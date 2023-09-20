
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import is_torch_sparse_tensor, add_self_loops, add_remaining_self_loops
import torch_sparse
from torch_sparse import fill_diag
from dgsparse import SparseTensor
from dgsparse import spmm_sum, spmm_max, spmm_mean
import torch_geometric.transforms as T
import time
import wandb

import dgl.sparse as dglsp
from dgl.nn import GINConv as DGL_GINConv
# from dgsparse.nn.gcnconv import GCN

device = "cuda:6" if torch.cuda.is_available() else "cpu"

class GIN_dgl(nn.Module):
    def __init__(
            self, in_size, out_size, hidden_size,
            aggregator_type="sum", init_eps=0, learn_eps=False,
            activation=None,
    ):
        super().__init__()
        self.conv1 = DGL_GINConv(
            nn.Linear(in_size, hidden_size, bias=False), aggregator_type, init_eps, learn_eps,
            activation
        )
        self.conv2 = DGL_GINConv(
            nn.Linear(hidden_size, out_size, bias=False), aggregator_type, init_eps, learn_eps,
            activation
        )

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = self.conv2(g, x)

        return x
    
    
class GINConv(nn.Module):
    def __init__(
        self,
        apply_func=None,
        aggregator_type="sum",
        init_eps=0,
        learn_eps=False,
        activation=None,
        cached=False,
    ):
        super().__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        self.cached = cached
        self._cached_dcsr = None
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

    def forward(self, edge_index, X, num_nodes):
        neigh = self.aggregate_neigh(edge_index, X, num_nodes, 0)
        rst = (1 + self.eps) * X + neigh

        if self.apply_func is not None:
            rst = self.apply_func(rst)
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


    def aggregate_neigh(self, edge_index, X, num_nodes, algorithm):
        adj_t = torch_sparse.SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.)
        rowptr, col, value = adj_t.csr()
        rowptr = rowptr.int()
        col = col.int()
        tcsr = torch.sparse_csr_tensor(
            rowptr, col, value, dtype=torch.float, size=(num_nodes, num_nodes),
            requires_grad=True,
            device=edge_index.device
        )
        dcsr = SparseTensor.from_torch_sparse_csr_tensor(
            tcsr.clone().detach(), True, requires_grad=True
        )
        if self._aggregator_type == "sum":
            rst = spmm_sum(dcsr, X, algorithm)
        elif self._aggregator_type == "max":
            rst = spmm_max(dcsr, X, algorithm)
        elif self._aggregator_type == "mean":
            rst = spmm_mean(dcsr, X, algorithm)
        else:
            rst = spmm_sum(dcsr, X, algorithm)
        return rst


class GIN(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        aggregator_type="sum",
        init_eps=0,
        learn_eps=False,
        activation=None,
        cached=False,
    ):
        super().__init__()
        self.conv1 = GINConv(
            nn.Linear(in_size, hidden_size),
            aggregator_type,
            init_eps,
            learn_eps,
            activation,
            cached,
        )
        self.conv2 = GINConv(
            nn.Linear(hidden_size, out_size),
            aggregator_type,
            init_eps,
            learn_eps,
            activation,
            cached,
        )

    def forward(self, edge_index, X, num_nodes):
        X = self.conv1(edge_index, X, num_nodes)
        X = self.conv2(edge_index, X, num_nodes)

        return X

    @property
    def eps(self):
        return self.conv1.eps
    

def evaluate(g, pred):
    label = g.ndata["label"].to(device)
    val_mask = g.ndata["val_mask"].to(device)
    test_mask = g.ndata["test_mask"].to(device)

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


def train(model, g):
    # g = g.to(device)
    features = g.ndata["feat"].to(device)
    label = g.ndata["label"].to(device)
    train_mask = g.ndata["train_mask"].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    # Preprocess to get the adjacency matrix of the graph.
    indices = torch.stack(g.edges()).to(device)
    N = g.num_nodes()

    for epoch in range(100):
        model.train()

        # Forward.
        if (isinstance(model, GIN_dgl)):
            logits = model(g, features)
        elif (isinstance(model, GIN)):
            logits = model(indices, features, N)

        print("前向结束")

        # Compute loss with nodes in the training set.
        loss = loss_fcn(logits[train_mask], label[train_mask])

        # Backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("后向结束")

        # Compute prediction.
        pred = logits.argmax(dim=1)

        # Evaluate the prediction.
        val_acc, test_acc = evaluate(g, pred)
        if epoch % 5 == 0:
            print(
                f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}"
                f", test acc: {test_acc:.3f}"
            )
        # wandb.log({
        #     "loss": loss,
        #     "val acc": val_acc,
        #     "test acc": test_acc
        # })



# Load graph from the existing dataset.
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Create model.s
feature = g.ndata["feat"]
in_size = feature.shape[1]
out_size = dataset.num_classes
gin_model = GIN_dgl(in_size, out_size, 16, activation=F.relu)
gin_model = gin_model.to(device)
dg_gin_model = GIN(in_size, out_size, 16, activation=F.relu)
dg_gin_model = dg_gin_model.to(device)
dg_gin_model_cache = GIN(in_size, out_size, 16, cached=True, activation=F.relu)
dg_gin_model_cache = dg_gin_model_cache.to(device)

# Kick off training.

dgl_train_time = []
dgsparse_train_time = []
dgsparse_cache_train_time = []

# wandb.init(project="dgsparse_time_compare", name="dgl")
# for i in range(10):
#     start = time.time()
#     train(gin_model, g)
#     end = time.time()
#     print(f"dgl time is: {end - start}")
#     dgl_train_time.append(end - start)
#     wandb.log({
#         "total_time": end - start
#     })
# wandb.finish()

# wandb.init(project="dgsparse_time_compare", name="dgsparse")
for i in range(10):
    start = time.time()
    train(dg_gin_model, g)
    end = time.time()
    print(f"dgsparse time is: {end - start}")
    dgsparse_train_time.append(end - start)
#     wandb.log({
#         "total_time": end - start
#     })
# wandb.finish()

# wandb.init(project="dgsparse_time_compare", name="dgsparse_cache")
for i in range(10):
    start = time.time()
    train(dg_gin_model_cache, g)
    end = time.time()
    print(f"dgsparse cache time is: {end - start}")
    dgsparse_cache_train_time.append(end - start)
#     wandb.log({
#         "total_time": end - start
#     })
# wandb.finish()

print(dgl_train_time)
print(dgsparse_train_time)
print(dgsparse_cache_train_time)









