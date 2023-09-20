
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import is_torch_sparse_tensor, add_self_loops, add_remaining_self_loops
import torch_sparse
from torch_sparse import fill_diag, mul, sum as sparsesum, matmul
from dgsparse import SparseTensor
from dgsparse import spmm_sum, spmm_max, spmm_mean
import torch_geometric.transforms as T
import time
import wandb
import matplotlib.pyplot as plt
import tqdm

import dgl.sparse as dglsp
from dgl.nn import GINConv as DGL_GINConv
# from dgsparse.nn.gcnconv import GCN

device = "cuda:6" if torch.cuda.is_available() else "cpu"


class GCNConv(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = nn.Linear(in_size, out_size, bias=False)

    def forward(self, dcsr, x):
        x = self.W(x)
        x = spmm_sum(dcsr, x, 0)
        return x
    
class GCN(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super().__init__()
        self.conv1 = GCNConv(in_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, out_size)
    
    def forward(self, dcsr, x):
        x = self.conv1(dcsr, x)
        x = F.relu(x)
        x = self.conv2(dcsr, x)

        return x

class GCNConv_torch_sparse(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = nn.Linear(in_size, out_size, bias=False)

    def forward(self, adj_t, x):
        x = self.W(x)
        x = matmul(adj_t, x)
        return x
    
class GCN_torch_sparse(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super().__init__()
        self.conv1 = GCNConv_torch_sparse(in_size, hidden_size)
        self.conv2 = GCNConv_torch_sparse(hidden_size, out_size)

    def forward(self, adj_t, x):
        x = self.conv1(adj_t, x)
        x = F.relu(x)
        x = self.conv2(adj_t, x)

        return x
    

def gcn_norm_from_edge_index(edge_index, num_nodes, add_self_loops=True):
    adj_t = torch_sparse.SparseTensor(
        row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)
    )
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1.0)
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t


def get_gcn_dcsr_from_edge_index(edge_index, num_nodes):
    adj_t = gcn_norm_from_edge_index(edge_index, num_nodes)
    rowptr, col, value = adj_t.csr()
    rowptr = rowptr.int()
    col = col.int()
    tcsr = torch.sparse_csr_tensor(
        rowptr,
        col,
        value,
        dtype=torch.float,
        size=(num_nodes, num_nodes),
        requires_grad=True,
        device=adj_t.device(),
    )
    dcsr = SparseTensor.from_torch_sparse_csr_tensor(
        tcsr.clone().detach(), True, requires_grad=True
    )
    return dcsr


def test_forward_time(model, g):
    # g = g.to(device)
    features = g.ndata["feat"].to(device)
    label = g.ndata["label"].to(device)
    train_mask = g.ndata["train_mask"].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    # Preprocess to get the adjacency matrix of the graph.
    indices = torch.stack(g.edges()).to(device)
    N = g.num_nodes()
    adj_t = gcn_norm_from_edge_index(indices, N)
    dcsr = get_gcn_dcsr_from_edge_index(indices, N)

    # warm_up
    if model is GCN_torch_sparse:
        for i in range(10):
            logits = model(adj_t, features)
    elif model is GCN:
        for i in range(10):
            logits = model(dcsr, features)
    
    torch.cuda.synchronize()
    start = time.time()

    if model is GCN_torch_sparse:
        for i in range(100):
            logits = model(adj_t, features)
    elif model is GCN:
        for i in range(100):
            logits = model(dcsr, features)
    
    torch.cuda.synchronize()
    end = time.time()

    return end - start


def test_time(model, g):
    # g = g.to(device)
    features = g.ndata["feat"].to(device)
    label = g.ndata["label"].to(device)
    train_mask = g.ndata["train_mask"].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    # Preprocess to get the adjacency matrix of the graph.
    indices = torch.stack(g.edges()).to(device)
    N = g.num_nodes()
    adj_t = gcn_norm_from_edge_index(indices, N)
    dcsr = get_gcn_dcsr_from_edge_index(indices, N)
    
    # warm_up
    if model is GCN_torch_sparse:
        for i in range(10):
            logits = model(adj_t, features)
    elif model is GCN:
        for i in range(10):
            logits = model(dcsr, features)

    torch.cuda.synchronize()
    start = time.time()
    forward_start = time.time()
    backward_start = time.time()

    if model is GCN_torch_sparse:
        for i in range(100):
            logits = model(adj_t, features)
    elif model is GCN:
        for i in range(100):
            logits = model(dcsr, features)

    torch.cuda.synchronize()
    forward_end = time.time()



class GCNConv_old(nn.Module):
    def __init__(self, in_size, out_size, cached=False):
        super().__init__()
        self.W = nn.Linear(in_size, out_size, bias=False)
        self.cached = cached
        self._cached_dcsr = None

    def forward(self, edge_index, x, num_nodes):
        cache = self._cached_dcsr
        if cache is None:
            adj_t = self.gcn_norm(edge_index, num_nodes)
            rowptr, col, value = adj_t.csr()
            rowptr = rowptr.int()
            col = col.int()
            tcsr = torch.sparse_csr_tensor(
                rowptr,
                col,
                value,
                dtype=torch.float,
                size=(num_nodes, num_nodes),
                requires_grad=True,
                device=edge_index.device,
            )
            dcsr = SparseTensor.from_torch_sparse_csr_tensor(
                tcsr.clone().detach(), True, requires_grad=True
            )
            if self.cached:
                self._cached_dcsr = dcsr
        else:
            dcsr = cache
        x = self.W(x)
        return spmm_sum(dcsr, x, 0)

    def gcn_norm(self, edge_index, num_nodes, add_self_loops=True):
        adj_t = torch_sparse.SparseTensor(
            row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)
        )
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0)
        if add_self_loops:
            adj_t = fill_diag(adj_t, 1.0)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t



class GCN_old(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, cached=False):
        super().__init__()
        self.conv1 = GCNConv_old(in_size, hidden_size, cached)
        self.conv2 = GCNConv_old(hidden_size, out_size, cached)

    def forward(self, edge_index, x, num_nodes):
        x = self.conv1(edge_index, x, num_nodes)
        x = F.relu(x)
        x = self.conv2(edge_index, x, num_nodes)
        return x
    
class GCNConv_torch_sparse_old(nn.Module):
    def __init__(self, in_size, out_size, cached=False):
        super().__init__()
        self.W = nn.Linear(in_size, out_size, bias=False)
        self.cached = cached
        self._cached_adj_t = None

    def forward(self, edge_index, x, num_nodes):
        cache = self._cached_adj_t
        if cache is None:
            adj_t = self.gcn_norm(edge_index, num_nodes)
            if self.cached:
                self._cached_adj_t = adj_t
        else:
            adj_t = cache
        x = self.W(x)
        rst = matmul(adj_t, x)
        return rst

    def gcn_norm(self, edge_index, num_nodes, add_self_loops=True):
        adj_t = torch_sparse.SparseTensor(
            row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)
        )
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1.0)
        if add_self_loops:
            adj_t = fill_diag(adj_t, 1.0)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

class GCN_torch_sparse_old(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, cached=False):
        super().__init__()
        self.conv1 = GCNConv_torch_sparse_old(in_size, hidden_size, cached)
        self.conv2 = GCNConv_torch_sparse_old(hidden_size, out_size, cached)

    def forward(self, edge_index, x, num_nodes):
        x = self.conv1(edge_index, x, num_nodes)
        x = F.relu(x)
        x = self.conv2(edge_index, x, num_nodes)
        return x


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
    adj_t = gcn_norm_from_edge_index(indices, N)
    dcsr = get_gcn_dcsr_from_edge_index(indices, N)

    # warm_up
    if isinstance(model, GCN_torch_sparse):
        for i in range(10):
            logits = model(adj_t, features)
    elif isinstance(model, GCN):
        for i in range(10):
            logits = model(dcsr, features)
    
    torch.cuda.synchronize()
    start = time.time()

    if isinstance(model, GCN_torch_sparse):
        print("进入GCN_torch_sparse")
        for i in range(100):
            logits = model(adj_t, features)

            loss = loss_fcn(logits[train_mask], label[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    elif isinstance(model, GCN):
        print("进入GCN")
        for i in range(100):
            logits = model(dcsr, features)

            loss = loss_fcn(logits[train_mask], label[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    torch.cuda.synchronize()
    end = time.time()

    return end - start

    # for epoch in range(50):
    #     model.train()

    #     # Forward.
    #     logits = model(indices, features, N)

    #     # Compute loss with nodes in the training set.
    #     loss = loss_fcn(logits[train_mask], label[train_mask])

    #     # Backward.
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

        # Compute prediction.
        # pred = logits.argmax(dim=1)

        # # Evaluate the prediction.
        # val_acc, test_acc = evaluate(g, pred)
        # if epoch % 5 == 0:
        #     print(
        #         f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}"
        #         f", test acc: {test_acc:.3f}"
        #     )
        # wandb.log({
        #     "loss": loss,
        #     "val acc": val_acc,
        #     "test acc": test_acc
        # })



# Load graph from the existing dataset.
dataset = dgl.data.RedditDataset()
# dataset = dgl.data.CoraGraphDataset()
g = dataset[0]

# Create model.s
feature = g.ndata["feat"]
in_size = feature.shape[1]
out_size = dataset.num_classes
gcn_model_old = GCN_old(in_size, out_size, 16)
gcn_model_old = gcn_model_old.to(device)
gcn_model = GCN(in_size, out_size, 16)
gcn_model = gcn_model.to(device)
# gcn_cachen_model = GCN(in_size, out_size, 16, cached=True)
# gcn_cachen_model = gcn_cachen_model.to(device)
gcn_torch_sparse_mdoel_old = GCN_torch_sparse_old(in_size, out_size, 16)
gcn_torch_sparse_mdoel_old = gcn_torch_sparse_mdoel_old.to(device)
gcn_torch_sparse_mdoel = GCN_torch_sparse(in_size, out_size, 16)
gcn_torch_sparse_mdoel = gcn_torch_sparse_mdoel.to(device)
# gcn_torch_sparse_cache_mdoel = GCN_torch_sparse(in_size, out_size, 16, cached=True)
# gcn_torch_sparse_cache_mdoel = gcn_torch_sparse_cache_mdoel.to(device)

# Kick off training.

dgsparse_time = []
dgsparse_cache_time = []
torch_sparse_time = []
torch_sparse_cache_time = []

epochs = 10

# wandb.init(project="dgsparse_time_compare2_torch_sparse_reddit", name="dgsparse")
# for i in range(epochs):
#     start = time.time()
#     train(gcn_model, g)
#     end = time.time()
#     print(f"dgsparse time is: {end - start}")
#     dgsparse_time.append(end - start)
#     wandb.log({
#         "total_time": end - start
#     })
#     torch.cuda
# wandb.finish()

# wandb.init(project="dgsparse_time_compare2_torch_sparse_reddit", name="dgsparse_cache")
# for i in range(epochs):
#     start = time.time()
#     train(gcn_cachen_model, g)
#     end = time.time()
#     print(f"dgsparse cached time is: {end - start}")
#     dgsparse_cache_time.append(end - start)
#     wandb.log({
#         "total_time": end - start
#     })
# wandb.finish()

# wandb.init(project="dgsparse_time_compare2_torch_sparse_reddit", name="torch_sparse")
# for i in range(epochs):
#     start = time.time()
#     train(gcn_torch_sparse_mdoel, g)
#     end = time.time()
#     print(f"torch_sparse time is: {end - start}")
#     torch_sparse_time.append(end - start)
#     wandb.log({
#         "total_time": end - start
#     })
# wandb.finish()

# wandb.init(project="dgsparse_time_compare2_torch_sparse_reddit", name="torch_sparse cache")
# for i in range(epochs):
#     start = time.time()
#     train(gcn_torch_sparse_cache_mdoel, g)
#     end = time.time()
#     print(f"torch_sparse cache time is: {end - start}")
#     torch_sparse_cache_time.append(end - start)
#     wandb.log({
#         "total_time": end - start
#     })
# wandb.finish()

# print(dgsparse_time)
# print(dgsparse_cache_time)
# print(torch_sparse_time)
# print(torch_sparse_cache_time)


def train_time(model, g):
    # g = g.to(device)
    features = g.ndata["feat"].to(device)
    label = g.ndata["label"].to(device)
    train_mask = g.ndata["train_mask"].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    # Preprocess to get the adjacency matrix of the graph.
    indices = torch.stack(g.edges()).to(device)
    N = g.num_nodes()

    # warm_up
    for i in range(10):
        logits = model(indices, features, N)

    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        logits = model(indices, features, N)
    torch.cuda.synchronize()
    end = time.time()

    return end - start


# dgsparse_time = train_time(gcn_model_old, g)
# torch_sparse_time = train_time(gcn_torch_sparse_mdoel_old, g)

dgsparse_time = []
torch_sparse_time = []

for i in tqdm.tqdm(range(10)):
    # dgsparse_time.append(train(gcn_model, g))
    torch_sparse_time.append(train(gcn_torch_sparse_mdoel, g))

print(dgsparse_time)
print(torch_sparse_time)

# print(train(gcn_model, g), train(gcn_torch_sparse_mdoel, g))













