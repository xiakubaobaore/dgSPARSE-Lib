

import torch
import dgsparse

device = "cuda:7" if torch.cuda.is_available() else "cpu"
rowptr = torch.tensor([0, 2, 3, 5, 6, 8])
col = torch.tensor([1, 4, 3, 0, 4, 3, 1, 4])
rowptr = rowptr.int().to(device)
col = col.int().to(device)
a = dgsparse.SparseTensor(None, rowptr, col, has_value=True)
stor = a.storage

print(a.csr())
print(stor._rowptr)
print(stor.row())

