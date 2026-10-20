import torch
from dimensiones import por_dia

dia_dec = torch.tensor(por_dia, dtype=torch.float32)
print(dia_dec)
print(dia_dec.dtype)
