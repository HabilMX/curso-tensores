import torch
from dimensiones import total, por_dia

t_total = torch.tensor(total)
t_dia = torch.tensor(por_dia)

print(t_total)
print(t_dia)
print(t_total.shape, t_dia.shape)
