import torch

piezas = torch.tensor([12, 30, 7])
precios = torch.tensor([35.5, 12.0, 450.0])
mezcla = torch.tensor([12, 30.5, 7])

print(piezas, piezas.dtype)
print(precios, precios.dtype)
print(mezcla, mezcla.dtype)
