import torch

# Banca: un lote de compras, como el detector de la sesión 7
compras = torch.zeros(64, 4)
print("compras:", compras.shape)

# Banca: el historial de cada cliente
historial = torch.zeros(32, 30, 4)
print("historial:", historial.shape)

# Seguros: las fotos de un siniestro
fotos = torch.zeros(8, 3, 480, 640)
print("fotos:", fotos.shape)
print("números en las fotos:", fotos.numel())

assert compras.shape == (64, 4), "Un lote de compras es (compras, datos)"
assert historial.shape == (32, 30, 4), "El historial es (clientes, compras en el tiempo, datos)"
assert fotos.shape == (8, 3, 480, 640), "Un lote de fotos es (fotos, colores, alto, ancho)"
print("Las tres formas cuadran")
