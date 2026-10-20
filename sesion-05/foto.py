import torch

foto = torch.zeros(28, 28)           # una foto de 28 × 28 puntos, toda negra
print(foto.shape, foto.numel())
print(foto.flatten().shape)

lote = torch.zeros(32, 28, 28)       # 32 fotos
print(lote.reshape(32, -1).shape)
print(lote.flatten().shape)
