# [repo] se añadió import torch; el resto es el bloque de la lección (paso 3, «La trampa del orden de los colores»)
import torch

fotos = torch.zeros(8, 3, 480, 640)
assert fotos.shape == (8, 480, 640, 3), "Un lote de fotos es (fotos, alto, ancho, colores)"
