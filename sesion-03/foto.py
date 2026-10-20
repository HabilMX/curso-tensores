import torch

# Foto de 2 × 3 puntos, en orden alto × ancho × color
#   eje 0 = renglón (alto), eje 1 = punto en el renglón (ancho), eje 2 = color (0 rojo, 1 verde, 2 azul)
foto = torch.tensor([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],          # arriba: rojo, verde, azul
    [[255, 255, 255], [0, 0, 0], [255, 255, 0]],      # abajo: blanco, negro, amarillo
])
print(foto.shape)

rojo = foto[:, :, 0]
print(rojo)
print(rojo.shape)
assert rojo.shape == (2, 3), "El canal rojo es una tabla de 2 × 3"
assert rojo.tolist() == [[255, 0, 0], [255, 0, 255]]
print("Canal rojo correcto")

print(foto[1, 2])

# La misma foto en orden color × alto × ancho
#   eje 0 = color, eje 1 = renglón, eje 2 = punto en el renglón
foto_cha = torch.tensor([
    [[255, 0, 0], [255, 0, 255]],      # canal rojo
    [[0, 255, 0], [255, 0, 255]],      # canal verde
    [[0, 0, 255], [255, 0, 0]],        # canal azul
])
print(foto_cha.shape)
print(foto_cha[0])
assert foto_cha[0].tolist() == foto[:, :, 0].tolist()
print("Mismo rojo, otro orden")
