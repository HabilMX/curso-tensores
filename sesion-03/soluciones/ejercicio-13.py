# Ejercicio 13 — solución (una respuesta posible), tal cual aparece en la lección.
# (sesión 3; ver el README.md de la carpeta para saber de dónde sale este archivo)
import torch

# Foto de 3 × 2 puntos, en orden color × alto × ancho
#   eje 0 = color (0 rojo, 1 verde, 2 azul)
#   eje 1 = renglón (0 arriba ... 2 abajo)
#   eje 2 = punto en el renglón (0 izquierda, 1 derecha)
# Los puntos: arriba rojo y verde; en medio azul y blanco; abajo negro y morado
foto = torch.tensor([
    [[255, 0], [0, 255], [0, 128]],      # canal rojo
    [[0, 255], [0, 255], [0, 0]],        # canal verde
    [[0, 0], [255, 255], [0, 128]],      # canal azul
])
assert foto.shape == (3, 3, 2), "3 colores × 3 renglones × 2 puntos"

verde = foto[1]
assert verde.shape == (3, 2)
assert verde.tolist() == [[0, 255], [0, 255], [0, 0]]

arriba = foto[:, 0]
assert arriba.shape == (3, 2)
assert arriba.tolist() == [[255, 0], [0, 255], [0, 0]]

esquina = foto[:, 2, 1]
assert esquina.shape == (3,)
assert esquina.tolist() == [128, 0, 128]
print("Ejercicio 13 superado")
