# cerro.py — una sola perilla: el precio de un cuaderno
import torch

precio = torch.tensor(0.0, requires_grad=True)   # empieza en $0: ni idea
tasa = 0.05                                      # qué tan largo es cada paso

for paso in range(1, 6):
    cobro = precio * 3                    # predicción: lo que cobraría por 3 cuadernos
    perdida = (cobro - 60) ** 2           # qué tan mal: la diferencia contra $60, al cuadrado
    perdida.backward()                    # PyTorch calcula la pendiente
    a_mano = 18 * precio.item() - 360     # la misma pendiente, sacada con lápiz
    assert abs(precio.grad.item() - a_mano) < 0.001, "PyTorch y el lápiz deberían coincidir"
    print(f"paso {paso}: precio {precio.item():6.2f}  pérdida {perdida.item():8.2f}  pendiente {precio.grad.item():8.2f}")
    with torch.no_grad():
        precio -= tasa * precio.grad      # un paso cuesta abajo
    precio.grad.zero_()                   # borrar la pendiente para el siguiente paso

assert abs(precio.item() - 20) < 0.1, "Tres cuadernos de $20 dan $60"
print("precio aprendido:", round(precio.item(), 2))
