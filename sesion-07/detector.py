# detector.py — un detector de fraude que aprende solo
import torch

torch.manual_seed(42)          # semilla fija: los mismos números "al azar" cada vez

# 1) DATOS INVENTADOS: 4,000 compras con tarjeta
n = 4000
monto      = torch.rand(n) * 20_000               # pesos, de 0 a 20,000
hora       = torch.randint(0, 24, (n,)).float()   # hora del día, de 0 a 23
distancia  = torch.rand(n) * 500                  # km desde la casa del cliente
extranjero = (torch.rand(n) < 0.1).float()        # 1 = comercio de otro país

# 2) LA REGLA ESCONDIDA (solo nosotros la conocemos; el modelo nunca la ve)
focos = (monto > 15_000).float() + (hora < 5).float() + (distancia > 400).float() + extranjero
y = (focos >= 2).float()       # la etiqueta: 1 = fraude, 0 = normal

# 3) EL TENSOR DE ENTRADA: compras × datos, todo entre 0 y 1
X = torch.stack([monto / 20_000, hora / 23, distancia / 500, extranjero], dim=1)
print("X:", tuple(X.shape), "| y:", tuple(y.shape), "| % fraude:", round(y.mean().item() * 100, 1))

# 4) Estudia con 3,200 compras; el examen es con las otras 800
X_ent, y_ent = X[:3200], y[:3200]
X_exa, y_exa = X[3200:], y[3200:]

# 5) LAS PERILLAS: empiezan al azar y el modelo las va girando
W1 = (torch.randn(4, 8) * 0.5).requires_grad_()
b1 = torch.zeros(8, requires_grad=True)
W2 = (torch.randn(8) * 0.5).requires_grad_()
b2 = torch.zeros((), requires_grad=True)   # () = un solo número: la perilla final

def predecir(X):
    oculta = torch.relu(X @ W1 + b1)   # (compras, 4) @ (4, 8) -> (compras, 8)
    return oculta @ W2 + b2            # (compras, 8) @ (8,)   -> (compras,)

perdida_fn = torch.nn.BCEWithLogitsLoss()
optimizador = torch.optim.SGD([W1, b1, W2, b2], lr=1.0)

# 6) EL CICLO DE ENTRENAMIENTO: una época = una vuelta a las 3,200 compras
for epoca in range(1, 401):
    puntaje = predecir(X_ent)               # 1. predecir
    perdida = perdida_fn(puntaje, y_ent)    # 2. medir qué tan mal le fue
    optimizador.zero_grad()                 # 3. borrar las pendientes de la vuelta anterior
    perdida.backward()                      # 4. calcular la pendiente de cada perilla
    optimizador.step()                      # 5. girar cada perilla un pasito cuesta abajo
    if epoca in (1, 10, 50, 100, 200, 400):
        print(f"época {epoca:3d}   pérdida {perdida.item():.4f}")

# 7) EL EXAMEN: 800 compras que el modelo nunca vio
with torch.no_grad():
    dice_fraude = (predecir(X_exa) > 0).float()   # puntaje positivo = "esto es fraude"

exactitud = (dice_fraude == y_exa).float().mean().item()
atrapados = (dice_fraude * y_exa).sum().item()          # dijo fraude y sí era
falsas    = (dice_fraude * (1 - y_exa)).sum().item()    # dijo fraude y no era
escapados = ((1 - dice_fraude) * y_exa).sum().item()    # dijo normal y era fraude

print("exactitud:", round(exactitud, 3))
print("fraudes en el examen:", int(y_exa.sum().item()))
print("atrapados:", int(atrapados), "| se escaparon:", int(escapados), "| alarmas falsas:", int(falsas))
print("recall:", round(atrapados / (atrapados + escapados), 3),
      "| precisión:", round(atrapados / (atrapados + falsas), 3))
