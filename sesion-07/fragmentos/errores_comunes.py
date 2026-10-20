# Errores comunes de la sesión 7 — escrito para el repositorio.
# La tabla «Errores comunes» de la lección cita mensajes y resultados; este script los
# provoca uno por uno y los imprime, para que el verificador compruebe que la tabla dice la verdad.
# (sesión 7; ver el README.md de la carpeta)
import torch

def probar(nombre, f):
    try:
        f()
        print(nombre, "-> no truena")
    except Exception as error:
        print(nombre, "->", type(error).__name__ + ":", str(error).split(". Saved")[0].split("\n")[0])


def sin_grabadora():
    precio = torch.tensor(0.0)
    cobro = precio * 3
    ((cobro - 60) ** 2).backward()


def etiquetas_con_otra_forma():
    torch.nn.BCEWithLogitsLoss()(torch.zeros(3200), torch.zeros(3200, 1))


def dos_backward():
    precio = torch.tensor(0.0, requires_grad=True)
    perdida = (precio * 3 - 60) ** 2
    perdida.backward()
    perdida.backward()


probar("requires_grad olvidado en cerro.py", sin_grabadora)
probar("etiquetas (3200, 1) contra puntajes (3200,)", etiquetas_con_otra_forma)
probar("backward dos veces", dos_backward)
