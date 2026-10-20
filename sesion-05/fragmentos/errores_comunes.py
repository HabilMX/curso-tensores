# Errores comunes de la sesión 5 — escrito para el repositorio.
# La tabla «Errores comunes» de la lección cita mensajes y resultados; este script los
# provoca uno por uno y los imprime, para que el verificador compruebe que la tabla dice la verdad.
# (sesión 5; ver el README.md de la carpeta)
import torch

def probar(nombre, f):
    try:
        print(nombre, "->", f())
    except Exception as error:
        print(nombre, "->", type(error).__name__ + ":", str(error).split("\n")[0])

meses = [60000, 55000, 60000, 50000, 65000, 50000, 70000, 100000, 70000, 60000, 55000, 90000]
ventas = torch.tensor(meses)

probar("reshape(-1, -1)", lambda: ventas.reshape(-1, -1))
probar("reshape(-2, -6)", lambda: ventas.reshape(-2, -6))
probar("reshape(-1, 5)", lambda: ventas.reshape(-1, 5))
probar("ventas.reshape(4, 3); ventas.shape", lambda: (ventas.reshape(4, 3), ventas.shape)[1])
probar("ventas[3, 2]", lambda: ventas[3, 2])
