# Errores comunes de la sesión 6 — escrito para el repositorio.
# La tabla «Errores comunes» de la lección cita mensajes y resultados; este script los
# provoca uno por uno y los imprime, para que el verificador compruebe que la tabla dice la verdad.
# (sesión 6; ver el README.md de la carpeta)
import torch

def probar(nombre, f):
    try:
        print(nombre, "->", f())
    except Exception as error:
        print(nombre, "->", type(error).__name__ + ":", str(error).split("\n")[0])

unidades = torch.tensor([[30, 80, 2], [40, 90, 3], [25, 70, 2], [35, 90, 2], [50, 120, 3], [30, 80, 1], [20, 50, 1]])
precios = torch.tensor([40, 10, 350])
por_dia = unidades.sum(dim=1)

probar("tabla (7, 3) / total por día (7,)", lambda: unidades / por_dia)
probar("tabla de enteros @ precios decimales", lambda: unidades @ precios.float())
probar("lista * 2 y tensor * 2", lambda: ([1, 2, 3] * 2, torch.tensor([1, 2, 3]) * 2))
