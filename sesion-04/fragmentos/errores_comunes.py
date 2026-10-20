# Errores comunes de la sesión 4 — escrito para el repositorio.
# La tabla «Errores comunes» de la lección cita mensajes y resultados; este script los
# provoca uno por uno y los imprime, para que el verificador compruebe que la tabla dice la verdad.
# (sesión 4; ver el README.md de la carpeta)
import torch

def probar(nombre, f):
    try:
        print(nombre, "->", f())
    except Exception as error:
        print(nombre, "->", type(error).__name__ + ":", str(error).split("\n")[0])

centro = [[1200, 800, 500], [1500, 900, 700], [1000, 700, 500], [1300, 900, 600],
          [2000, 1200, 700], [1200, 800, 400], [800, 500, 300]]
norte = [[300, 200, 100], [400, 250, 150], [350, 200, 100], [300, 150, 100],
         [500, 300, 200], [300, 200, 100], [200, 100, 100]]
papeleria = torch.tensor([centro, norte])

ventas_diarias = papeleria.sum(dim=2)
por_dia = ventas_diarias[0]

probar("max(dim=(1, 2))", lambda: ventas_diarias.max(dim=(1, 2)))
probar("tensor == lista", lambda: por_dia == [2500, 3100, 2200, 2800, 3900, 2400, 1600])
probar("2642.857177734375 == 18500 / 7", lambda: 2642.857177734375 == 18500 / 7)
