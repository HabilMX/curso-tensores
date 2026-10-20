# Errores comunes de la sesión 2 — escrito para el repositorio.
# La tabla «Errores comunes» de la lección cita mensajes y resultados; este script los
# provoca uno por uno y los imprime, para que el verificador compruebe que la tabla dice la verdad.
# (sesión 2; ver el README.md de la carpeta)
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

probar("t.shape()", lambda: papeleria.shape())
probar("numel sin paréntesis", lambda: str(papeleria.numel).split(" at 0x")[0])
probar("torch.Tensor", lambda: (torch.Tensor([1, 2, 3]), torch.Tensor([1, 2, 3]).dtype))
