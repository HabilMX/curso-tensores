# Errores comunes de la sesión 3 — escrito para el repositorio.
# La tabla «Errores comunes» de la lección cita mensajes y resultados; este script los
# provoca uno por uno y los imprime, para que el verificador compruebe que la tabla dice la verdad.
# (sesión 3; ver el README.md de la carpeta)
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

def asercion():
    assert papeleria[0, 0] == 1200


probar("papeleria[1, 7, 2]", lambda: papeleria[1, 7, 2])
probar("papeleria[4, 1, 2]", lambda: papeleria[4, 1, 2])
probar("papeleria[:][4]", lambda: papeleria[:][4])
probar("papeleria[0, 0].item()", lambda: papeleria[0, 0].item())
probar("assert papeleria[0, 0] == 1200", asercion)
probar("papeleria[0, 5:20].shape", lambda: papeleria[0, 5:20].shape)
