# Errores comunes de la sesión 8 — escrito para el repositorio.
# La tabla «Errores comunes» de la lección cita mensajes y resultados; este script los
# provoca uno por uno y los imprime, para que el verificador compruebe que la tabla dice la verdad.
# (sesión 8; ver el README.md de la carpeta)
import torch

def probar(nombre, f):
    try:
        print(nombre, "->", f())
    except Exception as error:
        print(nombre, "->", type(error).__name__ + ":", str(error).split("\n")[0])

diccionario = torch.nn.Embedding(5, 3)

probar("Embedding con decimales", lambda: diccionario(torch.tensor([0.0, 2.0])))
probar("producto 5 de un catálogo 0-4", lambda: diccionario(torch.tensor([5])))
