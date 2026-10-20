# Ejercicio 17 — solución, tal cual aparece en la lección.
# (sesión 4; ver el README.md de la carpeta para saber de dónde sale este archivo)
import torch

parciales = torch.tensor([
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
])                                   # alumno × materia × parcial
decimales = parciales.float()

por_alumno = decimales.mean(dim=(1, 2))           # a) quito materias y parciales
assert por_alumno.shape == (3,)
por_alumno_y_materia = decimales.mean(dim=2)      # b) quito parciales
assert por_alumno_y_materia.shape == (3, 4)
por_materia = decimales.mean(dim=(0, 2))          # d) quito alumnos y parciales
assert por_materia.shape == (4,)

mejor_alumno = por_alumno.argmax()                # c)
assert mejor_alumno.shape == ()
assert mejor_alumno.item() == 1, "El segundo alumno debería tener el mejor promedio"
assert torch.allclose(por_materia, torch.tensor([8.5556, 8.3333, 8.1111, 8.4444]), atol=1e-4)
assert por_materia.argmin().item() == 2, "La tercera materia debería ser la más difícil"  # d)

print(por_alumno)
print(por_alumno_y_materia)
print(mejor_alumno)
print(por_materia, por_materia.argmin())

suma = 0
cuantas = 0
for materia in parciales[0].tolist():
    for calificacion in materia:
        suma = suma + calificacion
        cuantas = cuantas + 1
assert round(por_alumno[0].item(), 2) == round(suma / cuantas, 2), "No coincide con el for"
print("Ejercicio 17 superado")
