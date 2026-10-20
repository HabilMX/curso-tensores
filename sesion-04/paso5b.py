from papeleria import papeleria, centro

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
decimales = ventas_diarias.float()
print(decimales.dtype)
print(decimales)

promedio_diario = decimales.mean(dim=1)     # quita el eje de los días
print(promedio_diario.shape)
print(promedio_diario)

# Con for: el total del Centro entre 7 días
total_centro = 0
for renglon in centro:
    total_centro = total_centro + sum(renglon)
print(total_centro / 7)

assert promedio_diario.shape == (2,), "Esperaba un promedio por sucursal"
assert round(promedio_diario[0].item(), 2) == round(total_centro / 7, 2), "No coincide con el for"
print("Promedio diario: cuadra")
