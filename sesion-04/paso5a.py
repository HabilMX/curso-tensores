from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
print(ventas_diarias.dtype)
promedio_diario = ventas_diarias.mean(dim=1)
