from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
resultado = ventas_diarias.max(dim=1)
print(resultado)
