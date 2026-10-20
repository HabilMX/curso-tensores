from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)
print(ventas_diarias.argmax(dim=1))
print(ventas_diarias.max())
print(ventas_diarias.min(dim=1))
