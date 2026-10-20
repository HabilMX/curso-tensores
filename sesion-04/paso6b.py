from papeleria import papeleria

dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
sucursales = ["Centro", "Norte"]

ventas_diarias = papeleria.sum(dim=2)
valores, indices = ventas_diarias.max(dim=1)
print(valores)
print(indices)

for s in range(2):
    dia = dias[indices[s].item()]
    print(sucursales[s], "vendió más el", dia, ":", valores[s].item())
