from papeleria import papeleria, centro, norte

por_sucursal = papeleria.sum(dim=(1, 2))   # junta días y productos
print(por_sucursal.shape)
print(por_sucursal)

por_sucursal_con_for = []
for pestana in [centro, norte]:
    total = 0
    for renglon in pestana:
        for venta in renglon:
            total = total + venta
    por_sucursal_con_for.append(total)
print(por_sucursal_con_for)

assert por_sucursal.shape == (2,), "Esperaba un número por sucursal: forma (2,)"
assert por_sucursal.tolist() == por_sucursal_con_for, "No coincide con el for"
assert por_sucursal[0].item() == 18500, "El Centro debería sumar 18,500"
print("Por sucursal: cuadra")
