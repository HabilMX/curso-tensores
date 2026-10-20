from papeleria import papeleria, centro

pestana_centro = papeleria[0]          # la pestaña del Centro: 7 días × 3 productos
print(pestana_centro.shape)

por_dia = pestana_centro.sum(dim=1)    # suma a lo largo del eje 1 (productos)
print(por_dia)
print(por_dia.shape)

# Lo mismo con for, como ya sabes hacerlo
por_dia_con_for = []
for renglon in centro:
    por_dia_con_for.append(sum(renglon))
print(por_dia_con_for)

assert por_dia.shape == (7,), "Esperaba un número por día: forma (7,)"
assert por_dia.tolist() == por_dia_con_for, "La suma con tensores no coincide con la del for"
print("Por día: cuadra")
