from papeleria import papeleria

semana = papeleria.sum(dim=1)    # sucursal × producto
print(semana)
print(semana.shape)

mal = semana.reshape(3, 2)
print(mal)

bien = semana.T
print(bien)
print(bien.shape)
