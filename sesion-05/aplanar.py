from papeleria import papeleria

print(papeleria.shape)
plana = papeleria.flatten()
print(plana.shape)
print(plana[:6])       # los primeros 6 números
print(plana[21:24])    # del 22 al 24
print(plana.sum(), papeleria.sum())

de_vuelta = plana.reshape(2, 7, 3)
assert de_vuelta.shape == (2, 7, 3), "Debería regresar a 2 sucursales × 7 días × 3 productos"
assert de_vuelta.tolist() == papeleria.tolist(), "Al regresar debería quedar idéntica"
print("Ida y vuelta sin perder nada")
