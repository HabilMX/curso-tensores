from papeleria import papeleria

sin_sucursal = papeleria.sum(dim=0)   # junta las dos pestañas
print(sin_sucursal.shape)
print(sin_sucursal)

sin_dia = papeleria.sum(dim=1)        # junta los 7 días
print(sin_dia.shape)
print(sin_dia)

sin_producto = papeleria.sum(dim=2)   # junta los 3 productos
print(sin_producto.shape)
print(sin_producto)
