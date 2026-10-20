def contar_dimensiones(x):
    dimensiones = 0
    while type(x) == list:
        dimensiones = dimensiones + 1
        x = x[0]
    return dimensiones

def forma(x):
    tamanos = []
    while type(x) == list:
        tamanos.append(len(x))
        x = x[0]
    return tuple(tamanos)

def es_rectangular(tabla):
    largo = len(tabla[0])
    for fila in tabla:
        if len(fila) != largo:
            return False
    return True

total = 18500

por_dia = [2500, 3100, 2200, 2800, 3900, 2400, 1600]

por_dia_y_producto = [
    [1200, 800, 500],    # lunes
    [1500, 900, 700],    # martes
    [1000, 700, 500],    # miércoles
    [1300, 900, 600],    # jueves
    [2000, 1200, 700],   # viernes
    [1200, 800, 400],    # sábado
    [800, 500, 300],     # domingo
]

centro = por_dia_y_producto
norte = [
    [300, 200, 100],
    [400, 250, 150],
    [350, 200, 100],
    [300, 150, 100],
    [500, 300, 200],
    [300, 200, 100],
    [200, 100, 100],
]
papeleria = [centro, norte]

if __name__ == "__main__":
    print("total", contar_dimensiones(total), forma(total))
    print("por_dia", contar_dimensiones(por_dia), forma(por_dia))
    print("por_dia_y_producto", contar_dimensiones(por_dia_y_producto), forma(por_dia_y_producto))
    print("papeleria", contar_dimensiones(papeleria), forma(papeleria))

    assert forma(papeleria) == (2, 7, 3), "La papelería debería ser 2 sucursales × 7 días × 3 productos"
    print("Todo cuadra")
