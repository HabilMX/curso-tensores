# Ejercicio 25 — solución (la función de la lección + los assert del enunciado).
# (sesión 6; ver el README.md de la carpeta para saber de dónde sale este archivo)
def se_pueden_estirar(forma_a, forma_b):
    cuantos = min(len(forma_a), len(forma_b))
    for i in range(1, cuantos + 1):
        a = forma_a[-i]
        b = forma_b[-i]
        if a != b and a != 1 and b != 1:
            return False
    return True

assert se_pueden_estirar((7, 3), (3,)) == True
assert se_pueden_estirar((7, 3), (7,)) == False
assert se_pueden_estirar((2, 7, 3), (7, 1)) == True
assert se_pueden_estirar((7, 3), (4,)) == False
assert se_pueden_estirar((7, 1), (1, 3)) == True
assert se_pueden_estirar((7, 3), ()) == True
print("Ejercicio 25 superado")
