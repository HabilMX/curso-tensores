# Ejemplos interactivos

Los ejemplos de las lecciones que se escriben en el intérprete de Python (las líneas que empiezan con `>>>`).
El verificador (`python verificar.py`) los corre con `doctest`, dentro de la carpeta de su sesión, y comprueba que lo
que imprimen es lo que muestra la lección.

Para abrir el intérprete: con el entorno activado, entra a la carpeta de la sesión y escribe `python3`.

## Sesión 1: una lista chueca

La lección muestra estas respuestas después de definir `chueca`. Las dos primeras líneas (importar lo de
`dimensiones.py` y definir `chueca`, que la lección escribe en un bloque aparte) y las dos últimas (el error de
`chueca[1][2]`, que la lección cita) se juntan aquí para que el ejemplo corra completo.

<!-- sesion: sesion-01 -->
```pycon
>>> from dimensiones import contar_dimensiones, forma, es_rectangular, por_dia_y_producto
>>> chueca = [[1, 2, 3], [4, 5]]
>>> contar_dimensiones(chueca)
2
>>> forma(chueca)
(2, 3)
>>> es_rectangular(por_dia_y_producto)
True
>>> es_rectangular(chueca)
False
>>> chueca[1][2]
Traceback (most recent call last):
  ...
IndexError: list index out of range
```

## Sesión 4: sumar a lo largo de una pestaña

La lección dice «en la terminal de Python, con `from papeleria import papeleria`»; esa primera línea se escribe aquí.

<!-- sesion: sesion-04 -->
```pycon
>>> from papeleria import papeleria
>>> papeleria[0].sum(dim=0)
tensor([9000, 5800, 3700])
>>> papeleria[0].sum(dim=0).shape
torch.Size([3])
```

## Sesión 4: la misma suma en TensorFlow (no lo corre el verificador)

TensorFlow no es requisito del curso, así que el verificador no corre estos bloques. Se probaron una vez, aparte, con
TensorFlow 2.21.0 en otro entorno virtual: las salidas coincidieron con las de la lección (ver `DIFERENCIAS.md`).
El código es el de la lección.

```python
import tensorflow as tf
from papeleria import centro, norte

papeleria_tf = tf.constant([centro, norte])
print(tf.reduce_sum(papeleria_tf, axis=(1, 2)))
print(tf.reduce_mean(tf.reduce_sum(papeleria_tf, axis=2), axis=1))
```

Salida (TensorFlow 2.21):

```
tf.Tensor([18500  4600], shape=(2,), dtype=int32)
tf.Tensor([2642  657], shape=(2,), dtype=int32)
```

```python
promedios_tf = tf.reduce_mean(
    tf.cast(tf.reduce_sum(papeleria_tf, axis=2), tf.float32),
    axis=1,
)
print(promedios_tf)
```

Salida (TensorFlow 2.21):

```
tf.Tensor([2642.8572  657.1429], shape=(2,), dtype=float32)
```

## Sesión 5: comprobar que `papeleria.py` se puede importar

Es un comando de la terminal (no del intérprete). La lección dice que debe imprimir `torch.Size([2, 7, 3])`.

```bash
python3 -c "from papeleria import papeleria; print(papeleria.shape)"
```
