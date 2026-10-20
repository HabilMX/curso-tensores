# Tensores desde cero (6): operar tensores completos

*Multiplicar una semana entera de ventas por sus precios en una sola línea, la regla que decide cuándo PyTorch «estira» un tensor, y la trampa de `*` contra `@` que no marca ningún error.*

*Tensores desde cero · lección B, sesión 6 de 8*

*Escenario: los personajes y la historia son inventados. Los datos técnicos, históricos y las cifras reales llevan su fuente.*

**Lunes 7 de diciembre, 7:40 p.m. Papelería Herrera, sucursal Centro.**

La cortina metálica está a medio bajar. Adentro huele a papel nuevo y a café recalentado. Doña Lupe cuenta monedas sobre el mostrador. Mateo tiene la laptop abierta entre un rollo de papel lustre y una caja de plumas.

Faltan dieciséis días para el 23.

Desde la semana pasada, Mateo sabe dos cosas. Que Norte iba bien hasta agosto. Y que en agosto un hombre de playera azul «cambió la terminal» sin que nadie lo hubiera llamado. Lo que no sabe es qué tiene que ver una cosa con la otra.

—Ya no quiero pesos, tía —dice—. Quiero piezas.

—¿Piezas de qué?

—De todo. Cuántos cuadernos, cuántas plumas, cuántas mochilas. Cada día. Del Centro y de Norte.

Doña Lupe lo mira por encima de los lentes.

—Para eso no necesitas computadora, mijo. Necesitas a alguien que cuente.

—Ya contaron. Fer me mandó lo de Norte en la mañana.

Hace dos semanas, nadie en la familia quería decir el nombre de Fer en voz alta. El Centro vendía unas cuatro veces lo que Norte, y ella era la encargada. Esa tarde no contestó el teléfono. Pero hoy mandó su conteo completo, renglón por renglón, sin que nadie se lo pidiera dos veces.

El celular de doña Lupe vibra. Un correo del banco. Del licenciado Ortega.

«Estimada señora Herrera: como lo solicitó, le comparto el total cobrado por la terminal de su sucursal Norte en la semana indicada. Le recuerdo que la fecha límite es el 23 de diciembre. Quedo atento.»

Abajo, un solo número: **$16,850**.

Doña Lupe le pasa el celular a Mateo sin decir nada.

—¿Eso es mucho o poco?

—Depende de cuánto vendimos.

Y ahí está el asunto. Para saber cuánto vendieron, hay que multiplicar cada pieza por su precio. Veintiuna casillas por sucursal. Con `for`, Mateo ya sabe hacerlo. Pero esta noche quiere una cuenta que tome la tabla entera de un jalón.

Porque si las piezas por sus precios dan menos de lo que cobró la terminal, alguien cobró algo que nunca salió de la papelería.

En la sesión anterior cambiaste la forma de un tensor sin tocar sus números: 12 meses en una fila se volvieron 4 trimestres de 3 meses. Hoy dejas de mover los números de lugar y empiezas a hacer cuentas con ellos, con el tensor completo a la vez.

## Repaso para abrir

Contesta sin ver la sesión anterior. Las respuestas están en las soluciones, al final.

1. ¿Cuándo un `reshape` es válido?
2. ¿Qué significa el `-1` en `reshape`?
3. ¿Por qué hacer `reshape` de una tabla no es lo mismo que girarla?

> **El código de esta sesión** está en el repositorio público del curso, en la carpeta `sesion-06`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-06). Escríbelo tú primero, porque así se aprende; el repositorio es para comparar cuando algo no te salga igual. Cada programa viene con la salida que debe darte.

## Objetivos de esta sesión

Al terminar, vas a poder hacer estas cuatro cosas. Al lado de cada una está el ejercicio que la demuestra.

| # | Al terminar puedes… | Lo demuestra |
|---|---|---|
| 1 | **Operar** dos tensores de la misma forma número por número, y **comprobarlo** contra un `for` | Ejercicio 23 |
| 2 | **Predecir**, antes de correr, si dos formas se pueden «estirar» para operar juntas y qué forma sale | Ejercicios 22 y 25 |
| 3 | **Calcular** con `@` un total por renglón y **comprobarlo** de dos maneras: con un `for` y con `sum(dim=1)` | Ejercicio 24 |
| 4 | **Reconocer** cuándo `*` y `@` dan resultados distintos sin marcar error | Ejercicio 22 |

**Qué necesitas saber antes:** las sesiones 1 a 5: crear un tensor, leer su `shape` y su `dtype`, sacarle pedazos con índices, sumar con `sum(dim=)` y cambiarlo de forma con `reshape`. **Qué necesitas tener:** el entorno de la sesión 2, con PyTorch instalado, y un archivo nuevo, `operar.py`, al que le vas a ir agregando código paso por paso. Cuando un paso te pida provocar un error, agrega la línea, córrelo, lee el error y bórrala para seguir.

---

## Por qué importa

Mateo podría hacer la cuenta como en las sesiones pasadas: un `for` dentro de otro `for`, casilla por casilla. Sirve. Pero así no trabaja un modelo de inteligencia artificial. Y tampoco el sistema que suspendió la terminal de Norte un domingo a medianoche.

Buena parte del trabajo de un modelo se va en dos operaciones: **multiplicar tensores número por número** y **multiplicar matrices**. Cuando un banco revisa una compra, el modelo toma la lista de números de la compra y la multiplica por una tabla de «pesos», que son números que el modelo aprendió. Eso es una multiplicación de matrices. Y no lo hace una vez: lo hace millones de veces por segundo. En algún lugar del banco, una cuenta así fue la que marcó los cobros de Norte como «inusuales».

Por eso existen las tarjetas de video para IA. Una analogía que sirve: la computadora normal es un puñado de chefs muy hábiles; la tarjeta de video son miles de ayudantes que solo saben picar, pero pican todos al mismo tiempo. Multiplicar matrices es «picar»: muchísimas multiplicaciones y sumas que se pueden repartir entre muchos ayudantes a la vez. AlexNet, la red de 2012 con la que arrancó la ola actual de la IA, según el propio artículo de sus autores «tarda entre cinco y seis días en entrenarse en dos GPU GTX 580 de 3 GB» ([Krizhevsky, Sutskever y Hinton, 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)). Dos tarjetas de videojuegos.

Hoy todo corre en tu computadora, sin tarjeta de video. Las dos operaciones son las mismas, y en la sesión que viene las vas a necesitar para entrenar tu propio modelo. Esta noche, Mateo las necesita para una pregunta más simple: ¿cuánto dinero dicen las piezas?

## El caso: ahora doña Lupe cuenta piezas

Hasta ahora, los números de doña Lupe eran pesos. Esta vez Mateo le pidió algo distinto: **cuántas piezas** vendió el Centro cada día de cada producto. Y aparte, su lista de precios, que ella dicta de memoria: el cuaderno a $40, la pluma a $10 y la mochila a $350.

—No han cambiado desde el regreso a clases —dice—. Y no van a cambiar hasta que suba el papel o me muera, lo que pase primero.

Con las piezas y los precios, cualquier cosa de dinero sale de una cuenta. Mateo empieza por el Centro, la sucursal que nadie sospecha: si la cuenta sale bien ahí, podrá confiar en ella cuando llegue a Norte.

### Paso 1 — Las piezas y los precios

Escribe esto al principio de `operar.py`. Primero las listas de Python, que ya conoces; después los tensores, con lo que viste en la sesión 2:

```python
import torch

lista_unidades = [
    [30, 80, 2],    # lunes
    [40, 90, 3],    # martes
    [25, 70, 2],    # miércoles
    [35, 90, 2],    # jueves
    [50, 120, 3],   # viernes
    [30, 80, 1],    # sábado
    [20, 50, 1],    # domingo
]
lista_precios = [40, 10, 350]   # cuaderno, pluma, mochila

unidades = torch.tensor(lista_unidades)
precios = torch.tensor(lista_precios)

print(unidades.shape, unidades.dtype)
print(precios.shape, precios.dtype)
```

Salida:

```
torch.Size([7, 3]) torch.int64
torch.Size([3]) torch.int64
```

Léelo como en la sesión 2: `unidades` es una tabla de 7 días por 3 productos; `precios` es una lista de 3 precios, uno por producto. Las dos guardan enteros. Guarda las listas de Python: te van a servir para comprobar con `for`.

Antes de seguir, una pregunta para que la pienses: el lunes se vendieron 30 cuadernos a $40. ¿Cuánto dinero entró por cuadernos el lunes? Si dijiste $1,200, ya sabes hacer la cuenta de hoy. Lo único nuevo es hacerla con los 21 números a la vez.

### Paso 2 — Elemento por elemento: dos tablas iguales

Ahora el archivo de Fer. Mateo lo abre con cuidado: en la sesión 2, un archivo de Norte traía un renglón chueco. Este viene completo. Misma forma que el del Centro: 7 días, 3 productos. Antes de nada, Mateo quiere ver las dos sucursales juntas, casilla con casilla. Agrega a tu archivo:

```python
lista_norte = [
    [10, 20, 1],
    [10, 25, 0],
    [5, 20, 1],
    [10, 15, 0],
    [15, 30, 1],
    [10, 20, 0],
    [5, 10, 1],
]
unidades_norte = torch.tensor(lista_norte)

total = unidades + unidades_norte
print(total)
```

Salida:

```
tensor([[ 40, 100,   3],
        [ 50, 115,   3],
        [ 30,  90,   3],
        [ 45, 105,   2],
        [ 65, 150,   4],
        [ 40, 100,   1],
        [ 25,  60,   2]])
```

El `+` tomó la casilla del lunes-cuadernos del Centro (30) y la sumó con la casilla del lunes-cuadernos de Norte (10): 40. Y así con las 21 casillas, cada una con su pareja. A esto se le llama operar **elemento por elemento**: cada número se opera con el que está en el mismo lugar del otro tensor.

Es como empalmar dos hojas de papel cuadriculado contra la ventana: cada cuadrito queda encima de su pareja. Por eso las dos hojas tienen que tener la misma forma.

Comprueba contra la versión con `for`. Agrega:

```python
total_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(lista_unidades[d][p] + lista_norte[d][p])
    total_for.append(renglon)

assert total.shape == (7, 3), "Deberían salir 7 días × 3 productos"
assert total.tolist() == total_for, "El tensor y el for deberían dar lo mismo"
print("Todo cuadra")
```

```
Todo cuadra
```

Como en la sesión 5, `tolist()` regresa el tensor a listas de Python para compararlo contra el `for`. Primero la forma, después el valor.

**Ojo: las listas de Python no hacen esto.** Pruébalo en la terminal de Python:

```python
print([1, 2, 3] + [10, 20, 30])
print(torch.tensor([1, 2, 3]) + torch.tensor([10, 20, 30]))
print([1, 2, 3] * 2)
print(torch.tensor([1, 2, 3]) * 2)
```

```
[1, 2, 3, 10, 20, 30]
tensor([11, 22, 33])
[1, 2, 3, 1, 2, 3]
tensor([2, 4, 6])
```

Con listas, `+` **pega** una lista detrás de la otra y `* 2` la **repite**. Con tensores, `+` suma casilla por casilla y `* 2` multiplica cada número. Si un día tu resultado sale el doble de largo en vez del doble de grande, estabas operando listas.

Mateo se queda viendo la última columna de Fer: 1, 0, 1, 0, 1, 0, 1. Cuatro mochilas en toda la semana. Lo más caro de la tienda, casi quieto, igual que en la sesión 3. Lo anota en un post-it. No sabe todavía para qué.

### Paso 3 — El estiramiento automático: piezas por precios

Ahora sí, la cuenta que importa. Las piezas son una tabla de `(7, 3)`; los precios, una lista de `(3,)`. No tienen la misma forma. Aun así, agrega esto:

```python
ingresos = unidades * precios
print(ingresos)
print(ingresos.shape)
```

```
tensor([[1200,  800,  700],
        [1600,  900, 1050],
        [1000,  700,  700],
        [1400,  900,  700],
        [2000, 1200, 1050],
        [1200,  800,  350],
        [ 800,  500,  350]])
torch.Size([7, 3])
```

Funcionó. Lunes: 30 cuadernos × $40 = $1,200; 80 plumas × $10 = $800; 2 mochilas × $350 = $700. Y lo mismo para los otros seis días.

¿Qué pasó? PyTorch vio que a la lista de precios le faltaba la dimensión de los días, y la **estiró**: hizo como si la lista de precios estuviera escrita siete veces, una debajo de otra, una por día.

```
   unidades (7, 3)          precios (3,)  →  estirado a (7, 3)
  30   80   2                40  10  350
  40   90   3                40  10  350
  25   70   2                40  10  350
  35   90   2      ×         40  10  350
  50  120   3                40  10  350
  30   80   1                40  10  350
  20   50   1                40  10  350
```

Y ya con las dos hojas del mismo tamaño, multiplicó elemento por elemento, como en el paso 2.

A esto en inglés le dicen ***broadcasting***; en español le vamos a decir **estiramiento automático**. Lo importante: PyTorch **no copia** los precios siete veces en la memoria. Solo hace la cuenta *como si* estuvieran copiados. Con 7 días da igual; con 7 millones de compras, esa diferencia es la que hace que tu computadora no se trabe.

Es como la cartulina de precios que doña Lupe tiene pegada detrás del mostrador, escrita con plumón: hay una sola, y sirve para cobrar todas las ventas del día. Nadie la vuelve a escribir para cada cliente.

Comprueba contra el `for`:

```python
ingresos_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(lista_unidades[d][p] * lista_precios[p])
    ingresos_for.append(renglon)

assert ingresos.shape == (7, 3), "Los ingresos deberían ser 7 días × 3 productos"
assert ingresos.tolist() == ingresos_for, "El tensor y el for deberían dar lo mismo"
print("Ingresos correctos")
print(ingresos.sum())
```

```
Ingresos correctos
tensor(19900)
```

Mira el `for` con calma: en `lista_precios[p]` solo aparece el producto, no el día. Eso es exactamente lo que hizo el estiramiento: usar el mismo precio para todos los días. Y el total de la semana, $19,900, sale con el `sum` que ya conoces de la sesión 4.

$19,900 en el Centro. La cuenta está probada dos veces: con tensores y con `for`. Ahora Mateo tiene una herramienta en la que puede confiar. Pero antes de usarla con Norte, quiere saber dónde se rompe.

### Paso 4 — La regla: ¿cuándo se puede estirar?

PyTorch no estira cualquier cosa. Sigue una regla, la misma que usa NumPy (la biblioteca de la que PyTorch copió su forma de trabajar) y que también sigue TensorFlow. Dice así:

> **Se comparan las formas de derecha a izquierda. Dos tamaños son compatibles si son iguales o si uno de los dos es 1.**

Y una parte extra: si a una forma le faltan dimensiones a la izquierda, PyTorch le pone unos (1) ahí, y esos unos se estiran.

Aplícala a piezas por precios. Escribe las formas alineadas **a la derecha**, como cuando sumas a mano en la primaria:

```
unidades   7   3
precios        3
           ↑   ↑
           │   └─ 3 contra 3: iguales, compatibles
           └───── 7 contra nada: a precios le falta, se le pone 1 y se estira a 7
resultado  7   3
```

Ahora un caso que **no** funciona. Doña Lupe empezó a vender colores a $25 y los agregó a su lista de precios, pero todavía no los cuenta en su tabla de piezas.

—Es que los niños los piden —se defiende—. Ya están en la cartulina. En el conteo, todavía no.

Agrega esto, córrelo y lee el error:

```python
precios_con_colores = torch.tensor([40, 10, 350, 25])   # agregó los colores
print(unidades * precios_con_colores)
```

```
Traceback (most recent call last):
  ...
RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1
```

Léelo palabra por palabra, como en la sesión 5:

- *The size of tensor a (3)*: el tamaño del tensor a, que es el de la izquierda del `*`, `unidades`, es 3…
- *must match the size of tensor b (4)*: …y tiene que coincidir con el tamaño del tensor b, `precios_con_colores`, que es 4…
- *at non-singleton dimension 1*: …en la dimensión 1, que no es de tamaño 1. (*Singleton* quiere decir «de un solo elemento». Si alguno de los dos midiera 1, se podría estirar; como ninguno mide 1, no hay nada que hacer.)

Con la regla:

```
unidades               7   3
precios_con_colores        4
                           ↑
                           3 contra 4: ni iguales ni alguno es 1 → error
```

El error te está diciendo algo cierto sobre tu negocio: tienes cuatro precios y tres columnas de piezas. ¿A qué columna le toca el precio de los colores? PyTorch no adivina, y qué bueno. Borra esas dos líneas para seguir.

### Paso 5 — Un escalar con todos: el IVA

El caso más sencillo de estiramiento lo has estado usando sin saberlo. Doña Lupe quiere ver sus ingresos con el IVA de 16 % incluido, «para que el contador no me regañe en enero». Aquí supón que los precios de su lista son antes de IVA; en un mostrador de verdad, el precio que ve el cliente ya debe traer los impuestos incluidos ([Ley Federal de Protección al Consumidor, artículo 7 bis](https://www.diputados.gob.mx/LeyesBiblio/pdf/LFPC.pdf)). Agrega:

```python
con_iva = ingresos * 1.16
print(con_iva)
print(con_iva.shape, con_iva.dtype)
print(ingresos.sum(), con_iva.sum())
```

```
tensor([[1392.,  928.,  812.],
        [1856., 1044., 1218.],
        [1160.,  812.,  812.],
        [1624., 1044.,  812.],
        [2320., 1392., 1218.],
        [1392.,  928.,  406.],
        [ 928.,  580.,  406.]])
torch.Size([7, 3]) torch.float32
tensor(19900) tensor(23084.)
```

El 1.16 es un escalar, un solo número; en la sesión 1 viste que no tiene dimensiones, su forma es `()`. Con la regla, le faltan todas las dimensiones, así que se le ponen unos y se estira a `(7, 3)`. Por eso multiplica a las 21 casillas. La semana con IVA: $23,084.

Dos cosas que notar en la salida:

- Los números ahora traen un punto (`1392.`) y el `dtype` cambió a `float32`. Multiplicar enteros por un número con decimales da decimales.
- La forma no cambió. Un escalar nunca cambia la forma de lo que multiplica.

Agrega su comprobación:

```python
assert con_iva.shape == ingresos.shape, "El IVA no cambia la forma"
assert con_iva[0][0] == 1392, "Lunes, cuadernos: 1,200 × 1.16 = 1,392"
print("IVA correcto")
```

```
IVA correcto
```

### Paso 6 — `@`: multiplicar y sumar de un jalón

—¿Y cuánto vendimos el viernes? —pregunta doña Lupe—. El viernes siempre es el bueno.

Quiere el ingreso de cada día. Ya sabrías hacerlo en dos pasos: multiplicar piezas por precios y luego sumar cada renglón con `sum(dim=1)`. Esa combinación, «multiplica parejas y suma», es tan común que tiene su propio símbolo: `@`. Se llama **producto de matrices** (en inglés, *matrix multiplication*; en PyTorch también existe como `torch.matmul`).

Agrega:

```python
por_dia = unidades @ precios
print(por_dia)
print(por_dia.shape)
```

```
tensor([2700, 3550, 2400, 3000, 4250, 2350, 1650])
torch.Size([7])
```

Siete totales, uno por día, en una sola operación. Hazlo a mano para el lunes y vas a ver qué hizo `@`:

```
lunes:   30  80   2
precios: 40  10  350

30 × 40  +  80 × 10  +  2 × 350
 1200    +   800     +   700     =  2700
```

`@` toma **un renglón** de la tabla de la izquierda y **la lista** de la derecha, multiplica las parejas y **suma todo**. Un renglón, un número. Siete renglones, siete números. Por eso la forma pasó de `(7, 3)` a `(7,)`: la dimensión de los productos se «consumió» en la suma.

Es exactamente lo que hace la cajera con el ticket: piezas por precio, renglón por renglón, y al final el total. `@` hace el ticket de cada día. Y el viernes, el día bueno, fue el más alto: $4,250.

**La regla de las formas para `@`**, con tablas y listas como las de esta sesión: el último tamaño de la izquierda tiene que ser igual al primero de la derecha, y ese tamaño desaparece:

```
unidades @ precios
 (7, 3)    (3,)
     └──┬──┘
   3 y 3: iguales, se consumen
resultado: (7,)
```

Ahora compruébalo de tres maneras. Esta es la autocorrección más importante de la sesión:

```python
# 1) Con for, como lo harías sin tensores
por_dia_for = []
for d in range(7):
    suma = 0
    for p in range(3):
        suma = suma + lista_unidades[d][p] * lista_precios[p]
    por_dia_for.append(suma)

# 2) Multiplicar y luego sumar por renglón
por_dia_dos_pasos = (unidades * precios).sum(dim=1)

assert por_dia.shape == (7,), "Debería salir un número por día"
assert por_dia.tolist() == por_dia_for, "El @ y el for deberían dar lo mismo"
assert por_dia.tolist() == por_dia_dos_pasos.tolist(), "El @ y el sum(dim=1) deberían dar lo mismo"
print("Los tres caminos dan lo mismo")
```

```
Los tres caminos dan lo mismo
```

Mira el `for`: tiene `suma = suma + …` adentro. Esa suma es la diferencia con el `for` del paso 3, que solo multiplicaba. `*` multiplica; `@` multiplica **y suma**.

**El orden importa.** Prueba al revés:

```python
print(precios @ unidades)
```

```
Traceback (most recent call last):
  ...
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 7x3)
```

*mat1 and mat2 shapes cannot be multiplied*: las formas de la matriz 1 y la matriz 2 no se pueden multiplicar. PyTorch trató los precios como una tabla de 1 × 3 y la puso contra una de 7 × 3: el último tamaño de la izquierda (3) no es igual al primero de la derecha (7). Con `@`, el que pone los renglones va a la izquierda.

### Paso 7 — La trampa más peligrosa: `*` y `@` con tablas cuadradas

Son las 9:15 p.m. y Mateo está a punto de cometer el error que casi todos cometen una vez.

Hasta aquí, cada vez que usaste el símbolo equivocado, PyTorch se quejó. Con tablas cuadradas, no.

Doña Lupe quiere comparar dos sucursales en dos productos: piezas de cuadernos y plumas, en el Centro y en Norte. En Norte vende un poco más caro, así que tiene una tabla de precios por sucursal. Lo que quiere es el ingreso de cada casilla: piezas por precio, cada una con su pareja. Eso es elemento por elemento, `*`. Pero imagina que alguien escribe `@` por costumbre:

```python
import torch
# Renglones: Centro y Norte. Columnas: cuadernos y plumas.
piezas = torch.tensor([[30, 80],
                       [10, 20]])
precio = torch.tensor([[40, 10],    # en el Centro
                       [45, 12]])   # en Norte, un poco más caro
print(piezas * precio)
print(piezas @ precio)
```

```
tensor([[1200,  800],
        [ 450,  240]])
tensor([[4800, 1260],
        [1300,  340]])
```

**Ninguno de los dos marcó error.** Los dos dieron una tabla de 2 × 2. Pero solo el primero es lo que doña Lupe quería: 30 cuadernos × $40 = $1,200. El segundo dice que en el Centro entraron $4,800 por cuadernos, cuatro veces más de lo real. ¿De dónde salió? Con dos tablas, `@` toma un renglón de la izquierda y una **columna** de la derecha, y hace el «multiplica y suma» del ticket: el renglón del Centro (30, 80) contra la columna de precios de cuadernos (40, 45) da 30 × 40 + 80 × 45 = 4,800. Mezcló las piezas de plumas del Centro con el precio de cuadernos de Norte. No tiene ningún sentido de negocio.

¿Por qué no hubo error? Porque con tablas cuadradas las dos reglas se cumplen a la vez: `(2, 2)` y `(2, 2)` son iguales, así que `*` funciona; y el último tamaño de la izquierda (2) es igual al primero de la derecha (2), así que `@` también funciona.

Es la trampa más peligrosa porque **un error que truena se arregla en cinco minutos; uno que no truena llega hasta el reporte del director.** O hasta el escritorio del licenciado Ortega. Mateo se imagina llegando al banco el 23 con una tabla que se ve perfecta y está mal. Se le enfría la espalda.

La defensa es la de toda la serie: compara contra el `for`, aunque sea de un renglón:

```python
lista_piezas = [[30, 80], [10, 20]]
lista_precio = [[40, 10], [45, 12]]
ingreso_for = []
for s in range(2):
    renglon = []
    for p in range(2):
        renglon.append(lista_piezas[s][p] * lista_precio[s][p])
    ingreso_for.append(renglon)
print(ingreso_for)
```

```
[[1200, 800], [450, 240]]
```

Coincide con `*`, no con `@`. Y para elegir el símbolo antes de escribir, una sola pregunta:

| Si quieres… | Usa | Qué le pasa a la forma |
|---|---|---|
| Cada número con su pareja (precio × piezas de la misma casilla) | `*` | Queda igual (o la del estiramiento) |
| Multiplicar parejas **y sumarlas** (el total del ticket) | `@` | La dimensión que se suma desaparece |

---

**¿Y en TensorFlow?** El estiramiento sigue la misma regla y da el mismo resultado. El `@` es más estricto: con una tabla y una lista marca error (*different ndims*, distinto número de dimensiones), y para ese caso se usa otra función, `tf.linalg.matvec`. Mismas ideas, distintos modales.

---

👉 **Si eres de Administración:** ya hiciste esto en tu hoja de cálculo. Cuando escribes `=B2*$F$1` y arrastras la fórmula hacia abajo, el precio fijo de `$F$1` se usa en todos los renglones: eso es el estiramiento. Y `@` es la función `SUMAPRODUCTO` de Excel: multiplica parejas y suma. La trampa del paso 7 también tiene su versión en Excel: una fórmula que no marca `#¡VALOR!` pero suma la columna equivocada. Por eso, en cualquier reporte, revisa un renglón a mano antes de presentarlo.

---

## 10:40 p.m. — Lo que no cuadra

Mateo hace con Norte lo mismo que hizo con el Centro: las piezas de Fer por los mismos precios, y luego la suma. Una línea. (Esa cuenta la haces tú en el ejercicio 23; no te adelanto el número.)

La compara con el correo de Ortega. La vuelve a hacer. La compara contra el `for`. Mismo resultado.

—Tía. Lo que vendió Norte en toda la semana no llega ni a la mitad de lo que cobró la terminal.

—A lo mejor casi todos pagaron con tarjeta.

—Aunque *todos* hubieran pagado con tarjeta. Aunque nadie hubiera dejado un peso en efectivo. La terminal cobró $16,850, y en Norte no salió mercancía para tanto dinero.

Doña Lupe deja las monedas.

—Entonces, ¿qué cobraron?

—Nada. Cobros sin mercancía detrás.

Afuera pasa un camión y la cortina vibra.

—Fer —dice doña Lupe, despacio—. Ella es la que cobra en Norte.

Mateo ya lo había pensado. Por eso, mientras corría el `for`, abrió el correo de la papelería. El que nadie revisa. El que doña Lupe dice que «es puro anuncio». Buscó «Norte». Buscó «Fernanda».

Ahí estaba. Martes 22 de septiembre, 6:12 p.m. Asunto: **Faltantes: no cuadra la terminal con el inventario**.

«Doña Lupe: hice el corte de la semana y la terminal marca más cobros de los que salen en mi inventario. Conté dos veces. Para lo que dice la terminal, me tendrían que faltar piezas en el estante, y no me faltan. Creo que algo anda mal con la terminal nueva. ¿Le aviso al banco o le aviso a usted? Fer.»

Sin respuesta. Sin abrir. Desde septiembre.

—Hizo la misma cuenta que yo —dice Mateo—. Piezas por precios contra lo cobrado. Con calculadora. Hace dos meses y medio.

Doña Lupe se quita los lentes y se talla los ojos.

—Y yo pensando que era ella.

—Pensábamos con sospechas, tía. Los números dicen otra cosa. Si Fer estuviera robando, no te escribe para avisarte que no cuadra.

—Mañana le hablo. —Una pausa—. Y le llevo conchas. De las de chocolate, que son las que le gustan.

Esa noche, Mateo aprende lo que ningún `assert` le iba a enseñar: **se decide con datos, no con sospechas.**

---

## Ejercicios de la sesión 6

Las soluciones están al final de esta entrega. Intenta primero; el `assert` te dice si acertaste.

**Ejercicio 22 — Predice antes de correr.**

a) Para cada par, escribe en papel si se pueden estirar y, si sí, qué forma sale. Usa la regla del paso 4: alinea a la derecha y compara. Después córrelo:

`torch.ones(...)` crea un tensor lleno de unos con la forma que le indiques. Aquí solo lo usamos para probar formas: los valores no importan.

```python
print((torch.ones(7, 3) * torch.ones(3)).shape)
print((torch.ones(2, 7, 3) * torch.ones(7, 1)).shape)
print((torch.ones(7, 1) * torch.ones(1, 3)).shape)
print((torch.ones(7, 3) * torch.tensor(2)).shape)
print((torch.ones(2, 7, 3) * torch.ones(2)).shape)
```

Ejemplo de cómo se ve uno resuelto: `(7, 3)` y `(3,)` → alineo: 3 con 3, iguales; a la lista le falta el 7, se estira → sale `(7, 3)`.

b) Con esta tabla, calcula **a mano**, en papel, `m * m` y `m @ m`. Después córrelo y compara.

```python
m = torch.tensor([[2, 1],
                  [0, 3]])
print(m * m)
print(m @ m)
```

Pista para `@`: la casilla de arriba a la derecha sale del renglón de arriba, (2, 1), contra la columna derecha, (1, 3): 2 × 1 + 1 × 3.

**Ejercicio 23 — Los ingresos de Norte.** Llena los huecos. Usa `unidades_norte` del paso 2 y los mismos `precios`.

```python
ingresos_norte = ____ * ____

ingresos_norte_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(____)
    ingresos_norte_for.append(renglon)

assert ingresos_norte.shape == (7, 3), "Norte también es 7 días × 3 productos"
assert ingresos_norte.tolist() == ingresos_norte_for, "El tensor y el for deberían dar lo mismo"
con_iva = ____
print(ingresos_norte.sum(), con_iva.sum())
print("Ejercicio 23 superado")
```

Antes de correrlo, predice: ¿cuánto entró por mochilas el martes en Norte? Mira la tabla de piezas.

**Ejercicio 24 — El Buen Fin.** Para el Buen Fin, doña Lupe le baja $5 al cuaderno, nada a la pluma y $50 a la mochila. Supón que vende las mismas piezas que esta semana.

1. Crea un tensor `descuento` con los tres descuentos (como números negativos) y súmalo a `precios` para obtener `precios_bf`.
2. Con `@`, calcula `por_dia_bf`, el ingreso de cada día con los precios del Buen Fin.
3. Escribe dos `assert`: el de la forma, y uno que compare tu `@` contra la versión con `*` y `sum(dim=1)`.
4. Calcula cuánto dejó de ganar cada día por los descuentos, y cuánto en toda la semana.

Reto: el dinero que dejó de ganar cada día se puede sacar con un solo `@`, sin usar `por_dia_bf`. ¿Con cuál?

**Ejercicio 25 — Escribe tú la regla.** Sin PyTorch, solo con Python. Escribe una función `se_pueden_estirar(forma_a, forma_b)` que reciba dos tuplas y devuelva `True` si se pueden estirar para operar juntas y `False` si no. Tiene que pasar estos `assert`:

```python
assert se_pueden_estirar((7, 3), (3,)) == True
assert se_pueden_estirar((7, 3), (7,)) == False
assert se_pueden_estirar((2, 7, 3), (7, 1)) == True
assert se_pueden_estirar((7, 3), (4,)) == False
assert se_pueden_estirar((7, 1), (1, 3)) == True
assert se_pueden_estirar((7, 3), ()) == True
print("Ejercicio 25 superado")
```

Pistas: `forma[-1]` es el último tamaño de una tupla y `forma[-2]` el penúltimo; así recorres de derecha a izquierda. Solo hace falta comparar tantos tamaños como tenga la forma más corta.

---

## Errores comunes (y cómo darte cuenta)

| Error | Cómo se ve | Cómo darte cuenta | Qué hacer |
|---|---|---|---|
| Una lista de precios con un producto de más (o de menos) | `RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1` | Los dos números entre paréntesis son distintos y ninguno es 1 | Imprime el `.shape` de los dos antes de operar; arregla los datos, no la cuenta |
| Dividir una tabla `(7, 3)` entre un total por día `(7,)` | `RuntimeError: The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 1` | La regla alinea a la derecha: puso los 7 días debajo de los 3 productos | Dale forma de columna: `por_dia.reshape(7, 1)` |
| Usar `@` donde querías `*` con tablas cuadradas | **No hay error.** Sale una tabla del tamaño esperado con números que no cuadran | Calcula una casilla a mano o compara contra el `for` | Pregúntate: ¿quiero parejas (`*`) o parejas sumadas (`@`)? |
| Poner al revés los lados de `@` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 7x3)` | El último tamaño de la izquierda no es igual al primero de la derecha | El que pone los renglones va a la izquierda: `unidades @ precios`. Girar con `.T` no sirve aquí: una lista `(3,)` sigue siendo `(3,)` |
| Mezclar enteros y decimales en `@` | `RuntimeError: addmv input tensors must have the same dtype, but got Float, Long, and Float` | Revisa el `.dtype` de los dos | Convierte el de enteros con `.float()` |
| Esperar que una lista de Python opere como tensor | `[1, 2, 3] * 2` da `[1, 2, 3, 1, 2, 3]` | El resultado sale más largo, no más grande | Convierte primero con `torch.tensor(...)` |

---

## Repaso para la próxima sesión

La sesión 7 empieza con estas tres preguntas. Contéstalas sin ver la lección:

1. ¿Cuándo dos formas se pueden «estirar» para operar juntas?
2. ¿Qué diferencia hay entre `*` y `@`?
3. ¿Qué forma sale de una tabla `(7, 3) @` un vector `(3,)`?

🔁 **La próxima semana:** Mateo ya puede probar que esos cobros no salieron de la papelería. Lo que no sabe es de dónde salieron. Para eso va a construir un detector de fraude que aprende de los datos: le das miles de compras inventadas con una regla escondida, y el modelo la descubre por su cuenta, a puro `@` y estiramiento automático, las dos operaciones de hoy.

*Los cobros no venían de la papelería. Venían de otro lado. Y el banco solo veía la terminal.*

---

## Soluciones (no las veas antes de intentarlo)

**Repaso para abrir.**

1. Cuando el total de números es igual antes y después: una forma `(12,)` cabe en `(4, 3)` porque 12 = 4 × 3, pero no en `(5, 3)`.
2. Que PyTorch calcule ese tamaño por ti, con los números que sobran. Solo puede haber un `-1` por `reshape`.
3. Porque `reshape` reacomoda los números en el orden en que están guardados, renglón por renglón; girar la tabla, que los renglones se vuelvan columnas, es `.T`.

**Ejercicio 22.**

a) Salida real:

```
torch.Size([7, 3])
torch.Size([2, 7, 3])
torch.Size([7, 3])
torch.Size([7, 3])
Traceback (most recent call last):
  ...
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 2
```

- `(7, 3)` y `(3,)`: 3 con 3; al de la derecha se le pone 1 y se estira a 7 → `(7, 3)`.
- `(2, 7, 3)` y `(7, 1)`: 3 con 1, se estira; 7 con 7; al de la derecha le falta el 2 → `(2, 7, 3)`.
- `(7, 1)` y `(1, 3)`: los dos se estiran → `(7, 3)`.
- `(7, 3)` y un escalar `()`: al escalar le faltan todas, se estira → `(7, 3)`.
- `(2, 7, 3)` y `(2,)`: el 2 se alinea contra el 3 de la derecha, no contra el 2 de la izquierda. 3 contra 2: error. Este es el que engaña: el 2 «se parece» a las sucursales, pero la regla compara por la derecha. Fíjate que el error dice *dimension 2*: en una forma de tres dimensiones, la dimensión 2 es la última.

b)

```
tensor([[4, 1],
        [0, 9]])
tensor([[4, 5],
        [0, 9]])
```

`m * m`: cada número al cuadrado. `m @ m`: arriba a la izquierda, 2 × 2 + 1 × 0 = 4; arriba a la derecha, 2 × 1 + 1 × 3 = 5; abajo a la izquierda, 0 × 2 + 3 × 0 = 0; abajo a la derecha, 0 × 1 + 3 × 3 = 9. Mira qué traicionero: las dos tablas coinciden en tres de cuatro casillas. Si solo revisaras la de arriba a la izquierda, creerías que daban lo mismo.

**Ejercicio 23.** Los huecos son `unidades_norte` y `precios`; `lista_norte[d][p] * lista_precios[p]`, y `ingresos_norte * 1.16`. La salida:

```
tensor(5400) tensor(6264.)
Ejercicio 23 superado
```

Las mochilas del martes en Norte: 0 piezas, así que $0. Si imprimes `ingresos_norte`, el martes es `[400, 250, 0]`.

**Ejercicio 24.**

```python
descuento = torch.tensor([-5, 0, -50])
precios_bf = precios + descuento
por_dia_bf = unidades @ precios_bf
print(precios_bf)
print(por_dia_bf)

assert por_dia_bf.shape == (7,), "Debería salir un número por día"
assert por_dia_bf.tolist() == (unidades * precios_bf).sum(dim=1).tolist(), "El @ y el sum(dim=1) deberían dar lo mismo"

dejo_de_ganar = unidades @ precios - por_dia_bf
print(dejo_de_ganar)
assert dejo_de_ganar.tolist() == (-(unidades @ descuento)).tolist()
print(dejo_de_ganar.sum())
print("Ejercicio 24 superado")
```

```
tensor([ 35,  10, 300])
tensor([2450, 3200, 2175, 2725, 3850, 2150, 1500])
tensor([250, 350, 225, 275, 400, 200, 150])
tensor(1850)
Ejercicio 24 superado
```

`precios + descuento` es elemento por elemento: dos listas de `(3,)`. En la semana dejó de ganar $1,850. El reto: `unidades @ descuento` da lo que dejó de ganar cada día, con signo negativo, porque los descuentos son negativos; el `-` de adelante lo vuelve positivo. Es lo que comprueba el último `assert`.

**Ejercicio 25.** Una solución posible:

```python
def se_pueden_estirar(forma_a, forma_b):
    cuantos = min(len(forma_a), len(forma_b))
    for i in range(1, cuantos + 1):
        a = forma_a[-i]
        b = forma_b[-i]
        if a != b and a != 1 and b != 1:
            return False
    return True
```

Con los `assert` del enunciado debajo, la salida es:

```
Ejercicio 25 superado
```

Recorre de derecha a izquierda con índices negativos: `-1`, `-2`… Solo compara tantos tamaños como tenga la forma más corta, porque a la otra le faltan dimensiones y esas siempre se pueden estirar. Si encuentra una pareja que no es igual y en la que ninguno es 1, ya no hay nada que hacer.

---

## Lo que sigue

En la sesión 7 juntas todo lo que llevas: un tensor de compras, un `@` contra una tabla de pesos y un estiramiento para sumar un ajuste, repetido hasta que el modelo aprende. Si quieres adelantar, vuelve a hacer el paso 6 a mano con el martes: si te sale $3,550 sin ver, estás listo.

---

*Escribo desde lo que vemos todos los días en Hábil, donde construimos sistemas para banca, seguros y empresas en México. Si en tu trabajo o en el negocio de tu familia se topan con un problema así, platiquemos: [habil.mx](https://habil.mx)*
