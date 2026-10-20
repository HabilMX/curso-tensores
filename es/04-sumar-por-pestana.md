# Tensores desde cero (4): sumar por pestaña

*Totales, promedios y el mejor día de la semana, sacados de un tensor en una sola línea, y la regla que nunca falla: el eje que sumas desaparece.*

*Tensores desde cero · lección A, sesión 4 de 8*

*Escenario: los personajes y la historia son inventados. Los datos técnicos, históricos y las cifras reales llevan su fuente.*

**Lunes 23 de noviembre, 7:40 p.m. Bodega de la sucursal Centro.**

La cortina metálica ya estaba a medio bajar. Olía a cartón nuevo y a café recalentado. Mateo tenía la laptop sobre una caja de cuadernos de raya, y doña Lupe, de pie detrás de él, no se había quitado el suéter.

—Faltan treinta días —dijo ella—. El licenciado Ortega me volvió a hablar. Quiere «algo concreto».

—Ya casi, tía.

—Eso dijiste el lunes pasado. Y el otro.

Mateo no contestó. Llevaba una semana rebanando el tensor de la papelería: un día por aquí, un producto por allá. Había visto cosas raras. El viernes de Norte, flojo como un martes. Las mochilas, quietas. Pero eran pedazos. Y con pedazos no se convence a un banco.

—Necesito ver todo junto —dijo—. No un día. No un producto. La semana entera de cada sucursal, en un solo número.

—¿Y eso la computadora lo hace?

—Lo hace en una línea. Si le digo bien qué juntar.

Doña Lupe se sentó en un bote de pintura volteado.

—Te lo digo desde ahorita, mijo: Norte está en la avenida, junto a la parada del camión. Por esa puerta entra el doble de gente que por esta. Si vende menos que aquí, quiero saber cuánto menos.

Mateo puso los dedos sobre el teclado. Hasta hoy había aprendido a **sacar pedazos** del tensor. Hoy le tocaba lo contrario: **resumir**. Juntar muchos números en uno solo, a lo largo del eje que él eligiera.

Afuera pasó un camión y la cortina vibró.

—Ándale pues —dijo doña Lupe—. Que hable tu tensor.

*Es la última sesión de la lección A. Vas a hacer lo mismo que Mateo, paso por paso.*

## Repaso para abrir

Contesta sin ver la sesión 3. Las respuestas están en las Soluciones, al final.

1. ¿Qué forma tiene `papeleria[:, 4]` y qué representa?
2. ¿Qué diferencia hay entre `papeleria[0]` y `papeleria[0:1]` en la forma?
3. ¿Cómo sacas las ventas de mochilas de las dos sucursales todos los días?

> **El código de esta sesión** está en el repositorio público del curso, en la carpeta `sesion-04`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-04). Escríbelo tú primero, porque así se aprende; el repositorio es para comparar cuando algo no te salga igual. Cada programa viene con la salida que debe darte.

## Objetivos de esta sesión

Al terminar, vas a poder hacer estas cuatro cosas. Al lado de cada una está el ejercicio que la demuestra.

| # | Al terminar puedes… | Lo demuestra |
|---|---|---|
| 1 | **Predecir** qué forma queda después de sumar un tensor sobre uno o varios ejes, antes de correr el código | Ejercicio 14 |
| 2 | **Calcular** totales y promedios por eje con `sum` y `mean`, y **arreglar** el error de `mean` con enteros | Ejercicios 15 y 17 |
| 3 | **Encontrar** el mayor o el menor a lo largo de un eje, y **en qué posición** está, con `max`, `min`, `argmax` y `argmin` | Ejercicios 16 y 17 |
| 4 | **Comprobar** cualquier resultado con tensores contra el mismo cálculo hecho con `for` | Ejercicios 15, 16 y 17 |

---

## Qué necesitas

- El entorno virtual de la sesión 2, con PyTorch instalado, activado.
- Un archivo `papeleria.py` con los datos de doña Lupe. Si ya lo tienes de la sesión 3, úsalo. Si no, cópialo así:

```python
import torch

# eje 0 = sucursal (0 Centro, 1 Norte)
# eje 1 = día      (0 lunes ... 6 domingo)
# eje 2 = producto (0 cuadernos, 1 plumas, 2 mochilas)
centro = [
    [1200, 800, 500],    # lunes
    [1500, 900, 700],    # martes
    [1000, 700, 500],    # miércoles
    [1300, 900, 600],    # jueves
    [2000, 1200, 700],   # viernes
    [1200, 800, 400],    # sábado
    [800, 500, 300],     # domingo
]
norte = [
    [300, 200, 100],
    [400, 250, 150],
    [350, 200, 100],
    [300, 150, 100],
    [500, 300, 200],
    [300, 200, 100],
    [200, 100, 100],
]
papeleria = torch.tensor([centro, norte])
```

Cada paso de hoy es un archivo nuevo en la misma carpeta, que empieza con `from papeleria import ...`. Así no copias los datos cada vez.

---

## Por qué importa

Un tensor guarda el detalle. Cada casilla, cada día, cada producto. Pero el detalle no acusa a nadie ni salva a nadie: lo que sirve son **respuestas**. ¿Cuánto vendió cada sucursal? ¿Qué día se vende más? ¿Cuánto entra en promedio al día?

Mateo se dio cuenta de algo esa noche: las tres preguntas son la misma operación. **Juntar muchos números en uno, a lo largo de una dirección.** En Excel es la columna de totales al final de cada renglón, o la fila de totales abajo. En un tensor se llama **reducir**, porque después de la operación quedan menos números.

Y no es un truco de papelería. Los sistemas de inteligencia artificial reducen todo el tiempo. Si un banco tiene las compras de 1,000 clientes en 30 días, con 5 datos por compra (forma `(1000, 30, 5)`, como en la sesión 1), y promedia sobre los días, obtiene cómo compra normalmente cada cliente: una tabla de `(1000, 5)`. Una compra que se aleja mucho de ese promedio es la que conviene revisar. Justo lo que el licenciado Ortega está haciendo, del otro lado del teléfono, con la terminal de Norte.

En las tiendas pasa lo mismo a lo grande. En 2020, una competencia internacional de pronósticos llamada M5 pidió adivinar las ventas diarias de 10 tiendas de Walmart en tres estados de Estados Unidos: 42,840 series de ventas organizadas por niveles (fuente: [M5 Forecasting - Accuracy, Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy); resultados en *International Journal of Forecasting*, [doi:10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013)). Sumar a lo largo de un eje, como vas a hacer hoy, es la forma de pasar de un nivel al de arriba.

## El caso: la papelería de doña Lupe, otra vez

El tensor `papeleria` tiene forma `(2, 7, 3)`: 2 sucursales, 7 días, 3 productos. Son los mismos 42 números de siempre. Esta noche Mateo les va a hacer las preguntas que de verdad le quitan el sueño a doña Lupe.

### Paso 1 — El total de todo

Mateo empezó por lo más sencillo, para calentar: ¿cuánto vendieron las dos sucursales juntas en toda la semana? Crea `paso1.py`:

```python
from papeleria import papeleria

total = papeleria.sum()
print(total)
print(total.shape)
print(total.item())
```

Salida:

```
tensor(23100)
torch.Size([])
23100
```

`sum()` sin nada adentro junta **los 42 números** en uno solo: 23,100 pesos. Fíjate en la forma: `torch.Size([])`, vacía. Es un escalar, como el `()` de la sesión 1, y `.item()` te lo da como número normal de Python. Cuadra: 18,500 del Centro más 4,600 del Norte.

—¿Veintitrés mil cien? —doña Lupe frunció la boca—. Eso ya lo sé, mijo. Eso me lo dice el cuaderno.

Tenía razón. Un solo número lo esconde todo. Ella quiere saber **cuánto vendió cada sucursal**, **cada día**, **cada producto**. Para eso le dices a `sum` a lo largo de qué eje juntar.

### Paso 2 — Una pestaña: sumar a lo ancho y a lo largo

Antes de meterse con el cubo entero, Mateo hizo lo que haría cualquier detective con prisa pero sin ganas de equivocarse: probar en terreno conocido. Una sola pestaña, la del Centro. Una tabla de 7 días × 3 productos. Con una tabla hay dos maneras de sumar, y las conoces de Excel:

```
             Cuad.  Plumas  Moch.   │ total del día
   Lun       1200    800    500     │  2500
   Mar       1500    900    700     │  3100
   ...                              │  ...
   Dom        800    500    300     │  1600
   ─────────────────────────────────┘
   total     9000   5800   3700
   del producto
```

- **A lo ancho** (la columna de la derecha): juntas los 3 productos de cada día. Quedan 7 números, uno por día.
- **A lo largo** (el renglón de abajo): juntas los 7 días de cada producto. Quedan 3 números, uno por producto.

En PyTorch le dices cuál con `dim=`, que quiere decir **dimensión**, es decir, eje. Recuerda cómo se numeran los ejes de la pestaña: el eje 0 son los días (los renglones) y el eje 1 son los productos (las columnas).

Crea `paso2.py`:

```python
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
```

Salida:

```
torch.Size([7, 3])
tensor([2500, 3100, 2200, 2800, 3900, 2400, 1600])
torch.Size([7])
[2500, 3100, 2200, 2800, 3900, 2400, 1600]
Por día: cuadra
```

—Esos números me los sé de memoria —dijo doña Lupe—. Lunes dos mil quinientos, martes tres mil cien…

Exacto. Son la lista `por_dia` que ella le dictó a Mateo en la sesión 1. El tensor y el `for` dan lo mismo. Primera prueba superada: la herramienta no miente en terreno conocido.

Ahora fíjate en la forma, que es lo importante de hoy. Entraste con `(7, 3)`. Sumaste con `dim=1`. Saliste con `(7,)`. **El 3, que era el eje 1, desapareció.** Lógico: juntaste los tres productos en uno, así que ya no hay «eje de productos».

Una herramienta nueva en ese código: `.tolist()`. Es pariente de `.item()`: `.item()` convierte un tensor de un solo número en número de Python, y `.tolist()` convierte un tensor de varios números en lista de Python. La usamos para que el `assert` pueda comparar lista contra lista.

Ahora a lo largo. En la terminal de Python, con `from papeleria import papeleria`:

```
>>> papeleria[0].sum(dim=0)
tensor([9000, 5800, 3700])
>>> papeleria[0].sum(dim=0).shape
torch.Size([3])
```

Entraste con `(7, 3)`, sumaste con `dim=0` y saliste con `(3,)`: **el 7, que era el eje 0, desapareció.** Juntaste los siete días y quedó un total por producto. (En el ejercicio 15 vas a comprobar algo parecido con un `for`.)

Mateo se quedó viendo las dos formas. `(7,)`. `(3,)`. Y escribió en un post-it, que pegó en la tapa de la laptop, la regla de toda la sesión. Apréndetela tal cual:

> **El eje que reduces desaparece de la forma.** Los demás se quedan, en el mismo orden.

Un truco para no confundirte: `dim=` no dice **qué te queda**, dice **qué juntas**. `dim=1` no es «dame los productos»; es «junta los productos». Por eso los productos desaparecen.

—O sea que lo que juntas se va —resumió doña Lupe—. Como el aguinaldo.

### Paso 3 — El cubo completo: tres maneras de sumar

Ahora sí, el cubo completo. `(2, 7, 3)`: tres ejes, así que hay tres maneras de sumar uno solo. Mateo no corrió nada todavía. Primero agarró una pluma —de las que vende la tía— y predijo cada forma en una servilleta. Hazlo tú también, con la regla: tacha el número del eje que juntas.

| Juntas | Qué eje | Forma que queda | Qué significa |
|---|---|---|---|
| las sucursales | `dim=0` | ~~2~~, 7, 3 → `(7, 3)` | cada día y producto, con las dos sucursales juntas |
| los días | `dim=1` | 2, ~~7~~, 3 → `(2, 3)` | cada sucursal y producto, en toda la semana |
| los productos | `dim=2` | 2, 7, ~~3~~ → `(2, 7)` | cada sucursal y día, con todos los productos |

Ahora compruébalo. Crea `paso3.py`:

```python
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
```

Salida:

```
torch.Size([7, 3])
tensor([[1500, 1000,  600],
        [1900, 1150,  850],
        [1350,  900,  600],
        [1600, 1050,  700],
        [2500, 1500,  900],
        [1500, 1000,  500],
        [1000,  600,  400]])
torch.Size([2, 3])
tensor([[9000, 5800, 3700],
        [2350, 1400,  850]])
torch.Size([2, 7])
tensor([[2500, 3100, 2200, 2800, 3900, 2400, 1600],
        [ 600,  800,  650,  550, 1000,  600,  400]])
```

Léelos como doña Lupe:

- **`dim=0`** encima una pestaña sobre la otra y suma casilla por casilla. El lunes, en cuadernos, Centro vendió 1,200 y Norte 300: juntos, 1,500. Es la primera casilla. Desaparecieron las sucursales: es como si doña Lupe tuviera una sola tienda grande.
- **`dim=1`** junta la semana. El renglón de arriba es el Centro: 9,000 de cuadernos, 5,800 de plumas, 3,700 de mochilas, los mismos del paso 2. El de abajo es el Norte. Desaparecieron los días.
- **`dim=2`** junta los productos. El renglón de arriba es el `por_dia` del Centro que ya conoces; el de abajo, el del Norte. Desaparecieron los productos.

Las tres predicciones de la servilleta cuadraron. Esa es la ventaja de la regla: **sabes la forma antes de correr el código.** Y si la forma que sale no es la que predijiste, sabes que juntaste el eje equivocado.

Pero Mateo ya no estaba viendo las formas. Estaba viendo el renglón de abajo de `dim=1`. *2350, 1400, 850.* La semana entera de Norte, producto por producto. Ni siquiera llegaba a lo que el Centro vende de cuadernos solos.

Le faltaba una cuenta. La que había venido a hacer.

### Paso 4 — Juntar dos ejes a la vez: el total de cada sucursal

La pregunta de doña Lupe, la de la puerta por la que entra el doble de gente: ¿cuánto vendió cada sucursal? Para eso hay que juntar **los días y los productos**: dos ejes a la vez. `sum` acepta una tupla de ejes. Crea `paso4.py`:

```python
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
```

Salida:

```
torch.Size([2])
tensor([18500,  4600])
[18500, 4600]
Por sucursal: cuadra
```

18,500 el Centro y 4,600 el Norte.

Los números estaban en la hoja desde la sesión 1. Pero nunca los habían puesto así, uno junto al otro, con la puerta de Norte en la cabeza.

Doña Lupe se acercó a la pantalla. No dijo nada durante un rato largo.

—Cuatro veces —dijo al fin—. El Centro vendió alrededor de cuatro veces lo que Norte; Norte, cerca de una cuarta parte. Y por su puerta entra el doble de gente.

—Tía, esto no dice por qué. Nada más dice cuánto.

—Ajá. —Ella ya estaba mirando su celular—. ¿Y quién cuadra la caja de Norte todas las noches, Mateo?

Él sabía la respuesta. Todos la sabían. Fer.

La regla, mientras tanto, sigue igual: 2, ~~7~~, ~~3~~ → `(2,)`. Tachaste dos números, desaparecieron dos ejes.

El `for` necesitó **tres ciclos**, uno dentro de otro. El tensor, una línea. Y no solo se escribe más corto: con arreglos grandes el `for` de Python puede tardar mucho más que `sum` de PyTorch. Cuánto más depende de tu computadora, de la versión y del tamaño, así que mídelo en tu equipo antes de citar una cifra.

**Cuidado si lo haces en dos pasos.** Mateo, con las manos frías, quiso volver a sacar la cuenta de otra manera, por si acaso: sumar primero un eje y luego el otro. Pero ojo: **después de la primera suma, los ejes se vuelven a numerar.** Mira este intento. Crea `paso4c.py`:

```python
from papeleria import papeleria

paso_a = papeleria.sum(dim=1)     # quito los días: queda (2, 3)
print(paso_a.shape)
paso_b = paso_a.sum(dim=2)        # ¿quito los productos?
```

Salida:

```
torch.Size([2, 3])
Traceback (most recent call last):
  File ".../paso4c.py", line 5, in <module>
    paso_b = paso_a.sum(dim=2)        # ¿quito los productos?
             ^^^^^^^^^^^^^^^^^
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

Léelo palabra por palabra: «dimensión fuera de rango: esperaba algo entre -2 y 1, y me diste 2». Después de quitar los días, `paso_a` tiene forma `(2, 3)`: solo dos ejes, el 0 y el 1. Los productos ya no son el eje 2; ahora son el eje 1. (Los números negativos cuentan los ejes desde el final; por ahora no los necesitas.)

La solución fácil: haz las dos cosas en una sola llamada, `dim=(1, 2)`. Ahí los números se refieren siempre a la forma original.

Mateo volvió a la llamada de una línea. 4,600. El número no se movió.

En la otra esquina de la bodega, doña Lupe marcaba un número. Lo dejó sonar. Colgó. Volvió a marcar.

### Paso 5 — El promedio diario, y el error que vas a ver seguro

—Sácame cuánto vende Norte en un día normal —dijo doña Lupe sin soltar el teléfono—. Para decírselo a ella en la cara.

O sea: ¿cuánto se vende **en promedio** al día en cada sucursal? El promedio se saca con `mean` (en inglés, «media»). Funciona igual que `sum`, con `dim=`. Primero sacas las ventas de cada día (juntando productos) y luego promedias los 7 días. Crea `paso5a.py`:

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
print(ventas_diarias.dtype)
promedio_diario = ventas_diarias.mean(dim=1)
```

Salida:

```
torch.int64
Traceback (most recent call last):
  File ".../paso5a.py", line 5, in <module>
    promedio_diario = ventas_diarias.mean(dim=1)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
```

Tronó.

—¿Ahora qué? —preguntó doña Lupe.

—Nada grave, tía. Se queja de que le di enteros.

—Pues como todos. A mí también me gustan enteros.

Léelo despacio, traducido:

- *mean(): could not infer output dtype*: «`mean` no supo de qué tipo hacer el resultado».
- *Input dtype must be either a floating point or complex dtype*: «lo que le entra tiene que ser de decimales» (*floating point* son los números con punto decimal; *complex* son otra clase de números que no vas a usar aquí).
- *Got: Long*: «y me diste `Long`». `Long` es como PyTorch llama por dentro a `int64`, el tipo de enteros que viste en la sesión 2. Por eso imprimimos el `dtype` antes: `torch.int64`.

¿Por qué se niega? Porque un promedio casi nunca sale entero: 18,500 entre 7 da 2,642.857…, y un tensor de enteros no tiene dónde guardar el «.857». En lugar de cortarlo a escondidas, PyTorch se detiene y te avisa. A Mateo eso le gustó. En esta historia ya había demasiadas cosas pasando a escondidas.

Hay dos maneras de arreglarlo.

**Arreglo 1: convertir a decimales con `.float()`.** Crea `paso5b.py`:

```python
from papeleria import papeleria, centro

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
decimales = ventas_diarias.float()
print(decimales.dtype)
print(decimales)

promedio_diario = decimales.mean(dim=1)     # quita el eje de los días
print(promedio_diario.shape)
print(promedio_diario)

# Con for: el total del Centro entre 7 días
total_centro = 0
for renglon in centro:
    total_centro = total_centro + sum(renglon)
print(total_centro / 7)

assert promedio_diario.shape == (2,), "Esperaba un promedio por sucursal"
assert round(promedio_diario[0].item(), 2) == round(total_centro / 7, 2), "No coincide con el for"
print("Promedio diario: cuadra")
```

Salida:

```
torch.float32
tensor([[2500., 3100., 2200., 2800., 3900., 2400., 1600.],
        [ 600.,  800.,  650.,  550., 1000.,  600.,  400.]])
torch.Size([2])
tensor([2642.8572,  657.1429])
2642.8571428571427
Promedio diario: cuadra
```

`.float()` no cambia los números, solo cómo se guardan: fíjate en el punto después de cada uno (`2500.`). Ahora son `float32`, decimales. El Centro vende en promedio $2,642.86 al día y el Norte, $657.14.

—Seiscientos cincuenta y siete pesos —repitió doña Lupe—. Con esa esquina. Eso lo vende un puesto de chicles.

Dos detalles de ese código:

- La forma: `(2, 7)` con `dim=1` dio `(2,)`. Desaparecieron los días, como siempre.
- El `assert` compara **redondeando a dos decimales**, porque el tensor dice 2642.8572 y el `for` 2642.8571428571427. Ninguno está mal: el tensor usa `float32` y Python divide con un `float` de 64 bits. Los dos son aproximaciones binarias, pero `float32` conserva menos precisión. Sin redondear, el `assert` fallaría con el cálculo bien hecho. Ojo: redondeamos solo para comparar o mostrar; para contabilidad real, guarda los importes en centavos como enteros o usa aritmética decimal, no `float32`.

**Arreglo 2: crear el tensor con decimales desde el principio.** Si los datos de doña Lupe vinieran con decimales, el tensor nacería de decimales y `mean` funcionaría directo (para dinero exacto, mejor centavos enteros). Crea `paso5c.py`:

```python
import torch

con_centavos = torch.tensor([2500.0, 3100.5, 2200.0])
print(con_centavos.dtype)
print(con_centavos.mean())
```

Salida:

```
torch.float32
tensor(2600.1667)
```

Basta con que los números lleven punto decimal para que PyTorch elija `float32` solo. Para la papelería, que ya existe con enteros, lo más práctico es el arreglo 1: `.float()` justo antes del `mean`.

### Paso 6 — El mejor día: `max` y `argmax`

Mateo tenía una sospecha distinta a la de su tía, y quería probarla antes de decir nada. Si alguien estuviera escondiendo ventas a lo loco, Norte se comportaría raro: días buenos donde no tocan, días muertos al azar. Así que hizo la última pregunta de la noche: ¿qué día se vende más en cada sucursal? Aquí hay **dos** respuestas que te interesan: **cuánto** se vendió ese día (el valor) y **qué día** fue (la posición). Crea `paso6.py`:

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
resultado = ventas_diarias.max(dim=1)
print(resultado)
```

Salida:

```
torch.return_types.max(
values=tensor([3900, 1000]),
indices=tensor([4, 4]))
```

Esto se ve raro, pero es sencillo. Cuando le das `dim=` a `max`, no regresa un tensor, sino **una cajita con dos cosas**:

- `values` (valores): la venta más alta de cada sucursal. 3,900 en el Centro, 1,000 en el Norte.
- `indices` (índices): **en qué posición** del eje estaba esa venta. 4 y 4.

¿Y qué es la posición 4 del eje de los días? Lunes 0, martes 1, miércoles 2, jueves 3… **viernes**. Las dos sucursales venden más el viernes.

Igual que con una tupla de Python, puedes sacar las dos cosas de un jalón, cada una en su variable. Y con una lista de nombres traduces el índice a palabras. Crea `paso6b.py`:

```python
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
```

Salida:

```
tensor([3900, 1000])
tensor([4, 4])
Centro vendió más el viernes : 3900
Norte vendió más el viernes : 1000
```

La regla de la forma también vale aquí: entraste con `(2, 7)`, juntaste los días con `dim=1`, y tanto `valores` como `indices` tienen forma `(2,)`: uno por sucursal.

Mateo se recargó en la silla. El mismo día para las dos. Norte no se portaba como una tienda descontrolada. Se portaba como el Centro, pero en chiquito. Como si a cada día, más o menos parejo, le faltara un pedazo.

Eso no se parece a alguien que saca dinero de la caja un día sí y otro no. Parece otra cosa. Pero ¿qué? Mateo todavía no tenía con qué averiguarlo, así que no dijo nada.

En el ejercicio 16 vas a comprobar un resultado así contra un `for`: es más largo, pero ya sabes escribirlo.

**Sus parientes.** Si solo quieres la posición, sin el valor, usa `argmax` («el argumento del máximo», o sea, dónde está). Si quieres el más bajo, `min` y `argmin` funcionan igual. Y si le quitas el `dim`, `max` busca en todo el tensor y regresa un solo número. Crea `paso6c.py`:

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)
print(ventas_diarias.argmax(dim=1))
print(ventas_diarias.max())
print(ventas_diarias.min(dim=1))
```

Salida:

```
tensor([4, 4])
tensor(3900)
torch.return_types.min(
values=tensor([1600,  400]),
indices=tensor([6, 6]))
```

El viernes es el mejor día de las dos sucursales; 3,900 es la venta diaria más alta de todo el tensor; y el día más flojo de las dos es el 6, el domingo. Mismo patrón, otra vez. Norte copia al Centro hasta en sus días malos.

Una diferencia que te va a ahorrar un rato: `sum` y `mean` aceptan una tupla de ejes, como `dim=(1, 2)`; **`max` y `min` no**, solo uno a la vez. Lo vas a ver en la tabla de errores.

**Para cuando lo necesites: `keepdim`.** Si le agregas `keepdim=True` a cualquiera de estas funciones, el eje que juntas no desaparece, sino que se queda con tamaño 1: `papeleria.sum(dim=2, keepdim=True)` da forma `(2, 7, 1)` en lugar de `(2, 7)`. Hoy no te hace falta; en la lección B vas a ver cuándo conviene.

### Todo en una tabla

Eran casi las nueve. Mateo copió en una sola hoja todo lo que había sacado esa noche, para que doña Lupe pudiera llevárselo al licenciado Ortega. Cada pregunta, una línea:

| Pregunta de doña Lupe | Código | Forma |
|---|---|---|
| ¿Cuánto vendí en total? | `papeleria.sum()` | `()` |
| ¿Cuánto vendió cada sucursal? | `papeleria.sum(dim=(1, 2))` | `(2,)` |
| ¿Cuánto vendió cada sucursal cada día? | `papeleria.sum(dim=2)` | `(2, 7)` |
| ¿Cuánto vendió cada sucursal de cada producto? | `papeleria.sum(dim=1)` | `(2, 3)` |
| ¿Cuánto vendí de cada producto cada día, juntando sucursales? | `papeleria.sum(dim=0)` | `(7, 3)` |
| ¿Cuánto vendo en promedio al día en cada sucursal? | `papeleria.sum(dim=2).float().mean(dim=1)` | `(2,)` |
| ¿Qué día vendo más en cada sucursal? | `papeleria.sum(dim=2).argmax(dim=1)` | `(2,)` |

---

## PyTorch y TensorFlow: la misma suma, con otro nombre

Antes de cerrar la laptop, Mateo quiso estar seguro de que el 4,600 no era un capricho de PyTorch. Corrió la misma cuenta en TensorFlow, la otra herramienta grande. Ahí las funciones se llaman `reduce_sum` y `reduce_mean` (ahí se ve la palabra «reducir»), y el eje se llama `axis` en lugar de `dim`. Este bloque es para leer, no para correr: el curso no instala TensorFlow.

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

La suma da igual: 4,600 en las dos herramientas. Pero mira el promedio: **2642 y 657, sin decimales y sin ningún aviso**. TensorFlow conserva el tipo entero y hace división entera: trunca la parte fraccionaria; PyTorch prefirió tronar. Un error al menos te avisa; un número chueco que nadie nota es peor. Si usas TensorFlow, convierte a decimales antes de promediar:

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

Mateo lo anotó con doble subrayado. Treinta días antes de una fecha límite, un número que cambia en silencio es lo último que necesitas.

---

👉 **Si eres de Administración:** lo que hizo Mateo esa noche lo haces tú con una tabla dinámica. Cuando pones «sucursal» en las filas y «suma de ventas» en los valores, Excel junta los días y los productos y te deja un número por sucursal: es `papeleria.sum(dim=(1, 2))`. Si pones «día» en las filas, junta sucursales y productos: `papeleria.sum(dim=(0, 2))`. La regla de hoy vale igual allá: lo que no pones en la tabla dinámica, desaparece del reporte. Y la lección del paso 5 también: antes de presentar un promedio, revisa que tu herramienta no haya cortado los centavos. Y una más, que la tabla dinámica no te va a decir: un total bajo te dice **cuánto** falta, nunca **quién** tiene la culpa.

---

## Ejercicios de la sesión 4

Las soluciones están al final de esta entrega. Intenta primero; el `assert` te dice si acertaste. Siempre en el mismo orden: **primero la forma, después el valor.**

**Ejercicio 14 — Predice la forma.** Escribe en papel qué forma queda en cada caso y qué significa para doña Lupe. Después convierte cada predicción en un `assert` y córrelo. El primero va resuelto:

```python
from papeleria import papeleria

assert papeleria.sum(dim=0).shape == (7, 3)          # ejemplo resuelto
assert papeleria.sum(dim=1).shape == ____
assert papeleria.sum(dim=2).shape == ____
assert papeleria.sum(dim=(0, 1)).shape == ____
assert papeleria.sum(dim=(0, 2)).shape == ____
assert papeleria.sum(dim=(0, 1, 2)).shape == ____
print("Ejercicio 14 superado")
```

Pregunta extra: ¿qué otra forma de escribir tienes para `papeleria.sum(dim=(0, 1, 2))`?

**Ejercicio 15 — El promedio por producto (a medias).** Doña Lupe quiere saber cuánto vende **en promedio al día** de cada producto en cada sucursal. Llena los dos huecos:

El prefijo `f` antes de unas comillas crea una **f-string**: deja meter un valor entre llaves dentro de un texto. Aquí hará que el mensaje del `assert` muestre la forma real si algo falla.

```python
from papeleria import papeleria, norte

promedio = papeleria.____().mean(dim=____)
print(promedio)

assert promedio.shape == (2, 3), f"Esperaba (sucursal, producto) = (2, 3) y salió {tuple(promedio.shape)}"

suma_plumas_norte = 0
for renglon in norte:
    suma_plumas_norte = suma_plumas_norte + renglon[1]

assert round(promedio[1, 1].item(), 2) == round(suma_plumas_norte / 7, 2), "No coincide con el for"
print("Ejercicio 15 superado")
```

Fíjate en el mensaje del primer `assert`: si te equivocas de eje, te dice qué forma te salió. Esa forma te dice qué eje juntaste por error.

**Ejercicio 16 — Lo mejor y lo peor de la semana.** Escribe tú el código. Para cada pregunta: un `assert` de la forma y luego el valor.

a) ¿Cuál es el día más flojo de cada sucursal, y cuánto se vendió ese día? Comprueba la respuesta del Norte con un `for`.
b) ¿Cuál es el producto estrella (el que más vendió en la semana) de cada sucursal?
c) Juntando las dos sucursales, ¿qué día se vendió más y cuánto? Debe dar 4,900.

Pista: en cada pregunta, primero decide qué ejes juntas para la suma y luego sobre qué eje buscas.

**Ejercicio 17 — Tus calificaciones.** Retoma las calificaciones del ejercicio 4 de la sesión 1. Si no guardaste las tuyas, usa estas (alumno × materia × parcial):

```python
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]
```

Conviértelas en tensor y saca, cada una con su `assert` de forma:

a) El promedio general de cada alumno.
b) El promedio de cada alumno en cada materia.
c) Qué alumno tiene el mejor promedio.
d) Cuál es la materia más difícil del grupo (la de promedio más bajo, juntando alumnos y parciales).

Comprueba el inciso a) del primer alumno con un `for`. Si usaste tus propios datos con otro orden de ejes, tus `dim` van a ser distintos a los de la solución, y está bien: lo que cuenta es que la forma que predices sea la que sale.

---

## Errores comunes (y cómo darte cuenta)

| Error | Cómo se ve | Cómo darte cuenta | Qué hacer |
|---|---|---|---|
| Pensar que `dim` dice qué te queda | Quieres un total por día y escribes `dim=1`, pensando «el eje de los días» | La forma sale `(2, 3)` en lugar de `(2, 7)` | `dim` dice qué **juntas**. Para un total por día, junta los otros ejes |
| Promediar enteros | `RuntimeError: mean(): could not infer output dtype. ... Got: Long` | El `dtype` dice `torch.int64` | `.float()` antes del `mean`, o datos con punto decimal |
| Sumar en dos pasos con la numeración vieja | `IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)` | Pediste un eje que ya no existe | Después de reducir, los ejes se renumeran. Mejor una sola llamada: `dim=(1, 2)` |
| Comparar un tensor contra una lista | `assert por_dia == [2500, 3100, ...]` lanza `AssertionError` aunque los números están bien | El `assert` truena con números que coinciden a ojo | Convierte primero: `por_dia.tolist() == [...]` |
| Comparar decimales sin redondear | `2642.857177734375 == 18500 / 7` da `False` | El `assert` truena por el último dígito | Redondea los dos lados: `round(x, 2)` |
| Darle una tupla a `max` | `TypeError: max() received an invalid combination of arguments - got (dim=tuple, ), ...` | Escribiste `max(dim=(1, 2))` | `max` y `min` aceptan un solo eje. Suma o reduce primero uno y luego busca en el otro |

---

## Autoevaluación de la lección A

Terminaste la primera lección. Antes de pasar a la lección B, revisa que puedes hacer cada cosa **sin ver las sesiones**. Si alguna te cuesta, regresa a la sesión que dice y repite su ejercicio.

**Sesión 1 — Un número, una lista, una tabla, un cubo**
- [ ] Digo cuántas dimensiones tiene un conjunto de datos y qué representa cada una.
- [ ] Escribo esos datos como listas anidadas de Python y señalo un número con sus índices.
- [ ] Calculo la forma a mano y digo cuántos números hay en total, multiplicándola.
- [ ] Reconozco una lista chueca y explico por qué no puede ser un tensor.
- [ ] Escribo un `assert` con mensaje en español para comprobar mi propio trabajo.

**Sesión 2 — Tu primer tensor**
- [ ] Convierto una lista de Python en tensor con `torch.tensor`.
- [ ] Leo `shape`, `ndim`, `dtype` y `numel()` de un tensor y explico qué dice cada uno.
- [ ] Sé qué `dtype` elige PyTorch para enteros y para decimales.

**Sesión 3 — Los ejes tienen nombre**
- [ ] Saco un número con `papeleria[1, 4, 2]` y lo convierto en número de Python con `.item()`.
- [ ] Saco una rebanada completa por cualquier eje, como `papeleria[:, 4]` o `papeleria[:, :, 2]`, y digo su forma antes de correrla.

**Sesión 4 — Sumar por pestaña**
- [ ] Predigo la forma después de reducir uno o varios ejes.
- [ ] Saco totales y promedios por eje con `sum` y `mean`, y arreglo el error de `mean` con enteros.
- [ ] Encuentro el mayor, el menor y su posición con `max`, `min`, `argmax` y `argmin`.
- [ ] Compruebo un resultado con tensores contra el mismo cálculo con `for`.

## Repaso de la lección A

Diez preguntas de las cuatro sesiones. Contéstalas sin ver nada; las respuestas están en las Soluciones.

1. ¿Cuál es la regla para saber cuántas dimensiones tiene un conjunto de datos?
2. Si doña Lupe guarda 4 semanas de ventas, por sucursal, día y producto, ¿qué forma tiene el tensor y cuántos números guarda?
3. ¿Qué hace `assert` cuando la condición es cierta, y qué hace cuando es falsa?
4. ¿Qué pasa si le das a `torch.tensor` la lista `[[1, 2, 3], [4, 5]]`?
5. ¿Qué te dicen `papeleria.ndim`, `papeleria.numel()` y `papeleria.dtype`?
6. ¿Qué `dtype` tiene `torch.tensor([1, 2, 3])`? ¿Y `torch.tensor([1.5, 2.0])`?
7. ¿Qué número da `papeleria[1, 4, 2]` y para qué sirve ponerle `.item()`?
8. ¿Qué forma tiene `papeleria[:, :, 2]` y qué representa?
9. ¿Qué forma deja `papeleria.sum(dim=0)` y qué significa para doña Lupe?
10. ¿Qué regresa `max(dim=1)` que no regresa `sum(dim=1)`?

---

## Repaso para la próxima sesión

La sesión 5 empieza con estas tres preguntas. Contéstalas sin ver la lección:

1. Si sumas `papeleria` con `dim=1`, ¿qué forma queda y qué significa?
2. ¿Por qué `mean` falla con enteros y cómo lo arreglas?
3. ¿Cómo compruebas que tu suma con tensores está bien?

🔁 **La próxima semana** arranca la lección B, **Los tensores trabajando**. Mateo va a necesitar mirar más atrás que una semana: meses enteros. Y para verlos va a tener que cambiar la forma del tensor sin cambiar sus datos: los mismos números, acomodados en otras cajas. Vas a leer también, palabra por palabra, el error que sale cuando la forma nueva no cabe.

Antes de bajar la cortina, doña Lupe le enseñó a Mateo su celular. Le había pedido a Fer el corte de caja de la semana desde las cuatro. Después vinieron las llamadas: las de la tarde, y las de esta noche. Ninguna devuelta. Mateo le pidió que esperara, que los números decían cuánto y no quién. Ella guardó el teléfono sin contestarle.

Fer no contestó el teléfono en toda la tarde.

---

## Soluciones (no las veas antes de intentarlo)

**Repaso para abrir (de la sesión 3).**

1. `papeleria[:, 4]` tiene forma `(2, 3)`: los tres productos del viernes en cada sucursal.
2. `papeleria[0]` tiene forma `(7, 3)`: la pestaña del Centro, sin el eje de sucursales. `papeleria[0:1]` tiene forma `(1, 7, 3)`: una rebanada que conserva el eje de sucursales, con una sola sucursal adentro.
3. `papeleria[:, :, 2]`: todas las sucursales, todos los días, el producto 2 (mochilas). Forma `(2, 7)`.

**Ejercicio 14.**

```python
assert papeleria.sum(dim=0).shape == (7, 3)          # cada día y producto, sucursales juntas
assert papeleria.sum(dim=1).shape == (2, 3)          # cada sucursal y producto, en la semana
assert papeleria.sum(dim=2).shape == (2, 7)          # cada sucursal y día, productos juntos
assert papeleria.sum(dim=(0, 1)).shape == (3,)       # cada producto, en todo
assert papeleria.sum(dim=(0, 2)).shape == (7,)       # cada día, en todo
assert papeleria.sum(dim=(0, 1, 2)).shape == ()      # un solo número: el total
```

Si además imprimes los valores:

```
tensor([11350,  7200,  4550])
tensor([3100, 3900, 2850, 3350, 4900, 3000, 2000])
tensor(23100)
Ejercicio 14 superado
```

Pregunta extra: `papeleria.sum(dim=(0, 1, 2))` es lo mismo que `papeleria.sum()`. Juntar todos los ejes es juntarlo todo: 23,100.

**Ejercicio 15.** Los huecos son `float` y `1`: conviertes a decimales y juntas los días.

```python
promedio = papeleria.float().mean(dim=1)
```

Salida:

```
tensor([[1285.7142,  828.5714,  528.5714],
        [ 335.7143,  200.0000,  121.4286]])
Ejercicio 15 superado
```

El Norte vende en promedio exactamente $200 de plumas al día (1,400 entre 7). Si pusiste `dim=0`, el mensaje del `assert` te dijo:

```
AssertionError: Esperaba (sucursal, producto) = (2, 3) y salió (7, 3)
```

El `(7, 3)` te delata: desaparecieron las sucursales, no los días.

**Ejercicio 16.**

```python
from papeleria import papeleria, norte

dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
productos = ["cuadernos", "plumas", "mochilas"]

# a) Día más flojo: junto productos (2, 7) y busco el mínimo a lo largo de los días
valores, indices = papeleria.sum(dim=2).min(dim=1)
assert valores.shape == (2,), "Esperaba un valor por sucursal"
assert valores.tolist() == [1600, 400], "Los mínimos deberían ser 1,600 y 400"
assert indices.tolist() == [6, 6], "El domingo debería ser el día más flojo"
print(valores, indices)

peor_dia = 0
for d in range(7):
    if sum(norte[d]) < sum(norte[peor_dia]):
        peor_dia = d
assert indices[1].item() == peor_dia, "No coincide con el for"

# b) Producto estrella: junto días (2, 3) y busco la posición del máximo en productos
estrella = papeleria.sum(dim=1).argmax(dim=1)
assert estrella.shape == (2,), "Esperaba un producto por sucursal"
assert estrella.tolist() == [0, 0], "Los cuadernos deberían ser el producto estrella"
print(productos[estrella[0].item()], productos[estrella[1].item()])

# c) Mejor día con las dos sucursales juntas: junto sucursales y productos (7,)
por_dia_juntas = papeleria.sum(dim=(0, 2))
assert por_dia_juntas.shape == (7,), "Esperaba un total por día"
print(dias[por_dia_juntas.argmax().item()], por_dia_juntas.max().item())
assert por_dia_juntas.max().item() == 4900, "El mejor día debería sumar 4,900"
assert por_dia_juntas.argmax().item() == 4, "El mejor día debería ser viernes"
print("Ejercicio 16 superado")
```

Salida:

```
tensor([1600,  400]) tensor([6, 6])
cuadernos cuadernos
viernes 4900
Ejercicio 16 superado
```

El domingo es el día más flojo de las dos sucursales (1,600 y 400), los cuadernos son el producto estrella de las dos y el viernes, juntando sucursales, se vendieron 4,900. En la c), `argmax()` sin `dim` sirve porque `por_dia_juntas` ya tiene un solo eje.

**Ejercicio 17.**

```python
import torch

parciales = torch.tensor([
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
])                                   # alumno × materia × parcial
decimales = parciales.float()

por_alumno = decimales.mean(dim=(1, 2))           # a) quito materias y parciales
assert por_alumno.shape == (3,)
por_alumno_y_materia = decimales.mean(dim=2)      # b) quito parciales
assert por_alumno_y_materia.shape == (3, 4)
por_materia = decimales.mean(dim=(0, 2))          # d) quito alumnos y parciales
assert por_materia.shape == (4,)

mejor_alumno = por_alumno.argmax()                # c)
assert mejor_alumno.shape == ()
assert mejor_alumno.item() == 1, "El segundo alumno debería tener el mejor promedio"
assert torch.allclose(por_materia, torch.tensor([8.5556, 8.3333, 8.1111, 8.4444]), atol=1e-4)
assert por_materia.argmin().item() == 2, "La tercera materia debería ser la más difícil"  # d)

print(por_alumno)
print(por_alumno_y_materia)
print(mejor_alumno)
print(por_materia, por_materia.argmin())

suma = 0
cuantas = 0
for materia in parciales[0].tolist():
    for calificacion in materia:
        suma = suma + calificacion
        cuantas = cuantas + 1
assert round(por_alumno[0].item(), 2) == round(suma / cuantas, 2), "No coincide con el for"
print("Ejercicio 17 superado")
```

Salida:

```
tensor([8.2500, 9.1667, 7.6667])
tensor([[8.6667, 8.0000, 9.3333, 7.0000],
        [9.6667, 9.0000, 8.3333, 9.6667],
        [7.3333, 8.0000, 6.6667, 8.6667]])
tensor(1)
tensor([8.5556, 8.3333, 8.1111, 8.4444]) tensor(2)
Ejercicio 17 superado
```

El segundo alumno (índice 1) tiene el mejor promedio, 9.17, y la materia más difícil es la tercera (índice 2), con 8.11. Si se te olvidó el `.float()`, viste el mismo `RuntimeError` del paso 5: las calificaciones también son enteros.

**Repaso de la lección A.**

1. El número de dimensiones es el número de datos que necesitas para señalar un solo número.
2. `(4, 2, 7, 3)`: semana, sucursal, día y producto. Guarda 4 × 2 × 7 × 3 = 168 números.
3. Si la condición es cierta, no hace nada y el programa sigue. Si es falsa, detiene el programa con un `AssertionError` y muestra tu mensaje.
4. PyTorch se niega a crearlo, porque la lista es chueca (un renglón de 3 y otro de 2): `ValueError: expected sequence of length 3 at dim 1 (got 2)`.
5. `ndim` da 3 (tres ejes), `numel()` da 42 (el total de números, 2 × 7 × 3) y `dtype` da `torch.int64` (enteros).
6. `torch.int64` para los enteros y `torch.float32` para los decimales.
7. Da 200: Norte, viernes, mochilas. `.item()` lo convierte de tensor de un número a número normal de Python, para imprimirlo o compararlo con un `assert`.
8. `(2, 7)`: las ventas de mochilas de las dos sucursales los siete días.
9. `(7, 3)`: junta las dos sucursales; queda cada día y cada producto como si doña Lupe tuviera una sola tienda.
10. Además del valor más alto, regresa **en qué posición** está (los índices). `sum` solo regresa los totales.

---

## Lo que sigue

Con esto cierras la lección A: sabes qué es un tensor, crearlo, recorrerlo por sus ejes y resumirlo. En la sesión 5 empieza la lección B. Antes de llegar, resuelve la autoevaluación completa; si todo te sale sin ver, vas con la base firme.

---

*Escribo desde lo que vemos todos los días en Hábil, donde construimos sistemas para banca, seguros y empresas en México. Si en tu trabajo o en el negocio de tu familia se topan con un problema así, platiquemos: [habil.mx](https://habil.mx)*
