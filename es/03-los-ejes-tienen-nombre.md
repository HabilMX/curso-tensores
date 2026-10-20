# Tensores desde cero (3): los ejes tienen nombre

*Cómo sacar un pedazo de un tensor por cualquiera de sus ejes —un número, un día, un producto, el canal rojo de una foto— y qué le pasa a la forma cuando lo haces.*

*Tensores desde cero · lección A, sesión 3 de 8*

*Escenario: los personajes y la historia son inventados. Los datos técnicos, históricos y las cifras reales llevan su fuente.*

**Lunes 16 de noviembre, 7:40 p.m. Papelería Centro.**

La cortina metálica está a medio bajar. Adentro huele a papel nuevo y a café recalentado. Doña Lupe cuenta billetes en el mostrador. Mateo tiene la laptop abierta junto a la caja registradora, entre un frasco de lápices y una pila de cartulinas.

Faltan 37 días para el 23 de diciembre.

La semana pasada arreglaron el archivo de Norte, el del renglón cortado. Ahora PyTorch ya lo acepta: un tensor de forma `(2, 7, 3)`. Dos sucursales, siete días, tres productos. Cuarenta y dos números.

—¿Y ya? —pregunta doña Lupe sin levantar la vista—. ¿Ya sabes quién me está robando?

—Todavía no, tía. Tengo los números. Me falta saber mirarlos.

—Pues míralos. Para eso tienes los ojos.

Mateo se ríe, pero tiene razón. Cuarenta y dos números juntos no dicen nada. Son como una bodega con todo amontonado: ahí está la respuesta, pero no la encuentras. Lo que necesita es sacar **pedazos**. Un día. Un producto. Una sucursal. Ponerlos lado a lado y ver qué no cuadra.

Escribe una línea. Le sale el viernes de las dos sucursales.

Se queda quieto.

—Tía. ¿Cuál es su mejor día de la semana?

—El viernes, ¿cuál otro? Salen de la escuela, vienen con los papás, se llevan de todo.

—En Centro, sí. El viernes vende 2,000 de cuadernos. Ningún otro día se le acerca.

—¿Y en Norte?

Mateo gira la pantalla. `500, 300, 200`.

Doña Lupe se pone los lentes. Los vuelve a quitar.

—Eso vende Norte un martes cualquiera.

Casi. El martes de Norte son 400, 250 y 150. El viernes, el día fuerte, apenas le gana por un pelito.

Y hay algo peor en la última columna. Las mochilas. Lo más caro de la tienda. En Norte, casi todos los días: 100. Como si nadie las tocara.

Mateo siente un hueco en el estómago. Todavía no sabe qué significa. Pero sabe que acaba de encontrar la primera pista de verdad.

---

En la sesión anterior instalaste PyTorch y convertiste la papelería de doña Lupe en tu primer tensor de verdad, con forma `(2, 7, 3)`. Hoy aprendes lo que hizo Mateo esa noche: sacarle pedazos. No solo un número, sino un día completo, un producto completo o una semana recortada, por el eje que tú elijas.

## Repaso para abrir

Contesta sin ver la sesión anterior. Las respuestas están en las soluciones, al final.

1. ¿Qué te dice `t.shape` y cómo lo compruebas con un `assert`?
2. ¿Por qué `torch.tensor([[1, 2, 3], [4, 5]])` falla?
3. ¿Qué `dtype` sale de `torch.tensor([1, 2, 3])` y cuál de `torch.tensor([1.0, 2.0])`?

> **El código de esta sesión** está en el repositorio público del curso, en la carpeta `sesion-03`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-03). Escríbelo tú primero, porque así se aprende; el repositorio es para comparar cuando algo no te salga igual. Cada programa viene con la salida que debe darte.

## Objetivos de esta sesión

Al terminar, vas a poder hacer estas cuatro cosas. Al lado de cada una está el ejercicio que la demuestra.

| # | Al terminar puedes… | Lo demuestra |
|---|---|---|
| 1 | **Nombrar** cada eje de un tensor y **escribirlo** en un comentario arriba del código | Ejercicios 10 y 13 |
| 2 | **Sacar** un solo número con un índice por eje, `papeleria[1, 4, 2]`, y **convertirlo** a número de Python con `.item()` | Ejercicio 10 |
| 3 | **Sacar** un pedazo por cualquier eje con `:` y con un tramo como `1:5`, y **predecir** su forma antes de correrlo | Ejercicios 11 y 12 |
| 4 | **Decidir** el orden de los ejes de una foto y **sacar** un solo color | Ejercicio 13 |

**Qué necesitas:** el entorno de la sesión 2 con PyTorch instalado (actívalo antes de empezar) y un editor de texto. Hoy no instalas nada nuevo.

---

## Por qué importa

Mateo no encontró la pista sumando todo. La encontró cortando: un día de dos sucursales, una columna de una sucursal. Así trabaja casi cualquier sistema con datos.

Un programa que revisa la foto de un coche chocado casi nunca la usa completa: recorta la zona de la defensa, o se queda con un solo color. Un banco que guarda las compras de mil clientes en 30 días, con forma `(1000, 30, 5)`, a veces quiere a un cliente completo y a veces solo el día de ayer de todos. Es el mismo tensor. Lo que cambia es por cuál eje lo cortas.

Y aquí viene la trampa. Muchos de los errores que vas a ver con modelos no son de matemáticas: son de forma. Pediste un pedazo, creíste que tenía una forma y tenía otra. Nada truena. Solo te da un número equivocado, y tú le crees. Hoy aprendes a predecir esa forma antes de correr el código, para que eso no te pase a ti.

## Paso 1 — Ponle nombre a cada eje

Antes de cortar nada, Mateo hace algo que parece de más y no lo es: escribe arriba del código qué es cada eje. Porque PyTorch no sabe qué es «Norte». No sabe qué es «viernes».

Para PyTorch, `papeleria` es una caja de `2 × 7 × 3` números y nada más. Quien sabe que el primer eje son las sucursales eres tú. Por eso la primera regla de hoy: **dilo en voz alta y escríbelo en un comentario**, como la etiqueta de un cajón de la cocina.

Crea un archivo `ejes.py` y empieza así:

```python
import torch

# Ejes de la papelería:
#   eje 0 = sucursal  (0 Centro, 1 Norte)
#   eje 1 = día       (0 lunes ... 6 domingo)
#   eje 2 = producto  (0 cuadernos, 1 plumas, 2 mochilas)
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
print(papeleria.shape)
```

Córrelo con `python3 ejes.py`:

```
torch.Size([2, 7, 3])
```

Léela con los nombres puestos: **2 sucursales, 7 días, 3 productos.** Los ejes se cuentan desde 0, como los índices de una lista: eje 0, eje 1, eje 2. Cuando leas «por el eje 1», traduce «por los días».

—¿Y eso para qué? —dice doña Lupe, asomándose—. Si ya sabes cuál es cuál.

—Hoy sí. En tres semanas, a las dos de la mañana, no me voy a acordar.

—Como tu tío con las llaves. Por eso les puse etiqueta.

## Paso 2 — Un solo número: `papeleria[1, 4, 2]`

Lo primero que Mateo quiere ver es un solo número: lo que vendió Norte el viernes en mochilas. Si ahí hay algo raro, quiere tocarlo con el dedo.

En la sesión 1 señalaste un número así: `papeleria[1][4][2]`. Con un tensor eso sigue funcionando, pero hay una forma mejor: **todos los índices dentro del mismo corchete, separados por comas.**

Desde aquí, ve agregando cada bloque al final de `ejes.py` y vuelve a correrlo; en cada salida te muestro solo las líneas nuevas. Agrega:

```python
print(papeleria[1][4][2])
print(papeleria[1, 4, 2])
```

```
tensor(200)
tensor(200)
```

Las dos dan lo mismo: Norte, viernes, mochilas, $200. En el mejor día de la semana. Mateo anota el número en un post-it y lo pega en la orilla de la pantalla.

La diferencia está en cómo trabajan por dentro:

- `papeleria[1][4][2]` son **tres pasos**: saca la pestaña de Norte, de ahí el viernes, del viernes las mochilas. Caja dentro de caja.
- `papeleria[1, 4, 2]` es **un solo paso**: le das la dirección completa y va directo.

En el paso 7 vas a ver un caso donde la de corchetes separados te engaña. Desde hoy, usa la de comas.

### Lo que sale no es un número: es un tensor de cero dimensiones

La salida no dice `200`, dice `tensor(200)`: el número viene envuelto. Agrega esto:

```python
x = papeleria[1, 4, 2]
print(x.shape, x.ndim)
print(type(x))
print(x.item())
print(type(x.item()))
```

```
torch.Size([]) 0
<class 'torch.Tensor'>
200
<class 'int'>
```

Forma vacía, cero dimensiones: es un **escalar**, como el total de $18,500 de la sesión 1, pero en versión tensor. Un dulce que todavía trae la envoltura. `.item()` le quita la envoltura y te da un `int` de Python normal, para imprimirlo, guardarlo en una lista o mandarlo a un reporte.

Y la autocorrección de siempre: primero la forma, después el valor.

```python
assert papeleria[1, 4, 2].shape == (), "Un solo número no tiene ejes"
assert papeleria[1, 4, 2].item() == 200, "Norte, viernes, mochilas: 200"
print("Paso 2 correcto")
```

```
Paso 2 correcto
```

`torch.Size([])` se compara bien contra la tupla vacía `()`: tus formas se siguen escribiendo como tuplas.

Un número suelto no prueba nada. 200 pesos de mochilas puede ser un mal viernes y ya. Mateo necesita ver la semana entera de Norte, de un jalón.

## Paso 3 — Un índice solo: el eje desaparece

¿Qué pasa si das menos índices que ejes? Prueba con uno solo:

```python
print(papeleria[1])
print(papeleria[1].shape)
```

```
tensor([[300, 200, 100],
        [400, 250, 150],
        [350, 200, 100],
        [300, 150, 100],
        [500, 300, 200],
        [300, 200, 100],
        [200, 100, 100]])
torch.Size([7, 3])
```

La pestaña completa de Norte. Mateo baja la vista por la última columna: 100, 150, 100, 100, 200, 100, 100. Plana como mesa.

Pero fíjate en la forma, que hoy es la protagonista. Pasó de `(2, 7, 3)` a `(7, 3)`: **el eje de las sucursales desapareció.** Ya elegiste una; frente a la pestaña de Norte ya no hay «sucursales», solo días y productos.

Con dos índices, lo mismo:

```python
print(papeleria[1, 4])
print(papeleria[1, 4].shape)
```

```
tensor([500, 300, 200])
torch.Size([3])
```

Norte, viernes, los tres productos. Se fueron dos ejes; queda el de productos. Primera mitad de la regla de hoy:

> **Cada índice que das se come su eje.** Tres ejes menos un índice: quedan dos. Tres ejes menos tres índices: no queda ninguno, y eso es un escalar.

## Paso 4 — Los dos puntos: «todos, por favor»

Ahora viene la línea que le dio el hueco en el estómago. Mateo no quiere a Norte sola: quiere a Norte **junto a** Centro, el mismo día, para que la diferencia no tenga dónde esconderse.

¿Y si quieres **el viernes de las dos sucursales**? Tienes que saltarte el primer eje sin elegir nada. Para eso están los dos puntos, `:`, que en un índice quieren decir «todos».

```python
viernes = papeleria[:, 4]
print(viernes)
print(viernes.shape)
```

```
tensor([[2000, 1200,  700],
        [ 500,  300,  200]])
torch.Size([2, 3])
```

Léelo con los nombres de los ejes: **todas las sucursales, el día 4 (viernes), y como no dijiste nada del producto, todos los productos.** Renglón de arriba, Centro; renglón de abajo, Norte.

Ahí está. Arriba, 2000. Abajo, 500. Mismo día, misma salida de la escuela. Una línea de código y la diferencia queda a la vista, sin pestañas ni copiar y pegar.

Forma `(2, 3)`: el eje de los días se fue porque elegiste uno; el de sucursales se quedó porque pediste todas; el de productos, porque no lo tocaste. Como pedir tacos «con todo»: no eliges cebolla o cilantro, te llevas lo que haya.

### Compruébalo contra la versión con `for`

Mateo no se fía de una sola línea. Si va a enseñarle esto al banco, quiere sacarlo por dos caminos. Con lo que ya sabías, sería número por número:

```python
a_mano = []
for s in range(2):
    a_mano.append([papeleria[s][4][p].item() for p in range(3)])
print(a_mano)
print(viernes.tolist())
```

```
[[2000, 1200, 700], [500, 300, 200]]
[[2000, 1200, 700], [500, 300, 200]]
```

`.tolist()` es primo de `.item()`: convierte un tensor completo de regreso a listas de Python, que se comparan con `==` como siempre. Las dos dan lo mismo; déjalo por escrito:

```python
assert viernes.shape == (2, 3), "Viernes: 2 sucursales × 3 productos"
assert viernes.tolist() == [[2000, 1200, 700], [500, 300, 200]]
print("Paso 4 correcto")
```

```
Paso 4 correcto
```

## Paso 5 — Un tramo: `papeleria[0, 1:5]`

—A ver —dice doña Lupe—. Enséñame cómo se ve una semana normal. La de Centro. Pero sin el lunes, que siempre está muerto, ni el fin de semana.

Mateo necesita un tramo: del martes al viernes. Los dos puntos también piden un tramo, como en las listas de Python: `lista[1:5]` da los elementos 1, 2, 3 y 4. **Incluye el primero, deja fuera el último.** Con tensores, en cualquier eje.

Doña Lupe quiere el Centro de martes a viernes:

```python
tramo = papeleria[0, 1:5]
print(tramo)
print(tramo.shape)
```

```
tensor([[1500,  900,  700],
        [1000,  700,  500],
        [1300,  900,  600],
        [2000, 1200,  700]])
torch.Size([4, 3])
```

Centro, días del 1 al 4 (el 5, sábado, queda fuera), todos los productos. Forma `(4, 3)`: el eje de sucursales se fue por el índice; el de días **se quedó**, pero mide 4 en vez de 7. Un tramo recorta el eje, no lo quita. Compáralo contra la lista `centro` de Python:

```python
print(centro[1:5])
assert tramo.shape == (4, 3), "De martes a viernes: 4 días × 3 productos"
assert tramo.tolist() == centro[1:5]
print("Paso 5 correcto")
```

```
[[1500, 900, 700], [1000, 700, 500], [1300, 900, 600], [2000, 1200, 700]]
Paso 5 correcto
```

A este pedazo, el que se saca con dos puntos, se le llama **rebanada**. Es la palabra que vas a ver en la documentación en español; en inglés es *slice*.

## Paso 6 — El último eje: `papeleria[:, :, 2]`

Doña Lupe deja los billetes. Ya no cuenta. Ahora pregunta ella: **¿cuánto vendió de mochilas cada sucursal cada día?** Las mochilas son lo que más deja. Si algo está mal, quiere verlo ahí.

El producto es el eje 2, el último; para llegar a él pasas por los otros dos con «todos»:

```python
mochilas = papeleria[:, :, 2]
print(mochilas)
print(mochilas.shape)
```

```
tensor([[500, 700, 500, 600, 700, 400, 300],
        [100, 150, 100, 100, 200, 100, 100]])
torch.Size([2, 7])
```

Forma `(2, 7)`: se fue el eje de productos; cada renglón es una sucursal y cada columna un día.

Doña Lupe se acerca tanto a la pantalla que Mateo tiene que hacerse para atrás. Arriba, Centro: 500, 700, 500, 600… Abajo, Norte: 100, 150, 100, 100…

—Mijo, en Norte pasa más gente que aquí. Está junto a la parada del camión.

Mateo no contesta. No tiene qué contestar todavía.

En la hoja de cálculo tendrías que copiar la última columna de cada pestaña y pegarlas lado a lado. Aquí fue una línea. Con `for`, son dos ciclos:

```python
m = []
for s in range(2):
    fila = []
    for d in range(7):
        fila.append(papeleria[s, d, 2].item())
    m.append(fila)
print(m)

assert mochilas.shape == (2, 7), "Mochilas: 2 sucursales × 7 días"
assert mochilas.tolist() == m
print("Paso 6 correcto")
```

```
[[500, 700, 500, 600, 700, 400, 300], [100, 150, 100, 100, 200, 100, 100]]
Paso 6 correcto
```

Siete líneas contra una. Ese es el chiste de los tensores: cuando tienes prisa —y a Mateo le quedan 37 días—, la pregunta y la respuesta caben en el mismo renglón.

## Paso 7 — Por qué la de comas: la trampa de `papeleria[:][4]`

Esa misma noche, Mateo casi se equivoca. Antes de dar con la línea buena del paso 4, probó la que le salía natural. Si pidieras el viernes de las dos sucursales con corchetes separados, lo natural sería esto:

```python
print(papeleria[:][4])
```

```
Traceback (most recent call last):
  ...
IndexError: index 4 is out of bounds for dimension 0 with size 2
```

Lee el error palabra por palabra: «el índice 4 está fuera de los límites de la dimensión 0, que mide 2». ¿Dimensión 0? ¡Si tú querías el día!

Los corchetes separados trabajan en pasos, y cada paso empieza otra vez desde el primer eje **de lo que quedó** del paso anterior, no desde el eje que tú tenías en mente:

1. `papeleria[:]` es «todas las sucursales»: te devuelve el tensor completo, igualito.
2. `[4]` cae otra vez en el primer eje: le pides la sucursal 4, y solo hay dos.

Con comas, cada posición es un eje fijo. Y esta vez tuviste suerte: tronó. Si doña Lupe tuviera 7 sucursales, `papeleria[:][4]` te daría la sucursal 4 sin quejarse, y creerías que es el viernes. **El error que no truena es el peligroso.**

Mateo se imagina la escena: él frente al licenciado Ortega, enseñando «los viernes» que en realidad son otra sucursal. El banco no le daría una segunda oportunidad. Desde esa noche, todo con comas.

## Paso 8 — `papeleria[0]` contra `papeleria[0:1]`

Una última cosa le hace ruido a Mateo. Hay dos maneras de pedir una sola sucursal, y parece que dan lo mismo. Parece.

Usa el índice y la rebanada para pedir **lo mismo**, la sucursal Centro:

```python
print(papeleria[0].shape)
print(papeleria[0:1].shape)
```

```
torch.Size([7, 3])
torch.Size([1, 7, 3])
```

Mismos 21 números, distinta forma. `papeleria[0]` es un índice: **el eje desaparece** y te quedas con una tabla. `papeleria[0:1]` es una rebanada de una sola sucursal: **el eje se queda**, aunque mida 1, y te quedas con una hoja de una sola pestaña:

```python
print(papeleria[0:1])
```

```
tensor([[[1200,  800,  500],
         [1500,  900,  700],
         [1000,  700,  500],
         [1300,  900,  600],
         [2000, 1200,  700],
         [1200,  800,  400],
         [ 800,  500,  300]]])
```

Tres corchetes al principio: sigue siendo de tres dimensiones. Es sacar una hoja de la carpeta contra quedarte con una carpeta de una sola hoja. Y le importa a PyTorch: un modelo que espera «una carpeta de fotos», con forma `(N, C, H, W)`, puede rechazar una foto suelta de forma `(C, H, W)`. La regla completa de hoy:

> **Con un índice, el eje desaparece. Con una rebanada, el eje se queda** (aunque mida 1).

| Lo que escribes | Qué eje se va | Forma | Qué es |
|---|---|---|---|
| `papeleria[1, 4, 2]` | los tres | `()` | Norte, viernes, mochilas |
| `papeleria[1]` | sucursal | `(7, 3)` | la pestaña de Norte |
| `papeleria[1, 4]` | sucursal y día | `(3,)` | Norte, viernes, los 3 productos |
| `papeleria[:, 4]` | día | `(2, 3)` | el viernes de las dos sucursales |
| `papeleria[0, 1:5]` | sucursal | `(4, 3)` | Centro de martes a viernes |
| `papeleria[:, :, 2]` | producto | `(2, 7)` | las mochilas, por sucursal y día |
| `papeleria[0:1]` | ninguno | `(1, 7, 3)` | una hoja con solo la pestaña Centro |

El truco para predecir la forma: **escribe la forma original, tacha cada eje donde pusiste un índice y cambia por el largo del tramo cada eje donde pusiste una rebanada.** Para `papeleria[0, 1:5]`: `(2, 7, 3)` → tacha el 2 → cambia el 7 por 4 (del 1 al 4) → `(4, 3)`.

## Paso 9 — Una foto de 2 × 3 puntos

**Martes 17 de noviembre, 1:15 a.m. Cuarto de Mateo.**

No puede dormir. En el teléfono tiene una foto que le mandó doña Lupe en la tarde: la terminal de Norte, con su cable enredado. «Para que veas que sí está conectada», escribió.

Mateo piensa que una foto también es un tensor. Y que si sabe cortar la papelería, sabe cortar una foto.

Para comprobarlo, empieza con una foto chiquita hecha a mano: 2 puntos de alto por 3 de ancho. Cada punto trae tres números, **rojo, verde y azul**, de 0 (nada) a 255 (todo). El amarillo, por ejemplo, es rojo y verde al máximo, sin azul.

Crea un archivo `foto.py`:

```python
import torch

# Foto de 2 × 3 puntos, en orden alto × ancho × color
#   eje 0 = renglón (alto), eje 1 = punto en el renglón (ancho), eje 2 = color (0 rojo, 1 verde, 2 azul)
foto = torch.tensor([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],          # arriba: rojo, verde, azul
    [[255, 255, 255], [0, 0, 0], [255, 255, 0]],      # abajo: blanco, negro, amarillo
])
print(foto.shape)
```

```
torch.Size([2, 3, 3])
```

La forma tiene dos 3: el primero son los puntos de ancho; el segundo, los colores. Sin el comentario, nadie sabe cuál es cuál, ni tú en dos semanas. Saca **solo el canal rojo**, igual que las mochilas:

```python
rojo = foto[:, :, 0]
print(rojo)
print(rojo.shape)
assert rojo.shape == (2, 3), "El canal rojo es una tabla de 2 × 3"
assert rojo.tolist() == [[255, 0, 0], [255, 0, 255]]
print("Canal rojo correcto")
```

```
tensor([[255,   0,   0],
        [255,   0, 255]])
torch.Size([2, 3])
Canal rojo correcto
```

Arriba, solo el punto rojo tiene rojo; abajo, el blanco y el amarillo sí, el negro no. Esa tabla es una foto en blanco y negro de cuánto rojo hay en cada punto: eso es «separar canales». Un punto suelto, con sus tres colores:

```python
print(foto[1, 2])
```

```
tensor([255, 255,   0])
```

Abajo, a la derecha: el amarillo.

Funciona igual que las mochilas. Mismos dos puntos, mismas reglas. Mateo sonríe a oscuras. Lo que aprendió con la papelería le sirve para cualquier caja de números.

## Paso 10 — El orden de los ejes es una decisión de diseño

Pero ahí, a la una de la mañana, se le ocurre una pregunta incómoda: ¿quién decidió que el color fuera el último eje? Él. Guardaste la foto como **alto × ancho × color**, pero nadie te obligó. Se puede guardar como **color × alto × ancho**: la tabla del rojo, luego la del verde, luego la del azul.

```python
# La misma foto en orden color × alto × ancho
#   eje 0 = color, eje 1 = renglón, eje 2 = punto en el renglón
foto_cha = torch.tensor([
    [[255, 0, 0], [255, 0, 255]],      # canal rojo
    [[0, 255, 0], [255, 0, 255]],      # canal verde
    [[0, 0, 255], [255, 0, 0]],        # canal azul
])
print(foto_cha.shape)
print(foto_cha[0])
assert foto_cha[0].tolist() == foto[:, :, 0].tolist()
print("Mismo rojo, otro orden")
```

```
torch.Size([3, 2, 3])
tensor([[255,   0,   0],
        [255,   0, 255]])
Mismo rojo, otro orden
```

Mismos 18 números, otra forma: `(3, 2, 3)`. Y el rojo ya no es `foto[:, :, 0]`, es solo `foto_cha[0]`. **Lo que pones primero es lo que se saca más fácil.**

¿Cuál es el correcto? Los dos. Es como acomodar el clóset por tipo de ropa o por conjunto, o como doña Lupe acomoda la bodega: por sucursal o por producto. La mercancía es la misma; cambia qué encuentras rápido. Por eso el orden de los ejes es una **decisión de diseño**: la tomas pensando en qué vas a pedir más seguido.

Ya hay costumbres. En PyTorch, la convención común para imágenes es **canales primero**, que en la documentación verás como **C × H × W** (*channels*, *height*, *width*: canales, alto y ancho; [documentación de TorchVision](https://docs.pytorch.org/vision/stable/transforms.html)); un lote de fotos agrega al principio el eje de cuál foto. Otras herramientas y muchos programas de imágenes acostumbran alto × ancho × color. Ninguno está mal; lo malo es no saber cuál tienes en la mano. Por eso, con fotos, **el comentario con los ejes va siempre arriba del tensor.**

| Orden | Forma de tu foto | El canal rojo | Un punto con sus 3 colores |
|---|---|---|---|
| alto × ancho × color | `(2, 3, 3)` | `foto[:, :, 0]` | `foto[1, 2]` |
| color × alto × ancho | `(3, 2, 3)` | `foto_cha[0]` | `foto_cha[:, 1, 2]` |

Si corres `foto_cha[:, 1, 2]`, te da `tensor([255, 255,   0])`: el mismo amarillo.

Mateo cierra la laptop a las 2:03. En el post-it de la pantalla sigue el 200. Ya no le parece un número. Le parece una pregunta.

---

👉 **Si eres de Administración:** lo que hizo Mateo esa noche lo haces tú con una hoja de cálculo cuando filtras o escondes columnas: «solo el viernes de todas las sucursales» o «solo las mochilas de todas las pestañas». Aquí cada una es una línea. Y lo del paso 10 te sirve en cualquier reporte: cuando te pidan «ventas por sucursal y por mes», pregunta qué va en los renglones y qué en las columnas. Es una decisión de diseño, y se toma pensando en quien lo va a leer.

---

## Ejercicios de la sesión 3

Las soluciones están al final. Trabaja en tu `ejes.py`, que ya tiene la `papeleria` y el comentario con los ejes. En el ejercicio 12 los `assert` ya escritos te dicen si acertaste; en el 10, el 11 y el 13, compara tu trabajo con las soluciones.

**Ejercicio 10 — Señala el número, ahora con comas.** Con un índice por eje y `.item()`, escribe el `assert` de cada dato:

a) Lo que vendió el Centro el miércoles en plumas.
b) Lo que vendió Norte el domingo en cuadernos.
c) Lo que vendió el Centro el sábado en mochilas.
d) Comprueba con un `assert` que tu respuesta de a) da lo mismo con los corchetes separados de la sesión 1.

Ejemplo de uno resuelto: `assert papeleria[0, 0, 0].item() == 1200` (Centro, lunes, cuadernos).

**Ejercicio 11 — Predice la forma antes de correr.** En papel, escribe la forma de cada expresión y qué representa, usando el truco del paso 8. Después córrelo y compara.

```python
for nombre, x in [
    ("a) papeleria[0]", papeleria[0]),
    ("b) papeleria[0, 6]", papeleria[0, 6]),
    ("c) papeleria[:, 6]", papeleria[:, 6]),
    ("d) papeleria[:, :, 0]", papeleria[:, :, 0]),
    ("e) papeleria[1, 0:3]", papeleria[1, 0:3]),
    ("f) papeleria[1, 0:3, 1]", papeleria[1, 0:3, 1]),
    ("g) papeleria[0:1, 6]", papeleria[0:1, 6]),
    ("h) papeleria[1, 4, 2]", papeleria[1, 4, 2]),
]:
    print(nombre, x.shape)
```

Si fallaste alguna, busca cuál eje creíste que se quedaba y se fue, o al revés.

**Ejercicio 12 — Completa los huecos.** Llena cada `____` y comprueba con los `assert`:

```python
# a) El fin de semana (sábado y domingo) de Norte, los 3 productos
fin_norte = papeleria[____, ____]
assert fin_norte.shape == (2, 3), "Fin de semana de Norte: 2 días × 3 productos"
assert fin_norte.tolist() == [[300, 200, 100], [200, 100, 100]]

# b) Las plumas del Centro de lunes a miércoles
plumas = papeleria[0, ____, ____]
assert plumas.shape == (3,), "Plumas del Centro: 3 días"
assert plumas.tolist() == [800, 900, 700]

# c) Lo mismo que b), con un for
a_mano = []
for d in range(____):
    a_mano.append(papeleria[0, d, 1].____)
assert plumas.tolist() == a_mano
print("Ejercicio 12 superado")
```

Pista para a): el domingo es el día 6, y la rebanada deja fuera su último número.

**Ejercicio 13 — Tu propia foto.** Crea a mano una foto de **3 puntos de alto por 2 de ancho**, guardada en el orden que acostumbra PyTorch: **color × alto × ancho** (`C × H × W`). Elige tú los colores de los 6 puntos (rojo, verde, azul, blanco, negro, amarillo o los que quieras) y escríbelos en un comentario. Después:

a) Escribe el `assert` de su forma.
b) Saca el canal verde. Escribe el `assert` de su forma y el de su valor contra una lista escrita a mano.
c) Saca el renglón de arriba con sus tres colores. ¿Qué forma tiene? Escribe sus dos `assert`.
d) Saca el punto de abajo a la derecha con sus tres colores. Escribe sus dos `assert`.

No tiene una sola respuesta; lo que sí tiene que cumplirse es que los `assert` pasen y el comentario diga qué es cada eje.

---

## Errores comunes (y cómo darte cuenta)

| Error | Cómo se ve | Cómo darte cuenta | Qué hacer |
|---|---|---|---|
| Pedir un índice que no existe | `papeleria[1, 7, 2]` da `IndexError: index 7 is out of bounds for dimension 1 with size 7` | El mensaje te dice qué eje (*dimension 1*, los días) y cuánto mide (*size 7*) | Los índices van de 0 al tamaño menos 1: los días, de 0 a 6 |
| Invertir el orden de los ejes | `papeleria[4, 1, 2]` en vez de `papeleria[1, 4, 2]` da `IndexError: index 4 is out of bounds for dimension 0 with size 2` | Truena en *dimension 0*, que tiene solo 2 sucursales | Lee tu comentario de ejes: el primer índice siempre es la sucursal |
| Encadenar corchetes con `:` | `papeleria[:][4]` da el mismo error de *dimension 0*, o peor, no truena y te da otra cosa | El resultado no es lo que pediste, o truena en un eje que no tocaste | Todos los índices en un solo corchete: `papeleria[:, 4]` |
| Creer que el tramo incluye el último | Pides de martes a viernes con `1:4` y te falta el viernes | El `assert` de la forma dice 3 días donde esperabas 4 | El último número queda fuera: martes a viernes es `1:5` |
| Confundir índice con rebanada | `papeleria[0:1]` tiene forma `(1, 7, 3)` y esperabas `(7, 3)` | El `assert` de la forma falla con un 1 de más | Índice: el eje se va. Rebanada: el eje se queda, aunque mida 1 |
| Usar `.item()` con varios números | `papeleria[0, 0].item()` da `RuntimeError: a Tensor with 3 elements cannot be converted to Scalar` | El mensaje dice cuántos elementos tenía (3) | `.item()` es solo para un número; para varios, `.tolist()` |
| Comparar un pedazo contra un número en un `assert` | `assert papeleria[0, 0] == 1200` da `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` | Truena el `assert`, no el índice | Te faltó un índice (querías `papeleria[0, 0, 0]`), o compara la forma y luego `.tolist()` contra una lista |

Una trampa que no truena: `papeleria[0, 5:20]` no marca error aunque solo hay 7 días; te da sábado y domingo, forma `(2, 3)`, igual que harían las listas de Python. Si un tramo te da menos de lo esperado, revisa sus números.

---

## Repaso para la próxima sesión

La sesión 4 empieza con estas tres preguntas. Contéstalas sin ver la lección:

1. ¿Qué forma tiene `papeleria[:, 4]` y qué representa?
2. ¿Qué diferencia hay entre `papeleria[0]` y `papeleria[0:1]` en la forma?
3. ¿Cómo sacas las ventas de mochilas de las dos sucursales todos los días?

🔁 **La próxima semana:**

Martes, 8:10 de la mañana. Mateo le marca a doña Lupe antes de entrar a clase. Le cuenta lo del viernes. Lo de las mochilas que no se mueven.

Del otro lado de la línea, un silencio largo. Se oye la cortina de Centro subiendo.

—Pues si no se venden las mochilas —dice al fin doña Lupe—, ¿qué le están cobrando a la gente?

Mateo no lo sabe. Pero sabe cómo seguir: dejar de sacar pedazos y empezar a **resumirlos**. Totales y promedios por eje: cuánto vendió cada sucursal en la semana, cuál fue el promedio de cada día, todo en una línea y comprobado contra un `for`. Cuando sume la semana de Norte, el número va a señalar a alguien.

---

## Soluciones (no las veas antes de intentarlo)

**Repaso para abrir.**

1. `t.shape` te dice cuántos elementos hay en cada eje: para la papelería, 2 sucursales, 7 días y 3 productos. Se comprueba con `assert papeleria.shape == (2, 7, 3)`.
2. Porque no es rectangular: el primer renglón tiene 3 números y el segundo, 2. PyTorch se niega y lo dice: `ValueError: expected sequence of length 3 at dim 1 (got 2)`.
3. `torch.tensor([1, 2, 3])` sale `torch.int64` (enteros) y `torch.tensor([1.0, 2.0])` sale `torch.float32` (decimales).

**Ejercicio 10.**

```python
assert papeleria[0, 2, 1].item() == 700    # a) Centro, miércoles, plumas
assert papeleria[1, 6, 0].item() == 200    # b) Norte, domingo, cuadernos
assert papeleria[0, 5, 2].item() == 400    # c) Centro, sábado, mochilas
assert papeleria[0][2][1].item() == papeleria[0, 2, 1].item()   # d)
print("Ejercicio 10 superado")
```

```
Ejercicio 10 superado
```

Los incisos a) y b) son los mismos datos que buscaste en los incisos a) y c) del ejercicio 2 de la sesión 1, ahora con comas.

**Ejercicio 11.**

```
a) papeleria[0] torch.Size([7, 3])
b) papeleria[0, 6] torch.Size([3])
c) papeleria[:, 6] torch.Size([2, 3])
d) papeleria[:, :, 0] torch.Size([2, 7])
e) papeleria[1, 0:3] torch.Size([3, 3])
f) papeleria[1, 0:3, 1] torch.Size([3])
g) papeleria[0:1, 6] torch.Size([1, 3])
h) papeleria[1, 4, 2] torch.Size([])
```

a) la pestaña Centro; b) el domingo del Centro; c) el domingo de las dos sucursales; d) los cuadernos de las dos, todos los días; e) Norte de lunes a miércoles; f) las plumas de Norte de lunes a miércoles; g) el domingo del Centro conservando el eje de sucursal, que mide 1; h) Norte, viernes, mochilas. La tramposa es g): se parece a b), pero la sucursal se pidió con rebanada.

**Ejercicio 12.**

```python
fin_norte = papeleria[1, 5:7]       # a)
plumas = papeleria[0, 0:3, 1]       # b)
for d in range(3):                  # c)
    a_mano.append(papeleria[0, d, 1].item())
```

```
Ejercicio 12 superado
```

En a), `5:7` toma el sábado (5) y el domingo (6) y deja fuera el 7, que de todos modos no existe.

**Ejercicio 13.** Una respuesta posible:

```python
import torch

# Foto de 3 × 2 puntos, en orden color × alto × ancho
#   eje 0 = color (0 rojo, 1 verde, 2 azul)
#   eje 1 = renglón (0 arriba ... 2 abajo)
#   eje 2 = punto en el renglón (0 izquierda, 1 derecha)
# Los puntos: arriba rojo y verde; en medio azul y blanco; abajo negro y morado
foto = torch.tensor([
    [[255, 0], [0, 255], [0, 128]],      # canal rojo
    [[0, 255], [0, 255], [0, 0]],        # canal verde
    [[0, 0], [255, 255], [0, 128]],      # canal azul
])
assert foto.shape == (3, 3, 2), "3 colores × 3 renglones × 2 puntos"

verde = foto[1]
assert verde.shape == (3, 2)
assert verde.tolist() == [[0, 255], [0, 255], [0, 0]]

arriba = foto[:, 0]
assert arriba.shape == (3, 2)
assert arriba.tolist() == [[255, 0], [0, 255], [0, 0]]

esquina = foto[:, 2, 1]
assert esquina.shape == (3,)
assert esquina.tolist() == [128, 0, 128]
print("Ejercicio 13 superado")
```

```
Ejercicio 13 superado
```

En c), la forma es `(3, 2)`, no `(2, 3)`: sacar un pedazo no reacomoda los ejes que quedan, los deja en el orden en que estaban, y el color va primero.

---

## Lo que sigue

En la sesión 4, el total del Centro será la suma de `papeleria[0]`. Si quieres adelantar, calcula con un `for` cuánto vendió cada sucursal en la semana. Centro ya lo sabes desde la sesión 1: $18,500. Te falta Norte.

---

*Escribo desde lo que vemos todos los días en Hábil, donde construimos sistemas para banca, seguros y empresas en México. Si en tu trabajo o en el negocio de tu familia se topan con un problema así, platiquemos: [habil.mx](https://habil.mx)*
