# Tensores desde cero (1): un número, una lista, una tabla, un cubo

*Qué es un tensor, contado con las listas de Python que ya sabes escribir, y por qué Einstein tuvo que pedirle ayuda a un amigo para entenderlo.*

*Tensores desde cero · lección A, sesión 1 de 8*

*Escenario: los personajes y la historia son inventados. Los datos técnicos, históricos y las cifras reales llevan su fuente.*

**Domingo 1 de noviembre, 11:52 p.m.**

El celular de doña Lupe vibró sobre el buró. Una vez. Ella, que nunca contesta nada después de las diez, lo volteó por pura costumbre.

> *Detectamos cargos inusuales en la terminal de su sucursal Norte. La terminal queda suspendida mientras se revisa.*

Lo leyó tres veces. Luego se sentó en la orilla de la cama, con los lentes en la frente, y ya no se volvió a dormir.

**Lunes 2 de noviembre, 9:10 a.m.**

—Señora Herrera, buenos días. Le habla el licenciado Ortega, del banco.

Voz formal. Prisa de alguien con otras cuarenta llamadas pendientes.

—Tiene hasta el 23 de diciembre para demostrarnos cuáles cobros de esa terminal son ventas reales. Si no, la cancelamos.

—¿El 23? Licenciado, ¡es temporada! Sin terminal en Norte, en diciembre…

—Por eso le doy el plazo. Que tenga buen día.

Clic.

**Lunes 2 de noviembre, 7:40 p.m. Papelería, sucursal Centro.**

Olor a papel nuevo y a goma de borrar. La cortina metálica a medio bajar, raspando cada vez que pasa un camión. Mateo Ríos, 20 años, segundo semestre de Sistemas, entra agachándose, con la mochila al hombro y la laptop bajo el brazo.

—Tía, me dijo mi mamá que era urgente.

Doña Lupe no lo saluda. Le pone el celular enfrente, con el mensaje del banco.

—Tú que sabes de computadoras.

Mateo lo lee. Siente el estómago frío.

—Tía, yo sé Python básico. Esto es… otra cosa.

—Pues aprende la otra cosa. Tienes hasta el 23 de diciembre. —Se cruza de brazos—. Y no me digas que no, que te pagué la inscripción.

Mateo abre la laptop sobre el mostrador, entre un frasco de lápices y una pila de cartulinas. Sabe una sola cosa con certeza: el banco no vio una papelería. Vio números. Si quiere defender a su tía, tiene que aprender a ver lo mismo que el banco.

Y tú vas a aprender con él.

---

Esta serie tiene dos lecciones de cuatro sesiones cada una, una sesión por semana. En la primera lección, **De la lista al cubo**, aprendes qué es un tensor, a crearlo, a recorrerlo y a resumirlo. En la segunda, **Los tensores trabajando**, lo cambias de forma, lo operas completo, entrenas con él un modelo que detecta fraudes y ves dónde aparece en un banco, una aseguradora y una tienda.

| Sesión | Qué ves | Práctica |
|---|---|---|
| **1 — Un número, una lista, una tabla, un cubo** | Qué es una dimensión, cómo contarlas y qué es la forma de un tensor, todo con listas de Python | Ejercicios 1 a 5 |
| **2 — Tu primer tensor** | Instalar PyTorch, convertir tus listas en tensores, `shape` y `dtype` | Ejercicios 6 a 9 |
| **3 — Los ejes tienen nombre** | Sacar un pedazo del tensor por cualquiera de sus ejes | Ejercicios 10 a 13 |
| **4 — Sumar por pestaña** | Totales y promedios por eje, comparados contra un `for` | Ejercicios 14 a 17, autoevaluación y repaso de la lección |

Cada sesión es de unas 2 horas: más o menos una para leer y seguir el ejemplo, y otra para practicar. Desde la sesión 2, cada una empieza con **3 preguntas de repaso** de la anterior, para contestar sin ver.

**Qué necesitas saber antes:** Python básico: variables, `if`, `for` y `while`, funciones, listas y tuplas. Si vienes del curso de Python de Hábil, son sus lecciones 0 a 4. **Qué necesitas tener:** Python 3 y un editor de texto. Hoy no instalas nada más.

> **El código de esta sesión** está en el repositorio público del curso, en la carpeta `sesion-01`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-01). Escríbelo tú primero, porque así se aprende; el repositorio es para comparar cuando algo no te salga igual. Cada programa viene con la salida que debe darte.

## Objetivos de esta sesión

Al terminar, vas a poder hacer estas cuatro cosas. Al lado de cada una está el ejercicio que la demuestra.

| # | Al terminar puedes… | Lo demuestra |
|---|---|---|
| 1 | Dado un conjunto de datos (ventas, calificaciones, una foto), **decir** cuántas dimensiones tiene y qué representa cada una | Ejercicio 1 |
| 2 | **Escribir** esos datos como listas anidadas de Python y **señalar** un número con sus índices | Ejercicios 2 y 4 |
| 3 | **Calcular** la forma de un tensor (cuántos elementos hay en cada dimensión), a mano y con una función | Ejercicios 1 y 3 |
| 4 | **Reconocer** cuándo una tabla no puede ser un tensor porque está chueca, y comprobarlo con código | Ejercicio 5 |

---

## Por qué importa

Esa noche, mientras doña Lupe cuenta el cambio de la caja, Mateo busca en su celular cómo hace un banco para decir «inusual». La respuesta le da un poco de miedo.

El banco no lee una compra como tú la lees. La convierte en números: el monto, la hora, la distancia a la última compra, cuántas compras hubo en la última hora. Una compra se vuelve una lista de números. Mil compras, una tabla. Las compras de cada cliente en los últimos 30 días, un cubo.

Nadie en el banco vio la cara de los clientes de Norte. Una máquina vio cubos de números y algo no le cuadró.

Y no es solo el banco. Cuando una aseguradora recibe la foto de un coche chocado, tampoco ve un coche. Ve una cuadrícula de puntos, y cada punto trae tres números: cuánto rojo, cuánto verde y cuánto azul tiene. Otra vez un cubo de números.

Casi todos los sistemas de inteligencia artificial que vas a oír nombrar en tu carrera trabajan así: primero convierten el mundo en cajas de números ordenadas, y después hacen cuentas con esas cajas. Esas cajas se llaman **tensores**. Si entiendes cómo están ordenadas, entiendes la mitad de lo que hace un modelo. La otra mitad la vemos en la segunda lección.

—¿Y eso con qué se come? —pregunta doña Lupe, sin levantar la vista de las monedas.

—Con listas, tía. Listas dentro de listas.

Es verdad, y es la buena noticia: ya sabes escribir los datos de un tensor. Son listas dentro de listas, de puros números y bien parejas. En la sesión 2 las vas a convertir en tensores de verdad.

## El caso: la papelería de doña Lupe

Doña Lupe tiene una papelería con dos sucursales, Centro y Norte. Vende tres cosas que le importan: cuadernos, plumas y mochilas. Los números de esta serie son de ejemplo, pero la forma de los datos es la de cualquier negocio.

Mateo no sabe todavía qué busca. Así que hace lo único sensato: empieza por lo fácil y va pidiendo más detalle. Le pregunta a su tía por las ventas de la semana pasada cuatro veces, cada vez con más lupa.

### Paso 1 — Un número

Doña Lupe baja la cortina hasta el fondo. El ruido metálico retumba en todo el local.

—Tía, ¿cuánto vendió la semana pasada?
—$18,500, en el Centro. Y no me preguntes de Norte todavía, que me sube la presión.

```python
total = 18500
```

Un solo número. Para señalarlo no necesitas ningún dato: solo hay uno. Decimos que tiene **cero dimensiones**. En el lenguaje de los tensores se llama **escalar**.

Un escalar dice mucho y explica poco. Con $18,500 nadie puede defender nada ante un banco.

### Paso 2 — Una lista

—¿Y día por día?
—Lunes 2,500; martes 3,100; miércoles 2,200; jueves 2,800; viernes 3,900; sábado 2,400; domingo 1,600. —Lo dice de memoria, como quien reza.

```python
por_dia = [2500, 3100, 2200, 2800, 3900, 2400, 1600]
```

Ahora sí necesitas un dato para señalar un número: **el día**. `por_dia[4]` es el viernes, 3,900 (recuerda que los índices empiezan en 0). Una dirección, **una dimensión**. Se llama **vector**.

Comprueba que no se perdió nada: `sum(por_dia)` da 18500, el mismo total del paso 1. El detalle no cambia el total; solo lo reparte.

Mateo anota algo en un post-it: *el detalle no miente; el total esconde.* Todavía no sabe cuánto le va a servir esa frase.

### Paso 3 — Una tabla

—¿Y por día y por producto?

Doña Lupe resopla, saca su laptop vieja, la de la calcomanía de la Virgen, y abre su hoja de cálculo:

| Día | Cuadernos | Plumas | Mochilas |
|---|---|---|---|
| Lunes | 1,200 | 800 | 500 |
| Martes | 1,500 | 900 | 700 |
| Miércoles | 1,000 | 700 | 500 |
| Jueves | 1,300 | 900 | 600 |
| Viernes | 2,000 | 1,200 | 700 |
| Sábado | 1,200 | 800 | 400 |
| Domingo | 800 | 500 | 300 |

En Python, cada renglón es una lista, y la tabla es una lista de renglones:

```python
por_dia_y_producto = [
    [1200, 800, 500],    # lunes
    [1500, 900, 700],    # martes
    [1000, 700, 500],    # miércoles
    [1300, 900, 600],    # jueves
    [2000, 1200, 700],   # viernes
    [1200, 800, 400],    # sábado
    [800, 500, 300],     # domingo
]
```

Para señalar un número necesitas **dos datos: el día y el producto**. `por_dia_y_producto[4][2]` es el viernes (renglón 4), mochilas (columna 2): 700. Dos direcciones, **dos dimensiones**. Se llama **matriz**.

Fíjate en el orden: primero el renglón, después la columna. Ese orden no es un capricho; es el orden en que escribiste las listas. La de afuera son los días; la de adentro, los productos.

—O sea que cada número tiene su domicilio —dice doña Lupe.

—Exacto, tía. Día y producto. Calle y número.

### Paso 4 — Un cubo: la hoja con pestañas

Entonces Mateo hace la pregunta que su tía estaba esquivando.

—¿Y la sucursal Norte?

Silencio. Doña Lupe da clic en la otra pestaña de la misma hoja, despacio, como quien abre un sobre con malas noticias. Misma forma —7 días, 3 productos—, otros números. Su hoja ahora tiene **dos pestañas, una por sucursal**, y cada pestaña es una tabla.

```
        Pestaña «Centro»                Pestaña «Norte»
      Cuad.  Plumas  Moch.            Cuad.  Plumas  Moch.
Lun   1200    800    500        Lun    300    200    100
Mar   1500    900    700        Mar    400    250    150
Mié   1000    700    500        Mié    350    200    100
Jue   1300    900    600        Jue    300    150    100
Vie   2000   1200    700        Vie    500    300    200
Sáb   1200    800    400        Sáb    300    200    100
Dom    800    500    300        Dom    200    100    100
```

Mateo mira las dos pestañas lado a lado. Algo le pica en la nuca. Pero primero tiene que meter todo en Python.

En Python, la hoja completa es una lista de pestañas:

```python
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
```

Para señalar un número necesitas **tres datos: la sucursal, el día y el producto**. `papeleria[1][4][2]` es Norte (pestaña 1), viernes (renglón 4), mochilas (columna 2): 200. Tres direcciones, **tres dimensiones**. Ya no tiene nombre especial: es un **tensor de tres dimensiones**.

Esta es la imagen que conviene que te quede de toda la serie: **un tensor de tres dimensiones es una hoja de cálculo con pestañas, donde todas las pestañas tienen las mismas columnas y el mismo número de renglones.**

| Lo que tienes | Para señalar un número necesitas | Dimensiones | Nombre |
|---|---|---|---|
| $18,500 | nada: solo hay uno | 0 | escalar |
| Ventas de 7 días | el día | 1 | vector |
| Ventas por día y producto | el día y el producto | 2 | matriz |
| Lo mismo, por sucursal | la sucursal, el día y el producto | 3 | tensor de 3 dimensiones |

**La regla que lo resume todo:** el número de dimensiones es el número de datos que necesitas para señalar un solo número. A cada una de esas direcciones también se le llama **eje**; vas a ver las dos palabras.

—¿Y eso es todo? ¿Un cubo? —Doña Lupe lo mira por encima de los lentes—. Yo pensé que el banco usaba cosas de la NASA.

—Usa cubos, tía. Nomás que muy grandes.

### Paso 5 — Contar las dimensiones con código

Contar corchetes a ojo funciona con ejemplos chicos. Con lo que el banco tiene en la mano, no. Mateo lo sabe: si va a pelear contra una máquina, la suya tiene que contar sola. Antes, el Python que vas a usar de aquí en adelante, por si no lo tienes fresco:

- `type(x) == list` pregunta si `x` es una lista.
- `x[0]` toma el primer elemento; `len(x)` cuenta cuántos hay.
- `lista.append(valor)` agrega un valor al final de una lista; `tuple(lista)` la convierte en tupla.

Crea un archivo `dimensiones.py` con las listas de los pasos 1 a 4 y, arriba de ellas, esta función:

```python
def contar_dimensiones(x):
    dimensiones = 0
    while type(x) == list:
        dimensiones = dimensiones + 1
        x = x[0]
    return dimensiones
```

Cómo funciona: piensa en las muñecas rusas. Mientras lo que tiene en la mano sea una lista, abre la caja —toma el primer elemento, `x[0]`— y cuenta uno. Cuando lo que sale ya es un número, se detiene. Una lista de listas de listas se abre tres veces: tres dimensiones. Un número no se abre ninguna: cero.

Agrega al final del archivo:

```python
if __name__ == "__main__":
    print(contar_dimensiones(total))
    print(contar_dimensiones(por_dia))
    print(contar_dimensiones(por_dia_y_producto))
    print(contar_dimensiones(papeleria))
```

Córrelo con `python3 dimensiones.py`:

```
0
1
2
3
```

La línea `if __name__ == "__main__":` es un detalle de Python: hace que el código de abajo solo corra cuando ejecutas este archivo directamente. Por ahora cópiala tal cual.

Cuatro números en la pantalla. Doña Lupe se asoma.

—¿Ya? ¿Eso salva a Norte?

—No, tía. Eso me dice que entiendo lo que tengo. Que no es poco.

### Paso 6 — La forma: cuántos hay en cada dirección

Saber que la papelería tiene tres dimensiones no basta. Un detective no anota «hay sospechosos»: anota cuántos. Mateo también quiere saber **cuántos elementos hay en cada una**: 2 sucursales, 7 días, 3 productos. Eso se llama la **forma** del tensor, y se escribe como una tupla: `(2, 7, 3)`.

La forma te dice mucho de un vistazo:

- Cuántas dimensiones hay: cuántos números tiene la tupla (3).
- Qué tan grande es cada una: 2, 7 y 3.
- Cuántos números hay en total: se multiplican, 2 × 7 × 3 = 42.

Cuarenta y dos números. Toda una semana de dos sucursales cabe en cuarenta y dos números.

La función es casi igual a la anterior. En lugar de contar, anota el largo de cada caja que abre:

```python
def forma(x):
    tamanos = []
    while type(x) == list:
        tamanos.append(len(x))
        x = x[0]
    return tuple(tamanos)
```

Cambia el bloque final de tu archivo por este:

```python
if __name__ == "__main__":
    print("total", contar_dimensiones(total), forma(total))
    print("por_dia", contar_dimensiones(por_dia), forma(por_dia))
    print("por_dia_y_producto", contar_dimensiones(por_dia_y_producto), forma(por_dia_y_producto))
    print("papeleria", contar_dimensiones(papeleria), forma(papeleria))

    assert forma(papeleria) == (2, 7, 3), "La papelería debería ser 2 sucursales × 7 días × 3 productos"
    print("Todo cuadra")
```

Salida:

```
total 0 ()
por_dia 1 (7,)
por_dia_y_producto 2 (7, 3)
papeleria 3 (2, 7, 3)
Todo cuadra
```

Dos cosas nuevas en esa salida:

- `(7,)` lleva una coma porque así escribe Python una tupla de un solo elemento. Sin la coma, `(7)` sería solo el número 7.
- `()` es una tupla vacía: el escalar no tiene ninguna dirección.

Y una línea nueva en el código: `assert`. Quiere decir **«esto tiene que ser cierto»**. Es como el guardia de la entrada: si lo es, no pasa nada y el programa sigue. Si no, el programa se detiene y te dice por qué. Haz la prueba: cambia el `(2, 7, 3)` por `(7, 2, 3)` y vuelve a correrlo:

```
Traceback (most recent call last):
  ...
AssertionError: La papelería debería ser 2 sucursales × 7 días × 3 productos
```

Regrésalo a `(2, 7, 3)`. Así vas a trabajar en toda la serie: cada ejercicio trae su `assert`, y el programa te dice solo si te equivocaste, sin esperar a que nadie te revise.

—«Todo cuadra» —lee doña Lupe en voz alta—. Ojalá el banco pensara igual.

### Paso 7 — Cuándo una lista de listas NO es un tensor

Son casi las diez. Mateo bosteza y se le ocurre la pregunta incómoda: ¿y si un día alguien manda una pestaña incompleta? ¿Su función se daría cuenta?

Hay una condición que no se negocia: **todas las cajas del mismo nivel tienen que medir lo mismo.** Todas las pestañas con 7 renglones, todos los renglones con 3 columnas. Un tensor es rectangular.

Mira esta lista:

```python
chueca = [[1, 2, 3], [4, 5]]
```

Python la acepta sin problema: es una lista de dos listas. Pero el primer renglón tiene 3 elementos y el segundo, 2. No es una tabla, es una tabla chueca. Y nuestras funciones te mienten en la cara:

```
>>> contar_dimensiones(chueca)
2
>>> forma(chueca)
(2, 3)
```

`forma` dice `(2, 3)` porque solo midió el primer renglón. Como un testigo que vio la primera cuadra y jura por toda la calle. Si confías en ella y pides `chueca[1][2]`, truena:

```
IndexError: list index out of range
```

Por eso hace falta revisar. Para una tabla (dos dimensiones), esta función compara el largo de cada renglón contra el del primero. Con más dimensiones la revisión es más larga; no te preocupes, en la próxima sesión PyTorch la hace completa por ti:

```python
def es_rectangular(tabla):
    largo = len(tabla[0])
    for fila in tabla:
        if len(fila) != largo:
            return False
    return True
```

```
>>> es_rectangular(por_dia_y_producto)
True
>>> es_rectangular(chueca)
False
```

Las bibliotecas de IA, como PyTorch, hacen esta revisión por ti: al convertir una lista chueca en un tensor normal, se niegan. Lo vas a ver la próxima sesión. (Existen tensores especiales para datos disparejos, pero en esta serie no los usamos.)

Mateo se queda viendo ese `False`. Un dato chueco no avisa. Se esconde, hasta que algo truena.

### Más allá de tres: la dirección postal

—¿Y si le pido cuatro semanas, tía, en vez de una?

—Te las doy. Pero me ayudas a acomodar las cartulinas.

Con las ventas de 4 semanas, Mateo necesitaría cuatro datos para señalar un número: semana, sucursal, día y producto. Cuatro dimensiones, forma `(4, 2, 7, 3)`.

Cuatro dimensiones no se dibujan, pero se entienden igual que una dirección postal. Para llegar a un departamento das el estado, la colonia, el edificio, el piso y el número de depa: cinco datos, cinco dimensiones. Nadie se asusta por eso. Cada dato elige una caja dentro de la anterior, igual que cada índice de `papeleria[1][4][2]`.

Algunos tensores reales que vas a encontrar:

| Qué es | Dimensiones | Forma de ejemplo |
|---|---|---|
| Una foto a color pequeña | alto × ancho × color | `(480, 640, 3)` |
| Un lote de 32 fotos | foto × alto × ancho × color | `(32, 480, 640, 3)` |
| 10 segundos de video a 30 cuadros por segundo | cuadro × alto × ancho × color | `(300, 720, 1280, 3)` |
| Resumen diario de 1,000 clientes durante 30 días, con 5 datos por cliente y día | cliente × día × dato | `(1000, 30, 5)` |

Aquí escribimos las fotos como alto × ancho × color porque así quedan nuestras listas. El orden no es universal: en muchos programas de PyTorch vas a ver color × alto × ancho. Lo importante es saber qué es cada eje.

Ese video, guardado sin comprimir, son 829,440,000 números (multiplica la forma). Por eso los videos se guardan comprimidos (un archivo MP4 lleva adentro video comprimido), y por eso trabajar con video sin comprimir pide mucha memoria.

Mateo mira el último renglón de la tabla: 1,000 clientes, 30 días, 5 datos. Así, más o menos, debe verse lo que el banco tiene de la terminal de Norte. Y él apenas va en 42 números.

---

## ¿De dónde salió la palabra «tensor»?

Ya pasada la medianoche, Mateo sigue despierto en su cuarto. No puede dejar de leer. Resulta que la herramienta que va a usar contra el banco tiene una historia de gente igual de atorada que él.

Nadie inventó los tensores solo.

**1846.** El irlandés William Rowan Hamilton usa por primera vez la palabra *tensor*, del latín *tendere*, «estirar». Pero para él significaba otra cosa: el tamaño de un cuaternión, un tipo de número que él mismo inventó. La palabra nace, pero todavía no sabe qué va a ser.

**1898.** El alemán Woldemar Voigt le da el sentido moderno. Estudiaba cristales, que se estiran y se deforman distinto según la dirección en que los jales, y necesitaba una sola cosa que guardara todas esas direcciones a la vez. Una caja para muchas direcciones. Te suena, ¿verdad?

**1900.** Los italianos Gregorio Ricci-Curbastro y Tullio Levi-Civita publican, bien ordenadas, las matemáticas que hoy llamamos cálculo tensorial (ellos le decían «cálculo diferencial absoluto»). Y aquí un detalle que parece pista de novela: en todo su artículo no aparece ni una vez la palabra «tensor».

**1912.** Albert Einstein regresa a Zúrich atorado con su teoría de la gravedad. No le salen las cuentas. Busca a su amigo de la universidad, el matemático Marcel Grossmann. Según recordaba años después un compañero de ambos, Einstein le dijo:

—Grossmann, ayúdame o me vuelvo loco.

Grossmann le enseña el trabajo de los italianos. El de Ricci-Curbastro y Levi-Civita. El que no decía «tensor».

**1915.** Marzo. Llega una carta para Einstein. La firma Levi-Civita. Le dice que una demostración de la versión de su teoría que tenía entonces está mal.

Imagínalo: Einstein, corregido por carta.

Se cruzan cartas hasta mayo. Einstein defiende su prueba con terquedad. Las cartas van y vienen. Al final, Einstein termina aceptando que estaba incompleta. En una carta del 2 de abril le escribió: «Nunca había tenido una correspondencia tan interesante». Meses después cambió de camino, y el 25 de noviembre presentó las ecuaciones finales de la relatividad general, escritas con tensores.

Hay una lección ahí que no es de matemáticas: corregir a alguien no es pelearse. El «tu demostración está mal» de Levi-Civita no terminó en pleito: le mostró a Einstein una falla de verdad, meses antes de las ecuaciones finales, y terminó en una amistad.

*Fuentes: Jeff Miller, «Earliest Known Uses of Some of the Words of Mathematics», entrada TENSOR (Hamilton 1846, Voigt 1898, Ricci-Curbastro y Levi-Civita), en MacTutor: [mathshistory.st-andrews.ac.uk/Miller/mathword/t/](https://mathshistory.st-andrews.ac.uk/Miller/mathword/t/). El artículo de Ricci-Curbastro y Levi-Civita es «Méthodes de calcul différentiel absolu et leurs applications», Mathematische Annalen 54, pp. 125–201; el tomo lleva fecha de 1900 y algunos catálogos lo citan como de 1901. Las cartas de 1915: Galina Weinstein, «Einstein the Stubborn: Correspondence between Einstein and Levi-Civita», [arxiv.org/abs/1202.4305](https://arxiv.org/abs/1202.4305).*

Mateo apaga la lámpara pensando en eso. Si a Einstein le decían «está mal», a él también se lo van a decir. Mejor que se lo diga un `assert` hoy que el banco el 23 de diciembre.

### Una verdad para que nadie te sorprenda

Si un día un físico te dice que un tensor «no es una tabla de números», tiene razón, desde su lado. En física y en matemáticas, un tensor es una caja de números que además obedece reglas precisas cuando giras los ejes o cambias de unidades. En programación y en IA, *tensor* es solo la caja ordenada. Esa caja puede guardar los números de un tensor de la física, pero no trae por sí sola sus reglas ni su significado. Los programadores le tomaron prestado el nombre a la física porque suena mejor que «arreglo de varias dimensiones», que es lo que es.

Y una trampa de vocabulario: en inglés, al número de dimensiones a veces le dicen *rank*. No lo confundas con el «rango» de una matriz que quizá viste en álgebra lineal; son cosas distintas.

---

👉 **Si eres de Administración:** la parte de números de tu hoja de cálculo con pestañas (sin encabezados ni notas) ya es un tensor de tres dimensiones, siempre que todas las pestañas tengan las mismas columnas y los mismos renglones. Doña Lupe lo tenía en su laptop desde hace años y no lo sabía. Cuando en una junta alguien diga «los datos entran al modelo como tensores», piensa en eso: tablas apiladas, todas con la misma forma. Y la regla del paso 7 vale para tus reportes: si cada sucursal te manda su pestaña con columnas distintas, nadie puede sumarlas sin arreglarlas primero.

---

## Ejercicios de la sesión 1

Las soluciones están al final de esta entrega. Intenta primero; el `assert` te dice si acertaste.

**Ejercicio 1 — Predice antes de correr.** Para cada dato, escribe en papel cuántas dimensiones tiene, su forma y qué representa cada dimensión. Después compruébalo con tus funciones.

```python
temperatura = 31
calificaciones = [[9, 8, 10], [7, 9, 8], [10, 10, 9], [6, 8, 7]]
foto = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
lote = [foto, foto]
```

Cuando ya tengas tus respuestas en papel, escríbelas en estos `assert` (cambia cada `?`) y corre:

```python
assert (contar_dimensiones(temperatura), forma(temperatura)) == (?, ?)
assert (contar_dimensiones(calificaciones), forma(calificaciones)) == (?, ?)
assert (contar_dimensiones(foto), forma(foto)) == (?, ?)
assert (contar_dimensiones(lote), forma(lote)) == (?, ?)
print("Ejercicio 1 superado")
```

Pista: `calificaciones` son 4 alumnos con 3 materias cada uno. `foto` es una imagen de 2 × 2 puntos, y cada punto trae tres números: rojo, verde y azul.

**Ejercicio 2 — Señala el número.** Con la `papeleria` del paso 4, escribe el índice que da cada dato y compruébalo con un `assert`:

a) Lo que vendió el Centro el miércoles en plumas.
b) Lo que vendió Norte el viernes en mochilas.
c) Lo que vendió Norte el domingo en cuadernos.

Ejemplo de cómo se ve uno resuelto: `assert papeleria[0][0][0] == 1200` (Centro, lunes, cuadernos).

**Ejercicio 3 — Completa la función.** Esta versión de `forma` tiene dos huecos. Llénalos sin ver el paso 6 y comprueba con los `assert`:

```python
def forma2(x):
    tamanos = []
    while type(x) == list:
        tamanos.append(____)
        x = ____
    return tuple(tamanos)

assert forma2(por_dia) == (7,)
assert forma2(papeleria) == (2, 7, 3)
assert forma2(31) == ()
print("Ejercicio 3 superado")
```

**Ejercicio 4 — Tus propios datos.** Escribe como listas anidadas las calificaciones de 3 alumnos, en 4 materias, en 3 parciales. Inventa los números. Decide tú el orden de las dimensiones y escríbelo en un comentario. Después:

a) Escribe el `assert` de su forma.
b) Escribe el índice del segundo parcial de la tercera materia del primer alumno, y compruébalo con un `assert` contra la calificación que tú inventaste en ese lugar.

Este ejercicio no tiene una sola respuesta correcta: depende del orden que elijas. Lo que sí tiene que cumplirse es que el `assert` pase y que tu comentario diga qué es cada dimensión.

**Ejercicio 5 — Encuentra la chueca.** Una de estas tablas no puede ser un tensor. Encuéntrala a ojo y compruébalo con `es_rectangular`:

```python
a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8]]
c = [[0], [0], [0]]
```

Escribe tus respuestas en estos `assert` (cambia cada `?` por `True` o `False`):

```python
assert es_rectangular(a) == ?
assert es_rectangular(b) == ?
assert es_rectangular(c) == ?
print("Ejercicio 5 superado")
```

¿Qué te diría `forma(b)`? ¿Por qué no es confiable?

---

## Errores comunes (y cómo darte cuenta)

| Error | Cómo se ve | Cómo darte cuenta | Qué hacer |
|---|---|---|---|
| Contar los índices desde 1 | Pides el viernes con `por_dia[5]` y te da el sábado | El número no coincide con la tabla | Los índices empiezan en 0: lunes es 0, viernes es 4 |
| Invertir el orden de los índices | `papeleria[4][1][2]` en vez de `papeleria[1][4][2]` | `IndexError`, porque solo hay 2 sucursales | El orden de los índices es el orden de las listas: de afuera hacia adentro |
| Confiar en `forma` con una lista chueca | `forma` dice `(2, 3)` y luego un índice truena | `es_rectangular` da `False` | Arregla los datos antes de usarlos; nunca rellenes con números inventados sin decirlo |
| Pasarle una lista vacía a `forma` o a `contar_dimensiones` | `IndexError: list index out of range` dentro de la función | Truena con `[]` | Una caja vacía no tiene primer elemento que abrir; revisa que tus datos tengan algo |
| Olvidar la coma en `(7,)` | `assert forma(por_dia) == (7)` falla aunque la forma es correcta | El `assert` truena con algo que «se ve bien» | Una tupla de un elemento lleva coma: `(7,)` |

---

## Repaso para la próxima sesión

La sesión 2 empieza con estas tres preguntas. Contéstalas sin ver la lección:

1. ¿Cómo sabes cuántas dimensiones tiene un conjunto de datos, sin contar corchetes?
2. ¿Qué te dice la forma `(2, 7, 3)`, y cuántos números hay en total?
3. ¿Qué condición tiene que cumplir una lista de listas para ser un tensor?

🔁 **La próxima semana:** Mateo deja las listas y agarra una herramienta en serio. Instalas PyTorch y conviertes la papelería de doña Lupe en tu primer tensor de verdad. Vas a ver que PyTorch ya trae la forma integrada (ahí se llama `shape`), y que se niega a aceptar listas chuecas.

Esa noche, antes de cerrar la laptop, Mateo volvió a abrir las dos pestañas. Centro, Norte. Centro, Norte. Las mismas columnas, los mismos siete días. Y sin embargo, en la pestaña de Norte había algo que no cuadraba. Pero para verlo necesitaba una herramienta de verdad.

---

## Soluciones (no las veas antes de intentarlo)

**Ejercicio 1.**

```
temperatura      0 ()             un solo número
calificaciones   2 (4, 3)         alumno × materia
foto             3 (2, 2, 3)      renglón de puntos × punto × color
lote             4 (2, 2, 2, 3)   foto × renglón × punto × color
```

Los `assert` quedan así: `(0, ())`, `(2, (4, 3))`, `(3, (2, 2, 3))` y `(4, (2, 2, 2, 3))`.

El `lote` agrega una dimensión al principio: cuál foto. Así trabajan casi siempre los modelos de imágenes: reciben lotes, y hasta una sola foto suele ir como un lote de una.

**Ejercicio 2.**

```python
assert papeleria[0][2][1] == 700   # a) Centro, miércoles, plumas
assert papeleria[1][4][2] == 200   # b) Norte, viernes, mochilas
assert papeleria[1][6][0] == 200   # c) Norte, domingo, cuadernos
```

**Ejercicio 3.** Los huecos son `len(x)` y `x[0]`: anota el largo de la caja y después ábrela.

**Ejercicio 4.** Una respuesta posible, con el orden alumno × materia × parcial:

```python
# alumno × materia × parcial
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]
assert forma(parciales) == (3, 4, 3)
assert parciales[0][2][1] == 9   # primer alumno, tercera materia, segundo parcial
```

Si elegiste materia × alumno × parcial, tu forma es `(4, 3, 3)` y tu índice es `parciales[2][0][1]`. Las dos están bien si el comentario lo dice. Elegir el orden de las dimensiones es una decisión de diseño, y en la sesión 3 vas a ver por qué importa.

**Ejercicio 5.** Los `assert` son `True`, `False` y `True`. La chueca es `b`: su último renglón tiene 2 elementos y los demás, 3. `es_rectangular(b)` da `False`. `forma(b)` diría `(3, 3)` porque solo mide el primer renglón, y no es confiable por eso: no revisa los demás.

---

## Lo que sigue

En la sesión 2 dejas las listas y usas una herramienta hecha para esto. Si quieres adelantar, revisa que tengas Python 3.11 o más nuevo con `python3 --version`.

---

*Escribo desde lo que vemos todos los días en Hábil, donde construimos sistemas para banca, seguros y empresas en México. Si en tu trabajo o en el negocio de tu familia se topan con un problema así, platiquemos: [habil.mx](https://habil.mx)*
