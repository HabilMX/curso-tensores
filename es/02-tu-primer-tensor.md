# Tensores desde cero (2): tu primer tensor

*Instalas PyTorch, conviertes la papelería de doña Lupe en un tensor de verdad y descubres que la función que programaste la semana pasada ya venía de fábrica.*

*Tensores desde cero · lección A, sesión 2 de 8*

*Escenario: los personajes y la historia son inventados. Los datos técnicos, históricos y las cifras reales llevan su fuente.*

**Lunes 9 de noviembre, 7:40 p.m. Trastienda de la sucursal Centro.**

Huele a papel nuevo y a café recalentado. Doña Lupe baja la cortina metálica hasta la mitad, con ese rechinido que se oye hasta la esquina, y se sienta junto a Mateo.

—Ya llegó el archivo de Norte —dice, y le pone el celular enfrente—. El de la semana. Ese es el que me importa.

Mateo lo pasa a su laptop. Siete días, tres productos: cuadernos, plumas y mochilas. Lo de siempre. En el calendario de la pared, alguien encerró en rojo el 23 de diciembre. Faltan cuarenta y cuatro días.

—¿Y ahora qué le vas a hacer? ¿Otra vez tus listitas?

La semana pasada Mateo escribió la papelería como listas dentro de listas y programó a mano dos funciones: una que cuenta dimensiones y otra que calcula la forma. Funcionaron. Pero también descubrió algo incómodo: su función `forma` se deja engañar por una lista chueca. Y no avisa.

—Las listas ya no me alcanzan, tía. Necesito una herramienta de verdad.

—¿Y cuánto cuesta?

—Nada. Es gratis.

Doña Lupe lo mira por encima de los lentes.

—Lo barato sale caro, mijo.

—Esta no. Es la que usan los que hacen inteligencia artificial.

—Peor tantito.

Mateo se ríe y abre la terminal. Pero antes de instalar nada, se pone una regla: contestar las tres preguntas de la semana pasada sin ver sus notas. Si no puede, no tiene caso seguir. Una herramienta nueva sobre una base floja nada más hace más rápido los errores.

## Repaso para abrir

Antes de empezar, contesta sin ver la sesión 1. Las respuestas están al final, en las soluciones.

1. ¿Cómo sabes cuántas dimensiones tiene un conjunto de datos, sin contar corchetes?
2. ¿Qué te dice la forma `(2, 7, 3)`, y cuántos números hay en total?
3. ¿Qué condición tiene que cumplir una lista de listas para ser un tensor?

Si dudaste en alguna, regresa a la sesión 1 antes de seguir.

> **El código de esta sesión** está en el repositorio público del curso, en la carpeta `sesion-02`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-02). Escríbelo tú primero, porque así se aprende; el repositorio es para comparar cuando algo no te salga igual. Cada programa viene con la salida que debe darte.

## Objetivos de esta sesión

Al terminar, vas a poder hacer estas cinco cosas. Al lado de cada una está el ejercicio que la demuestra.

| # | Al terminar puedes… | Lo demuestra |
|---|---|---|
| 1 | **Instalar** PyTorch en un entorno propio y **comprobar** que funciona | Paso 1 y ejercicio 6 |
| 2 | **Convertir** tus listas anidadas en un tensor con `torch.tensor` | Ejercicios 7 y 8 |
| 3 | **Leer** la forma, las dimensiones y el total de números de un tensor (`.shape`, `.ndim`, `.numel()`) y **comprobarlos** con un `assert` contra tus funciones de la sesión 1 | Ejercicios 6, 7 y 8 |
| 4 | **Predecir** el tipo de número de un tensor (`.dtype`) y **elegirlo** tú cuando haga falta | Ejercicios 6 y 8 |
| 5 | **Leer** el error que da PyTorch con una lista chueca y **decir** dónde está el problema | Ejercicio 9 |

**Qué necesitas:** la sesión 1 terminada, con tu archivo `dimensiones.py` (las listas de la papelería y las funciones `contar_dimensiones` y `forma`). Una computadora con Linux Mint (o cualquier Linux o macOS con Python 3.11 o más nuevo; en Windows el curso no se ha probado), internet para descargar PyTorch y una terminal abierta.

---

## Por qué importa

Mateo vuelve a ver su función `forma`. Un `while`, un `append`, un `tuple`. Funcionó con la papelería. Pero tiene un hueco: con una lista chueca, miente. Y miente muy segura de sí misma.

Eso es lo que no lo deja tranquilo. Si el archivo de Norte viniera mal, su función le daría una forma sin pestañear, y él le creería. Y si el archivo trajera un millón de números, cada cuenta sería otro `for` escrito a mano.

Quien trabaja con inteligencia artificial no programa esas funciones cada vez. Usa una biblioteca que ya guarda los números ordenados, sabe su forma, revisa que sean rectangulares y hace las cuentas muy rápido. La más usada en investigación se llama **PyTorch**.

Hoy aprendes lo primero que hace cualquiera que trabaja con IA cuando le llegan datos nuevos: **crear el tensor y preguntarle cómo es.** ¿Qué forma tiene? ¿Cuántas dimensiones? ¿Cuántos números? ¿De qué tipo? Si esas respuestas no son las que esperas, nada de lo que venga después va a salir bien.

Mateo todavía no lo sabe, pero esta noche una de esas preguntas va a contestar «no».

---

## Paso 1 — Instalar PyTorch

### Un cuarto aparte para tus herramientas

Antes de instalar, vas a crear un **entorno virtual**. Suena complicado y no lo es: es una carpeta donde guardas las bibliotecas de un proyecto, separadas de las del resto de tu computadora.

Piénsalo como la caja de herramientas de tu tío el plomero: si mete sus llaves en el cajón de la cocina, un día no encuentra nada. Con su propia caja, nadie le mueve nada. El entorno virtual es tu caja: lo que instales ahí no le estorba a Linux Mint, que usa su propio Python para funcionar.

En la terminal, primero revisa tu versión de Python:

```bash
python3 --version
```

Necesitas 3.11 o más nuevo: es lo que pide PyTorch desde su versión 2.15. Linux Mint 22 trae Python 3.12, así que vas bien.

**En Linux Mint y Ubuntu, primero esto.** La herramienta para crear entornos viene en un paquete aparte que una instalación nueva no trae. Instálalo (te va a pedir tu contraseña); si ya lo tenías, no pasa nada:

```bash
sudo apt install python3-venv
```

Ahora crea el entorno y actívalo:

```bash
python3 -m venv ~/tensores
source ~/tensores/bin/activate
```

La primera línea crea la caja: una carpeta llamada `tensores` en tu carpeta personal (eso significa `~`). La segunda la abre. Vas a notar que tu terminal ahora empieza con `(tensores)`. Esa es la señal de que estás dentro.

**Si usas Windows:** esas dos líneas son para Linux y macOS. En Windows, desde tu carpeta personal, crea el entorno con `python -m venv tensores` y ábrelo con `.\tensores\Scripts\activate` (así lo indica la [documentación de Python](https://docs.python.org/3/library/venv.html)). Ojo: el curso no se probó en Windows; si una salida no coincide, compárala con la del repositorio antes de suponer que te equivocaste.

**Si te sale este error:** te saltaste el paso de arriba. En Ubuntu 24.04 (la misma base que Linux Mint 22) el mensaje se ve así:

```text
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.12-venv

You may need to use sudo with that command.  After installing the python3-venv
package, recreate your virtual environment.
```

Fíjate que el mensaje propone `python3.12-venv`, con el número de versión, y aquí te pedimos `python3-venv`, sin número. Los dos sirven: `python3-venv` instala por dentro el de tu versión. Preferimos el que no lleva número porque sigue funcionando cuando tu sistema cambie de versión de Python. Corre `sudo apt install python3-venv` y vuelve a correr las dos líneas.

Ojo: **cada terminal nueva empieza con el entorno cerrado.** Vuelve a correr `source ~/tensores/bin/activate` y busca el `(tensores)`. Si no lo ves, Python no va a encontrar PyTorch.

### El comando oficial

Con el entorno activado, actualiza el instalador de paquetes y después instala PyTorch con el comando oficial de su página para Linux, sin tarjeta de video:

```bash
pip3 install --upgrade pip
pip3 install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

Qué dice ese comando, parte por parte:

- `torch==2.14.1` es PyTorch, en la versión exacta con la que se escribió este curso. La fijamos para que tus salidas coincidan con las de la lección: con otra versión los números pueden cambiar un poco y no sabrías si te equivocaste tú.
- `torch` es PyTorch. El nombre viene de **Torch**, una biblioteca anterior escrita en otro lenguaje (Lua), de cuya comunidad salió parte de la gente que lo creó.
- `torchvision` es un paquete compañero para imágenes. Viene en el comando oficial y no estorba.
- `--index-url https://download.pytorch.org/whl/cpu` le dice a `pip` de qué tienda descargar: la de la edición **solo CPU**. Sin esa parte, en Linux te bajaría la edición para tarjetas de video NVIDIA, que pesa mucho más y no la necesitas en este curso.

Es una descarga grande; tómalo con calma.

### Comprobar que quedó

Crea un archivo `verifica.py`, en la misma carpeta donde tienes `dimensiones.py`:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

Córrelo con `python3 verifica.py`:

```
2.14.1 False
```

El primer dato es la versión de PyTorch. Si a ti te sale `2.14.1+cpu`, también está bien: el `+cpu` es la etiqueta de la edición solo CPU. El segundo dato pregunta si PyTorch puede usar una tarjeta de video de NVIDIA (lo que PyTorch llama CUDA). Como instalaste la edición solo CPU, el `False` es justo lo que esperamos, aunque tu laptop tenga tarjeta de video.

**¿Qué es una GPU?** Es la tarjeta de video: nació para pintar millones de puntos de la pantalla al mismo tiempo en los videojuegos. La CPU es un puñado de chefs muy hábiles; la GPU es un equipo de miles de ayudantes que solo saben picar, pero pican todos a la vez, y multiplicar tensores grandes es puro picar. Todo este curso corre en la CPU de una laptop normal.

> Mateo ve el `False` y se ríe. Doña Lupe, que no le quita los ojos a la pantalla, pregunta si eso es malo.
> —Quiere decir que PyTorch no va a usar tarjeta de video en mi laptop, tía.
> —Ah, pues qué bueno. Así estudias.

---

## Paso 2 — Del número y la lista al tensor

> **8:25 p.m.** Mateo tiene el archivo de Norte a un clic. No lo abre. Primero va a practicar con los datos de la semana pasada, los que ya sabe que están bien. Si algo falla con datos buenos, el error es suyo. Si falla con datos nuevos, el error viene en los datos. Mezclar las dos cosas es la forma más rápida de no entender nada.

Hay una sola función que aprender hoy para crear tensores: `torch.tensor`. Le das tus listas y te regresa un tensor con los mismos números.

Crea `primer_tensor.py` en la misma carpeta que `dimensiones.py`. La línea `from dimensiones import ...` toma las listas que escribiste la semana pasada, sin volver a copiarlas:

```python
import torch
from dimensiones import total, por_dia

t_total = torch.tensor(total)
t_dia = torch.tensor(por_dia)

print(t_total)
print(t_dia)
print(t_total.shape, t_dia.shape)
```

Córrelo con `python3 primer_tensor.py`:

```
tensor(18500)
tensor([2500, 3100, 2200, 2800, 3900, 2400, 1600])
torch.Size([]) torch.Size([7])
```

Tres cosas que leer en esa salida:

- PyTorch envuelve tus números en `tensor(...)`: ya no es una lista de Python, es un tensor.
- El escalar se ve como `tensor(18500)`, sin corchetes. Cero dimensiones, igual que en la sesión 1.
- `.shape` es la **forma**. Para el escalar sale `torch.Size([])`, vacía, porque no tiene ninguna dirección. Para la semana sale `torch.Size([7])`: siete días.

Fíjate: `.shape` no lleva paréntesis. No es una función, es un dato que el tensor ya trae guardado, como la etiqueta con la talla de una camisa: no la mides, la lees.

> —¿Y eso qué? —dice doña Lupe, señalando el `18500`—. Es lo mismo que yo ya tenía apuntado.
> —Ese es el chiste, tía. Que sea lo mismo. Ahora la computadora también lo sabe.

---

## Paso 3 — La papelería completa

Ahora la hoja con sus dos pestañas. Crea `papeleria.py` en la misma carpeta; este archivo va a guardar el tensor que usarás en las sesiones siguientes:

```python
import torch
from dimensiones import centro, norte

papeleria = torch.tensor([centro, norte])

# Esto corre al abrir papeleria.py, no cuando otra sesión lo importa.
if __name__ == "__main__":
    print(papeleria)
    print(papeleria.shape)
    print(papeleria.ndim)
    print(papeleria.numel())
    print(papeleria.dtype)
```

Salida:

```
tensor([[[1200,  800,  500],
         [1500,  900,  700],
         [1000,  700,  500],
         [1300,  900,  600],
         [2000, 1200,  700],
         [1200,  800,  400],
         [ 800,  500,  300]],

        [[ 300,  200,  100],
         [ 400,  250,  150],
         [ 350,  200,  100],
         [ 300,  150,  100],
         [ 500,  300,  200],
         [ 300,  200,  100],
         [ 200,  100,  100]]])
torch.Size([2, 7, 3])
3
42
torch.int64
```

La condición `if __name__ == "__main__":` deja esas cinco comprobaciones para cuando corres este archivo directo. Las sesiones siguientes podrán usar `from papeleria import papeleria` sin volver a imprimir toda la tabla.

PyTorch alinea las columnas y deja un renglón en blanco entre la pestaña Centro y la Norte: la misma hoja con pestañas de la semana pasada. Ahora lee las cuatro respuestas:

| Le preguntas | Escribes | Te contesta | Quiere decir |
|---|---|---|---|
| ¿Qué forma tienes? | `papeleria.shape` | `torch.Size([2, 7, 3])` | 2 sucursales, 7 días, 3 productos |
| ¿Cuántas dimensiones? | `papeleria.ndim` | `3` | sucursal, día y producto |
| ¿Cuántos números traes? | `papeleria.numel()` | `42` | 2 × 7 × 3 |
| ¿De qué tipo son? | `papeleria.dtype` | `torch.int64` | enteros (lo vemos en el paso 4) |

Los nombres vienen del inglés: `shape` es «forma»; `ndim`, «número de dimensiones»; `numel`, *number of elements*, «número de elementos»; `dtype`, *data type*, «tipo de dato». Fíjate en un detalle: `numel()` sí lleva paréntesis, porque es una función que cuenta. Los otros tres no.

### `torch.Size([2, 7, 3])` es tu `forma` de la semana pasada

La sesión 1 terminó con esta salida de tu función:

```
papeleria 3 (2, 7, 3)
```

Y hoy PyTorch te dice `torch.Size([2, 7, 3])`. **Es exactamente lo mismo**, escrito con otra ropa. `torch.Size` es la manera en que PyTorch guarda una forma, y por dentro se comporta como una tupla. Por eso lo puedes comparar directo contra `(2, 7, 3)`.

Tu `while` con `append` y `.shape` hacen el mismo trabajo. La diferencia: la de PyTorch, como vas a ver en el paso 5, no se deja engañar por una lista chueca.

### Comprobarlo con `assert`: primero la forma, después el valor

No le creas a nadie, ni a PyTorch: compruébalo. Crea `compara.py`:

```python
import torch
from dimensiones import centro, norte, forma, contar_dimensiones

papeleria_lista = [centro, norte]
papeleria = torch.tensor([centro, norte])

# 1) Primero la forma: PyTorch contra tus funciones de la sesión 1
print(papeleria.shape, forma(papeleria_lista))
print(papeleria.ndim, contar_dimensiones(papeleria_lista))
assert papeleria.shape == (2, 7, 3), "La papelería debería ser 2 sucursales × 7 días × 3 productos"
assert papeleria.shape == forma(papeleria_lista), "PyTorch y tu función deberían dar la misma forma"
assert papeleria.ndim == contar_dimensiones(papeleria_lista), "Deberían contar las mismas dimensiones"

# 2) Después el total de números, contra un for
cuenta = 0
for pestana in papeleria_lista:
    for renglon in pestana:
        for numero in renglon:
            cuenta = cuenta + 1
print(papeleria.numel(), cuenta)
assert papeleria.numel() == cuenta, "Deberían ser los mismos números"

# 3) Al final, el valor: de regreso a listas, ¿quedó igual?
assert papeleria.tolist() == papeleria_lista, "Los datos no deberían cambiar al convertirlos"
print("Todo cuadra")
```

Salida:

```
torch.Size([2, 7, 3]) (2, 7, 3)
3 3
42 42
Todo cuadra
```

Tres comprobaciones, en este orden:

1. **La forma.** Es el `assert` que vas a escribir en cada ejercicio del curso. Si la forma está mal, todo lo demás está mal.
2. **El total contra un `for`.** Los tres `for` anidados cuentan uno por uno lo que `numel()` cuenta en una palabra. Si los dos dan 42, no se perdió ni sobró nada.
3. **El valor.** `.tolist()` hace el viaje de regreso: convierte el tensor otra vez en listas. La lista se llama `papeleria_lista` para no chocar con el tensor `papeleria`; si son iguales, los números no cambiaron en el camino.

Si cambias `(2, 7, 3)` por `(7, 2, 3)`, el primer `assert` truena con su mensaje, igual que en la sesión 1. El orden importa: no es lo mismo 2 pestañas de 7 renglones que 7 pestañas de 2.

> «Todo cuadra.» Mateo lo lee dos veces. Con los datos conocidos, la herramienta no se equivoca y no lo deja equivocarse. Ya puede confiar en ella. Falta una cosa antes del archivo de Norte: saber qué clase de números va a guardar.

---

## Paso 4 — El tipo de número: `dtype`

Todos los números de un tensor son del mismo tipo. No se vale mezclar, como en una lista de Python, un entero con un texto. Ese tipo único se llama **dtype**.

¿Por qué? Porque así la computadora sabe exactamente cuánto espacio ocupa cada número y los forma en fila sin revisar uno por uno, como cajas del mismo tamaño en un anaquel.

Para empezar te bastan dos tipos:

| dtype | Qué guarda | Ejemplo |
|---|---|---|
| `torch.int64` | enteros: piezas, pesos sin centavos, conteos | 12 cuadernos |
| `torch.float32` | números con decimales: mediciones, promedios, porcentajes | 9.7 de promedio |

El número del nombre es cuánto espacio ocupa cada uno, en bits: 64 o 32. No necesitas más por ahora.

Un aviso para cuando trabajes con dinero de verdad: `float32` guarda los decimales de forma aproximada (el 0.1 queda como 0.10000000149011612). Para importes exactos, los sistemas guardan centavos como enteros: $35.50 se guarda como `3550`. En el ejemplo de abajo usamos precios con decimales solo para ver qué `dtype` sale.

Crea `tipos.py`:

```python
import torch

piezas = torch.tensor([12, 30, 7])
precios = torch.tensor([35.5, 12.0, 450.0])
mezcla = torch.tensor([12, 30.5, 7])

print(piezas, piezas.dtype)
print(precios, precios.dtype)
print(mezcla, mezcla.dtype)
```

Salida:

```
tensor([12, 30,  7]) torch.int64
tensor([ 35.5000,  12.0000, 450.0000]) torch.float32
tensor([12.0000, 30.5000,  7.0000]) torch.float32
```

La regla que PyTorch sigue:

- **Si todos son enteros, sale `int64`.** Por eso tu papelería salió `torch.int64`: doña Lupe registra pesos sin centavos.
- **Si al menos uno trae punto decimal, sale `float32`,** y los enteros también se vuelven decimales: el 12 se convierte en `12.0000`. Basta un solo 30.5 para que toda la fila cambie.

> —Yo no apunto centavos —dice doña Lupe—. Los centavos son cosa del banco.
> —Por eso tu hoja salió `int64`, tía. Enteros, como tú.
> —¿Me estás diciendo cuadrada?
> —Te estoy diciendo exacta.

### Elegir tú el tipo

A veces quieres decimales aunque tus datos sean enteros. Por ejemplo, para sacar promedios, como vas a ver en la sesión 4. Para eso, `torch.tensor` acepta un segundo dato:

```python
import torch
from dimensiones import por_dia

dia_dec = torch.tensor(por_dia, dtype=torch.float32)
print(dia_dec)
print(dia_dec.dtype)
```

Salida:

```
tensor([2500., 3100., 2200., 2800., 3900., 2400., 1600.])
torch.float32
```

El punto después de cada número (`2500.`) es la pista de que ya son decimales. Mismos números, otro tipo.

---

## Paso 5 — La lista chueca: PyTorch dice que no

**10:15 p.m.** Ahora sí. Mateo copia los siete renglones del archivo de Norte a una lista, escribe la línea de siempre, `torch.tensor(...)`, y le da Enter.

La terminal se llena de letras rojas.

—¿Ya la descompusiste? —pregunta doña Lupe.

—No, tía. Se negó.

Mateo baja por el archivo, renglón por renglón, con el dedo en la pantalla. Lunes: tres números. Martes: dos. Miércoles: tres. Al martes le falta uno.

Antes de sacar conclusiones, lo reproduce en chiquito. En la sesión 1 viste que tu función `forma` se dejaba engañar por esta lista y tenías que revisarla con `es_rectangular`:

```python
chueca = [[1, 2, 3], [4, 5]]
```

Vamos a dársela a PyTorch. Crea `chueca.py`:

```python
import torch

chueca = [[1, 2, 3], [4, 5]]
t = torch.tensor(chueca)
```

Córrelo:

```
Traceback (most recent call last):
  File ".../chueca.py", line 4, in <module>
    t = torch.tensor(chueca)
        ^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 3 at dim 1 (got 2)
```

El mismo rojo que con el archivo de Norte. PyTorch se niega a crear el tensor. Lo importante es la última línea, y se lee palabra por palabra:

| Dice | Traducido | Qué significa aquí |
|---|---|---|
| `ValueError` | error de valor | el dato que le diste no sirve para esto |
| `expected sequence of length 3` | esperaba una lista de largo 3 | el primer renglón medía 3, así que todos deberían medir 3 |
| `at dim 1` | en la dimensión 1 | la dimensión de adentro: los renglones. La de afuera es la 0 |
| `(got 2)` | recibí 2 | encontró un renglón que mide 2 |

Fíjate en el `dim 1`. Las dimensiones se numeran desde 0, igual que los índices de una lista: la de afuera es la 0, la siguiente es la 1. En la papelería, la dimensión 0 son las sucursales, la 1 los días y la 2 los productos. Esa numeración la vas a usar mucho en la sesión 3.

Es tu `es_rectangular`, pero mejor: revisa todas las cajas de todos los niveles y te dice dónde está el problema. Por eso el tensor que creas con `torch.tensor` siempre es rectangular. (PyTorch tiene tensores especiales para datos disparejos, pero en este curso no los usamos.)

Tu función de la semana pasada se habría tragado el archivo de Norte sin decir nada. PyTorch no: prefiere tronar que mentirte.

**Cómo se arregla:** no rellenando con un número inventado. Mateo podría poner un cero en el martes y seguir. El tensor saldría perfecto, y estaría mintiendo: un cero dice «no se vendió nada»; un hueco dice «no sé». No es lo mismo. Pregúntale al que te dio los datos cuál falta. Si faltan las mochilas del martes, el tensor no se arregla con un cero; se arregla preguntando.

> —¿Quién arma este archivo, tía?
> —Lo manda la sucursal. Cada lunes. Nunca había llegado mal.
> Nunca. Mateo se queda viendo el martes.

---

## Recuadro: la pelea de dos gigantes (PyTorch y TensorFlow)

Mientras PyTorch se descargaba, Mateo se hizo una pregunta: si esta herramienta es tan buena, ¿por qué en todos lados oye otro nombre? Vas a oír dos nombres todo el tiempo, y detrás de ellos hay una pelea entre dos gigantes. Los datos son de la investigación del curso, con fecha.

**El que llegó primero.** **TensorFlow** es de Google. Lo liberó como código abierto el **9 de noviembre de 2015**, con permiso para que cualquiera lo use: once años exactos antes de la noche en que Mateo instaló a su rival. La versión vigente a octubre de 2026 es la **2.21.0**, publicada el 6 de marzo de 2026.

**El retador.** **PyTorch** nació en Facebook (hoy Meta), en su laboratorio de investigación. Sus primeras versiones en GitHub son de **septiembre de 2016**, diez meses después, y su lanzamiento oficial, de enero de 2017. Desde **septiembre de 2022** ya no es de una sola empresa: lo cuida la PyTorch Foundation, dentro de la Linux Foundation, y sus miembros fundadores fueron AMD, AWS (de Amazon), Google Cloud, Meta, Microsoft Azure y NVIDIA. Fíjate en un nombre de esa lista: Google Cloud, de la misma casa que el rival, también está sentado ahí. La versión vigente cuando escribimos esta sesión (octubre de 2026) es la **2.14.1**, publicada el 30 de septiembre de 2026. Si a ti se te instaló una más nueva, está bien.

**¿Quién va ganando?** Depende del ring.

- **En los modelos que la gente publica, PyTorch, por mucho.** En Hugging Face, el sitio donde la gente publica modelos de IA, el 18 de octubre de 2026 había 258,777 modelos con la etiqueta de PyTorch contra 17,469 con la de TensorFlow: unos 15 por cada uno. Eso mide lo que la gente sube, no lo que usa en su trabajo, pero la diferencia es clara. Y la biblioteca de modelos más popular de Hugging Face, Transformers, anunció que en su versión 5 deja TensorFlow y se queda solo con PyTorch.
- **Entre programadores, casi empate.** En la encuesta de Stack Overflow de 2024, el 10.6 % de quienes contestaron dijo usar PyTorch y el 10.1 %, TensorFlow. La encuesta pregunta qué usa cada quien, no qué corre en producción. TensorFlow llegó antes a las empresas y sigue vivo en muchas, y en celulares.

*Fuentes: [anuncio de Google sobre TensorFlow (2015)](https://blog.google/innovation-and-ai/products/tensorflow-smarter-machine-learning-for/); versiones en PyPI: [tensorflow](https://pypi.org/project/tensorflow/) y [torch](https://pypi.org/project/torch/); versiones de PyTorch en [GitHub](https://github.com/pytorch/pytorch/releases); [Linux Foundation, 12 de septiembre de 2022](https://www.linuxfoundation.org/press/press-release/meta-transitions-pytorch-to-the-linux-foundation); [modelos de PyTorch en Hugging Face](https://huggingface.co/models?library=pytorch), consultado el 18 de octubre de 2026 (la cifra cambia cada día); [encuesta de Stack Overflow 2024](https://survey.stackoverflow.co/2024/technology).*

**Una diferencia que vas a notar si un día usas los dos:** con los mismos enteros, PyTorch crea `int64` y TensorFlow, `int32`.

| | PyTorch | TensorFlow |
|---|---|---|
| Crear un tensor | `torch.tensor([1, 2, 3])` | `tf.constant([1, 2, 3])` |
| Entero por defecto | `int64` | `int32` |
| Con decimales | `float32` | `float32` |

**¿Cuál usar?** Para aprender y leer código de investigación, PyTorch. Si un día tu trabajo usa TensorFlow, lo aprendes en una tarde: forma, dimensiones y tipo funcionan igual.

---

👉 **Si eres de Administración:** el `dtype` es el formato que le das a una columna en tu hoja de cálculo: «número entero» o «moneda con dos decimales». Y el error de la lista chueca es el reporte de la sucursal que llega con una columna de menos: no se suma con los demás hasta que alguien pregunta qué falta. Cuando un analista te diga «el modelo no corrió porque los datos no tenían la forma correcta», ya sabes de qué habla.

---

## Ejercicios de la sesión 2

Las soluciones están al final de esta entrega. Trabaja en la misma carpeta que tu `dimensiones.py` y con el entorno activado.

**Ejercicio 6 — Predice antes de correr.** Escribe en papel la forma, las dimensiones, el tipo y el total de números de cada tensor. Después corre y compara.

```python
import torch

temperatura = torch.tensor(31)
calificaciones = torch.tensor([[9, 8, 10], [7, 9, 8], [10, 10, 9], [6, 8, 7]])
foto = torch.tensor([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])
promedios = torch.tensor([9.0, 8.0, 9.7, 7.0])

for nombre, t in [("temperatura", temperatura), ("calificaciones", calificaciones),
                  ("foto", foto), ("promedios", promedios)]:
    print(nombre, t.shape, t.ndim, t.dtype, t.numel())
```

Son los mismos datos del ejercicio 1 de la sesión 1, más uno con decimales. Pista: ¿cuántos números trae un escalar?

**Ejercicio 7 — Completa los huecos.** La pestaña Norte sola, como tensor. Llena los `____` y corre. Si el `print` final aparece, acertaste.

```python
import torch
from dimensiones import norte

norte_t = torch.____(norte)

assert norte_t.shape == ____, "Norte es una pestaña: 7 días × 3 productos"
assert norte_t.ndim == ____, "Una pestaña es una tabla"
assert norte_t.numel() == ____, "7 × 3"
assert norte_t.dtype == ____, "Son pesos enteros"
assert norte_t.tolist() == norte, "Los datos no deberían cambiar"
print("Ejercicio 7 superado")
```

**Ejercicio 8 — Tus propios datos, ahora en tensor.** Toma las calificaciones que escribiste en el ejercicio 4 de la sesión 1 (alumnos, materias y parciales) y:

a) Conviértelas en tensor y escribe el `assert` de su forma contra tu función `forma`.
b) Escribe el `assert` de sus dimensiones contra `contar_dimensiones`.
c) Cuenta sus números con `for` anidados y compáralo con un `assert` contra `numel()`.
d) Crea una segunda versión con `dtype=torch.float32` y comprueba con un `assert` que cambió el tipo pero no la forma.

Termina con `print("Ejercicio 8 superado")`.

**Ejercicio 9 — Encuentra la chueca, ahora con PyTorch.** Antes de correr, predice cuáles truenan y qué número de `dim` va a decir el error. Después prueba cada una por separado con `torch.tensor(...)`.

```python
a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8]]
c = [[[1, 2], [3, 4]], [[5, 6], [7]]]
d = [[1, 2], [3, 4, 5]]
```

Para las que truenan, contesta: ¿por qué en `d` el error dice que esperaba 2 y recibió 3, si el renglón que «sobra» es el segundo? Y arregla `b` agregándole el número que le falta (supón que es un 9).

---

## Errores comunes (y cómo darte cuenta)

| Error | Cómo se ve | Cómo darte cuenta | Qué hacer |
|---|---|---|---|
| Olvidar activar el entorno | `ModuleNotFoundError: No module named 'torch'` | Tu terminal no empieza con `(tensores)` | Corre `source ~/tensores/bin/activate` en cada terminal nueva |
| Llamar a tu archivo `torch.py` | `AttributeError: partially initialized module 'torch' has no attribute 'tensor' (most likely due to a circular import)` | Truena en la primera línea que usa `torch.` | Python importó tu archivo en lugar de PyTorch. Cámbiale el nombre, por ejemplo a `primer_tensor.py` |
| Poner paréntesis a `.shape` | `TypeError: 'torch.Size' object is not callable` | Escribiste `t.shape()` | `.shape`, `.ndim` y `.dtype` van sin paréntesis; solo `.numel()` los lleva |
| Olvidar los paréntesis de `numel` | Sale algo como `<built-in method numel of Tensor object at 0x...>` en vez de un número | El `print` no muestra un número | Escribe `t.numel()`, con paréntesis |
| Escribir `torch.Tensor` con mayúscula | `torch.Tensor([1, 2, 3])` da `tensor([1., 2., 3.])`, decimales | El `dtype` sale `torch.float32` con datos enteros | Usa `torch.tensor`, con minúscula: respeta el tipo de tus datos |

---

## Repaso para la próxima sesión

La sesión 3 empieza con estas tres preguntas. Contéstalas sin ver la lección:

1. ¿Qué te dice `t.shape` y cómo lo compruebas con un `assert`?
2. ¿Por qué `torch.tensor([[1, 2, 3], [4, 5]])` falla?
3. ¿Qué `dtype` sale de `torch.tensor([1, 2, 3])` y cuál de `torch.tensor([1.0, 2.0])`?

🔁 **La próxima semana:** ya tienes la papelería como tensor. Ahora vas a sacarle pedazos por cualquiera de sus ejes: solo la pestaña Norte, solo los viernes de las dos sucursales, solo las mochilas de toda la semana. Y vas a ver que una foto se recorta exactamente igual.

Pasan de las once cuando Mateo cierra la laptop. Doña Lupe ya contó la caja y espera junto a la cortina con las llaves en la mano. En el celular sigue abierto el archivo de Norte, con su martes de dos números. No parece un dedazo cualquiera: es el único renglón chueco en semanas de archivos derechos. Y llegó justo la semana en que el banco suspendió la terminal.

¿Quién deja un día a medias?

---

## Soluciones (no las veas antes de intentarlo)

**Repaso para abrir.**

1. Cuentas cuántos datos necesitas para señalar un solo número. En la papelería necesitas la sucursal, el día y el producto: tres datos, tres dimensiones. Y si no quieres contar tú, lo cuenta `contar_dimensiones` (y desde hoy, `.ndim`).
2. Que hay 2 elementos en la primera dimensión (sucursales), 7 en la segunda (días) y 3 en la tercera (productos). En total, 2 × 7 × 3 = 42 números.
3. Que sea rectangular: todas las cajas del mismo nivel tienen que medir lo mismo. Todas las pestañas con 7 renglones, todos los renglones con 3 columnas.

**Ejercicio 6.**

```
temperatura torch.Size([]) 0 torch.int64 1
calificaciones torch.Size([4, 3]) 2 torch.int64 12
foto torch.Size([2, 2, 3]) 3 torch.int64 12
promedios torch.Size([4]) 1 torch.float32 4
```

La trampa era el escalar: su forma está vacía y tiene cero dimensiones, pero trae **un** número. `promedios` sale `float32` porque trae decimales. Fíjate también en que `calificaciones` y `foto` traen los mismos 12 números con formas distintas: el total no te dice la forma, pero la forma sí te dice el total.

**Ejercicio 7.**

```python
norte_t = torch.tensor(norte)

assert norte_t.shape == (7, 3), "Norte es una pestaña: 7 días × 3 productos"
assert norte_t.ndim == 2, "Una pestaña es una tabla"
assert norte_t.numel() == 21, "7 × 3"
assert norte_t.dtype == torch.int64, "Son pesos enteros"
```

Salida:

```
Ejercicio 7 superado
```

**Ejercicio 8.** Una respuesta posible, con la lista `parciales` (alumno × materia × parcial) de la solución del ejercicio 4 de la sesión 1:

```python
import torch
from dimensiones import forma, contar_dimensiones

# alumno × materia × parcial
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]

p = torch.tensor(parciales)
print(p.shape, p.ndim, p.numel(), p.dtype)
assert p.shape == forma(parciales), "PyTorch y tu función deberían coincidir"
assert p.ndim == contar_dimensiones(parciales)

cuenta = 0
for alumno in parciales:
    for materia in alumno:
        for calificacion in materia:
            cuenta = cuenta + 1
assert p.numel() == cuenta

p_dec = torch.tensor(parciales, dtype=torch.float32)
print(p_dec.dtype)
assert p_dec.dtype == torch.float32
assert p_dec.shape == p.shape, "Cambiar el dtype no cambia la forma"
print("Ejercicio 8 superado")
```

Salida:

```
torch.Size([3, 4, 3]) 3 36 torch.int64
torch.float32
Ejercicio 8 superado
```

Si tus datos tienen otro orden u otros tamaños, tu forma será distinta, y está bien: lo que importa es que los `assert` pasen.

**Ejercicio 9.**

- `a` funciona: `torch.Size([3, 2])`.
- `b` truena: `ValueError: expected sequence of length 3 at dim 1 (got 2)`. El último renglón mide 2 y los demás, 3.
- `c` truena: `ValueError: expected sequence of length 2 at dim 2 (got 1)`. Aquí el problema está más adentro, en la dimensión 2: el último par, `[7]`, solo trae un número.
- `d` truena: `ValueError: expected sequence of length 2 at dim 1 (got 3)`. PyTorch toma el **primer** renglón como la regla. El primero mide 2, así que el segundo, que mide 3, es el que «está mal», aunque a ti te parezca que al primero le falta uno. Por eso el error te dice dónde se rompió la regla, pero decidir qué dato es el correcto te toca a ti, con quien te dio los datos.

`b` arreglado:

```python
b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(torch.tensor(b).shape)
```

```
torch.Size([3, 3])
```

---

## Lo que sigue

En la sesión 3 dejas de ver el tensor completo y empiezas a sacarle pedazos. Si quieres adelantar, abre tu `papeleria.py` y deja listo el tensor `papeleria`: es lo primero que vas a usar.

---

*Escribo desde lo que vemos todos los días en Hábil, donde construimos sistemas para banca, seguros y empresas en México. Si en tu trabajo o en el negocio de tu familia se topan con un problema así, platiquemos: [habil.mx](https://habil.mx)*
