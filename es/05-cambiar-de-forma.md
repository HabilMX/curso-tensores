# Tensores desde cero (5): cambiar de forma sin cambiar los datos

*Doce meses en una fila, cuatro trimestres o dos semestres: los mismos números acomodados de otra manera, y cómo leer el error que sale cuando pides una forma imposible.*

*Tensores desde cero · lección B, sesión 5 de 8*

*Escenario: los personajes y la historia son inventados. Los datos técnicos, históricos y las cifras reales llevan su fuente.*

**Lunes 30 de noviembre, 9:15 p.m. Cocina de doña Lupe.**

La olla de los frijoles seguía tibia en la estufa. Sobre el mantel de hule había una carpeta verde, gorda, amarrada con una liga.

—Fer sigue sin contestar —dijo doña Lupe—. Una semana, Mateo. Una semana.

—A lo mejor está enferma, tía.

—A lo mejor. —No sonaba convencida.

Le acercó la carpeta. Adentro venía la libreta del contador: lo que vendió el Centro cada mes del año pasado. Doce renglones a mano, con pluma azul.

—El contador los quiere por trimestre. Y el licenciado Ortega, por semestre, para «su revisión». Cada quien los quiere en su cajita.

—Son los mismos doce números.

—Pues díselo a ellos.

Mateo los capturó en la laptop. Doce números en fila. No tenía que cambiar ninguno. Solo acomodarlos distinto. Y algo le daba vueltas: si veía un año sano acomodado por trimestres, sabría qué buscar en Norte. Qué mes debía ser fuerte. Cuál no.

—¿Qué mes vendes más?

—Agosto, mijo. Regreso a clases. Las mamás hacen fila hasta la calle. Casi todas pagan con tarjeta.

Mateo apuntó *agosto* en un post-it y lo pegó en la orilla de la pantalla. No sabía todavía por qué.

En el celular de doña Lupe brilló una notificación. Un correo del banco. Ella lo abrió, lo leyó con los lentes en la punta de la nariz y se lo pasó sin decir nada.

*«Le informamos que los primeros cargos inusuales registrados en la terminal de su sucursal Norte corresponden a la segunda quincena de agosto.»*

El post-it seguía ahí. Amarillo. *Agosto*.

---

Con esta sesión empieza la segunda lección, **Los tensores trabajando**. Hasta aquí, Mateo aprendió a mirar: crear un tensor, sacarle pedazos y resumirlo. Así descubrió que el Centro vende unas cuatro veces lo que Norte: $18,500 contra $4,600 en la semana. Ahora los tensores se ponen a trabajar, y cada herramienta lo acerca un paso más a lo que pasó en Norte: le cambias la forma a un tensor sin tocar sus datos, haces cuentas con tensores completos sin un solo `for`, entrenas un modelo que detecta fraudes y ves dónde aparecen en un banco, una aseguradora y una tienda. Faltan 23 días para el plazo del banco.

| Sesión | Qué ves | Práctica |
|---|---|---|
| **5 — Cambiar de forma sin cambiar los datos** | `reshape` y `flatten`, y cómo leer el error de forma | Ejercicios 18 a 21 |
| **6 — Operar tensores completos** | Hacer cuentas con todo el tensor a la vez, sin `for` | Ejercicios 22 a 25 |
| **7 — Un detector de fraude que aprende solo** | Entrenar un modelo pequeño con tensores | Ejercicios 26 a 29 |
| **8 — Tensores en el banco, la aseguradora y la tienda** | Leer un caso real y decir qué forma tiene su tensor | Ejercicios 30 a 33, autoevaluación y repaso de la lección B |

Como en la lección A, cada sesión es de unas 2 horas: una para leer y otra para practicar.

En la sesión anterior sumaste por pestaña con `sum(dim=)` y lo comparaste contra un `for`. Hoy no cambias ningún número: cambias cómo están acomodados. Igual que Mateo esa noche, con la libreta del contador abierta junto a la laptop.

## Repaso para abrir

Contesta sin ver la sesión 4. Las respuestas están en las soluciones, al final.

1. Si sumas `papeleria` con `dim=1`, ¿qué forma queda y qué significa?
2. ¿Por qué `mean` falla con enteros y cómo lo arreglas?
3. ¿Cómo compruebas que tu suma con tensores está bien?

> **El código de esta sesión** está en el repositorio público del curso, en la carpeta `sesion-05`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-05). Escríbelo tú primero, porque así se aprende; el repositorio es para comparar cuando algo no te salga igual. Cada programa viene con la salida que debe darte.

## Objetivos de esta sesión

Al lado de cada objetivo está el ejercicio que lo demuestra.

| # | Al terminar puedes… | Lo demuestra |
|---|---|---|
| 1 | **Decir**, sin correr nada, si una forma nueva (con o sin un `-1`) es válida para un tensor, y **calcular** la dimensión que falta cuando hay un `-1` | Ejercicio 18 |
| 2 | **Reacomodar** un tensor con `reshape`, usando `-1` para que PyTorch calcule un tamaño, y **comprobarlo** contra un `for` | Ejercicios 19 y 20 |
| 3 | **Leer** palabra por palabra el error de forma de PyTorch y **arreglar** el código que lo provoca | Ejercicio 20 |
| 4 | **Distinguir** entre reacomodar una tabla y girarla, y **detectar** con un `assert` cuando alguien las confundió | Ejercicio 21 |
| 5 | **Aplanar** un tensor con `flatten` y **restaurar** su forma sin perder valores | Ejercicio 19, parte c |

**Qué necesitas saber antes:** las sesiones 1 a 4: crear un tensor, leer su `shape`, indexar, rebanar y sumar con `sum(dim=)`; y en Python, usar `for`, `range`, listas, `append`, `tolist()` y `assert` (abajo, en el paso 3, va un recordatorio). **Qué necesitas tener:** el entorno de la sesión 2 con PyTorch, y el archivo `papeleria.py` de la sesión 4 (el tensor `papeleria` de forma `(2, 7, 3)`). Los archivos de hoy van en esa misma carpeta.

**Alcance de hoy:** la decisión nueva que practicas es si `reshape` conserva el orden que necesitas. `flatten()` es el caso de llevar todo a una fila. `.T` aparece solo para contrastar con `reshape`: escribe esa línea y deja que los `assert` la revisen. El paso 10 y el recuadro de TensorFlow son lectura para ver dónde se usa esto en IA: corre `foto.py` si quieres, pero ningún ejercicio lo necesita, y no hace falta instalar TensorFlow.

---

## Por qué importa

—¿Y cómo los acomodas sin que se te revuelvan? —preguntó doña Lupe.

Mateo señaló el cartón de huevos junto a la estufa.

Una docena cabe en un cartón de 2 filas por 6 o en uno de 3 por 4: mismos 12 huevos, otro cartón. Y si te piden meterla en uno de 5 por 3, sabes que no se puede: tiene 15 huecos y tú tienes 12 huevos.

—Eso hasta yo lo sé —dijo ella—. Y si sobra un hueco, alguien se comió un huevo.

Con los tensores pasa todo el tiempo: los datos llegan con una forma y el siguiente paso los necesita con otra. Doña Lupe tiene ventas por mes y su contador las quiere por trimestre. Una foto llega como cuadrícula y el modelo la quiere como una fila de números. No se inventa ni se pierde un solo número: solo se cambia el cartón. Eso es **reshape** («cambiar de forma», en inglés).

Y una razón práctica: unos de los errores que más vas a ver con tensores son los de forma: pedir un cartón que no cuadra con tus huevos. Hoy aprendes a leerlos sin asustarte. Y vas a conocer el otro error, el peligroso: el que no truena. El cartón se ve igual, cuadra, nadie se queja… y los huevos ya no son los que eran.

## El caso: las ventas del año de doña Lupe

Son los doce renglones de la libreta del contador: lo que vendió el Centro cada mes del año pasado, en pesos. Un año sano, sin sobresaltos. Agosto es su mes fuerte, por el regreso a clases, y diciembre también, por las mochilas de regalo. Mateo lo va a usar como molde: así se ve un año normal en una papelería de doña Lupe.

### Paso 1 — Los doce meses en una fila

Crea un archivo `cambiar_forma.py`:

```python
import torch

#        ene    feb    mar    abr    may    jun    jul    ago     sep    oct    nov    dic
meses = [60000, 55000, 60000, 50000, 65000, 50000, 70000, 100000, 70000, 60000, 55000, 90000]
ventas = torch.tensor(meses)
print(ventas.shape)
print(ventas.sum())
```

Córrelo con `python3 cambiar_forma.py`:

```
torch.Size([12])
tensor(785000)
```

Nada nuevo: 12 números que suman $785,000. Ese total es tu testigo: vas a cambiar la forma muchas veces, y nunca debe moverse.

### Paso 2 — Cuatro trimestres de tres meses

El contador pide las ventas por trimestre: enero a marzo, abril a junio, y así. Quieres una tabla de **4 renglones (trimestres) por 3 columnas (meses)**.

Agrega al final del archivo:

```python
trimestres = ventas.reshape(4, 3)
print(trimestres)
print(trimestres.shape)
print(trimestres.sum())
```

Las líneas nuevas de la salida (arriba siguen saliendo las de antes):

```
tensor([[ 60000,  55000,  60000],
        [ 50000,  65000,  50000],
        [ 70000, 100000,  70000],
        [ 60000,  55000,  90000]])
torch.Size([4, 3])
tensor(785000)
```

Cuatro renglones, tres columnas y el mismo total. `reshape(4, 3)` quiere decir «acomódalos en 4 por 3».

**¿Cómo decide PyTorch qué número va en cada lugar?** Lee la fila original de izquierda a derecha, como un libro, y llena la tabla nueva renglón por renglón, igual que acomodas huevos en un cartón: primero la primera fila, luego la segunda.

Como los meses venían en orden, cada renglón quedó justo como un trimestre. No es suerte: el orden de lectura coincide con lo que quieres. Guarda esta idea; en el paso 9 la vas a ver fallar.

Ahora ya puedes usar todo lo de la lección A sobre la tabla nueva. Agrega:

```python
print(trimestres.sum(dim=1))   # total de cada trimestre
print(trimestres[3, 2])        # cuarto trimestre, tercer mes: diciembre
print(ventas.shape)            # ¿se descompuso la original?
```

Salida nueva:

```
tensor([175000, 165000, 240000, 205000])
tensor(90000)
torch.Size([12])
```

- El tercer trimestre (julio a septiembre) es el más fuerte: $240,000, por el regreso a clases.
- `trimestres[3, 2]` es diciembre, $90,000: renglón 3 (cuarto trimestre), columna 2 (tercer mes).
- `ventas` sigue con forma `(12,)`. **`reshape` no le cambia la forma al tensor original: te entrega uno con la forma nueva.** Por eso lo guardas en una variable, como `trimestres`. Ojo: no siempre es una copia aparte; muchas veces los dos comparten los mismos números por dentro. Hoy no importa, porque no modificas ninguno.

> Doña Lupe se asomó por encima de su hombro.
>
> —¿Y cómo sabes que la máquina no se los inventó?
>
> —No lo sé, tía. Por eso lo voy a comprobar.
>
> —Eso me gusta. Ni a la computadora hay que creerle a la primera.

### Paso 3 — Comprobarlo contra un `for`

No le creas a PyTorch solo porque sí. Calcula los totales con un `for` y compara.

**Recordatorio de Python que vas a usar:** `range(4)` produce 0, 1, 2 y 3; `lista.append(x)` agrega `x` al final de una lista; `tensor.tolist()` convierte un tensor en una lista normal para compararla con otra lista.

Y una herramienta que usarás en toda la serie. Un `assert` es una revisión automática: si la condición es verdadera, no imprime nada y el programa sigue; si es falsa, el programa se detiene y muestra el mensaje entre comillas. Aquí funciona como tu control del error: no tienes que creerle al resultado, puedes comprobarlo. Si quieres ver cómo se ve un fallo, prueba esto en una consola de Python:

```python
# Prueba opcional para reconocer un fallo:
assert 2 == 3, "2 no es igual a 3"
# Sale: AssertionError: 2 no es igual a 3
```

Ahora sí, agrega:

```python
a_mano = []
for t in range(4):
    total = 0
    for m in range(3):
        total = total + meses[t * 3 + m]
    a_mano.append(total)

assert trimestres.shape == (4, 3), "Deberían ser 4 trimestres × 3 meses"
assert trimestres.sum(dim=1).tolist() == a_mano, "Los totales por trimestre no cuadran con el for"
print("Trimestres comprobados")
```

Salida nueva:

```
Trimestres comprobados
```

Fíjate en `meses[t * 3 + m]`: es la cuenta que hace `reshape` por dentro. Diciembre (trimestre 3, mes 2) está en la posición 3 × 3 + 2 = 11 de la fila original. `reshape` te ahorra esa cuenta. Y como siempre: primero el `assert` de la forma, después el de los valores.

### Paso 4 — Dos semestres, y hasta tres dimensiones

El licenciado Ortega le pide a doña Lupe sus ventas por semestre: dos mitades de seis meses. Agrega:

```python
semestres = ventas.reshape(2, 6)
print(semestres)
print(semestres.sum(dim=1))

anio = ventas.reshape(2, 2, 3)   # semestre × trimestre × mes
print(anio)
print(anio[1, 0, 2])
```

Salida nueva:

```
tensor([[ 60000,  55000,  60000,  50000,  65000,  50000],
        [ 70000, 100000,  70000,  60000,  55000,  90000]])
tensor([340000, 445000])
tensor([[[ 60000,  55000,  60000],
         [ 50000,  65000,  50000]],

        [[ 70000, 100000,  70000],
         [ 60000,  55000,  90000]]])
tensor(70000)
```

Primer semestre, $340,000; segundo, $445,000. Suman los $785,000 de siempre.

La forma `(2, 2, 3)` convierte el año en un **cubo**: la hoja con pestañas de la sesión 1, una pestaña por semestre y en cada una una tabla de trimestres por meses. `anio[1, 0, 2]` es el segundo semestre, su primer trimestre, su tercer mes: septiembre, $70,000. Doce números, cuatro formas, y nada se perdió.

> Mateo se quedó viendo el segundo semestre. Ahí estaba el número más grande de todo el año, solo, en medio de su trimestre: agosto, $100,000. En un año sano, agosto es el mes en que más gente pasa la tarjeta.
>
> Miró el post-it. Miró el correo del banco: *segunda quincena de agosto*.
>
> Si alguien quisiera esconder cobros falsos, ¿qué mes escogería? El mes en que nadie se fija, porque todo el mundo está pagando.

### Paso 5 — La regla: multiplica la forma

> **Un `reshape` con tamaños positivos es válido cuando la forma nueva, multiplicada, da el mismo total de números que la original.** Con `-1` hay una regla aparte, en el paso 6.

Revísalo antes de correr:

| Forma que pides | Multiplicada | ¿Sirve para 12 meses? |
|---|---|---|
| `(4, 3)` | 4 × 3 = 12 | Sí |
| `(2, 2, 3)` | 2 × 2 × 3 = 12 | Sí |
| `(5, 3)` | 5 × 3 = 15 | No: te faltan 3 huevos |
| `(5, 2)` | 5 × 2 = 10 | No: te sobran 2 huevos |

`reshape` nunca rellena con ceros ni tira lo que sobra. Si no cuadra, se niega.

### Paso 6 — El `-1`, que calcula solo

A veces sabes cuánto quieres en una dimensión y te da flojera dividir para la otra. Pones **`-1`** en el lugar que no quieres calcular. Agrega:

```python
print(ventas.reshape(-1, 3).shape)
print(ventas.reshape(6, -1).shape)
print(ventas.reshape(2, -1, 3).shape)
print(ventas.reshape(-1).shape)
```

Salida nueva:

```
torch.Size([4, 3])
torch.Size([6, 2])
torch.Size([2, 2, 3])
torch.Size([12])
```

**`-1` quiere decir «tú calcula cuánto va aquí para que cuadre»**. En `reshape(-1, 3)`, PyTorch piensa: «12 números en columnas de 3: son 4 renglones». En `reshape(2, -1, 3)`: «2 × algo × 3 = 12; algo es 2».

Y `reshape(-1)`, solito, quiere decir «ponlo todo en una sola dimensión», tenga la forma que tenga: una lista de números, como la `(12,)` del principio.

**La regla con `-1`:** multiplica los tamaños que sí escribiste; ese producto tiene que repartir exacto el total de números (sin que sobre nada). El `-1` vale el total entre ese producto. En `reshape(2, -1, 3)`: 2 × 3 = 6, y 12 ÷ 6 = 2.

- **Solo puede haber uno.** Con `ventas.reshape(-1, -1)` sale `RuntimeError: only one dimension can be inferred` («solo se puede deducir una dimensión»).
- **No hace milagros.** Con `reshape(-1, 5)` no hay número que por 5 dé 12, y truena. El `-1` calcula; no inventa huevos.
- **Es el único negativo que vale.** Con `ventas.reshape(-2, -6)` sale `RuntimeError: invalid shape dimension -2 at index 0 of shape [-2, -6]`, aunque -2 × -6 sea 12.

### Paso 7 — El error de forma, leído palabra por palabra

Provoquemos el error a propósito. Un compañero quiso «5 grupos de 3 meses». Agrega al final del archivo:

```python
quintos = ventas.reshape(5, 3)
```

Córrelo (recorté la ruta con `...`; en tu computadora es otra):

```
Traceback (most recent call last):
  File ".../cambiar_forma.py", line 42, in <module>
    quintos = ventas.reshape(5, 3)
              ^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[5, 3]' is invalid for input of size 12
```

Un mensaje de error se lee **de abajo hacia arriba**: la última línea dice qué pasó; las de arriba, dónde. La última, palabra por palabra:

| Pedazo | Qué quiere decir |
|---|---|
| `RuntimeError` | «Error al correr»: el código estaba bien escrito, pero falló mientras corría. |
| `shape '[5, 3]'` | «La forma [5, 3]». Es la forma que **tú pediste**. |
| `is invalid` | «no es válida». |
| `for input of size 12` | «para una entrada de tamaño 12». Es lo que **tú tienes**: 12 números. |

Junto: **«La forma 5 × 3 no sirve para algo que tiene 12 números.»** 5 × 3 = 15, y 15 no es 12. El error no te regaña: te da lo que pediste y lo que tienes.

Las de arriba dicen dónde: en tu archivo, línea 42 (a ti te puede salir otro número si dejaste más o menos renglones en blanco), y copian la línea culpable con `^^^^` debajo del pedazo que tronó.

Si querías grupos de 3 meses, el arreglo es `reshape(4, 3)` o `reshape(-1, 3)`. Si de verdad necesitabas 5 grupos, el problema no es el código: 12 no se reparte en 5 iguales. Borra esa línea antes de seguir.

**La receta para cualquier error de forma:** busca en la última línea los dos tamaños (lo que pediste y lo que tienes), multiplica, y pregúntate qué querías en realidad.

> —Qué decente —dijo doña Lupe cuando Mateo le leyó el error—. Te dice qué pediste y qué tienes. Ojalá el banco fuera así.
>
> Mateo se rio. Luego dejó de reírse. *Lo que pediste y lo que tienes.* Lo que se vendió y lo que se cobró. Si alguien pusiera esos dos números juntos, uno al lado del otro…
>
> Lo apuntó en otro post-it. Eso tendría que esperar.

### Paso 8 — Aplanar la papelería con `flatten`

Regresemos a la papelería `(2, 7, 3)`. A veces necesitas sus 42 números en una sola fila, para guardarlos o dárselos a un modelo. Eso es **aplanar**, como desarmar una caja de cartón: el mismo cartón, sin volumen.

En PyTorch se llama **`flatten`** («aplanar»). Antes de crear el archivo, comprueba que tu `papeleria.py` se puede importar. Abre la Terminal en la carpeta donde lo guardaste y corre:

```bash
python3 -c "from papeleria import papeleria; print(papeleria.shape)"
```

Debe imprimir `torch.Size([2, 7, 3])`. Si dice `ModuleNotFoundError`, coloca `papeleria.py` en esta misma carpeta y abre la Terminal ahí. Si importa pero dice que no encuentra `papeleria`, vuelve al archivo de la sesión 4 y revisa que esa sea la variable guardada.

Ahora sí, crea `aplanar.py`:

```python
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
```

Guarda el archivo y córrelo desde la misma carpeta con `python3 aplanar.py`. Si aparece un error, no avances: compara primero la última línea con la salida esperada.

Salida:

```
torch.Size([2, 7, 3])
torch.Size([42])
tensor([1200,  800,  500, 1500,  900,  700])
tensor([300, 200, 100])
tensor(23100) tensor(23100)
Ida y vuelta sin perder nada
```

- **2 × 7 × 3 = 42** números, que suman los mismos $23,100.
- **El orden de lectura es el de siempre**, de la caja de afuera hacia la de adentro: toda la pestaña Centro, día por día y producto por producto, y después toda la Norte. Los primeros seis son Centro lunes y martes. Los primeros 21 son del Centro (7 × 3), así que las posiciones 21 a 23 son Norte lunes: 300, 200, 100.
- **Se puede regresar**, siempre que recuerdes qué forma tenía.

`flatten()` hace lo mismo que `reshape(-1)`, pero su nombre dice lo que hace.

### Paso 9 — La trampa: `reshape` no es girar la tabla

Lo más importante de la sesión, porque aquí el error **no truena**.

`papeleria.sum(dim=1)` es la tabla de la semana: 2 sucursales × 3 productos. Doña Lupe la quiere al revés para un reporte: **productos en renglones, sucursales en columnas**, 3 × 2. ¿`reshape(3, 2)`?

Crea `girar.py`:

```python
from papeleria import papeleria

semana = papeleria.sum(dim=1)    # sucursal × producto
print(semana)
print(semana.shape)

mal = semana.reshape(3, 2)
print(mal)
```

Córrelo con `python3 girar.py`. Salida:

```
tensor([[9000, 5800, 3700],
        [2350, 1400,  850]])
torch.Size([2, 3])
tensor([[9000, 5800],
        [3700, 2350],
        [1400,  850]])
```

PyTorch no se quejó: 3 × 2 = 2 × 3. Pero léelo como reporte:

- El renglón de cuadernos debería traer Centro 9,000 y Norte 2,350. Trae **9,000 y 5,800**: cuadernos y plumas del Centro.
- El de plumas debería traer 5,800 y 1,400. Trae **3,700 y 2,350**: mochilas del Centro y cuadernos del Norte.

Un revoltijo. `reshape` hizo lo único que sabe: leer en orden (9000, 5800, 3700, 2350, 1400, 850) y llenar renglones de 2 en 2. **`reshape` cambia el cartón; no sabe qué huevo es de quién.**

Lo que querías es **girar** la tabla: renglones a columnas. En PyTorch es `.T` (de *transponer*, el nombre matemático). Se escribe `semana.T`, **sin paréntesis**: no se llama como función, describe la tabla girada. Agrega:

```python
bien = semana.T
print(bien)
print(bien.shape)
```

Salida nueva:

```
tensor([[9000, 2350],
        [5800, 1400],
        [3700,  850]])
torch.Size([3, 2])
```

Ahora sí: cuadernos 9,000 y 2,350; plumas 5,800 y 1,400; mochilas 3,700 y 850. Misma forma `(3, 2)` que el revoltijo: **mismo tamaño, distinto contenido.** Por eso el `assert` de la forma nunca va solo. De `.T` no necesitas más por ahora: es lo que sí gira.

**¿Y por qué con los trimestres sí funcionó?** Porque ahí no girabas nada: cortabas una fila en pedazos de 3, en el mismo orden. La pregunta antes de cada `reshape`: **¿quiero cortar en pedazos, en el mismo orden de lectura?** Si sí, `reshape`. Si quieres que los renglones se vuelvan columnas, `reshape` te da un revoltijo sin avisarte.

> *Mismo tamaño, distinto contenido.* Mateo lo dijo en voz alta.
>
> —¿Qué cosa? —preguntó doña Lupe, que ya estaba lavando la olla.
>
> —Algo que se ve igual por fuera puede ser otra cosa por dentro. Como el revoltijo: pasa la revisión de la forma y nadie se da cuenta.
>
> Doña Lupe cerró la llave del agua. Se secó las manos en el mandil, despacio.
>
> —La terminal de Norte la cambiaron en agosto.
>
> Mateo levantó la vista.
>
> —¿Quién?
>
> —Un técnico. Fer me mandó una foto al celular: «Ya vino el del banco, ya quedó la terminal nueva». Igualita a la otra. Y en la orilla de la foto, un señor de playera azul, de espaldas. Yo pensé que lo había mandado el banco.
>
> —¿Tú lo llamaste?
>
> —No.
>
> —¿Y Fer?
>
> Doña Lupe no contestó. En la mesa, el post-it seguía diciendo *agosto*.

### Paso 10 — Por qué la IA vive cambiando de forma (lectura; correrlo es opcional)

**Una foto se aplana.** Una foto en blanco y negro de 28 × 28 puntos es una tabla, y muchos modelos sencillos esperan una fila de números por ejemplo.

**Un lote se reacomoda.** Los modelos ven fotos en lotes, por ejemplo de 32: un cubo `(32, 28, 28)` que el modelo quiere como 32 filas, una por foto.

Si quieres verlo correr, crea `foto.py` y córrelo con `python3 foto.py`. `torch.zeros` crea un tensor de la forma que pidas lleno de ceros, para no escribir miles de números:

```python
import torch

foto = torch.zeros(28, 28)           # una foto de 28 × 28 puntos, toda negra
print(foto.shape, foto.numel())
print(foto.flatten().shape)

lote = torch.zeros(32, 28, 28)       # 32 fotos
print(lote.reshape(32, -1).shape)
print(lote.flatten().shape)
```

Salida:

```
torch.Size([28, 28]) 784
torch.Size([784])
torch.Size([32, 784])
torch.Size([25088])
```

- La foto aplanada: 784 números en fila (28 × 28).
- `lote.reshape(32, -1)`: 32 renglones, uno por foto; el `-1` calcula los 784.
- Ojo con `lote.flatten()`: aplana **todo**, también la separación entre fotos. Quedan 25,088 números (32 × 784) y la forma ya no dice dónde empieza cada foto: solo lo sabes si recuerdas que eran 32 de 784. Con lotes quieres `reshape(32, -1)`, que guarda esa separación en la forma.

---

> **Recuadro: lo mismo en TensorFlow.** Solo para comparar; no lo ejecutes ni instales TensorFlow para esta sesión: todos los objetivos y ejercicios se resuelven con PyTorch. La idea es idéntica: `tf.reshape(ventas, (4, 3))` en lugar de `ventas.reshape(4, 3)`, con el mismo `-1`. El mensaje de error incluye: `Input to reshape is a tensor with 12 values, but the requested shape has 15`. TensorFlow multiplica por ti (5 × 3 = 15); PyTorch te da la forma y tú multiplicas.

---

👉 **Si eres de Administración:** esto hace el contador de doña Lupe cada fin de trimestre, y esto haces tú al convertir un reporte mensual en trimestral: los mismos números agrupados distinto, y el total del año no se mueve. Esa es tu prueba de control, como el `assert` del paso 3. Y la trampa del paso 9 vive en Excel: si reacomodas una tabla a mano en vez de usar «Pegado especial → Transponer», los números caen en celdas ajenas sin ninguna alerta. Después de reacomodar, revisa un dato que te sepas de memoria.

---

## Ejercicios de la sesión 5

Las soluciones están al final. Intenta primero. Los `assert` revisan las partes de código; para las predicciones y explicaciones, compara tu respuesta con la salida esperada y con la solución al final.

**Ejercicio 18 — Predice antes de correr.** Con los 12 meses de `ventas`, escribe en papel si cada forma es válida y qué forma final daría:

```
(3, 4)    (6, 2)    (5, 2)    (2, -1)    (-1, 5)    (2, 3, 2)    (12, 1)
```

Luego compruébalas una por una en la consola de Python con `ventas` ya creada. Si todavía tienes en `cambiar_forma.py` la línea `quintos = ventas.reshape(5, 3)` del paso 7, bórrala (si no, el archivo se detiene en ese error) y abre la consola con:

```bash
python3 -i cambiar_forma.py
```

Cuando aparezca `>>>`, escribe una forma por vez. Si no aparece `>>>`, el archivo se detuvo en otro error: resuélvelo antes de hacer el ejercicio. Para salir, escribe `exit()`. Se ve así (no copies el `>>>`: lo pone Python, tú escribes solo lo que sigue):

```text
>>> ventas.reshape(3, 4).shape
```

Así se ve una resuelta: `(4, 3)` → 4 × 3 = 12 → válida → `torch.Size([4, 3])`. En las inválidas, revisa que el error diga los dos tamaños que esperabas.

**Ejercicio 19 — La tabla larga.** Muchas hojas guardan los datos en una tabla larga: un renglón por sucursal y día, una columna por producto. **Partes a) y b):** completa los huecos y contesta la pregunta de abajo.

```python
from papeleria import papeleria

larga = papeleria.reshape(____, 3)
assert larga.shape == (14, 3), "Deberían ser 14 renglones (sucursal y día) × 3 productos"

a_mano = []
for s in range(2):
    for d in range(7):
        a_mano.append(papeleria[s, d].____())
assert larga.tolist() == a_mano, "La tabla larga no coincide con la armada a mano"
print("Ejercicio 19 superado")
```

Después, sin correr nada, di qué renglón de `larga` ocupa Norte miércoles. Compruébalo con `print(larga[9])`, copia los tres números que salieron y agrega:

```python
assert larga[9].tolist() == papeleria[1, 2].tolist(), "Ese renglón debería ser Norte miércoles"
```

**Parte c) — Aplanar.** Sin ver el paso 8, completa los huecos para aplanar la papelería y regresarla a su forma:

```python
plana = papeleria.____()
de_vuelta = plana.____(2, 7, 3)

assert plana.shape == (42,), "Deberían quedar 42 números en una fila"
assert de_vuelta.tolist() == papeleria.tolist(), "La ida y vuelta debe conservar cada dato"
print("Ejercicio 19c superado")
```

**Ejercicio 20 — Lee el error y arréglalo.** Las ventas diarias del Centro en un febrero de 28 días que empezó en lunes: 4 semanas exactas. Un compañero quiso acomodarlas en semanas:

```python
import torch

dias_febrero = [
    2500, 3100, 2200, 2800, 3900, 2400, 1600,
    2600, 3000, 2300, 2700, 4000, 2500, 1500,
    2400, 3200, 2100, 2900, 3800, 2300, 1700,
    2700, 3300, 2400, 3000, 4200, 2600, 1800,
]
febrero = torch.tensor(dias_febrero)
por_semana = febrero.reshape(7, 5)
```

a) Córrelo, copia la última línea del error y explica estas cuatro partes: `RuntimeError`, `shape '[7, 5]'`, `is invalid` e `input of size 28`. Después explica por qué 7 × 5 no puede acomodar 28 días.

b) Arréglalo para que `por_semana` quede como 4 semanas × 7 días (lunes a domingo). Escribe el `assert` de la forma y compara los totales por semana contra un `for` sobre la lista `dias_febrero`, como en el paso 3 (el día `d` de la semana `s` está en `dias_febrero[s * 7 + d]`).

c) Otro compañero dice: «yo lo arreglé con `reshape(7, 4)`, que también da 28». No truena. ¿Por qué está mal para lo que querían? Para verlo, agrega:

```python
mal = febrero.reshape(7, 4)
print(mal[0])
```

Compara esa salida con los siete días de lunes a domingo de la primera semana.

**Ejercicio 21 — Atrapa la trampa.** Cubre el paso 9 antes de intentarlo y escribe un programa que:

1. Calcule `semana = papeleria.sum(dim=1)`.
2. Construya la tabla de productos × sucursales (3 × 2) correctamente.
3. Tenga un `assert` de la forma y otro que revise que el renglón de cuadernos trae `[9000, 2350]`.
4. Compruebe con dos `for` que `girada[p, s]` es igual a `semana[s, p]` para todos.

Después, para probar la trampa, reemplaza **solo** la línea de la tabla girada (en la solución es `girada = semana.T`) por:

```python
girada = semana.reshape(3, 2)
```

No cambies los `assert` ni los dos `for`. Córrelo. ¿Cuál `assert` truena? ¿Por qué PyTorch no te avisó antes?

---

## Errores comunes (y cómo darte cuenta)

| Error | Cómo se ve | Cómo darte cuenta | Qué hacer |
|---|---|---|---|
| Pedir una forma que no cuadra | `RuntimeError: shape '[5, 3]' is invalid for input of size 12` | El programa truena en la línea del `reshape` | Multiplica la forma que pediste y compárala con el tamaño que dice el error |
| Creer que `reshape` cambió el original | Tras `ventas.reshape(4, 3)`, `ventas.shape` sigue en `(12,)` | `ventas[3, 2]` truena | Guarda el resultado: `trimestres = ventas.reshape(4, 3)` |
| Usar dos `-1` | `RuntimeError: only one dimension can be inferred` | Truena aunque el total sí cuadraría | Deja un solo `-1` y escribe los demás tamaños |
| Usar `reshape` para girar una tabla | Números en renglones ajenos, sin ningún error | Pasa el `assert` de la forma y truena el de los valores | Para girar, `.T`; `reshape` solo corta en el orden de lectura |

---

## Repaso para la próxima sesión

La sesión 6 empieza con estas tres preguntas. Contéstalas sin ver la lección:

1. ¿Cuándo un `reshape` es válido?
2. ¿Qué significa el `-1` en `reshape`?
3. ¿Por qué hacer `reshape` de una tabla no es lo mismo que girarla?

🔁 **La próxima semana:** operas tensores completos, sin `for`. Vas a multiplicar precios por cantidades de todas las sucursales y todos los días de un jalón, y vas a conocer una trampa que, como la de hoy, no marca error. Mateo va a poner por fin dos números lado a lado: lo que se vendió y lo que se cobró.

Esa noche, antes de irse, le escribió al licenciado Ortega: *¿El banco mandó un técnico a cambiar la terminal de la sucursal Norte en agosto?*

La respuesta llegó a las 9:03 de la mañana siguiente. Una sola línea:

*«El banco no ha enviado personal técnico a su sucursal Norte en lo que va del año.»*

Mateo amplió la foto que Fer le había mandado a su tía en agosto. La terminal nueva, brillante. Y en la orilla, cortado por el encuadre, un hombre de espaldas.

**¿Quién era el hombre de la playera azul?**

---

## Soluciones (no las veas antes de intentarlo)

**Repaso para abrir.**

1. Queda `(2, 3)`: los días desaparecen porque se sumaron. Cada número es el total de la semana de una sucursal en un producto (la usaste hoy en el paso 9).
2. Porque `mean` solo acepta tensores de decimales (o complejos): con enteros truena siempre, aunque el promedio diera exacto. La razón de esa regla es la de la sesión 4: un promedio casi nunca da entero y PyTorch no corta los decimales a escondidas. Se convierte a decimales antes: `papeleria.float().mean(dim=1)`.
3. Calculando lo mismo con un `for`, que ya sabes que funciona, y comparando con un `assert`: primero la forma, después los valores.

**Ejercicio 18.**

| Forma | Multiplicada | Resultado |
|---|---|---|
| `(3, 4)` | 12 | `torch.Size([3, 4])` |
| `(6, 2)` | 12 | `torch.Size([6, 2])` |
| `(5, 2)` | 10 | `RuntimeError: shape '[5, 2]' is invalid for input of size 12` |
| `(2, -1)` | 2 × 6 | `torch.Size([2, 6])` |
| `(-1, 5)` | ningún número × 5 da 12 | `RuntimeError: shape '[-1, 5]' is invalid for input of size 12` |
| `(2, 3, 2)` | 12 | `torch.Size([2, 3, 2])` |
| `(12, 1)` | 12 | `torch.Size([12, 1])` |

La última es curiosa: 12 renglones de un solo número cada uno, una columna. A diferencia de `(12,)`, que tiene una sola dimensión, `(12, 1)` tiene dos. Parece inútil, pero la vas a encontrar seguido en ejemplos de IA.

**Ejercicio 19.** Los huecos son `-1` (o `14`) y `tolist`. Este bloque no es para copiar: muestra solo las dos líneas con huecos ya llenos, y los `...` son las líneas de tu archivo que no cambian:

```text
larga = papeleria.reshape(-1, 3)
...
        a_mano.append(papeleria[s, d].tolist())
```

Norte miércoles es el renglón 9: los primeros 7 renglones (0 a 6) son el Centro, Norte lunes es el 7, martes el 8 y miércoles el 9. `print(larga[9])` da:

```
tensor([350, 200, 100])
```

La cuenta es la misma del paso 3: sucursal × 7 + día = 1 × 7 + 2 = 9.

Parte c): los huecos son `flatten` y `reshape`:

```python
plana = papeleria.flatten()
de_vuelta = plana.reshape(2, 7, 3)
```

Salida: `Ejercicio 19c superado`.

**Ejercicio 20.**

a) La última línea es:

```
RuntimeError: shape '[7, 5]' is invalid for input of size 28
```

`RuntimeError` es un error al correr: el código estaba bien escrito, pero falló mientras corría. `shape '[7, 5]'` es la forma que pidió el compañero. `is invalid` dice que no sirve. `input of size 28` es lo que tiene: los días de febrero. 7 × 5 = 35, y 35 no es 28, así que 28 días no caben en 7 renglones de 5.

b) Cambia la última línea por `por_semana = febrero.reshape(4, 7)` (o `reshape(-1, 7)`: «de 7 en 7, tú calcula cuántas semanas»). El `for` es el del paso 3, con 4 y 7 en lugar de 4 y 3, sobre la lista `dias_febrero`:

```python
por_semana = febrero.reshape(4, 7)

a_mano = []
for s in range(4):
    total = 0
    for d in range(7):
        total = total + dias_febrero[s * 7 + d]
    a_mano.append(total)

assert por_semana.shape == (4, 7), "Deberían ser 4 semanas × 7 días"
assert por_semana.sum(dim=1).tolist() == a_mano, "Los totales por semana no cuadran con el for"
print(por_semana.sum(dim=1))
print("Ejercicio 20 superado")
```

Salida:

```
tensor([18500, 18600, 18400, 20000])
Ejercicio 20 superado
```

c) `reshape(7, 4)` corta en pedazos de 4 días. Su primer renglón es `tensor([2500, 3100, 2200, 2800])`: lunes a jueves de la primera semana. Otro revoltijo que no truena (el siguiente renglón empieza en el viernes). Que el total cuadre solo dice que el `reshape` es posible, no que sea el que querías.

**Ejercicio 21.**

```python
from papeleria import papeleria

semana = papeleria.sum(dim=1)        # sucursal × producto
girada = semana.T                    # producto × sucursal
assert girada.shape == (3, 2), "Deberían ser 3 productos × 2 sucursales"
assert girada[0].tolist() == [9000, 2350], "El renglón de cuadernos debería traer Centro y Norte"

for p in range(3):
    for s in range(2):
        assert girada[p, s] == semana[s, p], "Girar es cambiar renglones por columnas"
print("Ejercicio 21 superado")
```

Salida:

```
Ejercicio 21 superado
```

Con `semana.reshape(3, 2)` pasa el `assert` de la forma y truena el de cuadernos: el primer renglón trae `[9000, 5800]`. PyTorch no avisó porque para él no hay error: pediste 6 números en 3 × 2 y eso te dio. Si los datos tienen sentido, solo lo revisas tú, con un `assert` de los valores.

---

## Lo que sigue

*Fuentes de los datos técnicos: documentación oficial de PyTorch para [`reshape`](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html), [`flatten`](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html) y [`mean`](https://docs.pytorch.org/docs/stable/generated/torch.mean.html), [`Tensor.numel`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numel.html), [`torch.zeros`](https://docs.pytorch.org/docs/stable/generated/torch.zeros.html) y el atributo `Tensor.T` (en la [página de tensores](https://docs.pytorch.org/docs/stable/tensors.html)), y de TensorFlow para [`tf.reshape`](https://www.tensorflow.org/api_docs/python/tf/reshape).*

En la sesión 6 dejas de hacer cuentas número por número. Si quieres adelantar, escribe con un `for` cómo multiplicarías por 2 cada uno de los 42 números de la papelería. La semana que entra lo haces en una línea.

---

*Escribo desde lo que vemos todos los días en Hábil, donde construimos sistemas para banca, seguros y empresas en México. Si en tu trabajo o en el negocio de tu familia se topan con un problema así, platiquemos: [habil.mx](https://habil.mx)*
