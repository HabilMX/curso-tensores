# Tensores desde cero (8): tensores en el banco, la aseguradora y la tienda

*Cómo leer un caso real y decir qué forma tiene su tensor, por qué en la competencia de pronóstico de Walmart ganó un método que no era una red neuronal, y qué cuidados no son opcionales.*

*Tensores desde cero · lección B, sesión 8 de 8*

*Escenario: los personajes y la historia son inventados. Los datos técnicos, históricos y las cifras reales llevan su fuente.*

**Miércoles 23 de diciembre, 9:40 a.m. Sucursal del banco, segundo piso.**

El último día del plazo.

La sala de espera huele a café recalentado y a pino de plástico. Alguien puso un arbolito junto al dispensador de agua. Tiene tres esferas y una se está cayendo.

Doña Lupe trae su bolsa en las piernas, agarrada con las dos manos, como si alguien se la fuera a quitar. Mateo trae la laptop. Adentro está todo: la hoja de Norte, los totales, el cambio de agosto, los cobros sin mercancía y el detector de la semana pasada.

—Si nos dicen que no —dice doña Lupe—, cierro Norte el 2 de enero. Ya lo pensé.

—No nos van a decir que no.

—Eso decía tu tío del aguinaldo, y mira.

Mateo no contesta. Repasa en su cabeza lo que le preocupa desde hace nueve días: el detector acertaba 91 % de las veces. Y se le escapaban fraudes. ¿Qué banco le va a creer a un modelo hecho en una laptop, en la mesa de una papelería?

—¿Señora Herrera?

El licenciado Ortega es más joven de lo que sonaba por teléfono. Corbata roja, gafete colgando, prisa en los zapatos. Los pasa a una oficina con una ventana que da al estacionamiento.

—Tengo veinte minutos —dice—. Enséñeme lo que trae.

Mateo abre la laptop. Le muestra la caída de Norte en agosto. Los cobros de madrugada. Los montos que no corresponden a ninguna mercancía. El detector.

Ortega no dice nada durante un rato largo. Luego gira su propio monitor hacia ellos.

—¿Sabe qué es esto?

En la pantalla hay una lista. Cobros. Muchos. Algunos marcados en rojo.

—Es lo mismo que usted hizo —dice Ortega—. Solo que nosotros lo hacemos con millones de compras al día. Siéntese, joven. Le voy a enseñar cómo se ve esto en grande.

Mateo mira el reloj de la pared. Los veinte minutos acaban de empezar a correr.

## Repaso para abrir

Contesta sin ver la sesión 7. Las respuestas están en las Soluciones.

1. ¿Qué forma tiene el tensor de entrada del detector y qué es cada eje?
2. ¿Qué mide la pérdida y por qué tiene que bajar?
3. ¿Por qué acertar 91 % de las veces puede no ser suficiente para un banco?

> **El código de esta sesión** está en el repositorio público del curso, en la carpeta `sesion-08`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-08). Escríbelo tú primero, porque así se aprende; el repositorio es para comparar cuando algo no te salga igual. Cada programa viene con la salida que debe darte.

## Objetivos de esta sesión

| # | Al terminar puedes… | Lo demuestra |
|---|---|---|
| 1 | **Leer** un caso real y **escribir** la forma de su tensor, diciendo qué es cada eje | Ejercicio 30 |
| 2 | **Crear** con `torch.zeros` un tensor con la forma de un caso y **comprobarla** con `assert` | Ejercicios 30 y 31 |
| 3 | **Explicar** qué es un embedding y **predecir** la forma de su salida | Ejercicio 32 |
| 4 | **Convertir** una secuencia en tabla y **decir** qué se pierde | Ejercicio 33 |

**Qué necesitas:** tu entorno de PyTorch de la sesión 2 y las sesiones 5 a 7. Hoy no instalas nada.

---

## Por qué importa

Hasta aquí, los datos los inventaste tú, o salieron de la hoja de doña Lupe. Hoy vas a ver lo que vio Mateo en la pantalla de Ortega: casos reales de bancos, aseguradoras y tiendas.

Nadie te los va a dar con su forma escrita. Te llegan con una frase como «el modelo analiza el historial del cliente». Tu trabajo es traducirla a una forma: cuántos ejes, qué es cada uno y de qué tamaño. Si sabes hacerlo, ninguna noticia de IA te suena a magia. Suena a una caja con direcciones, como la hoja de la papelería.

Ortega se lo dijo a Mateo con otras palabras: «Aquí nadie se impresiona con la palabra *inteligencia*. Aquí se pregunta qué entra y qué sale».

**Dos advertencias.** Primera: casi ninguna empresa publica las formas exactas de sus tensores; publica qué técnica usa. Así que las formas de hoy dicen **cómo es ese tipo de dato en general**, no la receta secreta de cada empresa. Segunda: cuando una cifra la reporta la propia empresa, sin revisión de fuera, va con un **«según la empresa»**. Como tu primo que dice medir 1.80: puede ser cierto, pero nadie lo ha medido.

---

## Paso 1 — Las tres preguntas para leer un caso

—Antes de que le enseñe nada —dice Ortega—, dígame una cosa. Cuando usted ve un sistema de estos, ¿qué es lo primero que pregunta?

Mateo lo piensa. En noviembre habría dicho «qué tan bueno es». Ahora no.

—Qué forma tiene.

Ortega sonríe por primera vez en la mañana.

La habilidad central nueva de hoy es traducir un caso a ejes y forma, y cabe en tres preguntas, en este orden (además conocerás dos convenciones nuevas, el orden de los colores en PyTorch y los embeddings; hoy basta con reconocer sus formas):

1. **¿Qué es una cosa?** Una compra, un cliente, una foto. Casi siempre es el primer eje: el lote.
2. **¿Qué trae cada cosa adentro?** Una lista de datos agrega un eje. Una historia en el tiempo agrega otro antes de los datos. Una foto trae colores, alto y ancho.
3. **¿Qué número va en cada casilla?** Un monto, un color, el número de un producto.

Con las respuestas, «dibujas» la forma con `torch.zeros`, que crea un tensor de ceros con la forma que le pidas: no importan los valores, importa la caja. Crea `casos.py`:

```python
import torch

# Banca: un lote de compras, como el detector de la sesión 7
compras = torch.zeros(64, 4)
print("compras:", compras.shape)

# Banca: el historial de cada cliente
historial = torch.zeros(32, 30, 4)
print("historial:", historial.shape)

# Seguros: las fotos de un siniestro
fotos = torch.zeros(8, 3, 480, 640)
print("fotos:", fotos.shape)
print("números en las fotos:", fotos.numel())

assert compras.shape == (64, 4), "Un lote de compras es (compras, datos)"
assert historial.shape == (32, 30, 4), "El historial es (clientes, compras en el tiempo, datos)"
assert fotos.shape == (8, 3, 480, 640), "Un lote de fotos es (fotos, colores, alto, ancho)"
print("Las tres formas cuadran")
```

Salida:

```
compras: torch.Size([64, 4])
historial: torch.Size([32, 30, 4])
fotos: torch.Size([8, 3, 480, 640])
números en las fotos: 7372800
Las tres formas cuadran
```

`numel()` cuenta cuántos números caben en la caja: 8 × 3 × 480 × 640 = 7,372,800. Ocho fotos medianas ya son más de siete millones de números.

Doña Lupe se asoma a la laptop.

—¿Puros ceros?

—Es el molde, tía. Como cuando compras el marco antes que la foto.

—Ah. Pues el marco ya me costó siete millones.

---

## Paso 2 — El banco: cinco maneras de cazar un fraude

Ortega abre una carpeta en su computadora. Adentro hay recortes: comunicados, notas, artículos de blogs de empresas.

—Esto lo junto para las capacitaciones —dice—. Fíjese en que cada uno mira algo distinto.

Todos los bancos quieren lo mismo: detectar la compra rara sin molestar al cliente honesto. Lo que cambia es **qué miran**, y eso cambia la forma. Esto es lo que Ortega fue abriendo, uno por uno:

- **Mastercard.** Su comunicado de febrero de 2024 dice que Decision Intelligence ayuda a puntuar 143 mil millones de transacciones al año (la versión en español del mismo comunicado dice 125 mil millones: por eso se lee la fuente), y que la versión nueva evalúa **la relación entre varias entidades** alrededor de cada compra. Según la empresa, mejora la detección 20 % en promedio, y hasta 300 % en algunos casos.
- **Visa.** En 2019, según la empresa, analizaba el 100 % de 127 mil millones de transacciones al año, en más o menos un milisegundo cada una.
- **BBVA.** En una nota de 2018 sobre un trabajo con el MIT, BBVA dice que generaron **más de 200 variables nuevas por transacción** y que su modelo «puede reducir en un 54 % el nivel de falsos positivos» (alarmas que molestan a un cliente honesto). Es BBVA global, no BBVA México.
- **HSBC.** El primo del fraude: el lavado de dinero. En un texto de 2023 en el blog de Google Cloud, HSBC dice que revisa más de 1,200 millones de transacciones al mes buscando **movimientos rápidos entre cuentas y cambios bruscos de comportamiento**, y que encuentra de dos a cuatro veces más actividad sospechosa que antes, con 60 % menos alertas. Según la empresa.
- **Nubank.** En su blog de ingeniería cuenta que convierte las transacciones de cada cliente en **secuencias de fichas**, como si fueran palabras, y las lee con el tipo de modelo que se usa para texto. Cada transacción puede producir varias fichas (monto, fecha, texto de la descripción). La empresa reporta 131 millones de clientes en total a finales de 2025; eso no quiere decir que este modelo llegue a todos ellos.

Mateo nota algo en cada recorte: casi todos dicen «según la empresa». Ortega lo ve notarlo.

—Eso es lo que más me gusta de usted, joven. Que lee las letras chiquitas.

Ahora, las tres preguntas:

| Caso | ¿Qué es una cosa? | ¿Qué trae adentro? | Forma (tipo de dato) |
|---|---|---|---|
| Visa: una compra a la vez | una compra | sus datos | `(datos,)` |
| Mastercard, BBVA: muchas compras | una compra | sus datos (en BBVA, más de 200) | `(compras, datos)` |
| HSBC: comportamiento en el tiempo | una cuenta | sus movimientos en orden, cada uno con sus datos | `(cuentas, movimientos, datos)` |
| Nubank: historial como texto | un cliente | sus compras en orden, en fichas | modelo didáctico, con una ficha por compra: `(clientes, compras)` de enteros → `(clientes, compras, números)`; en el caso real, como una compra da varias fichas: `(clientes, fichas)` → `(clientes, fichas, números)` |

En BBVA, el eje de los datos pasó de 4 (tu detector) a más de 200: **la forma no cambió de idea, solo de tamaño.** En HSBC, mirar el tiempo agrega un eje: el paso de la tabla al cubo de la sesión 1, con una pestaña por cuenta.

Mateo se queda viendo la fila de BBVA. Su detector tenía cuatro datos por cobro: monto, hora, distancia y si el comercio era extranjero. El del banco tiene más de doscientos. Y sin embargo es la misma caja: `(compras, datos)`.

Su laptop no estaba jugando a otra cosa. Estaba jugando a lo mismo, en chiquito.

---

## Paso 3 — La aseguradora: el choque en fotos

—El banco no es el único que hace esto —dice Ortega, y abre otro recorte—. Mi cuñado trabaja en seguros. Allá ya no mandan a un ajustador a ver cada golpe. Le piden fotos.

En marzo de 2020, la aseguradora británica Ageas anunció que usaba el sistema de la empresa Tractable para evaluar choques de auto. Según Ageas, con fotos el sistema identifica qué piezas están dañadas y cuánto, arma el presupuesto completo y decide en minutos, a veces durante la primera llamada.

Las tres preguntas: una cosa es **una foto** (un siniestro trae varias: el lote); cada foto trae **3 colores**, y cada color es una tabla de **alto por ancho**; en cada casilla va cuánto de ese color tiene ese punto. Forma: `(fotos, 3, alto, ancho)`, el tensor `fotos` del paso 1.

Doña Lupe se acomoda los lentes.

—O sea que si choco, le tomo foto al coche y la computadora me dice cuánto.

—Algo así, tía.

—Pues que le tome foto a mi cuenta de la luz, a ver qué opina.

**La trampa del orden de los colores.** En la sesión 1 escribiste una foto como `(480, 640, 3)`: alto, ancho, color. En PyTorch la costumbre es poner el color **antes**: `(3, 480, 640)`. Mismos números, otro acomodo. Si mezclas las costumbres, el `assert` te avisa:

```python
fotos = torch.zeros(8, 3, 480, 640)
assert fotos.shape == (8, 480, 640, 3), "Un lote de fotos es (fotos, alto, ancho, colores)"
```

```
Traceback (most recent call last):
  ...
AssertionError: Un lote de fotos es (fotos, alto, ancho, colores)
```

Ninguna costumbre está mal; lo que está mal es suponer.

¿Y la salida? Por cada pieza, qué tan probable es «reparar» y qué tan probable es «reemplazar»: `(piezas, 2)`. Entran cuatro ejes, sale una tabla chiquita.

---

## Paso 4 — La tienda: la lección de humildad de Walmart

Mateo pregunta lo que lleva toda la mañana queriendo preguntar:

—Entonces, ¿todo esto lo hacen con redes neuronales? ¿Lo más avanzado gana siempre?

Ortega se ríe.

—Le voy a enseñar una que le va a gustar a su tía. Es de una tienda.

En 2020 se hizo la competencia de pronóstico M5. Estos datos salen del artículo académico de sus organizadores (Makridakis, Spiliotis y Assimakopoulos):

- Había que pronosticar **42,840 series jerárquicas** de ventas de **Walmart**: 10 tiendas en California, Texas y Wisconsin, datos diarios de unos 5.5 años. En el nivel más desagregado eran 30,490 series, cada una «cuánto vendió este producto en esta tienda, día por día»; las demás eran sumas por categoría, departamento, tienda o estado.
- Participaron **7,092 personas en 5,507 equipos de 101 países**.
- **El primer lugar usó LightGBM**, un método de árboles de decisión, no redes neuronales: un ensamble grande de 220 modelos, no algo trivial. El segundo lugar combinó LightGBM con una red (N-BEATS). El tercero sí usó redes: una combinación de 43 redes neuronales (el equipo usó DeepAR, que lee secuencias).

La lección de humildad: **las redes neuronales no ganan siempre.** Con datos de tabla, un método que no es una red puede ganarles. Y aun así, LightGBM recibe una tabla `(renglones, datos)`: **el tensor es la forma de acomodar los datos, no un tipo de modelo.**

| Quién | Cómo ve los datos | Forma |
|---|---|---|
| Una red que lee secuencias (tercer lugar) | cada serie es una historia | `(series, días, datos)` |
| LightGBM (ganador) | cada día de cada serie es un renglón | `(series × días, datos)` |

—¿Ve? —dice doña Lupe—. Lo sencillo también sirve. Llevo treinta años diciéndolo.

Pasar de una a otra es la sesión 5. Mateo lo hace ahí mismo, en la oficina, con la papelería de doña Lupe: 2 sucursales × 7 días se vuelven 14 renglones. Crea `tienda.py`:

```python
import torch

centro = [[1200, 800, 500], [1500, 900, 700], [1000, 700, 500], [1300, 900, 600],
          [2000, 1200, 700], [1200, 800, 400], [800, 500, 300]]
norte = [[300, 200, 100], [400, 250, 150], [350, 200, 100], [300, 150, 100],
         [500, 300, 200], [300, 200, 100], [200, 100, 100]]
papeleria = torch.tensor([centro, norte])
print("papeleria:", papeleria.shape)

tabla = papeleria.reshape(-1, 3)
print("tabla:", tabla.shape)

# Comprobación: con for, el total no cambia
total_for = 0
for sucursal in [centro, norte]:
    for dia in sucursal:
        for venta in dia:
            total_for = total_for + venta

assert tabla.shape == (14, 3), "La tabla debería ser 14 renglones (sucursal y día) × 3 productos"
assert tabla.sum().item() == total_for == 23100, "Cambiar de forma no cambia el total"
print("Total:", tabla.sum().item(), "| con for:", total_for)
```

Salida:

```
papeleria: torch.Size([2, 7, 3])
tabla: torch.Size([14, 3])
Total: 23100 | con for: 23100
```

Los renglones 0 a 6 son el Centro; los 7 a 13, el Norte. El total, 18,500 + 4,600 = 23,100, no cambió. ¿Qué se perdió? Que el renglón 7 sea «Norte, lunes» ya no lo dice la forma; en la práctica se agregan columnas como «sucursal» y «día».

Ortega mira los dos números: 18,500 y 4,600.

—Esa diferencia fue lo que nos hizo voltear a Norte —dice—. Nuestro sistema vio cobros raros en su terminal. Lo que no vio fue lo que usted me acaba de enseñar: que no había mercancía detrás.

---

## Paso 5 — La tienda que te recomienda: el diccionario que se aprende solo

Suena el celular de doña Lupe. Una notificación de una tienda en línea: «Quizá también te guste: juego de geometría».

—¿Y esta cómo sabe que vendo juegos de geometría?

Ortega y Mateo se miran. Esa pregunta tiene respuesta.

Detrás de «quizá también te guste…» muchas veces hay un **embedding**. **Facebook (hoy Meta)** publicó en 2019 un modelo de recomendación de código abierto en PyTorch (DLRM) donde «qué producto es» pasa por **tablas de embeddings**. **Alibaba** publicó ese año uno que lee la secuencia de lo que un usuario vio y compró; según la empresa, funciona en su tienda Taobao.

Piensa en un diccionario de Python: `{"cuadernos": [1.5, -0.3, -2.2], ...}`, una lista de números por producto. La diferencia es que **esos números no los escribe nadie: el modelo los aprende solo**, como aprendió los pesos del detector, y se ajustan con los datos y la pérdida. En una tarea de recomendación, productos con patrones de compra parecidos **pueden** terminar con números cercanos, pero no es una regla automática.

**Un embedding es un diccionario que se aprende solo.** En PyTorch la llave es un entero, así que primero numeras tu catálogo. Crea `embedding.py`:

```python
import torch

torch.manual_seed(0)

# El catálogo: cada producto tiene un número
catalogo = {"cuadernos": 0, "plumas": 1, "mochilas": 2, "colores": 3, "tijeras": 4}

# La tabla de embeddings: 5 productos, 3 números por producto
diccionario = torch.nn.Embedding(5, 3)
print("tabla completa:", diccionario.weight.shape)

# Una clienta compró cuadernos, mochilas y otra vez cuadernos
compra = torch.tensor([catalogo["cuadernos"], catalogo["mochilas"], catalogo["cuadernos"]])
print("entrada:", compra.shape, compra.dtype)

salida = diccionario(compra)
print("salida:", salida.shape)
print(salida)

assert salida.shape == (3, 3), "3 productos comprados × 3 números por producto"
assert torch.equal(salida[0], salida[2]), "El mismo producto da el mismo renglón"
assert torch.equal(salida[1], diccionario.weight[2]), "Buscar un producto es sacar su renglón"
print("Forma y valores cuadran")

# Un lote: 2 clientas, 4 compras cada una
historiales = torch.tensor([[0, 1, 1, 3],
                            [2, 4, 0, 0]])
print("historiales:", historiales.shape, "->", diccionario(historiales).shape)
```

Salida:

```
tabla completa: torch.Size([5, 3])
entrada: torch.Size([3]) torch.int64
salida: torch.Size([3, 3])
tensor([[ 1.5410, -0.2934, -2.1788],
        [ 0.4033,  0.8380, -0.7193],
        [ 1.5410, -0.2934, -2.1788]], grad_fn=<EmbeddingBackward0>)
Forma y valores cuadran
historiales: torch.Size([2, 4]) -> torch.Size([2, 4, 3])
```

`torch.equal(a, b)` responde `True` solo si los dos tensores tienen la misma forma y los mismos valores. Por eso los dos `assert` comprueban que buscar el mismo producto da el mismo renglón.

Léelo despacio:

- `Embedding(5, 3)` es una tabla de 5 renglones (uno por producto) y 3 columnas, al principio con números al azar.
- Buscar un producto es **sacar su renglón**: cuadernos aparece dos veces, idéntico.
- `grad_fn` es PyTorch avisando que lleva la cuenta para poder ajustar esos números al entrenar. Esa es la parte de «se aprende solo».
- **La regla de la forma:** la salida es la forma de la entrada **más un eje al final**, del tamaño del segundo número del Embedding. `(3,)` → `(3, 3)`; `(2, 4)` → `(2, 4, 3)`.

Esa última línea es el modelo didáctico de Nubank: con una ficha por compra, `(clientes, compras)` de enteros se vuelve `(clientes, compras, números)`. En el caso real una compra puede dar varias fichas, y el eje sería el de fichas. Tu historial de compras se lee como un texto.

—Entonces la tienda no sabe que vendo juegos de geometría —dice doña Lupe—. Sabe que la gente que compra lo que yo compro también compra eso.

—Exacto, tía.

—Pues es más chismosa que mi comadre.

Si pides un producto que no está en la tabla (el 5, cuando solo hay del 0 al 4), sale:

```
IndexError: index out of range in self
```

---

## Paso 6 — México y los cuidados que no son opcionales

**10:05 a.m.** Ortega mira el reloj. Se pasó cinco minutos de los veinte y no parece importarle.

—Ahora le voy a decir algo que no le dije por teléfono —dice—. El mensaje que le llegó a su tía el 1 de noviembre lo mandó una máquina. Un sistema vio cobros raros y suspendió la terminal. Solo.

Doña Lupe aprieta la bolsa.

—¿Y por qué me dieron hasta hoy?

—Porque una máquina puede marcar. Pero una decisión así no se debe quedar en la máquina. Alguien tiene que revisar. Hoy ese alguien soy yo.

Mateo entiende de golpe por qué está sentado ahí. No vino a convencer a un modelo. Vino a que una persona revisara lo que el modelo decidió.

Lo que sigue es por qué ese «alguien tiene que revisar» no es cortesía del banco.

**Banxico.** El Banco de México publicó el 10 de junio de 2026 el recuadro «Inteligencia artificial y estabilidad financiera: canales potenciales de transmisión de riesgos», donde advierte de la concentración en **pocos proveedores**, de modelos **opacos** y difíciles de supervisar, y de la posible **exclusión no justificada de ciertos grupos de población**. Lo comenta también una nota de Expansión del 18 de junio de 2026.

**El sesgo: German Credit.** Es una base pública muy usada para practicar crédito, del repositorio de la Universidad de California en Irvine: 1,000 personas con 20 datos cada una, de 1994. Como tabla original, `X` tiene 1,000 personas y 20 atributos mixtos (números y categorías); antes de usarla en un tensor numérico hay que codificar las categorías, y entonces el eje de datos puede dejar de medir 20. El problema está en las columnas: una mezcla **estado civil y sexo**, otra es **la edad** y otra dice si la persona es **«trabajador extranjero»**. Entrenar con eso tal cual es pedirle al modelo que preste o no según el sexo o la nacionalidad. Por eso el curso no la usó en la sesión 7.

¿Y si borras esas columnas? No basta: otras pueden **delatar** el mismo dato. Es como quitarle el nombre a un examen y dejarle la letra: el maestro igual sabe de quién es. Hay que medir cómo trata el modelo a cada grupo.

**La ley europea.** El *AI Act* de la Unión Europea, Anexo III, punto 5(b), pone como **alto riesgo** los sistemas que evalúan si una persona es buena para un crédito o le ponen calificación de crédito, **excepto los que detectan fraude financiero**. Tu detector quedaría fuera; un modelo que califique crédito, como los que se practican con German Credit, dentro. Para pensar: ¿por qué la ley trata distinto al fraude que al crédito? Pista: si el detector se equivoca, te bloquean una compra; si se equivoca el de crédito, te quedas sin la casa.

**La ley mexicana.** La nueva Ley Federal de Protección de Datos Personales en Posesión de los Particulares se publicó el 20 de marzo de 2025. Su **artículo 26, fracción II**, te permite **oponerte** a que tus datos pasen por un **tratamiento automatizado** que te afecte de forma importante y que te evalúe **sin intervención humana**: tu situación económica, tu salud, tu fiabilidad o tu comportamiento, entre otras cosas. Traducido: si un modelo decide solo si te dan un crédito, en esos supuestos puedes oponerte. No quiere decir que la ley ordene, por sí sola, una revisión humana en cada decisión, y el derecho tiene excepciones: no procede, por ejemplo, cuando el tratamiento es necesario para cumplir una obligación legal. Para quien diseña, «el modelo decide y ya» no basta: conviene un humano en el camino.

Ojo con lo que dice y lo que no dice ese artículo: habla de **tus datos personales**, de decisiones sobre **ti**. La terminal de una papelería es otra historia, y en el relato el banco la revisa a mano por su propia regla. Pero la idea es la misma, y Mateo la vio en la oficina de Ortega: **la máquina marca; una persona revisa; y a quien le afecta tiene por dónde reclamar.**

> **10:20 a.m. La oficina del licenciado Ortega.**
>
> Ortega teclea algo. Imprime una hoja. La firma.
>
> —La terminal de Norte queda reactivada a partir de hoy —dice—. Pero antes va a ir un técnico. Uno de verdad, con gafete y orden de servicio. Revisa el aparato.
>
> Doña Lupe se queda callada. Por primera vez en siete semanas no tiene un dicho a la mano.
>
> **1:15 p.m. Sucursal Norte.**
>
> El técnico del banco abre la terminal sobre el mostrador. Fer está a un lado, de brazos cruzados. Mateo, atrás.
>
> Dentro de la ranura donde entra la tarjeta hay algo que no debería estar ahí: una laminita delgada, con un cable del grueso de un cabello que corre hacia adentro del aparato.
>
> —Un lector —dice el técnico—. Copia la tarjeta cada vez que alguien paga. La gente ni se entera.
>
> Fer suelta el aire despacio.
>
> —Lo puso el de la playera azul. En agosto.
>
> Nadie lo había llamado. Nadie le preguntó su nombre. Pagaba la gente en Norte, y sus tarjetas se usaban de madrugada, lejos, en otro lado. Por eso el banco solo veía la terminal.
>
> El técnico mete la laminita en una bolsa de plástico. Mateo piensa en la tabla de cobros de la sesión 6: los que no tenían mercancía detrás. Ahí estaba la mano que faltaba.

---

👉 **Si eres de Administración:** cuando un proveedor te presente su sistema de IA, hazle las tres preguntas del paso 1; si no sabe contestarlas, no entiende su producto. Pregunta de dónde sale cada cifra. Y si el sistema decide sobre personas (crédito, seguros, contratación), pregunta dónde está el humano que revisa (conviene aunque la ley no lo exija en todos los casos).

---

## Ejercicios de la sesión 8

Los últimos del curso. Las soluciones están al final.

**Ejercicio 30 — Tres casos, tres formas.** Contesta las tres preguntas en papel, escribe la forma, dibújala con `torch.zeros` y compruébala con `assert`. El inciso a) va resuelto.

a) Un lote de 500 compras con 200 datos cada una, como en la nota de BBVA:

```python
bbva = torch.zeros(500, 200)
assert bbva.shape == (500, 200), "a) compras × datos"
```

b) El historial de 16 clientes en secuencias de 100 compras, cada compra pasada por un Embedding de 64 números, como en el modelo didáctico de Nubank. Pista: `torch.zeros(16, 100, dtype=torch.long)` y `torch.nn.Embedding(1000, 64)`.
c) Un siniestro con 5 fotos de 720 de alto por 1,280 de ancho, en el orden de PyTorch.

Termina con `print("Ejercicio 30 superado")`.

**Ejercicio 31 — El lote de fotos.** Llena los huecos: un siniestro con 6 fotos de 224 × 224, y su cantidad de números calculada de dos maneras que deben coincidir:

```python
siniestro = ____
assert siniestro.shape == ____, "6 fotos × 3 colores × 224 de alto × 224 de ancho"
cuenta = 1
for tamano in (6, 3, 224, 224):
    cuenta = ____
assert siniestro.numel() == cuenta, "numel debe dar lo mismo que multiplicar la forma con un for"
print("Ejercicio 31 superado:", siniestro.numel(), "números")
```

**Ejercicio 32 — El diccionario de doña Lupe.** Su tienda en línea tiene 4 productos y quiere 2 números por producto. Predice las formas **antes** de correr:

```python
torch.manual_seed(1)
catalogo = {"cuadernos": 0, "plumas": 1, "mochilas": 2, "colores": 3}
diccionario = torch.nn.Embedding(____, ____)
historiales = torch.tensor([[0, 1, 1, 2, 3],
                            [3, 3, 0, 1, 2],
                            [2, 2, 2, 0, 1]])
salida = diccionario(historiales)
assert diccionario.weight.shape == ____, "4 productos × 2 números"
assert historiales.shape == ____, "3 clientas × 5 compras"
assert salida.shape == ____, "3 clientas × 5 compras × 2 números"
assert torch.equal(salida[0][3], diccionario.weight[catalogo["mochilas"]]), "La cuarta compra de la primera clienta son mochilas"
print("Ejercicio 32 superado")
```

**Ejercicio 33 — De secuencia a tabla, tú solo.** Una M5 chiquita: 100 series, 28 días, 3 datos por día (precio, promoción, festivo). a) Crea con `torch.zeros` lo que vería una red que lee secuencias. b) Conviértelo con `reshape` y `-1` en la tabla que vería LightGBM. c) Escribe tres `assert`: las dos formas y que tengan la misma cantidad de números. d) En un comentario, di qué es un renglón de la tabla y qué ya no dice la forma.

---

## Errores comunes (y cómo darte cuenta)

| Error | Cómo se ve | Cómo darte cuenta | Qué hacer |
|---|---|---|---|
| Mezclar el orden de los colores | Escribes `(fotos, alto, ancho, 3)` y PyTorch espera `(fotos, 3, alto, ancho)` | Truena el `assert` de la forma | Pregunta en qué orden vienen las fotos |
| Pedir un producto fuera de la tabla | `IndexError: index out of range in self` | Algún número es ≥ al primer tamaño del Embedding | N productos van de 0 a N − 1 |
| Darle decimales a un Embedding | `RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)` | Dice `FloatTensor` donde esperaba enteros | `torch.tensor([0, 2])`, sin punto |
| Repetir una cifra de la empresa como hecho | «Mastercard mejora 300 % la detección» | La fuente es un comunicado de la empresa | «Según la empresa», y di si es «hasta» o «en promedio» |

---

## Autoevaluación de la lección B

Marca con honestidad. Si algo te sale «todavía no», regresa a esa sesión.

| Puedo… | Sesión | Sí / Todavía no |
|---|---|---|
| Cambiar de forma con `reshape`, usar `-1` y leer el error cuando no cuadra | 5 | |
| Predecir la forma que sale al operar tensores de formas distintas | 6 | |
| Distinguir `*` de `@` y saber cuál usar | 6 | |
| Explicar pérdida, gradiente, optimizador y época | 7 | |
| Explicar por qué la exactitud sola engaña, y qué miden precisión y recall | 7 | |
| Leer un caso real y escribir la forma de su tensor | 8 | |

## Repaso de la lección B

Contesta sin ver. Las de código, predice y después córrelas.

1. Si `meses = torch.arange(12)`, ¿qué forma da `meses.reshape(4, -1)`? ¿Y `meses.reshape(-1, 6)`?
2. ¿Qué pasa con `meses.reshape(5, -1)`, y por qué?
3. Con la papelería `(2, 7, 3)` y `precios = torch.tensor([45.0, 12.0, 350.0])`, ¿qué forma tiene `papeleria * precios`?
4. ¿Funciona `torch.zeros(2, 7, 3) * torch.zeros(7)`?
5. Con `a = [[1, 2], [3, 4]]` y `b = [[10, 20], [30, 40]]` como tensores, ¿cuánto dan `a * b` y `a @ b`? ¿Por qué `*` es una trampa que no marca error?
6. Si `tabla` es `(14, 3)` y `precios` es `(3,)`, ¿qué forma tiene `tabla @ precios` y qué es cada número?
7. ¿Qué hacen `perdida.backward()` y `optimizador.step()`?
8. ¿Qué es una época?
9. De 1,000 compras, 50 son fraude. El modelo lanza 40 alertas y 30 son fraude de verdad. Calcula exactitud, precisión y recall. ¿Cuál le preocupa más al banco?
10. Escribe la forma de: (a) 64 compras con 4 datos; (b) 32 clientes con sus últimas 30 compras de 4 datos; (c) 8 fotos de 480 × 640 en el orden de PyTorch.

---

🔁 **Para cerrar el curso**

**Miércoles 23 de diciembre, 7:30 p.m. Papelería, sucursal Centro.**

Doña Lupe llega con una caja de pastelería amarrada con listón. Tres leches. La pone sobre el mostrador, entre las engrapadoras y los rollos de papel lustre.

—Este lo pagué con tarjeta —dice—. En la terminal de Norte. A ver si ahora sí me cobran lo que es.

Fer corta el pastel. Le toca el pedazo más grande, y nadie dice nada, porque todos saben por qué: fue ella la que en septiembre escribió que faltaba mercancía, y nadie leyó su correo.

Mateo come de pie, recargado en la vitrina. Afuera alguien baja una cortina metálica. Huele a papel nuevo y a canela.

—¿Y tú qué vas a hacer ahora? —le pregunta doña Lupe—. Ya no hay nada que salvar.

Mateo lo ha pensado toda la semana. En noviembre quería terminar la carrera y ya. Ahora sabe lo que le gusta: que una pila de números diga quién miente. Quiere estudiar esto. Datos. Modelos. Y las letras chiquitas.

—Esto, tía. Lo que hicimos.

—¿Cuidar papelerías?

—Algo así.

Esa noche abre la laptop. Una hoja nueva, en blanco. Ninguna forma todavía, ningún eje.

Escribe una sola línea:

`# ¿Y ahora qué?`

Tú estás igual que él. En ocho sesiones fuiste de un número suelto a leer cómo trabajan un banco, una aseguradora y una tienda. No te queda una fórmula: te queda una manera de mirar.

| Sesión | Lo que aprendiste |
|---|---|
| 1 | Qué es una dimensión y qué es la forma, con listas de Python |
| 2 | Tu primer tensor: `torch.tensor`, `shape` y `dtype` |
| 3 | Sacar un pedazo del tensor por cualquiera de sus ejes |
| 4 | Sumar y promediar por eje, comparado contra un `for` |
| 5 | Cambiar de forma con `reshape` sin cambiar los datos |
| 6 | Operar tensores completos, y la diferencia entre `*` y `@` |
| 7 | Entrenar un detector de fraude, y desconfiar de la exactitud |
| 8 | Leer casos reales, embeddings y los cuidados del crédito automático |

---

## Soluciones (no las veas antes de intentarlo)

### Repaso para abrir

1. `(compras, datos de cada compra)`: en la sesión 7, `(4000, 4)`.
2. Mide **qué tan mal predice** el modelo, en un solo número; si baja, el modelo está aprendiendo.
3. Porque casi todas las compras son buenas: un «detector» que nunca alarma ya acierta alrededor de 84 %. Con 91 %, el nuestro todavía dejaba pasar 1 de cada 3 fraudes.

### Ejercicios

**30.** b) `(16, 100, 64)`: clientes × compras × números del embedding. El 1000 es cuántos productos distintos caben en la tabla; no cambia la forma de la salida. c) `(5, 3, 720, 1280)`.

```python
historiales = torch.zeros(16, 100, dtype=torch.long)
embedding = torch.nn.Embedding(1000, 64)
salida = embedding(historiales)

assert historiales.shape == (16, 100), "clientes × compras"
assert salida.shape == (16, 100, 64), "clientes × compras × números"

fotos_siniestro = torch.zeros(5, 3, 720, 1280)
assert fotos_siniestro.shape == (5, 3, 720, 1280), "fotos × colores × alto × ancho"

print("Ejercicio 30 superado")
```

Salida: `Ejercicio 30 superado`.

**31.** Los huecos son `torch.zeros(6, 3, 224, 224)`, `(6, 3, 224, 224)` y `cuenta * tamano`. Salida: `Ejercicio 31 superado: 903168 números`.

**32.** `Embedding(4, 2)`, `(4, 2)`, `(3, 5)` y `(3, 5, 2)`: a la forma de la entrada se le agrega al final un eje de tamaño 2. Salida: `Ejercicio 32 superado`.

**33.**

```python
series = torch.zeros(100, 28, 3)    # series × días × datos
tabla = series.reshape(-1, 3)       # (serie y día) × datos
assert series.shape == (100, 28, 3), "series × días × datos"
assert tabla.shape == (2800, 3), "(series y día) × datos"
assert tabla.numel() == series.numel() == 8400, "Cambiar de forma no cambia cuántos números hay"
print("Ejercicio 33 superado")
# Un renglón es un día de una serie; la forma ya no dice de cuál serie ni de cuál día.
```

### Repaso de la lección B

Salidas reales de las preguntas 1 a 6:

```
1) torch.Size([4, 3]) torch.Size([2, 6])
2) shape '[5, -1]' is invalid for input of size 12
3) torch.Size([2, 7, 3])
4) The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 2
5) a * b = [[10, 40], [90, 160]] | a @ b = [[70, 100], [150, 220]]
6) torch.Size([14])
```

1. `(4, 3)` y `(2, 6)`: el `-1` es «calcula tú».
2. Truena: 12 números no se reparten en 5 renglones iguales.
3. `(2, 7, 3)`: los precios se estiran a todas las sucursales y días.
4. No: PyTorch compara las formas de derecha a izquierda, y el último eje mide 3 contra 7.
5. `*` es casilla por casilla; `@`, renglón por columna. Con estas formas las dos funcionan: usar la equivocada no truena, solo da otro número. Por eso se compara contra un `for`.
6. `(14,)`: el total en pesos de cada sucursal en cada día (cantidades por precios, sumadas).
7. `perdida.backward()` calcula los gradientes de la pérdida respecto de cada peso; `optimizador.step()` usa esos gradientes para mover los pesos un pasito y buscar que baje la pérdida.
8. Una pasada completa por todos los datos de entrenamiento.
9. Compras buenas sin alerta: 1,000 − 30 − 10 − 20 = 940. Exactitud = 970 ÷ 1,000 = 0.97; precisión = 30 ÷ 40 = 0.75; recall = 30 ÷ 50 = 0.6. El 97 % suena excelente, pero se cuelan 4 de cada 10 fraudes.
10. `(64, 4)`, `(32, 30, 4)` y `(8, 3, 480, 640)`: los tensores del paso 1.

---

## Lo que sigue

Un orden posible para seguir (en inglés, salvo lo último):

- **Para la intuición:** la serie *Neural Networks* de **3Blue1Brown**, videos animados que muestran qué hace una red por dentro. Empieza por «But what is a neural network?». Sirven para entender, no para practicar.
- **Para construir una red desde un solo número:** *Neural Networks: Zero to Hero* de **Andrej Karpathy**. Su primer video, *micrograd* (2 horas 25 minutos), construye desde cero lo que hace `perdida.backward()`; él dice que basta Python básico y un vago recuerdo del cálculo de prepa. Va a ritmo de experto: ve con pausas.
- **Para redes más grandes en PyTorch:** *Learn PyTorch for Deep Learning*, de Daniel Bourke (learnpytorch.io), con ejercicios y soluciones para comparar.
- **En español:** los videos de **DotCSV** para motivarte y, como meta, el curso de posgrado *Aprendizaje Profundo* de la UNAM, con su material abierto en GitHub. Si un día entiendes sus libretas, este curso cumplió.

Un último consejo: cuando algo no funcione, imprime `.shape`. Tú ya sabes leerla.

---

*Escribo desde lo que vemos todos los días en Hábil, donde construimos sistemas para banca, seguros y empresas en México. Si en tu trabajo o en el negocio de tu familia se topan con un problema así, platiquemos: [habil.mx](https://habil.mx)*
