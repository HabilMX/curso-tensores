# Tensores desde cero (7): un detector de fraude que aprende solo

*Le das 4,000 compras con tarjeta y la respuesta de cada una, pero nunca le escribes la regla. En menos de un segundo aprende buena parte de ella. Y luego aprendes por qué acertar 91 % de las veces puede esconder que se le escapa más de 1 de cada 3 fraudes.*

*Tensores desde cero · lección B, sesión 7 de 8*

*Escenario: los personajes y la historia son inventados. Los datos técnicos, históricos y las cifras reales llevan su fuente.*

**Lunes 14 de diciembre, 8:18 p.m. Papelería Herrera, sucursal Centro.**

La cortina metálica ya está abajo y todavía tiembla un poco, como si no quisiera quedarse quieta. El foco de la trastienda zumba. Huele a papel nuevo, a café recalentado y a la canela de un pan que doña Lupe dejó olvidado junto a la caja. Afuera pasa un camión y vibra el vidrio de la puerta.

Mateo tiene la laptop abierta sobre una caja de cartulinas. A un lado, dos hojas impresas. En una, las ventas que sí salieron de Norte: cuadernos, plumas, alguna mochila. En la otra, los cobros que mandó el banco. Algunos llegaron de madrugada. Otros son altos. Otros aparecen muy lejos de la sucursal, en lugares donde la papelería no ha vendido ni un clip.

—Este de las tres de la mañana no puede ser una mochila —dice Mateo, y lo encierra con pluma roja.

—A esa hora yo solo vendo sueño —contesta doña Lupe—. Y barato, porque nadie lo compra.

Hace una semana multiplicaron piezas por precios y la cuenta no cerró: hay cobros sin mercancía detrás. Fer no los inventó; ella había avisado de los faltantes en septiembre y nadie leyó su correo. El problema no estaba en el inventario. Estaba en la terminal que cambió aquel hombre de playera azul. Un lector escondido pudo copiar tarjetas en Norte. Después, alguien las usó en otros lados.

Pero una sospecha no alcanza para el banco. El licenciado Ortega lo dijo por teléfono, sin rodeos: «Señora, necesito saber cuáles cobros sí son suyos. No una corazonada. Cuáles».

Y ahí está el nudo. Una compra cara no es un fraude: hay quien junta todo el año para su pantalla. Una compra de madrugada tampoco: hay farmacias abiertas y gente con urgencias. Un cobro lejano puede ser alguien de vacaciones. Ninguna pista sola condena a nadie. Lo que pesa es cuando se juntan.

Mateo intenta revisar la hoja del banco renglón por renglón y se rinde en el cuarenta. Así no va a acabar ni en enero.

En el calendario de la pared, el que regaló la tlapalería de enfrente, alguien encerró en rojo el 23. Faltan nueve días.

—No voy a escribir una regla para cada caso, tía —dice Mateo—. Le voy a dar miles de ejemplos a la computadora: cuánto costó, a qué hora, qué tan lejos y si el comercio era de otro país. Que ella encuentre el patrón.

—¿Y si la computadora se equivoca?

Mateo se queda mirando el círculo rojo del calendario.

—Entonces nos va a decir por dónde seguir buscando. Pero no va a decidir sola.

En la sesión anterior operaste tensores completos de un jalón: piezas por precio de toda una semana de un jalón, con broadcasting y con `@`. Hoy esas dos herramientas se ponen a trabajar solas: vas a ver a un programa **aprender**.

Es la sesión más difícil del curso, así que vamos despacio. Si algo no te queda claro a la primera, es normal: léelo otra vez con el código corriendo enfrente.

## Repaso para abrir

Contesta sin ver la sesión 6. Las respuestas están en las soluciones, al final.

1. ¿Cuándo dos formas se pueden «estirar» para operar juntas?
2. ¿Qué diferencia hay entre `*` y `@`?
3. ¿Qué forma sale de una tabla `(7, 3) @` un vector `(3,)`?

> **El código de esta sesión** está en el repositorio público del curso, en la carpeta `sesion-07`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-07). Escríbelo tú primero, porque así se aprende; el repositorio es para comparar cuando algo no te salga igual. Cada programa viene con la salida que debe darte.

## Objetivos de esta sesión

| # | Al terminar puedes… | Lo demuestra |
|---|---|---|
| 1 | **Nombrar** en orden los cinco pasos del ciclo de entrenamiento (predecir, medir la pérdida, borrar las pendientes viejas, calcular la pendiente, girar las perillas) y **decir** qué línea de código hace cada uno | Ejercicio 26 |
| 2 | **Predecir** qué le pasa a la pérdida si cambias el número de épocas o la tasa de aprendizaje, y **comprobarlo** corriendo el modelo | Ejercicios 26 y 27 |
| 3 | **Calcular** con tensores la exactitud, el recall y la precisión de un detector, y **explicar** por qué la exactitud sola engaña | Ejercicio 28 |
| 4 | **Relacionar** lo que aprendió el modelo con la regla escondida que generó los datos | Ejercicio 29 |

**Qué necesitas:** el mismo entorno de la sesión 2 (Python 3 con PyTorch en tu venv) y todo lo de las sesiones 1 a 6: crear tensores, `shape`, `dtype`, indexar, reducir con `sum` y `mean`, `reshape`, broadcasting y `@`. Hoy no instalas nada.

También necesitas reconocer tres piezas de Python: `for` repite las líneas indentadas; `def nombre(X):` nombra una receta reutilizable; y `with ...:` agrupa líneas que se ejecutan bajo una misma regla. No tienes que dominarlas todavía: copia los bloques completos y cambia solo lo que señale cada ejercicio.

---

## Por qué importa

Hasta hoy tú escribías la regla: suma este eje, multiplica estas dos tablas. Mateo tiene otro tipo de problema. Ve miles de cobros y sabe cuáles terminaron siendo fraude, pero no tiene una frase corta que explique todos los casos.

Cada cobro trae pistas. Monto. Hora. Distancia. Comercio extranjero. Una sola pista no condena a nadie. Varias juntas pueden pedir una revisión. El trabajo del programa es encontrar esa combinación en ejemplos viejos, no adivinar la historia de una persona.

Piensa en cómo aprendiste a saber si un tianguis es bueno. Nadie te dio un manual. Fuiste muchas veces. Viste dónde había fila, qué puesto daba la fruta madura y cuál la escondía abajo. Con el tiempo, sin poder explicarlo en una frase, ya sabías a qué puesto no acercarte. Eso fue aprender de ejemplos.

O piensa en tu abuela con los frijoles. No mide nada. Prueba, le falta sal, le echa una pizca, vuelve a probar. Nunca le pasaron la fórmula exacta: la encontró a fuerza de corregir. La computadora va a hacer exactamente eso, pero con números y miles de veces por segundo.

Entonces se le da la vuelta: **en lugar de escribir la regla, le das ejemplos al programa y dejas que busque una regla útil.** Eso es entrenar un modelo. Está detrás de mucho de lo que hoy llamamos inteligencia artificial.

Lo bueno es que no aparece magia nueva. Entrenar es **pura cuenta con tensores**. Los datos son un tensor. Las perillas que aprende el modelo son tensores. La predicción usa `@`. Y al final, Mateo todavía necesita comparar, preguntar y reunir pruebas para que alguien revise el caso.

> **En la libreta de Mateo:** un modelo no prueba que un cobro fue fraude. Solo ordena miles de pistas y señala cuáles vale la pena revisar primero.

---

## Paso 1 — La idea en una frase

Mateo empieza pequeño. Si entiende una perilla, puede entender cuarenta y nueve. Si la pérdida baja en el ejemplo chico, ya sabe qué buscar cuando entrenen las compras.

Entrenar un modelo es esto, repetido muchas veces:

1. **Predecir:** el modelo adivina.
2. **Medir:** comparas su adivinanza con la respuesta correcta y sacas un número que dice qué tan mal le fue.
3. **Borrar las pendientes viejas:** limpias la cuenta de la vuelta anterior (en el Paso 2 verás por qué hace falta).
4. **Calcular la pendiente:** averiguas hacia dónde mover cada perilla para que le vaya un poquito menos mal.
5. **Girar las perillas:** las mueves un pasito en esa dirección.

Y vuelta a empezar: el `for` repite esos cinco pasos.

Como los tiros libres en la cancha: tiras, ves que te fuiste a la izquierda, corriges un poco a la derecha y vuelves a tirar. Nadie te dio la fórmula del tiro perfecto: la encontraste a puro corregir.

Antes de ver el detector de fraude, que tiene 49 perillas, vamos a hacer el ciclo completo con **una sola**.

> **8:24 p.m.** —¿Cinco pasos y ya? —Mateo frunce la boca—. Suena demasiado fácil. Si fuera así, cualquiera haría inteligencia artificial.
>
> —Cualquiera hace tortillas, mijo —dice doña Lupe sin levantar la vista de las monedas—. Hacerlas bien es otra cosa. La receta es corta; lo largo es la práctica.
>
> Mateo se ríe, pero se queda pensando. Tiene razón. La receta del ciclo cabe en una servilleta. Lo difícil será confiar en ella cuando las perillas no sean una, sino cuarenta y nueve, y no pueda ver con sus propios ojos para dónde se mueven. Por eso empieza con una. Si la ve funcionar en chiquito, le va a creer en grande.

## Paso 2 — Una sola perilla: el precio de un cuaderno

Doña Lupe perdió la lista de precios. Lo único que recuerda es que un cliente se llevó 3 cuadernos y pagó $60. ¿Cuánto cuesta un cuaderno?

Tú sacas 20 de cabeza. Pero vamos a dejar que PyTorch lo descubra «a ciegas», con el ciclo de 5 pasos. Crea `cerro.py`:

```python
# cerro.py — una sola perilla: el precio de un cuaderno
import torch

precio = torch.tensor(0.0, requires_grad=True)   # empieza en $0: ni idea
tasa = 0.05                                      # qué tan largo es cada paso

for paso in range(1, 6):
    cobro = precio * 3                    # predicción: lo que cobraría por 3 cuadernos
    perdida = (cobro - 60) ** 2           # qué tan mal: la diferencia contra $60, al cuadrado
    perdida.backward()                    # PyTorch calcula la pendiente
    a_mano = 18 * precio.item() - 360     # la misma pendiente, sacada con lápiz
    assert abs(precio.grad.item() - a_mano) < 0.001, "PyTorch y el lápiz deberían coincidir"
    print(f"paso {paso}: precio {precio.item():6.2f}  pérdida {perdida.item():8.2f}  pendiente {precio.grad.item():8.2f}")
    with torch.no_grad():
        precio -= tasa * precio.grad      # un paso cuesta abajo
    precio.grad.zero_()                   # borrar la pendiente para el siguiente paso

assert abs(precio.item() - 20) < 0.1, "Tres cuadernos de $20 dan $60"
print("precio aprendido:", round(precio.item(), 2))
```

Córrelo con `python3 cerro.py`:

```
paso 1: precio   0.00  pérdida  3600.00  pendiente  -360.00
paso 2: precio  18.00  pérdida    36.00  pendiente   -36.00
paso 3: precio  19.80  pérdida     0.36  pendiente    -3.60
paso 4: precio  19.98  pérdida     0.00  pendiente    -0.36
paso 5: precio  20.00  pérdida     0.00  pendiente    -0.04
precio aprendido: 20.0
```

En cinco pasos llegó a $20 sin que nadie le dijera la respuesta. Vamos línea por línea, porque aquí está toda la sesión en chiquito.

**La perilla.** `precio` es el número que el modelo puede mover. Empieza en 0. El `requires_grad=True` le dice a PyTorch: «anota todo lo que le hagas a este número, porque luego te voy a pedir su pendiente». Es prender la grabadora antes de la junta.

¿Por qué «perilla»? Piensa en la estufa. La perilla no sabe cocinar. Solo sube o baja la flama. Quien decide cuánto girarla es quien prueba la comida. Aquí pasa igual: `precio` no sabe nada de cuadernos. Es un número que se puede mover un poquito para un lado o para el otro. El que decide cuánto moverlo es el ciclo.

**La predicción.** `cobro = precio * 3`: con el precio que cree ahora, ¿cuánto cobraría por 3 cuadernos? En el paso 1, $0.

**La pérdida.** `(cobro - 60) ** 2`: la diferencia entre lo que predijo y lo que pasó de verdad, al cuadrado. El cuadrado hace que la pérdida nunca sea negativa (pasarse por $10 o quedarse corto por $10 cuesta igual) y que los errores grandes pesen mucho más. En el paso 1 vale 3,600: le fue pésimo. **La pérdida es una calificación al revés: entre más baja, mejor.** Cero es perfecto.

Es como el golf, donde gana el que hace menos golpes. O como las multas del reglamento: entre menos, mejor te portaste. La pérdida no te dice *qué* hiciste mal. Solo te dice *cuánto*. Con eso basta para empezar a corregir.

**La pendiente.** Imagínate parado en un cerro **con los ojos vendados**, queriendo llegar al fondo del valle. No ves nada, así que tanteas el piso con el pie: si hacia adelante baja, das un paso hacia adelante; si sube, te regresas. Eso es la **pendiente** (en los libros, **gradiente**): para qué lado baja el piso y qué tan empinado está. El cerro es la pérdida, donde estás parado es el valor de la perilla, y el fondo es la pérdida más baja posible.

Hazlo de verdad, aunque sea con la imaginación. Es de noche, se fue la luz en toda la colonia y tienes que bajar la calle empinada de tu casa sin lámpara. No ves la esquina. Pero tus pies sienten la inclinación. Donde se siente más empinado, das un paso más largo. Donde el piso ya casi está plano, das pasitos cortos, porque ya llegaste o casi. Nunca viste el mapa de la calle. Te bastó sentir el piso, paso por paso. Eso hace la computadora: no ve la forma completa del cerro, solo siente la pendiente donde está parada.

`perdida.backward()` es tantear el piso. PyTorch revisa lo que anotó la grabadora (multiplicaste por 3, restaste 60, elevaste al cuadrado) y calcula la pendiente solito. A eso se le llama **autograd**: pendientes automáticas. El resultado queda en `precio.grad`.

¿No le crees? Para eso está el `assert` con `a_mano`: con el cálculo de la prepa, la pendiente es `18 × precio − 360`, y el `assert` revisa en cada paso que PyTorch diga lo mismo. Opcional, no lo memorices: al derivar `(3 × precio − 60)²` sale `2 × (3 × precio − 60) × 3`, que se simplifica a `18 × precio − 360`. No necesitas saber derivar; esa línea solo verifica que PyTorch hizo la misma cuenta, para que veas que no hay truco.

Y ese `assert` es una revisión automática: `assert condición, "mensaje"` no imprime nada si la condición es verdadera. Si es falsa, detiene el programa y muestra el mensaje. Mientras aprendes, no borres los `assert`: son tu semáforo.

**El paso.** `precio -= tasa * precio.grad`. La pendiente es −360: negativa quiere decir que el piso baja si subes el precio. Restar un negativo es sumar: 0 − 0.05 × (−360) = 18. El precio salta a $18. La `tasa` (*learning rate*, tasa de aprendizaje) es **qué tan largo es cada paso**: muy corto y tardas una eternidad; muy largo y te pasas al otro lado del valle. Es como estacionarte en reversa en un lugar apretado: si mueves el volante a puros jalones, te subes a la banqueta; si lo mueves de milímetro en milímetro, el de atrás te empieza a pitar. El `with torch.no_grad():` apaga la grabadora mientras giras la perilla, porque ajustar no es parte de la cuenta.

**Borrar la pendiente.** PyTorch **suma** cada pendiente nueva a la que ya tenía. Si no la borras con `precio.grad.zero_()`, el paso siguiente usaría la vieja más la nueva y se iría de largo.

Fíjate: la pérdida pasa de 3,600 a 36 a 0.36. Cerca del fondo la pendiente se hace chiquita, así que los pasos se achican solos. Nadie le dijo «frénate»: el cerro se lo dijo.

> **8:36 p.m.** Doña Lupe se asoma por encima del hombro de Mateo y ve el precio llegar a 20.
>
> —Veinte pesos. Eso te lo decía yo sin computadora.
>
> —Ya sé, tía. Pero ella no lo sabía. Empezó en cero, sin idea, y llegó sola.
>
> —Como tu primo con la bicicleta. Se cayó como veinte veces y a la veintiuno ya iba solito.
>
> Mateo se ríe y mira otra vez las cinco líneas. No encontró una respuesta escondida. Corrigió una mala respuesta hasta que dejó de ser mala. Y entiende algo que no esperaba: el primer intento no importa. Puede empezar pésimo. Lo que importa es que cada vuelta sea un poquito menos pésima. Lo que sigue es lo mismo, con 4,000 compras y 49 perillas.

## Paso 3 — Los datos: 4,000 compras inventadas

Las compras las **inventamos**: no hay datos personales de nadie, funciona sin internet y, sobre todo, **nosotros sabemos la regla que decide qué es fraude**, así que podemos revisar si el modelo la descubrió. Son el ensayo de Mateo antes de tocar otra vez los cobros de Norte.

De cada compra guardamos cuatro datos: **monto** (de $0 hasta casi $20,000), **hora** (de 0 a 23), **distancia** a la casa del cliente (de 0 hasta casi 500 km) y si el comercio es **extranjero** (1 = sí, 0 = no; 1 de cada 10).

Y esta es **la regla escondida**, que solo sabemos tú y yo. Cada compra prende «focos rojos»:

- un foco si el monto es de más de $15,000,
- un foco si la compra es de madrugada (antes de las 5),
- un foco si es a más de 400 km de la casa,
- un foco si el comercio es extranjero.

**Si una compra prende dos focos o más, es fraude.** Uno solo no basta: comprar una pantalla cara a mediodía es normal; comprarla a las 3 de la mañana ya huele raro.

El modelo **nunca** ve esta regla escrita: solo recibe los cuatro datos de cada compra y, mientras estudia, la respuesta correcta (fraude o no). Este archivo hace dos trabajos. Hasta aquí **nosotros** fabricamos el laboratorio con `focos`; desde `predecir(X_ent)`, el modelo solo recibe `X_ent`. `focos` nunca entra a `predecir`, y `y_ent` solo sirve para medir qué tan mal predijo. En datos reales, `y` vendría de casos ya investigados.

Crea `detector.py`. Va por partes. Copia los bloques 1 a 7 en el mismo archivo, en el orden en que aparecen (en el código van numerados con comentarios). Esta es la primera:

```python
# detector.py — un detector de fraude que aprende solo
import torch

torch.manual_seed(42)          # semilla fija: los mismos números "al azar" cada vez

# 1) DATOS INVENTADOS: 4,000 compras con tarjeta
n = 4000
monto      = torch.rand(n) * 20_000               # pesos, de 0 a 20,000
hora       = torch.randint(0, 24, (n,)).float()   # hora del día, de 0 a 23
distancia  = torch.rand(n) * 500                  # km desde la casa del cliente
extranjero = (torch.rand(n) < 0.1).float()        # 1 = comercio de otro país

# 2) LA REGLA ESCONDIDA (solo nosotros la conocemos; el modelo nunca la ve)
focos = (monto > 15_000).float() + (hora < 5).float() + (distancia > 400).float() + extranjero
y = (focos >= 2).float()       # la etiqueta: 1 = fraude, 0 = normal

# 3) EL TENSOR DE ENTRADA: compras × datos, todo entre 0 y 1
X = torch.stack([monto / 20_000, hora / 23, distancia / 500, extranjero], dim=1)
print("X:", tuple(X.shape), "| y:", tuple(y.shape), "| % fraude:", round(y.mean().item() * 100, 1))
```

Funciones nuevas, todas sencillas:

- `torch.rand(n)` da `n` números al azar entre 0 y 1; por 20,000, montos de 0 a 20,000 (broadcasting, como en la sesión 6). `torch.randint(0, 24, (n,))` da enteros al azar del 0 al 23.
- `torch.manual_seed(42)` fija el «azar». La computadora no tira dados: sigue una receta, y con la misma semilla da los mismos números cada vez que lo corres en la misma computadora.
- `(monto > 15_000)` da un `True` o `False` por compra; `.float()` lo vuelve 1.0 o 0.0. Sumar cuatro de esos cuenta los focos de las 4,000 compras a la vez.
- `torch.stack([...], dim=1)` pega las cuatro listas como **columnas** de una tabla.

Dividir cada columna (`monto / 20_000`, `hora / 23`…) pone todo entre 0 y 1, para que el monto (miles) no aplaste a la hora (decenas) solo por ser un número más grande.

La salida de esta parte:

```
X: (4000, 4) | y: (4000,) | % fraude: 15.8
```

**`X` tiene forma `(4000, 4)`: compras × datos de cada compra.** Una hoja de 4,000 renglones y 4 columnas. Al primer eje se le dice **lote** (cuántos ejemplos) y al segundo, **características** (qué sabemos de cada uno). Esa forma la vas a ver en todos lados.

**`y` tiene forma `(4000,)`:** una respuesta por compra. Se les llama **etiquetas**: la columna de respuestas correctas del examen. El 15.8 % de las compras son fraude.

Para verlo de cerca, agrega `print(X[:3])` y `print(y[:3])` justo después de ese `print`. Son una inspección temporal: ejecútalos una vez aquí y bórralos antes de comparar tu salida con la «Salida completa» del Paso 8:

```
tensor([[0.8823, 0.1304, 0.4309, 0.0000],
        [0.9150, 0.6087, 0.3986, 0.0000],
        [0.3829, 0.4348, 0.1907, 0.0000]])
tensor([1., 0., 0.])
```

La compra 0 es de $17,645 (0.8823 × 20,000) a las 3 de la mañana (0.1304 × 23): dos focos, fraude. La compra 1 es más cara, $18,300, pero a las 2 de la tarde: un solo foco, normal. Ese matiz es el que el modelo tiene que descubrir.

> **8:52 p.m.** —Ya está —dice Mateo, muy seguro—. Todo lo de más de quince mil es fraude. Lo marco y listo.
>
> Doña Lupe deja de contar monedas.
>
> —¿Y la señora que vino en agosto a pagar los útiles de sus cuatro hijos de un jalón? ¿También es ratera?
>
> Mateo abre la boca y la vuelve a cerrar. Mira la compra 1 en la pantalla: $18,300, a las dos de la tarde, normal. Su regla la habría acusado. **La pista no vive en una sola columna.** Ni el monto solo, ni la hora sola. Lo raro es cuando se juntan.
>
> —Una golondrina no hace verano —remata doña Lupe, y vuelve a sus monedas.
>
> Mateo tacha su regla. Por eso no la va a escribir él.

## Paso 4 — Estudiar y presentar examen

Si calificas a un alumno con las mismas preguntas que estudió, no sabes si aprendió o memorizó. Con los modelos pasa igual:

```python
# 4) Estudia con 3,200 compras; el examen es con las otras 800
X_ent, y_ent = X[:3200], y[:3200]
X_exa, y_exa = X[3200:], y[3200:]
```

Es rebanar por el primer eje, como en la sesión 3. Las últimas 800 compras se guardan bajo llave: el modelo nunca las ve mientras aprende.

Mateo llama a las primeras 3,200 «estudio» y a las otras 800 «examen». No es un adorno. Si le enseñara las respuestas del examen, el número final sonaría bonito y no diría nada.

Es la diferencia entre aprenderte la guía de memoria y entender el tema. Si el examen trae exactamente las preguntas de la guía, sacas diez aunque no sepas nada. Si el maestro cambia los números, se nota quién entendió. Las 800 compras apartadas son el examen con números cambiados.

Un detalle honesto: en los ejercicios 26 y 27 vas a comparar épocas y tasas mirando estas mismas 800. En cuanto eliges con ellas, ya no son un examen limpio, sino una práctica para decidir. En un proyecto real se aparta un tercer grupo, el examen final, que nadie consulta hasta el último día.

> **9:05 p.m.** El celular de doña Lupe vibra sobre el mostrador. Es Fer, desde Norte: «Ya cerré. Hoy todo fue en efectivo. Tres clientes de siempre se fueron sin comprar porque no traían. Le mando foto de las notas del día.»
>
> Doña Lupe lee el mensaje dos veces y se lo enseña a Mateo.
>
> —Y yo que hace un mes no quería ni decir su nombre —dice en voz baja—. Uno ve moros con tranchete donde no los hay.
>
> Mateo no contesta. Guarda la foto de las notas en una carpeta nueva. Si el modelo dice algo de los cobros de Norte, va a necesitar con qué compararlo. Datos de verdad, apartados, que el modelo nunca haya visto. Igual que el examen.

## Paso 5 — Las perillas y la predicción con `@`

Ahora las perillas. En `cerro.py` había una. Aquí hay cuatro tensores de perillas:

```python
# 5) LAS PERILLAS: empiezan al azar y el modelo las va girando
W1 = (torch.randn(4, 8) * 0.5).requires_grad_()
b1 = torch.zeros(8, requires_grad=True)
W2 = (torch.randn(8) * 0.5).requires_grad_()
b2 = torch.zeros((), requires_grad=True)   # () = un solo número: la perilla final

def predecir(X):
    oculta = torch.relu(X @ W1 + b1)   # (compras, 4) @ (4, 8) -> (compras, 8)
    return oculta @ W2 + b2            # (compras, 8) @ (8,)   -> (compras,)
```

En los libros a estas perillas les dicen **pesos** (la `W`, de *weights*) y **sesgos** (la `b`, de *bias*). `W1` tiene 4 × 8 = 32, `b1` tiene 8, `W2` tiene 8 y `b2` tiene 1: **49 perillas**. Empiezan al azar (`torch.randn`) o en cero, como el precio empezó en $0. Dos detalles de sintaxis: el `()` de `torch.zeros((), ...)` quiere decir «un solo número», la perilla final; y el guion bajo de `requires_grad_()` prende la «grabadora» sobre ese mismo tensor, ya creado.

¿Has visto la consola de un sonidero en una fiesta? Una mesa llena de perillas: graves, agudos, volumen de cada micrófono, eco. Ninguna perilla sola hace que suene bien. Lo que suena bien es la combinación. Y el sonidero no la calcula: mueve una, escucha, mueve otra, vuelve a escuchar. Los **pesos** son eso. Cada uno dice cuánto importa un dato en una mezcla. Cada peso de `W1` ajusta cuánto cuenta un dato en una de las sospechas intermedias: grande, «súbele a esto»; cerca de cero, «a esto ni caso»; negativo, «réstale». Por sí solo no mide la importancia global del dato ni si es señal de fraude o de normalidad: eso depende también de `W2`, de los sesgos y de `relu`. La diferencia es que nuestro sonidero tiene 49 perillas y las mueve todas a la vez, guiado por la pérdida en lugar del oído.

Lee `predecir` mirando solo las formas, como en la sesión 6:

- `X @ W1`: `(compras, 4) @ (4, 8)` da `(compras, 8)`. Cada compra pasa de 4 datos a **8 «sospechas» intermedias**, cada una una mezcla distinta de monto, hora, distancia y extranjero. Una podría acabar fijándose en «caro y de madrugada»; otra, en «lejos y extranjero». Nadie le dice cuáles: las descubre al girar las perillas.
- `+ b1`: un vector `(8,)` sumado a cada renglón. Broadcasting.
- `torch.relu(...)`: **si un número sale negativo, lo vuelve cero; si sale positivo, lo deja igual.** «Si esta sospecha no aplica, ni la menciones». Sin `relu`, las dos capas se juntarían en una sola mezcla, una sola cuenta de multiplicar y sumar, que no alcanza a imitar los cortes de la regla (más de $15,000, antes de las 5, más de 400 km) ni cómo se combinan. (Si los focos ya vinieran contados como entradas, «dos o más» sí sería una sola mezcla.)
- `oculta @ W2 + b2`: `(compras, 8) @ (8,)` da `(compras,)`, la cuenta de la tercera pregunta del repaso. **Un número por compra**: el **puntaje**.

**Puntaje mayor que cero: el modelo dice «fraude»; menor o igual que cero: «normal».** Entre más lejos del cero, más seguro está el modelo, lo cual no es una garantía. En los libros a este puntaje le dicen **logit** (de ahí el nombre de la pérdida del Paso 6). Un puntaje de 0 equivale a «mitad y mitad»: 50 % de probabilidad de fraude según el modelo. Por eso el corte está en el cero, y el código rompe el empate exacto como «normal» (`> 0`, no `>= 0`). A esto se le llama **predicción** o *forward* («hacia adelante»): los datos entran por un lado y el puntaje sale por el otro, como la masa en la máquina de tortillas.

> **9:20 p.m.** Mateo se queda viendo `W1` como si le debiera dinero.
>
> —¿Y cuál de las ocho es la de «caro y de madrugada»? —pregunta en voz alta, a nadie.
>
> Imprime `W1` con `print(W1)`. Le salen treinta y dos números al azar, ninguno con nombre. Se desespera tantito. Quería ver la regla escrita, bonita, en una casilla. No va a pasar. Las sospechas no traen etiqueta: se van formando solas mientras las perillas giran.
>
> —¿Ya te peleaste con la máquina? —pregunta doña Lupe.
>
> —Todavía no. Nomás no me dice qué está pensando.
>
> —Como tu tío. Treinta años de casada y sigo sin saber.
>
> Mateo suelta la risa y decide no pelear con lo que no puede ver. Le basta con cuidar las formas: cuatro datos entran, ocho señales salen y al final queda un puntaje por cobro.

## Paso 6 — La pérdida para un «sí o no»

En `cerro.py`, la pérdida era la diferencia al cuadrado. Para preguntas de «sí o no» se usa otra, ya hecha en PyTorch:

```python
perdida_fn = torch.nn.BCEWithLogitsLoss()
optimizador = torch.optim.SGD([W1, b1, W2, b2], lr=1.0)
```

No te asustes por el nombre. `BCEWithLogitsLoss` hace el papel de nuestra resta al cuadrado, pero para fraude/no fraude: recibe puntajes y etiquetas y devuelve **un solo número** con qué tan mal le fue en todas las compras juntas. Su regla es de sentido común: si era fraude y el puntaje fue muy positivo, pérdida casi cero; si era fraude y el puntaje fue muy negativo, pérdida enorme (estar muy seguro y equivocarse es lo peor); si el puntaje está cerca de cero, pérdida mediana.

Piensa en el amigo que te jura que el partido ya está ganado y al final pierden 3 a 0. No es lo mismo que el amigo que dijo «quién sabe». Los dos se equivocaron, pero al primero le cobras más caro la apuesta. Esta pérdida cobra igual: dudar cuesta poquito; equivocarse muy seguro cuesta muchísimo. Así el modelo aprende a no presumir seguridad que no tiene.

La segunda línea es el **optimizador**, el que gira las perillas. `SGD` hace lo mismo que hiciste a mano en `cerro.py`, `perilla -= tasa * pendiente`, pero con las 49 a la vez. `lr=1.0` es la tasa.

> **9:41 p.m.** La máquina ya tiene los datos, las perillas y una manera de calificar sus errores. Solo falta soltarla.
>
> Mateo tiene el dedo encima de la tecla y no la aprieta. ¿Y si no aprende nada? ¿Y si la pérdida se queda donde está y él pierde la noche, y la siguiente, y el 23 llega con las manos vacías?
>
> Doña Lupe apaga la cafetera, que ya huele a quemado.
>
> —El que no arriesga no gana, mijo. Y el que no le pica al botón no se entera.
>
> El reloj de la pared marca casi las diez. Faltan nueve días, y cada uno pesa. Mateo respira. Cada vuelta del ciclo tiene que acercarlo a una pista, no solo a un número bonito.

## Paso 7 — El ciclo completo y las épocas

Ahora sí, el corazón de la sesión:

```python
# 6) EL CICLO DE ENTRENAMIENTO: una época = una vuelta a las 3,200 compras
for epoca in range(1, 301):
    puntaje = predecir(X_ent)               # 1. predecir
    perdida = perdida_fn(puntaje, y_ent)    # 2. medir qué tan mal le fue
    optimizador.zero_grad()                 # 3. borrar las pendientes de la vuelta anterior
    perdida.backward()                      # 4. calcular la pendiente de cada perilla
    optimizador.step()                      # 5. girar cada perilla un pasito cuesta abajo
    if epoca in (1, 10, 50, 100, 200, 300):
        print(f"época {epoca:3d}   pérdida {perdida.item():.4f}")
```

Compáralo con `cerro.py`. Es el mismo ciclo, renglón por renglón:

| Paso | En `cerro.py` | En `detector.py` |
|---|---|---|
| Predecir | `cobro = precio * 3` | `puntaje = predecir(X_ent)` |
| Medir | `(cobro - 60) ** 2` | `perdida_fn(puntaje, y_ent)` |
| Borrar pendientes viejas | `precio.grad.zero_()` | `optimizador.zero_grad()` |
| Calcular pendientes | `perdida.backward()` | `perdida.backward()` |
| Girar perillas | `precio -= tasa * precio.grad` | `optimizador.step()` |

Cada vuelta del `for` le enseña al modelo **todas** las 3,200 compras de estudio. Una vuelta completa a todos los datos se llama **época**: repasar todo el temario una vez. Aquí le damos 300.

¿Por qué tantas? Porque en cada vuelta las perillas se mueven solo un pasito. Es como aprenderte una canción en la guitarra: la primera vez que la tocas completa te equivocas en todos los cambios de acorde. La décima, solo en el difícil. La número cien, ya ni lo piensas. Cada época es tocar la canción entera una vez más. Nadie la aprende a la primera, ni tú ni la computadora.

Corre lo que llevas y mira la pérdida:

```
época   1   pérdida 0.6819
época  10   pérdida 0.4098
época  50   pérdida 0.2528
época 100   pérdida 0.2070
época 200   pérdida 0.1984
época 300   pérdida 0.1910
```

**La pérdida de estudio baja:** el modelo está reduciendo errores en las 3,200 compras que ya vio. Es una buena señal, pero todavía no prueba que funcione con compras nuevas: eso lo revisa el examen del Paso 8. Baja rápido al principio y luego cada vez más despacio, como con el cuaderno: cerca del fondo el piso casi no se inclina. En mi máquina, las 300 épocas tardaron menos de una décima de segundo.

> **¿Por qué 300 y no más?** Lo medimos. Cada computadora redondea los decimales en un orden un poco distinto; la diferencia es microscópica, del tamaño de la séptima cifra decimal. Con pasos largos como `lr=1.0`, después de unas 300 vueltas esas diferencias empiezan a crecer: con 400, dos computadoras con las mismas versiones terminaron con conteos distintos de fraudes atrapados. Con 300, el resultado no se movió ni cuando le metimos a propósito diferencias mil veces más grandes que las que hay de una computadora a otra. Por eso el curso usa 300: para que, si tu número sale distinto, sepas que el error está en tu código y no en tu máquina.

**9:47 p.m.** Mateo aprieta la tecla.

La terminal escupe la primera línea antes de que él alcance a parpadear: `época 1 pérdida 0.6819`. Se le hace un nudo en el estómago. Ese número no le dice nada todavía. Es el modelo adivinando a ciegas, con las perillas al azar, como un perro que oye su nombre en otro idioma.

Luego aparece la segunda: `0.4098`.

Mateo se inclina hacia la pantalla.

`0.2528`.

—Tía. Tía, ven.

Doña Lupe llega secándose las manos en el mandil.

—¿Qué pasó? ¿Se descompuso?

—No. Mire. Está bajando.

`0.2070`. `0.1984`. `0.1910`.

Seis números, uno debajo del otro, cada uno más chico. Ya se acabó. Duró menos de lo que tarda en hervir el agua del café. Mateo se queda viendo la columna como quien ve un marcador al final de un partido que no esperaba ganar.

Nadie le escribió a la máquina la regla «dos focos o más». Nadie le habló de madrugadas ni de comercios extranjeros. Sí le dieron 3,200 compras, sus cuatro datos y la respuesta fraude/no fraude de cada una, y una manera de medir qué tan mal le iba. Con esos ejemplos, vuelta por vuelta, fue moviendo sus 49 perillas hasta equivocarse menos. Mateo siente algo raro, entre orgullo y escalofrío. Él escribió unas cuantas líneas. Lo demás lo hizo ella.

—¿Y eso es bueno? —pregunta doña Lupe.

—Es que está aprendiendo, tía. La estoy viendo aprender.

Doña Lupe mira la pantalla, luego a su sobrino, que tiene los ojos brillosos como cuando era niño y le salía el trompo.

—Pues ojalá aprenda rápido. Que el 23 no espera.

Mateo asiente, pero ya no la oye del todo. **La pérdida es una señal, no una sentencia.** Le dice que el modelo se equivoca menos en el estudio. Todavía falta lo importante: ver qué hace cuando le quitan las respuestas.

## Paso 8 — El examen: 91 %

Falta lo importante: ¿le va bien con compras que **nunca vio**? Agrega la última parte:

```python
# 7) EL EXAMEN: 800 compras que el modelo nunca vio
with torch.no_grad():
    dice_fraude = (predecir(X_exa) > 0).float()   # puntaje positivo = "esto es fraude"

exactitud = (dice_fraude == y_exa).float().mean().item()
atrapados = (dice_fraude * y_exa).sum().item()          # dijo fraude y sí era
falsas    = (dice_fraude * (1 - y_exa)).sum().item()    # dijo fraude y no era
escapados = ((1 - dice_fraude) * y_exa).sum().item()    # dijo normal y era fraude

print("exactitud:", round(exactitud, 3))
print("fraudes en el examen:", int(y_exa.sum().item()))
print("atrapados:", int(atrapados), "| se escaparon:", int(escapados), "| alarmas falsas:", int(falsas))
print("recall:", round(atrapados / (atrapados + escapados), 3),
      "| precisión:", round(atrapados / (atrapados + falsas), 3))
```

Las dos últimas cifras que imprime (recall y precisión) todavía no tienen nombre para ti: ignóralas hasta el Paso 9, donde se explican y las vuelves a leer. Por ahora mira la exactitud y las tres cuentas.

El truco es `*` elemento por elemento, el de la sesión 6: `dice_fraude * y_exa` solo da 1 donde **las dos** son 1 (dijo fraude **y** era). `1 - y_exa` voltea unos y ceros, así que `dice_fraude * (1 - y_exa)` da 1 donde dijo fraude y **no** era. Sumas y tienes cuántos de cada caso.

El script completo tiene 61 líneas, 41 sin comentarios ni líneas en blanco. Córrelo con `python3 detector.py`. Salida completa, idéntica las dos veces que lo corrí:

```
X: (4000, 4) | y: (4000,) | % fraude: 15.8
época   1   pérdida 0.6819
época  10   pérdida 0.4098
época  50   pérdida 0.2528
época 100   pérdida 0.2070
época 200   pérdida 0.1984
época 300   pérdida 0.1910
exactitud: 0.908
fraudes en el examen: 125
atrapados: 81 | se escaparon: 44 | alarmas falsas: 30
recall: 0.648 | precisión: 0.73
```

**Exactitud: 0.908.** En compras que nunca vio, acertó casi 91 de cada 100, sin que nadie le dijera la regla de los focos.

Sobre la semilla: con la misma versión de PyTorch y una computadora como la mía, lo normal es que te salga exactamente esto. Pero PyTorch no garantiza números idénticos entre versiones ni entre computadoras distintas ([documentación de PyTorch sobre reproducibilidad](https://docs.pytorch.org/docs/2.14/notes/randomness.html)): los últimos decimales pueden moverse un poco. Con 300 épocas medimos que no se mueven (ver «¿Por qué 300 y no más?» en el Paso 7); en los ejercicios con muchas más vueltas, sí pueden moverse. Lo que **sí** tiene que pasar siempre: la pérdida baja y la exactitud queda alrededor de 91 %. Si te sale muy distinto, revisa la tabla de errores.

**10:12 p.m.** Mateo ve el 91 y levanta los brazos como si hubiera metido gol.

—¡Noventa y uno, tía! ¡Acertó noventa y uno de cada cien!

Doña Lupe no levanta los brazos. Se acomoda los lentes y se acerca a la pantalla con la calma de quien ha contado mil cortes de caja.

—¿Entonces qué tan bueno es?

—Pues… muy bueno. Es un nueve.

—Un nueve en qué. ¿Qué dice ese renglón de abajo?

Mateo lo lee en voz alta, más despacio: «se escaparon: 44».

Se le baja la emoción como globo pinchado. Cuarenta y cuatro. De 125 fraudes en el examen, cuarenta y cuatro pasaron como si nada. El modelo los vio, los revisó y dijo «normal».

Son compras inventadas, ya lo sabe. Pero no puede evitar ponerles cara. Se imagina a una de esas cuarenta y cuatro. Una señora que compró cuadernos en Norte en septiembre, pagó con tarjeta, dio las gracias y se fue. Y una mañana, en la fila de las tortillas, abre la app del banco y ve un cargo de diecisiete mil pesos en una tienda donde nunca ha estado. Su aguinaldo, o lo que iba a ser su aguinaldo. Luego se imagina al muchacho que juntó para un celular y se lo vaciaron un martes a las tres de la mañana. Y a otro. Y a otro. Cuarenta y cuatro veces.

—No es un nueve, tía —dice, y la voz le sale chiquita—. Para una tarea, nueve es buena nota. Para 125 fraudes, cuarenta y cuatro es una puerta abierta.

Doña Lupe le pone una mano en el hombro.

—Ya ves. No todo lo que brilla es oro, mijo. Ahora averigua de qué está hecho.

> **Recuadro de alarma:** la exactitud cuenta todos los aciertos juntos. No te dice cuántos fraudes dejó pasar. Por eso un banco no puede quedarse con ese número.

¿91 % es bueno? Suena a 9 de calificación. Ahora viene lo más importante de la sesión.

## Paso 9 — La trampa: el detector flojo saca 84

Imagina un «detector» que a todas las compras les dice «no es fraude», sin mirarlas.

En el examen hay 800 compras y 125 son fraude. Las otras 675 son normales, y el flojo las acierta todas: 675 de 800 es **84.4 %**. Lo vas a comprobar con un tensor en el ejercicio 28.

Un programa de una línea que no sirve para nada saca 84; el nuestro, 91. La diferencia es mucho más chica de lo que parecía. **La exactitud engaña cuando lo importante es raro**, y en un banco el fraude suele ser raro: casi todas las compras son de gente honesta.

Piénsalo con un detector de metales en la entrada de un concierto. Casi nadie trae nada peligroso. Si el guardia deja pasar a todos sin revisar, «acierta» con casi todo el público. Y aun así es el peor guardia del mundo, porque el único caso que importaba se le pasó. Acertar con lo común es fácil. Lo difícil, y lo que vale, es atrapar lo raro.

> **10:31 p.m.** Mateo escribe el detector flojo en un papel, sin computadora: «a todo, no». Lo mira un rato. Es una línea. Es ridícula. Y saca 84.
>
> —O sea que mi modelo de 49 perillas le gana a no hacer nada por menos de siete puntos —murmura.
>
> —¿Y eso es poco? —pregunta doña Lupe.
>
> —Es como presumir que corres más rápido que alguien sentado.
>
> Doña Lupe suelta una carcajada que hace sonar los vasos del anaquel.
>
> —Pues con eso no le ganas ni al camión de la basura.
>
> Mateo no se ofende. Por primera vez en la noche siente que entiende de verdad. El número que importa no es el más grande. Es el que cuenta lo que se escapó.

Por eso los bancos miran otros dos números. Primero, los cuatro casos posibles del examen:

| | De verdad era fraude | De verdad era normal |
|---|---|---|
| **El modelo dijo «fraude»** | 81 atrapados | 30 alarmas falsas |
| **El modelo dijo «normal»** | 44 se escaparon | 645 bien dejados pasar |

(Las cuatro casillas suman 800. Los aciertos son 81 + 645 = 726, y 726 / 800 = 0.9075, que redondeado da 0.908, la exactitud de arriba.)

**Recall: de todos los fraudes, ¿cuántos atrapó?** Hubo 125 fraudes y atrapó 81: 81 / 125 = **0.648**. Se le escaparon 44 de 125: más de 1 de cada 3 fraudes. Cada uno de esos 44 es alguien que mañana revisa su estado de cuenta y encuentra un cargo de $17,000 en una tienda donde nunca ha estado. Tiene que reclamar el cargo, el banco revisa el caso y muchas veces termina devolviéndole el dinero; el susto no se lo devuelve nadie. El detector flojo tiene recall **cero**: no atrapa ni uno.

**Precisión: de todas las alarmas que dio, ¿cuántas eran ciertas?** Dio 81 + 30 = 111 alarmas, y 81 eran fraude: 81 / 111 = **0.73**. Más o menos 1 de cada 4 alarmas fue una falsa alarma. Cada una de esas 30 es una persona honesta a la que, según cómo trabaje el banco, le detuvieron la compra en la caja del súper, con la fila detrás, o le pidieron confirmar que era ella antes de dejarla pasar. Si eso pasa seguido, el cliente se cambia de banco.

Las dos cosas pesan. Si el banco quiere atrapar **todos** los fraudes, marca todo como sospechoso: recall perfecto y todos sus clientes enojados. Si no quiere molestar a **nadie**, no marca nada: cero alarmas falsas y todos los fraudes pasan, como con el flojo. El trabajo de verdad está en el medio, y depende de cuánto cuesta cada error. Por eso, para juzgar un sistema contra fraude, importa más cuántos fraudes atrapa y cuántas alarmas falsas da que la «exactitud» sola.

**La lección grande de hoy:** acertar mucho no basta. Antes de creerle a un modelo, pregunta qué tan raro es lo que busca y cuántos de esos casos raros atrapa.

> **11:05 p.m.** Mateo dibuja la tabla de cuatro casillas en la parte de atrás de un recibo de luz. Arriba a la izquierda, 81: los que atrapó. Abajo a la izquierda, 44: los que se fueron. Arriba a la derecha, 30: gente honesta con la tarjeta rechazada en la caja. Se queda viendo esa casilla. Si el banco le hiciera eso a los clientes de Norte, la papelería perdería a los pocos que todavía regresan.
>
> —No es un número, tía. Son dos. Y se jalan uno al otro, como cobija chiquita: si te tapas los pies, se te destapan los hombros.
>
> Doña Lupe asiente despacio. Esa la entiende perfecto. Mateo guarda el recibo en la bolsa de la chamarra. Mañana lo va a necesitar.

---

👉 **Si eres de Administración:** no necesitas entrenar el modelo para hacer la pregunta correcta en la junta. Cuando alguien presente «nuestro modelo acierta 91 %», pregunta: ¿cuánto saca uno que no hace nada? ¿Cuántos de los casos que nos importan atrapa (recall)? ¿Cuántas de sus alarmas son ciertas (precisión)? Luego traduce cada error a pesos: cuánto cuesta un fraude que se escapa y cuánto un cliente molesto. Esa tabla de cuatro casillas es una herramienta de negocio, y sirve igual para decidir a quién dar crédito o qué reclamación de seguro revisar.

---

## Ejercicios de la sesión 7

Las soluciones están al final. Antes de cada cambio, **escribe en papel qué crees que va a pasar**. Predecir antes de correr es la mitad del aprendizaje. Llena esta tabla en papel (los ejercicios 26 y 27 te piden estas cuatro corridas):

| Configuración | Mi predicción | Pérdida final | Exactitud | Atrapados | Alarmas falsas | ¿Qué aprendí? |
|---|---|---:|---:|---:|---:|---|
| 300 épocas, `lr=1.0` | | | | | | |
| 2000 épocas, `lr=1.0` | | | | | | |
| 300 épocas, `lr=0.1` | | | | | | |
| 300 épocas, `lr=10` | | | | | | |

**Ejercicio 26 — Más vueltas, menos vueltas.** En `detector.py`, el número de épocas está en `range(1, 301)`. Ya lo corrí con 100 épocas (`range(1, 101)`): la pérdida termina en 0.2070, la exactitud en 0.902, atrapa 77 fraudes y da 30 alarmas falsas. Ahora tú: córrelo con **2000** épocas. Agrega `1000, 2000` a la lista de épocas que se imprimen para que veas la pérdida.

a) Antes de correr: ¿la pérdida final va a ser mayor o menor que 0.1910?
b) ¿Qué pasó con la exactitud, los atrapados y las alarmas falsas?
c) Señala en tu código cuál renglón es cada uno de los cinco pasos del ciclo (predecir, medir, borrar pendientes, calcular pendientes, girar perillas).

**Ejercicio 27 — El largo del paso.** Regresa a 300 épocas. Ahora cambia la tasa, `lr=1.0`, primero a `lr=0.1` (pasitos de hormiga) y después a `lr=10` (zancadas de gigante). Antes de correr cada una, predice: ¿baja más o menos la pérdida? ¿Por qué?

Aviso: con una de las dos el programa truena al final con un `ZeroDivisionError`. No es un error tuyo. Lee las líneas de arriba del error y explica por qué pasó. Pista: mira cuántas alarmas dio en total.

**Ejercicio 28 — Calcula al detector flojo.** Agrega esto al final de `detector.py` y llena los huecos. El detector flojo es un tensor de 800 ceros: «no es fraude» para todas las compras del examen.

```python
# Ejercicio 28: el detector flojo que siempre dice "no es fraude"
siempre_no = torch.____(800)
assert siempre_no.shape == y_exa.shape, "Tiene que haber una respuesta por cada compra del examen"
exactitud_flojo = (siempre_no == ____).float().mean().item()
atrapados_flojo = (siempre_no * ____).sum().item()
print("exactitud del flojo:", round(exactitud_flojo, 3))
print("fraudes que atrapa:", int(atrapados_flojo), "de", int(y_exa.sum().item()))

# la misma cuenta con un for, como antes de los tensores
aciertos = 0
for i in range(800):
    if y_exa[i].item() == ____:
        aciertos = aciertos + 1
assert round(aciertos / 800, 3) == round(exactitud_flojo, 3), "El for y el tensor deberían coincidir"
print("El for dice lo mismo:", aciertos, "aciertos de 800")
```

**Segunda parte.** Calcula ahora el recall y la precisión del **detector entrenado**, sin mirar las dos últimas líneas del Paso 8. Ya tienes `atrapados`, `falsas` y `escapados` en tu archivo:

```python
recall_modelo = ______________________________
precision_modelo = ___________________________

print("recall del modelo:", round(recall_modelo, 3))
print("precisión del modelo:", round(precision_modelo, 3))
assert round(recall_modelo, 3) == 0.648 and round(precision_modelo, 3) == 0.73, "Revisa cuál de las cuatro casillas falta en tu fracción"
```

Con la corrida base (300 épocas, `lr=1.0`) deben salir `0.648` y `0.73`. Si cambiaste épocas o tasa, quita el `assert`.

Después contesta: ¿cuál es el recall del detector flojo? ¿Por qué no puedes calcular su precisión?

**Ejercicio 29 — ¿Qué aprendió el modelo?** La regla usa los cuatro datos. ¿El modelo se dio cuenta? Pregúntaselo **revolviendo una columna**: si revuelves la hora entre las 800 compras del examen, cada compra se queda con la hora de otra al azar. La columna sigue ahí, pero ya no dice nada. Si la exactitud cae mucho en esta corrida, es evidencia de que el modelo estaba usando esa columna para predecir. No prueba por sí sola que haya descubierto la regla exacta ni que esa columna cause fraude.

Vas a repetirlo para cada una de las 4 columnas: copiar el examen, revolver **solo esa columna**, volver a predecir y mostrar la exactitud y los fraudes atrapados. Dos herramientas nuevas:

- `X_exa.clone()` hace una copia, para no echar a perder el examen original.
- `torch.randperm(800)` da los números del 0 al 799 en desorden. `X_exa[torch.randperm(800), col]` es la columna `col` revuelta.

Recuerda predecir dentro de `with torch.no_grad():`, como en el examen. Un ejemplo resuelto, con la columna `monto` (la columna 0):

```python
X_original = X_exa.clone()
X_revuelto = X_exa.clone()
X_revuelto[:, 0] = X_exa[torch.randperm(800), 0]

with torch.no_grad():
    dice = (predecir(X_revuelto) > 0).float()

print("monto: exactitud", round((dice == y_exa).float().mean().item(), 3))
```

Ahora completa tú, con un `for`, las columnas 1, 2 y 3 (hora, distancia y extranjero), y muestra también los fraudes atrapados:

```python
for col in range(1, 4):
    X_revuelto = X_exa.________()
    X_revuelto[:, col] = X_exa[torch.________(800), col]
    with torch.no_grad():
        dice = (________(X_revuelto) > 0).float()
    # imprime la exactitud y los fraudes atrapados de esta columna
    assert torch.equal(X_exa, X_original), "No debes alterar el examen original"
```

Después contesta: ¿qué columna le dolió más perder al modelo? ¿Alguna no le importó? ¿Cuadra con la regla escondida de los focos?

---

## Errores comunes (y cómo darte cuenta)

| Error | Cómo se ve | Cómo darte cuenta | Qué hacer |
|---|---|---|---|
| Olvidar `requires_grad=True` en una perilla | En `cerro.py`, donde es la única perilla: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` al llamar `backward()`. En `detector.py`, si lo olvidas solo en `W1`, **no truena**: `W1.grad` se queda en `None`, `W1` nunca se mueve, la pérdida se atora en 0.3366 y atrapa solo 13 de 125 | En `cerro.py` truena en la línea del `backward`; en `detector.py`, la pérdida baja mucho menos de lo normal y `print(W1.grad)` dice `None` | Prende la «grabadora» al crear cada perilla: `requires_grad=True` |
| Olvidar borrar las pendientes (`zero_grad`) | En `cerro.py`, el `assert` te salva: truena en el paso 2 con `AssertionError: PyTorch y el lápiz deberían coincidir`. Sin los `assert`, no truena, y eso es lo peligroso: el precio salta a 37.80, luego a 41.58, la pérdida sube hasta 4,191.27 (en el paso 4) y, al terminar, el precio aprendido queda en 4.95 en lugar de 20 | La pérdida sube y baja en lugar de bajar | PyTorch suma las pendientes nuevas a las viejas; bórralas en cada vuelta |
| Etiquetas con otra forma que los puntajes | `ValueError: Target size (torch.Size([3200, 1])) must be the same as input size (torch.Size([3200]))` | Imprime `y.shape` y el `shape` del puntaje: tienen que ser iguales | Deja `y` con forma `(4000,)`, sin `reshape(4000, 1)` |
| Tasa demasiado grande | El entrenamiento no truena. Con `lr=10` la pérdida se atora en 0.4365 y el modelo deja de atrapar fraudes; al final truena la precisión con `ZeroDivisionError`, porque no dio ninguna alarma (ejercicio 27) | La pérdida deja de bajar o brinca | Baja la tasa; prueba de 10 en 10 veces (1, 0.1, 0.01…) |
| Llamar `backward()` dos veces sobre la misma pérdida | `RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).` | Truena en el segundo `backward` | Un `backward` por vuelta; la siguiente vuelta vuelve a predecir y a medir |
| Creerle a la exactitud sola | «¡91 %!» | Calculas al detector flojo y saca 84 % | Mira también recall y precisión |

---

## Repaso para la próxima sesión

La sesión 8 empieza con estas tres preguntas. Contéstalas sin ver la lección:

1. ¿Qué forma tiene el tensor de entrada del detector y qué es cada eje?
2. ¿Qué mide la pérdida y por qué tiene que bajar?
3. ¿Por qué acertar 91 % de las veces puede no ser suficiente para un banco?

🔁 **La próxima semana:** los tensores en el banco, la aseguradora y la tienda. Casos reales, con su fuente, y el reto de escribir tú la forma del tensor de cada uno. Spoiler: el detector de hoy tiene exactamente la forma de los de verdad.

---

## Soluciones (no las veas antes de intentarlo)

**Repaso para abrir.**

1. Se comparan las formas **de derecha a izquierda**, eje por eje. En cada eje, las dos medidas tienen que ser **iguales, o una de ellas tiene que ser 1** (el 1 es el que se «estira»). Si en algún eje no se cumple, PyTorch da error.
2. `*` multiplica **elemento por elemento**: casilla con casilla, y las formas tienen que coincidir o estirarse. `@` es el **producto de matrices**: renglón por columna, multiplicando y sumando. Lo engañoso es que a veces los dos corren sin error y dan cosas distintas.
3. `(7,)`. El 3 de en medio se «come» (se multiplica y se suma) y queda un número por cada uno de los 7 renglones. Hoy lo usaste: `(compras, 8) @ (8,)` dio `(compras,)`.

**Ejercicio 26.** Con 2000 épocas (las primeras líneas son iguales a las de 300; pego desde la 300):

```
época 300   pérdida 0.1910
época 1000   pérdida 0.1641
época 2000   pérdida 0.1142
exactitud: 0.93
fraudes en el examen: 125
atrapados: 83 | se escaparon: 42 | alarmas falsas: 14
recall: 0.664 | precisión: 0.856
```

a) Menor: 0.1142. Más vueltas, más tiempo para bajar el cerro. Las primeras épocas dan los mismos números que con 300 gracias a la semilla fija. Del 1000 en adelante, en tu computadora la última cifra de la pérdida y algún conteo pueden salir un poco distintos: son las diferencias de redondeo del recuadro «¿Por qué 300 y no más?», que con tantas vueltas tienen tiempo de crecer.

b) La exactitud sube a 93 %, atrapa 83 fraudes en lugar de 81 y las alarmas falsas bajan de 30 a 14. Ojo: más épocas no siempre ayuda. Con datos reales, demasiadas vueltas pueden hacer que el modelo **memorice** los ejemplos de estudio y le vaya peor en el examen. Por eso el examen va con datos apartados.

c) `puntaje = predecir(X_ent)` predice; `perdida = perdida_fn(puntaje, y_ent)` mide; `optimizador.zero_grad()` borra pendientes; `perdida.backward()` calcula pendientes; `optimizador.step()` gira perillas.

**Ejercicio 27.** Con `lr=0.1` (quito algunas épocas de en medio):

```
época   1   pérdida 0.6819
época  50   pérdida 0.4238
época 300   pérdida 0.3194
exactitud: 0.859
fraudes en el examen: 125
atrapados: 12 | se escaparon: 113 | alarmas falsas: 0
recall: 0.096 | precisión: 1.0
```

La pérdida baja, pero mucho más despacio: en 300 épocas llega a 0.3194, peor que lo que la tasa 1.0 lograba en 50. Pasitos de hormiga: necesita muchas más vueltas. Y atrapa solo 12 de 125. Fíjate en la precisión: 1.0, perfecta, porque las 12 alarmas que dio eran fraude. Un detector que casi nunca suena casi nunca se equivoca cuando suena, y aun así deja pasar 113 fraudes: otra razón para no juzgarlo con un solo número.

Con `lr=10` (igual, recortado):

```
época   1   pérdida 0.6819
época  10   pérdida 0.8124
época  50   pérdida 0.4365
época 300   pérdida 0.4365
exactitud: 0.844
fraudes en el examen: 125
atrapados: 0 | se escaparon: 125 | alarmas falsas: 0
Traceback (most recent call last):
  ...
    "| precisión:", round(atrapados / (atrapados + falsas), 3))
                          ~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
```

Las zancadas eran tan largas que en la época 10 la pérdida **subió** (de 0.6819 a 0.8124): brincó al otro lado del valle. Luego se atoró en 0.4365. ¿Y en qué terminó? Exactitud **0.844**, cero atrapados, cero alarmas: **el modelo terminó comportándose como el detector flojo.** Con `lr=10`, en esta corrida las actualizaciones apagaron las ocho señales intermedias (`relu` dejó en cero las 25,600 de las compras de estudio) y el modelo quedó con un puntaje constante y negativo, el mismo para todas las compras (−1.67 en el examen). Con el corte en cero, nunca da alarma. Ojo: sigue minimizando la misma pérdida de siempre; no «eligió» la exactitud. Que le salga 84 % es consecuencia de que el fraude es raro.

El `ZeroDivisionError` sale porque la precisión es atrapados entre alarmas, y hubo 0 alarmas: 0 / 0. No se puede medir la puntería de alguien que nunca disparó.

**Ejercicio 28.** Los huecos son `zeros`, `y_exa`, `y_exa` y `0`. Este bloque no es para copiar: muestra solo las líneas con huecos ya llenos, y los `...` son las líneas de tu archivo que no cambian:

```text
siempre_no = torch.zeros(800)
exactitud_flojo = (siempre_no == y_exa).float().mean().item()
atrapados_flojo = (siempre_no * y_exa).sum().item()
...
    if y_exa[i].item() == 0:
```

Salida (las últimas tres líneas, después de la salida normal del detector):

```
exactitud del flojo: 0.844
fraudes que atrapa: 0 de 125
El for dice lo mismo: 675 aciertos de 800
```

El recall es 0 / 125 = **0**: no atrapa ni un fraude. La precisión no se puede calcular por la misma razón que en el ejercicio 27: no da ninguna alarma, y 0 / 0 no está definido. Un detector con 84 % de exactitud y recall cero: esa es toda la lección en dos números.

**Ejercicio 29.** Una solución:

```python
# Ejercicio 29: ¿qué dato le importa más al modelo?
nombres = ["monto", "hora", "distancia", "extranjero"]
for col in range(4):
    X_revuelto = X_exa.clone()                                  # copia del examen
    X_revuelto[:, col] = X_exa[torch.randperm(800), col]        # revuelve UNA columna
    with torch.no_grad():
        dice = (predecir(X_revuelto) > 0).float()
    exac = (dice == y_exa).float().mean().item()
    atrap = (dice * y_exa).sum().item()
    print(f"revolviendo {nombres[col]:10s} exactitud {exac:.3f}   atrapados {int(atrap)} de 125")
```

Salida (al final del detector, con la misma semilla):

```
revolviendo monto      exactitud 0.848   atrapados 54 de 125
revolviendo hora       exactitud 0.873   atrapados 63 de 125
revolviendo distancia  exactitud 0.870   atrapados 64 de 125
revolviendo extranjero exactitud 0.849   atrapados 57 de 125
```

Sin revolver, acertaba 90.8 % y atrapaba 81. Revolver **cualquiera** de las cuatro lo tumba: la exactitud cae a entre 84.8 % y 87.3 %, apenas arriba del flojo o igual que él, y deja de atrapar entre 17 y 27 fraudes. Las que más le dolieron: monto y extranjero.

Es compatible con la regla escondida: las cuatro columnas prenden un foco, y el modelo recibió ejemplos etiquetados producidos con esas cuatro columnas. Es evidencia de que las usa, no una prueba de que haya recuperado la regla exacta. Si tus números salieron distintos porque tu `randperm` revolvió de otra forma, está bien: lo que tiene que pasar es que las cuatro bajen.

---

## La próxima semana

En la sesión 8, la última, Mateo sale de la papelería y entra al banco para ver cómo trabajan estos mismos tensores con millones de compras, y con personas que revisan la decisión.

Si quieres adelantar, piensa: si el detector viera no una compra, sino las últimas 30 compras de cada cliente, ¿qué forma tendría su tensor de entrada?

**11:48 p.m.**

La trastienda está en silencio. Doña Lupe se quedó dormida en la silla de plástico, con el mandil puesto y la bolsa de monedas en el regazo. El pan de canela sigue junto a la caja, ya duro.

Mateo no corre el detector sobre los cobros de Norte. Lo piensa, con el dedo sobre la tecla, y no lo hace: el ejercicio aprendió con compras y una regla inventadas, y no sirve para decidir sobre personas reales. En vez de eso, saca la tabla de cuatro casillas y la pone junto a las notas que Fer anotó a mano. Ordena las preguntas: qué cargos no tienen mercancía detrás, cuáles caen de madrugada o lejos, qué evidencia puede pedirle al banco.

Debería sentirse contento, y lo está a medias. Tiene un método. Pero un método no es una prueba.

Abre el correo para escribirle al licenciado Ortega. Teclea «Le comparto un modelo que» y se detiene. Se imagina la cara del licenciado. Un muchacho de veinte años, una laptop con una calcomanía despegada, un programa que acierta 91 % y deja escapar más de uno de cada tres. Y del otro lado, un banco con sus propios sistemas, sus propios analistas, sus propias reglas.

Borra la frase.

Por la ventana pasa la última patrulla de la noche. El foco zumba. En el calendario, el círculo rojo del 23 se ve más grande que en la mañana.

Mateo cierra la laptop despacio, para no despertar a su tía. Ya sabe lo que tiene. Ya sabe también lo que le falta: no un número más alto, sino alguien del otro lado dispuesto a revisar.

Faltaban nueve días para el 23. Y el banco no iba a creerle a un modelo hecho en una laptop.

---

*Escribo desde lo que vemos todos los días en Hábil, donde construimos sistemas para banca, seguros y empresas en México. Si en tu trabajo o en el negocio de tu familia se topan con un problema así, platiquemos: [habil.mx](https://habil.mx)*
