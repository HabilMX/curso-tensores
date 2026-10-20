# Tensores desde cero

**Por Dorian Chávez, fundador de Hábil y arquitecto de integración.**

De la lista de Python que ya se sabe escribir a un detector de fraude que aprende de los datos.

**Para quién es:** estudiantes de Sistemas que ya programan un poco en Python, y gente de Administración que quiere
entender qué hay debajo de la inteligencia artificial que se usa en bancos, aseguradoras y tiendas. Es un curso de
nivel intermedio: no explica qué es una variable ni un bucle, pero cada palabra nueva se explica cuando aparece. Cada
sesión trae un recuadro «Si eres de Administración» con la pregunta que vale la pena hacer en una junta.

**Qué vas a terminar sabiendo:**

- Crear un tensor de PyTorch a partir de datos reales y decir su forma antes de correr el código.
- Sacar y resumir datos por cualquiera de sus ejes, y comprobar cada resultado contra el mismo cálculo hecho con un `for`.
- Leer un error de forma de PyTorch, palabra por palabra, y arreglarlo.
- Entrenar un detector de fraude pequeño y explicar por qué acertar 91 % no basta.

**Lo que necesitas antes de empezar:** Python básico (variables, listas, bucles `for` y funciones; las lecciones 0 a 4
del [curso de Python](https://github.com/HabilMX/curso-python) bastan), Python 3.11 o más nuevo y una terminal. La
primera sesión no instala nada; PyTorch se instala en la segunda. Probado en Linux (Mint 22.3 y Ubuntu 24.04) y en
macOS; en Windows no se ha probado. No hace falta tarjeta de video.

## Las ocho sesiones

Dos lecciones de cuatro sesiones, de unas 2 horas cada una: más o menos una para leer y seguir el ejemplo, y otra para
practicar. Todas siguen la misma historia: una papelería, una terminal bancaria que cobra de más y un estudiante que
tiene hasta el 23 de diciembre para entender qué pasó.

| Sesión | | Qué aprendes |
|---|---|---|
| 1 | [Un número, una lista, una tabla, un cubo](01-un-numero-una-lista-una-tabla-un-cubo.md) | qué es un tensor, con las listas de Python; contar dimensiones; de dónde salió la palabra |
| 2 | [Tu primer tensor](02-tu-primer-tensor.md) | instalar PyTorch; `torch.tensor`, `shape` y `dtype`; PyTorch y TensorFlow |
| 3 | [Los ejes tienen nombre](03-los-ejes-tienen-nombre.md) | sacar un pedazo por cualquier eje; qué le pasa a la forma |
| 4 | [Sumar por pestaña](04-sumar-por-pestana.md) | sumar y promediar sobre un eje con `dim` |
| 5 | [Cambiar de forma sin cambiar los datos](05-cambiar-de-forma.md) | `reshape` y cómo leer un error de forma |
| 6 | [Operar tensores completos](06-operar-tensores-completos.md) | broadcasting, y la trampa de `*` contra `@` |
| 7 | [Un detector de fraude que aprende de los datos](07-detector-de-fraude.md) | el ciclo de entrenamiento; exactitud, recall y precisión |
| 8 | [Tensores en el banco, la aseguradora y la tienda](08-banco-aseguradora-tienda.md) | leer un caso real y decir la forma de su tensor; los cuidados que no son opcionales |

## El código

Cada sesión tiene su carpeta en este repositorio ([`sesion-01`](../sesion-01) a [`sesion-08`](../sesion-08)), con cada
programa y la salida que debe darte. `verificar.py` los corre todos y los compara con lo que dice la lección. La
instalación paso a paso está en el [README principal](../README.md).

## Dónde se publica

Una sesión por semana en «En construcción», el Substack de Hábil, los miércoles a las 8:00, del 4 de noviembre al 23 de
diciembre de 2026: https://enconstruccionmx.substack.com
