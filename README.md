# Tensores desde cero

Curso de Hábil para aprender qué es un tensor, y para qué sirve, con Python y PyTorch. Se parte de lo que ya sabes
(listas de listas de Python) y se termina entrenando un detector de fraude pequeño y leyendo cómo usan los tensores
un banco, una aseguradora y una tienda. Está pensado para quien empieza desde cero en este tema.

Autor: Dorian Chávez, Hábil.

El hilo del curso es una papelería con dos sucursales. Cada sesión agrega una sola dificultad nueva y trae ejercicios
que se corrigen solos con `assert`. Este repositorio tiene el código de las ocho sesiones, la salida que debe dar cada
archivo y un verificador que lo comprueba.

## Las ocho sesiones

| Sesión | Carpeta | De qué trata | Archivos de Python |
|---|---|---|---|
| 1 | [`sesion-01`](sesion-01/) | Un número, una lista, una tabla, un cubo | 6 |
| 2 | [`sesion-02`](sesion-02/) | Tu primer tensor | 13 |
| 3 | [`sesion-03`](sesion-03/) | Los ejes tienen nombre | 8 |
| 4 | [`sesion-04`](sesion-04/) | Sumar por pestaña | 17 |
| 5 | [`sesion-05`](sesion-05/) | Cambiar de forma sin cambiar los datos | 15 |
| 6 | [`sesion-06`](sesion-06/) | Operar tensores completos | 11 |
| 7 | [`sesion-07`](sesion-07/) | Un detector de fraude que aprende de los datos | 13 |
| 8 | [`sesion-08`](sesion-08/) | Tensores en el banco, la aseguradora y la tienda | 10 |

Las dos lecciones del curso son la A (sesiones 1 a 4: de la lista al cubo) y la B (sesiones 5 a 8: los tensores
trabajando). Las lecciones completas, con la historia, las explicaciones y los ejercicios, se publican por entregas en
el Substack de Hábil: https://enconstruccionmx.substack.com

Requisito de conocimientos: lecciones 0 a 4 del curso de Python de Hábil (instalación, bucles y funciones, colecciones).

## Requisitos

- Python **3.11 o más nuevo** (probado con 3.12), con **PyTorch 2.14.1** y **torchvision 0.29.1** (versiones fijas: con otras, los números pueden
  cambiar un poco y no sabrías si te equivocaste tú). Desde PyTorch 2.15 el mínimo es Python 3.11.
- Internet para descargar PyTorch (la edición solo CPU pesa menos que la de tarjeta de video).
- Una terminal. Probado en Linux (Mint 22.3 y Ubuntu 24.04) y en macOS.

Todo corre en la CPU de una laptop normal. No hace falta tarjeta de video.

## Instalación

El entorno virtual es una carpeta donde viven las bibliotecas de este curso, separadas de las del resto de tu
computadora. Se instala una vez; en cada terminal nueva hay que activarlo otra vez.

### Linux (Mint, Ubuntu, Debian)

```bash
python3 --version
sudo apt install python3-venv
python3 -m venv ~/tensores
source ~/tensores/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

La línea de `sudo apt` es necesaria en una instalación nueva: el paquete que crea entornos viene aparte. Sin él,
`python3 -m venv` falla con código 1 y un mensaje sobre `ensurepip` que propone `apt install python3.12-venv`; con
`sudo apt install python3-venv` funciona (comprobado en Ubuntu 24.04 limpio, con Python 3.12.3). Si ya lo tenías,
no pasa nada. Cuando el entorno está activo, la terminal empieza con `(tensores)`.

### macOS

```bash
python3 --version
python3 -m venv ~/tensores
source ~/tensores/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

Si `python3 --version` marca una versión anterior a 3.11, instala una más nueva desde [python.org](https://www.python.org/downloads/).

### Windows

El curso no se ha probado en Windows. Según la [documentación de Python](https://docs.python.org/3/library/venv.html),
en PowerShell el entorno se crea con `python -m venv tensores` y se activa con `.\tensores\Scripts\activate`, y los
comandos que aquí dicen `python3` se escriben `python`. Si una salida no coincide, compárala con la de
`salida-esperada/` antes de suponer que te equivocaste.

### Comprobar que quedó

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Debe imprimir `2.14.1 False` (o `2.14.1+cpu False`). El `False` es lo esperado: instalaste la edición solo CPU.

## Cómo correr un archivo

Entra a la carpeta de la sesión y corre el archivo:

```bash
cd sesion-04
python3 paso2.py
```

La salida esperada de cada archivo está en `salida-esperada/<archivo>.txt`, dentro de su carpeta.

## Cómo correr el verificador

`verificar.py` corre cada archivo de Python de todas las sesiones (los de la lección, las soluciones y los fragmentos),
compara lo que imprime con su salida esperada y te dice cuántos corrió de cuántos. Termina con código distinto de 0
si algo no coincide.

```bash
python3 verificar.py                 # todo (93 archivos y los ejemplos interactivos)
python3 verificar.py --solo sesion-07  # solo una carpeta o un archivo
python3 verificar.py --estricto      # sin ninguna tolerancia numérica
python3 verificar.py --autoprueba    # comprueba que el verificador se pone en rojo cuando debe
python3 verificar.py --revisar-md    # títulos sin emojis y sin palabras vetadas
python3 verificar.py --revisar-secretos
python3 verificar.py --lista         # el manifiesto: qué se corre y con qué reglas
```

Lo único que el verificador tolera está documentado en la cabecera de `verificar.py` y, archivo por archivo, en
`manifiesto.json`: el cuerpo de los mensajes de error (rutas y números de línea), el sufijo `+cpu` de la versión, y una
tolerancia numérica pequeña en los pocos programas que dependen de una semilla aleatoria (`detector.py` y sus variantes,
`embedding.py`).

`DIFERENCIAS.md` anota todo lo que, al correr el código con las versiones fijadas, no coincide con lo que dice la lección.

## Qué hay en el repositorio

- `sesion-NN/`: el código de cada sesión. Los archivos que pide la lección están en la carpeta; las soluciones de los
  ejercicios, en `soluciones/`; los bloques sueltos y las comprobaciones de afirmaciones de la lección, en `fragmentos/`.
- `ejemplos-interactivos.md`: los ejemplos de la lección que se escriben en el intérprete (`>>>`), que el verificador
  también corre.
- `verificar.py`: el verificador, con autoprueba.
- `medir_palabras.py`: el medidor de palabras de una lección (cuenta la prosa, sin bloques de código ni frontmatter;
  la regla está en su cabecera).
- `manifiesto.json`: la lista de archivos que se corren, con su salida esperada y las reglas de comparación.
- `CI.md`: lo que debe comprobar la integración continua.
- `DIFERENCIAS.md`: diferencias entre lo que dice la lección y lo que da el código.

## Licencia

Pendiente: la decide Hábil. Ver `LICENSE-PENDIENTE.md`.
