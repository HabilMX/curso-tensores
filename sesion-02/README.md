# Sesión 2: Tu primer tensor

Lección A del curso Tensores desde cero. La lección completa está en el Substack de Hábil: https://enconstruccionmx.substack.com

## Cómo correr un archivo

Con el entorno virtual activado (ver el README de la raíz), desde esta carpeta:

```bash
python3 nombre_del_archivo.py
```

Cada archivo tiene su salida esperada en `salida-esperada/` (las soluciones, en `soluciones/salida-esperada/`; los fragmentos, en `fragmentos/salida-esperada/`). Desde la raíz, `python verificar.py --solo sesion-02` corre y compara toda esta carpeta.

## Archivos

| Archivo | De dónde sale | Nota |
|---|---|---|
| `chueca.py` | código de la lección, tal cual | Provoca el error a propósito. El verificador compara el encabezado del Traceback y la última línea; el cuerpo (rutas, números de línea, marcas ^^^^) varía con la versión de Python. |
| `compara.py` | código de la lección, tal cual |  |
| `dimensiones.py` | código de la lección, tal cual | Copia del dimensiones.py de la sesión 1: la lección 2 dice que se importa desde la misma carpeta. |
| `fragmentos/dtype_elegido.py` | bloque de la lección armado con líneas añadidas (marcadas con [repo]) | Bloque de la lección que no pide nombre de archivo; se corre tal cual. |
| `fragmentos/errores_comunes.py` | escrito para este repositorio | Comprueba la tabla «Errores comunes» de la lección 2. |
| `papeleria.py` | código de la lección, tal cual |  |
| `primer_tensor.py` | código de la lección, tal cual |  |
| `soluciones/ejercicio-06.py` | código de la lección, tal cual |  |
| `soluciones/ejercicio-07.py` | código de la lección con los huecos llenados según su solución |  |
| `soluciones/ejercicio-08.py` | código de la lección, tal cual |  |
| `soluciones/ejercicio-09.py` | escrito para este repositorio | Escrito para el repositorio; las líneas esperadas son las frases de la solución de la lección. |
| `tipos.py` | código de la lección, tal cual |  |
| `verifica.py` | código de la lección, tal cual | La lección acepta 2.14.1+cpu (ediciones solo CPU de Linux/Windows): el verificador quita el sufijo +cpu. |

## Cómo leer la columna «De dónde sale»

- Los archivos de esta carpeta que no están en `soluciones/` ni en `fragmentos/` son los que la lección le pide crear al alumno, con el código tal cual aparece en ella. Cuando la lección los va armando por pasos («agrega al final del archivo»), aquí aparece el archivo ya armado, con los bloques en el orden de la lección.
- En `soluciones/`, el código sale de las soluciones de la lección. Donde el enunciado trae huecos (`____`), están llenados con lo que dice la solución. Las líneas entre `# [repo]` y `# [/repo]` son un preludio que se añadió para que el archivo corra solo.
- En `fragmentos/` están los bloques de la lección que no piden un nombre de archivo (errores a propósito, comparaciones) y las comprobaciones de afirmaciones de la lección escritas para el repositorio.
