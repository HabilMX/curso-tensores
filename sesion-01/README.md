# Sesión 1: Un número, una lista, una tabla, un cubo

Lección A del curso Tensores desde cero. La lección completa está en el Substack de Hábil: https://enconstruccionmx.substack.com

## Cómo correr un archivo

Con el entorno virtual activado (ver el README de la raíz), desde esta carpeta:

```bash
python3 nombre_del_archivo.py
```

Cada archivo tiene su salida esperada en `salida-esperada/` (las soluciones, en `soluciones/salida-esperada/`; los fragmentos, en `fragmentos/salida-esperada/`). Desde la raíz, `python verificar.py --solo sesion-01` corre y compara toda esta carpeta.

## Archivos

| Archivo | De dónde sale | Nota |
|---|---|---|
| `dimensiones.py` | código de la lección, tal cual |  |
| `soluciones/ejercicio-01.py` | código de la lección con los huecos llenados según su solución |  |
| `soluciones/ejercicio-02.py` | código de la lección, tal cual | La lección no imprime nada en este ejercicio: la salida esperada es vacía. |
| `soluciones/ejercicio-03.py` | código de la lección con los huecos llenados según su solución |  |
| `soluciones/ejercicio-04.py` | código de la lección, tal cual | La lección no imprime nada en este ejercicio: la salida esperada es vacía. |
| `soluciones/ejercicio-05.py` | código de la lección con los huecos llenados según su solución |  |

## Cómo leer la columna «De dónde sale»

- Los archivos de esta carpeta que no están en `soluciones/` ni en `fragmentos/` son los que la lección le pide crear al alumno, con el código tal cual aparece en ella. Cuando la lección los va armando por pasos («agrega al final del archivo»), aquí aparece el archivo ya armado, con los bloques en el orden de la lección.
- En `soluciones/`, el código sale de las soluciones de la lección. Donde el enunciado trae huecos (`____`), están llenados con lo que dice la solución. Las líneas entre `# [repo]` y `# [/repo]` son un preludio que se añadió para que el archivo corra solo.
- En `fragmentos/` están los bloques de la lección que no piden un nombre de archivo (errores a propósito, comparaciones) y las comprobaciones de afirmaciones de la lección escritas para el repositorio.
