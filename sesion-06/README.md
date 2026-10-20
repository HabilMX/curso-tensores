# Sesión 6: Operar tensores completos

Lección B del curso Tensores desde cero. La lección completa está en el Substack de Hábil: https://enconstruccionmx.substack.com

## Cómo correr un archivo

Con el entorno virtual activado (ver el README de la raíz), desde esta carpeta:

```bash
python3 nombre_del_archivo.py
```

Cada archivo tiene su salida esperada en `salida-esperada/` (las soluciones, en `soluciones/salida-esperada/`; los fragmentos, en `fragmentos/salida-esperada/`). Desde la raíz, `python verificar.py --solo sesion-06` corre y compara toda esta carpeta.

## Archivos

| Archivo | De dónde sale | Nota |
|---|---|---|
| `fragmentos/errores_comunes.py` | escrito para este repositorio | Comprueba la tabla «Errores comunes» de la lección 6. |
| `fragmentos/listas_vs_tensores.py` | bloque de la lección armado con líneas añadidas (marcadas con [repo]) |  |
| `operar.py` | código de la lección, tal cual | Archivo acumulado: pasos 1, 2, 3, 5 y 6 de la lección, en orden. Las líneas que provocan errores a propósito (pasos 4 y 6) están en operar_error_colores.py y operar_error_arroba.py. |
| `operar_error_arroba.py` | bloque de la lección armado con líneas añadidas (marcadas con [repo]) | Paso 6 de la lección: el orden de @ importa. |
| `operar_error_colores.py` | bloque de la lección armado con líneas añadidas (marcadas con [repo]) | Paso 4 de la lección. |
| `soluciones/ejercicio-22a.py` | código de la lección, tal cual | La última línea falla a propósito (el que engaña). |
| `soluciones/ejercicio-22b.py` | código de la lección, tal cual |  |
| `soluciones/ejercicio-23.py` | código de la lección con los huecos llenados según su solución |  |
| `soluciones/ejercicio-24.py` | código de la lección, tal cual |  |
| `soluciones/ejercicio-25.py` | código de la lección, tal cual |  |
| `trampa_cuadrada.py` | código de la lección, tal cual | Paso 7 de la lección: los dos bloques (la trampa y la defensa con for), en orden. |

## Cómo leer la columna «De dónde sale»

- Los archivos de esta carpeta que no están en `soluciones/` ni en `fragmentos/` son los que la lección le pide crear al alumno, con el código tal cual aparece en ella. Cuando la lección los va armando por pasos («agrega al final del archivo»), aquí aparece el archivo ya armado, con los bloques en el orden de la lección.
- En `soluciones/`, el código sale de las soluciones de la lección. Donde el enunciado trae huecos (`____`), están llenados con lo que dice la solución. Las líneas entre `# [repo]` y `# [/repo]` son un preludio que se añadió para que el archivo corra solo.
- En `fragmentos/` están los bloques de la lección que no piden un nombre de archivo (errores a propósito, comparaciones) y las comprobaciones de afirmaciones de la lección escritas para el repositorio.
