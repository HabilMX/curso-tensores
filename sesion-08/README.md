# Sesión 8: Tensores en el banco, la aseguradora y la tienda

Lección B del curso Tensores desde cero. La lección completa está en el Substack de Hábil: https://enconstruccionmx.substack.com

## Cómo correr un archivo

Con el entorno virtual activado (ver el README de la raíz), desde esta carpeta:

```bash
python3 nombre_del_archivo.py
```

Cada archivo tiene su salida esperada en `salida-esperada/` (las soluciones, en `soluciones/salida-esperada/`; los fragmentos, en `fragmentos/salida-esperada/`). Desde la raíz, `python verificar.py --solo sesion-08` corre y compara toda esta carpeta.

## Archivos

| Archivo | De dónde sale | Nota |
|---|---|---|
| `casos.py` | código de la lección, tal cual |  |
| `embedding.py` | código de la lección, tal cual |  |
| `fragmentos/errores_comunes.py` | escrito para este repositorio | Comprueba el paso 5 y la tabla «Errores comunes» de la lección 8. |
| `fragmentos/orden_colores.py` | bloque de la lección armado con líneas añadidas (marcadas con [repo]) |  |
| `soluciones/ejercicio-30.py` | código de la lección, tal cual |  |
| `soluciones/ejercicio-31.py` | código de la lección con los huecos llenados según su solución |  |
| `soluciones/ejercicio-32.py` | código de la lección con los huecos llenados según su solución |  |
| `soluciones/ejercicio-33.py` | código de la lección, tal cual |  |
| `soluciones/repaso-leccion-b.py` | escrito para este repositorio | Escrito para el repositorio; las líneas esperadas son las «salidas reales» que da la lección. |
| `tienda.py` | código de la lección, tal cual |  |

## Cómo leer la columna «De dónde sale»

- Los archivos de esta carpeta que no están en `soluciones/` ni en `fragmentos/` son los que la lección le pide crear al alumno, con el código tal cual aparece en ella. Cuando la lección los va armando por pasos («agrega al final del archivo»), aquí aparece el archivo ya armado, con los bloques en el orden de la lección.
- En `soluciones/`, el código sale de las soluciones de la lección. Donde el enunciado trae huecos (`____`), están llenados con lo que dice la solución. Las líneas entre `# [repo]` y `# [/repo]` son un preludio que se añadió para que el archivo corra solo.
- En `fragmentos/` están los bloques de la lección que no piden un nombre de archivo (errores a propósito, comparaciones) y las comprobaciones de afirmaciones de la lección escritas para el repositorio.
