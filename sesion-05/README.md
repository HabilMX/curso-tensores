# Sesión 5: Cambiar de forma sin cambiar los datos

Lección B del curso Tensores desde cero. La lección completa está en el Substack de Hábil: https://enconstruccionmx.substack.com

## Cómo correr un archivo

Con el entorno virtual activado (ver el README de la raíz), desde esta carpeta:

```bash
python3 nombre_del_archivo.py
```

Cada archivo tiene su salida esperada en `salida-esperada/` (las soluciones, en `soluciones/salida-esperada/`; los fragmentos, en `fragmentos/salida-esperada/`). Desde la raíz, `python verificar.py --solo sesion-05` corre y compara toda esta carpeta.

## Archivos

| Archivo | De dónde sale | Nota |
|---|---|---|
| `aplanar.py` | código de la lección, tal cual |  |
| `cambiar_forma.py` | código de la lección, tal cual | Archivo acumulado: pasos 1 a 4 y 6 de la lección, en orden (el paso 3 incluye su comprobación con for). |
| `cambiar_forma_error.py` | bloque de la lección armado con líneas añadidas (marcadas con [repo]) | Paso 7 de la lección: cambiar_forma.py completo más la línea quintos = ventas.reshape(5, 3) al final (sin ningún comentario añadido, para que el error caiga en la línea 42 que cita la lección). |
| `foto.py` | código de la lección, tal cual |  |
| `fragmentos/errores_comunes.py` | escrito para este repositorio | Comprueba el paso 6 y la tabla «Errores comunes» de la lección 5. El mensaje de ventas[3, 2] no lo cita la lección (solo dice que truena): se compara el tipo de error. |
| `girar.py` | código de la lección, tal cual | Archivo acumulado: el bloque de girar.py y el del paso 9 (.T) que se agrega al final. |
| `papeleria.py` | código de la lección, tal cual | Copia del papeleria.py de la sesión 4 (la lección 5 lo pide en la misma carpeta). |
| `soluciones/ejercicio-18.py` | escrito para este repositorio | Escrito para el repositorio; las líneas esperadas son la tabla de la solución de la lección. |
| `soluciones/ejercicio-19.py` | código de la lección con los huecos llenados según su solución |  |
| `soluciones/ejercicio-19c.py` | código de la lección con los huecos llenados según su solución |  |
| `soluciones/ejercicio-20a.py` | código de la lección, tal cual | La lección da la última línea del error (solución, inciso a); el cuerpo del Traceback se elide. |
| `soluciones/ejercicio-20b.py` | código de la lección, tal cual |  |
| `soluciones/ejercicio-20c.py` | código de la lección, tal cual | La lección dice que el primer renglón es tensor([2500, 3100, 2200, 2800]) (solución, inciso c). |
| `soluciones/ejercicio-21-trampa.py` | código de la lección con los huecos llenados según su solución | La lección dice que truena el assert del renglón de cuadernos; el mensaje es el del assert de la solución. |
| `soluciones/ejercicio-21.py` | código de la lección, tal cual |  |

## Cómo leer la columna «De dónde sale»

- Los archivos de esta carpeta que no están en `soluciones/` ni en `fragmentos/` son los que la lección le pide crear al alumno, con el código tal cual aparece en ella. Cuando la lección los va armando por pasos («agrega al final del archivo»), aquí aparece el archivo ya armado, con los bloques en el orden de la lección.
- En `soluciones/`, el código sale de las soluciones de la lección. Donde el enunciado trae huecos (`____`), están llenados con lo que dice la solución. Las líneas entre `# [repo]` y `# [/repo]` son un preludio que se añadió para que el archivo corra solo.
- En `fragmentos/` están los bloques de la lección que no piden un nombre de archivo (errores a propósito, comparaciones) y las comprobaciones de afirmaciones de la lección escritas para el repositorio.
