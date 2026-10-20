# Sesión 7: Un detector de fraude que aprende solo

Lección B del curso Tensores desde cero. La lección completa está en el Substack de Hábil: https://enconstruccionmx.substack.com

## Cómo correr un archivo

Con el entorno virtual activado (ver el README de la raíz), desde esta carpeta:

```bash
python3 nombre_del_archivo.py
```

Cada archivo tiene su salida esperada en `salida-esperada/` (las soluciones, en `soluciones/salida-esperada/`; los fragmentos, en `fragmentos/salida-esperada/`). Desde la raíz, `python verificar.py --solo sesion-07` corre y compara toda esta carpeta.

## Archivos

| Archivo | De dónde sale | Nota |
|---|---|---|
| `cerro.py` | código de la lección, tal cual |  |
| `detector.py` | código de la lección, tal cual | Bloques 1 a 7 de la lección en orden, con el bloque de la pérdida y el optimizador antes del ciclo. La lección dice 61 líneas, 41 sin comentarios ni líneas en blanco; este archivo tiene 61 y 41. |
| `fragmentos/cerro_sin_zero_grad.py` | escrito para este repositorio | Comprueba la fila «Olvidar borrar las pendientes» de la lección 7. Ver DIFERENCIAS.md: con los assert de cerro.py puestos, el programa SÍ truena. |
| `fragmentos/detector_inspeccion.py` | bloque de la lección armado con líneas añadidas (marcadas con [repo]) | Paso 3 de la lección 7: «Para verlo de cerca, agrega print(X[:3]) y print(y[:3])». |
| `fragmentos/detector_lr10_diagnostico.py` | bloque de la lección armado con líneas añadidas (marcadas con [repo]) | Solución del ejercicio 27, con lr=10: las 25,600 señales en cero y el puntaje constante de -1.67. |
| `fragmentos/detector_sin_grad_en_W1.py` | escrito para este repositorio | Comprueba la fila «Olvidar requires_grad=True» de la lección 7: la pérdida se atora en 0.3366 y atrapa 13 de 125 (los 112 que se escapan y las 0 alarmas falsas son lo que sale del mismo conteo; la lección no los cita). |
| `fragmentos/detector_tiempo.py` | bloque de la lección armado con líneas añadidas (marcadas con [repo]) | Tiempo: no determinista. Solo se comprueba el umbral de un segundo, no el valor. |
| `fragmentos/errores_comunes.py` | escrito para este repositorio | Comprueba la tabla «Errores comunes» de la lección 7. La lección cierra el mensaje de backward con un punto; se compara por contenido. |
| `soluciones/ejercicio-26.py` | código de la lección con los huecos llenados según su solución | La lección pega solo desde la época 300; el verificador exige esas líneas en orden. |
| `soluciones/ejercicio-27-lr0.1.py` | código de la lección con los huecos llenados según su solución | La lección recorta algunas épocas de en medio; el verificador exige esas líneas en orden. |
| `soluciones/ejercicio-27-lr10.py` | código de la lección con los huecos llenados según su solución | Falla a propósito: 0 alarmas y la precisión divide entre cero. |
| `soluciones/ejercicio-28.py` | código de la lección con los huecos llenados según su solución | Las dos últimas líneas esperadas no se imprimen en la lección; salen del assert de la lección (0.648 y 0.73). |
| `soluciones/ejercicio-29.py` | código de la lección, tal cual | La lección dice «al final del detector, con la misma semilla». |

## Cómo leer la columna «De dónde sale»

- Los archivos de esta carpeta que no están en `soluciones/` ni en `fragmentos/` son los que la lección le pide crear al alumno, con el código tal cual aparece en ella. Cuando la lección los va armando por pasos («agrega al final del archivo»), aquí aparece el archivo ya armado, con los bloques en el orden de la lección.
- En `soluciones/`, el código sale de las soluciones de la lección. Donde el enunciado trae huecos (`____`), están llenados con lo que dice la solución. Las líneas entre `# [repo]` y `# [/repo]` son un preludio que se añadió para que el archivo corra solo.
- En `fragmentos/` están los bloques de la lección que no piden un nombre de archivo (errores a propósito, comparaciones) y las comprobaciones de afirmaciones de la lección escritas para el repositorio.
