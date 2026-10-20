# Diferencias entre la lección y el código

Se corrió todo con Python 3.12.12 en macOS, `torch==2.14.1` y `torchvision==0.29.1` (edición CPU, instalada con el
comando de la lección 2), en un entorno virtual nuevo. Regla seguida: si lo obtenido difiere de la lección, no se
cambia nada del código ni de la salida esperada; se anota aquí.

Resultado: 93 archivos de Python corridos, 93 coinciden con la lección, también con `python verificar.py --estricto`
(sin ninguna tolerancia numérica). Los dos ejemplos interactivos coinciden. La única diferencia de contenido es la
primera de esta lista.

## Diferencias

> **Estado:** la diferencia 1 quedó resuelta en la lección 7 v0.5.2 (19-oct-2026), que ahora dice lo que se obtuvo. La observación de la versión de Python quedó alineada: el README pide 3.11 o más nuevo, como la lección 2.

### 1. Olvidar `zero_grad` en `cerro.py`: la lección dice «no truena»; con el `assert` puesto, sí truena

- **Sesión:** 7 (tabla «Errores comunes», fila «Olvidar borrar las pendientes»).
- **Esperado (lección):** «No truena. En `cerro.py` el precio salta a 37.80, luego a 41.58, la pérdida sube hasta 4,191.27
  (en el paso 4) y, al terminar, el precio aprendido queda en 4.95 en lugar de 20».
- **Obtenido:** si se borra solo la línea `precio.grad.zero_()` de `cerro.py` tal como está en la lección, el programa
  se detiene en el paso 2 con `AssertionError: PyTorch y el lápiz deberían coincidir`, porque el `assert` de `a_mano`
  compara la pendiente acumulada contra la del lápiz. Los números que cita la lección (37.80, 41.58, 4191.27 y 4.95) salen
  exactamente, pero solo si además se quitan los dos `assert` de `cerro.py`. Comprobado en
  `sesion-07/fragmentos/cerro_sin_zero_grad.py`, que tiene ambas cosas quitadas y coincide con esos cuatro números.

## Observaciones que no son diferencias de salida

- **Versión mínima de Python.** La lección 2 dice «Necesitas 3.11 o más nuevo: es lo que pide PyTorch desde su versión
  2.15». Con la versión fijada del curso (2.14.1) también funciona Python 3.10; el README de este repositorio dice 3.10 a 3.13
  y que desde PyTorch 2.15 el mínimo es 3.11.
- **Número de línea del error de `cambiar_forma.py`.** La lección cita `line 42`. En `sesion-05/cambiar_forma_error.py`
  (el archivo acumulado más la línea `quintos = ventas.reshape(5, 3)`, sin ningún comentario añadido) el error cae en la
  línea 42. Coincide. El verificador no compara el cuerpo de los Traceback; esto se comprobó aparte.
- **Líneas del `detector.py`.** La lección dice 61 líneas, 41 sin comentarios ni líneas en blanco. El archivo armado con los
  bloques de la lección tiene 61 y 41.
- **Salidas de la lección 7 que dependen de la semilla.** `detector.py` y sus variantes (ejercicios 26 a 29, `lr=0.1`,
  `lr=10`) dieron en esta máquina exactamente los números de la lección. El manifiesto les da una tolerancia pequeña
  (decimales 0.03, enteros 6; en el ejercicio 29, 0.05 y 12) porque la propia lección avisa que PyTorch no garantiza números
  idénticos entre computadoras; `embedding.py` tiene tolerancia de 0.001. Con `--estricto` se apaga.
- **Tiempos (no deterministas).** La lección dice que las 400 épocas tardan «menos de una décima de segundo»: aquí
  tardaron 0.06 s. El verificador solo comprueba que quepan en un segundo (`sesion-07/fragmentos/detector_tiempo.py`).
- **Versión de `torch`.** En Linux y Windows la edición CPU se reporta como `2.14.1+cpu`; en macOS, como `2.14.1`. La lección
  acepta las dos; el verificador quita el sufijo `+cpu` en `sesion-02/verifica.py`.
- **`papeleria.py` tiene dos versiones.** La lección 2 lo crea importando `centro` y `norte` de `dimensiones.py`; la 4 lo
  da autocontenido. Cada carpeta lleva la versión de su lección y las dos exportan `papeleria`, `centro` y `norte`.

## Afirmaciones de la lección comprobadas que sí coinciden

Además de los archivos que pide la lección, se comprobaron los mensajes y cifras que cita en prosa y en las tablas
«Errores comunes» (`fragmentos/errores_comunes.py` de las sesiones 2 a 8, y en la sesión 7 `detector_inspeccion.py`,
`detector_lr10_diagnostico.py` y `detector_sin_grad_en_W1.py`). Coincidieron todos: por ejemplo, las 25,600 señales en
cero y el puntaje constante de -1.67 con `lr=10`, y la pérdida de 0.3353 con 14 fraudes atrapados al olvidar
`requires_grad` en `W1`.

## No verificado por el verificador

- **TensorFlow (sesión 4, y menciones en las sesiones 5 y 6).** No es requisito del curso. Se probaron una vez, aparte, con
  TensorFlow 2.21.0 en otro entorno virtual: las tres salidas del recuadro de la sesión 4 (`[18500  4600]`, `[2642  657]` con
  `int32` y `[2642.8572  657.1429]`) coincidieron; el mensaje de `tf.reshape` («Input to reshape is a tensor with 12 values,
  but the requested shape has 15») y el de `@` con una tabla y una lista («different ndims») también. Una nota: el bloque de la
  lección 4 hace `from papeleria import centro, norte`, y `papeleria.py` importa `torch`, así que necesita PyTorch instalado
  en el mismo entorno (lo normal en el curso).
- **Comandos de terminal** (instalación, `python3 -i cambiar_forma.py`): no se corren. `python3 -c "from papeleria import
  papeleria; print(papeleria.shape)"` se probó a mano y dio `torch.Size([2, 7, 3])`.
- **Linux.** Todo se corrió en macOS. Según el aviso del coordinador, una corrida en Ubuntu 24.04 limpio (Python 3.12.3,
  torch 2.14.1+cpu) fue aparte; esta carpeta no la repite.
