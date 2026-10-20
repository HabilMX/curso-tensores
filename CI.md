# Integración continua

Lo que la integración continua (CI) de este repositorio debe comprobar en cada cambio. Si cualquiera de estas
comprobaciones falla, el cambio no se publica.

## Entorno

- Python 3.12 como versión principal. Opcional, en matriz: 3.11 y 3.13 (el curso pide 3.11 o más nuevo).
- En Linux nuevo: `sudo apt install python3-venv` antes de crear el entorno virtual.
- Instalar con `pip install -r requirements.txt` (trae PyTorch 2.14.1 y torchvision 0.29.1, edición solo CPU). La
  descarga es grande: conviene guardar en caché la carpeta de pip.

## La lista exacta

| # | Qué se comprueba | Comando | Debe pasar |
|---|---|---|---|
| 1 | Todos los archivos de Python (lección, soluciones, fragmentos) dan la salida esperada, y los ejemplos interactivos también | `python verificar.py` | código de salida 0; el resumen dice «Scripts corridos: N de N» con los dos N iguales y «RESULTADO: TODO EN VERDE» |
| 2 | El verificador se pone en rojo cuando debe (mete diferencias a propósito en una copia y comprueba que las detecta) | `python verificar.py --autoprueba` | código de salida 0 y «AUTOPRUEBA SUPERADA» |
| 3 | Ningún título de un `.md` lleva emoji y ningún archivo de texto trae palabras vetadas | `python verificar.py --revisar-md` | código de salida 0 |
| 4 | No hay secretos en el repositorio (llaves, tokens, contraseñas asignadas) | `python verificar.py --revisar-secretos` | código de salida 0 |

Notas de cada comprobación:

1. **Verificador.** Además de correr los archivos, falla si hay un `.py` en `sesion-NN/` que no esté en
   `manifiesto.json` (un archivo nuevo sin salida esperada no pasa). Hoy el manifiesto tiene 93 archivos.
   La CI no debe usar `--estricto`: las tolerancias numéricas (solo en los programas con semilla) existen porque Linux
   y macOS pueden dar los últimos decimales distintos. En una máquina de desarrollo, correr también
   `python verificar.py --estricto` sirve para ver si hubo corrimiento.
2. **Autoprueba.** Un verificador que nunca falla no sirve. La autoprueba trabaja en una copia temporal y no toca el repositorio.
3. **Títulos y palabras vetadas.** Los títulos son las líneas que empiezan con `#` fuera de bloques de código. La lista
   de palabras vetadas vive en `verificar.py` (`VETADAS`).
4. **Secretos.** Los patrones viven en `verificar.py` (`SECRETOS`). Si aparece un falso positivo, se corrige el
   texto; no se agrega una excepción.

## Ejemplo para GitLab CI

```yaml
verificar:
  image: python:3.12
  script:
    - python -m venv .entorno
    - . .entorno/bin/activate
    - pip install -r requirements.txt
    - python verificar.py
    - python verificar.py --autoprueba
    - python verificar.py --revisar-md
    - python verificar.py --revisar-secretos
```

## Ejemplo para GitHub Actions

```yaml
name: verificar
on: [push, pull_request]
jobs:
  verificar:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: pip
      - run: pip install -r requirements.txt
      - run: python verificar.py
      - run: python verificar.py --autoprueba
      - run: python verificar.py --revisar-md
      - run: python verificar.py --revisar-secretos
```

## Qué no cubre

- Los bloques de la lección que usan TensorFlow (sesión 4) no se corren: TensorFlow no es requisito del curso.
  Están en `ejemplos-interactivos.md`, marcados como no verificados por el verificador.
- Las comparaciones dicen que el código da lo que la lección afirma; no juzgan si la lección explica bien.
- Los mensajes de error se comparan por su primera y última línea (el cuerpo del Traceback cambia con la versión de Python).
