#!/usr/bin/env python3
"""Verificador del curso «Tensores desde cero» (Hábil).

Qué hace
--------
1. Corre cada script de las carpetas sesion-NN/ (los de la lección, las soluciones y los
   fragmentos) con el mismo Python que corre este verificador, y compara su salida
   (stdout y stderr juntos) con el archivo de salida-esperada/ que le toca.
2. Corre los ejemplos interactivos (>>>) de ejemplos-interactivos.md con doctest.
3. Termina con código 0 solo si TODO coincide; con 1 si algo falla; con 2 si faltan
   requisitos (por ejemplo, PyTorch).

Qué tolera (y nada más)
-----------------------
La lista completa vive en manifiesto.json, una entrada por script. Las únicas tolerancias son:

* Tracebacks: se compara el encabezado «Traceback (most recent call last):» y la última
  línea (el error). El cuerpo (rutas, números de línea, marcas ^^^^) cambia con la versión
  de Python y con la ruta de cada computadora, y la propia lección lo elide con «...».
* Espacios al final de cada línea y líneas en blanco al final del texto.
* "normalizar": expresiones regulares que se aplican a lo OBTENIDO antes de comparar
  (hoy solo se usa para aceptar «2.14.1+cpu» donde la lección dice «2.14.1»).
* "modo": "subsecuencia": la lección pega solo algunas líneas de la salida; cada línea
  esperada debe aparecer, en orden, como línea completa de lo obtenido. "contiene": igual,
  pero basta con que la línea esperada sea un pedazo de una línea obtenida (la lección cita
  mensajes de error cortados con «...»).
* "tolerancia": solo en los scripts que dependen de una semilla aleatoria. Los números con
  punto decimal pueden diferir hasta "decimales_abs" y los enteros hasta "enteros_abs";
  el texto alrededor de los números tiene que ser idéntico. Con --estricto se apaga.

Uso
---
    python verificar.py                  # todo
    python verificar.py --estricto       # sin tolerancia numérica
    python verificar.py --solo sesion-07 # solo una carpeta o un script (puede repetirse)
    python verificar.py --autoprueba     # comprueba que el verificador SÍ se pone en rojo
    python verificar.py --revisar-md     # títulos sin emojis y sin palabras vetadas
    python verificar.py --revisar-secretos
    python verificar.py --lista          # muestra el manifiesto
"""
import argparse
import concurrent.futures
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
VERSION_TORCH = "2.14.1"
ENCABEZADO_TB = "Traceback (most recent call last):"
NUM = re.compile(r"-?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------- texto

def limpiar(texto):
    lineas = [l.rstrip() for l in texto.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lineas and lineas[-1] == "":
        lineas.pop()
    return lineas


def colapsar_traceback(lineas):
    """Deja de cada Traceback solo el encabezado, una línea «  ...» y la línea del error."""
    salida = []
    i = 0
    while i < len(lineas):
        if lineas[i].startswith(ENCABEZADO_TB):
            salida.append(lineas[i])
            j = i + 1
            while j < len(lineas) and (lineas[j].startswith(" ") or lineas[j] == ""):
                j += 1
            salida.append("  ...")
            i = j
        else:
            salida.append(lineas[i])
            i += 1
    return salida


def linea_igual(esp, obt, tol):
    if esp == obt:
        return True
    if not tol:
        return False
    ne, no = NUM.findall(esp), NUM.findall(obt)
    if len(ne) != len(no) or not ne:
        return False
    if NUM.sub("#", esp) != NUM.sub("#", obt):
        return False
    for x, y in zip(ne, no):
        decimal = "." in x or "." in y
        limite = tol.get("decimales_abs", 0) if decimal else tol.get("enteros_abs", 0)
        if abs(float(x) - float(y)) > limite + 1e-12:
            return False
    return True


def comparar(esperado, obtenido, modo, tol):
    """Devuelve la lista de problemas (vacía si coincide)."""
    if modo == "contiene":
        pos = 0
        for e in esperado:
            while pos < len(obtenido) and e not in obtenido[pos]:
                pos += 1
            if pos == len(obtenido):
                return [f"no aparece, en orden, dentro de alguna línea: {e!r}"]
            pos += 1
        return []
    if modo == "subsecuencia":
        pos = 0
        for e in esperado:
            while pos < len(obtenido) and not linea_igual(e, obtenido[pos], tol):
                pos += 1
            if pos == len(obtenido):
                return [f"no aparece, en orden, la línea esperada: {e!r}"]
            pos += 1
        return []
    problemas = []
    if len(esperado) != len(obtenido):
        problemas.append(f"líneas esperadas {len(esperado)}, obtenidas {len(obtenido)}")
    for n, (e, o) in enumerate(zip(esperado, obtenido), 1):
        if not linea_igual(e, o, tol):
            problemas.append(f"línea {n}: esperaba {e!r}, obtuve {o!r}")
    return problemas


# ------------------------------------------------------------------------ ejecución

def entorno(sesion_dir):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    previo = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(sesion_dir) + (os.pathsep + previo if previo else "")
    return env


def correr_script(raiz, entrada, python):
    ruta = raiz / entrada["ruta"]
    sesion_dir = raiz / Path(entrada["ruta"]).parts[0]
    try:
        p = subprocess.run([python, str(ruta)], cwd=ruta.parent, env=entorno(sesion_dir),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    except subprocess.TimeoutExpired:
        return None, "se pasó de 300 segundos"
    return p, None


def evaluar(raiz, entrada, python, estricto):
    """Corre un script y lo compara. Devuelve (ok, detalle)."""
    ruta = entrada["ruta"]
    esp_ruta = raiz / entrada["salida"]
    if not (raiz / ruta).exists():
        return False, "el script no existe"
    if not esp_ruta.exists():
        return False, f"falta el archivo de salida esperada {entrada['salida']}"
    p, error = correr_script(raiz, entrada, python)
    if error:
        return False, error
    obtenido = p.stdout.decode("utf-8", errors="replace")
    for patron, reemplazo in entrada.get("normalizar", []):
        obtenido = re.sub(patron, reemplazo, obtenido)
    esperado = colapsar_traceback(limpiar(esp_ruta.read_text(encoding="utf-8")))
    obtenido_l = colapsar_traceback(limpiar(obtenido))
    tol = None if estricto else entrada.get("tolerancia")
    problemas = comparar(esperado, obtenido_l, entrada.get("modo", "exacto"), tol)
    if p.returncode != entrada.get("codigo_salida", 0):
        problemas.append(f"código de salida {p.returncode}, esperaba {entrada.get('codigo_salida', 0)}")
    if not problemas:
        return True, ""
    detalle = list(problemas[:6])
    if entrada.get("modo", "exacto") == "exacto":
        diff = list(difflib.unified_diff(esperado, obtenido_l, "esperado", "obtenido", lineterm="", n=1))
        detalle += diff[:24]
    else:
        detalle += ["--- obtenido ---"] + obtenido_l[:30]
    return False, "\n      ".join(detalle)


def cargar_manifiesto(raiz):
    return json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))["scripts"]


def scripts_en_disco(raiz):
    return sorted(str(p.relative_to(raiz)).replace(os.sep, "/")
                  for d in raiz.glob("sesion-*") for p in d.rglob("*.py")
                  if "__pycache__" not in p.parts)


def seleccionar(entradas, solo):
    if not solo:
        return entradas
    return [e for e in entradas if any(e["ruta"] == s or e["ruta"].startswith(s.rstrip("/") + "/") for s in solo)]


def verificar_scripts(raiz, solo=None, estricto=False, python=None, callar=False, trabajadores=4):
    python = python or sys.executable
    entradas = cargar_manifiesto(raiz)
    problemas_globales = []
    if not solo:
        en_manifiesto = {e["ruta"] for e in entradas}
        for ruta in scripts_en_disco(raiz):
            if ruta not in en_manifiesto:
                problemas_globales.append(f"el script {ruta} no está en manifiesto.json (no tiene salida esperada)")
    elegidas = seleccionar(entradas, solo)
    resultados = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=trabajadores) as ex:
        futuros = [ex.submit(evaluar, raiz, e, python, estricto) for e in elegidas]
        for e, f in zip(elegidas, futuros):
            ok, detalle = f.result()
            resultados.append((e, ok, detalle))
            if not callar:
                print(("  [ok]    " if ok else "  [FALLA] ") + e["ruta"])
                if not ok:
                    print("      " + detalle)
    for pg in problemas_globales:
        if not callar:
            print("  [FALLA] " + pg)
    return resultados, problemas_globales


# ------------------------------------------------------------ ejemplos interactivos

def bloques_pycon(raiz):
    """Lee ejemplos-interactivos.md: <!-- sesion: sesion-NN --> seguido de un bloque ```pycon."""
    md = raiz / "ejemplos-interactivos.md"
    if not md.exists():
        return []
    texto = md.read_text(encoding="utf-8")
    patron = re.compile(r"<!--\s*sesion:\s*(sesion-\d\d)\s*-->\s*\n```pycon\n(.*?)\n```", re.S)
    return [(m.group(1), m.group(2)) for m in patron.finditer(texto)]


def verificar_ejemplos(raiz, python=None, callar=False):
    python = python or sys.executable
    resultados = []
    for n, (sesion, codigo) in enumerate(bloques_pycon(raiz), 1):
        with tempfile.TemporaryDirectory() as tmp:
            archivo = Path(tmp) / "ejemplo.txt"
            archivo.write_text(codigo + "\n", encoding="utf-8")
            cwd = raiz / sesion
            p = subprocess.run([python, "-m", "doctest", "-o", "ELLIPSIS", str(archivo)], cwd=cwd,
                               env=entorno(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        ok = p.returncode == 0
        nombre = f"ejemplos-interactivos.md #{n} ({sesion})"
        resultados.append((nombre, ok, p.stdout.decode("utf-8", errors="replace")))
        if not callar:
            print(("  [ok]    " if ok else "  [FALLA] ") + nombre)
            if not ok:
                print("      " + "\n      ".join(p.stdout.decode("utf-8", errors="replace").splitlines()[:30]))
    return resultados


# --------------------------------------------------------------------- revisión de texto

# Las palabras vetadas se parten para que este archivo no las contenga al pie de la letra
# y la propia revisión pueda recorrerlo.
VETADAS = [
    ("tont" + "os", re.I), ("dum" + "mies", re.I), ("Cla" + "ude", 0), ("A" + "XO", 0),
    ("En" + "ti", 0), ("Unu" + "hi", 0),
]
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍⌀-⏿]")
SECRETOS = [
    ("llave de AWS", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("llave privada", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("token de Slack", re.compile(r"xox[abprs]-[0-9A-Za-z-]{10,}")),
    ("token de GitHub", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("token de GitLab", re.compile("glp" + r"at-[A-Za-z0-9_-]{15,}")),   # partido para no disparar a los revisores de secretos
    ("llave tipo sk-", re.compile(r"\bsk-[A-Za-z0-9]{20,}")),
    ("asignación de secreto", re.compile(r"(?i)\b(password|passwd|secret|token|api[_-]?key)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]")),
]
IGNORAR_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules", "tensores"}


def archivos_de_texto(raiz, extensiones):
    for p in sorted(raiz.rglob("*")):
        if not p.is_file() or any(parte in IGNORAR_DIRS for parte in p.relative_to(raiz).parts):
            continue
        if p.suffix in extensiones or p.name in extensiones:
            yield p


def revisar_md(raiz):
    problemas = []
    for p in archivos_de_texto(raiz, {".md"}):
        en_codigo = False
        for n, linea in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
            if linea.startswith("```"):
                en_codigo = not en_codigo
            if not en_codigo and re.match(r"#{1,6}\s", linea) and EMOJI.search(linea):
                problemas.append(f"{p.relative_to(raiz)}:{n}: emoji en un título: {linea.strip()}")
    for p in archivos_de_texto(raiz, {".md", ".py", ".txt", ".json", ".gitignore"}):
        texto = p.read_text(encoding="utf-8")
        for palabra, flags in VETADAS:
            patron = re.compile(r"\b" + palabra + r"\b", flags)
            for n, linea in enumerate(texto.split("\n"), 1):
                if patron.search(linea):
                    problemas.append(f"{p.relative_to(raiz)}:{n}: palabra vetada")
    return problemas


def revisar_secretos(raiz):
    problemas = []
    for p in archivos_de_texto(raiz, {".md", ".py", ".txt", ".json", ".gitignore", ".yml", ".yaml", ".cfg", ".toml"}):
        for n, linea in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
            for nombre, patron in SECRETOS:
                if patron.search(linea):
                    problemas.append(f"{p.relative_to(raiz)}:{n}: posible secreto ({nombre})")
    return problemas


# ------------------------------------------------------------------------- autoprueba

def autoprueba(python):
    """Mete a propósito diferencias en una copia del repositorio y comprueba que el
    verificador se pone en rojo en cada una. También comprueba que la copia sin tocar da verde."""
    print("Autoprueba del verificador (se trabaja en una copia temporal; el repositorio no se toca)")
    tmp = Path(tempfile.mkdtemp(prefix="autoprueba-tensores-"))
    copia = tmp / "copia"
    try:
        shutil.copytree(RAIZ, copia, ignore=shutil.ignore_patterns(".git", "__pycache__", "venv", ".venv", "tensores"))
        base = ["sesion-01/dimensiones.py", "sesion-07/cerro.py", "sesion-07/detector.py"]
        fallas = []

        def correr(solo, estricto=False):
            res, pg = verificar_scripts(copia, solo=solo, estricto=estricto, python=python, callar=True)
            return res, pg

        def esperar(nombre, solo, debe_ser_rojo, estricto=False):
            res, pg = correr(solo, estricto)
            rojo = any(not ok for _, ok, _ in res) or bool(pg) or not res
            bien = (rojo == debe_ser_rojo)
            etiqueta = "se puso en rojo" if rojo else "dio verde"
            print(f"  [{'ok' if bien else 'FALLA'}] {nombre}: {etiqueta} ({'esperado' if bien else 'NO ERA LO ESPERADO'})")
            if not bien:
                fallas.append(nombre)

        manifest_path = copia / "manifiesto.json"

        def editar(ruta, antes, despues):
            p = copia / ruta
            t = p.read_text(encoding="utf-8")
            assert antes in t, f"la autoprueba no encontró {antes!r} en {ruta}"
            p.write_text(t.replace(antes, despues, 1), encoding="utf-8")

        esperar("0. la copia sin tocar da verde", base, False)

        editar("sesion-01/salida-esperada/dimensiones.txt", "Todo cuadra", "Todo cuadra!")
        esperar("1. una línea de la salida esperada cambia", ["sesion-01/dimensiones.py"], True)
        editar("sesion-01/salida-esperada/dimensiones.txt", "Todo cuadra!", "Todo cuadra")

        editar("sesion-01/dimensiones.py", 'forma(papeleria) == (2, 7, 3)', 'forma(papeleria) == (7, 2, 3)')
        esperar("2. el script cambia y su assert truena", ["sesion-01/dimensiones.py"], True)
        editar("sesion-01/dimensiones.py", 'forma(papeleria) == (7, 2, 3)', 'forma(papeleria) == (2, 7, 3)')

        editar("sesion-07/salida-esperada/cerro.txt", "precio aprendido: 20.0", "precio aprendido: 21.0")
        esperar("3. un número cambia en un script sin tolerancia", ["sesion-07/cerro.py"], True)
        editar("sesion-07/salida-esperada/cerro.txt", "precio aprendido: 21.0", "precio aprendido: 20.0")

        e26 = "sesion-07/soluciones/salida-esperada/ejercicio-26.txt"
        editar(e26, "exactitud: 0.93", "exactitud: 0.73")
        esperar("4. un número se sale de la tolerancia del script con semilla", ["sesion-07/soluciones/ejercicio-26.py"], True)
        editar(e26, "exactitud: 0.73", "exactitud: 0.94")
        esperar("5. un número cae DENTRO de la tolerancia (debe tolerarse)", ["sesion-07/soluciones/ejercicio-26.py"], False)
        esperar("6. el mismo cambio con --estricto (debe detectarse)", ["sesion-07/soluciones/ejercicio-26.py"], True, estricto=True)
        editar(e26, "exactitud: 0.94", "exactitud: 0.93")
        editar("sesion-07/salida-esperada/detector.txt", "exactitud: 0.908", "exactitud: 0.918")
        esperar("6b. el detector ya no tiene tolerancia: 0.918 contra 0.908 (debe detectarse)", ["sesion-07/detector.py"], True)
        editar("sesion-07/salida-esperada/detector.txt", "exactitud: 0.918", "exactitud: 0.908")

        original = manifest_path.read_text(encoding="utf-8")
        datos = json.loads(original)
        for e in datos["scripts"]:
            if e["ruta"] == "sesion-01/dimensiones.py":
                e["codigo_salida"] = 3
        manifest_path.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
        esperar("7. el código de salida esperado no coincide", ["sesion-01/dimensiones.py"], True)
        manifest_path.write_text(original, encoding="utf-8")

        os.rename(copia / "sesion-01/salida-esperada/dimensiones.txt", copia / "sesion-01/salida-esperada/x.txt")
        esperar("8. falta el archivo de salida esperada", ["sesion-01/dimensiones.py"], True)
        os.rename(copia / "sesion-01/salida-esperada/x.txt", copia / "sesion-01/salida-esperada/dimensiones.txt")

        (copia / "sesion-01/nuevo.py").write_text("print('hola')\n", encoding="utf-8")
        res, pg = correr(None)  # sin filtro: debe delatar el script que no está en el manifiesto
        rojo = bool(pg)
        print(f"  [{'ok' if rojo else 'FALLA'}] 9. un script nuevo sin salida esperada: {'se puso en rojo' if rojo else 'dio verde'}")
        if not rojo:
            fallas.append("9")
        os.remove(copia / "sesion-01/nuevo.py")

        esperar("10. al final la copia vuelve a dar verde", base, False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if fallas:
        print(f"AUTOPRUEBA FALLIDA: el verificador no se comportó como debía en {len(fallas)} caso(s): {', '.join(fallas)}")
        return 1
    print("AUTOPRUEBA SUPERADA: el verificador detectó todas las diferencias introducidas a propósito.")
    return 0


# ----------------------------------------------------------------------------- principal

def comprobar_requisitos(python):
    p = subprocess.run([python, "-c", "import torch; print(torch.__version__)"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        print("No se pudo importar PyTorch con este Python. Instala los requisitos "
              "(pip install -r requirements.txt, ver README.md) y vuelve a correr.", file=sys.stderr)
        return None
    return p.stdout.decode().strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verificador del curso Tensores desde cero")
    ap.add_argument("--estricto", action="store_true", help="sin tolerancia numérica")
    ap.add_argument("--solo", action="append", help="una carpeta o un script (puede repetirse)")
    ap.add_argument("--autoprueba", action="store_true", help="comprueba que el verificador detecta diferencias")
    ap.add_argument("--revisar-md", action="store_true", help="títulos sin emojis y sin palabras vetadas")
    ap.add_argument("--revisar-secretos", action="store_true", help="busca posibles secretos")
    ap.add_argument("--lista", action="store_true", help="muestra el manifiesto")
    ap.add_argument("--raiz", default=str(RAIZ), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    raiz = Path(a.raiz).resolve()
    python = sys.executable

    if a.lista:
        for e in cargar_manifiesto(raiz):
            extra = [k for k in ("modo", "tolerancia", "normalizar") if k in e]
            print(f"{e['ruta']:55s} salida={e.get('codigo_salida', 0)} origen={e.get('origen', '-')} {','.join(extra)}")
        return 0
    if a.revisar_md or a.revisar_secretos:
        malos = []
        if a.revisar_md:
            malos += revisar_md(raiz)
        if a.revisar_secretos:
            malos += revisar_secretos(raiz)
        for m in malos:
            print("  [FALLA] " + m)
        print("Revisión de texto: " + ("todo limpio" if not malos else f"{len(malos)} problema(s)"))
        return 1 if malos else 0

    version = comprobar_requisitos(python)
    if version is None:
        return 2
    if a.autoprueba:
        return autoprueba(python)

    print(f"Python {sys.version.split()[0]} · PyTorch {version}" + ("" if version.split('+')[0] == VERSION_TORCH
          else f"  (AVISO: el curso se escribió con {VERSION_TORCH}; los números pueden variar)"))
    print("Scripts:")
    resultados, globales = verificar_scripts(raiz, solo=a.solo, estricto=a.estricto, python=python)
    ejemplos = []
    if not a.solo:
        print("Ejemplos interactivos:")
        ejemplos = verificar_ejemplos(raiz, python=python)
    n_ok = sum(1 for _, ok, _ in resultados if ok)
    e_ok = sum(1 for _, ok, _ in ejemplos if ok)
    rojo = (n_ok != len(resultados)) or bool(globales) or (e_ok != len(ejemplos)) or not resultados
    print()
    print(f"Scripts corridos: {len(resultados)} de {len(seleccionar(cargar_manifiesto(raiz), a.solo))}; "
          f"coinciden con la lección: {n_ok}; fallan: {len(resultados) - n_ok}"
          + (f"; scripts sin salida esperada: {len(globales)}" if globales else ""))
    if not a.solo:
        print(f"Ejemplos interactivos corridos: {len(ejemplos)}; coinciden: {e_ok}; fallan: {len(ejemplos) - e_ok}")
    if a.estricto:
        print("(modo estricto: sin tolerancia numérica)")
    print("RESULTADO: " + ("TODO EN VERDE" if not rojo else "HAY FALLAS"))
    return 1 if rojo else 0


if __name__ == "__main__":
    sys.exit(main())
