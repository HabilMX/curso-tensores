#!/usr/bin/env python3
"""Medidor canónico de palabras de una lección (Markdown).

Cuenta la PROSA de la lección. La regla, completa:

1. Se quita el frontmatter: si el archivo empieza con una línea «---», se descarta todo
   hasta la siguiente línea «---».
2. Se quitan los bloques de código cercados (``` o ~~~, con o sin lenguaje), junto con sus
   cercas. Un bloque que no se cierra descarta el resto del archivo (y se avisa).
3. Se quitan los comentarios HTML (<!-- ... -->).
4. Los enlaces [texto](url) cuentan por su texto; la url no cuenta. Las imágenes, los
   autoenlaces <http://...> y las direcciones sueltas (http/https) no cuentan.
5. El código dentro de una línea (`así`) SÍ cuenta: es parte de la frase.
6. Los títulos, las listas, las citas y las tablas cuentan; los símbolos de Markdown
   (#, >, -, *, |, ---) no, porque no son palabras.
7. Una palabra es una racha de letras o dígitos (con tildes y ñ) que puede llevar
   adentro un punto, una coma, un apóstrofo o un guion seguidos de más letras o dígitos:
   «doña», «14,500», «2.14.1», «torch.tensor» y «ex-alumno» son UNA palabra cada una.
   Las rayas largas (—), los signos de puntuación y los emojis no cuentan.

Opcional: --sin-seccion «Para Instagram» quita esa sección (desde su título hasta el
siguiente título del mismo nivel o mayor). Se puede repetir.

Uso:
    python medir_palabras.py leccion.md [otra.md ...]
    python medir_palabras.py leccion.md --sin-seccion "Para Instagram" --sin-seccion "Para Slack"
"""
import argparse
import re
import sys
from pathlib import Path

PALABRA = re.compile(r"[^\W_]+(?:[.,'’\-][^\W_]+)*")


def quitar_frontmatter(lineas):
    if lineas and lineas[0].strip() == "---":
        for i in range(1, len(lineas)):
            if lineas[i].strip() == "---":
                return lineas[i + 1:]
    return lineas


def quitar_codigo(lineas):
    salida, cerca = [], None
    aviso = False
    for linea in lineas:
        s = linea.lstrip()
        if cerca is None:
            if s.startswith("```") or s.startswith("~~~"):
                cerca = s[:3]
                continue
            salida.append(linea)
        elif s.startswith(cerca):
            cerca = None
    if cerca is not None:
        aviso = True
    return salida, aviso


def quitar_secciones(lineas, titulos):
    if not titulos:
        return lineas
    salida, nivel_excluido = [], None
    for linea in lineas:
        m = re.match(r"(#{1,6})\s+(.*)", linea)
        if m:
            nivel = len(m.group(1))
            if nivel_excluido is not None and nivel <= nivel_excluido:
                nivel_excluido = None
            if nivel_excluido is None and any(t.lower() in m.group(2).lower() for t in titulos):
                nivel_excluido = nivel
        if nivel_excluido is None:
            salida.append(linea)
    return salida


def contar(texto, sin_secciones=()):
    """Devuelve (palabras, aviso_de_bloque_sin_cerrar)."""
    lineas = texto.replace("\r\n", "\n").split("\n")
    lineas = quitar_frontmatter(lineas)
    lineas, aviso = quitar_codigo(lineas)
    lineas = quitar_secciones(lineas, sin_secciones)
    t = "\n".join(lineas)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"<https?://[^>]*>", " ", t)
    t = re.sub(r"https?://\S+", " ", t)
    return len(PALABRA.findall(t)), aviso


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cuenta las palabras de la prosa de una lección")
    ap.add_argument("archivos", nargs="+")
    ap.add_argument("--sin-seccion", action="append", default=[], metavar="TITULO")
    a = ap.parse_args(argv)
    total = 0
    for nombre in a.archivos:
        n, aviso = contar(Path(nombre).read_text(encoding="utf-8"), a.sin_seccion)
        total += n
        print(f"{n:7d}  {nombre}" + ("   (AVISO: bloque de código sin cerrar)" if aviso else ""))
    if len(a.archivos) > 1:
        print(f"{total:7d}  total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
