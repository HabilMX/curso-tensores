# Tensors from scratch

**By Dorian Chávez, founder of Hábil and integration architect.**

From the Python list you already know how to write to a fraud detector that learns from data.

**Who it is for:** Computer Systems students who already program a bit in Python, and Business Administration people who want to
understand what lies beneath the artificial intelligence used in banks, insurance companies and stores. It is an
intermediate-level course: it does not explain what a variable or a loop is, but every new word is explained when it appears. Each
session includes an “If you come from Business Administration” box with the question worth asking in a meeting.

**What you will know by the end:**

- Create a PyTorch tensor from real data and state its shape before running the code.
- Extract and summarize data along any of its axes, and check each result against the same calculation done with a `for`.
- Read a PyTorch shape error, word by word, and fix it.
- Train a small fraud detector and explain why getting 91 % right is not enough.

**What you need before you start:** basic Python (variables, lists, `for` loops and functions; lessons 0 to 4
of the [Python course](https://github.com/HabilMX/curso-python) are enough), Python 3.11 or newer and a terminal. The
first session installs nothing; PyTorch is installed in the second. Tested on Linux (Mint 22.3 and Ubuntu 24.04) and on
macOS; it has not been tested on Windows. No graphics card needed.

## The eight sessions

Two lessons of four sessions, about 2 hours each: roughly one hour to read and follow the example, and another to
practice. They all follow the same story: a stationery store, a card terminal that overcharges, and a student who
has until December 23 to understand what happened.

| Session | | What you learn |
|---|---|---|
| 1 | [A number, a list, a table, a cube](01-un-numero-una-lista-una-tabla-un-cubo.md) | what a tensor is, with Python lists; counting dimensions; where the word came from |
| 2 | [Your first tensor](02-tu-primer-tensor.md) | installing PyTorch; `torch.tensor`, `shape` and `dtype`; PyTorch and TensorFlow |
| 3 | [Axes have names](03-los-ejes-tienen-nombre.md) | taking a slice along any axis; what happens to the shape |
| 4 | [Summing by tab](04-sumar-por-pestana.md) | summing and averaging over an axis with `dim` |
| 5 | [Changing shape without changing the data](05-cambiar-de-forma.md) | `reshape` and how to read a shape error |
| 6 | [Operating on full tensors](06-operar-tensores-completos.md) | broadcasting, and the trap of `*` versus `@` |
| 7 | [A fraud detector that learns from data](07-detector-de-fraude.md) | the training loop; accuracy, recall and precision |
| 8 | [Tensors in the bank, the insurance company and the store](08-banco-aseguradora-tienda.md) | reading a real case and stating the shape of its tensor; the precautions that are not optional |

## The code

Each session has its folder in this repository ([`sesion-01`](../sesion-01) to [`sesion-08`](../sesion-08)), with every
program and the output it should give you. `verificar.py` runs all of them and compares them with what the lesson says. The
step-by-step installation is in the [main README](../README.md).

## Where it is published

One session per week on “En construcción”, Hábil’s Substack, on Wednesdays at 8:00, from November 4 to
December 23, 2026: https://enconstruccionmx.substack.com
