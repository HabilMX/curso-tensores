# Tensors from scratch (2): your first tensor

*You install PyTorch, turn doña Lupe’s stationery store into a real tensor, and discover that the function you programmed last week was already built in.*

*Tensors from scratch · lesson A, session 2 of 8*

*Setting: the characters and the story are fictional. Technical details, historical facts, and real figures include their sources.*

**Monday, November 9, 7:40 p.m. Back room of the Centro branch.**

It smells of fresh paper and reheated coffee. Doña Lupe pulls the metal shutter halfway down, with that screech you can hear all the way to the corner, and sits down next to Mateo.

“The Norte file has arrived,” she says, holding the phone in front of him. “The weekly one. That's the one I care about.”

Mateo transfers it to his laptop. Seven days, three products: notebooks, pens, and backpacks. The usual. On the wall calendar, someone circled December 23 in red. Forty-four days left.

“And what are you going to do with it now? Your little lists again?”

Last week Mateo wrote out the stationery store data as lists inside lists and coded two functions by hand: one that counts dimensions and another that calculates the shape. They worked. But he also discovered something uncomfortable: his `forma` function gets fooled by a crooked list. And it doesn't give a warning.

“Lists aren't enough for me anymore, Auntie. I need a real tool.”

“And how much does it cost?”

“Nothing. It's free.”

Doña Lupe looks at him over her glasses.

“Cheap things turn out expensive, mijo.”

“Not this one. It's what the people who make artificial intelligence use.”

“Even worse.”

Mateo laughs and opens the terminal. But before installing anything, he gives himself a rule: answer last week's three questions without looking at his notes. If he can't, there's no point in continuing. A new tool on a shaky foundation only makes mistakes happen faster.

## Warm-up review

Before starting, answer without looking at session 1. The answers are at the end, in the solutions.

1. How do you know how many dimensions a dataset has, without counting brackets?
2. What does the shape `(2, 7, 3)` tell you, and how many numbers are there in total?
3. What condition must a list of lists meet to be a tensor?

If you hesitated on any of them, go back to session 1 before continuing.

> **The code for this session** is in the course's public repository, in the `sesion-02` folder: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-02). Type it yourself first, because that's how you learn; the repository is for comparing when something doesn't come out the same. Each program comes with the output it should give you.

## Objectives of this session

By the end, you will be able to do these five things. Next to each one is the exercise that demonstrates it.

| # | By the end you can… | Demonstrated by |
|---|---|---|
| 1 | **Install** PyTorch in its own environment and **verify** that it works | Step 1 and exercise 6 |
| 2 | **Convert** your nested lists into a tensor with `torch.tensor` | Exercises 7 and 8 |
| 3 | **Read** a tensor's shape, dimensions, and total number of elements (`.shape`, `.ndim`, `.numel()`) and **verify** them with an `assert` against your session 1 functions | Exercises 6, 7, and 8 |
| 4 | **Predict** a tensor's number type (`.dtype`) and **choose** it yourself when necessary | Exercises 6 and 8 |
| 5 | **Read** the error PyTorch gives with a crooked list and **state** where the problem is | Exercise 9 |

**What you need:** session 1 completed, with your `dimensiones.py` file (the stationery store lists and the `contar_dimensiones` and `forma` functions). A computer running Linux Mint (or any Linux or macOS with Python 3.11 or newer; the course has not been tested on Windows), internet to download PyTorch, and an open terminal.

---

## Why it matters

Mateo looks back at his `forma` function. A `while`, an `append`, a `tuple`. It worked with the stationery store. But it has a hole: with a crooked list, it lies. And it lies with utter confidence.

That is what keeps him uneasy. If the Norte file arrived malformed, his function would give him a shape without batting an eye, and he would believe it. And if the file contained a million numbers, every calculation would be another `for` loop written by hand.

People who work with artificial intelligence don't program those functions every time. They use a library that already stores numbers in order, knows their shape, checks that they are rectangular, and does calculations very fast. The most widely used in research is called **PyTorch**.

Today you learn the first thing anyone working with AI does when new data arrives: **create the tensor and ask it what it's like.** What shape does it have? How many dimensions? How many numbers? Of what type? If those answers aren't what you expect, nothing that comes after will go well.

Mateo doesn't know it yet, but tonight one of those questions is going to answer “no”.

---

## Step 1 — Installing PyTorch

### A separate room for your tools

Before installing, you are going to create a **virtual environment**. It sounds complicated, and it isn't: it's a folder where you store a project's libraries, separated from the rest of your computer.

Think of it like your plumber uncle's toolbox: if he puts his wrenches in the kitchen drawer, one day he won't find anything. With his own box, nobody moves anything around. The virtual environment is your box: what you install there won't get in the way of Linux Mint, which uses its own Python to function.

In the terminal, first check your Python version:

```bash
python3 --version
```

You need 3.11 or newer: that's what PyTorch has required since version 2.15. Linux Mint 22 comes with Python 3.12, so you're on track.

**On Linux Mint and Ubuntu, this first.** The tool for creating environments comes in a separate package that a fresh installation does not include. Install it (it will ask for your password); if you already had it, no harm done:

```bash
sudo apt install python3-venv
```

Now create the environment and activate it:

```bash
python3 -m venv ~/tensores
source ~/tensores/bin/activate
```

The first line creates the box: a folder called `tensores` in your home directory (that's what `~` means). The second opens it. You will notice that your terminal now starts with `(tensores)`. That's the sign that you are inside.

**If you use Windows:** those two lines are for Linux and macOS. On Windows, from your home folder, create the environment with `python -m venv tensores` and open it with `.\tensores\Scripts\activate` (as stated in the [Python documentation](https://docs.python.org/3/library/venv.html)). Note: the course was not tested on Windows; if an output does not match, compare it with the repository before assuming you made a mistake.

**If you get this error:** you skipped the step above. On Ubuntu 24.04 (the same base as Linux Mint 22) the message looks like this:

```text
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.12-venv

You may need to use sudo with that command.  After installing the python3-venv
package, recreate your virtual environment.
```

Notice that the message proposes `python3.12-venv`, with the version number, and here we ask you for `python3-venv`, without a number. Both work: under the hood, `python3-venv` installs the one for your version. We prefer the one without a number because it keeps working when your system upgrades its Python version. Run `sudo apt install python3-venv` and run the two lines again.

Note: **every new terminal starts with the environment closed.** Run `source ~/tensores/bin/activate` again and look for `(tensores)`. If you don't see it, Python won't find PyTorch.

### The official command

With the environment activated, update the package installer and then install PyTorch with the official command from its website for Linux, without a graphics card:

```bash
pip3 install --upgrade pip
pip3 install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

What that command says, part by part:

- `torch==2.14.1` is PyTorch, at the exact version this course was written with. We lock it so that your outputs match those of the lesson: with another version the numbers may change slightly and you wouldn't know if you made a mistake.
- `torch` is PyTorch. The name comes from **Torch**, an earlier library written in another language (Lua), from whose community some of the people who created it emerged.
- `torchvision` is a companion package for images. It comes in the official command and does not get in the way.
- `--index-url https://download.pytorch.org/whl/cpu` tells `pip` which store to download from: the **CPU-only** edition. Without that part, on Linux it would download the edition for NVIDIA graphics cards, which weighs much more and you do not need in this course.

It is a large download; take it easy.

### Verifying it's set up

Create a file `verifica.py`, in the same folder where you have `dimensiones.py`:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

Run it with `python3 verifica.py`:

```
2.14.1 False
```

The first value is the PyTorch version. If yours shows `2.14.1+cpu`, that's fine too: `+cpu` is the tag for the CPU-only edition. The second value tells you whether PyTorch can use an NVIDIA graphics card (what PyTorch calls CUDA). Since you installed the CPU-only edition, `False` is exactly what we expect, even if your laptop has a graphics card.

**What is a GPU?** It's the graphics card: it was born to render millions of dots on the screen simultaneously in video games. The CPU is a handful of very skilled chefs; the GPU is a team of thousands of assistants who only know how to chop, but they all chop at once, and multiplying large tensors is pure chopping. This whole course runs on the CPU of an ordinary laptop.

> Mateo sees the `False` and laughs. Doña Lupe, who hasn't taken her eyes off the screen, asks if that's bad.
> “It means PyTorch won't use a graphics card on my laptop, Auntie.”
> “Oh, well good. That way you study.”

---

## Step 2 — From number and list to tensor

> **8:25 p.m.** Mateo is one click away from the Norte file. He doesn't open it. First he is going to practice with last week's data, the ones he already knows are good. If something fails with good data, the error is his. If it fails with new data, the error comes in the data. Mixing the two is the fastest way to understand nothing.

There is only one function to learn today to create tensors: `torch.tensor`. You give it your lists and it returns a tensor with the same numbers.

Create `primer_tensor.py` in the same folder as `dimensiones.py`. The line `from dimensiones import ...` takes the lists you wrote last week, without copying them again:

```python
import torch
from dimensiones import total, por_dia

t_total = torch.tensor(total)
t_dia = torch.tensor(por_dia)

print(t_total)
print(t_dia)
print(t_total.shape, t_dia.shape)
```

Run it with `python3 primer_tensor.py`:

```
tensor(18500)
tensor([2500, 3100, 2200, 2800, 3900, 2400, 1600])
torch.Size([]) torch.Size([7])
```

Three things to read in that output:

- PyTorch wraps your numbers in `tensor(...)`: it is no longer a Python list, it is a tensor.
- The scalar looks like `tensor(18500)`, without brackets. Zero dimensions, just like in session 1.
- `.shape` is the **shape**. For the scalar it comes out `torch.Size([])`, empty, because it doesn't have any direction. For the week it comes out `torch.Size([7])`: seven days.

Notice: `.shape` does not take parentheses. It is not a function, it is data the tensor already carries, like the tag with a shirt's size: you don't measure it, you read it.

> “And so what?” says doña Lupe, pointing at `18500`. “That's the same thing I already had written down.”
> “That's the point, Auntie. That it's the same. Now the computer knows it too.”

---

## Step 3 — The complete stationery store

Now the spreadsheet with its two tabs. Create `papeleria.py` in the same folder; this file will store the tensor you will use in subsequent sessions:

```python
import torch
from dimensiones import centro, norte

papeleria = torch.tensor([centro, norte])

# Esto corre al abrir papeleria.py, no cuando otra sesión lo importa.
if __name__ == "__main__":
    print(papeleria)
    print(papeleria.shape)
    print(papeleria.ndim)
    print(papeleria.numel())
    print(papeleria.dtype)
```

Output:

```
tensor([[[1200,  800,  500],
         [1500,  900,  700],
         [1000,  700,  500],
         [1300,  900,  600],
         [2000, 1200,  700],
         [1200,  800,  400],
         [ 800,  500,  300]],

        [[ 300,  200,  100],
         [ 400,  250,  150],
         [ 350,  200,  100],
         [ 300,  150,  100],
         [ 500,  300,  200],
         [ 300,  200,  100],
         [ 200,  100,  100]]])
torch.Size([2, 7, 3])
3
42
torch.int64
```

The condition `if __name__ == "__main__":` leaves those five checks for when you run this file directly. Subsequent sessions will be able to use `from papeleria import papeleria` without printing the whole table again.

PyTorch aligns the columns and leaves a blank line between the Centro tab and the Norte tab: the same tabbed spreadsheet from last week. Now read the four answers:

| You ask it | You write | It answers | It means |
|---|---|---|---|
| What shape do you have? | `papeleria.shape` | `torch.Size([2, 7, 3])` | 2 branches, 7 days, 3 products |
| How many dimensions? | `papeleria.ndim` | `3` | branch, day, and product |
| How many numbers do you carry? | `papeleria.numel()` | `42` | 2 × 7 × 3 |
| What type are they? | `papeleria.dtype` | `torch.int64` | integers (we see this in step 4) |

The names come from English: `shape` is “forma”; `ndim`, “number of dimensions”; `numel`, *number of elements*, “número de elementos”; `dtype`, *data type*, “tipo de dato”. Notice a detail: `numel()` does take parentheses, because it is a function that counts. The other three do not.

### `torch.Size([2, 7, 3])` is your `forma` from last week

Session 1 ended with this output from your function:

```
papeleria 3 (2, 7, 3)
```

And today PyTorch tells you `torch.Size([2, 7, 3])`. **It is exactly the same thing**, dressed in different clothes. `torch.Size` is the way PyTorch stores a shape, and internally it behaves like a tuple. That is why you can compare it directly against `(2, 7, 3)`.

Your `while` with `append` and `.shape` do the same job. The difference: PyTorch's, as you will see in step 5, does not get fooled by a crooked list.

### Verifying with `assert`: shape first, value second

Don't believe anyone, not even PyTorch: verify it. Create `compara.py`:

```python
import torch
from dimensiones import centro, norte, forma, contar_dimensiones

papeleria_lista = [centro, norte]
papeleria = torch.tensor([centro, norte])

# 1) Primero la forma: PyTorch contra tus funciones de la sesión 1
print(papeleria.shape, forma(papeleria_lista))
print(papeleria.ndim, contar_dimensiones(papeleria_lista))
assert papeleria.shape == (2, 7, 3), "La papelería debería ser 2 sucursales × 7 días × 3 productos"
assert papeleria.shape == forma(papeleria_lista), "PyTorch y tu función deberían dar la misma forma"
assert papeleria.ndim == contar_dimensiones(papeleria_lista), "Deberían contar las mismas dimensiones"

# 2) Después el total de números, contra un for
cuenta = 0
for pestana in papeleria_lista:
    for renglon in pestana:
        for numero in renglon:
            cuenta = cuenta + 1
print(papeleria.numel(), cuenta)
assert papeleria.numel() == cuenta, "Deberían ser los mismos números"

# 3) Al final, el valor: de regreso a listas, ¿quedó igual?
assert papeleria.tolist() == papeleria_lista, "Los datos no deberían cambiar al convertirlos"
print("Todo cuadra")
```

Output:

```
torch.Size([2, 7, 3]) (2, 7, 3)
3 3
42 42
Todo cuadra
```

Three checks, in this order:

1. **The shape.** It is the `assert` you will write in every exercise of the course. If the shape is wrong, everything else is wrong.
2. **The total against a `for`.** The three nested `for` loops count one by one what `numel()` counts in a single word. If both give 42, nothing was lost and nothing was extra.
3. **The value.** `.tolist()` makes the return trip: it turns the tensor back into lists. The list is called `papeleria_lista` so it doesn't clash with the tensor `papeleria`; if they are equal, the numbers didn't change along the way.

If you change `(2, 7, 3)` to `(7, 2, 3)`, the first `assert` blows up with its message, just like in session 1. Order matters: 2 tabs of 7 rows is not the same as 7 tabs of 2.

> “Todo cuadra.” Mateo reads it twice. With familiar data, the tool doesn't make mistakes and doesn't let him make them. Now he can trust it. One thing is missing before the Norte file: knowing what kind of numbers it will store.

---

## Step 4 — The number type: `dtype`

All numbers in a tensor are of the same type. You cannot mix, as in a Python list, an integer and a piece of text. That single type is called **dtype**.

Why? Because that way the computer knows exactly how much space each number occupies and lines them up without checking one by one, like boxes of the same size on a shelf.

To begin, two types are enough:

| dtype | What it stores | Example |
|---|---|---|
| `torch.int64` | integers: items, pesos without cents, counts | 12 notebooks |
| `torch.float32` | numbers with decimals: measurements, averages, percentages | 9.7 average |

The number in the name is how much space each one occupies, in bits: 64 or 32. You don't need more for now.

A warning for when you work with real money: `float32` stores decimals approximately (0.1 is stored as 0.10000000149011612). For exact amounts, systems store cents as integers: $35.50 is stored as `3550`. In the example below we use prices with decimals only to see what `dtype` comes out.

Create `tipos.py`:

```python
import torch

piezas = torch.tensor([12, 30, 7])
precios = torch.tensor([35.5, 12.0, 450.0])
mezcla = torch.tensor([12, 30.5, 7])

print(piezas, piezas.dtype)
print(precios, precios.dtype)
print(mezcla, mezcla.dtype)
```

Output:

```
tensor([12, 30,  7]) torch.int64
tensor([ 35.5000,  12.0000, 450.0000]) torch.float32
tensor([12.0000, 30.5000,  7.0000]) torch.float32
```

The rule PyTorch follows:

- **If all are integers, `int64` comes out.** That's why your stationery store came out `torch.int64`: doña Lupe records pesos without cents.
- **If at least one has a decimal point, `float32` comes out,** and the integers also become decimals: 12 turns into `12.0000`. A single 30.5 is enough for the entire row to change.

> “I don't write down cents,” says doña Lupe. “Cents are the bank's business.”
> “That's why your sheet came out `int64`, Auntie. Whole numbers, like you.”
> “Are you calling me rigid?”
> “I'm calling you exact.”

### Choosing the type yourself

Sometimes you want decimals even if your data are integers. For example, to calculate averages, as you will see in session 4. For that, `torch.tensor` accepts a second argument:

```python
import torch
from dimensiones import por_dia

dia_dec = torch.tensor(por_dia, dtype=torch.float32)
print(dia_dec)
print(dia_dec.dtype)
```

Output:

```
tensor([2500., 3100., 2200., 2800., 3900., 2400., 1600.])
torch.float32
```

The dot after each number (`2500.`) is the clue that they are already decimals. Same numbers, different type.

---

## Step 5 — The crooked list: PyTorch says no

**10:15 p.m.** This time for real. Mateo copies the seven rows from the Norte file into a list, writes the usual line, `torch.tensor(...)`, and hits Enter.

The terminal fills with red letters.

“Did you break it already?” asks doña Lupe.

“No, Auntie. It refused.”

Mateo goes down the file, row by row, finger on the screen. Monday: three numbers. Tuesday: two. Wednesday: three. Tuesday is missing one.

Before jumping to conclusions, he reproduces it in miniature. In session 1 you saw that your `forma` function was fooled by this list and you had to check it with `es_rectangular`:

```python
chueca = [[1, 2, 3], [4, 5]]
```

Let's give it to PyTorch. Create `chueca.py`:

```python
import torch

chueca = [[1, 2, 3], [4, 5]]
t = torch.tensor(chueca)
```

Run it:

```
Traceback (most recent call last):
  File ".../chueca.py", line 4, in <module>
    t = torch.tensor(chueca)
        ^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 3 at dim 1 (got 2)
```

The same red as with the Norte file. PyTorch refuses to create the tensor. The important thing is the last line, and it is read word by word:

| It says | Translated | What it means here |
|---|---|---|
| `ValueError` | value error | the data you gave it cannot be used for this |
| `expected sequence of length 3` | expected a list of length 3 | the first row had length 3, so all should have length 3 |
| `at dim 1` | at dimension 1 | the inside dimension: the rows. The outside one is 0 |
| `(got 2)` | received 2 | it found a row that has length 2 |

Notice `dim 1`. Dimensions are numbered from 0, just like the indices of a list: the outside one is 0, the next is 1. In the stationery store, dimension 0 is branches, 1 is days, and 2 is products. You will use that numbering a lot in session 3.

It is your `es_rectangular`, but better: it checks all the boxes at all levels and tells you where the problem is. That is why the tensor you create with `torch.tensor` is always rectangular. (PyTorch has special tensors for ragged data, but we do not use them in this course.)

Your function from last week would have swallowed the Norte file without saying anything. PyTorch doesn't: it prefers to blow up rather than lie to you.

**How it is fixed:** not by padding with an invented number. Mateo could put a zero on Tuesday and continue. The tensor would come out perfect, and it would be lying: a zero says “nothing was sold”; a blank says “I don't know”. It is not the same. Ask the person who gave you the data which one is missing. If Tuesday's backpacks are missing, the tensor is not fixed with a zero; it is fixed by asking.

> “Who puts this file together, Auntie?”
> “The branch sends it. Every Monday. It had never arrived wrong before.”
> Never. Mateo stares at Tuesday.

---

## Sidebar: the fight of two giants (PyTorch and TensorFlow)

While PyTorch was downloading, Mateo asked himself a question: if this tool is so good, why does he hear another name everywhere? You are going to hear two names all the time, and behind them is a fight between two giants. The facts are from the course's research, with dates.

**The one that arrived first.** **TensorFlow** is from Google. Google released it as open source on **November 9, 2015**, with permission for anyone to use it: eleven years to the day before the night Mateo installed its rival. The version in effect as of October 2026 is **2.21.0**, published on March 6, 2026.

**The challenger.** **PyTorch** was born at Facebook (now Meta), in its research laboratory. Its first versions on GitHub are from **September 2016**, ten months later, and its official launch was in January 2017. Since **September 2022** it no longer belongs to a single company: it is cared for by the PyTorch Foundation, within the Linux Foundation, and its founding members were AMD, AWS (from Amazon), Google Cloud, Meta, Microsoft Azure, and NVIDIA. Notice a name on that list: Google Cloud, from the same home as the rival, is also seated there. The version in effect when we wrote this session (October 2026) is **2.14.1**, published on September 30, 2026. If a newer one installed for you, that's fine.

**Who is winning?** It depends on the ring.

- **In the models people publish, PyTorch, by far.** On Hugging Face, the site where people publish AI models, on October 18, 2026 there were 258,777 models with the PyTorch tag versus 17,469 with TensorFlow: about 15 for each one. That measures what people upload, not what they use in their work, but the difference is clear. And Hugging Face's most popular model library, Transformers, announced that in version 5 it drops TensorFlow and stays solely with PyTorch.
- **Among programmers, almost a tie.** In the 2024 Stack Overflow survey, 10.6% of respondents said they use PyTorch and 10.1%, TensorFlow. The survey asks what each person uses, not what runs in production. TensorFlow arrived earlier at companies and remains alive in many, and on mobile phones.

*Sources: [Google announcement on TensorFlow (2015)](https://blog.google/innovation-and-ai/products/tensorflow-smarter-machine-learning-for/); versions on PyPI: [tensorflow](https://pypi.org/project/tensorflow/) and [torch](https://pypi.org/project/torch/); PyTorch versions on [GitHub](https://github.com/pytorch/pytorch/releases); [Linux Foundation, September 12, 2022](https://www.linuxfoundation.org/press/press-release/meta-transitions-pytorch-to-the-linux-foundation); [PyTorch models on Hugging Face](https://huggingface.co/models?library=pytorch), accessed October 18, 2026 (the figure changes every day); [Stack Overflow Survey 2024](https://survey.stackoverflow.co/2024/technology).*

**A difference you will notice if one day you use both:** with the same integers, PyTorch creates `int64` and TensorFlow, `int32`.

| | PyTorch | TensorFlow |
|---|---|---|
| Create a tensor | `torch.tensor([1, 2, 3])` | `tf.constant([1, 2, 3])` |
| Default integer | `int64` | `int32` |
| With decimals | `float32` | `float32` |

**Which one to use?** To learn and read research code, PyTorch. If one day your job uses TensorFlow, you learn it in an afternoon: shape, dimensions, and type work the same.

---

👉 **If you come from Business Administration:** `dtype` is the format you give a column in your spreadsheet: “whole number” or “currency with two decimal places”. And the crooked list error is the branch report that arrives with one column missing: it is not added to the others until someone asks what is missing. When an analyst tells you “the model didn't run because the data didn't have the right shape”, you already know what they are talking about.

---

## Session 2 exercises

The solutions are at the end of this issue. Work in the same folder as your `dimensiones.py` and with the environment activated.

**Exercise 6 — Predict before running.** Write on paper the shape, dimensions, type, and total number of numbers for each tensor. Then run and compare.

```python
import torch

temperatura = torch.tensor(31)
calificaciones = torch.tensor([[9, 8, 10], [7, 9, 8], [10, 10, 9], [6, 8, 7]])
foto = torch.tensor([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])
promedios = torch.tensor([9.0, 8.0, 9.7, 7.0])

for nombre, t in [("temperatura", temperatura), ("calificaciones", calificaciones),
                  ("foto", foto), ("promedios", promedios)]:
    print(nombre, t.shape, t.ndim, t.dtype, t.numel())
```

These are the same data from exercise 1 of session 1, plus one with decimals. Hint: how many numbers does a scalar carry?

**Exercise 7 — Fill in the blanks.** The Norte tab alone, as a tensor. Fill in the `____` and run. If the final `print` appears, you got it right.

```python
import torch
from dimensiones import norte

norte_t = torch.____(norte)

assert norte_t.shape == ____, "Norte es una pestaña: 7 días × 3 productos"
assert norte_t.ndim == ____, "Una pestaña es una tabla"
assert norte_t.numel() == ____, "7 × 3"
assert norte_t.dtype == ____, "Son pesos enteros"
assert norte_t.tolist() == norte, "Los datos no deberían cambiar"
print("Ejercicio 7 superado")
```

**Exercise 8 — Your own data, now in a tensor.** Take the grades you wrote in exercise 4 of session 1 (students, subjects, and midterms) and:

a) Convert them to a tensor and write the `assert` for its shape against your `forma` function.
b) Write the `assert` for its dimensions against `contar_dimensiones`.
c) Count its numbers with nested `for` loops and compare it with an `assert` against `numel()`.
d) Create a second version with `dtype=torch.float32` and check with an `assert` that the type changed but not the shape.

End with `print("Ejercicio 8 superado")`.

**Exercise 9 — Find the crooked one, now with PyTorch.** Before running, predict which ones blow up and what `dim` number the error will say. Then test each one separately with `torch.tensor(...)`.

```python
a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8]]
c = [[[1, 2], [3, 4]], [[5, 6], [7]]]
d = [[1, 2], [3, 4, 5]]
```

For the ones that blow up, answer: why in `d` does the error say it expected 2 and received 3, if the row that “overflows” is the second one? And fix `b` by adding the number it's missing (assume it's a 9).

---

## Common errors (and how to spot them)

| Error | What it looks like | How to notice it | What to do |
|---|---|---|---|
| Forgetting to activate the environment | `ModuleNotFoundError: No module named 'torch'` | Your terminal doesn't start with `(tensores)` | Run `source ~/tensores/bin/activate` in every new terminal |
| Calling your file `torch.py` | `AttributeError: partially initialized module 'torch' has no attribute 'tensor' (most likely due to a circular import)` | It blows up on the first line that uses `torch.` | Python imported your file instead of PyTorch. Rename it, for example to `primer_tensor.py` |
| Putting parentheses on `.shape` | `TypeError: 'torch.Size' object is not callable` | You wrote `t.shape()` | `.shape`, `.ndim`, and `.dtype` go without parentheses; only `.numel()` takes them |
| Forgetting parentheses on `numel` | Something like `<built-in method numel of Tensor object at 0x...>` comes out instead of a number | The `print` doesn't show a number | Write `t.numel()`, with parentheses |
| Writing `torch.Tensor` with a capital letter | `torch.Tensor([1, 2, 3])` gives `tensor([1., 2., 3.])`, decimals | The `dtype` comes out `torch.float32` with integer data | Use `torch.tensor`, with lowercase: it respects the type of your data |

---

## Review for next session

Session 3 begins with these three questions. Answer them without looking at the lesson:

1. What does `t.shape` tell you and how do you verify it with an `assert`?
2. Why does `torch.tensor([[1, 2, 3], [4, 5]])` fail?
3. What `dtype` comes out of `torch.tensor([1, 2, 3])` and which from `torch.tensor([1.0, 2.0])`?

🔁 **Next week:** you now have the stationery store as a tensor. Now you are going to extract slices along any of its axes: only the Norte tab, only Fridays for both branches, only backpacks for the whole week. And you will see that a photo is cropped exactly the same way.

It's past eleven when Mateo closes the laptop. Doña Lupe has already counted the register and is waiting by the shutter with keys in hand. On the phone, the Norte file is still open, with its Tuesday of two numbers. It doesn't look like an ordinary typo: it is the only crooked row in weeks of straight files. And it arrived the very week the bank suspended the terminal.

Who leaves a day half-finished?

---

## Solutions (don't look before trying)

**Warm-up review.**

1. You count how many pieces of data you need to point out a single number. In the stationery store you need the branch, the day, and the product: three pieces of data, three dimensions. And if you don't want to count yourself, `contar_dimensiones` counts it (and starting today, `.ndim`).
2. That there are 2 elements in the first dimension (branches), 7 in the second (days), and 3 in the third (products). In total, 2 × 7 × 3 = 42 numbers.
3. That it must be rectangular: all boxes at the same level have to measure the same. All tabs with 7 rows, all rows with 3 columns.

**Exercise 6.**

```
temperatura torch.Size([]) 0 torch.int64 1
calificaciones torch.Size([4, 3]) 2 torch.int64 12
foto torch.Size([2, 2, 3]) 3 torch.int64 12
promedios torch.Size([4]) 1 torch.float32 4
```

The trap was the scalar: its shape is empty and it has zero dimensions, but it carries **one** number. `promedios` comes out `float32` because it has decimals. Notice also that `calificaciones` and `foto` carry the same 12 numbers with different shapes: the total doesn't tell you the shape, but the shape does tell you the total.

**Exercise 7.**

```python
norte_t = torch.tensor(norte)

assert norte_t.shape == (7, 3), "Norte es una pestaña: 7 días × 3 productos"
assert norte_t.ndim == 2, "Una pestaña es una tabla"
assert norte_t.numel() == 21, "7 × 3"
assert norte_t.dtype == torch.int64, "Son pesos enteros"
```

Output:

```
Ejercicio 7 superado
```

**Exercise 8.** A possible answer, with the `parciales` list (student × subject × midterm) from the solution to exercise 4 of session 1:

```python
import torch
from dimensiones import forma, contar_dimensiones

# alumno × materia × parcial
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]

p = torch.tensor(parciales)
print(p.shape, p.ndim, p.numel(), p.dtype)
assert p.shape == forma(parciales), "PyTorch y tu función deberían coincidir"
assert p.ndim == contar_dimensiones(parciales)

cuenta = 0
for alumno in parciales:
    for materia in alumno:
        for calificacion in materia:
            cuenta = cuenta + 1
assert p.numel() == cuenta

p_dec = torch.tensor(parciales, dtype=torch.float32)
print(p_dec.dtype)
assert p_dec.dtype == torch.float32
assert p_dec.shape == p.shape, "Cambiar el dtype no cambia la forma"
print("Ejercicio 8 superado")
```

Output:

```
torch.Size([3, 4, 3]) 3 36 torch.int64
torch.float32
Ejercicio 8 superado
```

If your data have another order or other sizes, your shape will be different, and that's fine: what matters is that the `assert`s pass.

**Exercise 9.**

- `a` works: `torch.Size([3, 2])`.
- `b` blows up: `ValueError: expected sequence of length 3 at dim 1 (got 2)`. The last row measures 2 and the others, 3.
- `c` blows up: `ValueError: expected sequence of length 2 at dim 2 (got 1)`. Here the problem is further inside, in dimension 2: the last pair, `[7]`, only brings one number.
- `d` blows up: `ValueError: expected sequence of length 2 at dim 1 (got 3)`. PyTorch takes the **first** row as the rule. The first measures 2, so the second, which measures 3, is the one that is “wrong”, even if to you it seems that the first is missing one. That's why the error tells you where the rule broke, but deciding which value is correct is up to you, with whoever gave you the data.

Fixed `b`:

```python
b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(torch.tensor(b).shape)
```

```
torch.Size([3, 3])
```

---

## What comes next

In session 3 you stop seeing the tensor as a whole and start taking pieces out of it. If you want to get ahead, open your `papeleria.py` and leave the `papeleria` tensor ready: it's the first thing you will use.

---

*I write from what we see every day at Hábil, where we build systems for banking, insurance, and enterprise in Mexico. If at your work or your family's business you run into a problem like this, let's talk: [habil.mx](https://www.habil.mx/en/)*
