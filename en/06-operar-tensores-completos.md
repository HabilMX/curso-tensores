# Tensors from scratch (6): operating on full tensors

*Multiplying an entire week of sales by their prices in a single line, the rule that decides when PyTorch “stretches” a tensor, and the trap of `*` versus `@` that throws no error.*

*Tensors from scratch · lesson B, session 6 of 8*

*Setting: the characters and story are fictional. Technical and historical data and real figures cite their sources.*

**Monday, December 7, 7:40 p.m. Papelería Herrera, Centro branch.**

The metal security shutter is half-lowered. Inside, it smells of fresh paper and reheated coffee. Doña Lupe is counting coins on the counter. Mateo has his laptop open between a roll of gloss paper and a box of pens.

Sixteen days left until the 23rd.

Since last week, Mateo has known two things. That Norte was doing fine until August. And that in August, a man in a blue t-shirt “swapped the terminal” without anyone having called him. What he doesn't know is what one thing has to do with the other.

“I don't want pesos anymore, Auntie,” he says. “I want unit counts.”

“Unit counts of what?”

“Of everything. How many notebooks, how many pens, how many backpacks. Each day. For Centro and for Norte.”

Doña Lupe looks at him over her glasses.

“You don't need a computer for that, mijo. You need someone to count.”

“They already counted. Fer sent me Norte's numbers this morning.”

Two weeks ago, no one in the family wanted to say Fer's name out loud. Centro sold about four times what Norte did, and she was the store manager. That afternoon she didn't answer the phone. But today she sent her complete count, row by row, without anyone having to ask twice.

Doña Lupe's phone vibrates. An email from the bank. From Licenciado Ortega.

“Dear Mrs. Herrera: As requested, I am sharing the total processed by the payment terminal at your Norte branch for the indicated week. I remind you that the deadline is December 23. Sincerely.”

Below, a single number: **$16,850**.

Doña Lupe hands the phone to Mateo without a word.

“Is that a lot or a little?”

“Depends on how much we sold.”

And that is the catch. To know how much they sold, you have to multiply each unit by its price. Twenty-one cells per branch. With a `for` loop, Mateo already knows how to do it. But tonight he wants a calculation that takes the entire table in one shot.

Because if the units times their prices amount to less than what the terminal charged, someone charged for something that never left the stationery shop.

In the previous session, you changed the shape of a tensor without touching its numbers: 12 months in a single row became 4 quarters of 3 months. Today, you stop rearranging numbers and start doing math with them, with the full tensor all at once.

## Warm-up review

Answer without looking at the previous session. The answers are in the solutions at the end.

1. When is a `reshape` valid?
2. What does the `-1` mean in `reshape`?
3. Why is reshaping a table not the same as transposing (rotating) it?

> **The code for this session** is in the course's public repository, in the `sesion-06` folder: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-06). Type it out yourself first, because that is how you learn; the repository is there for comparison when something doesn't match. Each program includes the output it should give you.

## Objectives of this session

By the end, you will be able to do these four things. Next to each one is the exercise that demonstrates it.

| # | By the end you can… | Demonstrated by |
|---|---|---|
| 1 | **Operate** on two tensors of the same shape number by number, and **verify** it against a `for` loop | Exercise 23 |
| 2 | **Predict**, before running, whether two shapes can be “stretched” to operate together and what shape results | Exercises 22 and 25 |
| 3 | **Calculate** a total per row with `@` and **verify** it in two ways: with a `for` loop and with `sum(dim=1)` | Exercise 24 |
| 4 | **Recognize** when `*` and `@` yield different results without throwing an error | Exercise 22 |

**What you need to know beforehand:** sessions 1 through 5: creating a tensor, reading its `shape` and `dtype`, slicing pieces with indices, summing with `sum(dim=)`, and changing its shape with `reshape`. **What you need to have:** the environment from session 2, with PyTorch installed, and a new file, `operar.py`, to which you will add code step by step. When a step asks you to trigger an error, add the line, run it, read the error, and delete it before continuing.

---

## Why it matters

Mateo could do the math as in previous sessions: a `for` loop inside another `for` loop, cell by cell. It works. But that is not how an artificial intelligence model works. Nor is it how the system that suspended Norte's payment terminal at midnight on a Sunday works.

A large portion of a model's workload comes down to two operations: **multiplying tensors number by number** and **matrix multiplication**. When a bank reviews a transaction, the model takes the transaction's list of numbers and multiplies it by a table of “weights,” which are numbers the model learned. That is a matrix multiplication. And it doesn't do it once: it does it millions of times per second. Somewhere at the bank, a calculation like that was what flagged Norte's charges as “unusual.”

That is why video cards for AI exist. A helpful analogy: a standard computer is a handful of highly skilled chefs; a video card is thousands of kitchen assistants who only know how to chop, but they all chop at the same time. Multiplying matrices is “chopping”: vast numbers of multiplications and additions that can be distributed across many assistants at once. AlexNet, the 2012 neural network that kicked off the current wave of AI, according to the authors' own paper, “takes between five and six days to train on two 3GB GTX 580 GPUs” ([Krizhevsky, Sutskever, and Hinton, 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)). Two video game cards.

Today, everything runs on your computer, without a dedicated video card. The two operations are the same, and in the next session you will need them to train your own model. Tonight, Mateo needs them for a simpler question: how much money do the unit counts represent?

## The case: now doña Lupe counts units

Until now, doña Lupe's numbers were in pesos. This time, Mateo asked her for something different: **how many units** Centro sold each day for each product. And separately, her price list, which she recites from memory: notebooks at $40, pens at $10, and backpacks at $350.

“They haven't changed since back-to-school,” she says. “And they won't change until paper goes up or I drop dead, whichever comes first.”

With the units and the prices, any monetary figure comes out of a single calculation. Mateo starts with Centro, the branch no one suspects: if the math checks out there, he will be able to trust it when he gets to Norte.

### Step 1 — Units and prices

Write this at the beginning of `operar.py`. First the Python lists you already know; then the tensors, using what you saw in session 2:

```python
import torch

lista_unidades = [
    [30, 80, 2],    # lunes
    [40, 90, 3],    # martes
    [25, 70, 2],    # miércoles
    [35, 90, 2],    # jueves
    [50, 120, 3],   # viernes
    [30, 80, 1],    # sábado
    [20, 50, 1],    # domingo
]
lista_precios = [40, 10, 350]   # cuaderno, pluma, mochila

unidades = torch.tensor(lista_unidades)
precios = torch.tensor(lista_precios)

print(unidades.shape, unidades.dtype)
print(precios.shape, precios.dtype)
```

Output:

```
torch.Size([7, 3]) torch.int64
torch.Size([3]) torch.int64
```

Read it as in session 2: `unidades` is a table of 7 days by 3 products; `precios` is a list of 3 prices, one per product. Both store integers. Keep the Python lists: they will be useful for verifying with a `for` loop.

Before moving on, a question to think through: on Monday, 30 notebooks were sold at $40. How much money came in from notebooks on Monday? If you said $1,200, you already know how to do today's math. The only new thing is doing it with all 21 numbers at once.

### Step 2 — Element-wise: two identical tables

Now for Fer's file. Mateo opens it carefully: in session 2, a file from Norte had a crooked row. This one comes in complete. Same shape as Centro's: 7 days, 3 products. First of all, Mateo wants to see both branches side by side, cell by cell. Add to your file:

```python
lista_norte = [
    [10, 20, 1],
    [10, 25, 0],
    [5, 20, 1],
    [10, 15, 0],
    [15, 30, 1],
    [10, 20, 0],
    [5, 10, 1],
]
unidades_norte = torch.tensor(lista_norte)

total = unidades + unidades_norte
print(total)
```

Output:

```
tensor([[ 40, 100,   3],
        [ 50, 115,   3],
        [ 30,  90,   3],
        [ 45, 105,   2],
        [ 65, 150,   4],
        [ 40, 100,   1],
        [ 25,  60,   2]])
```

The `+` took the Monday-notebooks cell from Centro (30) and added it to the Monday-notebooks cell from Norte (10): 40. And so on for all 21 cells, each paired with its match. This is called operating **element-wise**: each number is operated on with the one in the exact same position in the other tensor.

It's like holding two sheets of grid paper up against the window: each little square lines up directly over its partner. That is why both sheets must have the exact same shape.

Verify against the `for` loop version. Add:

```python
total_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(lista_unidades[d][p] + lista_norte[d][p])
    total_for.append(renglon)

assert total.shape == (7, 3), "Deberían salir 7 días × 3 productos"
assert total.tolist() == total_for, "El tensor y el for deberían dar lo mismo"
print("Todo cuadra")
```

```
Todo cuadra
```

Just as in session 5, `tolist()` converts the tensor back to Python lists to compare it against the `for` loop. First the shape, then the value.

**Watch out: Python lists do not do this.** Try it in the Python interactive shell:

```python
print([1, 2, 3] + [10, 20, 30])
print(torch.tensor([1, 2, 3]) + torch.tensor([10, 20, 30]))
print([1, 2, 3] * 2)
print(torch.tensor([1, 2, 3]) * 2)
```

```
[1, 2, 3, 10, 20, 30]
tensor([11, 22, 33])
[1, 2, 3, 1, 2, 3]
tensor([2, 4, 6])
```

With lists, `+` **concatenates** one list after the other, and `* 2` **repeats** it. With tensors, `+` adds cell by cell, and `* 2` multiplies each number. If one day your result turns out twice as long instead of twice as large, you were operating on lists.

Mateo stares at Fer's last column: 1, 0, 1, 0, 1, 0, 1. Four backpacks in the entire week. The most expensive item in the shop, barely moving, just like in session 3. He jots it down on a Post-it note. He doesn't know why yet.

### Step 3 — Automatic stretching: units times prices

Now for the calculation that matters. Units are a table of shape `(7, 3)`; prices, a list of shape `(3,)`. They do not have the same shape. Even so, add this:

```python
ingresos = unidades * precios
print(ingresos)
print(ingresos.shape)
```

Output:

```
tensor([[1200,  800,  700],
        [1600,  900, 1050],
        [1000,  700,  700],
        [1400,  900,  700],
        [2000, 1200, 1050],
        [1200,  800,  350],
        [ 800,  500,  350]])
torch.Size([7, 3])
```

It worked. Monday: 30 notebooks × $40 = $1,200; 80 pens × $10 = $800; 2 backpacks × $350 = $700. And the same for the other six days.

What happened? PyTorch saw that the price list was missing the days dimension, and **stretched** it: it acted as if the price list were written out seven times, one below the other, one for each day.

```
   unidades (7, 3)          precios (3,)  →  estirado a (7, 3)
  30   80   2                40  10  350
  40   90   3                40  10  350
  25   70   2                40  10  350
  35   90   2      ×         40  10  350
  50  120   3                40  10  350
  30   80   1                40  10  350
  20   50   1                40  10  350
```

And once both sheets were the same size, it multiplied element-wise, just like in Step 2.

In English this is called ***broadcasting***; throughout this course we will also refer to it as **automatic stretching**. The key point: PyTorch **does not copy** the prices seven times into memory. It only computes *as if* they were copied. With 7 days it doesn't matter much; with 7 million purchases, that difference is what keeps your computer from freezing.

It's like the price poster doña Lupe has taped behind the counter, written in marker: there is only one, and it is used to ring up every sale of the day. Nobody rewrites it for every customer.

Verify against the `for` loop:

```python
ingresos_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(lista_unidades[d][p] * lista_precios[p])
    ingresos_for.append(renglon)

assert ingresos.shape == (7, 3), "Los ingresos deberían ser 7 días × 3 productos"
assert ingresos.tolist() == ingresos_for, "El tensor y el for deberían dar lo mismo"
print("Ingresos correctos")
print(ingresos.sum())
```

```
Ingresos correctos
tensor(19900)
```

Look closely at the `for` loop: inside `lista_precios[p]`, only the product index appears, not the day. That is exactly what stretching did: use the same price for every day. And the week's total, $19,900, comes from the `sum` you already know from session 4.

$19,900 at Centro. The calculation has been verified twice: with tensors and with a `for` loop. Mateo now has a tool he can trust. But before using it on Norte, he wants to know where it breaks.

### Step 4 — The rule: when can shapes be stretched?

PyTorch doesn't stretch arbitrary shapes. It follows a rule—the same one used by NumPy (the library from which PyTorch adopted its design) and also followed by TensorFlow. It goes like this:

> **Shapes are compared from right to left. Two dimensions are compatible if they are equal, or if one of them is 1.**

And an extra part: if a shape is missing dimensions on the left, PyTorch prepends ones (1) there, and those ones get stretched.

Apply it to units times prices. Write the shapes aligned **to the right**, just like when adding numbers by hand in elementary school:

```
unidades   7   3
precios        3
           ↑   ↑
           │   └─ 3 contra 3: iguales, compatibles
           └───── 7 contra nada: a precios le falta, se le pone 1 y se estira a 7
resultado  7   3
```

Now a case that does **not** work. Doña Lupe started selling colored pencils at $25 and added them to her price list, but she hasn't counted them in her unit sales table yet.

“The children keep asking for them,” she defends herself. “They're already on the poster. In the tally, not yet.”

Add this, run it, and read the error:

```python
precios_con_colores = torch.tensor([40, 10, 350, 25])   # agregó los colores
print(unidades * precios_con_colores)
```

```
Traceback (most recent call last):
  ...
RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1
```

Read it word for word, as in session 5:

- *The size of tensor a (3)*: the size of tensor a, which is the one to the left of `*`, `unidades`, is 3…
- *must match the size of tensor b (4)*: …and must match the size of tensor b, `precios_con_colores`, which is 4…
- *at non-singleton dimension 1*: …at dimension 1, which is not of size 1. (*Singleton* means “consisting of a single element.” If either one were size 1, it could be stretched; since neither is size 1, there is nothing that can be done.)

Under the rule:

```
unidades               7   3
precios_con_colores        4
                           ↑
                           3 contra 4: ni iguales ni alguno es 1 → error
```

The error is telling you something true about your business: you have four prices and three columns of unit sales. Which column should the price of colored pencils apply to? PyTorch doesn't guess, and that is a good thing. Delete those two lines before continuing.

### Step 5 — A scalar with everything: VAT

The simplest case of stretching is one you have been using without realizing it. Doña Lupe wants to see her revenue with the 16% VAT (IVA) included, “so the accountant doesn't scold me in January.” Here, assume the prices on her list are before VAT; at an actual retail counter, the price displayed to the customer must already include taxes ([Ley Federal de Protección al Consumidor, article 7 bis](https://www.diputados.gob.mx/LeyesBiblio/pdf/LFPC.pdf)). Add:

```python
con_iva = ingresos * 1.16
print(con_iva)
print(con_iva.shape, con_iva.dtype)
print(ingresos.sum(), con_iva.sum())
```

Output:

```
tensor([[1392.,  928.,  812.],
        [1856., 1044., 1218.],
        [1160.,  812.,  812.],
        [1624., 1044.,  812.],
        [2320., 1392., 1218.],
        [1392.,  928.,  406.],
        [ 928.,  580.,  406.]])
torch.Size([7, 3]) torch.float32
tensor(19900) tensor(23084.)
```

The number 1.16 is a scalar, a single number; in session 1 you saw that it has no dimensions—its shape is `()`. Under the rule, it is missing all dimensions, so ones are prepended and it is stretched to `(7, 3)`. That is why it multiplies all 21 cells. The week with VAT: $23,084.

Two things to notice in the output:

- The numbers now have a decimal point (`1392.`) and the `dtype` changed to `float32`. Multiplying integers by a floating-point number produces floats.
- The shape did not change. A scalar never changes the shape of what it multiplies.

Add its verification:

```python
assert con_iva.shape == ingresos.shape, "El IVA no cambia la forma"
assert con_iva[0][0] == 1392, "Lunes, cuadernos: 1,200 × 1.16 = 1,392"
print("IVA correcto")
```

```
IVA correcto
```

### Step 6 — `@`: multiplying and summing in one shot

“And how much did we sell on Friday?” Doña Lupe asks. “Friday is always the big day.”

She wants each day's revenue. You already know how to do that in two steps: multiply units by prices, and then sum each row with `sum(dim=1)`. That combination—“multiply pairs and sum”—is so common that it has its own operator: `@`. It is called **matrix multiplication** (in PyTorch it also exists as `torch.matmul`).

Add:

```python
por_dia = unidades @ precios
print(por_dia)
print(por_dia.shape)
```

Output:

```
tensor([2700, 3550, 2400, 3000, 4250, 2350, 1650])
torch.Size([7])
```

Seven totals, one per day, in a single operation. Work it out by hand for Monday and you will see what `@` did:

```
lunes:   30  80   2
precios: 40  10  350

30 × 40  +  80 × 10  +  2 × 350
 1200    +   800     +   700     =  2700
```

`@` takes **one row** from the table on the left and **the list** on the right, multiplies matching pairs, and **sums everything up**. One row, one number. Seven rows, seven numbers. That is why the shape went from `(7, 3)` to `(7,)`: the product dimension was “consumed” in the summation.

It is exactly what the cashier does with the receipt: items times price, row by row, and the grand total at the bottom. `@` calculates each day's receipt. And Friday, the big day, had the highest total: $4,250.

**The shape rule for `@`**, with 2D tables and 1D lists like those in this session: the last dimension of the left must match the first dimension of the right, and that dimension disappears:

```
unidades @ precios
 (7, 3)    (3,)
     └──┬──┘
   3 y 3: iguales, se consumen
resultado: (7,)
```

Now verify it in three different ways. This is the most important self-check of the session:

```python
# 1) Con for, como lo harías sin tensores
por_dia_for = []
for d in range(7):
    suma = 0
    for p in range(3):
        suma = suma + lista_unidades[d][p] * lista_precios[p]
    por_dia_for.append(suma)

# 2) Multiplicar y luego sumar por renglón
por_dia_dos_pasos = (unidades * precios).sum(dim=1)

assert por_dia.shape == (7,), "Debería salir un número por día"
assert por_dia.tolist() == por_dia_for, "El @ y el for deberían dar lo mismo"
assert por_dia.tolist() == por_dia_dos_pasos.tolist(), "El @ y el sum(dim=1) deberían dar lo mismo"
print("Los tres caminos dan lo mismo")
```

```
Los tres caminos dan lo mismo
```

Look closely at the `for` loop: it has `suma = suma + …` inside. That summation is the difference from the `for` loop in Step 3, which only multiplied. `*` multiplies; `@` multiplies **and sums**.

**Order matters.** Try reversing the sides:

```python
print(precios @ unidades)
```

```
Traceback (most recent call last):
  ...
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 7x3)
```

*mat1 and mat2 shapes cannot be multiplied*: the shapes of matrix 1 and matrix 2 cannot be multiplied. PyTorch treated the prices as a 1 × 3 table and set it against a 7 × 3 table: the last dimension of the left (3) is not equal to the first dimension of the right (7). With `@`, whichever tensor supplies the rows must go on the left.

### Step 7 — The most dangerous trap: `*` and `@` with square tables

It is 9:15 p.m., and Mateo is on the verge of making the mistake almost everyone makes at least once.

Up to this point, every time you used the wrong symbol, PyTorch complained. With square tables, it does not.

Doña Lupe wants to compare two branches across two products: unit sales of notebooks and pens, at Centro and at Norte. At Norte she sells slightly higher, so she has a price table per branch. What she wants is the revenue for each cell: units times price, each paired with its match. That is element-wise, `*`. But imagine someone writes `@` out of habit:

```python
import torch
# Renglones: Centro y Norte. Columnas: cuadernos y plumas.
piezas = torch.tensor([[30, 80],
                       [10, 20]])
precio = torch.tensor([[40, 10],    # en el Centro
                       [45, 12]])   # en Norte, un poco más caro
print(piezas * precio)
print(piezas @ precio)
```

```
tensor([[1200,  800],
        [ 450,  240]])
tensor([[4800, 1260],
        [1300,  340]])
```

**Neither one threw an error.** Both produced a 2 × 2 table. But only the first is what doña Lupe wanted: 30 notebooks × $40 = $1,200. The second claims that Centro took in $4,800 for notebooks—four times the actual figure. Where did that come from? With two 2D tables, `@` takes a row from the left and a **column** from the right, performing the receipt's “multiply and sum”: Centro's row (30, 80) against the notebook price column (40, 45) yields 30 × 40 + 80 × 45 = 4,800. It mixed Centro's pen sales with Norte's notebook prices. It makes zero business sense.

Why was there no error? Because with square tables both rules are satisfied simultaneously: `(2, 2)` and `(2, 2)` are identical, so `*` works; and the last dimension of the left (2) equals the first dimension of the right (2), so `@` also works.

It is the most dangerous trap because **an error that crashes is fixed in five minutes; an error that runs silently makes it all the way to the executive report.** Or right onto Licenciado Ortega's desk. Mateo pictures himself showing up at the bank on the 23rd with a table that looks flawless and is wrong. A chill runs down his spine.

The defense is the same one used throughout this series: check against a `for` loop, even if only for a single row:

```python
lista_piezas = [[30, 80], [10, 20]]
lista_precio = [[40, 10], [45, 12]]
ingreso_for = []
for s in range(2):
    renglon = []
    for p in range(2):
        renglon.append(lista_piezas[s][p] * lista_precio[s][p])
    ingreso_for.append(renglon)
print(ingreso_for)
```

```
[[1200, 800], [450, 240]]
```

It matches `*`, not `@`. And to choose the right operator before writing, ask yourself a single question:

| If you want… | Use | What happens to the shape |
|---|---|---|
| Each number paired with its match (price × units of the same cell) | `*` | Stays the same (or takes the broadcasted shape) |
| Multiplying pairs **and summing them** (the receipt total) | `@` | The summed dimension disappears |

---

**And in TensorFlow?** Broadcasting follows the same rule and gives the same result. The `@` operator is stricter: with a 2D table and a 1D list it raises an error (*different ndims*, different number of dimensions), and for that case a different function is used, `tf.linalg.matvec`. Same ideas, different manners.

---

👉 **If you come from Business Administration:** you have already done this in spreadsheets. When you write `=B2*$F$1` and drag the formula down, the fixed price in `$F$1` is used across all rows: that is broadcasting. And `@` is Excel's `SUMPRODUCT` (`SUMAPRODUCTO`) function: multiply pairs and sum. The trap in Step 7 also has its Excel counterpart: a formula that doesn't display `#VALUE!` (`#¡VALOR!`) but sums the wrong column. That is why, in any report, you check one row by hand before presenting it.

---

## 10:40 p.m. — What doesn't add up

Mateo does with Norte the same thing he did with Centro: Fer's unit counts times the same prices, and then the sum. One line. (You will do that calculation yourself in Exercise 23; I won't spoil the number.)

He compares it with Ortega's email. He runs it again. He checks it against the `for` loop. Same result.

“Auntie. What Norte sold the entire week doesn't even come to half of what the terminal charged.”

“Maybe almost everyone paid by card.”

“Even if *everyone* had paid by card. Even if not a single person had paid a single peso in cash. The terminal processed $16,850, and Norte didn't have anywhere near that much merchandise go out the door.”

Doña Lupe stops counting the coins.

“Then what did they charge for?”

“Nothing. Charges with no merchandise behind them.”

Outside, a truck rumbles by and the metal shutter rattles.

“Fer,” Doña Lupe says, slowly. “She is the one who rings up sales at Norte.”

Mateo had already thought of that. That was why, while the `for` loop was running, he opened the stationery shop's email inbox. The one nobody checks. The one doña Lupe says “is nothing but junk mail.” He searched for “Norte.” He searched for “Fernanda.”

There it was. Tuesday, September 22, 6:12 p.m. Subject: **Discrepancies: terminal doesn't match inventory**.

“Doña Lupe: I did the weekly cutoff and the terminal shows more charges than what shows up in my inventory. I counted twice. For what the terminal says, I ought to be missing items from the shelf, and I’m not missing any. I think something is wrong with the new terminal. Should I notify the bank or notify you? Fer.”

Unanswered. Unopened. Since September.

“She did the exact same math I did,” Mateo says. “Units times prices against what was charged. With a calculator. Two and a half months ago.”

Doña Lupe takes off her glasses and rubs her eyes.

“And here I was thinking it was her.”

“We were thinking with suspicions, Auntie. The numbers say something else. If Fer were stealing, she wouldn’t write to warn you that things don’t add up.”

“I’ll call her tomorrow.” A pause. “And I’ll bring her pan dulce—conchas. Chocolate ones, the ones she likes.”

That night, Mateo learns what no `assert` could ever teach him: **you make decisions with data, not with suspicions.**

---

## Exercises for session 6

The solutions are at the end of this installment. Try them first; the `assert` statements will tell you if you got them right.

**Exercise 22 — Predict before running.**

a) For each pair, write down on paper whether they can be broadcasted and, if so, what shape results. Use the rule from Step 4: align to the right and compare. Then run it:

`torch.ones(...)` creates a tensor full of ones with the shape you specify. Here we only use it to test shapes: the values don't matter.

```python
print((torch.ones(7, 3) * torch.ones(3)).shape)
print((torch.ones(2, 7, 3) * torch.ones(7, 1)).shape)
print((torch.ones(7, 1) * torch.ones(1, 3)).shape)
print((torch.ones(7, 3) * torch.tensor(2)).shape)
print((torch.ones(2, 7, 3) * torch.ones(2)).shape)
```

Example of what a worked solution looks like: `(7, 3)` and `(3,)` → align: 3 with 3, equal; the list is missing the 7, it is stretched → yields `(7, 3)`.

b) With this table, compute **by hand**, on paper, `m * m` and `m @ m`. Then run it and compare.

```python
m = torch.tensor([[2, 1],
                  [0, 3]])
print(m * m)
print(m @ m)
```

Hint for `@`: the top-right cell comes from the top row, (2, 1), against the right column, (1, 3): 2 × 1 + 1 × 3.

**Exercise 23 — Norte's revenue.** Fill in the blanks. Use `unidades_norte` from Step 2 and the same `precios`.

```python
ingresos_norte = ____ * ____

ingresos_norte_for = []
for d in range(7):
    renglon = []
    for p in range(3):
        renglon.append(____)
    ingresos_norte_for.append(renglon)

assert ingresos_norte.shape == (7, 3), "Norte también es 7 días × 3 productos"
assert ingresos_norte.tolist() == ingresos_norte_for, "El tensor y el for deberían dar lo mismo"
con_iva = ____
print(ingresos_norte.sum(), con_iva.sum())
print("Ejercicio 23 superado")
```

Before running it, predict: how much revenue came from backpacks on Tuesday at Norte? Look at the unit sales table.

**Exercise 24 — El Buen Fin.** For El Buen Fin, doña Lupe discounts $5 off notebooks, nothing off pens, and $50 off backpacks. Assume she sells the same unit counts as this week.

1. Create a `descuento` tensor with the three discounts (as negative numbers) and add it to `precios` to obtain `precios_bf`.
2. Using `@`, calculate `por_dia_bf`, the revenue for each day with the Buen Fin prices.
3. Write two `assert` statements: one for the shape, and one that compares your `@` calculation against the version with `*` and `sum(dim=1)`.
4. Calculate how much revenue was forgone each day due to the discounts, and how much for the entire week.

Challenge: the revenue forgone each day can be obtained with a single `@`, without using `por_dia_bf`. Which one?

**Exercise 25 — Write the rule yourself.** Without PyTorch, using plain Python only. Write a function `se_pueden_estirar(forma_a, forma_b)` that takes two tuples and returns `True` if they can be broadcasted to operate together and `False` otherwise. It must pass these `assert` statements:

```python
assert se_pueden_estirar((7, 3), (3,)) == True
assert se_pueden_estirar((7, 3), (7,)) == False
assert se_pueden_estirar((2, 7, 3), (7, 1)) == True
assert se_pueden_estirar((7, 3), (4,)) == False
assert se_pueden_estirar((7, 1), (1, 3)) == True
assert se_pueden_estirar((7, 3), ()) == True
print("Ejercicio 25 superado")
```

Hints: `forma[-1]` is the last dimension of a tuple and `forma[-2]` the second to last; that is how you traverse from right to left. You only need to compare as many dimensions as the shorter shape has.

---

## Common errors (and how to spot them)

| Error | What it looks like | How to spot it | What to do |
|---|---|---|---|
| A price list with one extra (or missing) product | `RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1` | The two numbers in parentheses differ and neither is 1 | Print the `.shape` of both before operating; fix the data, not the calculation |
| Dividing a `(7, 3)` table by a `(7,)` daily total | `RuntimeError: The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 1` | The rule aligns to the right: it placed the 7 days under the 3 products | Reshape it into a column: `por_dia.reshape(7, 1)` |
| Using `@` where you wanted `*` with square tables | **No error.** A table of the expected shape comes out with numbers that don't add up | Compute one cell by hand or compare against a `for` loop | Ask yourself: do I want element-wise pairs (`*`) or summed pairs (`@`)? |
| Swapping the sides of `@` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 7x3)` | The last dimension of the left does not match the first of the right | The one that sets the rows goes on the left: `unidades @ precios`. Transposing with `.T` doesn't help here: a `(3,)` list is still `(3,)` |
| Mixing integers and floating-point numbers in `@` | `RuntimeError: addmv input tensors must have the same dtype, but got Float, Long, and Float` | Check the `.dtype` of both | Convert the integer tensor with `.float()` |
| Expecting a Python list to operate like a tensor | `[1, 2, 3] * 2` gives `[1, 2, 3, 1, 2, 3]` | The result turns out longer, not larger | Convert it first with `torch.tensor(...)` |

---

## Review for next session

Session 7 begins with these three questions. Answer them without looking at the lesson:

1. When can two shapes be “stretched” (broadcasted) to operate together?
2. What is the difference between `*` and `@`?
3. What shape results from a `(7, 3)` table `@` a `(3,)` vector?

🔁 **Next week:** Mateo can now prove that those charges did not originate from the stationery shop. What he doesn't know is where they came from. To find out, he will build a fraud detector that learns from data: you feed it thousands of synthetic transactions with a hidden rule, and the model discovers it on its own, purely through `@` and automatic stretching, today's two operations.

*The charges were not coming from the stationery shop. They were coming from somewhere else. And the bank only saw the terminal.*

---

## Solutions (don't look before trying)

**Warm-up review.**

1. When the total number of elements is identical before and after: a `(12,)` shape fits into `(4, 3)` because 12 = 4 × 3, but not into `(5, 3)`.
2. To have PyTorch calculate that dimension size for you, using whatever elements remain. There can only be one `-1` per `reshape`.
3. Because `reshape` rearranges the numbers in the order they are stored in memory, row by row; transposing the table so that rows become columns is `.T`.

**Exercise 22.**

a) Actual output:

```
torch.Size([7, 3])
torch.Size([2, 7, 3])
torch.Size([7, 3])
torch.Size([7, 3])
Traceback (most recent call last):
  ...
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 2
```

- `(7, 3)` and `(3,)`: 3 with 3; the right one has a 1 prepended and stretches to 7 → `(7, 3)`.
- `(2, 7, 3)` and `(7, 1)`: 3 with 1, stretches; 7 with 7; the right one is missing the 2 → `(2, 7, 3)`.
- `(7, 1)` and `(1, 3)`: both stretch → `(7, 3)`.
- `(7, 3)` and a scalar `()`: the scalar is missing all dimensions, stretches → `(7, 3)`.
- `(2, 7, 3)` and `(2,)`: the 2 aligns against the 3 on the right, not against the 2 on the left. 3 against 2: error. This is the deceptive one: the 2 “looks like” the number of branches, but the rule compares from the right. Notice that the error says *dimension 2*: in a 3D shape, dimension 2 is the last one.

b)

```
tensor([[4, 1],
        [0, 9]])
tensor([[4, 5],
        [0, 9]])
```

`m * m`: each number squared. `m @ m`: top-left, 2 × 2 + 1 × 0 = 4; top-right, 2 × 1 + 1 × 3 = 5; bottom-left, 0 × 2 + 3 × 0 = 0; bottom-right, 0 × 1 + 3 × 3 = 9. Notice how tricky this is: both tables match in three out of four cells. If you only checked the top-left cell, you would think they gave the same result.

**Exercise 23.** The blanks are `unidades_norte` and `precios`; `lista_norte[d][p] * lista_precios[p]`, and `ingresos_norte * 1.16`. The output:

```
tensor(5400) tensor(6264.)
Ejercicio 23 superado
```

Backpack sales on Tuesday at Norte: 0 units, so $0. If you print `ingresos_norte`, Tuesday is `[400, 250, 0]`.

**Exercise 24.**

```python
descuento = torch.tensor([-5, 0, -50])
precios_bf = precios + descuento
por_dia_bf = unidades @ precios_bf
print(precios_bf)
print(por_dia_bf)

assert por_dia_bf.shape == (7,), "Debería salir un número por día"
assert por_dia_bf.tolist() == (unidades * precios_bf).sum(dim=1).tolist(), "El @ y el sum(dim=1) deberían dar lo mismo"

dejo_de_ganar = unidades @ precios - por_dia_bf
print(dejo_de_ganar)
assert dejo_de_ganar.tolist() == (-(unidades @ descuento)).tolist()
print(dejo_de_ganar.sum())
print("Ejercicio 24 superado")
```

```
tensor([ 35,  10, 300])
tensor([2450, 3200, 2175, 2725, 3850, 2150, 1500])
tensor([250, 350, 225, 275, 400, 200, 150])
tensor(1850)
Ejercicio 24 superado
```

`precios + descuento` is element-wise: two lists of shape `(3,)`. Over the week, she gave up $1,850 in revenue. The challenge: `unidades @ descuento` yields the revenue forgone each day as a negative value, because the discounts are negative; the leading `-` turns it positive. That is what the final `assert` verifies.

**Exercise 25.** One possible solution:

```python
def se_pueden_estirar(forma_a, forma_b):
    cuantos = min(len(forma_a), len(forma_b))
    for i in range(1, cuantos + 1):
        a = forma_a[-i]
        b = forma_b[-i]
        if a != b and a != 1 and b != 1:
            return False
    return True
```

With the `assert` statements from the problem prompt below it, the output is:

```
Ejercicio 25 superado
```

It traverses from right to left using negative indices: `-1`, `-2`… It only compares as many dimensions as the shorter shape has, because the other shape is merely missing dimensions on the left, and those can always be stretched. If it finds a pair that are not equal and neither of which is 1, there is nothing that can be done.

---

## What's next

In session 7, you bring together everything you have learned so far: a tensor of transactions, an `@` against a weight matrix, and automatic stretching to add an adjustment, repeated until the model learns. If you want to jump ahead, redo Step 6 by hand for Tuesday: if you get $3,550 without looking, you are ready.

---

*I write from what we see every day at Hábil, where we build systems for banking, insurance, and enterprise in Mexico. If at your work or your family's business you run into a problem like this, let's talk: [habil.mx](https://www.habil.mx/en/)*
