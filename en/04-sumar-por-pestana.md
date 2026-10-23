# Tensors from scratch (4): summing by tab

*Totals, averages, and the best day of the week, pulled from a tensor in a single line, and the rule that never fails: the axis you sum disappears.*

*Tensors from scratch · lesson A, session 4 of 8*

*Setting: the characters and the story are fictional. Technical details, historical facts, and real figures include their sources.*

**Monday, November 23, 7:40 p.m. Back room of the Centro branch.**

The metal shutter was already halfway down. It smelled of fresh cardboard and reheated coffee. Mateo had his laptop on a box of lined notebooks, and doña Lupe, standing behind him, hadn't taken off her sweater.

“Thirty days left,” she said. “Licenciado Ortega called me again. He wants ‘something concrete.’”

“Almost there, Auntie.”

“That's what you said last Monday. And the one before that.”

Mateo didn't answer. He had spent a week slicing the stationery store tensor: a day here, a product there. He had seen strange things. Friday at Norte, as slow as a Tuesday. Backpacks, dead in the water. But those were pieces. And you don't convince a bank with pieces.

“I need to see everything together,” he said. “Not one day. Not one product. The entire week for each branch, in a single number.”

“And the computer does that?”

“It does it in one line. If I tell it properly what to combine.”

Doña Lupe sat down on an upturned paint bucket.

“I'm telling you right now, mijo: Norte is on the avenue, right next to the bus stop. Twice as many people walk through that door as this one. If it's selling less than here, I want to know how much less.”

Mateo placed his fingers on the keyboard. Up until today he had learned to **extract slices** from the tensor. Today was the opposite: **summarizing**. Combining many numbers into a single one, along whatever axis he chose.

Outside, a bus rumbled past and the shutter rattled.

“Go on then,” said doña Lupe. “Let your tensor speak.”

*This is the last session of lesson A. You are going to do the same thing Mateo did, step by step.*

## Warm-up review

Answer without looking at session 3. The answers are at the end, in the Solutions.

1. What shape does `papeleria[:, 4]` have and what does it represent?
2. What is the difference between `papeleria[0]` and `papeleria[0:1]` in terms of shape?
3. How do you get the backpack sales for both branches across all days?

> **The code for this session** is in the course's public repository, in the `sesion-04` folder: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-04). Type it yourself first, because that's how you learn; the repository is for comparing when something doesn't turn out the same for you. Each program comes with the output it should give you.

## Objectives of this session

By the end, you will be able to do these four things. Next to each one is the exercise that demonstrates it.

| # | By the end you can… | Demonstrated by |
|---|---|---|
| 1 | **Predict** what shape remains after summing a tensor over one or more axes, before running the code | Exercise 14 |
| 2 | **Calculate** totals and averages by axis using `sum` and `mean`, and **fix** the `mean` error with integers | Exercises 15 and 17 |
| 3 | **Find** the maximum or minimum along an axis, and **what position** it is in, with `max`, `min`, `argmax`, and `argmin` | Exercises 16 and 17 |
| 4 | **Verify** any tensor result against the same calculation done with a `for` loop | Exercises 15, 16, and 17 |

---

## What you need

- The virtual environment from session 2, with PyTorch installed, activated.
- A `papeleria.py` file with doña Lupe's data. If you already have it from session 3, use it. If not, copy it like this:

```python
import torch

# eje 0 = sucursal (0 Centro, 1 Norte)
# eje 1 = día      (0 lunes ... 6 domingo)
# eje 2 = producto (0 cuadernos, 1 plumas, 2 mochilas)
centro = [
    [1200, 800, 500],    # lunes
    [1500, 900, 700],    # martes
    [1000, 700, 500],    # miércoles
    [1300, 900, 600],    # jueves
    [2000, 1200, 700],   # viernes
    [1200, 800, 400],    # sábado
    [800, 500, 300],     # domingo
]
norte = [
    [300, 200, 100],
    [400, 250, 150],
    [350, 200, 100],
    [300, 150, 100],
    [500, 300, 200],
    [300, 200, 100],
    [200, 100, 100],
]
papeleria = torch.tensor([centro, norte])
```

Each step today is a new file in the same folder, starting with `from papeleria import ...`. That way you don't copy the data every time.

---

## Why it matters

A tensor stores the detail. Every cell, every day, every product. But the detail neither accuses anyone nor saves anyone: what actually helps are **answers**. How much did each branch sell? What day sells the most? How much comes in on average per day?

Mateo realized something that night: all three questions are the same operation. **Combining many numbers into one, along a single direction.** In Excel, it is the total column at the end of each row, or the total row at the bottom. In a tensor it is called **reducing**, because fewer numbers remain after the operation.

And it's not a stationery store trick. Artificial intelligence systems reduce all the time. If a bank has the purchases of 1,000 customers across 30 days, with 5 data points per purchase (shape `(1000, 30, 5)`, as in session 1), and averages over days, it obtains how each customer typically shops: a `(1000, 5)` table. A purchase that strays far from that average is the one worth reviewing. Exactly what Licenciado Ortega is doing, on the other end of the phone, with the Norte terminal.

In retail, the exact same thing happens on a massive scale. In 2020, an international forecasting competition called M5 asked participants to predict daily sales for 10 stores of an American retail chain (public data from the Kaggle M5 competition) across three U.S. states: 42,840 sales time series organized hierarchically (source: [M5 Forecasting - Accuracy, Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy); results in *International Journal of Forecasting*, [doi:10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013)). Summing along an axis, as you will do today, is how you move from one level up to the next.

## The case: doña Lupe's stationery store, once again

The `papeleria` tensor has shape `(2, 7, 3)`: 2 branches, 7 days, 3 products. It's the same 42 numbers as always. Tonight Mateo is going to ask them the questions that truly keep doña Lupe awake at night.

### Step 1 — The grand total

Mateo started with the simplest thing, to warm up: how much did both branches sell together across the entire week? Create `paso1.py`:

```python
from papeleria import papeleria

total = papeleria.sum()
print(total)
print(total.shape)
print(total.item())
```

Output:

```
tensor(23100)
torch.Size([])
23100
```

`sum()` with nothing inside combines **all 42 numbers** into a single one: 23,100 pesos. Notice the shape: `torch.Size([])`, empty. It is a scalar, like the `()` from session 1, and `.item()` gives it to you as a regular Python number. It checks out: 18,500 from Centro plus 4,600 from Norte.

“Twenty-three thousand one hundred?” Doña Lupe pursed her lips. “I already know that, mijo. The notebook tells me that much.”

She was right. A single number hides everything. She wants to know **how much each branch sold**, **each day**, **each product**. For that, you tell `sum` along which axis to combine.

### Step 2 — A single tab: summing across and down

Before tackling the entire cube, Mateo did what any detective in a hurry with no desire to make mistakes would do: test on familiar ground. A single tab, the one for Centro. A table of 7 days × 3 products. With a table, there are two ways to sum, and you know them from Excel:

```
             Cuad.  Plumas  Moch.   │ total del día
   Lun       1200    800    500     │  2500
   Mar       1500    900    700     │  3100
   ...                              │  ...
   Dom        800    500    300     │  1600
   ─────────────────────────────────┘
   total     9000   5800   3700
   del producto
```

- **Across** (the column on the right): you combine the 3 products for each day. 7 numbers remain, one per day.
- **Down** (the bottom row): you combine the 7 days for each product. 3 numbers remain, one per product.

In PyTorch, you tell it which one using `dim=`, which stands for **dimension**, meaning axis. Remember how the axes of a tab are numbered: axis 0 is the days (the rows) and axis 1 is the products (the columns).

Create `paso2.py`:

```python
from papeleria import papeleria, centro

pestana_centro = papeleria[0]          # la pestaña del Centro: 7 días × 3 productos
print(pestana_centro.shape)

por_dia = pestana_centro.sum(dim=1)    # suma a lo largo del eje 1 (productos)
print(por_dia)
print(por_dia.shape)

# Lo mismo con for, como ya sabes hacerlo
por_dia_con_for = []
for renglon in centro:
    por_dia_con_for.append(sum(renglon))
print(por_dia_con_for)

assert por_dia.shape == (7,), "Esperaba un número por día: forma (7,)"
assert por_dia.tolist() == por_dia_con_for, "La suma con tensores no coincide con la del for"
print("Por día: cuadra")
```

Output:

```
torch.Size([7, 3])
tensor([2500, 3100, 2200, 2800, 3900, 2400, 1600])
torch.Size([7])
[2500, 3100, 2200, 2800, 3900, 2400, 1600]
Por día: cuadra
```

“I know those numbers by heart,” doña Lupe said. “Monday two thousand five hundred, Tuesday three thousand one hundred…”

Exactly. They are the `por_dia` list she dictated to Mateo in session 1. The tensor and the `for` loop yield the exact same thing. First test passed: the tool doesn't lie on familiar ground.

Now look at the shape, which is today's key lesson. You went in with `(7, 3)`. You summed with `dim=1`. You came out with `(7,)`. **The 3, which was axis 1, disappeared.** Makes sense: you combined the three products into one, so there is no longer a “products axis.”

A new tool in that code: `.tolist()`. It's related to `.item()`: `.item()` converts a single-number tensor into a Python number, and `.tolist()` converts a multi-number tensor into a Python list. We use it so that `assert` can compare list against list.

Now down the columns. In the Python terminal, with `from papeleria import papeleria`:

```
>>> papeleria[0].sum(dim=0)
tensor([9000, 5800, 3700])
>>> papeleria[0].sum(dim=0).shape
torch.Size([3])
```

You went in with `(7, 3)`, summed with `dim=0`, and came out with `(3,)`: **the 7, which was axis 0, disappeared.** You combined the seven days and were left with a total per product. (In exercise 15 you will verify something similar with a `for` loop.)

Mateo stared at the two shapes. `(7,)`. `(3,)`. And he wrote on a Post-it note, which he stuck to the laptop lid, the rule for the entire session. Learn it by heart:

> **The axis you reduce disappears from the shape.** The rest remain, in the same order.

A trick to avoid getting confused: `dim=` doesn't say **what you keep**, it says **what you combine**. `dim=1` does not mean “give me the products”; it means “combine the products.” That is why the products disappear.

“So whatever you combine goes away,” doña Lupe summed up. “Just like the Christmas bonus.”

### Step 3 — The full cube: three ways to sum

Now for the full cube. `(2, 7, 3)`: three axes, so there are three ways to sum just one. Mateo didn't run anything yet. First he grabbed a pen—one of the ones his aunt sells—and predicted each shape on a napkin. Do it yourself too, using the rule: cross out the number of the axis you combine.

| You combine | Which axis | Resulting shape | What it means |
|---|---|---|---|
| branches | `dim=0` | ~~2~~, 7, 3 → `(7, 3)` | each day and product, with both branches combined |
| days | `dim=1` | 2, ~~7~~, 3 → `(2, 3)` | each branch and product, across the entire week |
| products | `dim=2` | 2, 7, ~~3~~ → `(2, 7)` | each branch and day, with all products combined |

Now verify it. Create `paso3.py`:

```python
from papeleria import papeleria

sin_sucursal = papeleria.sum(dim=0)   # junta las dos pestañas
print(sin_sucursal.shape)
print(sin_sucursal)

sin_dia = papeleria.sum(dim=1)        # junta los 7 días
print(sin_dia.shape)
print(sin_dia)

sin_producto = papeleria.sum(dim=2)   # junta los 3 productos
print(sin_producto.shape)
print(sin_producto)
```

Output:

```
torch.Size([7, 3])
tensor([[1500, 1000,  600],
        [1900, 1150,  850],
        [1350,  900,  600],
        [1600, 1050,  700],
        [2500, 1500,  900],
        [1500, 1000,  500],
        [1000,  600,  400]])
torch.Size([2, 3])
tensor([[9000, 5800, 3700],
        [2350, 1400,  850]])
torch.Size([2, 7])
tensor([[2500, 3100, 2200, 2800, 3900, 2400, 1600],
        [ 600,  800,  650,  550, 1000,  600,  400]])
```

Read them like doña Lupe:

- **`dim=0`** stacks one tab on top of the other and sums cell by cell. On Monday, in notebooks, Centro sold 1,200 and Norte 300: together, 1,500. That is the first cell. Branches disappeared: it's as if doña Lupe had a single large store.
- **`dim=1`** combines the week. The top row is Centro: 9,000 in notebooks, 5,800 in pens, 3,700 in backpacks, the same ones from step 2. The bottom row is Norte. Days disappeared.
- **`dim=2`** combines products. The top row is Centro's `por_dia` that you already know; the bottom row, Norte's. Products disappeared.

All three napkin predictions checked out. That's the power of the rule: **you know the shape before running the code.** And if the shape that comes out isn't what you predicted, you know you combined the wrong axis.

But Mateo wasn't looking at the shapes anymore. He was looking at the bottom row of `dim=1`. *2350, 1400, 850.* Norte's entire week, product by product. It didn't even reach what Centro sells in notebooks alone.

He was still missing one calculation. The one he had come to do.

### Step 4 — Combining two axes at once: each branch's total

Doña Lupe's question, the one about the door twice as many people walk through: how much did each branch sell? For that, you need to combine **days and products**: two axes at once. `sum` accepts a tuple of axes. Create `paso4.py`:

```python
from papeleria import papeleria, centro, norte

por_sucursal = papeleria.sum(dim=(1, 2))   # junta días y productos
print(por_sucursal.shape)
print(por_sucursal)

por_sucursal_con_for = []
for pestana in [centro, norte]:
    total = 0
    for renglon in pestana:
        for venta in renglon:
            total = total + venta
    por_sucursal_con_for.append(total)
print(por_sucursal_con_for)

assert por_sucursal.shape == (2,), "Esperaba un número por sucursal: forma (2,)"
assert por_sucursal.tolist() == por_sucursal_con_for, "No coincide con el for"
assert por_sucursal[0].item() == 18500, "El Centro debería sumar 18,500"
print("Por sucursal: cuadra")
```

Output:

```
torch.Size([2])
tensor([18500,  4600])
[18500, 4600]
Por sucursal: cuadra
```

18,500 for Centro and 4,600 for Norte.

The numbers had been on the sheet since session 1. But they had never placed them like this, side by side, with Norte's front door in mind.

Doña Lupe stepped closer to the screen. She didn't say anything for a long time.

“Four times,” she finally said. “Centro sold about four times what Norte did; Norte, roughly a fourth. And twice as many people walk through its door.”

“Auntie, this doesn't say why. It only says how much.”

“Uh-huh.” She was already looking at her phone. “And who balances Norte's register every night, Mateo?”

He knew the answer. Everyone knew. Fer.

The rule, meanwhile, remains unchanged: 2, ~~7~~, ~~3~~ → `(2,)`. You crossed out two numbers, two axes disappeared.

The `for` approach required **three loops**, one inside another. The tensor, a single line. And it's not just shorter to write: on large arrays, a Python `for` loop can take much longer than PyTorch's `sum`. How much longer depends on your computer, your version, and the array size, so measure it on your own machine before quoting a number.

**Watch out if you do it in two steps.** Mateo, his hands cold, wanted to run the calculation another way, just in case: sum one axis first and then the other. But beware: **after the first sum, the axes are renumbered.** Look at this attempt. Create `paso4c.py`:

```python
from papeleria import papeleria

paso_a = papeleria.sum(dim=1)     # quito los días: queda (2, 3)
print(paso_a.shape)
paso_b = paso_a.sum(dim=2)        # ¿quito los productos?
```

Output:

```
torch.Size([2, 3])
Traceback (most recent call last):
  File ".../paso4c.py", line 5, in <module>
    paso_b = paso_a.sum(dim=2)        # ¿quito los productos?
             ^^^^^^^^^^^^^^^^^
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

Read it word for word: “dimension out of range: expected something between -2 and 1, but got 2.” After removing days, `paso_a` has shape `(2, 3)`: only two axes, 0 and 1. Products are no longer axis 2; they are now axis 1. (Negative numbers count axes from the end; you don't need them for now.)

The easy solution: do both in a single call, `dim=(1, 2)`. There, the numbers always refer to the original shape.

Mateo went back to the one-line call. 4,600. The number didn't budge.

In the other corner of the back room, doña Lupe was dialing a number. She let it ring. Hung up. Dialed again.

### Step 5 — Daily average, and the error you are bound to see

“Get me how much Norte sells on a normal day,” doña Lupe said without putting down the phone. “So I can say it to her face.”

In other words: how much is sold **on average** per day at each branch? The average is calculated using `mean`. It works just like `sum`, with `dim=`. First you get the sales for each day (combining products) and then you average across the 7 days. Create `paso5a.py`:

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
print(ventas_diarias.dtype)
promedio_diario = ventas_diarias.mean(dim=1)
```

Output:

```
torch.int64
Traceback (most recent call last):
  File ".../paso5a.py", line 5, in <module>
    promedio_diario = ventas_diarias.mean(dim=1)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
```

It blew up.

“Now what?” asked doña Lupe.

“Nothing serious, Auntie. It's complaining that I gave it integers.”

“Well, like everyone else. I like them whole too.”

Read it slowly, translated:

- *mean(): could not infer output dtype*: “`mean` couldn't figure out what type to make the result.”
- *Input dtype must be either a floating point or complex dtype*: “what goes into it must have decimals” (*floating point* are numbers with a decimal point; *complex* are another kind of number you won't use here).
- *Got: Long*: “and you gave me `Long`.” `Long` is how PyTorch internally refers to `int64`, the integer type you saw in session 2. That's why we printed `dtype` beforehand: `torch.int64`.

Why does it refuse? Because an average almost never comes out whole: 18,500 divided by 7 is 2,642.857…, and an integer tensor has nowhere to store the “.857”. Instead of silently truncating it, PyTorch halts and warns you. Mateo liked that. Too many things had been happening under the table in this story already.

There are two ways to fix it.

**Fix 1: convert to floating-point with `.float()`.** Create `paso5b.py`:

```python
from papeleria import papeleria, centro

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
decimales = ventas_diarias.float()
print(decimales.dtype)
print(decimales)

promedio_diario = decimales.mean(dim=1)     # quita el eje de los días
print(promedio_diario.shape)
print(promedio_diario)

# Con for: el total del Centro entre 7 días
total_centro = 0
for renglon in centro:
    total_centro = total_centro + sum(renglon)
print(total_centro / 7)

assert promedio_diario.shape == (2,), "Esperaba un promedio por sucursal"
assert round(promedio_diario[0].item(), 2) == round(total_centro / 7, 2), "No coincide con el for"
print("Promedio diario: cuadra")
```

Output:

```
torch.float32
tensor([[2500., 3100., 2200., 2800., 3900., 2400., 1600.],
        [ 600.,  800.,  650.,  550., 1000.,  600.,  400.]])
torch.Size([2])
tensor([2642.8572,  657.1429])
2642.8571428571427
Promedio diario: cuadra
```

`.float()` does not change the numbers, only how they are stored: notice the dot after each one (`2500.`). Now they are `float32`, floating-point decimals. Centro sells on average $2,642.86 per day, and Norte, $657.14.

“Six hundred fifty-seven pesos,” doña Lupe repeated. “On that corner. A chewing gum stand sells that much.”

Two details in that code:

- The shape: `(2, 7)` with `dim=1` yielded `(2,)`. Days disappeared, as always.
- The `assert` compares by **rounding to two decimal places**, because the tensor says 2642.8572 and the `for` loop says 2642.8571428571427. Neither is wrong: the tensor uses `float32` and Python divides with a 64-bit `float`. Both are binary approximations, but `float32` preserves less precision. Without rounding, the `assert` would fail even though the calculation is correct. Careful: we round only to compare or display; for real accounting, store amounts in cents as integers or use decimal arithmetic, not `float32`.

**Fix 2: create the tensor with decimals from the start.** If doña Lupe's data came with decimals, the tensor would be born with decimals and `mean` would work directly (for exact currency, whole cents are better). Create `paso5c.py`:

```python
import torch

con_centavos = torch.tensor([2500.0, 3100.5, 2200.0])
print(con_centavos.dtype)
print(con_centavos.mean())
```

Output:

```
torch.float32
tensor(2600.1667)
```

Having a decimal point in the numbers is enough for PyTorch to choose `float32` on its own. For the stationery store, which already exists with integers, the most practical approach is fix 1: `.float()` right before `mean`.

### Step 6 — The best day: `max` and `argmax`

Mateo had a suspicion different from his aunt's, and he wanted to test it before saying anything. If someone were wildly pocketing sales, Norte would behave erratically: good days when least expected, random dead days. So he asked the final question of the night: which day sells the most at each branch? Here there are **two** answers you care about: **how much** was sold that day (the value) and **which day** it was (the position). Create `paso6.py`:

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
resultado = ventas_diarias.max(dim=1)
print(resultado)
```

Output:

```
torch.return_types.max(
values=tensor([3900, 1000]),
indices=tensor([4, 4]))
```

This looks unusual, but it's simple. When you pass `dim=` to `max`, it doesn't return a plain tensor, but **a container with two things**:

- `values`: the highest sales figure for each branch. 3,900 at Centro, 1,000 at Norte.
- `indices`: **what position** on the axis that sale was at. 4 and 4.

And what is position 4 on the days axis? Monday 0, Tuesday 1, Wednesday 2, Thursday 3… **Friday**. Both branches sell the most on Friday.

Just like with a Python tuple, you can unpack both at once, each into its own variable. And with a list of names, you translate the index into words. Create `paso6b.py`:

```python
from papeleria import papeleria

dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
sucursales = ["Centro", "Norte"]

ventas_diarias = papeleria.sum(dim=2)
valores, indices = ventas_diarias.max(dim=1)
print(valores)
print(indices)

for s in range(2):
    dia = dias[indices[s].item()]
    print(sucursales[s], "vendió más el", dia, ":", valores[s].item())
```

Output:

```
tensor([3900, 1000])
tensor([4, 4])
Centro vendió más el viernes : 3900
Norte vendió más el viernes : 1000
```

The shape rule applies here too: you started with `(2, 7)`, combined days with `dim=1`, and both `valores` and `indices` have shape `(2,)`: one per branch.

Mateo leaned back in his chair. The same day for both. Norte wasn't behaving like an out-of-control store. It was behaving like Centro, but miniaturized. As if every single day, more or less evenly, was missing a piece.

That doesn't look like someone dipping their hand into the till on random days. It looks like something else. But what? Mateo didn't yet have the tools to figure it out, so he said nothing.

In exercise 16 you will verify a result like this against a `for` loop: it is longer, but you already know how to write it.

**Its relatives.** If you only want the position without the value, use `argmax` (“the argument of the maximum”, that is, where it is located). If you want the lowest, `min` and `argmin` work the same way. And if you omit `dim`, `max` searches across the entire tensor and returns a single number. Create `paso6c.py`:

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)
print(ventas_diarias.argmax(dim=1))
print(ventas_diarias.max())
print(ventas_diarias.min(dim=1))
```

Output:

```
tensor([4, 4])
tensor(3900)
torch.return_types.min(
values=tensor([1600,  400]),
indices=tensor([6, 6]))
```

Friday is the best day for both branches; 3,900 is the highest daily sales figure in the entire tensor; and the slowest day for both is 6, Sunday. The same pattern, once again. Norte mimics Centro even on its bad days.

A distinction that will save you time: `sum` and `mean` accept a tuple of axes, like `dim=(1, 2)`; **`max` and `min` do not**, only one at a time. You will see this in the error table.

**For when you need it: `keepdim`.** If you add `keepdim=True` to any of these functions, the axis you combine doesn't vanish; instead, it stays with size 1: `papeleria.sum(dim=2, keepdim=True)` yields shape `(2, 7, 1)` instead of `(2, 7)`. You don't need it today; in lesson B you will see when it comes in handy.

### Everything in a single table

It was almost nine. Mateo copied everything he had uncovered that night onto a single sheet of paper, so doña Lupe could take it to Licenciado Ortega. Each question, one line:

| Doña Lupe's question | Code | Shape |
|---|---|---|
| How much did I sell in total? | `papeleria.sum()` | `()` |
| How much did each branch sell? | `papeleria.sum(dim=(1, 2))` | `(2,)` |
| How much did each branch sell each day? | `papeleria.sum(dim=2)` | `(2, 7)` |
| How much did each branch sell of each product? | `papeleria.sum(dim=1)` | `(2, 3)` |
| How much did I sell of each product each day, combining branches? | `papeleria.sum(dim=0)` | `(7, 3)` |
| How much do I sell on average per day at each branch? | `papeleria.sum(dim=2).float().mean(dim=1)` | `(2,)` |
| Which day do I sell the most at each branch? | `papeleria.sum(dim=2).argmax(dim=1)` | `(2,)` |

---

## PyTorch and TensorFlow: the same sum, by another name

Before closing the laptop, Mateo wanted to make sure the 4,600 wasn't some quirk of PyTorch. He ran the same calculation in TensorFlow, the other major tool. There, the functions are named `reduce_sum` and `reduce_mean` (there you see the word “reduce”), and the axis is called `axis` instead of `dim`. This block is for reading, not running: the course does not install TensorFlow.

```python
import tensorflow as tf
from papeleria import centro, norte

papeleria_tf = tf.constant([centro, norte])
print(tf.reduce_sum(papeleria_tf, axis=(1, 2)))
print(tf.reduce_mean(tf.reduce_sum(papeleria_tf, axis=2), axis=1))
```

Output (TensorFlow 2.21):

```
tf.Tensor([18500  4600], shape=(2,), dtype=int32)
tf.Tensor([2642  657], shape=(2,), dtype=int32)
```

The sum gives the same result: 4,600 in both tools. But look at the average: **2642 and 657, with no decimals and no warning whatsoever**. TensorFlow preserves the integer type and performs integer division: truncating the fractional part; PyTorch preferred to crash. An error at least warns you; a skewed number that goes unnoticed is worse. If you use TensorFlow, convert to floating-point before averaging:

```python
promedios_tf = tf.reduce_mean(
    tf.cast(tf.reduce_sum(papeleria_tf, axis=2), tf.float32),
    axis=1,
)
print(promedios_tf)
```

Output (TensorFlow 2.21):

```
tf.Tensor([2642.8572  657.1429], shape=(2,), dtype=float32)
```

Mateo wrote it down with a double underline. Thirty days before a deadline, a number that quietly alters itself is the last thing you need.

---

👉 **If you come from Business Administration:** what Mateo did that night, you do with a pivot table. When you put “branch” in the rows and “sum of sales” in the values, Excel combines days and products and leaves you with one number per branch: that's `papeleria.sum(dim=(1, 2))`. If you put “day” in the rows, it combines branches and products: `papeleria.sum(dim=(0, 2))`. Today's rule applies just the same there: whatever you don't put in the pivot table disappears from the report. And the lesson from step 5 too: before presenting an average, make sure your tool hasn't clipped the cents. And one more thing a pivot table won't tell you: a low total tells you **how much** is missing, never **who** is to blame.

---

## Session 4 exercises

The solutions are at the end of this installment. Try them first; the `assert` tells you if you got it right. Always in the same order: **shape first, value second.**

**Exercise 14 — Predict the shape.** Write on paper what shape remains in each case and what it means for doña Lupe. Then turn each prediction into an `assert` and run it. The first one is already solved:

```python
from papeleria import papeleria

assert papeleria.sum(dim=0).shape == (7, 3)          # ejemplo resuelto
assert papeleria.sum(dim=1).shape == ____
assert papeleria.sum(dim=2).shape == ____
assert papeleria.sum(dim=(0, 1)).shape == ____
assert papeleria.sum(dim=(0, 2)).shape == ____
assert papeleria.sum(dim=(0, 1, 2)).shape == ____
print("Ejercicio 14 superado")
```

Bonus question: what other way do you have to write `papeleria.sum(dim=(0, 1, 2))`?

**Exercise 15 — The average by product (halfway there).** Doña Lupe wants to know how much she sells **on average per day** for each product at each branch. Fill in the two blanks:

The `f` prefix before quotation marks creates an **f-string**: it lets you insert a value inside curly braces within text. Here it will make the `assert` message show the actual shape if something fails.

```python
from papeleria import papeleria, norte

promedio = papeleria.____().mean(dim=____)
print(promedio)

assert promedio.shape == (2, 3), f"Esperaba (sucursal, producto) = (2, 3) y salió {tuple(promedio.shape)}"

suma_plumas_norte = 0
for renglon in norte:
    suma_plumas_norte = suma_plumas_norte + renglon[1]

assert round(promedio[1, 1].item(), 2) == round(suma_plumas_norte / 7, 2), "No coincide con el for"
print("Ejercicio 15 superado")
```

Notice the message in the first `assert`: if you pick the wrong axis, it tells you what shape came out. That shape reveals which axis you combined by mistake.

**Exercise 16 — The best and worst of the week.** Write the code yourself. For each question: an `assert` for the shape, then the value.

a) What is the slowest day for each branch, and how much was sold that day? Verify Norte's answer with a `for` loop.
b) What is the flagship product (the one that sold the most during the week) for each branch?
c) Combining both branches, which day sold the most, and how much? It should yield 4,900.

Hint: for each question, first decide which axes you combine for the sum, and then along which axis you search.

**Exercise 17 — Your grades.** Revisit the grades from exercise 4 of session 1. If you didn't save yours, use these (student × subject × midterm):

```python
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]
```

Convert them into a tensor and calculate, each with its own shape `assert`:

a) The overall average for each student.
b) The average for each student in each subject.
c) Which student has the highest average.
d) Which subject is the hardest for the group (the one with the lowest average, combining students and midterms).

Verify part a) for the first student with a `for` loop. If you used your own data with a different axis order, your `dim` arguments will differ from the solution's, and that is fine: what counts is that the shape you predict matches the one that comes out.

---

## Common errors (and how to spot them)

| Error | What it looks like | How to spot it | What to do |
|---|---|---|---|
| Thinking `dim` says what you keep | You want a daily total and write `dim=1`, thinking “the days axis” | The shape comes out `(2, 3)` instead of `(2, 7)` | `dim` says what you **combine**. For a daily total, combine the other axes |
| Averaging integers | `RuntimeError: mean(): could not infer output dtype. ... Got: Long` | The `dtype` says `torch.int64` | `.float()` before `mean`, or data with decimal points |
| Summing in two steps with old numbering | `IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)` | You asked for an axis that no longer exists | After reducing, axes are renumbered. Better to use a single call: `dim=(1, 2)` |
| Comparing a tensor against a list | `assert por_dia == [2500, 3100, ...]` raises `AssertionError` even though numbers match | The `assert` blows up with numbers that look identical to the naked eye | Convert first: `por_dia.tolist() == [...]` |
| Comparing floats without rounding | `2642.857177734375 == 18500 / 7` gives `False` | The `assert` blows up over the final digit | Round both sides: `round(x, 2)` |
| Passing a tuple to `max` | `TypeError: max() received an invalid combination of arguments - got (dim=tuple, ), ...` | You wrote `max(dim=(1, 2))` | `max` and `min` accept only a single axis. Sum or reduce one first, then search along the other |

---

## Lesson A self-assessment

You have finished the first lesson. Before moving on to lesson B, check that you can do each thing **without looking at the sessions**. If any of them gives you trouble, go back to the indicated session and redo its exercise.

**Session 1 — A number, a list, a table, a cube**
- [ ] I can state how many dimensions a dataset has and what each one represents.
- [ ] I can write those data as nested Python lists and pinpoint a number using its indices.
- [ ] I can calculate the shape by hand and find how many numbers there are in total by multiplying it.
- [ ] I can recognize a jagged list and explain why it cannot be a tensor.
- [ ] I can write an `assert` with a custom message to verify my own work.

**Session 2 — Your first tensor**
- [ ] I can convert a Python list into a tensor using `torch.tensor`.
- [ ] I can read `shape`, `ndim`, `dtype`, and `numel()` from a tensor and explain what each one means.
- [ ] I know which `dtype` PyTorch chooses for integers and for floating-point numbers.

**Session 3 — Axes have names**
- [ ] I can extract a number with `papeleria[1, 4, 2]` and convert it into a Python number with `.item()`.
- [ ] I can extract a full slice along any axis, like `papeleria[:, 4]` or `papeleria[:, :, 2]`, and state its shape before running it.

**Session 4 — Summing by tab**
- [ ] I can predict the shape after reducing one or more axes.
- [ ] I can calculate totals and averages by axis using `sum` and `mean`, and fix the `mean` error with integers.
- [ ] I can find the maximum, minimum, and their position using `max`, `min`, `argmax`, and `argmin`.
- [ ] I can verify a tensor calculation against the same math done with a `for` loop.

## Lesson A review

Ten questions from the four sessions. Answer them without looking at anything; the answers are in the Solutions.

1. What is the rule for determining how many dimensions a dataset has?
2. If doña Lupe saves 4 weeks of sales, by branch, day, and product, what shape does the tensor have and how many numbers does it hold?
3. What does `assert` do when the condition is true, and what does it do when it is false?
4. What happens if you pass the list `[[1, 2, 3], [4, 5]]` to `torch.tensor`?
5. What do `papeleria.ndim`, `papeleria.numel()`, and `papeleria.dtype` tell you?
6. What `dtype` does `torch.tensor([1, 2, 3])` have? And `torch.tensor([1.5, 2.0])`?
7. What number does `papeleria[1, 4, 2]` return, and what is the purpose of adding `.item()`?
8. What shape does `papeleria[:, :, 2]` have and what does it represent?
9. What shape does `papeleria.sum(dim=0)` leave, and what does it mean for doña Lupe?
10. What does `max(dim=1)` return that `sum(dim=1)` does not?

---

## Review for next session

Session 5 begins with these three questions. Answer them without looking at the lesson:

1. If you sum `papeleria` with `dim=1`, what shape remains and what does it mean?
2. Why does `mean` fail with integers and how do you fix it?
3. How do you verify that your tensor sum is correct?

🔁 **Next week** begins lesson B, **Tensors at Work**. Mateo is going to need to look further back than a single week: entire months. And to see them, he will have to reshape the tensor without changing its data: the same numbers, arranged in different boxes. You will also read, word for word, the error that appears when the new shape doesn't fit.

Before pulling down the shutter, doña Lupe showed Mateo her phone. She had asked Fer for the week's register closing since four. Then came the calls: the ones from the afternoon, and the ones from tonight. Not one returned. Mateo urged her to wait, that the numbers showed how much and not who. She put her phone away without answering him.

Fer didn't answer the phone all afternoon.

---

## Solutions (don't look before trying)

**Warm-up review (from session 3).**

1. `papeleria[:, 4]` has shape `(2, 3)`: Friday's three products at each branch.
2. `papeleria[0]` has shape `(7, 3)`: the Centro tab, without the branches axis. `papeleria[0:1]` has shape `(1, 7, 3)`: a slice that preserves the branches axis, containing only one branch inside.
3. `papeleria[:, :, 2]`: all branches, all days, product 2 (backpacks). Shape `(2, 7)`.

**Exercise 14.**

```python
assert papeleria.sum(dim=0).shape == (7, 3)          # cada día y producto, sucursales juntas
assert papeleria.sum(dim=1).shape == (2, 3)          # cada sucursal y producto, en la semana
assert papeleria.sum(dim=2).shape == (2, 7)          # cada sucursal y día, productos juntos
assert papeleria.sum(dim=(0, 1)).shape == (3,)       # cada producto, en todo
assert papeleria.sum(dim=(0, 2)).shape == (7,)       # cada día, en todo
assert papeleria.sum(dim=(0, 1, 2)).shape == ()      # un solo número: el total
```

If you also print the values:

```
tensor([11350,  7200,  4550])
tensor([3100, 3900, 2850, 3350, 4900, 3000, 2000])
tensor(23100)
Ejercicio 14 superado
```

Bonus question: `papeleria.sum(dim=(0, 1, 2))` is the same as `papeleria.sum()`. Combining all axes combines everything: 23,100.

**Exercise 15.** The blanks are `float` and `1`: you convert to decimals and combine the days.

```python
promedio = papeleria.float().mean(dim=1)
```

Output:

```
tensor([[1285.7142,  828.5714,  528.5714],
        [ 335.7143,  200.0000,  121.4286]])
Ejercicio 15 superado
```

Norte sells on average exactly $200 in pens per day (1,400 divided by 7). If you used `dim=0`, the `assert` message told you:

```
AssertionError: Esperaba (sucursal, producto) = (2, 3) y salió (7, 3)
```

The `(7, 3)` gives you away: branches disappeared, not days.

**Exercise 16.**

```python
from papeleria import papeleria, norte

dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
productos = ["cuadernos", "plumas", "mochilas"]

# a) Día más flojo: junto productos (2, 7) y busco el mínimo a lo largo de los días
valores, indices = papeleria.sum(dim=2).min(dim=1)
assert valores.shape == (2,), "Esperaba un valor por sucursal"
assert valores.tolist() == [1600, 400], "Los mínimos deberían ser 1,600 y 400"
assert indices.tolist() == [6, 6], "El domingo debería ser el día más flojo"
print(valores, indices)

peor_dia = 0
for d in range(7):
    if sum(norte[d]) < sum(norte[peor_dia]):
        peor_dia = d
assert indices[1].item() == peor_dia, "No coincide con el for"

# b) Producto estrella: junto días (2, 3) y busco la posición del máximo en productos
estrella = papeleria.sum(dim=1).argmax(dim=1)
assert estrella.shape == (2,), "Esperaba un producto por sucursal"
assert estrella.tolist() == [0, 0], "Los cuadernos deberían ser el producto estrella"
print(productos[estrella[0].item()], productos[estrella[1].item()])

# c) Mejor día con las dos sucursales juntas: junto sucursales y productos (7,)
por_dia_juntas = papeleria.sum(dim=(0, 2))
assert por_dia_juntas.shape == (7,), "Esperaba un total por día"
print(dias[por_dia_juntas.argmax().item()], por_dia_juntas.max().item())
assert por_dia_juntas.max().item() == 4900, "El mejor día debería sumar 4,900"
assert por_dia_juntas.argmax().item() == 4, "El mejor día debería ser viernes"
print("Ejercicio 16 superado")
```

Output:

```
tensor([1600,  400]) tensor([6, 6])
cuadernos cuadernos
viernes 4900
Ejercicio 16 superado
```

Sunday is the slowest day for both branches (1,600 and 400), notebooks are the flagship product for both, and on Friday, combining branches, 4,900 was sold. In c), `argmax()` without `dim` works because `por_dia_juntas` already has only one axis.

**Exercise 17.**

```python
import torch

parciales = torch.tensor([
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
])                                   # alumno × materia × parcial
decimales = parciales.float()

por_alumno = decimales.mean(dim=(1, 2))           # a) quito materias y parciales
assert por_alumno.shape == (3,)
por_alumno_y_materia = decimales.mean(dim=2)      # b) quito parciales
assert por_alumno_y_materia.shape == (3, 4)
por_materia = decimales.mean(dim=(0, 2))          # d) quito alumnos y parciales
assert por_materia.shape == (4,)

mejor_alumno = por_alumno.argmax()                # c)
assert mejor_alumno.shape == ()
assert mejor_alumno.item() == 1, "El segundo alumno debería tener el mejor promedio"
assert torch.allclose(por_materia, torch.tensor([8.5556, 8.3333, 8.1111, 8.4444]), atol=1e-4)
assert por_materia.argmin().item() == 2, "La tercera materia debería ser la más difícil"  # d)

print(por_alumno)
print(por_alumno_y_materia)
print(mejor_alumno)
print(por_materia, por_materia.argmin())

suma = 0
cuantas = 0
for materia in parciales[0].tolist():
    for calificacion in materia:
        suma = suma + calificacion
        cuantas = cuantas + 1
assert round(por_alumno[0].item(), 2) == round(suma / cuantas, 2), "No coincide con el for"
print("Ejercicio 17 superado")
```

Output:

```
tensor([8.2500, 9.1667, 7.6667])
tensor([[8.6667, 8.0000, 9.3333, 7.0000],
        [9.6667, 9.0000, 8.3333, 9.6667],
        [7.3333, 8.0000, 6.6667, 8.6667]])
tensor(1)
tensor([8.5556, 8.3333, 8.1111, 8.4444]) tensor(2)
Ejercicio 17 superado
```

The second student (index 1) has the highest average, 9.17, and the hardest subject is the third one (index 2), with 8.11. If you forgot `.float()`, you saw the same `RuntimeError` from step 5: the grades are also integers.

**Lesson A review.**

1. The number of dimensions is the number of pieces of information you need to pinpoint a single number.
2. `(4, 2, 7, 3)`: week, branch, day, and product. It holds 4 × 2 × 7 × 3 = 168 numbers.
3. If the condition is true, it does nothing and the program continues. If it is false, it halts the program with an `AssertionError` and displays your message.
4. PyTorch refuses to create it, because the list is jagged (one row of 3 and another of 2): `ValueError: expected sequence of length 3 at dim 1 (got 2)`.
5. `ndim` gives 3 (three axes), `numel()` gives 42 (the total numbers, 2 × 7 × 3), and `dtype` gives `torch.int64` (integers).
6. `torch.int64` for integers and `torch.float32` for floating-point decimals.
7. It yields 200: Norte, Friday, backpacks. `.item()` converts it from a single-number tensor into a regular Python number, to print it or compare it with an `assert`.
8. `(2, 7)`: backpack sales for both branches across all seven days.
9. `(7, 3)`: it combines both branches; each day and each product remains, as if doña Lupe had a single store.
10. In addition to the highest value, it returns **at what position** it is located (the indices). `sum` only returns the totals.

---

## What comes next

With this, you conclude lesson A: you know what a tensor is, how to create it, traverse it along its axes, and summarize it. In session 5, lesson B begins. Before getting there, solve the entire self-assessment; if you can do everything without looking, you're on solid ground.

---

*I write from what we see every day at Hábil, where we build systems for banking, insurance, and enterprise in Mexico. If at your work or your family's business you run into a problem like this, let's talk: [habil.mx](https://www.habil.mx/en/)*
