# Tensors from scratch (5): changing shape without changing the data

*Twelve months in a row, four quarters or two semesters: the same numbers arranged a different way, and how to read the error that appears when you ask for an impossible shape.*

*Tensors from scratch · lesson B, session 5 of 8*

*Setting: the characters and the story are fictional. Technical details, historical facts, and real figures include their sources.*

**Monday, November 30, 9:15 p.m. Doña Lupe's kitchen.**

The pot of beans was still warm on the stove. On the oilcloth tablecloth was a thick, green folder, tied with a rubber band.

“Fer still isn't answering,” said doña Lupe. “A whole week, Mateo. A week.”

“Maybe she's sick, Auntie.”

“Maybe.” She didn't sound convinced.

She pushed the folder toward him. Inside was the accountant's ledger: what the Centro branch sold each month of last year. Twelve handwritten rows, in blue ink.

“The accountant wants them by quarter. And Licenciado Ortega, by semester, for ‘his review.’ Everyone wants them in their own little box.”

“They're the same twelve numbers.”

“Well, tell that to them.”

Mateo typed them into the laptop. Twelve numbers in a row. He didn't have to change any of them. Just arrange them differently. And something was turning over in his mind: if he could see a healthy year arranged by quarter, he'd know what to look for at Norte. Which month should be strong. Which one shouldn't.

“Which month do you sell the most?”

“August, mijo. Back to school. The moms line up all the way into the street. Almost all of them pay with a card.”

Mateo wrote *August* on a post-it and stuck it to the edge of the screen. He didn't know why yet.

On doña Lupe's cell phone, a notification lit up. An email from the bank. She opened it, read it with her glasses on the tip of her nose, and handed it to him without saying a word.

*“We inform you that the first unusual charges recorded on the terminal at your Norte branch correspond to the second half of August.”*

The post-it was still there. Yellow. *August*.

---

This session begins the second lesson, **Tensors at Work**. Up to this point, Mateo learned how to observe: creating a tensor, extracting slices from it, and summarizing it. That’s how he discovered that Centro sells about four times as much as Norte: $18,500 versus $4,600 during the week. Now tensors are put to work, and each tool brings him one step closer to what happened at Norte: you change a tensor's shape without touching its data, you compute with entire tensors without a single `for`, you train a model that detects fraud, and you see where they appear in a bank, an insurance company, and a store. Twenty-three days left until the bank's deadline.

| Session | What you see | Practice |
|---|---|---|
| **5 — Changing shape without changing the data** | `reshape` and `flatten`, and how to read a shape error | Exercises 18 to 21 |
| **6 — Operating on entire tensors** | Calculating with the entire tensor at once, without a `for` | Exercises 22 to 25 |
| **7 — A fraud detector that learns from data** | Training a small model with tensors | Exercises 26 to 29 |
| **8 — Tensors in the bank, the insurance company and the store** | Reading a real case and stating what shape its tensor has | Exercises 30 to 33, self-assessment and review of lesson B |

As in lesson A, each session is about 2 hours: one to read and one to practice.

In the previous session you summed by tab with `sum(dim=)` and compared it against a `for`. Today you don't change any numbers: you change how they are arranged. Just like Mateo that night, with the accountant's ledger open next to the laptop.

## Warm-up review

Answer without looking at session 4. The answers are in the solutions, at the end.

1. If you sum `papeleria` with `dim=1`, what shape is left and what does it mean?
2. Why does `mean` fail on integers, and how do you fix it?
3. How do you verify that your sum with tensors is correct?

> **The code for this session** is in the course's public repository, in the `sesion-05` folder: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-05). Write it yourself first, because that is how you learn; the repository is for comparing when something doesn't come out the same. Each program comes with the output it should give you.

## Objectives of this session

Next to each objective is the exercise that demonstrates it.

| # | By the end you can… | Demonstrated by |
|---|---|---|
| 1 | **State**, without running anything, whether a new shape (with or without a `-1`) is valid for a tensor, and **calculate** the missing dimension when there is a `-1` | Exercise 18 |
| 2 | **Rearrange** a tensor with `reshape`, using `-1` so PyTorch calculates a size, and **verify it** against a `for` | Exercises 19 and 20 |
| 3 | **Read** PyTorch's shape error word by word and **fix** the code causing it | Exercise 20 |
| 4 | **Distinguish** between rearranging a table and transposing it, and **detect** with an `assert` when someone mixed them up | Exercise 21 |
| 5 | **Flatten** a tensor with `flatten` and **restore** its shape without losing values | Exercise 19, part c |

**What you need to know beforehand:** sessions 1 to 4: creating a tensor, reading its `shape`, indexing, slicing, and summing with `sum(dim=)`; and in Python, using `for`, `range`, lists, `append`, `tolist()`, and `assert` (below, in step 3, is a refresher). **What you need to have:** the environment from session 2 with PyTorch, and the `papeleria.py` file from session 4 (the `papeleria` tensor with shape `(2, 7, 3)`). Today's files go in that same folder.

**Today's scope:** the new decision you practice is whether `reshape` preserves the order you need. `flatten()` is the case of bringing everything into a single row. `.T` appears only to contrast with `reshape`: write that line and let the `assert` statements check it. Step 10 and the TensorFlow box are reading material to see where this is used in AI: run `foto.py` if you want, but no exercise needs it, and there is no need to install TensorFlow.

---

## Why it matters

“And how do you arrange them without mixing them up?” asked doña Lupe.

Mateo pointed to the egg carton next to the stove.

A dozen fits in a carton of 2 rows by 6 or in one of 3 by 4: the same 12 eggs, a different carton. And if they ask you to put it in one of 5 by 3, you know it's impossible: it has 15 slots and you have 12 eggs.

“Even I know that,” she said. “And if there's an empty slot, someone ate an egg.”

With tensors it happens all the time: data arrives in one shape and the next step needs it in another. Doña Lupe has sales by month and her accountant wants them by quarter. A photo arrives as a grid and the model wants it as a row of numbers. Not a single number is invented or lost: only the carton is changed. That is **reshape**.

And a practical reason: some of the errors you will see most often with tensors are shape errors: asking for a carton that doesn't fit your eggs. Today you learn to read them without panic. And you're going to see the other error, the dangerous one: the one that doesn't crash. The carton looks the same, it fits, nobody complains… and the eggs are no longer what they used to be.

## The case: doña Lupe's sales for the year

These are the twelve rows from the accountant's ledger: what Centro sold each month of last year, in pesos. A healthy year, without surprises. August is its strong month, because of back-to-school shopping, and December too, because of backpacks given as gifts. Mateo is going to use it as a blueprint: this is what a normal year looks like in one of doña Lupe's stationery stores.

### Step 1 — The twelve months in a row

Create a file `cambiar_forma.py`:

```python
import torch

#        ene    feb    mar    abr    may    jun    jul    ago     sep    oct    nov    dic
meses = [60000, 55000, 60000, 50000, 65000, 50000, 70000, 100000, 70000, 60000, 55000, 90000]
ventas = torch.tensor(meses)
print(ventas.shape)
print(ventas.sum())
```

Run it with `python3 cambiar_forma.py`:

```
torch.Size([12])
tensor(785000)
```

Nothing new: 12 numbers that sum to $785,000. That total is your control: you are going to change the shape many times, and it must never move.

### Step 2 — Four quarters of three months

The accountant asks for sales by quarter: January to March, April to June, and so on. You want a table of **4 rows (quarters) by 3 columns (months)**.

Add to the end of the file:

```python
trimestres = ventas.reshape(4, 3)
print(trimestres)
print(trimestres.shape)
print(trimestres.sum())
```

The new lines of output (the previous ones still appear above):

```
tensor([[ 60000,  55000,  60000],
        [ 50000,  65000,  50000],
        [ 70000, 100000,  70000],
        [ 60000,  55000,  90000]])
torch.Size([4, 3])
tensor(785000)
```

Four rows, three columns, and the same total. `reshape(4, 3)` means “arrange them into 4 by 3.”

**How does PyTorch decide which number goes in each spot?** It reads the original row from left to right, like a book, and fills the new table row by row, just like you arrange eggs in a carton: first the first row, then the second.

Since the months were in order, each row ended up exactly as a quarter. It is not luck: the reading order matches what you want. Keep this idea in mind; in step 9 you are going to see it fail.

Now you can use everything from lesson A on the new table. Add:

```python
print(trimestres.sum(dim=1))   # total de cada trimestre
print(trimestres[3, 2])        # cuarto trimestre, tercer mes: diciembre
print(ventas.shape)            # ¿se descompuso la original?
```

New output:

```
tensor([175000, 165000, 240000, 205000])
tensor(90000)
torch.Size([12])
```

- The third quarter (July to September) is the strongest: $240,000, due to back-to-school.
- `trimestres[3, 2]` is December, $90,000: row 3 (fourth quarter), column 2 (third month).
- `ventas` still has shape `(12,)`. **`reshape` does not change the shape of the original tensor: it gives you one with the new shape.** That is why you save it in a variable, such as `trimestres`. Note: it is not always a separate copy; often both share the same numbers internally. Today that doesn't matter, because you don't modify any of them.

> Doña Lupe peered over his shoulder.
>
> “And how do you know the machine didn't make them up?”
>
> “I don't know, Auntie. That's why I'm going to verify it.”
>
> “I like that. You shouldn't trust the computer right off the bat either.”

### Step 3 — Checking it against a `for`

Don't just take PyTorch's word for it. Calculate the totals with a `for` loop and compare.

**Python reminder that you will use:** `range(4)` produces 0, 1, 2, and 3; `lista.append(x)` adds `x` to the end of a list; `tensor.tolist()` converts a tensor into a normal list to compare it with another list.

And a tool you will use throughout the entire series. An `assert` is an automatic check: if the condition is true, it prints nothing and the program continues; if it is false, the program stops and shows the message in quotes. Here it works as your error control: you don't have to believe the result, you can verify it. If you want to see what a failure looks like, try this in a Python console:

```python
# Prueba opcional para reconocer un fallo:
assert 2 == 3, "2 no es igual a 3"
# Sale: AssertionError: 2 no es igual a 3
```

Now, add:

```python
a_mano = []
for t in range(4):
    total = 0
    for m in range(3):
        total = total + meses[t * 3 + m]
    a_mano.append(total)

assert trimestres.shape == (4, 3), "Deberían ser 4 trimestres × 3 meses"
assert trimestres.sum(dim=1).tolist() == a_mano, "Los totales por trimestre no cuadran con el for"
print("Trimestres comprobados")
```

New output:

```
Trimestres comprobados
```

Notice `meses[t * 3 + m]`: that's the calculation `reshape` does under the hood. December (quarter 3, month 2) is at position 3 × 3 + 2 = 11 of the original row. `reshape` saves you that calculation. And as always: first the shape `assert`, then the values `assert`.

### Step 4 — Two semesters, and up to three dimensions

Licenciado Ortega asks doña Lupe for her sales by semester: two halves of six months. Add:

```python
semestres = ventas.reshape(2, 6)
print(semestres)
print(semestres.sum(dim=1))

anio = ventas.reshape(2, 2, 3)   # semestre × trimestre × mes
print(anio)
print(anio[1, 0, 2])
```

New output:

```
tensor([[ 60000,  55000,  60000,  50000,  65000,  50000],
        [ 70000, 100000,  70000,  60000,  55000,  90000]])
tensor([340000, 445000])
tensor([[[ 60000,  55000,  60000],
         [ 50000,  65000,  50000]],

        [[ 70000, 100000,  70000],
         [ 60000,  55000,  90000]]])
tensor(70000)
```

First semester, $340,000; second, $445,000. They sum to the usual $785,000.

The shape `(2, 2, 3)` turns the year into a **cube**: the sheet with tabs from session 1, one tab per semester, and on each one a table of quarters by months. `anio[1, 0, 2]` is the second semester, its first quarter, its third month: September, $70,000. Twelve numbers, four shapes, and nothing was lost.

> Mateo stared at the second semester. There was the biggest number of the whole year, sitting alone in the middle of its quarter: August, $100,000. In a healthy year, August is the month when the most people swipe their card.
>
> He looked at the post-it. He looked at the email from the bank: *second half of August*.
>
> If someone wanted to hide fake charges, which month would they choose? The month nobody pays attention, because everyone is paying.

### Step 5 — The rule: multiply the shape

> **A `reshape` with positive sizes is valid when the new shape, multiplied together, gives the same total number of elements as the original.** With `-1` there is a separate rule, in step 6.

Check it before running:

| Shape you ask for | Multiplied | Does it work for 12 months? |
|---|---|---|
| `(4, 3)` | 4 × 3 = 12 | Yes |
| `(2, 2, 3)` | 2 × 2 × 3 = 12 | Yes |
| `(5, 3)` | 5 × 3 = 15 | No: you are missing 3 eggs |
| `(5, 2)` | 5 × 2 = 10 | No: you have 2 eggs left over |

`reshape` never pads with zeros or discards what is left over. If it doesn't match, it refuses.

### Step 6 — `-1`, which calculates on its own

Sometimes you know how much you want in one dimension and you don't feel like dividing for the other. You put **`-1`** in the place you don't want to calculate. Add:

```python
print(ventas.reshape(-1, 3).shape)
print(ventas.reshape(6, -1).shape)
print(ventas.reshape(2, -1, 3).shape)
print(ventas.reshape(-1).shape)
```

New output:

```
torch.Size([4, 3])
torch.Size([6, 2])
torch.Size([2, 2, 3])
torch.Size([12])
```

**`-1` means “you calculate how much goes here so it fits.”** In `reshape(-1, 3)`, PyTorch thinks: “12 numbers in columns of 3: that's 4 rows.” In `reshape(2, -1, 3)`: “2 × something × 3 = 12; something is 2.”

And `reshape(-1)`, all by itself, means “put everything into a single dimension,” whatever shape it has: a list of numbers, like the `(12,)` from the beginning.

**The rule with `-1`:** multiply the sizes you did write down; that product must divide the total count of numbers evenly (with no remainder). The `-1` equals the total divided by that product. In `reshape(2, -1, 3)`: 2 × 3 = 6, and 12 ÷ 6 = 2.

- **There can only be one.** With `ventas.reshape(-1, -1)` you get `RuntimeError: only one dimension can be inferred`.
- **It doesn't work miracles.** With `reshape(-1, 5)` there is no integer that multiplied by 5 gives 12, and it crashes. The `-1` calculates; it doesn't invent eggs.
- **It is the only negative number allowed.** With `ventas.reshape(-2, -6)` you get `RuntimeError: invalid shape dimension -2 at index 0 of shape [-2, -6]`, even though -2 × -6 is 12.

### Step 7 — The shape error, read word by word

Let's trigger the error on purpose. A classmate wanted “5 groups of 3 months.” Add to the end of the file:

```python
quintos = ventas.reshape(5, 3)
```

Run it (I shortened the path with `...`; on your computer it will be different):

```
Traceback (most recent call last):
  File ".../cambiar_forma.py", line 42, in <module>
    quintos = ventas.reshape(5, 3)
              ^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[5, 3]' is invalid for input of size 12
```

An error message is read **from bottom to top**: the last line says what happened; the ones above, where. The last line, word by word:

| Piece | What it means |
|---|---|
| `RuntimeError` | “Runtime error”: the code was written correctly, but failed while running. |
| `shape '[5, 3]'` | “The shape [5, 3]”. It is the shape **you asked for**. |
| `is invalid` | “is not valid”. |
| `for input of size 12` | “for an input of size 12”. It is what **you have**: 12 numbers. |

All together: **“The shape 5 × 3 does not work for something that has 12 numbers.”** 5 × 3 = 15, and 15 is not 12. The error doesn't scold you: it gives you what you asked for and what you have.

The lines above tell you where: in your file, line 42 (you may get a different number if you left more or fewer blank lines), and they copy the offending line with `^^^^` under the part that blew up.

If you wanted groups of 3 months, the fix is `reshape(4, 3)` or `reshape(-1, 3)`. If you really needed 5 groups, the problem is not the code: 12 cannot be divided into 5 equal parts. Delete that line before continuing.

**The recipe for any shape error:** look in the last line for the two sizes (what you asked for and what you have), multiply, and ask yourself what you really wanted.

> “How decent,” said doña Lupe when Mateo read her the error. “It tells you what you asked for and what you have. If only the bank were like that.”
>
> Mateo laughed. Then he stopped laughing. *What you asked for and what you have.* What was sold and what was charged. If someone were to put those two numbers together, side by side…
>
> He wrote it down on another post-it. That would have to wait.

### Step 8 — Flattening the stationery store with `flatten`

Let's go back to the `(2, 7, 3)` stationery store. Sometimes you need its 42 numbers in a single row, to store them or pass them to a model. That is **flattening**, like breaking down a cardboard box: the same cardboard, without volume.

In PyTorch it's called **`flatten`**. Before creating the file, make sure your `papeleria.py` can be imported. Open the Terminal in the folder where you saved it and run:

```bash
python3 -c "from papeleria import papeleria; print(papeleria.shape)"
```

It should print `torch.Size([2, 7, 3])`. If it says `ModuleNotFoundError`, place `papeleria.py` in this same folder and open the Terminal there. If it imports but says it cannot find `papeleria`, go back to the session 4 file and check that that is the saved variable.

Now, create `aplanar.py`:

```python
from papeleria import papeleria

print(papeleria.shape)
plana = papeleria.flatten()
print(plana.shape)
print(plana[:6])       # los primeros 6 números
print(plana[21:24])    # del 22 al 24
print(plana.sum(), papeleria.sum())

de_vuelta = plana.reshape(2, 7, 3)
assert de_vuelta.shape == (2, 7, 3), "Debería regresar a 2 sucursales × 7 días × 3 productos"
assert de_vuelta.tolist() == papeleria.tolist(), "Al regresar debería quedar idéntica"
print("Ida y vuelta sin perder nada")
```

Save the file and run it from the same folder with `python3 aplanar.py`. If an error appears, do not proceed: first compare the last line with the expected output.

Output:

```
torch.Size([2, 7, 3])
torch.Size([42])
tensor([1200,  800,  500, 1500,  900,  700])
tensor([300, 200, 100])
tensor(23100) tensor(23100)
Ida y vuelta sin perder nada
```

- **2 × 7 × 3 = 42** numbers, which sum to the same $23,100.
- **The reading order is the usual one**, from the outermost box to the innermost: the entire Centro tab, day by day and product by product, and then the entire Norte tab. The first six are Centro Monday and Tuesday. The first 21 belong to Centro (7 × 3), so positions 21 to 23 are Norte Monday: 300, 200, 100.
- **You can go back**, as long as you remember what shape it had.

`flatten()` does the same thing as `reshape(-1)`, but its name states what it does.

### Step 9 — The trap: `reshape` is not transposing the table

The most important part of the session, because here the error **does not crash**.

`papeleria.sum(dim=1)` is the table for the week: 2 branches × 3 products. Doña Lupe wants it the other way around for a report: **products in rows, branches in columns**, 3 × 2. `reshape(3, 2)`?

Create `girar.py`:

```python
from papeleria import papeleria

semana = papeleria.sum(dim=1)    # sucursal × producto
print(semana)
print(semana.shape)

mal = semana.reshape(3, 2)
print(mal)
```

Run it with `python3 girar.py`. Output:

```
tensor([[9000, 5800, 3700],
        [2350, 1400,  850]])
torch.Size([2, 3])
tensor([[9000, 5800],
        [3700, 2350],
        [1400,  850]])
```

PyTorch didn't complain: 3 × 2 = 2 × 3. But read it like a report:

- The notebooks row should have Centro 9,000 and Norte 2,350. It has **9,000 and 5,800**: notebooks and pens from Centro.
- The pens row should have 5,800 and 1,400. It has **3,700 and 2,350**: backpacks from Centro and notebooks from Norte.

A jumbled mess. `reshape` did the only thing it knows how to do: read in order (9000, 5800, 3700, 2350, 1400, 850) and fill rows 2 by 2. **`reshape` changes the carton; it doesn't know which egg belongs to whom.**

What you wanted was to **transpose** the table: rows to columns. In PyTorch it is `.T` (from *transpose*, the mathematical name). It is written `semana.T`, **without parentheses**: it is not called like a function; it describes the transposed table. Add:

```python
bien = semana.T
print(bien)
print(bien.shape)
```

New output:

```
tensor([[9000, 2350],
        [5800, 1400],
        [3700,  850]])
torch.Size([3, 2])
```

Now it's right: notebooks 9,000 and 2,350; pens 5,800 and 1,400; backpacks 3,700 and 850. The same shape `(3, 2)` as the jumbled mess: **same size, different content.** That is why the shape `assert` never goes alone. You don't need anything more from `.T` for now: it is what actually transposes.

**And why did it work with the quarters?** Because there you weren't transposing anything: you were cutting a row into chunks of 3, in the same order. The question before each `reshape`: **do I want to cut into chunks, in the same reading order?** If yes, `reshape`. If you want rows to become columns, `reshape` gives you a jumbled mess without warning you.

> *Same size, different content.* Mateo said it out loud.
>
> “What was that?” asked doña Lupe, who was already washing the pot.
>
> “Something that looks the same on the outside can be something else on the inside. Like the jumbled mess: it passes the shape check and nobody notices.”
>
> Doña Lupe turned off the faucet. She dried her hands on her apron, slowly.
>
> “The Norte terminal was replaced in August.”
>
> Mateo looked up.
>
> “Who did it?”
>
> “A technician. Fer sent me a photo on my cell phone: ‘The bank guy came, the new terminal is all set.’ Identical to the other one. And at the edge of the photo, a man in a blue t-shirt, with his back turned. I thought the bank had sent him.”
>
> “Did you call him?”
>
> “No.”
>
> “And Fer?”
>
> Doña Lupe didn't answer. On the table, the post-it still said *August*.

### Step 10 — Why AI is constantly changing shape (reading; running it is optional)

**A photo is flattened.** A black-and-white photo of 28 × 28 pixels is a table, and many simple models expect a row of numbers per example.

**A batch is rearranged.** Models see photos in batches, for example of 32: a cube `(32, 28, 28)` that the model wants as 32 rows, one per photo.

If you want to see it run, create `foto.py` and run it with `python3 foto.py`. `torch.zeros` creates a tensor of whatever shape you ask for filled with zeros, so you don't have to type thousands of numbers:

```python
import torch

foto = torch.zeros(28, 28)           # una foto de 28 × 28 puntos, toda negra
print(foto.shape, foto.numel())
print(foto.flatten().shape)

lote = torch.zeros(32, 28, 28)       # 32 fotos
print(lote.reshape(32, -1).shape)
print(lote.flatten().shape)
```

Output:

```
torch.Size([28, 28]) 784
torch.Size([784])
torch.Size([32, 784])
torch.Size([25088])
```

- The flattened photo: 784 numbers in a row (28 × 28).
- `lote.reshape(32, -1)`: 32 rows, one per photo; the `-1` calculates the 784.
- Watch out with `lote.flatten()`: it flattens **everything**, including the boundary between photos. That leaves 25,088 numbers (32 × 784) and the shape no longer tells you where each photo begins: you only know if you remember that there were 32 photos of 784. With batches, you want `reshape(32, -1)`, which preserves that separation in the shape.

---

> **Sidebar: the same thing in TensorFlow.** Just for comparison; do not run this or install TensorFlow for this session: all objectives and exercises are solved with PyTorch. The idea is identical: `tf.reshape(ventas, (4, 3))` instead of `ventas.reshape(4, 3)`, with the same `-1`. The error message includes: `Input to reshape is a tensor with 12 values, but the requested shape has 15`. TensorFlow multiplies for you (5 × 3 = 15); PyTorch gives you the shape and you multiply.

---

👉 **If you come from Business Administration:** this is what doña Lupe's accountant does at the end of each quarter, and this is what you do when turning a monthly report into a quarterly one: the same numbers grouped differently, and the year's total doesn't budge. That is your control check, like the `assert` from step 3. And the trap from step 9 lives in Excel: if you rearrange a table by hand instead of using “Paste Special → Transpose”, the numbers land in someone else's cells without any alert. After rearranging, check a figure you know by heart.

---

## Exercises for session 5

The solutions are at the end. Try them first. The `assert` statements check the code parts; for predictions and explanations, compare your answer with the expected output and with the solution at the end.

**Exercise 18 — Predict before running.** With the 12 months of `ventas`, write on paper whether each shape is valid and what final shape it would give:

```
(3, 4)    (6, 2)    (5, 2)    (2, -1)    (-1, 5)    (2, 3, 2)    (12, 1)
```

Then verify them one by one in the Python console with `ventas` already created. If you still have the line `quintos = ventas.reshape(5, 3)` from step 7 in `cambiar_forma.py`, delete it (otherwise the file stops at that error) and open the console with:

```bash
python3 -i cambiar_forma.py
```

When `>>>` appears, enter one shape at a time. If `>>>` doesn't appear, the file stopped at another error: resolve it before doing the exercise. To exit, type `exit()`. It looks like this (do not copy the `>>>`: Python puts that, you only type what follows):

```text
>>> ventas.reshape(3, 4).shape
```

This is what a solved one looks like: `(4, 3)` → 4 × 3 = 12 → valid → `torch.Size([4, 3])`. On the invalid ones, check that the error states the two sizes you expected.

**Exercise 19 — The long table.** Many spreadsheets store data in a long table: one row per branch and day, one column per product. **Parts a) and b):** fill in the blanks and answer the question below.

```python
from papeleria import papeleria

larga = papeleria.reshape(____, 3)
assert larga.shape == (14, 3), "Deberían ser 14 renglones (sucursal y día) × 3 productos"

a_mano = []
for s in range(2):
    for d in range(7):
        a_mano.append(papeleria[s, d].____())
assert larga.tolist() == a_mano, "La tabla larga no coincide con la armada a mano"
print("Ejercicio 19 superado")
```

Then, without running anything, state which row of `larga` is occupied by Norte Wednesday. Verify it with `print(larga[9])`, copy the three numbers that came out, and add:

```python
assert larga[9].tolist() == papeleria[1, 2].tolist(), "Ese renglón debería ser Norte miércoles"
```

**Part c) — Flattening.** Without looking at step 8, fill in the blanks to flatten the stationery store and return it to its shape:

```python
plana = papeleria.____()
de_vuelta = plana.____(2, 7, 3)

assert plana.shape == (42,), "Deberían quedar 42 números en una fila"
assert de_vuelta.tolist() == papeleria.tolist(), "La ida y vuelta debe conservar cada dato"
print("Ejercicio 19c superado")
```

**Exercise 20 — Read the error and fix it.** Daily sales of Centro in a 28-day February that started on a Monday: exactly 4 weeks. A classmate wanted to arrange them into weeks:

```python
import torch

dias_febrero = [
    2500, 3100, 2200, 2800, 3900, 2400, 1600,
    2600, 3000, 2300, 2700, 4000, 2500, 1500,
    2400, 3200, 2100, 2900, 3800, 2300, 1700,
    2700, 3300, 2400, 3000, 4200, 2600, 1800,
]
febrero = torch.tensor(dias_febrero)
por_semana = febrero.reshape(7, 5)
```

a) Run it, copy the last line of the error, and explain these four parts: `RuntimeError`, `shape '[7, 5]'`, `is invalid`, and `input of size 28`. Then explain why 7 × 5 cannot accommodate 28 days.

b) Fix it so that `por_semana` ends up as 4 weeks × 7 days (Monday through Sunday). Write the shape `assert` and compare the weekly totals against a `for` loop over the `dias_febrero` list, as in step 3 (day `d` of week `s` is at `dias_febrero[s * 7 + d]`).

c) Another classmate says: “I fixed it with `reshape(7, 4)`, which also gives 28.” It doesn't crash. Why is it wrong for what they wanted? To see it, add:

```python
mal = febrero.reshape(7, 4)
print(mal[0])
```

Compare that output with the seven days from Monday to Sunday of the first week.

**Exercise 21 — Catch the trap.** Cover step 9 before trying it and write a program that:

1. Calculates `semana = papeleria.sum(dim=1)`.
2. Builds the products × branches table (3 × 2) correctly.
3. Has a shape `assert` and another checking that the notebooks row contains `[9000, 2350]`.
4. Verifies with two `for` loops that `girada[p, s]` equals `semana[s, p]` for all of them.

Then, to test the trap, replace **only** the line of the transposed table (in the solution it is `girada = semana.T`) with:

```python
girada = semana.reshape(3, 2)
```

Do not change the `assert` statements or the two `for` loops. Run it. Which `assert` fails? Why didn't PyTorch warn you beforehand?

---

## Common errors (and how to spot them)

| Error | How it looks | How to spot it | What to do |
|---|---|---|---|
| Asking for a shape that doesn't fit | `RuntimeError: shape '[5, 3]' is invalid for input of size 12` | The program crashes on the `reshape` line | Multiply the shape you asked for and compare it with the size stated in the error |
| Thinking that `reshape` changed the original | After `ventas.reshape(4, 3)`, `ventas.shape` is still `(12,)` | `ventas[3, 2]` crashes | Save the result: `trimestres = ventas.reshape(4, 3)` |
| Using two `-1`s | `RuntimeError: only one dimension can be inferred` | Crashes even though the total would match | Leave only a single `-1` and write out the other sizes |
| Using `reshape` to transpose a table | Numbers in the wrong rows, without any error | Passes the shape `assert` and fails on the values `assert` | To transpose, `.T`; `reshape` only cuts in reading order |

---

## Review for next session

Session 6 begins with these three questions. Answer them without looking at the lesson:

1. When is a `reshape` valid?
2. What does `-1` mean in `reshape`?
3. Why is reshaping a table not the same as transposing it?

🔁 **Next week:** you operate on entire tensors, without `for`. You will multiply prices by quantities for all branches and all days in one shot, and you'll encounter a trap that, like today's, throws no error. Mateo will finally put two numbers side by side: what was sold and what was charged.

That night, before leaving, he wrote to Licenciado Ortega: *Did the bank send a technician to replace the terminal at the Norte branch in August?*

The answer arrived at 9:03 the following morning. A single line:

*“The bank has not dispatched any technical personnel to your Norte branch so far this year.”*

Mateo zoomed in on the photo Fer had sent his aunt in August. The shiny new terminal. And at the edge, cut off by the frame, a man with his back turned.

**Who was the man in the blue t-shirt?**

---

## Solutions (don't peek before trying)

**Warm-up review.**

1. `(2, 3)` is left: the days disappear because they were summed. Each number is the weekly total for one branch in one product (you used it today in step 9).
2. Because `mean` only accepts decimal (or complex) tensors: on integers it always crashes, even if the average would come out exact. The reason for that rule is the one from session 4: an average almost never yields an integer, and PyTorch doesn't silently chop off decimals. It is converted to floats beforehand: `papeleria.float().mean(dim=1)`.
3. Calculating the same thing with a `for` loop, which you already know works, and comparing with an `assert`: first the shape, then the values.

**Exercise 18.**

| Shape | Multiplied | Result |
|---|---|---|
| `(3, 4)` | 12 | `torch.Size([3, 4])` |
| `(6, 2)` | 12 | `torch.Size([6, 2])` |
| `(5, 2)` | 10 | `RuntimeError: shape '[5, 2]' is invalid for input of size 12` |
| `(2, -1)` | 2 × 6 | `torch.Size([2, 6])` |
| `(-1, 5)` | no number × 5 gives 12 | `RuntimeError: shape '[-1, 5]' is invalid for input of size 12` |
| `(2, 3, 2)` | 12 | `torch.Size([2, 3, 2])` |
| `(12, 1)` | 12 | `torch.Size([12, 1])` |

The last one is curious: 12 rows of a single number each, one column. Unlike `(12,)`, which has a single dimension, `(12, 1)` has two. It seems useless, but you will encounter it often in AI examples.

**Exercise 19.** The blanks are `-1` (or `14`) and `tolist`. This block is not for copying: it shows only the two lines with blanks already filled in, and the `...` are the lines of your file that do not change:

```text
larga = papeleria.reshape(-1, 3)
...
        a_mano.append(papeleria[s, d].tolist())
```

Norte Wednesday is row 9: the first 7 rows (0 to 6) are Centro, Norte Monday is 7, Tuesday is 8, and Wednesday is 9. `print(larga[9])` gives:

```
tensor([350, 200, 100])
```

The calculation is the same as in step 3: branch × 7 + day = 1 × 7 + 2 = 9.

Part c): the blanks are `flatten` and `reshape`:

```python
plana = papeleria.flatten()
de_vuelta = plana.reshape(2, 7, 3)
```

Output: `Ejercicio 19c superado`.

**Exercise 20.**

a) The last line is:

```
RuntimeError: shape '[7, 5]' is invalid for input of size 28
```

`RuntimeError` is a runtime error: the code was written correctly, but failed while running. `shape '[7, 5]'` is the shape the classmate asked for. `is invalid` says that it doesn't work. `input of size 28` is what it has: the days of February. 7 × 5 = 35, and 35 is not 28, so 28 days cannot fit in 7 rows of 5.

b) Change the last line to `por_semana = febrero.reshape(4, 7)` (or `reshape(-1, 7)`: “in chunks of 7, you calculate how many weeks”). The `for` loop is the one from step 3, with 4 and 7 instead of 4 and 3, over the `dias_febrero` list:

```python
por_semana = febrero.reshape(4, 7)

a_mano = []
for s in range(4):
    total = 0
    for d in range(7):
        total = total + dias_febrero[s * 7 + d]
    a_mano.append(total)

assert por_semana.shape == (4, 7), "Deberían ser 4 semanas × 7 días"
assert por_semana.sum(dim=1).tolist() == a_mano, "Los totales por semana no cuadran con el for"
print(por_semana.sum(dim=1))
print("Ejercicio 20 superado")
```

Output:

```
tensor([18500, 18600, 18400, 20000])
Ejercicio 20 superado
```

c) `reshape(7, 4)` cuts into chunks of 4 days. Its first row is `tensor([2500, 3100, 2200, 2800])`: Monday through Thursday of the first week. Another jumbled mess that doesn't crash (the next row starts on Friday). That the total matches only says that the `reshape` is possible, not that it is the one you wanted.

**Exercise 21.**

```python
from papeleria import papeleria

semana = papeleria.sum(dim=1)        # sucursal × producto
girada = semana.T                    # producto × sucursal
assert girada.shape == (3, 2), "Deberían ser 3 productos × 2 sucursales"
assert girada[0].tolist() == [9000, 2350], "El renglón de cuadernos debería traer Centro y Norte"

for p in range(3):
    for s in range(2):
        assert girada[p, s] == semana[s, p], "Girar es cambiar renglones por columnas"
print("Ejercicio 21 superado")
```

Output:

```
Ejercicio 21 superado
```

With `semana.reshape(3, 2)` the shape `assert` passes and the notebooks one crashes: the first row has `[9000, 5800]`. PyTorch didn't warn you because for it there is no error: you asked for 6 numbers in 3 × 2 and that's what it gave you. Whether the data makes sense can only be checked by you, with a values `assert`.

---

## What comes next

*Sources of technical data: official PyTorch documentation for [`reshape`](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html), [`flatten`](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html), and [`mean`](https://docs.pytorch.org/docs/stable/generated/torch.mean.html), [`Tensor.numel`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numel.html), [`torch.zeros`](https://docs.pytorch.org/docs/stable/generated/torch.zeros.html), and the `Tensor.T` attribute (on the [tensors page](https://docs.pytorch.org/docs/stable/tensors.html)), and TensorFlow for [`tf.reshape`](https://www.tensorflow.org/api_docs/python/tf/reshape).*

In session 6 you stop calculating number by number. If you want to get ahead, write with a `for` loop how you would multiply each of the stationery store's 42 numbers by 2. Next week you do it in a single line.

---

*I write from what we see every day at Hábil, where we build systems for banking, insurance, and enterprise in Mexico. If at your work or your family's business you run into a problem like this, let's talk: [habil.mx](https://www.habil.mx/en/)*
