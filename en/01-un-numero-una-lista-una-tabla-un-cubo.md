# Tensors from scratch (1): a number, a list, a table, a cube

*What a tensor is, told through the Python lists you already know how to write, and why Einstein had to ask a friend for help to understand it.*

*Tensors from scratch · lesson A, session 1 of 8*

*Setting: the characters and the story are fictional. Technical and historical data, and real figures, cite their sources.*

**Sunday, November 1, 11:52 p.m.**

Doña Lupe’s cell phone vibrated on the nightstand. Once. She, who never answers anything after ten, turned it over out of sheer habit.

> *We detected unusual charges on the terminal at your Norte branch. The terminal remains suspended while under review.*

She read it three times. Then she sat on the edge of the bed, her glasses on her forehead, and didn't go back to sleep.

**Monday, November 2, 9:10 a.m.**

“Mrs. Herrera, good morning. This is Licenciado Ortega, from the bank.”

Formal voice. The haste of someone with forty other pending calls.

“You have until December 23 to prove to us which charges from that terminal are real sales. If not, we will cancel it.”

“The 23rd? Licenciado, it's peak season! Without a terminal in Norte, in December…”

“That is why I am giving you the deadline. Have a good day.”

Click.

**Monday, November 2, 7:40 p.m. Papelería, Centro branch.**

The smell of fresh paper and erasers. The metal security shutter halfway down, scraping every time a bus rumbles past. Mateo Ríos, 20 years old, second semester of Computer Systems, ducks inside, backpack slung over his shoulder and laptop tucked under his arm.

“Aunt Lupe, my mom told me it was urgent.”

Doña Lupe doesn't greet him. She thrusts her cell phone in front of him, showing the message from the bank.

“You're the one who knows about computers.”

Mateo reads it. His stomach drops.

“Aunt Lupe, I know basic Python. This is… something else.”

“Then learn the other thing. You have until December 23.” She crosses her arms. “And don't tell me no; I paid your tuition.”

Mateo opens his laptop on the counter, between a jar of pencils and a stack of poster boards. He knows only one thing for certain: the bank didn't see a stationery shop. It saw numbers. If he wants to defend his aunt, he has to learn to see what the bank sees.

And you are going to learn along with him.

---

This series has two lessons of four sessions each, one session per week. In the first lesson, **From the List to the Cube**, you learn what a tensor is, how to create it, traverse it, and summarize it. In the second, **Tensors at Work**, you reshape it, operate on it as a whole, train a fraud detection model with it, and see where it appears in a bank, an insurance company, and a retail store.

| Session | What you see | Practice |
|---|---|---|
| **1 — A number, a list, a table, a cube** | What a dimension is, how to count them, and what the shape of a tensor is, all using Python lists | Exercises 1 to 5 |
| **2 — Your first tensor** | Installing PyTorch, converting your lists into tensors, `shape` and `dtype` | Exercises 6 to 9 |
| **3 — Axes have names** | Extracting a slice of the tensor along any of its axes | Exercises 10 to 13 |
| **4 — Summing by tab** | Totals and averages by axis, compared against a `for` loop | Exercises 14 to 17, self-assessment and lesson review |

Each session takes about 2 hours: roughly one to read and follow the example, and another to practice. Starting in Session 2, each one begins with **3 review questions** from the previous one, to answer without looking.

**What you need to know beforehand:** Basic Python: variables, `if`, `for`, and `while`, functions, lists, and tuples. If you come from Hábil's Python course, those are lessons 0 to 4. **What you need to have:** Python 3 and a text editor. Today you install nothing else.

> **The code for this session** is in the course's public repository, in the `sesion-01` folder: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-01). Write it yourself first, because that is how you learn; the repository is for comparing when something does not come out the same for you. Each program comes with the output it should give you.

## Objectives of this session

By the end, you will be able to do these four things. Next to each is the exercise that demonstrates it.

| # | By the end you will be able to… | Demonstrated by |
|---|---|---|
| 1 | Given a dataset (sales, grades, a photo), **state** how many dimensions it has and what each one represents | Exercise 1 |
| 2 | **Write** those data as nested Python lists and **pinpoint** a number using its indices | Exercises 2 and 4 |
| 3 | **Calculate** the shape of a tensor (how many elements are in each dimension), by hand and with a function | Exercises 1 and 3 |
| 4 | **Recognize** when a table cannot be a tensor because it is jagged, and verify it with code | Exercise 5 |

---

## Why it matters

That night, while doña Lupe counts the change in the cash drawer, Mateo searches on his phone for how a bank determines that something is “unusual.” The answer unnerves him a bit.

The bank does not read a purchase the way you read it. It turns it into numbers: the amount, the time, the distance to the last purchase, how many purchases occurred in the last hour. A purchase becomes a list of numbers. A thousand purchases, a table. Each customer's purchases over the last 30 days, a cube.

No one at the bank saw the faces of the customers at Norte. A machine saw cubes of numbers, and something didn't add up.

And it is not just the bank. When an insurance company receives a photo of a crashed car, it does not see a car either. It sees a grid of pixels, and each pixel carries three numbers: how much red, how much green, and how much blue it has. Once again, a cube of numbers.

Almost all artificial intelligence systems you will hear about in your career work this way: first they turn the world into ordered boxes of numbers, and then they perform calculations with those boxes. Those boxes are called **tensors**. If you understand how they are ordered, you understand half of what a model does. We will see the other half in the second lesson.

“And how does that work?” asks doña Lupe, without looking up from the coins.

“With lists, Aunt Lupe. Lists inside lists.”

It's true, and that is the good news: you already know how to write the data for a tensor. They are lists inside lists, made purely of numbers and completely even. In Session 2 you will turn them into real tensors.

## The case: doña Lupe's stationery shop

Doña Lupe has a stationery shop with two branches, Centro and Norte. She sells three things she cares about: notebooks, pens, and backpacks. The numbers in this series are examples, but the shape of the data is that of any business.

Mateo does not yet know what he is looking for. So he does the only sensible thing: he starts with the simplest level and asks for progressively more detail. He asks his aunt about last week's sales four times, each time looking through a stronger magnifying glass.

### Step 1 — A number

Doña Lupe pulls the shutter all the way down. The metallic crash echoes through the entire shop.

“Aunt Lupe, how much did you sell last week?”
“$18,500, at Centro. And don't ask me about Norte yet, or my blood pressure will go up.”

```python
total = 18500
```

A single number. To pinpoint it you don't need any information: there is only one. We say it has **zero dimensions**. In the language of tensors, it is called a **scalar**.

A scalar says a lot and explains very little. With $18,500, no one can defend anything to a bank.

### Step 2 — A list

“And day by day?”
“Monday 2,500; Tuesday 3,100; Wednesday 2,200; Thursday 2,800; Friday 3,900; Saturday 2,400; Sunday 1,600.” She recites it from memory, like a prayer.

```python
por_dia = [2500, 3100, 2200, 2800, 3900, 2400, 1600]
```

Now you do need one piece of information to pinpoint a number: **the day**. `por_dia[4]` is Friday, 3,900 (remember that indices start at 0). One direction, **one dimension**. It is called a **vector**.

Verify that nothing was lost: `sum(por_dia)` gives 18500, the same total as in Step 1. The detail does not change the total; it only distributes it.

Mateo jots down a note on a Post-it: *the detail doesn't lie; the total hides.* He doesn't yet know how useful that phrase will turn out to be.

### Step 3 — A table

“And by day and by product?”

Doña Lupe sighs, pulls out her old laptop, the one with the Virgin Mary sticker, and opens her spreadsheet:

| Day | Notebooks | Pens | Backpacks |
|---|---|---|---|
| Monday | 1,200 | 800 | 500 |
| Tuesday | 1,500 | 900 | 700 |
| Wednesday | 1,000 | 700 | 500 |
| Thursday | 1,300 | 900 | 600 |
| Friday | 2,000 | 1,200 | 700 |
| Saturday | 1,200 | 800 | 400 |
| Sunday | 800 | 500 | 300 |

In Python, each row is a list, and the table is a list of rows:

```python
por_dia_y_producto = [
    [1200, 800, 500],    # lunes
    [1500, 900, 700],    # martes
    [1000, 700, 500],    # miércoles
    [1300, 900, 600],    # jueves
    [2000, 1200, 700],   # viernes
    [1200, 800, 400],    # sábado
    [800, 500, 300],     # domingo
]
```

To pinpoint a number you need **two pieces of data: the day and the product**. `por_dia_y_producto[4][2]` is Friday (row 4), backpacks (column 2): 700. Two directions, **two dimensions**. It is called a **matrix**.

Notice the order: first the row, then the column. That order is not arbitrary; it is the order in which you wrote the lists. The outer one is the days; the inner one, the products.

“So every number has its own address,” says doña Lupe.

“Exactly, Aunt Lupe. Day and product. Street and number.”

### Step 4 — A cube: the spreadsheet with tabs

Then Mateo asks the question his aunt was avoiding.

“And the Norte branch?”

Silence. Doña Lupe clicks on the other tab of the same spreadsheet, slowly, like someone opening an envelope with bad news. Same shape—7 days, 3 products—different numbers. Her spreadsheet now has **two tabs, one per branch**, and each tab is a table.

```
        Pestaña «Centro»                Pestaña «Norte»
      Cuad.  Plumas  Moch.            Cuad.  Plumas  Moch.
Lun   1200    800    500        Lun    300    200    100
Mar   1500    900    700        Mar    400    250    150
Mié   1000    700    500        Mié    350    200    100
Jue   1300    900    600        Jue    300    150    100
Vie   2000   1200    700        Vie    500    300    200
Sáb   1200    800    400        Sáb    300    200    100
Dom    800    500    300        Dom    200    100    100
```

Mateo looks at the two tabs side by side. Something prickles at the back of his neck. But first he has to put everything into Python.

In Python, the complete spreadsheet is a list of tabs:

```python
centro = por_dia_y_producto
norte = [
    [300, 200, 100],
    [400, 250, 150],
    [350, 200, 100],
    [300, 150, 100],
    [500, 300, 200],
    [300, 200, 100],
    [200, 100, 100],
]
papeleria = [centro, norte]
```

To pinpoint a number you need **three pieces of data: the branch, the day, and the product**. `papeleria[1][4][2]` is Norte (tab 1), Friday (row 4), backpacks (column 2): 200. Three directions, **three dimensions**. It no longer has a special name: it is a **three-dimensional tensor**.

This is the mental picture worth keeping for the entire series: **a three-dimensional tensor is a spreadsheet with tabs, where all tabs have the same columns and the same number of rows.**

| What you have | To pinpoint a number you need | Dimensions | Name |
|---|---|---|---|
| $18,500 | nothing: there is only one | 0 | scalar |
| Sales for 7 days | the day | 1 | vector |
| Sales by day and product | the day and the product | 2 | matrix |
| The same, by branch | the branch, the day, and the product | 3 | 3-dimensional tensor |

**The rule that sums it all up:** the number of dimensions is the number of pieces of information you need to pinpoint a single number. Each of those directions is also called an **axis**; you will see both words.

“And that's it? A cube?” Doña Lupe peers at him over her glasses. “I thought the bank used NASA stuff.”

“They use cubes, Aunt Lupe. Just very large ones.”

### Step 5 — Counting dimensions with code

Counting brackets by eye works for small examples. With the volumes the bank handles, it doesn't. Mateo knows it: if he is going to fight a machine, his own machine has to count by itself. First, the Python you will use from here on out, in case you need a refresher:

- `type(x) == list` asks whether `x` is a list.
- `x[0]` takes the first element; `len(x)` counts how many there are.
- `lista.append(valor)` adds a value to the end of a list; `tuple(lista)` converts it into a tuple.

Create a file named `dimensiones.py` with the lists from steps 1 to 4 and, above them, this function:

```python
def contar_dimensiones(x):
    dimensiones = 0
    while type(x) == list:
        dimensiones = dimensiones + 1
        x = x[0]
    return dimensiones
```

How it works: think of Russian nesting dolls. As long as what it holds in its hands is a list, it opens the box—takes the first element, `x[0]`—and counts one. When what emerges is a number, it stops. A list of lists of lists opens three times: three dimensions. A number opens zero times: zero.

Add this to the end of the file:

```python
if __name__ == "__main__":
    print(contar_dimensiones(total))
    print(contar_dimensiones(por_dia))
    print(contar_dimensiones(por_dia_y_producto))
    print(contar_dimensiones(papeleria))
```

Run it with `python3 dimensiones.py`:

```
0
1
2
3
```

The line `if __name__ == "__main__":` is a Python detail: it ensures that the code below only runs when you execute this file directly. For now, copy it as is.

Four numbers on the screen. Doña Lupe leans in.

“Is that it? Does that save Norte?”

“No, Aunt Lupe. That tells me I understand what I have. Which is no small thing.”

### Step 6 — The shape: how many there are in each direction

Knowing that the stationery shop has three dimensions is not enough. A detective doesn't write down “there are suspects”: he writes down how many. Mateo also wants to know **how many elements are in each one**: 2 branches, 7 days, 3 products. That is called the **shape** of the tensor, and it is written as a tuple: `(2, 7, 3)`.

The shape tells you a lot at a glance:

- How many dimensions there are: how many numbers the tuple has (3).
- How large each one is: 2, 7, and 3.
- How many numbers there are in total: multiply them, 2 × 7 × 3 = 42.

Forty-two numbers. An entire week across two branches fits into forty-two numbers.

The function is almost identical to the previous one. Instead of counting, it records the length of each box it opens:

```python
def forma(x):
    tamanos = []
    while type(x) == list:
        tamanos.append(len(x))
        x = x[0]
    return tuple(tamanos)
```

Replace the final block of your file with this one:

```python
if __name__ == "__main__":
    print("total", contar_dimensiones(total), forma(total))
    print("por_dia", contar_dimensiones(por_dia), forma(por_dia))
    print("por_dia_y_producto", contar_dimensiones(por_dia_y_producto), forma(por_dia_y_producto))
    print("papeleria", contar_dimensiones(papeleria), forma(papeleria))

    assert forma(papeleria) == (2, 7, 3), "La papelería debería ser 2 sucursales × 7 días × 3 productos"
    print("Todo cuadra")
```

Output:

```
total 0 ()
por_dia 1 (7,)
por_dia_y_producto 2 (7, 3)
papeleria 3 (2, 7, 3)
Todo cuadra
```

Two new things in that output:

- `(7,)` has a comma because that is how Python writes a single-element tuple. Without the comma, `(7)` would just be the number 7.
- `()` is an empty tuple: the scalar has no directions.

And a new line in the code: `assert`. It means **“this has to be true.”** It is like the guard at the entrance: if it is, nothing happens and the program continues. If not, the program halts and tells you why. Try it yourself: change `(2, 7, 3)` to `(7, 2, 3)` and run it again:

```
Traceback (most recent call last):
  ...
AssertionError: La papelería debería ser 2 sucursales × 7 días × 3 productos
```

Change it back to `(2, 7, 3)`. This is how you will work throughout the series: each exercise comes with its `assert`, and the program tells you by itself if you made a mistake, without waiting for anyone to grade you.

“‘Todo cuadra’ (‘Everything adds up’),” Doña Lupe reads aloud. “If only the bank thought the same.”

### Step 7 — When a list of lists is NOT a tensor

It is almost ten. Mateo yawns, and the uncomfortable question hits him: what if one day someone sends an incomplete tab? Would his function notice?

There is one non-negotiable condition: **all boxes at the same level must measure the same.** All tabs with 7 rows, all rows with 3 columns. A tensor is rectangular.

Look at this list:

```python
chueca = [[1, 2, 3], [4, 5]]
```

Python accepts it without complaint: it is a list of two lists. But the first row has 3 elements and the second, 2. It is not a table; it is a jagged table. And our functions lie straight to your face:

```
>>> contar_dimensiones(chueca)
2
>>> forma(chueca)
(2, 3)
```

`forma` says `(2, 3)` because it only measured the first row. Like a witness who saw the first block and swears for the whole street. If you trust it and ask for `chueca[1][2]`, it blows up:

```
IndexError: list index out of range
```

That is why validation is necessary. For a table (two dimensions), this function compares the length of each row against that of the first. With more dimensions the check is longer; don't worry, in the next session PyTorch does the whole check for you:

```python
def es_rectangular(tabla):
    largo = len(tabla[0])
    for fila in tabla:
        if len(fila) != largo:
            return False
    return True
```

```
>>> es_rectangular(por_dia_y_producto)
True
>>> es_rectangular(chueca)
False
```

AI libraries, like PyTorch, do this check for you: when converting a jagged list into a standard tensor, they refuse. You will see this in the next session. (Special tensors exist for uneven data, but we do not use them in this series.)

Mateo stares at that `False`. A crooked data point doesn't warn you. It hides, until something blows up.

### Beyond three: the postal address

“And what if I ask you for four weeks, Aunt Lupe, instead of one?”

“I'll give them to you. But you help me organize the poster boards.”

With 4 weeks of sales, Mateo would need four pieces of information to pinpoint a number: week, branch, day, and product. Four dimensions, shape `(4, 2, 7, 3)`.

Four dimensions cannot be drawn, but they are understood just like a postal address. To get to an apartment you give the state, the neighborhood, the building, the floor, and the apartment number: five pieces of information, five dimensions. Nobody gets intimidated by that. Each piece of information chooses a box inside the previous one, just like each index in `papeleria[1][4][2]`.

Some real tensors you will encounter:

| What it is | Dimensions | Example shape |
|---|---|---|
| A small color photo | height × width × color | `(480, 640, 3)` |
| A batch of 32 photos | photo × height × width × color | `(32, 480, 640, 3)` |
| 10 seconds of video at 30 frames per second | frame × height × width × color | `(300, 720, 1280, 3)` |
| Daily summary of 1,000 customers over 30 days, with 5 data points per customer and day | customer × day × data point | `(1000, 30, 5)` |

Here we write photos as height × width × color because that is how our lists end up structured. The order is not universal: in many PyTorch programs you will see color × height × width. The important thing is knowing what each axis represents.

That video, stored uncompressed, is 829,440,000 numbers (multiply the shape). That is why videos are stored compressed (an MP4 file contains compressed video inside), and why working with uncompressed video requires so much memory.

Mateo looks at the last row of the table: 1,000 customers, 30 days, 5 data points. That is roughly what the bank must have on the Norte terminal. And he is only up to 42 numbers.

---

## Where did the word “tensor” come from?

Past midnight, Mateo is still awake in his room. He can't stop reading. It turns out that the tool he is going to use against the bank has a history of people just as stuck as he is.

Nobody invented tensors alone.

**1846.** The Irishman William Rowan Hamilton uses the word *tensor* for the first time, from the Latin *tendere*, “to stretch.” But to him it meant something else: the magnitude of a quaternion, a type of number he invented himself. The word is born, but it doesn't yet know what it is destined to become.

**1898.** The German Woldemar Voigt gives it its modern meaning. He was studying crystals, which stretch and deform differently depending on the direction in which you pull them, and he needed a single object that could hold all those directions at once. A box for multiple directions. Sounds familiar, doesn't it?

**1900.** The Italians Gregorio Ricci-Curbastro and Tullio Levi-Civita publish, cleanly organized, the mathematics we call tensor calculus today (they called it “absolute differential calculus”). And here is a detail straight out of a mystery novel: throughout their entire paper, the word “tensor” does not appear even once.

**1912.** Albert Einstein returns to Zurich stuck on his theory of gravity. The math just isn't working out. He seeks out his university friend, the mathematician Marcel Grossmann. As a classmate of both later recalled, Einstein told him:

“Grossmann, help me or I'll go crazy.”

Grossmann shows him the work of the Italians. That of Ricci-Curbastro and Levi-Civita. The one that never said “tensor.”

**1915.** March. A letter arrives for Einstein. It is signed by Levi-Civita. He tells him that a proof of the version of his theory he held at the time is wrong.

Imagine that: Einstein, corrected by mail.

Letters go back and forth through May. Einstein defends his proof stubbornly. Letters come and go. In the end, Einstein concedes that it was incomplete. In a letter dated April 2, he wrote to him: “I have never had such an interesting correspondence.” Months later he changed course, and on November 25 he presented the final equations of general relativity, written with tensors.

There is a lesson there that has nothing to do with mathematics: correcting someone is not picking a fight. Levi-Civita's “your proof is wrong” did not end in an argument: it showed Einstein a real flaw, months before the final equations, and ended in a friendship.

*Sources: Jeff Miller, “Earliest Known Uses of Some of the Words of Mathematics,” entry TENSOR (Hamilton 1846, Voigt 1898, Ricci-Curbastro and Levi-Civita), on MacTutor: [mathshistory.st-andrews.ac.uk/Miller/mathword/t/](https://mathshistory.st-andrews.ac.uk/Miller/mathword/t/). The paper by Ricci-Curbastro and Levi-Civita is “Méthodes de calcul différentiel absolu et leurs applications,” Mathematische Annalen 54, pp. 125–201; the volume bears the date 1900, though some catalogs cite it as 1901. The 1915 letters: Galina Weinstein, “Einstein the Stubborn: Correspondence between Einstein and Levi-Civita,” [arxiv.org/abs/1202.4305](https://arxiv.org/abs/1202.4305).*

Mateo turns off the lamp thinking about that. If Einstein was told “that's wrong,” he will be told so too. Better to hear it from an `assert` today than from the bank on December 23.

### A truth so nobody catches you off guard

If a physicist ever tells you that a tensor “is not a table of numbers,” they are right, from their perspective. In physics and mathematics, a tensor is a box of numbers that additionally obeys precise rules when you rotate the axes or change units. In programming and AI, *tensor* is simply the ordered box. That box can hold the numbers of a physics tensor, but on its own it brings neither its rules nor its meaning. Programmers borrowed the name from physics because it sounds better than “multidimensional array,” which is what it actually is.

And a vocabulary trap: in English, the number of dimensions is sometimes called *rank*. Do not confuse it with the “rank” of a matrix that you might have seen in linear algebra; they are different things.

---

👉 **If you come from Business Administration:** the numerical part of your spreadsheet with tabs (without headers or notes) is already a three-dimensional tensor, provided all tabs have the same columns and the same rows. Doña Lupe had had it on her laptop for years without knowing it. When someone in a meeting says “the data enters the model as tensors,” think of that: stacked tables, all sharing the same shape. And the rule from Step 7 applies to your reports: if each branch sends you its tab with different columns, no one can aggregate them without fixing them first.

---

## Exercises for session 1

The solutions are at the end of this installment. Try them first; the `assert` tells you if you got it right.

**Exercise 1 — Predict before running.** For each dataset, write on paper how many dimensions it has, its shape, and what each dimension represents. Then verify it with your functions.

```python
temperatura = 31
calificaciones = [[9, 8, 10], [7, 9, 8], [10, 10, 9], [6, 8, 7]]
foto = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
lote = [foto, foto]
```

Once you have your answers on paper, write them into these `assert` statements (replace each `?`) and run:

```python
assert (contar_dimensiones(temperatura), forma(temperatura)) == (?, ?)
assert (contar_dimensiones(calificaciones), forma(calificaciones)) == (?, ?)
assert (contar_dimensiones(foto), forma(foto)) == (?, ?)
assert (contar_dimensiones(lote), forma(lote)) == (?, ?)
print("Ejercicio 1 superado")
```

Hint: `calificaciones` represents 4 students with 3 subjects each. `foto` is an image of 2 × 2 pixels, and each pixel carries three numbers: red, green, and blue.

**Exercise 2 — Pinpoint the number.** Using `papeleria` from Step 4, write the index expression that gives each value and verify it with an `assert`:

a) What Centro sold on Wednesday in pens.
b) What Norte sold on Friday in backpacks.
c) What Norte sold on Sunday in notebooks.

Example of how a solved one looks: `assert papeleria[0][0][0] == 1200` (Centro, Monday, notebooks).

**Exercise 3 — Complete the function.** This version of `forma` has two blanks. Fill them in without looking back at Step 6, and verify with the `assert` statements:

```python
def forma2(x):
    tamanos = []
    while type(x) == list:
        tamanos.append(____)
        x = ____
    return tuple(tamanos)

assert forma2(por_dia) == (7,)
assert forma2(papeleria) == (2, 7, 3)
assert forma2(31) == ()
print("Ejercicio 3 superado")
```

**Exercise 4 — Your own data.** Write as nested lists the grades of 3 students, in 4 subjects, across 3 midterms. Make up the numbers. Decide the order of the dimensions yourself and write it in a comment. Afterwards:

a) Write the `assert` for its shape.
b) Write the index of the second midterm of the third subject of the first student, and verify it with an `assert` against the grade you invented at that position.

This exercise does not have a single correct answer: it depends on the order you choose. What must hold true is that the `assert` passes and that your comment states what each dimension represents.

**Exercise 5 — Spot the jagged one.** One of these tables cannot be a tensor. Find it by eye and check it with `es_rectangular`:

```python
a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8]]
c = [[0], [0], [0]]
```

Write your answers into these `assert` statements (replace each `?` with `True` or `False`):

```python
assert es_rectangular(a) == ?
assert es_rectangular(b) == ?
assert es_rectangular(c) == ?
print("Ejercicio 5 superado")
```

What would `forma(b)` tell you? Why is it not reliable?

---

## Common errors (and how to spot them)

| Error | How it looks | How to spot it | What to do |
|---|---|---|---|
| Counting indices from 1 | You ask for Friday with `por_dia[5]` and get Saturday | The number does not match the table | Indices start at 0: Monday is 0, Friday is 4 |
| Reversing the order of indices | `papeleria[4][1][2]` instead of `papeleria[1][4][2]` | `IndexError`, because there are only 2 branches | The order of indices matches the order of the lists: from outside in |
| Trusting `forma` on a jagged list | `forma` says `(2, 3)` and then an index blows up | `es_rectangular` returns `False` | Clean the data before using them; never fill in missing values with fabricated numbers without disclosing it |
| Passing an empty list to `forma` or `contar_dimensiones` | `IndexError: list index out of range` inside the function | It blows up with `[]` | An empty box has no first element to open; check that your data contain something |
| Forgetting the comma in `(7,)` | `assert forma(por_dia) == (7)` fails even though the shape is correct | The `assert` blows up on something that “looks fine” | A single-element tuple requires a comma: `(7,)` |

---

## Review for next session

Session 2 begins with these three questions. Answer them without looking at the lesson:

1. How do you know how many dimensions a dataset has, without counting brackets?
2. What does the shape `(2, 7, 3)` tell you, and how many numbers are there in total?
3. What condition must a list of lists satisfy to be a tensor?

🔁 **Next week:** Mateo leaves lists behind and grabs a serious tool. You install PyTorch and convert doña Lupe's papelería into your first real tensor. You will see that PyTorch already comes with shape built in (there it is called `shape`), and that it refuses to accept jagged lists.

That night, before closing his laptop, Mateo reopened both tabs. Centro, Norte. Centro, Norte. The same columns, the same seven days. And yet, on the Norte tab there was something that didn't add up. But to see it, he needed a real tool.

---

## Solutions (don't look at them before trying)

**Exercise 1.**

```
temperatura      0 ()             un solo número
calificaciones   2 (4, 3)         alumno × materia
foto             3 (2, 2, 3)      renglón de puntos × punto × color
lote             4 (2, 2, 2, 3)   foto × renglón × punto × color
```

The `assert` statements end up like this: `(0, ())`, `(2, (4, 3))`, `(3, (2, 2, 3))`, and `(4, (2, 2, 2, 3))`.

The `lote` adds a dimension at the beginning: which photo. That is almost always how image models work: they receive batches, and even a single photo usually goes in as a batch of one.

**Exercise 2.**

```python
assert papeleria[0][2][1] == 700   # a) Centro, miércoles, plumas
assert papeleria[1][4][2] == 200   # b) Norte, viernes, mochilas
assert papeleria[1][6][0] == 200   # c) Norte, domingo, cuadernos
```

**Exercise 3.** The blanks are `len(x)` and `x[0]`: record the length of the box and then open it.

**Exercise 4.** One possible answer, with the order student × subject × midterm:

```python
# alumno × materia × parcial
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]
assert forma(parciales) == (3, 4, 3)
assert parciales[0][2][1] == 9   # primer alumno, tercera materia, segundo parcial
```

If you chose subject × student × midterm, your shape is `(4, 3, 3)` and your index is `parciales[2][0][1]`. Both are fine as long as the comment says so. Choosing the order of dimensions is a design decision, and in Session 3 you will see why it matters.

**Exercise 5.** The `assert` values are `True`, `False`, and `True`. The jagged one is `b`: its last row has 2 elements and the others, 3. `es_rectangular(b)` gives `False`. `forma(b)` would say `(3, 3)` because it only measures the first row, and it is not reliable for that reason: it does not check the rest.

---

## What comes next

In Session 2 you leave lists behind and use a tool built for this purpose. If you want to get ahead, verify that you have Python 3.11 or newer with `python3 --version`.

---

*I write from what we see every day at Hábil, where we build systems for banking, insurance, and enterprise in Mexico. If at your work or your family's business you run into a problem like this, let's talk: [habil.mx](https://www.habil.mx/en/)*
