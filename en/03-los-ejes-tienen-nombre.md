# Tensors from scratch (3): axes have names

*How to extract a slice of a tensor along any of its axes—a number, a day, a product, the red channel of a photo—and what happens to the shape when you do.*

*Tensors from scratch · lesson A, session 3 of 8*

*Setting: the characters and the story are fictional. Technical details, historical facts, and real figures include their sources.*

**Monday, November 16, 7:40 p.m. Centro stationery store.**

The metal shutter is halfway down. Inside, it smells of fresh paper and reheated coffee. Doña Lupe is counting bills at the counter. Mateo has his laptop open next to the cash register, between a jar of pencils and a stack of poster boards.

Thirty-seven days left until December 23.

Last week they fixed the Norte file, the one with the cut-off row. Now PyTorch accepts it: a tensor of shape `(2, 7, 3)`. Two branches, seven days, three products. Forty-two numbers.

“So that's it?” asks doña Lupe without looking up. “Do you already know who's stealing from me?”

“Not yet, Auntie. I have the numbers. What I need is to know how to look at them.”

“Well, look at them. That's what you have eyes for.”

Mateo laughs, but she's right. Forty-two numbers lumped together say nothing. They're like a storage room with everything piled up: the answer is right there, but you can't find it. What he needs is to pull out **pieces**. A day. A product. A branch. Put them side by side and see what doesn't add up.

He types a line. Out comes Friday for both branches.

He freezes.

“Auntie. What's your best day of the week?”

“Friday, which else? Kids get out of school, they come with their parents, they buy everything.”

“At Centro, yes. On Friday it sells 2,000 in notebooks. No other day comes close.”

“And at Norte?”

Mateo turns the screen. `500, 300, 200`.

Doña Lupe puts on her glasses. Takes them off again.

“That's what Norte sells on any random Tuesday.”

Almost. Norte's Tuesday is 400, 250, and 150. Friday, the big day, barely beats it by a hair.

And there's something worse in the last column. Backpacks. The most expensive item in the store. At Norte, almost every day: 100. As if no one ever touched them.

Mateo feels a sinking feeling in his stomach. He doesn't know what it means yet. But he knows he just found his first real lead.

---

In the previous session, you installed PyTorch and turned doña Lupe’s stationery store into your first real tensor, with shape `(2, 7, 3)`. Today you learn what Mateo did that night: pull out pieces from it. Not just a single number, but an entire day, an entire product, or a trimmed week, along whichever axis you choose.

## Warm-up review

Answer without looking at the previous session. The answers are at the end, in the solutions.

1. What does `t.shape` tell you, and how do you verify it with an `assert`?
2. Why does `torch.tensor([[1, 2, 3], [4, 5]])` fail?
3. What `dtype` comes out of `torch.tensor([1, 2, 3])` and which from `torch.tensor([1.0, 2.0])`?

> **The code for this session** is in the course's public repository, in the `sesion-03` folder: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-03). Type it yourself first, because that's how you learn; the repository is for comparing when something doesn't come out the same for you. Each program comes with the output it should give you.

## Objectives of this session

By the end, you will be able to do these four things. Next to each one is the exercise that demonstrates it.

| # | By the end you can… | Demonstrated by |
|---|---|---|
| 1 | **Name** each axis of a tensor and **write it** in a comment above the code | Exercises 10 and 13 |
| 2 | **Extract** a single number with an index per axis, `papeleria[1, 4, 2]`, and **convert it** to a Python number with `.item()` | Exercise 10 |
| 3 | **Extract** a slice along any axis with `:` and with a range like `1:5`, and **predict** its shape before running it | Exercises 11 and 12 |
| 4 | **Decide** the axis order of a photo and **extract** a single color | Exercise 13 |

**What you need:** the session 2 environment with PyTorch installed (activate it before starting) and a text editor. Today you install nothing new.

---

## Why it matters

Mateo didn't find the lead by adding everything up. He found it by slicing: one day across two branches, one column for one branch. That's how almost any data system works.

A program that inspects a photo of a crashed car almost never uses it whole: it crops the bumper area, or keeps only a single color. A bank that stores the purchases of a thousand customers over 30 days, with shape `(1000, 30, 5)`, sometimes wants a single customer in full and sometimes just yesterday's data for everyone. It's the same tensor. What changes is which axis you slice it along.

And here comes the trap. Many of the errors you'll see with models aren't about math: they're about shape. You asked for a slice, thought it had one shape, and it had another. Nothing blows up. It just gives you the wrong number, and you believe it. Today you learn to predict that shape before running the code, so that doesn't happen to you.

## Step 1 — Name every axis

Before slicing anything, Mateo does something that seems redundant and isn't: he writes what each axis is above the code. Because PyTorch doesn't know what “Norte” is. It doesn't know what “Friday” is.

To PyTorch, `papeleria` is a box of `2 × 7 × 3` numbers and nothing more. The one who knows the first axis is branches is you. That's why today's first rule is: **say it out loud and write it in a comment**, like the label on a kitchen drawer.

Create a file `ejes.py` and start like this:

```python
import torch

# Ejes de la papelería:
#   eje 0 = sucursal  (0 Centro, 1 Norte)
#   eje 1 = día       (0 lunes ... 6 domingo)
#   eje 2 = producto  (0 cuadernos, 1 plumas, 2 mochilas)
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
print(papeleria.shape)
```

Run it with `python3 ejes.py`:

```
torch.Size([2, 7, 3])
```

Read it with the names attached: **2 branches, 7 days, 3 products.** Axes are numbered from 0, just like the indices of a list: axis 0, axis 1, axis 2. When you read “along axis 1”, translate “along days”.

“And what's that for?” says doña Lupe, leaning in. “You already know which is which.”

“Today I do. Three weeks from now, at two in the morning, I won't remember.”

“Like your uncle with the keys. That's why I put labels on them.”

## Step 2 — A single number: `papeleria[1, 4, 2]`

The first thing Mateo wants to see is a single number: what Norte sold on Friday in backpacks. If there's something strange there, he wants to touch it with his finger.

In session 1 you pinpointed a number like this: `papeleria[1][4][2]`. With a tensor that still works, but there is a better way: **all indices inside the same bracket, separated by commas.**

From here on, keep adding each block to the end of `ejes.py` and run it again; in each output I show you only the new lines. Add:

```python
print(papeleria[1][4][2])
print(papeleria[1, 4, 2])
```

```
tensor(200)
tensor(200)
```

Both give the same thing: Norte, Friday, backpacks, $200. On the best day of the week. Mateo jots down the number on a Post-it and sticks it to the edge of the screen.

The difference is in how they work under the hood:

- `papeleria[1][4][2]` is **three steps**: it pulls out the Norte tab, from there Friday, from Friday backpacks. Box inside box.
- `papeleria[1, 4, 2]` is **a single step**: you give it the full address and it goes straight there.

In step 7 you will see a case where separate brackets deceive you. From now on, use commas.

### What comes out is not a number: it is a zero-dimensional tensor

The output doesn't say `200`, it says `tensor(200)`: the number comes wrapped. Add this:

```python
x = papeleria[1, 4, 2]
print(x.shape, x.ndim)
print(type(x))
print(x.item())
print(type(x.item()))
```

```
torch.Size([]) 0
<class 'torch.Tensor'>
200
<class 'int'>
```

Empty shape, zero dimensions: it's a **scalar**, like the $18,500 total from session 1, but in tensor form. A candy that still has its wrapper on. `.item()` removes the wrapper and gives you a normal Python `int`, to print it, save it in a list, or send it to a report.

And the usual self-correction: shape first, value second.

```python
assert papeleria[1, 4, 2].shape == (), "Un solo número no tiene ejes"
assert papeleria[1, 4, 2].item() == 200, "Norte, viernes, mochilas: 200"
print("Paso 2 correcto")
```

```
Paso 2 correcto
```

`torch.Size([])` compares cleanly against the empty tuple `()`: your shapes are still written as tuples.

A single number doesn't prove anything. Two hundred pesos in backpacks could just be one bad Friday and that's it. Mateo needs to see Norte's entire week in one go.

## Step 3 — A single index: the axis disappears

What happens if you provide fewer indices than axes? Try with just one:

```python
print(papeleria[1])
print(papeleria[1].shape)
```

```
tensor([[300, 200, 100],
        [400, 250, 150],
        [350, 200, 100],
        [300, 150, 100],
        [500, 300, 200],
        [300, 200, 100],
        [200, 100, 100]])
torch.Size([7, 3])
```

The complete Norte tab. Mateo casts his eyes down the last column: 100, 150, 100, 100, 200, 100, 100. Flat as a tabletop.

But look at the shape, which is today's main character. It went from `(2, 7, 3)` to `(7, 3)`: **the branch axis disappeared.** You already picked one; looking at the Norte tab there are no longer “branches”, only days and products.

With two indices, the same thing:

```python
print(papeleria[1, 4])
print(papeleria[1, 4].shape)
```

```
tensor([500, 300, 200])
torch.Size([3])
```

Norte, Friday, all three products. Two axes gone; the product axis remains. The first half of today's rule:

> **Every index you provide eats its axis.** Three axes minus one index: two remain. Three axes minus three indices: none remain, and that's a scalar.

## Step 4 — The colon: “all of them, please”

Now comes the line that gave him the sinking feeling in his stomach. Mateo doesn't want Norte alone: he wants Norte **next to** Centro, on the same day, so the difference has nowhere to hide.

And what if you want **Friday for both branches**? You have to skip the first axis without choosing anything. That's what the colon, `:`, is for—inside an index, it means “all of them”.

```python
viernes = papeleria[:, 4]
print(viernes)
print(viernes.shape)
```

```
tensor([[2000, 1200,  700],
        [ 500,  300,  200]])
torch.Size([2, 3])
```

Read it with the axis names: **all branches, day 4 (Friday), and since you said nothing about the product, all products.** Top row, Centro; bottom row, Norte.

There it is. Top, 2000. Bottom, 500. Same day, same school dismissal. One line of code and the difference is out in plain sight, without tabs or copy-pasting.

Shape `(2, 3)`: the day axis is gone because you chose one; the branch axis stayed because you asked for all of them; the product axis, because you didn't touch it. Like ordering tacos “with everything”: you don't choose onion or cilantro, you take whatever is there.

### Check it against the version with a `for`

Mateo doesn't trust a single line. If he's going to show this to the bank, he wants to derive it by two paths. With what you already knew, it would be number by number:

```python
a_mano = []
for s in range(2):
    a_mano.append([papeleria[s][4][p].item() for p in range(3)])
print(a_mano)
print(viernes.tolist())
```

```
[[2000, 1200, 700], [500, 300, 200]]
[[2000, 1200, 700], [500, 300, 200]]
```

`.tolist()` is a cousin of `.item()`: it converts an entire tensor back into Python lists, which you compare with `==` as always. Both yield the same; leave it verified in writing:

```python
assert viernes.shape == (2, 3), "Viernes: 2 sucursales × 3 productos"
assert viernes.tolist() == [[2000, 1200, 700], [500, 300, 200]]
print("Paso 4 correcto")
```

```
Paso 4 correcto
```

## Step 5 — A range: `papeleria[0, 1:5]`

“Let's see,” says doña Lupe. “Show me what a normal week looks like. Centro's. But without Monday, which is always dead, and without the weekend.”

Mateo needs a range: from Tuesday to Friday. The colon can also request a range, just like in Python lists: `lista[1:5]` gives elements 1, 2, 3, and 4. **It includes the first, leaves out the last.** With tensors, along any axis.

Doña Lupe wants Centro from Tuesday to Friday:

```python
tramo = papeleria[0, 1:5]
print(tramo)
print(tramo.shape)
```

```
tensor([[1500,  900,  700],
        [1000,  700,  500],
        [1300,  900,  600],
        [2000, 1200,  700]])
torch.Size([4, 3])
```

Centro, days 1 through 4 (day 5, Saturday, is left out), all products. Shape `(4, 3)`: the branch axis was removed by the index; the days axis **stayed**, but measures 4 instead of 7. A range trims the axis, it doesn't remove it. Compare it against the Python `centro` list:

```python
print(centro[1:5])
assert tramo.shape == (4, 3), "De martes a viernes: 4 días × 3 productos"
assert tramo.tolist() == centro[1:5]
print("Paso 5 correcto")
```

```
[[1500, 900, 700], [1000, 700, 500], [1300, 900, 600], [2000, 1200, 700]]
Paso 5 correcto
```

This extracted piece, the one obtained with a colon, is called a **slice** (in Spanish documentation you'll see the term *rebanada*).

## Step 6 — The last axis: `papeleria[:, :, 2]`

Doña Lupe puts down the bills. She isn't counting anymore. Now she's the one asking: **how much did each branch sell in backpacks each day?** Backpacks bring in the most money. If something is wrong, she wants to see it there.

Product is axis 2, the last one; to reach it, you pass through the other two with “all”:

```python
mochilas = papeleria[:, :, 2]
print(mochilas)
print(mochilas.shape)
```

```
tensor([[500, 700, 500, 600, 700, 400, 300],
        [100, 150, 100, 100, 200, 100, 100]])
torch.Size([2, 7])
```

Shape `(2, 7)`: the product axis is gone; each row is a branch and each column a day.

Doña Lupe leans so close to the screen that Mateo has to lean back. Top, Centro: 500, 700, 500, 600… Bottom, Norte: 100, 150, 100, 100…

“Mijo, more people walk past Norte than here. It's right next to the bus stop.”

Mateo doesn't answer. He doesn't have an answer yet.

In a spreadsheet you would have to copy the last column of each tab and paste them side by side. Here it was a single line. With `for`, it takes two loops:

```python
m = []
for s in range(2):
    fila = []
    for d in range(7):
        fila.append(papeleria[s, d, 2].item())
    m.append(fila)
print(m)

assert mochilas.shape == (2, 7), "Mochilas: 2 sucursales × 7 días"
assert mochilas.tolist() == m
print("Paso 6 correcto")
```

```
[[500, 700, 500, 600, 700, 400, 300], [100, 150, 100, 100, 200, 100, 100]]
Paso 6 correcto
```

Seven lines versus one. That's the beauty of tensors: when you're in a hurry—and Mateo has 37 days left—the question and the answer fit on the same line.

## Step 7 — Why commas: the trap of `papeleria[:][4]`

That same night, Mateo almost made a mistake. Before arriving at the good line in step 4, he tried what came naturally to him. If you were to ask for Friday for both branches with separate brackets, the natural way would be this:

```python
print(papeleria[:][4])
```

```
Traceback (most recent call last):
  ...
IndexError: index 4 is out of bounds for dimension 0 with size 2
```

Read the error word by word: “index 4 is out of bounds for dimension 0 with size 2”. Dimension 0? But you wanted the day!

Separate brackets work in steps, and each step starts again from the first axis **of whatever was left** from the previous step, not from the axis you had in mind:

1. `papeleria[:]` is “all branches”: it returns the entire tensor, identical.
2. `[4]` falls right back on the first axis: you are asking for branch 4, and there are only two.

With commas, each position is a fixed axis. And this time you got lucky: it blew up. If doña Lupe had 7 branches, `papeleria[:][4]` would give you branch 4 without complaining, and you would believe it was Friday. **The error that doesn't blow up is the dangerous one.**

Mateo imagines the scene: him standing in front of Licenciado Ortega, showing “the Fridays” that are actually another branch. The bank wouldn't give him a second chance. From that night on, everything with commas.

## Step 8 — `papeleria[0]` versus `papeleria[0:1]`

One last thing bothers Mateo. There are two ways to ask for a single branch, and they seem to give the same thing. Seem to.

Use the index and the slice to request **the same thing**, the Centro branch:

```python
print(papeleria[0].shape)
print(papeleria[0:1].shape)
```

```
torch.Size([7, 3])
torch.Size([1, 7, 3])
```

Same 21 numbers, different shape. `papeleria[0]` is an index: **the axis disappears** and you are left with a table. `papeleria[0:1]` is a slice of a single branch: **the axis stays**, even though it measures 1, and you are left with a spreadsheet with a single tab:

```python
print(papeleria[0:1])
```

```
tensor([[[1200,  800,  500],
         [1500,  900,  700],
         [1000,  700,  500],
         [1300,  900,  600],
         [2000, 1200,  700],
         [1200,  800,  400],
         [ 800,  500,  300]]])
```

Three brackets at the beginning: it is still three-dimensional. It's taking a sheet out of the folder versus keeping a folder with a single sheet. And it matters to PyTorch: a model expecting “a folder of photos”, with shape `(N, C, H, W)`, can reject a standalone photo with shape `(C, H, W)`. Today's complete rule:

> **With an index, the axis disappears. With a slice, the axis stays** (even if it measures 1).

| What you write | Which axis goes away | Shape | What it is |
|---|---|---|---|
| `papeleria[1, 4, 2]` | all three | `()` | Norte, Friday, backpacks |
| `papeleria[1]` | branch | `(7, 3)` | the Norte tab |
| `papeleria[1, 4]` | branch and day | `(3,)` | Norte, Friday, the 3 products |
| `papeleria[:, 4]` | day | `(2, 3)` | Friday for both branches |
| `papeleria[0, 1:5]` | branch | `(4, 3)` | Centro from Tuesday to Friday |
| `papeleria[:, :, 2]` | product | `(2, 7)` | backpacks, by branch and day |
| `papeleria[0:1]` | none | `(1, 7, 3)` | a spreadsheet with only the Centro tab |

The trick to predicting the shape: **write down the original shape, cross out each axis where you put an index, and replace with the length of the range each axis where you put a slice.** For `papeleria[0, 1:5]`: `(2, 7, 3)` → cross out 2 → change 7 to 4 (from 1 to 4) → `(4, 3)`.

## Step 9 — A 2 × 3 pixel photo

**Tuesday, November 17, 1:15 a.m. Mateo's room.**

He can't sleep. On his phone he has a photo doña Lupe sent him in the afternoon: the Norte terminal, with its tangled cord. “So you can see it really is connected,” she wrote.

Mateo thinks that a photo is also a tensor. And that if he knows how to slice the stationery store, he knows how to slice a photo.

To check, he starts with a tiny handmade photo: 2 pixels high by 3 wide. Each pixel carries three numbers, **red, green, and blue**, from 0 (nothing) to 255 (full). Yellow, for example, is red and green at maximum, without blue.

Create a file `foto.py`:

```python
import torch

# Foto de 2 × 3 puntos, en orden alto × ancho × color
#   eje 0 = renglón (alto), eje 1 = punto en el renglón (ancho), eje 2 = color (0 rojo, 1 verde, 2 azul)
foto = torch.tensor([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],          # arriba: rojo, verde, azul
    [[255, 255, 255], [0, 0, 0], [255, 255, 0]],      # abajo: blanco, negro, amarillo
])
print(foto.shape)
```

```
torch.Size([2, 3, 3])
```

The shape has two 3s: the first is the pixels of width; the second, the colors. Without the comment, nobody knows which is which, not even you in two weeks. Extract **only the red channel**, just like backpacks:

```python
rojo = foto[:, :, 0]
print(rojo)
print(rojo.shape)
assert rojo.shape == (2, 3), "El canal rojo es una tabla de 2 × 3"
assert rojo.tolist() == [[255, 0, 0], [255, 0, 255]]
print("Canal rojo correcto")
```

```
tensor([[255,   0,   0],
        [255,   0, 255]])
torch.Size([2, 3])
Canal rojo correcto
```

On the top row, only the red pixel has red; on the bottom, white and yellow do, black does not. That table is a black-and-white photo of how much red there is in each pixel: that's what “separating channels” is. A single pixel, with its three colors:

```python
print(foto[1, 2])
```

```
tensor([255, 255,   0])
```

Bottom right: yellow.

It works just like backpacks. Same colons, same rules. Mateo smiles in the dark. What he learned with the stationery store works for any box of numbers.

## Step 10 — Axis order is a design decision

Right there, at one in the morning, an uncomfortable question occurs to him: who decided that color should be the last axis? He did. You stored the photo as **height × width × color**, but nobody forced you to. It can be stored as **color × height × width**: the red table, then the green, then the blue.

```python
# La misma foto en orden color × alto × ancho
#   eje 0 = color, eje 1 = renglón, eje 2 = punto en el renglón
foto_cha = torch.tensor([
    [[255, 0, 0], [255, 0, 255]],      # canal rojo
    [[0, 255, 0], [255, 0, 255]],      # canal verde
    [[0, 0, 255], [255, 0, 0]],        # canal azul
])
print(foto_cha.shape)
print(foto_cha[0])
assert foto_cha[0].tolist() == foto[:, :, 0].tolist()
print("Mismo rojo, otro orden")
```

```
torch.Size([3, 2, 3])
tensor([[255,   0,   0],
        [255,   0, 255]])
Mismo rojo, otro orden
```

Same 18 numbers, different shape: `(3, 2, 3)`. And red is no longer `foto[:, :, 0]`, it's just `foto_cha[0]`. **What you put first is what is easiest to extract.**

Which one is correct? Both. It's like organizing your closet by type of clothing or by outfit, or like doña Lupe organizes the back room: by branch or by product. The merchandise is the same; what changes is what you find quickly. That's why axis order is a **design decision**: you make it thinking about what you will ask for most often.

Conventions already exist. In PyTorch, the common convention for images is **channels first**, which in the documentation you will see as **C × H × W** (*channels*, *height*, *width*; [TorchVision documentation](https://docs.pytorch.org/vision/stable/transforms.html)); a batch of photos adds the axis of which photo at the beginning. Other tools and many image programs use height × width × color. Neither is wrong; what is bad is not knowing which one you have in hand. That's why, with photos, **the comment with the axes always goes above the tensor.**

| Order | Shape of your photo | The red channel | A pixel with its 3 colors |
|---|---|---|---|
| height × width × color | `(2, 3, 3)` | `foto[:, :, 0]` | `foto[1, 2]` |
| color × height × width | `(3, 2, 3)` | `foto_cha[0]` | `foto_cha[:, 1, 2]` |

If you run `foto_cha[:, 1, 2]`, it gives you `tensor([255, 255,   0])`: the exact same yellow.

Mateo closes the laptop at 2:03. On the Post-it on the screen the 200 is still there. It no longer seems like a number to him. It seems like a question.

---

👉 **If you come from Business Administration:** what Mateo did that night is what you do with a spreadsheet when you filter or hide columns: “only Friday across all branches” or “only backpacks across all tabs.” Here each is a single line. And what you learned in step 10 helps you in any report: when they ask for “sales by branch and by month,” ask what goes in the rows and what in the columns. It is a design decision, made thinking about who is going to read it.

---

## Session 3 exercises

The solutions are at the end. Work in your `ejes.py`, which already has `papeleria` and the comment with the axes. In exercise 12, the already written `assert`s tell you if you got it right; in 10, 11, and 13, compare your work with the solutions.

**Exercise 10 — Pinpoint the number, now with commas.** With an index per axis and `.item()`, write the `assert` for each value:

a) What Centro sold on Wednesday in pens.
b) What Norte sold on Sunday in notebooks.
c) What Centro sold on Saturday in backpacks.
d) Check with an `assert` that your answer to a) yields the same thing using the separate brackets from session 1.

Example of a solved one: `assert papeleria[0, 0, 0].item() == 1200` (Centro, Monday, notebooks).

**Exercise 11 — Predict the shape before running.** On paper, write the shape of each expression and what it represents, using the trick from step 8. Afterwards, run it and compare.

```python
for nombre, x in [
    ("a) papeleria[0]", papeleria[0]),
    ("b) papeleria[0, 6]", papeleria[0, 6]),
    ("c) papeleria[:, 6]", papeleria[:, 6]),
    ("d) papeleria[:, :, 0]", papeleria[:, :, 0]),
    ("e) papeleria[1, 0:3]", papeleria[1, 0:3]),
    ("f) papeleria[1, 0:3, 1]", papeleria[1, 0:3, 1]),
    ("g) papeleria[0:1, 6]", papeleria[0:1, 6]),
    ("h) papeleria[1, 4, 2]", papeleria[1, 4, 2]),
]:
    print(nombre, x.shape)
```

If you missed any, look for which axis you thought stayed and disappeared, or vice versa.

**Exercise 12 — Fill in the blanks.** Fill in each `____` and check with the `assert`s:

```python
# a) El fin de semana (sábado y domingo) de Norte, los 3 productos
fin_norte = papeleria[____, ____]
assert fin_norte.shape == (2, 3), "Fin de semana de Norte: 2 días × 3 productos"
assert fin_norte.tolist() == [[300, 200, 100], [200, 100, 100]]

# b) Las plumas del Centro de lunes a miércoles
plumas = papeleria[0, ____, ____]
assert plumas.shape == (3,), "Plumas del Centro: 3 días"
assert plumas.tolist() == [800, 900, 700]

# c) Lo mismo que b), con un for
a_mano = []
for d in range(____):
    a_mano.append(papeleria[0, d, 1].____)
assert plumas.tolist() == a_mano
print("Ejercicio 12 superado")
```

Hint for a): Sunday is day 6, and a slice leaves out its last number.

**Exercise 13 — Your own photo.** Create by hand a photo **3 pixels high by 2 wide**, stored in the order customary for PyTorch: **color × height × width** (`C × H × W`). Choose the colors of the 6 pixels yourself (red, green, blue, white, black, yellow, or whichever you want) and write them in a comment. Afterwards:

a) Write the `assert` for its shape.
b) Extract the green channel. Write the `assert` for its shape and the one for its value against a hand-written list.
c) Extract the top row with its three colors. What shape does it have? Write its two `assert`s.
d) Extract the bottom-right pixel with its three colors. Write its two `assert`s.

There is no single correct answer; what must hold true is that the `assert`s pass and the comment says what each axis is.

---

## Common errors (and how to spot them)

| Error | What it looks like | How to notice it | What to do |
|---|---|---|---|
| Asking for an index that doesn't exist | `papeleria[1, 7, 2]` gives `IndexError: index 7 is out of bounds for dimension 1 with size 7` | The message tells you which axis (*dimension 1*, days) and how large it is (*size 7*) | Indices run from 0 to size minus 1: days run from 0 to 6 |
| Reversing the axis order | `papeleria[4, 1, 2]` instead of `papeleria[1, 4, 2]` gives `IndexError: index 4 is out of bounds for dimension 0 with size 2` | It blows up on *dimension 0*, which has only 2 branches | Read your axis comment: the first index is always the branch |
| Chaining brackets with `:` | `papeleria[:][4]` gives the same *dimension 0* error, or worse, doesn't blow up and gives you something else | The result is not what you asked for, or it blows up on an axis you didn't touch | All indices inside a single bracket: `papeleria[:, 4]` |
| Believing the range includes the last element | You ask for Tuesday to Friday with `1:4` and Friday is missing | The shape `assert` says 3 days where you expected 4 | The last number is excluded: Tuesday to Friday is `1:5` |
| Confusing an index with a slice | `papeleria[0:1]` has shape `(1, 7, 3)` and you expected `(7, 3)` | The shape `assert` fails with an extra 1 | Index: the axis disappears. Slice: the axis stays, even if it measures 1 |
| Using `.item()` on multiple numbers | `papeleria[0, 0].item()` gives `RuntimeError: a Tensor with 3 elements cannot be converted to Scalar` | The message tells you how many elements it had (3) | `.item()` is only for a single number; for several, use `.tolist()` |
| Comparing a slice against a number in an `assert` | `assert papeleria[0, 0] == 1200` gives `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` | The `assert` blows up, not the indexing | You missed an index (you meant `papeleria[0, 0, 0]`), or compare the shape and then `.tolist()` against a list |

A trap that doesn't blow up: `papeleria[0, 5:20]` produces no error even though there are only 7 days; it gives you Saturday and Sunday, shape `(2, 3)`, just as Python lists would do. If a range gives you less than expected, check its numbers.

---

## Review for next session

Session 4 begins with these three questions. Answer them without looking at the lesson:

1. What shape does `papeleria[:, 4]` have and what does it represent?
2. What difference in shape is there between `papeleria[0]` and `papeleria[0:1]`?
3. How do you extract backpack sales for both branches across all days?

🔁 **Next week:**

Tuesday, 8:10 in the morning. Mateo calls doña Lupe before going into class. He tells her about Friday. About the backpacks that don't budge.

On the other end of the line, a long silence. You can hear Centro's shutter rolling up.

“Well, if backpacks aren't selling,” doña Lupe says at last, “what are they charging people for?”

Mateo doesn't know. But he knows how to keep going: stop pulling out slices and start **summarizing** them. Totals and averages by axis: how much each branch sold over the week, what the average was for each day, all on a single line and verified against a `for`. When he sums up Norte's week, the number is going to point at someone.

---

## Solutions (don't look before trying)

**Warm-up review.**

1. `t.shape` tells you how many elements are in each axis: for the stationery store, 2 branches, 7 days, and 3 products. It is verified with `assert papeleria.shape == (2, 7, 3)`.
2. Because it is not rectangular: the first row has 3 numbers and the second, 2. PyTorch refuses and says so: `ValueError: expected sequence of length 3 at dim 1 (got 2)`.
3. `torch.tensor([1, 2, 3])` comes out as `torch.int64` (integers) and `torch.tensor([1.0, 2.0])` comes out as `torch.float32` (decimals).

**Exercise 10.**

```python
assert papeleria[0, 2, 1].item() == 700    # a) Centro, miércoles, plumas
assert papeleria[1, 6, 0].item() == 200    # b) Norte, domingo, cuadernos
assert papeleria[0, 5, 2].item() == 400    # c) Centro, sábado, mochilas
assert papeleria[0][2][1].item() == papeleria[0, 2, 1].item()   # d)
print("Ejercicio 10 superado")
```

```
Ejercicio 10 superado
```

Parts a) and b) are the same data points you looked up in parts a) and c) of exercise 2 in session 1, now with commas.

**Exercise 11.**

```
a) papeleria[0] torch.Size([7, 3])
b) papeleria[0, 6] torch.Size([3])
c) papeleria[:, 6] torch.Size([2, 3])
d) papeleria[:, :, 0] torch.Size([2, 7])
e) papeleria[1, 0:3] torch.Size([3, 3])
f) papeleria[1, 0:3, 1] torch.Size([3])
g) papeleria[0:1, 6] torch.Size([1, 3])
h) papeleria[1, 4, 2] torch.Size([])
```

a) the Centro tab; b) Centro's Sunday; c) Sunday for both branches; d) notebooks for both, across all days; e) Norte from Monday to Wednesday; f) Norte's pens from Monday to Wednesday; g) Centro's Sunday preserving the branch axis, which measures 1; h) Norte, Friday, backpacks. The tricky one is g): it looks like b), but the branch was requested with a slice.

**Exercise 12.**

```python
fin_norte = papeleria[1, 5:7]       # a)
plumas = papeleria[0, 0:3, 1]       # b)
for d in range(3):                  # c)
    a_mano.append(papeleria[0, d, 1].item())
```

```
Ejercicio 12 superado
```

In a), `5:7` takes Saturday (5) and Sunday (6) and leaves out 7, which doesn't exist anyway.

**Exercise 13.** One possible answer:

```python
import torch

# Foto de 3 × 2 puntos, en orden color × alto × ancho
#   eje 0 = color (0 rojo, 1 verde, 2 azul)
#   eje 1 = renglón (0 arriba ... 2 abajo)
#   eje 2 = punto en el renglón (0 izquierda, 1 derecha)
# Los puntos: arriba rojo y verde; en medio azul y blanco; abajo negro y morado
foto = torch.tensor([
    [[255, 0], [0, 255], [0, 128]],      # canal rojo
    [[0, 255], [0, 255], [0, 0]],        # canal verde
    [[0, 0], [255, 255], [0, 128]],      # canal azul
])
assert foto.shape == (3, 3, 2), "3 colores × 3 renglones × 2 puntos"

verde = foto[1]
assert verde.shape == (3, 2)
assert verde.tolist() == [[0, 255], [0, 255], [0, 0]]

arriba = foto[:, 0]
assert arriba.shape == (3, 2)
assert arriba.tolist() == [[255, 0], [0, 255], [0, 0]]

esquina = foto[:, 2, 1]
assert esquina.shape == (3,)
assert esquina.tolist() == [128, 0, 128]
print("Ejercicio 13 superado")
```

```
Ejercicio 13 superado
```

In c), the shape is `(3, 2)`, not `(2, 3)`: extracting a slice doesn't rearrange the remaining axes; it leaves them in the order they were in, and color comes first.

---

## What comes next

In session 4, the Centro total will be the sum of `papeleria[0]`. If you want to get ahead, calculate with a `for` how much each branch sold in the week. You already know Centro since session 1: $18,500. You're just missing Norte.

---

*I write from what we see every day at Hábil, where we build systems for banking, insurance, and enterprise in Mexico. If at your work or your family's business you run into a problem like this, let's talk: [habil.mx](https://www.habil.mx/en/)*
