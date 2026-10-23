# Tensors from scratch (8): tensors in the bank, the insurance company, and the store

*How to read a real-world case and determine the shape of its tensor, why a method that was not a neural network won a forecasting competition with data from a US retail chain, and which precautions are not optional.*

*Tensors from scratch · lesson B, session 8 of 8*

*Setting: the characters and the story are fictional. Technical and historical data, and real figures, cite their sources.*

**Wednesday, December 23, 9:40 a.m. Bank branch, second floor.**

The final day of the deadline.

The waiting room smells of reheated coffee and plastic pine. Someone placed a small tree next to the water dispenser. It has three ornaments, and one of them is falling off.

Doña Lupe holds her purse on her lap, clutched with both hands, as if someone were about to snatch it away. Mateo carries the laptop. Everything is inside: the Norte spreadsheet, the totals, the terminal swap in August, the charges with no merchandise, and last week's detector.

“If they say no,” says doña Lupe, “I'm closing Norte on January 2. My mind is made up.”

“They're not going to say no.”

“That's what your uncle said about his Christmas bonus, and look what happened.”

Mateo doesn't answer. In his head, he goes over what has worried him for the past nine days: the detector was right 91% of the time. And frauds were still slipping past it. What bank is going to believe a model built on a laptop, on a stationery store counter?

“Mrs. Herrera?”

Licenciado Ortega is younger than he sounded on the phone. Red tie, badge dangling, haste in his shoes. He ushers them into an office with a window overlooking the parking lot.

“I have twenty minutes,” he says. “Show me what you've got.”

Mateo opens the laptop. He shows him Norte's slump in August. The late-night charges. The amounts that don't match any merchandise. The detector.

Ortega says nothing for a long while. Then he turns his own monitor toward them.

“Do you know what this is?”

On the screen is a list. Charges. Many. Some flagged in red.

“It's the same thing you did,” says Ortega. “Except we do it with millions of purchases a day. Sit down, young man. I'm going to show you what this looks like at scale.”

Mateo looks at the wall clock. The twenty minutes have just begun to tick.

## Warm-up review

Answer without looking at session 7. The answers are in the Solutions.

1. What shape does the detector's input tensor have, and what is each axis?
2. What does loss measure, and why does it have to decrease?
3. Why might being right 91% of the time not be enough for a bank?

> **The code for this session** is in the course's public repository, in the `sesion-08` folder: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-08). Type it yourself first, because that's how you learn; the repository is for comparing when something doesn't come out the same. Each program comes with the output it should give you.

## Objectives of this session

| # | By the end you can… | Demonstrated by |
|---|---|---|
| 1 | **Read** a real case and **write** the shape of its tensor, stating what each axis is | Exercise 30 |
| 2 | **Create** a tensor with the shape of a case using `torch.zeros` and **verify** it with `assert` | Exercises 30 and 31 |
| 3 | **Explain** what an embedding is and **predict** the shape of its output | Exercise 32 |
| 4 | **Convert** a sequence into a table and **state** what is lost | Exercise 33 |

**What you need:** your PyTorch environment from session 2 and sessions 5 to 7. Today you install nothing.

---

## Why it matters

Up to this point, the data was either made up by you or came from doña Lupe's spreadsheet. Today you are going to see what Mateo saw on Ortega's screen: real-world cases from banks, insurance companies, and retail stores.

Nobody is going to give them to you with their shape already written out. They arrive with a phrase like “the model analyzes the customer's history.” Your job is to translate that into a shape: how many axes, what each one represents, and what size. If you know how to do that, no AI news sounds like magic. It sounds like a box with addresses, just like the stationery store spreadsheet.

Ortega put it to Mateo in different words: “Around here, nobody is impressed by the word *intelligence*. Around here, we ask what goes in and what comes out.”

**Two caveats.** First: almost no company publishes the exact shapes of its tensors; they publish what technique they use. So today's shapes describe **what that kind of data looks like in general**, not each company's secret recipe. Second: when a figure is reported by the company itself, without external review, it comes with an **“according to the company.”** Like your cousin who claims to be 1.80 m tall: it might be true, but nobody has measured him.

---

## Step 1 — The three questions for reading a case

“Before I show you anything,” says Ortega, “tell me one thing. When you look at one of these systems, what is the first thing you ask?”

Mateo thinks about it. In November, he would have said “how good is it.” Not now.

“What shape it has.”

Ortega smiles for the first time this morning.

Today's central new skill is translating a case into axes and shape, and it fits into three questions, in this order (you will also encounter two new conventions: the color order in PyTorch and embeddings; today it is enough to recognize their shapes):

1. **What is one thing?** A purchase, a customer, a photo. It is almost always the first axis: the batch.
2. **What does each thing contain inside?** A list of data adds an axis. A history over time adds another before the data. A photo contains colors, height, and width.
3. **What number goes in each cell?** An amount, a color, a product ID number.

With the answers, you “sketch” the shape using `torch.zeros`, which creates a tensor of zeros with whatever shape you request: the values don't matter, the box matters. Create `casos.py`:

```python
import torch

# Banca: un lote de compras, como el detector de la sesión 7
compras = torch.zeros(64, 4)
print("compras:", compras.shape)

# Banca: el historial de cada cliente
historial = torch.zeros(32, 30, 4)
print("historial:", historial.shape)

# Seguros: las fotos de un siniestro
fotos = torch.zeros(8, 3, 480, 640)
print("fotos:", fotos.shape)
print("números en las fotos:", fotos.numel())

assert compras.shape == (64, 4), "Un lote de compras es (compras, datos)"
assert historial.shape == (32, 30, 4), "El historial es (clientes, compras en el tiempo, datos)"
assert fotos.shape == (8, 3, 480, 640), "Un lote de fotos es (fotos, colores, alto, ancho)"
print("Las tres formas cuadran")
```

Output:

```
compras: torch.Size([64, 4])
historial: torch.Size([32, 30, 4])
fotos: torch.Size([8, 3, 480, 640])
números en las fotos: 7372800
Las tres formas cuadran
```

`numel()` counts how many numbers fit in the box: 8 × 3 × 480 × 640 = 7,372,800. Eight medium photos are already more than seven million numbers.

Doña Lupe leans over the laptop.

“Nothing but zeros?”

“It's the template, Aunt Lupe. Like buying the frame before the photo.”

“Ah. Well, the frame has already cost me seven million.”

---

## Step 2 — The bank: five ways to hunt fraud

Ortega opens a folder on his computer. Inside are clippings: press releases, articles, company blog posts.

“I collect these for training sessions,” he says. “Notice that each one looks at something different.”

All banks want the same thing: to spot the unusual purchase without inconveniencing the honest customer. What changes is **what they look at**, and that changes the shape. This is what Ortega opened, one by one:

- **Mastercard.** Its February 2024 press release states that Decision Intelligence helps score 143 billion transactions a year (the Spanish version of the same press release says 125 billion: that is why you check the source), and that the new version evaluates **the relationship between multiple entities** surrounding each purchase. According to the company, it improves detection by 20% on average, and up to 300% in some cases.
- **Visa.** In 2019, according to the company, it analyzed 100% of 127 billion transactions a year, in about one millisecond each.
- **BBVA.** In a 2018 note about joint work with MIT, BBVA says they generated **more than 200 new variables per transaction** and that their model “can reduce the level of false positives by 54%” (alarms that annoy an honest customer). That is BBVA global, not BBVA Mexico.
- **HSBC.** The cousin of fraud: money laundering. In a 2023 post on the Google Cloud blog, HSBC says it reviews more than 1.2 billion transactions a month searching for **rapid movements between accounts and sudden shifts in behavior**, and that it finds two to four times more suspicious activity than before, with 60% fewer alerts. According to the company.
- **Nubank.** On its engineering blog, it explains that it converts each customer's transactions into **sequences of tokens**, as if they were words, and reads them with the type of model used for text. Each transaction can produce multiple tokens (amount, date, description text). The company reports 131 million customers in total at the end of 2025; that does not mean this model reaches all of them.

Mateo notices something in every clipping: almost all of them say “according to the company.” Ortega catches him noticing.

“That's what I like most about you, young man. You read the fine print.”

Now, the three questions:

| Case | What is one thing? | What does it contain inside? | Shape (data type) |
|---|---|---|---|
| Visa: one purchase at a time | a purchase | its data | `(datos,)` |
| Mastercard, BBVA: many purchases | a purchase | its data (at BBVA, more than 200) | `(compras, datos)` |
| HSBC: behavior over time | an account | its transactions in order, each with its data | `(cuentas, movimientos, datos)` |
| Nubank: history as text | a customer | their purchases in order, in tokens | instructional model, with one token per purchase: `(clientes, compras)` of integers → `(clientes, compras, números)`; in the real case, since a purchase yields multiple tokens: `(clientes, fichas)` → `(clientes, fichas, números)` |

At BBVA, the data axis went from 4 (your detector) to more than 200: **the shape didn't change its concept, only its size.** At HSBC, looking at time adds an axis: moving from the table to the cube of session 1, with one tab per account.

Mateo stares at the BBVA row. His detector had four data points per charge: amount, time, distance, and whether the merchant was foreign. The bank's has more than two hundred. And yet it is the very same box: `(compras, datos)`.

His laptop wasn't playing a different game. It was playing the exact same game, on a small scale.

---

## Step 3 — The insurance company: the car crash in photos

“The bank isn't the only one doing this,” says Ortega, opening another clipping. “My brother-in-law works in insurance. Over there, they don't send an adjuster out to look at every dent anymore. They ask for photos.”

In March 2020, British insurer Ageas announced it was using the system from the company Tractable to assess car crashes. According to Ageas, using photos, the system identifies which parts are damaged and to what extent, compiles the complete repair estimate, and makes a decision in minutes, sometimes during the initial call.

The three questions: one thing is **a photo** (a claim includes several: the batch); each photo contains **3 colors**, and each color is a table of **height by width**; in each cell goes how much of that color that point has. Shape: `(fotos, 3, alto, ancho)`, the `fotos` tensor from step 1.

Doña Lupe adjusts her glasses.

“So if I get into a crash, I take a photo of the car and the computer tells me how much.”

“Something like that, Aunt Lupe.”

“Well, let it take a photo of my electric bill and see what it thinks of that.”

**The color order trap.** In session 1, you wrote a photo as `(480, 640, 3)`: height, width, color. In PyTorch, the convention is to put color **first**: `(3, 480, 640)`. The same numbers, a different layout. If you mix up the conventions, the `assert` warns you:

```python
fotos = torch.zeros(8, 3, 480, 640)
assert fotos.shape == (8, 480, 640, 3), "Un lote de fotos es (fotos, alto, ancho, colores)"
```

```
Traceback (most recent call last):
  ...
AssertionError: Un lote de fotos es (fotos, alto, ancho, colores)
```

Neither convention is wrong; what is wrong is assuming.

And the output? For each part, how likely it is to “repair” and how likely it is to “replace”: `(piezas, 2)`. Four axes go in; a small table comes out.

---

## Step 4 — The store: forecasting's lesson in humility

Mateo asks what he has been wanting to ask all morning:

“So, is all of this done with neural networks? Does the most advanced approach always win?”

Ortega laughs.

“I'm going to show you one your aunt will like. It's from a retail store.”

In 2020, the M5 forecasting competition took place. These figures come from the academic paper by its organizers (Makridakis, Spiliotis, and Assimakopoulos):

- Contestants had to forecast **42,840 hierarchical sales series** from **a US retail chain** (public data from the Kaggle M5 competition): 10 stores across California, Texas, and Wisconsin, daily data covering roughly 5.5 years. At the most granular level, there were 30,490 series, each representing “how much this product sold in this store, day by day”; the rest were aggregations by category, department, store, or state.
- **7,092 people across 5,507 teams from 101 countries** participated.
- **First place used LightGBM**, a decision tree method, not neural networks: a large ensemble of 220 models, not something trivial. Second place combined LightGBM with a neural network (N-BEATS). Third place did use neural networks: a combination of 43 neural networks (the team used DeepAR, which reads sequences).

The lesson in humility: **neural networks do not always win.** With tabular data, a method that is not a neural network can beat them. And even so, LightGBM receives a table `(renglones, datos)`: **the tensor is the way data is arranged, not a type of model.**

| Who | How they view the data | Shape |
|---|---|---|
| A sequence-reading network (third place) | each series is a story | `(series, días, datos)` |
| LightGBM (winner) | each day of each series is a row | `(series × días, datos)` |

“See?” says doña Lupe. “Simple things work too. I've been saying it for thirty years.”

Moving from one to the other is session 5. Mateo does it right there in the office, using doña Lupe's stationery store: 2 branches × 7 days become 14 rows. Create `tienda.py`:

```python
import torch

centro = [[1200, 800, 500], [1500, 900, 700], [1000, 700, 500], [1300, 900, 600],
          [2000, 1200, 700], [1200, 800, 400], [800, 500, 300]]
norte = [[300, 200, 100], [400, 250, 150], [350, 200, 100], [300, 150, 100],
         [500, 300, 200], [300, 200, 100], [200, 100, 100]]
papeleria = torch.tensor([centro, norte])
print("papeleria:", papeleria.shape)

tabla = papeleria.reshape(-1, 3)
print("tabla:", tabla.shape)

# Comprobación: con for, el total no cambia
total_for = 0
for sucursal in [centro, norte]:
    for dia in sucursal:
        for venta in dia:
            total_for = total_for + venta

assert tabla.shape == (14, 3), "La tabla debería ser 14 renglones (sucursal y día) × 3 productos"
assert tabla.sum().item() == total_for == 23100, "Cambiar de forma no cambia el total"
print("Total:", tabla.sum().item(), "| con for:", total_for)
```

Output:

```
papeleria: torch.Size([2, 7, 3])
tabla: torch.Size([14, 3])
Total: 23100 | con for: 23100
```

Rows 0 to 6 are Centro; rows 7 to 13 are Norte. The total, 18,500 + 4,600 = 23,100, did not change. What was lost? The fact that row 7 is “Norte, Monday” is no longer indicated by the shape; in practice, columns like “branch” and “day” are added.

Ortega looks at the two numbers: 18,500 and 4,600.

“That difference was what made us look at Norte in the first place,” he says. “Our system detected strange charges on your terminal. What it didn't see was what you just showed me: that there was no merchandise behind them.”

---

## Step 5 — The store that recommends to you: the dictionary that learns itself

Doña Lupe's cell phone chimes. A notification from an online store: “You might also like: geometry set.”

“And how does this thing know I sell geometry sets?”

Ortega and Mateo look at each other. That question has an answer.

Behind “you might also like…” there is often an **embedding**. In 2019, **Facebook (now Meta)** published an open-source recommendation model in PyTorch (DLRM) where “which product it is” passes through **embedding tables**. That same year, **Alibaba** published one that reads the sequence of what a user viewed and bought; according to the company, it runs in its Taobao store.

Think of a Python dictionary: `{"cuadernos": [1.5, -0.3, -2.2], ...}`, a list of numbers per product. The difference is that **nobody writes those numbers: the model learns them on its own**, just as it learned the weights for the detector, and they adjust with the data and the loss. In a recommendation task, products with similar purchasing patterns **can** end up with numbers close to each other, but it is not an automatic rule.

**An embedding is a dictionary that learns itself.** In PyTorch, the key is an integer, so you first number your catalog. Create `embedding.py`:

```python
import torch

torch.manual_seed(0)

# El catálogo: cada producto tiene un número
catalogo = {"cuadernos": 0, "plumas": 1, "mochilas": 2, "colores": 3, "tijeras": 4}

# La tabla de embeddings: 5 productos, 3 números por producto
diccionario = torch.nn.Embedding(5, 3)
print("tabla completa:", diccionario.weight.shape)

# Una clienta compró cuadernos, mochilas y otra vez cuadernos
compra = torch.tensor([catalogo["cuadernos"], catalogo["mochilas"], catalogo["cuadernos"]])
print("entrada:", compra.shape, compra.dtype)

salida = diccionario(compra)
print("salida:", salida.shape)
print(salida)

assert salida.shape == (3, 3), "3 productos comprados × 3 números por producto"
assert torch.equal(salida[0], salida[2]), "El mismo producto da el mismo renglón"
assert torch.equal(salida[1], diccionario.weight[2]), "Buscar un producto es sacar su renglón"
print("Forma y valores cuadran")

# Un lote: 2 clientas, 4 compras cada una
historiales = torch.tensor([[0, 1, 1, 3],
                            [2, 4, 0, 0]])
print("historiales:", historiales.shape, "->", diccionario(historiales).shape)
```

Output:

```
tabla completa: torch.Size([5, 3])
entrada: torch.Size([3]) torch.int64
salida: torch.Size([3, 3])
tensor([[ 1.5410, -0.2934, -2.1788],
        [ 0.4033,  0.8380, -0.7193],
        [ 1.5410, -0.2934, -2.1788]], grad_fn=<EmbeddingBackward0>)
Forma y valores cuadran
historiales: torch.Size([2, 4]) -> torch.Size([2, 4, 3])
```

`torch.equal(a, b)` returns `True` only if both tensors have the same shape and the same values. That is why the two `assert` statements verify that looking up the same product yields the same row.

Read it slowly:

- `Embedding(5, 3)` is a table of 5 rows (one per product) and 3 columns, initially populated with random numbers.
- Looking up a product means **extracting its row**: notebooks appears twice, identical.
- `grad_fn` is PyTorch signaling that it is keeping track so it can adjust those numbers during training. That is the “learns itself” part.
- **The shape rule:** the output has the shape of the input **plus an additional axis at the end**, of the size of the Embedding's second dimension. `(3,)` → `(3, 3)`; `(2, 4)` → `(2, 4, 3)`.

That last line is Nubank's instructional model: with one token per purchase, `(clientes, compras)` of integers becomes `(clientes, compras, números)`. In the real-world case, a purchase can yield several tokens, and that axis would be the tokens axis. Your purchase history is read like text.

“So the store doesn't actually know I sell geometry sets,” says doña Lupe. “It knows that people who buy what I buy also buy that.”

“Exactly, Aunt Lupe.”

“Well, it's a bigger gossip than my comadre.”

If you request a product that is not in the table (5, when there are only 0 through 4), you get:

```
IndexError: index out of range in self
```

---

## Step 6 — Mexico and the precautions that are not optional

**10:05 a.m.** Ortega looks at the clock. He is five minutes past the twenty and doesn't seem to mind.

“Now I'm going to tell you something I didn't tell you over the phone,” he says. “The message your aunt received on November 1 was sent by a machine. A system spotted unusual charges and suspended the terminal. All on its own.”

Doña Lupe grips her handbag tighter.

“And why did you give me until today?”

“Because a machine can flag. But a decision like that shouldn't rest entirely on a machine. Someone has to review it. Today, that someone is me.”

Mateo suddenly understands why he is sitting there. He didn't come to persuade a model. He came so that a human being would review what the model decided.

What follows is why that “someone has to review it” is not a bank courtesy.

**Banxico.** On June 10, 2026, the Bank of Mexico published the feature box “Artificial Intelligence and Financial Stability: Potential Risk Transmission Channels,” in which it warns about concentration in **few providers**, **opaque** models that are difficult to supervise, and the potential **unjustified exclusion of certain population groups**. A report by Expansión on June 18, 2026, also discusses it.

**Bias: German Credit.** This is a public dataset widely used for practicing credit scoring, from the University of California, Irvine repository: 1,000 individuals with 20 attributes each, from 1994. As an original table, `X` has 1,000 people and 20 mixed attributes (numerical and categorical); before using it in a numerical tensor, the categories must be encoded, at which point the data axis may no longer measure 20. The problem lies in the columns: one mixes **marital status and sex**, another is **age**, and another indicates whether the person is a **“foreign worker.”** Training on that directly amounts to asking the model to lend or deny credit based on sex or nationality. That is why the course did not use it in session 7.

And what if you delete those columns? That isn't enough: other columns can **give away** the same information. It is like erasing the name from an exam paper but leaving the handwriting: the teacher still knows whose it is. You have to measure how the model treats each group.

**European law.** The European Union's *AI Act*, Annex III, point 5(b), classifies as **high risk** systems intended to evaluate the creditworthiness of natural persons or establish their credit score, **with the exception of AI systems used for the purpose of detecting financial fraud**. Your detector would be exempt; a credit scoring model, like those practiced with German Credit, would fall under it. Something to think about: why does the law treat fraud differently from credit? Hint: if the fraud detector makes a mistake, a purchase gets blocked; if the credit model makes a mistake, you lose out on a house.

**Mexican law.** The new Federal Law on Protection of Personal Data Held by Private Parties was published on March 20, 2025. Its **Article 26, Section II**, allows you to **object** to having your data subjected to **automated processing** that significantly affects you and evaluates you **without human intervention**: your economic situation, health, reliability, or behavior, among other things. In plain language: if a model alone decides whether you get a loan, in those circumstances you can object. That does not mean the law mandates human review for every decision on its own, and the right has exceptions: it does not apply, for instance, when processing is necessary to fulfill a legal obligation. For system designers, “the model decides and that's it” is not enough: keeping a human in the loop is well advised.

Pay close attention to what that article says and what it does not say: it speaks of **your personal data**, of decisions about **you**. The terminal of a stationery store is another matter, and in our story the bank reviews it manually under its own internal policy. But the underlying idea is the same, and Mateo saw it in Ortega's office: **the machine flags; a person reviews; and whoever is affected has a mechanism to appeal.**

> **10:20 a.m. Licenciado Ortega's office.**
>
> Ortega types something. Prints out a sheet. Signs it.
>
> “The Norte terminal is reactivated starting today,” he says. “But first, a technician is going to go out. A real one, with a badge and a work order. To inspect the device.”
>
> Doña Lupe remains silent. For the first time in seven weeks, she doesn't have a saying ready at hand.
>
> **1:15 p.m. Norte branch.**
>
> The bank technician opens up the terminal on the counter. Fer stands to one side, arms crossed. Mateo, behind her.
>
> Inside the slot where the card is inserted, there is something that shouldn't be there: a thin shim, with a wire as fine as a hair running deep into the device.
>
> “A skimmer,” says the technician. “It copies the card every time someone pays. Customers don't notice a thing.”
>
> Fer lets out her breath slowly.
>
> “The guy in the blue shirt put it there. Back in August.”
>
> Nobody had called him. Nobody had asked his name. People paid at Norte, and their cards were used in the dead of night, far away, somewhere else. That was why the bank only saw the terminal.
>
> The technician slips the thin shim into a plastic bag. Mateo thinks about the transaction table from session 6: the ones that had no merchandise behind them. There was the missing hand.

---

👉 **If you come from Business Administration:** when a vendor pitches you their AI system, ask them the three questions from step 1; if they cannot answer them, they don't understand their product. Ask where each figure comes from. And if the system makes decisions about people (credit, insurance, hiring), ask where the human reviewer is (it is advisable even where the law does not require it in all cases).

---

## Exercises of session 8

The last ones of the course. The solutions are at the end.

**Exercise 30 — Three cases, three shapes.** Answer the three questions on paper, write down the shape, sketch it with `torch.zeros`, and verify it with `assert`. Item a) is already solved.

a) A batch of 500 purchases with 200 data points each, as in the BBVA note:

```python
bbva = torch.zeros(500, 200)
assert bbva.shape == (500, 200), "a) compras × datos"
```

b) The history of 16 customers across sequences of 100 purchases, each purchase passed through an Embedding of 64 numbers, as in Nubank's instructional model. Hint: `torch.zeros(16, 100, dtype=torch.long)` and `torch.nn.Embedding(1000, 64)`.
c) An insurance claim with 5 photos of 720 height by 1,280 width, in PyTorch order.

End with `print("Ejercicio 30 superado")`.

**Exercise 31 — The batch of photos.** Fill in the blanks: an insurance claim with 6 photos of 224 × 224, and its total count of numbers calculated in two ways that must match:

```python
siniestro = ____
assert siniestro.shape == ____, "6 fotos × 3 colores × 224 de alto × 224 de ancho"
cuenta = 1
for tamano in (6, 3, 224, 224):
    cuenta = ____
assert siniestro.numel() == cuenta, "numel debe dar lo mismo que multiplicar la forma con un for"
print("Ejercicio 31 superado:", siniestro.numel(), "números")
```

**Exercise 32 — Doña Lupe's dictionary.** Her online store has 4 products and wants 2 numbers per product. Predict the shapes **before** running:

```python
torch.manual_seed(1)
catalogo = {"cuadernos": 0, "plumas": 1, "mochilas": 2, "colores": 3}
diccionario = torch.nn.Embedding(____, ____)
historiales = torch.tensor([[0, 1, 1, 2, 3],
                            [3, 3, 0, 1, 2],
                            [2, 2, 2, 0, 1]])
salida = diccionario(historiales)
assert diccionario.weight.shape == ____, "4 productos × 2 números"
assert historiales.shape == ____, "3 clientas × 5 compras"
assert salida.shape == ____, "3 clientas × 5 compras × 2 números"
assert torch.equal(salida[0][3], diccionario.weight[catalogo["mochilas"]]), "La cuarta compra de la primera clienta son mochilas"
print("Ejercicio 32 superado")
```

**Exercise 33 — From sequence to table, on your own.** A mini M5: 100 series, 28 days, 3 data points per day (price, promotion, holiday). a) Create with `torch.zeros` what a sequence-reading network would see. b) Convert it with `reshape` and `-1` into the table that LightGBM would see. c) Write three `assert` statements: the two shapes and that they have the same quantity of numbers. d) In a comment, state what a row of the table is and what the shape no longer indicates.

---

## Common errors (and how to spot them)

| Error | How it looks | How to spot it | What to do |
|---|---|---|---|
| Mixing up the color order | You write `(fotos, alto, ancho, 3)` and PyTorch expects `(fotos, 3, alto, ancho)` | The shape `assert` blows up | Ask in what order the photos arrive |
| Requesting a product outside the table | `IndexError: index out of range in self` | Some number is ≥ the first size of the Embedding | N products go from 0 to N − 1 |
| Passing floats to an Embedding | `RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)` | It says `FloatTensor` where it expected integers | `torch.tensor([0, 2])`, without a decimal point |
| Repeating a company figure as fact | “Mastercard improves detection by 300%” | The source is a press release from the company | “According to the company,” and specify whether it is “up to” or “on average” |

---

## Self-assessment for lesson B

Check honestly. If something comes out as “not yet,” go back to that session.

| I can… | Session | Yes / Not yet |
|---|---|---|
| Change shape with `reshape`, use `-1`, and read the error when it doesn't match | 5 | |
| Predict the shape that results from operating on tensors of different shapes | 6 | |
| Distinguish `*` from `@` and know which one to use | 6 | |
| Explain loss, gradient, optimizer, and epoch | 7 | |
| Explain why accuracy alone is misleading, and what precision and recall measure | 7 | |
| Read a real-world case and write the shape of its tensor | 8 | |

## Review of lesson B

Answer without looking. For the code questions, predict first and then run them.

1. If `meses = torch.arange(12)`, what shape does `meses.reshape(4, -1)` produce? And `meses.reshape(-1, 6)`?
2. What happens with `meses.reshape(5, -1)`, and why?
3. With the stationery store `(2, 7, 3)` and `precios = torch.tensor([45.0, 12.0, 350.0])`, what shape does `papeleria * precios` have?
4. Does `torch.zeros(2, 7, 3) * torch.zeros(7)` work?
5. With `a = [[1, 2], [3, 4]]` and `b = [[10, 20], [30, 40]]` as tensors, what do `a * b` and `a @ b` evaluate to? Why is `*` a trap that does not trigger an error?
6. If `tabla` is `(14, 3)` and `precios` is `(3,)`, what shape does `tabla @ precios` have and what is each number?
7. What do `perdida.backward()` and `optimizador.step()` do?
8. What is an epoch?
9. Out of 1,000 purchases, 50 are fraud. The model triggers 40 alerts, and 30 are actual fraud. Calculate accuracy, precision, and recall. Which one concerns the bank more?
10. Write the shape of: (a) 64 purchases with 4 data points; (b) 32 customers with their last 30 purchases of 4 data points; (c) 8 photos of 480 × 640 in PyTorch order.

---

🔁 **To close the course**

**Wednesday, December 23, 7:30 p.m. Stationery store, Centro branch.**

Doña Lupe arrives with a bakery box tied with ribbon. Tres leches. She places it on the counter, between the staplers and rolls of glossy craft paper.

“I paid for this one by card,” she says. “On the Norte terminal. Let's see if this time they actually charge me the right amount.”

Fer slices the cake. She gets the biggest piece, and nobody says a word, because everyone knows why: back in September, she was the one who wrote that merchandise was missing, and nobody read her email.

Mateo eats standing up, leaning against the glass display case. Outside, someone pulls down a metal shutter. It smells of fresh paper and cinnamon.

“And what are you going to do now?” asks doña Lupe. “There's nothing left to save.”

Mateo has thought about it all week. In November, he just wanted to finish his degree and be done with it. Now he knows what he likes: having a pile of numbers reveal who is lying. He wants to study this. Data. Models. And the fine print.

“This, Aunt Lupe. What we just did.”

“Looking after stationery stores?”

“Something like that.”

That night he opens the laptop. A fresh, blank sheet. No shape yet, no axes.

He types a single line:

`# What now?`

You are in the same place as him. Over eight sessions, you went from an isolated number to reading how a bank, an insurance company, and a retail store work. You aren't left with a formula: you are left with a way of seeing.

| Session | What you learned |
|---|---|
| 1 | What a dimension is and what shape is, using Python lists |
| 2 | Your first tensor: `torch.tensor`, `shape`, and `dtype` |
| 3 | Extracting a slice of the tensor along any of its axes |
| 4 | Summing and averaging by axis, compared against a `for` loop |
| 5 | Changing shape with `reshape` without altering the data |
| 6 | Operating on full tensors, and the difference between `*` and `@` |
| 7 | Training a fraud detector, and being skeptical of accuracy |
| 8 | Reading real-world cases, embeddings, and the safeguards of automated credit |

---

## Solutions (don't peek before trying)

### Warm-up review

1. `(compras, datos de cada compra)`: in session 7, `(4000, 4)`.
2. It measures **how poorly the model predicts**, in a single number; if it goes down, the model is learning.
3. Because nearly all purchases are legitimate: a “detector” that never raises an alarm already gets around 84% accuracy. At 91%, ours was still letting 1 out of every 3 frauds slip through.

### Exercises

**30.** b) `(16, 100, 64)`: customers × purchases × embedding numbers. The 1000 is how many distinct products fit in the table; it doesn't change the shape of the output. c) `(5, 3, 720, 1280)`.

```python
historiales = torch.zeros(16, 100, dtype=torch.long)
embedding = torch.nn.Embedding(1000, 64)
salida = embedding(historiales)

assert historiales.shape == (16, 100), "clientes × compras"
assert salida.shape == (16, 100, 64), "clientes × compras × números"

fotos_siniestro = torch.zeros(5, 3, 720, 1280)
assert fotos_siniestro.shape == (5, 3, 720, 1280), "fotos × colores × alto × ancho"

print("Ejercicio 30 superado")
```

Output: `Ejercicio 30 superado`.

**31.** The blanks are `torch.zeros(6, 3, 224, 224)`, `(6, 3, 224, 224)`, and `cuenta * tamano`. Output: `Ejercicio 31 superado: 903168 números`.

**32.** `Embedding(4, 2)`, `(4, 2)`, `(3, 5)`, and `(3, 5, 2)`: an axis of size 2 is appended to the end of the input shape. Output: `Ejercicio 32 superado`.

**33.**

```python
series = torch.zeros(100, 28, 3)    # series × días × datos
tabla = series.reshape(-1, 3)       # (serie y día) × datos
assert series.shape == (100, 28, 3), "series × días × datos"
assert tabla.shape == (2800, 3), "(series y día) × datos"
assert tabla.numel() == series.numel() == 8400, "Cambiar de forma no cambia cuántos números hay"
print("Ejercicio 33 superado")
# Un renglón es un día de una serie; la forma ya no dice de cuál serie ni de cuál día.
```

### Review of lesson B

Real outputs for questions 1 to 6:

```
1) torch.Size([4, 3]) torch.Size([2, 6])
2) shape '[5, -1]' is invalid for input of size 12
3) torch.Size([2, 7, 3])
4) The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 2
5) a * b = [[10, 40], [90, 160]] | a @ b = [[70, 100], [150, 220]]
6) torch.Size([14])
```

1. `(4, 3)` and `(2, 6)`: the `-1` means “calculate it yourself.”
2. It blows up: 12 numbers cannot be split into 5 equal rows.
3. `(2, 7, 3)`: prices are broadcast (stretched) across all branches and days.
4. No: PyTorch compares shapes from right to left, and the last axis measures 3 versus 7.
5. `*` is element-wise; `@` is row by column. With these shapes both work: using the wrong one doesn't blow up, it just gives a different number. That's why it is checked against a `for` loop.
6. `(14,)`: the total in pesos for each branch on each day (quantities multiplied by prices, summed).
7. `perdida.backward()` computes the gradients of the loss with respect to each weight; `optimizador.step()` uses those gradients to adjust the weights by a small step to decrease the loss.
8. One complete pass over all the training data.
9. Legitimate purchases with no alert: 1,000 − 30 − 10 − 20 = 940. Accuracy = 970 ÷ 1,000 = 0.97; precision = 30 ÷ 40 = 0.75; recall = 30 ÷ 50 = 0.6. The 97% sounds excellent, but 4 out of 10 frauds slip through.
10. `(64, 4)`, `(32, 30, 4)`, and `(8, 3, 480, 640)`: the tensors from step 1.

---

## What comes next

One possible path forward (in English, except for the last one):

- **For intuition:** the *Neural Networks* series by **3Blue1Brown**, animated videos showing what a network does on the inside. Start with “But what is a neural network?”. They are meant for understanding, not hands-on coding.
- **To build a network starting from a single number:** *Neural Networks: Zero to Hero* by **Andrej Karpathy**. His first video, *micrograd* (2 hours 25 minutes), builds from scratch what `perdida.backward()` does; he says basic Python and a vague recollection of high school calculus are enough. It moves at an expert pace: watch it with pauses.
- **For larger networks in PyTorch:** *Learn PyTorch for Deep Learning*, by Daniel Bourke (learnpytorch.io), with exercises and solutions to compare.
- **In Spanish:** the videos by **DotCSV** for motivation and, as a milestone goal, UNAM's graduate course *Aprendizaje Profundo*, with its open material on GitHub. If you can understand its notebooks one day, this course has done its job.

One last piece of advice: whenever something doesn't work, print `.shape`. You already know how to read it.

---

*I write from what we see every day at Hábil, where we build systems for banking, insurance, and enterprise in Mexico. If at your work or your family's business you run into a problem like this, let's talk: [habil.mx](https://www.habil.mx/en/)*
