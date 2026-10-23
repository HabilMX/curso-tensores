# Tensors from scratch (7): a fraud detector that learns from data

*You give it 4,000 card purchases and the answer for each one, but you never write down the rule. In less than a second it learns a good part of it. And then you learn why getting it right 91 % of the time can hide that it misses more than 1 out of every 3 frauds.*

*Tensors from scratch · lesson B, session 7 of 8*

*Setting: the characters and the story are fictional. Technical details, historical facts, and real figures include their sources.*

**Monday, December 14, 8:18 p.m. Papelería Herrera, Centro branch.**

The metal shutter is already pulled down and still trembling slightly, as if reluctant to stay still. The light bulb in the back room buzzes. It smells of fresh paper, reheated coffee, and the cinnamon of a sweet bread that doña Lupe left forgotten next to the cash register. Outside, a bus rumbles past and the glass on the door vibrates.

Mateo has his laptop open on top of a box of poster boards. To one side, two printed sheets. On one, the sales that actually came out of Norte: notebooks, pens, an occasional backpack. On the other, the charges sent by the bank. Some arrived in the early morning hours. Others are high. Others appear far away from the branch, in places where the stationery shop hasn't sold so much as a paperclip.

“This one at three in the morning can't be a backpack,” says Mateo, circling it in red pen.

“At that hour, sleep is the only thing I sell,” answers doña Lupe. “And cheap, because nobody buys it.”

A week ago they multiplied units by prices and the numbers didn't add up: there are charges with no merchandise behind them. Fer didn't invent them; she had warned about the shortages back in September and nobody read her email. The problem wasn't in the inventory. It was in the card terminal replaced by that man in the blue t-shirt. A hidden skimmer could have copied cards at Norte. Afterwards, someone used them elsewhere.

But a suspicion isn't enough for the bank. Licenciado Ortega said it over the phone, bluntly: “Ma'am, I need to know which charges are indeed yours. Not a hunch. Which ones.”

And that's the knot. An expensive purchase isn't fraud: some people save all year for their TV screen. An early morning purchase isn't either: there are 24-hour pharmacies and people with emergencies. A distant charge could be someone on vacation. No single clue condemns anyone. What carries weight is when they come together.

Mateo tries to go over the bank's printout row by row and gives up at line forty. Working like this, he won't even finish by January.

On the wall calendar, the one handed out by the hardware store across the street, someone circled the 23rd in red. Nine days left.

“I'm not going to write a rule for every case, Auntie,” says Mateo. “I'm going to give the computer thousands of examples: how much it cost, what time, how far away, and whether the merchant was from another country. Let it find the pattern.”

“And what if the computer makes a mistake?”

Mateo stares at the red circle on the calendar.

“Then it will tell us where to keep looking. But it's not going to decide on its own.”

In the previous session you operated on entire tensors in one shot: units times price for an entire week in one go, with broadcasting and with `@`. Today those two tools get put to work on their own: you are going to watch a program **learn**.

This is the hardest session in the course, so let's take it slowly. If something isn't clear on the first pass, that is normal: read it again with the code running in front of you.

## Warm-up review

Answer without looking at session 6. The answers are in the solutions, at the end.

1. When can two shapes be “stretched” to operate together?
2. What is the difference between `*` and `@`?
3. What shape comes out of a table `(7, 3) @` a vector `(3,)`?

> **The code for this session** is in the course's public repository, in the `sesion-07` folder: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-07). Type it yourself first, because that's how you learn; the repository is for comparing when something doesn't come out the same. Each program comes with the output it should give you.

## Objectives of this session

| # | By the end you can… | Demonstrated by |
|---|---|---|
| 1 | **Name** in order the five steps of the training loop (predict, measure the loss, clear old slopes, calculate the slope, turn the knobs) and **state** which line of code does each one | Exercise 26 |
| 2 | **Predict** what happens to the loss if you change the number of epochs or the learning rate, and **verify it** by running the model | Exercises 26 and 27 |
| 3 | **Calculate** with tensors the accuracy, recall, and precision of a detector, and **explain** why accuracy alone is deceptive | Exercise 28 |
| 4 | **Connect** what the model learned to the hidden rule that generated the data | Exercise 29 |

**What you need:** the same environment from session 2 (Python 3 with PyTorch in your venv) and everything from sessions 1 to 6: creating tensors, `shape`, `dtype`, indexing, reducing with `sum` and `mean`, `reshape`, broadcasting, and `@`. Today you won't install anything.

You also need to recognize three Python constructs: `for` repeats the indented lines; `def name(X):` names a reusable recipe; and `with ...:` groups lines that run under the same rule. You don't have to master them yet: copy the full blocks and change only what each exercise points out.

---

## Why it matters

Up until today, you wrote the rule: sum this axis, multiply these two tables. Mateo has a different kind of problem. He sees thousands of charges and knows which ones ended up being fraud, but he doesn't have a short sentence that explains every case.

Each charge carries clues. Amount. Time. Distance. Foreign merchant. A single clue condemns no one. Several together may call for a review. The program's job is to find that combination in past examples, not guess a person's life story.

Think about how you learned to tell whether a street market is good. Nobody handed you a manual. You went many times. You saw where there was a line, which stall offered ripe fruit, and which one hid the bad stuff at the bottom. Over time, without being able to explain it in a single sentence, you already knew which stall to stay away from. That was learning from examples.

Or think of your grandmother making beans. She doesn't measure anything. She tastes, it lacks salt, she tosses in a pinch, tastes again. Nobody ever handed her the exact formula: she found it through sheer trial and correction. The computer is going to do exactly that, but with numbers and thousands of times per second.

So we turn it around: **instead of writing the rule, you give the program examples and let it look for a useful rule.** That is training a model. It lies behind much of what we call artificial intelligence today.

The good news is that no new magic appears. Training is **pure tensor computation**. The data is a tensor. The knobs the model learns are tensors. Prediction uses `@`. And in the end, Mateo still needs to compare, ask questions, and gather evidence so someone can review the case.

> **In Mateo's notebook:** a model doesn't prove that a charge was fraud. It only sorts thousands of clues and points out which ones are worth checking first.

---

## Step 1 — The idea in one sentence

Mateo starts small. If he understands one knob, he can understand forty-nine. If the loss goes down in the small example, he'll already know what to look for when training on the purchases.

Training a model is this, repeated many times:

1. **Predict:** the model guesses.
2. **Measure:** you compare its guess with the correct answer and get a number that says how poorly it did.
3. **Clear old slopes:** you clear the running total from the previous round (in Step 2 you'll see why that's needed).
4. **Calculate the slope:** you find out which way to move each knob so it does a little less poorly.
5. **Turn the knobs:** you move them one small step in that direction.

And back to the start: the `for` loop repeats those five steps.

Like free throws on the basketball court: you shoot, see that you missed to the left, correct a bit to the right, and shoot again. Nobody gave you the formula for the perfect shot: you found it simply by making corrections.

Before looking at the fraud detector, which has 49 knobs, we are going to run the entire loop with **just one**.

> **8:24 p.m.** “Five steps and that's it?” Mateo purses his lips. “It sounds too easy. If it were like that, anyone could do artificial intelligence.”
>
> “Anyone can make tortillas, mijo,” says doña Lupe without looking up from the coins. “Making them well is another story. The recipe is short; what takes time is practice.”
>
> Mateo laughs, but trails off into thought. She's right. The recipe for the loop fits on a napkin. The hard part will be trusting it when there isn't just one knob, but forty-nine, and he can't see with his own eyes which way they are moving. That's why he starts with one. If he sees it work on a small scale, he'll believe it on a large one.

## Step 2 — A single knob: the price of a notebook

Doña Lupe lost the price list. The only thing she remembers is that a customer took 3 notebooks and paid $60. How much does a notebook cost?

You calculate 20 in your head. But let's let PyTorch discover it “blindly,” with the 5-step loop. Create `cerro.py`:

```python
# cerro.py — una sola perilla: el precio de un cuaderno
import torch

precio = torch.tensor(0.0, requires_grad=True)   # empieza en $0: ni idea
tasa = 0.05                                      # qué tan largo es cada paso

for paso in range(1, 6):
    cobro = precio * 3                    # predicción: lo que cobraría por 3 cuadernos
    perdida = (cobro - 60) ** 2           # qué tan mal: la diferencia contra $60, al cuadrado
    perdida.backward()                    # PyTorch calcula la pendiente
    a_mano = 18 * precio.item() - 360     # la misma pendiente, sacada con lápiz
    assert abs(precio.grad.item() - a_mano) < 0.001, "PyTorch y el lápiz deberían coincidir"
    print(f"paso {paso}: precio {precio.item():6.2f}  pérdida {perdida.item():8.2f}  pendiente {precio.grad.item():8.2f}")
    with torch.no_grad():
        precio -= tasa * precio.grad      # un paso cuesta abajo
    precio.grad.zero_()                   # borrar la pendiente para el siguiente paso

assert abs(precio.item() - 20) < 0.1, "Tres cuadernos de $20 dan $60"
print("precio aprendido:", round(precio.item(), 2))
```

Run it with `python3 cerro.py`:

```
paso 1: precio   0.00  pérdida  3600.00  pendiente  -360.00
paso 2: precio  18.00  pérdida    36.00  pendiente   -36.00
paso 3: precio  19.80  pérdida     0.36  pendiente    -3.60
paso 4: precio  19.98  pérdida     0.00  pendiente    -0.36
paso 5: precio  20.00  pérdida     0.00  pendiente    -0.04
precio aprendido: 20.0
```

In five steps it reached $20 without anyone telling it the answer. Let's go line by line, because the entire session is right here in miniature.

**The knob.** `precio` is the number the model can move. It starts at 0. The `requires_grad=True` tells PyTorch: “record everything you do to this number, because later I'm going to ask you for its slope.” It's like turning on the tape recorder before the meeting.

Why a “knob”? Think of a stove. The knob doesn't know how to cook. It only raises or lowers the flame. The person who decides how much to turn it is the one tasting the food. Here it's the same: `precio` knows nothing about notebooks. It is a number that can move a little bit one way or the other. The thing that decides how much to move it is the loop.

**The prediction.** `cobro = precio * 3`: at the price it currently believes, how much would it charge for 3 notebooks? In step 1, $0.

**The loss.** `(cobro - 60) ** 2`: the difference between what it predicted and what actually happened, squared. Squaring makes the loss never negative (overshooting by $10 or undershooting by $10 costs the same) and makes large errors weigh much more heavily. In step 1 it is 3,600: it did terribly. **The loss is a grade in reverse: the lower, the better.** Zero is perfect.

It's like golf, where whoever takes the fewest strokes wins. Or like traffic tickets: the fewer, the better you behaved. The loss doesn't tell you *what* you did wrong. It only tells you *how much*. That's enough to start correcting.

**The slope.** Imagine standing on a hill **blindfolded**, wanting to reach the bottom of the valley. You can't see anything, so you feel the ground with your foot: if it slopes down in front of you, you take a step forward; if it slopes up, you step back. That is the **slope** (in textbooks, **gradient**): which way the ground slopes down and how steep it is. The hill is the loss, where you are standing is the value of the knob, and the bottom is the lowest possible loss.

Do it for real, even if only in your imagination. It's night, the power went out across the whole neighborhood, and you have to walk down the steep street by your house without a flashlight. You can't see the corner. But your feet feel the slope. Where it feels steeper, you take a longer stride. Where the ground is almost flat, you take short steps, because you have arrived or almost arrived. You never saw the map of the street. Feeling the ground, step by step, was enough. That is what the computer does: it doesn't see the full shape of the hill, it only feels the slope right where it stands.

`perdida.backward()` is feeling the ground. PyTorch checks what the recorder noted down (you multiplied by 3, subtracted 60, squared it) and computes the slope all on its own. That is called **autograd**: automatic gradients. The result is stored in `precio.grad`.

Don't believe it? That's what the `assert` with `a_mano` is for: using high school calculus, the slope is `18 × precio − 360`, and the `assert` checks at each step that PyTorch says the same thing. Optional, don't memorize it: taking the derivative of `(3 × precio − 60)²` gives `2 × (3 × precio − 60) × 3`, which simplifies to `18 × precio − 360`. You don't need to know calculus; that line only verifies that PyTorch did the exact same calculation, so you can see there is no trick.

And that `assert` is an automatic check: `assert condition, "message"` prints nothing if the condition is true. If it is false, it stops the program and displays the message. While you learn, don't delete the `assert` statements: they are your traffic light.

**The step.** `precio -= tasa * precio.grad`. The slope is −360: negative means the ground slopes down if you increase the price. Subtracting a negative is adding: 0 − 0.05 × (−360) = 18. The price jumps to $18. The `tasa` (*learning rate*) is **how long each step is**: too short and you take an eternity; too long and you overshoot to the other side of the valley. It's like parallel parking in reverse in a tight spot: if you jerk the steering wheel with big yanks, you hit the curb; if you turn it millimeter by millimeter, the driver behind you starts honking. The `with torch.no_grad():` turns off the recorder while you turn the knob, because making the adjustment is not part of the calculation.

**Clearing the slope.** PyTorch **accumulates** each new slope onto whatever it already had. If you don't clear it with `precio.grad.zero_()`, the next step would use the old one plus the new one and shoot right past the mark.

Notice: the loss goes from 3,600 to 36 to 0.36. Near the bottom, the slope becomes tiny, so the steps shrink on their own. Nobody told it to “slow down”: the hill told it.

> **8:36 p.m.** Doña Lupe peers over Mateo's shoulder and watches the price reach 20.
>
> “Twenty pesos. I could have told you that without a computer.”
>
> “I know, Auntie. But it didn't know. It started at zero, clueless, and got there on its own.”
>
> “Just like your cousin with the bicycle. He fell about twenty times, and on the twenty-first he was riding all by himself.”
>
> Mateo laughs and looks again at the five lines. It didn't find a hidden answer. It corrected a bad answer until it stopped being bad. And he understands something he didn't expect: the first attempt doesn't matter. It can start out terribly. What matters is that each round is a little less terrible. What follows is the same thing, with 4,000 purchases and 49 knobs.

## Step 3 — The data: 4,000 fictional purchases

We **made up** the purchases: there is no personal data from anyone, it works without the internet, and above all, **we know the rule that decides what counts as fraud**, so we can check whether the model discovered it. They are Mateo's trial run before touching the Norte charges again.

For each purchase we record four pieces of data: **amount** (from $0 up to almost $20,000), **hour** (from 0 to 23), **distance** to the customer's home (from 0 up to almost 500 km), and whether the merchant is **foreign** (1 = yes, 0 = no; 1 out of 10).

And this is **the hidden rule**, which only you and I know. Each purchase sets off “red flags”:

- one flag if the amount is over $15,000,
- one flag if the purchase is in the early morning hours (before 5 a.m.),
- one flag if it is more than 400 km from home,
- one flag if the merchant is foreign.

**If a purchase sets off two flags or more, it is fraud.** A single one is not enough: buying an expensive TV screen at noon is normal; buying it at 3 a.m. already smells suspicious.

The model **never** sees this rule written down: it only receives the four data points for each purchase and, while it studies, the correct answer (fraud or not). This file does two jobs. Up to this point, **we** build the lab setup with `focos`; starting from `predecir(X_ent)`, the model only receives `X_ent`. `focos` never enters `predecir`, and `y_ent` is only used to measure how poorly it predicted. In real data, `y` would come from cases that have already been investigated.

Create `detector.py`. It comes in parts. Copy blocks 1 through 7 into the same file, in the order they appear (in the code they are numbered with comments). This is the first one:

```python
# detector.py — un detector de fraude que aprende de los datos
import torch

torch.manual_seed(42)          # semilla fija: los mismos números "al azar" cada vez

# 1) DATOS INVENTADOS: 4,000 compras con tarjeta
n = 4000
monto      = torch.rand(n) * 20_000               # pesos, de 0 a 20,000
hora       = torch.randint(0, 24, (n,)).float()   # hora del día, de 0 a 23
distancia  = torch.rand(n) * 500                  # km desde la casa del cliente
extranjero = (torch.rand(n) < 0.1).float()        # 1 = comercio de otro país

# 2) LA REGLA ESCONDIDA (solo nosotros la conocemos; el modelo nunca la ve)
focos = (monto > 15_000).float() + (hora < 5).float() + (distancia > 400).float() + extranjero
y = (focos >= 2).float()       # la etiqueta: 1 = fraude, 0 = normal

# 3) EL TENSOR DE ENTRADA: compras × datos, todo entre 0 y 1
X = torch.stack([monto / 20_000, hora / 23, distancia / 500, extranjero], dim=1)
print("X:", tuple(X.shape), "| y:", tuple(y.shape), "| % fraude:", round(y.mean().item() * 100, 1))
```

New functions, all straightforward:

- `torch.rand(n)` gives `n` random numbers between 0 and 1; times 20,000, amounts from 0 to 20,000 (broadcasting, as in session 6). `torch.randint(0, 24, (n,))` gives random integers from 0 to 23.
- `torch.manual_seed(42)` fixes the “randomness.” The computer doesn't roll dice: it follows a recipe, and with the same seed it gives the same numbers every time you run it on the same computer.
- `(monto > 15_000)` gives a `True` or `False` per purchase; `.float()` turns it into 1.0 or 0.0. Adding four of those counts the flags for all 4,000 purchases at once.
- `torch.stack([...], dim=1)` stacks the four lists as **columns** of a table.

Dividing each column (`monto / 20_000`, `hora / 23`…) puts everything between 0 and 1, so that the amount (thousands) doesn't crush the hour (tens) simply for being a larger number.

The output for this part:

```
X: (4000, 4) | y: (4000,) | % fraude: 15.8
```

**`X` has shape `(4000, 4)`: purchases × data for each purchase.** A sheet of 4,000 rows and 4 columns. The first axis is called the **batch** (how many examples) and the second, **features** (what we know about each one). You will see that shape everywhere.

**`y` has shape `(4000,)`:** one answer per purchase. These are called **labels**: the column of correct answers on the exam. 15.8 % of the purchases are fraud.

To see it up close, add `print(X[:3])` and `print(y[:3])` right after that `print`. They are a temporary inspection: run them once here and delete them before comparing your output with the “Complete output” in Step 8:

```
tensor([[0.8823, 0.1304, 0.4309, 0.0000],
        [0.9150, 0.6087, 0.3986, 0.0000],
        [0.3829, 0.4348, 0.1907, 0.0000]])
tensor([1., 0., 0.])
```

Purchase 0 is for $17,645 (0.8823 × 20,000) at 3 in the morning (0.1304 × 23): two flags, fraud. Purchase 1 is more expensive, $18,300, but at 2 in the afternoon: a single flag, normal. That nuance is what the model has to discover.

> **8:52 p.m.** “That's it,” says Mateo, very confident. “Everything over fifteen thousand is fraud. I'll flag it and be done with it.”
>
> Doña Lupe stops counting coins.
>
> “And what about the lady who came in August to pay for the school supplies of her four kids all at once? Is she a thief too?”
>
> Mateo opens his mouth and closes it again. He looks at purchase 1 on the screen: $18,300, at two in the afternoon, normal. His rule would have accused her. **The clue does not live in a single column.** Neither the amount alone, nor the hour alone. What's unusual is when they come together.
>
> “One swallow does not make a summer.” Doña Lupe delivers the final word and goes back to her coins.
>
> Mateo crosses out his rule. That is why he is not going to write it himself.

## Step 4 — Studying and taking the exam

If you grade a student with the same questions they studied, you don't know whether they learned or memorized. It's the same with models:

```python
# 4) Estudia con 3,200 compras; el examen es con las otras 800
X_ent, y_ent = X[:3200], y[:3200]
X_exa, y_exa = X[3200:], y[3200:]
```

This is slicing along the first axis, as in session 3. The last 800 purchases are kept under lock and key: the model never sees them while learning.

Mateo calls the first 3,200 “study” and the other 800 “exam.” That is not an embellishment. If he showed it the answers to the exam, the final number would sound pretty and say nothing.

It's the difference between memorizing the study guide and understanding the subject. If the exam has the exact questions from the guide, you get an A even if you know nothing. If the teacher changes the numbers, it becomes clear who understood. The 800 set-aside purchases are the exam with changed numbers.

An honest detail: in exercises 26 and 27 you are going to compare epochs and learning rates by looking at these same 800. As soon as you make choices based on them, they are no longer a clean exam, but a practice set for decision-making. In a real-world project, a third group is set aside—the final exam—which nobody looks at until the very last day.

> **9:05 p.m.** Doña Lupe's cell phone vibrates on the counter. It's Fer, from Norte: “I've closed up. Everything was in cash today. Three regular customers left without buying because they didn't have cash on them. Sending you a photo of today's receipts.”
>
> Doña Lupe reads the message twice and shows it to Mateo.
>
> “And just a month ago I didn't even want to say her name,” she says softly. “You see ghosts around every corner when there aren't any.”
>
> Mateo doesn't reply. He saves the photo of the receipts into a new folder. If the model says anything about the charges at Norte, he is going to need something to compare it with. Real data, set aside, that the model has never seen. Just like the exam.

## Step 5 — The knobs and prediction with `@`

Now the knobs. In `cerro.py` there was one. Here there are four knob tensors:

```python
# 5) LAS PERILLAS: empiezan al azar y el modelo las va girando
W1 = (torch.randn(4, 8) * 0.5).requires_grad_()
b1 = torch.zeros(8, requires_grad=True)
W2 = (torch.randn(8) * 0.5).requires_grad_()
b2 = torch.zeros((), requires_grad=True)   # () = un solo número: la perilla final

def predecir(X):
    oculta = torch.relu(X @ W1 + b1)   # (compras, 4) @ (4, 8) -> (compras, 8)
    return oculta @ W2 + b2            # (compras, 8) @ (8,)   -> (compras,)
```

In textbooks these knobs are called **weights** (the `W`) and **biases** (the `b`). `W1` has 4 × 8 = 32, `b1` has 8, `W2` has 8, and `b2` has 1: **49 knobs**. They start at random (`torch.randn`) or at zero, just as the price started at $0. Two syntax details: the `()` in `torch.zeros((), ...)` means “a single number,” the final knob; and the trailing underscore in `requires_grad_()` turns on the “recorder” in place on that existing tensor.

Have you seen a DJ's soundboard at a street party? A board full of knobs: bass, treble, volume for each microphone, echo. No single knob makes it sound good. What sounds good is the combination. And the sound tech doesn't calculate it: he turns one, listens, turns another, listens again. **Weights** are just that. Each one says how much a piece of data matters in a mix. Each weight in `W1` adjusts how much a feature counts toward one of the intermediate suspicions: large means “turn this up”; near zero means “pay no attention to this”; negative means “subtract this.” On its own, it doesn't measure the global importance of the feature or whether it signals fraud or normality: that also depends on `W2`, on the biases, and on `relu`. The difference is that our sound tech has 49 knobs and moves them all at once, guided by the loss instead of his ears.

Read `predecir` looking only at the shapes, as in session 6:

- `X @ W1`: `(compras, 4) @ (4, 8)` gives `(compras, 8)`. Each purchase goes from 4 features to **8 intermediate “suspicions,”** each one a different mix of amount, hour, distance, and foreign merchant. One might end up focusing on “expensive and early morning”; another, on “far away and foreign.” Nobody tells it which ones: it discovers them by turning the knobs.
- `+ b1`: a vector `(8,)` added to each row. Broadcasting.
- `torch.relu(...)`: **if a number comes out negative, it turns it to zero; if it comes out positive, it leaves it alone.** “If this suspicion doesn't apply, don't even mention it.” Without `relu`, the two layers would collapse into a single mix, a single computation of multiplying and adding, which is not enough to mimic the rule's cutoffs (over $15,000, before 5 a.m., over 400 km) or how they combine. (If the flags were already pre-counted as inputs, “two or more” would indeed be a single mix.)
- `oculta @ W2 + b2`: `(compras, 8) @ (8,)` gives `(compras,)`, the calculation from the third review question. **One number per purchase**: the **score**.

**Score greater than zero: the model says “fraud”; less than or equal to zero: “normal.”** The farther from zero, the more confident the model is—which is no guarantee. In textbooks this score is called a **logit** (hence the name of the loss in Step 6). A score of 0 corresponds to “fifty-fifty”: 50 % probability of fraud according to the model. That is why the cutoff is at zero, and the code breaks an exact tie as “normal” (`> 0`, not `>= 0`). This is called **prediction** or the *forward pass*: data goes in one side and the score comes out the other, like dough in a tortilla press.

> **9:20 p.m.** Mateo stares at `W1` as if it owed him money.
>
> “And which of the eight is the one for ‘expensive and early morning’?” he asks aloud, to no one in particular.
>
> He prints `W1` with `print(W1)`. Thirty-two random numbers come out, none with a name. He gets a little frustrated. He wanted to see the rule written out, nice and neat, in a box. It's not going to happen. Suspicions don't come with labels: they form on their own as the knobs turn.
>
> “Have you picked a fight with the machine already?” asks doña Lupe.
>
> “Not yet. It just won't tell me what it's thinking.”
>
> “Just like your uncle. Thirty years of marriage and I still don't know.”
>
> Mateo bursts out laughing and decides not to fight with what he cannot see. Keeping track of the shapes is enough: four features enter, eight signals exit, and in the end a score remains for each charge.

## Step 6 — The loss for a “yes or no”

In `cerro.py`, the loss was the squared difference. For “yes or no” questions, a different one is used, already built into PyTorch:

```python
perdida_fn = torch.nn.BCEWithLogitsLoss()
optimizador = torch.optim.SGD([W1, b1, W2, b2], lr=1.0)
```

Don't let the name scare you. `BCEWithLogitsLoss` plays the role of our squared difference, but for fraud/no fraud: it takes scores and labels and returns **a single number** indicating how poorly it did across all purchases combined. Its rule is common sense: if it was fraud and the score was strongly positive, loss is near zero; if it was fraud and the score was strongly negative, loss is massive (being very confident and being wrong is the worst); if the score is near zero, loss is medium.

Think of the friend who swears to you that the match is already won and in the end they lose 3-0. It's not the same as the friend who said “who knows.” Both were wrong, but you make the first one pay much more heavily on the bet. This loss charges the same way: being doubtful costs very little; being dead certain and wrong costs immensely. That way the model learns not to brag about confidence it doesn't have.

The second line is the **optimizer**, the thing that turns the knobs. `SGD` does the same thing you did by hand in `cerro.py`, `perilla -= tasa * pendiente`, but with all 49 at once. `lr=1.0` is the learning rate.

> **9:41 p.m.** The machine now has the data, the knobs, and a way to score its mistakes. All that's left is to set it loose.
>
> Mateo has his finger poised over the key and hesitates to press it. What if it doesn't learn anything? What if the loss stays right where it is and he wastes the night, and the next one, and the 23rd arrives with nothing to show for it?
>
> Doña Lupe switches off the coffee maker, which is already smelling burnt.
>
> “Nothing ventured, nothing gained, mijo. And if you don't hit the button, you'll never find out.”
>
> The wall clock shows almost ten. Nine days left, and each one weighs. Mateo takes a breath. Each pass through the loop has to bring him closer to a clue, not just a pretty number.

## Step 7 — The complete loop and epochs

Now, the heart of the session:

```python
# 6) EL CICLO DE ENTRENAMIENTO: una época = una vuelta a las 3,200 compras
for epoca in range(1, 301):
    puntaje = predecir(X_ent)               # 1. predecir
    perdida = perdida_fn(puntaje, y_ent)    # 2. medir qué tan mal le fue
    optimizador.zero_grad()                 # 3. borrar las pendientes de la vuelta anterior
    perdida.backward()                      # 4. calcular la pendiente de cada perilla
    optimizador.step()                      # 5. girar cada perilla un pasito cuesta abajo
    if epoca in (1, 10, 50, 100, 200, 300):
        print(f"época {epoca:3d}   pérdida {perdida.item():.4f}")
```

Compare it with `cerro.py`. It's the same loop, row by row:

| Step | In `cerro.py` | In `detector.py` |
|---|---|---|
| Predict | `cobro = precio * 3` | `puntaje = predecir(X_ent)` |
| Measure | `(cobro - 60) ** 2` | `perdida_fn(puntaje, y_ent)` |
| Clear old slopes | `precio.grad.zero_()` | `optimizador.zero_grad()` |
| Calculate slopes | `perdida.backward()` | `perdida.backward()` |
| Turn knobs | `precio -= tasa * precio.grad` | `optimizador.step()` |

Each pass of the `for` loop shows the model **all** 3,200 study purchases. A complete pass over all the data is called an **epoch**: going through the entire syllabus once. Here we give it 300.

Why so many? Because on each pass the knobs move only a tiny step. It's like learning a song on the guitar: the first time you play it all the way through, you mess up every chord change. The tenth time, only on the difficult one. The hundredth time, you don't even think about it. Each epoch is playing the entire song one more time. Nobody learns it on the first try, not you and not the computer.

Run what you have so far and look at the loss:

```
época   1   pérdida 0.6819
época  10   pérdida 0.4098
época  50   pérdida 0.2528
época 100   pérdida 0.2070
época 200   pérdida 0.1984
época 300   pérdida 0.1910
```

**The training loss goes down:** the model is reducing errors on the 3,200 purchases it has already seen. It's a good sign, but it still doesn't prove that it works on new purchases: that is tested by the exam in Step 8. It drops quickly at first and then more and more slowly, just like with the notebook: near the bottom, the ground hardly slopes at all. On my machine, the 300 epochs took less than a tenth of a second.

> **Why 300 and not more?** We measured it. Every computer rounds decimal points in a slightly different order; the difference is microscopic, on the order of the seventh decimal place. With large steps like `lr=1.0`, after about 300 passes those differences start to grow: with 400, two computers running the same versions ended up with different counts of caught frauds. With 300, the result didn't budge even when we intentionally introduced differences a thousand times larger than those between one computer and another. That is why the course uses 300: so that, if your number comes out different, you know the error is in your code and not in your machine.

**9:47 p.m.** Mateo presses the key.

The terminal spits out the first line before he can even blink: `época 1 pérdida 0.6819`. A knot forms in his stomach. That number tells him nothing yet. It's the model guessing blindly, with random knobs, like a dog hearing its name in another language.

Then the second appears: `0.4098`.

Mateo leans toward the screen.

`0.2528`.

“Auntie. Auntie, come here.”

Doña Lupe walks over, wiping her hands on her apron.

“What happened? Did it break?”

“No. Look. It's going down.”

`0.2070`. `0.1984`. `0.1910`.

Six numbers, one below the other, each one smaller. It's already over. It lasted less time than it takes to boil water for coffee. Mateo stares at the column like someone staring at the scoreboard at the end of a match they didn't expect to win.

Nobody wrote the rule “two flags or more” for the machine. Nobody told it about early mornings or foreign merchants. They did give it 3,200 purchases, their four features, and the fraud/not fraud answer for each one, along with a way to measure how poorly it did. With those examples, round by round, it kept moving its 49 knobs until it made fewer mistakes. Mateo feels something strange, between pride and a shiver down his spine. He wrote a few lines. The rest was done by the machine.

“And is that good?” asks doña Lupe.

“It's learning, Auntie. I'm watching it learn.”

Doña Lupe looks at the screen, then at her nephew, whose eyes are gleaming just like when he was a boy and finally got his spinning top to spin.

“Well, let's hope it learns fast. Because the 23rd won't wait.”

Mateo nods, but he barely hears her anymore. **The loss is a signal, not a verdict.** It tells him that the model makes fewer mistakes during study. The important part is still missing: seeing what it does when the answers are taken away.

## Step 8 — The exam: 91 %

The important part is still missing: does it do well with purchases it has **never seen**? Add the final part:

```python
# 7) EL EXAMEN: 800 compras que el modelo nunca vio
with torch.no_grad():
    dice_fraude = (predecir(X_exa) > 0).float()   # puntaje positivo = "esto es fraude"

exactitud = (dice_fraude == y_exa).float().mean().item()
atrapados = (dice_fraude * y_exa).sum().item()          # dijo fraude y sí era
falsas    = (dice_fraude * (1 - y_exa)).sum().item()    # dijo fraude y no era
escapados = ((1 - dice_fraude) * y_exa).sum().item()    # dijo normal y era fraude

print("exactitud:", round(exactitud, 3))
print("fraudes en el examen:", int(y_exa.sum().item()))
print("atrapados:", int(atrapados), "| se escaparon:", int(escapados), "| alarmas falsas:", int(falsas))
print("recall:", round(atrapados / (atrapados + escapados), 3),
      "| precisión:", round(atrapados / (atrapados + falsas), 3))
```

The last two figures it prints (recall and precision) don't have a meaning for you yet: ignore them until Step 9, where they are explained and you read them again. For now, look at accuracy and the three counts.

The trick is element-wise `*`, the one from session 6: `dice_fraude * y_exa` only gives 1 where **both** are 1 (it said fraud **and** it was). `1 - y_exa` flips ones and zeros, so `dice_fraude * (1 - y_exa)` gives 1 where it said fraud and it **wasn't**. You sum them up and you have the count for each case.

The full script has 61 lines, 41 without comments or blank lines. Run it with `python3 detector.py`. Complete output, identical both times I ran it:

```
X: (4000, 4) | y: (4000,) | % fraude: 15.8
época   1   pérdida 0.6819
época  10   pérdida 0.4098
época  50   pérdida 0.2528
época 100   pérdida 0.2070
época 200   pérdida 0.1984
época 300   pérdida 0.1910
exactitud: 0.908
fraudes en el examen: 125
atrapados: 81 | se escaparon: 44 | alarmas falsas: 30
recall: 0.648 | precisión: 0.73
```

**Accuracy: 0.908.** On purchases it had never seen, it got nearly 91 out of 100 right, without anyone telling it the rule of the flags.

About the seed: with the same version of PyTorch and a computer like mine, you should normally get exactly this. But PyTorch does not guarantee identical numbers across versions or across different computers ([PyTorch documentation on reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)): the last decimal places may shift slightly. With 300 epochs we measured that they do not move (see “Why 300 and not more?” in Step 7); in exercises with many more passes, they might shift. What **must** always happen: the loss drops and the accuracy stays around 91 %. If your result is very different, check the troubleshooting table.

**10:12 p.m.** Mateo sees the 91 and raises his arms as if he had scored a goal.

“Ninety-one, Auntie! It got ninety-one out of every hundred right!”

Doña Lupe doesn't raise her arms. She adjusts her glasses and steps closer to the screen with the calm of someone who has tallied a thousand cash register closes.

“So how good is it?”

“Well… very good. It's a nine.”

“A nine on what? What does that line below say?”

Mateo reads it aloud, more slowly: “se escaparon: 44”.

The excitement deflates out of him like a punctured balloon. Forty-four. Out of 125 frauds on the exam, forty-four slipped through as if nothing happened. The model saw them, reviewed them, and said “normal.”

They are fictional purchases, he already knows. But he can't help putting a face to them. He imagines one of those forty-four. A woman who bought notebooks at Norte in September, paid by card, said thank you, and left. And one morning, standing in line for tortillas, she opens her banking app and sees a seventeen-thousand-peso charge at a store she has never set foot in. Her year-end bonus, or what was going to be her year-end bonus. Then he imagines the young man who saved up for a cell phone and had his account emptied at three in the morning on a Tuesday. And another. And another. Forty-four times.

“It's not a nine, Auntie,” he says, his voice coming out quiet. “For homework, a nine is a good grade. For 125 frauds, forty-four is an open door.”

Doña Lupe rests a hand on his shoulder.

“There you go. Not all that glitters is gold, mijo. Now find out what it's made of.”

> **Warning box:** accuracy counts all correct predictions together. It doesn't tell you how many frauds it let through. That is why a bank cannot rely on that number alone.

Is 91 % good? It sounds like a 9 out of 10. Now comes the most important part of the session.

## Step 9 — The trap: the lazy detector scores 84

Imagine a “detector” that tells every purchase “not fraud,” without even looking at them.

On the exam there are 800 purchases and 125 are fraud. The other 675 are normal, and the lazy one gets every single one of them right: 675 out of 800 is **84.4 %**. You will verify this with a tensor in exercise 28.

A one-line program that is completely useless scores 84; ours scores 91. The difference is much smaller than it seemed. **Accuracy is deceptive when what matters is rare**, and in a bank fraud is usually rare: almost all purchases come from honest people.

Think about a metal detector at the entrance to a concert. Almost nobody carries anything dangerous. If the guard lets everyone through without checking, he “gets it right” with almost the entire audience. And yet he is the worst guard in the world, because the one case that mattered slipped right past him. Getting the common stuff right is easy. The hard part—and what is valuable—is catching the rare case.

> **10:31 p.m.** Mateo writes the lazy detector down on a piece of paper, without a computer: “say no to everything.” He stares at it for a while. It's one line. It's ridiculous. And it scores 84.
>
> “So my 49-knob model beats doing nothing by less than seven points,” he murmurs.
>
> “And is that little?” asks doña Lupe.
>
> “It's like bragging that you run faster than someone who is sitting down.”
>
> Doña Lupe bursts into laughter that rattles the glasses on the shelf.
>
> “Well, with that you couldn't outrun a garbage truck.”
>
> Mateo doesn't take offense. For the first time tonight, he feels he truly understands. The number that matters isn't the biggest one. It's the one that counts what slipped away.

That's why banks look at two other numbers. First, the four possible outcomes of the exam:

| | Actually fraud | Actually normal |
|---|---|---|
| **Model said “fraud”** | 81 caught | 30 false alarms |
| **Model said “normal”** | 44 slipped through | 645 correctly let through |

(The four cells add up to 800. The correct predictions are 81 + 645 = 726, and 726 / 800 = 0.9075, which rounded gives 0.908, the accuracy above.)

**Recall: out of all the frauds, how many did it catch?** There were 125 frauds and it caught 81: 81 / 125 = **0.648**. It missed 44 out of 125: more than 1 out of every 3 frauds. Each of those 44 is someone who checks their bank statement tomorrow and finds a $17,000 charge at a store they have never visited. They have to dispute the charge, the bank investigates the case and often ends up refunding the money; nobody refunds the scare. The lazy detector has **zero** recall: it doesn't catch a single one.

**Precision: out of all the alarms it raised, how many were real?** It raised 81 + 30 = 111 alarms, and 81 were fraud: 81 / 111 = **0.73**. Roughly 1 out of every 4 alarms was a false alarm. Each of those 30 is an honest person who, depending on how the bank operates, had their card declined at the supermarket checkout with a line of people behind them, or was asked to verify their identity before being allowed through. If that happens often, the customer switches banks.

Both things matter. If the bank wants to catch **all** frauds, it flags everything as suspicious: perfect recall and all its customers angry. If it doesn't want to inconvenience **anyone**, it flags nothing: zero false alarms and every fraud slips through, just like with the lazy detector. The real job lies in the middle, and it depends on how much each mistake costs. That is why, when evaluating a fraud detection system, how many frauds it catches and how many false alarms it raises matter more than “accuracy” alone.

**Today's big takeaway:** getting a high percentage right is not enough. Before trusting a model, ask how rare what it looks for is, and how many of those rare cases it catches.

> **11:05 p.m.** Mateo draws the four-cell table on the back of an electric bill. Top left, 81: the ones it caught. Bottom left, 44: the ones that got away. Top right, 30: honest people whose cards got declined at the register. He stares at that box. If the bank did that to the customers at Norte, the stationery shop would lose the few who still come back.
>
> “It's not one number, Auntie. It's two. And they pull against each other, like a blanket that's too small: if you cover your feet, your shoulders are bare.”
>
> Doña Lupe nods slowly. That one she understands completely. Mateo tucks the receipt into his jacket pocket. Tomorrow he'll need it.

---

👉 **If you come from Business Administration:** you don't need to train the model to ask the right question in a meeting. When someone presents “our model is 91 % accurate,” ask: what would a model that does nothing score? How many of the cases we care about does it catch (recall)? How many of its alarms are accurate (precision)? Then translate each error into pesos: how much does a fraud that slips through cost, and how much does an upset customer cost? That four-cell table is a business tool, and it works just as well for deciding who to extend credit to or which insurance claim to investigate.

---

## Exercises for session 7

The solutions are at the end. Before each change, **write down on paper what you think will happen**. Predicting before running is half of learning. Fill in this table on paper (exercises 26 and 27 ask for these four runs):

| Configuration | My prediction | Final loss | Accuracy | Caught | False alarms | What did I learn? |
|---|---|---:|---:|---:|---:|---|
| 300 epochs, `lr=1.0` | | | | | | |
| 2000 epochs, `lr=1.0` | | | | | | |
| 300 epochs, `lr=0.1` | | | | | | |
| 300 epochs, `lr=10` | | | | | | |

**Exercise 26 — More passes, fewer passes.** In `detector.py`, the number of epochs is in `range(1, 301)`. I already ran it with 100 epochs (`range(1, 101)`): the loss ends at 0.2070, accuracy at 0.902, it catches 77 frauds, and raises 30 false alarms. Now it's your turn: run it with **2000** epochs. Add `1000, 2000` to the list of epochs that get printed so you can see the loss.

a) Before running: will the final loss be greater or less than 0.1910?
b) What happened to accuracy, caught frauds, and false alarms?
c) Point out in your code which line corresponds to each of the five steps of the loop (predict, measure, clear slopes, calculate slopes, turn knobs).

**Exercise 27 — Step length.** Go back to 300 epochs. Now change the learning rate, `lr=1.0`, first to `lr=0.1` (ant steps) and then to `lr=10` (giant strides). Before running each one, predict: does the loss decrease more or less? Why?

Warning: with one of the two, the program crashes at the end with a `ZeroDivisionError`. It's not your mistake. Read the lines above the error and explain why it happened. Hint: look at how many alarms it raised in total.

**Exercise 28 — Calculate the lazy detector.** Add this to the end of `detector.py` and fill in the blanks. The lazy detector is a tensor of 800 zeros: “not fraud” for all purchases on the exam.

```python
# Ejercicio 28: el detector flojo que siempre dice "no es fraude"
siempre_no = torch.____(800)
assert siempre_no.shape == y_exa.shape, "Tiene que haber una respuesta por cada compra del examen"
exactitud_flojo = (siempre_no == ____).float().mean().item()
atrapados_flojo = (siempre_no * ____).sum().item()
print("exactitud del flojo:", round(exactitud_flojo, 3))
print("fraudes que atrapa:", int(atrapados_flojo), "de", int(y_exa.sum().item()))

# la misma cuenta con un for, como antes de los tensores
aciertos = 0
for i in range(800):
    if y_exa[i].item() == ____:
        aciertos = aciertos + 1
assert round(aciertos / 800, 3) == round(exactitud_flojo, 3), "El for y el tensor deberían coincidir"
print("El for dice lo mismo:", aciertos, "aciertos de 800")
```

**Part two.** Now calculate the recall and precision of the **trained detector**, without looking at the last two lines of Step 8. You already have `atrapados`, `falsas`, and `escapados` in your file:

```python
recall_modelo = ______________________________
precision_modelo = ___________________________

print("recall del modelo:", round(recall_modelo, 3))
print("precisión del modelo:", round(precision_modelo, 3))
assert round(recall_modelo, 3) == 0.648 and round(precision_modelo, 3) == 0.73, "Revisa cuál de las cuatro casillas falta en tu fracción"
```

With the baseline run (300 epochs, `lr=1.0`) they should come out to `0.648` and `0.73`. If you changed epochs or learning rate, remove the `assert`.

Then answer: what is the recall of the lazy detector? Why can't you calculate its precision?

**Exercise 29 — What did the model learn?** The rule uses all four features. Did the model figure that out? Ask it by **shuffling a column**: if you shuffle the hour among the 800 exam purchases, each purchase ends up with the hour of another one at random. The column is still there, but it no longer carries any signal. If accuracy drops sharply in this run, that is evidence that the model was using that column to make predictions. It does not prove by itself that it recovered the exact rule or that this column causes fraud.

You are going to repeat this for each of the 4 columns: copy the exam, shuffle **only that column**, predict again, and show the accuracy and caught frauds. Two new tools:

- `X_exa.clone()` creates a copy, so you don't ruin the original exam.
- `torch.randperm(800)` gives the numbers from 0 to 799 shuffled. `X_exa[torch.randperm(800), col]` is column `col` shuffled.

Remember to predict inside `with torch.no_grad():`, just like in the exam. A solved example, with the `monto` column (column 0):

```python
X_original = X_exa.clone()
X_revuelto = X_exa.clone()
X_revuelto[:, 0] = X_exa[torch.randperm(800), 0]

with torch.no_grad():
    dice = (predecir(X_revuelto) > 0).float()

print("monto: exactitud", round((dice == y_exa).float().mean().item(), 3))
```

Now complete columns 1, 2, and 3 (hour, distance, and foreign merchant) yourself using a `for` loop, and also display the caught frauds:

```python
for col in range(1, 4):
    X_revuelto = X_exa.________()
    X_revuelto[:, col] = X_exa[torch.________(800), col]
    with torch.no_grad():
        dice = (________(X_revuelto) > 0).float()
    # imprime la exactitud y los fraudes atrapados de esta columna
    assert torch.equal(X_exa, X_original), "No debes alterar el examen original"
```

Then answer: which column hurt the model the most to lose? Did it not care about any of them? Does it square with the hidden rule of the flags?

---

## Common errors (and how to spot them)

| Error | What it looks like | How to spot it | What to do |
|---|---|---|---|
| Forgetting `requires_grad=True` on a knob | In `cerro.py`, where it is the only knob: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` when calling `backward()`. In `detector.py`, if you forget it only on `W1`, **it doesn't crash**: `W1.grad` stays `None`, `W1` never moves, the loss gets stuck at 0.3366 and it catches only 13 out of 125 | In `cerro.py` it crashes on the `backward` line; in `detector.py`, the loss drops much less than usual and `print(W1.grad)` shows `None` | Turn on the “recorder” when creating each knob: `requires_grad=True` |
| Forgetting to clear slopes (`zero_grad`) | In `cerro.py`, the `assert` saves you: it crashes on step 2 with `AssertionError: PyTorch y el lápiz deberían coincidir`. Without the `assert` statements, it doesn't crash, and that's what's dangerous: the price jumps to 37.80, then 41.58, the loss climbs up to 4,191.27 (at step 4) and, by the end, the learned price is 4.95 instead of 20 | The loss goes up and down instead of down | PyTorch adds new slopes to the old ones; clear them on every pass |
| Labels with a different shape than scores | `ValueError: Target size (torch.Size([3200, 1])) must be the same as input size (torch.Size([3200]))` | Print `y.shape` and the score's `shape`: they must be identical | Leave `y` with shape `(4000,)`, without `reshape(4000, 1)` |
| Learning rate too large | Training doesn't crash. With `lr=10` the loss gets stuck at 0.4365 and the model stops catching fraud; at the end precision crashes with `ZeroDivisionError`, because it didn't raise any alarms (exercise 27) | The loss stops going down or bounces around | Lower the learning rate; try factors of 10 (1, 0.1, 0.01…) |
| Calling `backward()` twice on the same loss | `RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).` | Crashes on the second `backward` | One `backward` per pass; the next pass predicts and measures again |
| Trusting accuracy alone | “91 %!” | You calculate the lazy detector and it scores 84 % | Look at recall and precision too |

---

## Review for next session

Session 8 starts with these three questions. Answer them without looking at the lesson:

1. What shape does the detector's input tensor have and what is each axis?
2. What does the loss measure and why does it have to decrease?
3. Why can getting it right 91 % of the time not be enough for a bank?

🔁 **Next week:** tensors in the bank, the insurance company, and the store. Real cases, with sources, and the challenge of writing the shape of each one's tensor yourself. Spoiler: today's detector has the exact same shape as the real ones.

---

## Solutions (don't look before trying)

**Warm-up review.**

1. Shapes are compared **from right to left**, axis by axis. Along each axis, the two dimensions must be **equal, or one of them must be 1** (the 1 is the one that gets “stretched”). If any axis doesn't meet this condition, PyTorch raises an error.
2. `*` multiplies **element by element**: cell by cell, and the shapes must match or stretch. `@` is the **matrix product**: row by column, multiplying and summing. The tricky part is that sometimes both run without error and give different things.
3. `(7,)`. The 3 in the middle gets “consumed” (multiplied and summed) leaving one number for each of the 7 rows. You used it today: `(compras, 8) @ (8,)` gave `(compras,)`.

**Exercise 26.** With 2000 epochs (the first lines are identical to those of 300; pasted starting from 300):

```
época 300   pérdida 0.1910
época 1000   pérdida 0.1641
época 2000   pérdida 0.1142
exactitud: 0.93
fraudes en el examen: 125
atrapados: 83 | se escaparon: 42 | alarmas falsas: 14
recall: 0.664 | precisión: 0.856
```

a) Lower: 0.1142. More passes, more time to descend the hill. The early epochs produce the same numbers as with 300 thanks to the fixed seed. From 1000 onward, on your computer the last digit of the loss and some count may come out slightly different: these are the rounding differences mentioned in the “Why 300 and not more?” box, which have time to grow over so many passes.

b) Accuracy rises to 93 %, it catches 83 frauds instead of 81, and false alarms drop from 30 to 14. Note: more epochs doesn't always help. With real data, too many passes can cause the model to **memorize** the study examples and perform worse on the exam. That is why the exam uses set-aside data.

c) `puntaje = predecir(X_ent)` predicts; `perdida = perdida_fn(puntaje, y_ent)` measures; `optimizador.zero_grad()` clears slopes; `perdida.backward()` calculates slopes; `optimizador.step()` turns knobs.

**Exercise 27.** With `lr=0.1` (omitting some middle epochs):

```
época   1   pérdida 0.6819
época  50   pérdida 0.4238
época 300   pérdida 0.3194
exactitud: 0.859
fraudes en el examen: 125
atrapados: 12 | se escaparon: 113 | alarmas falsas: 0
recall: 0.096 | precisión: 1.0
```

The loss drops, but much more slowly: in 300 epochs it reaches 0.3194, worse than what a learning rate of 1.0 achieved in 50. Ant steps: it needs many more passes. And it catches only 12 out of 125. Look at the precision: 1.0, perfect, because all 12 alarms it raised were indeed fraud. A detector that almost never sounds an alarm is almost never wrong when it does, and yet it lets 113 frauds slip through: another reason not to judge it by a single number.

With `lr=10` (similarly truncated):

```
época   1   pérdida 0.6819
época  10   pérdida 0.8124
época  50   pérdida 0.4365
época 300   pérdida 0.4365
exactitud: 0.844
fraudes en el examen: 125
atrapados: 0 | se escaparon: 125 | alarmas falsas: 0
Traceback (most recent call last):
  ...
    "| precisión:", round(atrapados / (atrapados + falsas), 3))
                          ~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
```

The strides were so long that at epoch 10 the loss **increased** (from 0.6819 to 0.8124): it jumped to the other side of the valley. Then it got stuck at 0.4365. And what was the end result? Accuracy **0.844**, zero caught, zero alarms: **the model ended up behaving like the lazy detector.** With `lr=10`, in this run the updates shut off all eight intermediate signals (`relu` zeroed out all 25,600 across the study purchases) and the model was left with a constant negative score, identical for every purchase (−1.67 on the exam). With the cutoff at zero, it never raises an alarm. Note: it is still minimizing the same loss as always; it didn't “choose” accuracy. Scoring 84 % is simply a consequence of fraud being rare.

The `ZeroDivisionError` occurs because precision is caught divided by alarms, and there were 0 alarms: 0 / 0. You can't measure the aim of someone who never fired a shot.

**Exercise 28.** The blanks are `zeros`, `y_exa`, `y_exa`, and `0`. This block is not meant to be copied: it shows only the lines with blanks already filled, and `...` represents the lines of your file that don't change:

```text
siempre_no = torch.zeros(800)
exactitud_flojo = (siempre_no == y_exa).float().mean().item()
atrapados_flojo = (siempre_no * y_exa).sum().item()
...
    if y_exa[i].item() == 0:
```

Output (the last three lines, following the normal output of the detector):

```
exactitud del flojo: 0.844
fraudes que atrapa: 0 de 125
El for dice lo mismo: 675 aciertos de 800
```

Recall is 0 / 125 = **0**: it doesn't catch a single fraud. Precision cannot be calculated for the same reason as in exercise 27: it raises zero alarms, and 0 / 0 is undefined. A detector with 84 % accuracy and zero recall: that is the entire lesson in two numbers.

**Exercise 29.** One solution:

```python
# Ejercicio 29: ¿qué dato le importa más al modelo?
nombres = ["monto", "hora", "distancia", "extranjero"]
for col in range(4):
    X_revuelto = X_exa.clone()                                  # copia del examen
    X_revuelto[:, col] = X_exa[torch.randperm(800), col]        # revuelve UNA columna
    with torch.no_grad():
        dice = (predecir(X_revuelto) > 0).float()
    exac = (dice == y_exa).float().mean().item()
    atrap = (dice * y_exa).sum().item()
    print(f"revolviendo {nombres[col]:10s} exactitud {exac:.3f}   atrapados {int(atrap)} de 125")
```

Output (at the end of the detector, with the same seed):

```
revolviendo monto      exactitud 0.848   atrapados 54 de 125
revolviendo hora       exactitud 0.873   atrapados 63 de 125
revolviendo distancia  exactitud 0.870   atrapados 64 de 125
revolviendo extranjero exactitud 0.849   atrapados 57 de 125
```

Without shuffling, it got 90.8 % right and caught 81. Shuffling **any** of the four knocks it down: accuracy drops to between 84.8 % and 87.3 %, barely above the lazy detector or equal to it, and it misses between 17 and 27 additional frauds. The ones that hurt it most: amount and foreign merchant.

This is consistent with the hidden rule: all four columns trigger a flag, and the model received labeled examples produced with those four columns. It is evidence that it uses them, not proof that it recovered the exact rule. If your numbers came out different because your `randperm` shuffled differently, that is fine: what must happen is that all four drop.

---

## Next week

In session 8, the last one, Mateo leaves the stationery shop and enters the bank to see how these same tensors work with millions of purchases, and with humans reviewing the decisions.

If you want a head start, think: if the detector looked not at a single purchase, but at the last 30 purchases of each customer, what shape would its input tensor have?

**11:48 p.m.**

The back room is quiet. Doña Lupe fell asleep in the plastic chair, her apron still on and the coin bag in her lap. The cinnamon bread sits next to the register, now stale.

Mateo doesn't run the detector on the charges from Norte. He thinks about it, finger hovering over the key, and doesn't do it: the exercise learned on synthetic purchases and a fictional rule, and is not fit for making decisions about real people. Instead, he takes out the four-cell table and places it alongside the handwritten notes Fer jotted down. He arranges the questions: which charges have no merchandise behind them, which ones occur in the early morning or far away, what evidence he can ask the bank for.

He should feel happy, and he is, halfway. He has a method. But a method is not proof.

He opens his email to write to Licenciado Ortega. He types “Sharing with you a model that” and pauses. He pictures the licenciado's face. A twenty-year-old kid, a laptop with a peeling sticker, a program that gets 91 % right and lets more than one out of every three get away. And on the other side, a bank with its own systems, its own analysts, its own rules.

He deletes the sentence.

Past the window, the night's last patrol car drives by. The light bulb buzzes. On the calendar, the red circle around the 23rd looks larger than it did this morning.

Mateo closes the laptop quietly, so as not to wake his aunt. He already knows what he has. He also knows what he is missing: not a higher number, but someone on the other side willing to review.

Nine days left until the 23rd. And the bank was not about to believe a model built on a laptop.

---

*I write from what we see every day at Hábil, where we build systems for banking, insurance, and enterprise in Mexico. If at your work or your family's business you run into a problem like this, let's talk: [habil.mx](https://www.habil.mx/en/)*
