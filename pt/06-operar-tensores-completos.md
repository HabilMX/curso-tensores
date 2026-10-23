# Tensores do zero (6): operar tensores inteiros

*Multiplicar uma semana inteira de vendas pelos preços em uma única linha, a regra que decide quando o PyTorch “estica” um tensor, e a armadilha de `*` versus `@` que não dá nenhum erro.*

*Tensores do zero · lição B, sessão 6 de 8*

*Cenário: os personagens e a história são inventados. Os dados técnicos, históricos e os números reais trazem a fonte.*

**Segunda-feira, 7 de dezembro, 19h40. Papelaria Herrera, filial Centro.**

A porta de enrolar está meio abaixada. Lá dentro há cheiro de papel novo e de café requentado. Dona Lupe conta moedas sobre o balcão. Mateo está com o notebook aberto entre um rolo de papel laminado e uma caixa de canetas.

Faltam dezesseis dias para o dia 23.

Desde a semana passada, Mateo sabe duas coisas. Que o Norte ia bem até agosto. E que em agosto um homem de camiseta azul “trocou a maquininha” sem que ninguém o tivesse chamado. O que ele não sabe é o que uma coisa tem a ver com a outra.

— Já não quero pesos, tia — diz ele. — Quero peças.

— Peças de quê?

— De tudo. Quantos cadernos, quantas canetas, quantas mochilas. Cada dia. Do Centro e do Norte.

Dona Lupe olha para ele por cima dos óculos.

— Para isso não precisa de computador, filho. Precisa de alguém que conte.

— Já contaram. A Fer me mandou a do Norte de manhã.

Há duas semanas, ninguém na família queria dizer o nome da Fer em voz alta. O Centro vendia umas quatro vezes o que o Norte vendia, e ela era a encarregada. Naquela tarde, ela não atendeu o telefone. Mas hoje mandou a contagem completa, linha por linha, sem que ninguém precisasse pedir duas vezes.

O celular de dona Lupe vibra. Um e-mail do banco. Do doutor Ortega.

“Prezada senhora Herrera: conforme solicitado, envio o total cobrado pela maquininha da filial Norte na semana indicada. Lembro que o prazo final é 23 de dezembro. Fico à disposição.”

Embaixo, um único número: **$16.850**.

Dona Lupe passa o celular para Mateo sem dizer nada.

— Isso é muito ou pouco?

— Depende de quanto a gente vendeu.

E aí está a questão. Para saber quanto venderam, é preciso multiplicar cada peça pelo preço dela. Vinte e uma células por filial. Com `for`, Mateo já sabe fazer. Mas esta noite ele quer uma conta que pegue a tabela inteira de uma vez só.

Porque, se as peças vezes os preços derem menos do que a maquininha cobrou, alguém cobrou algo que nunca saiu da papelaria.

Na sessão anterior, mudou-se a forma de um tensor sem tocar nos números: 12 meses em uma fila viraram 4 trimestres de 3 meses. Hoje os números param de mudar de lugar e começam as contas com eles, com o tensor inteiro de uma vez.

## Revisão para abrir

A ideia é responder sem olhar a sessão anterior. As respostas estão nas soluções, no final.

1. Quando um `reshape` é válido?
2. O que significa o `-1` em `reshape`?
3. Por que fazer `reshape` de uma tabela não é o mesmo que girá-la?

> **O código desta sessão** está no repositório público do curso, na pasta `sesion-06`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-06). Vale escrevê-lo primeiro, porque é assim que se aprende; o repositório serve para comparar quando algo não sair igual. Cada programa vem com a saída que deve produzir.

## Objetivos desta sessão

Ao terminar, estas quatro coisas serão possíveis. Ao lado de cada uma está o exercício que a demonstra.

| # | Ao terminar, é possível… | Demonstrado em |
|---|---|---|
| 1 | **Operar** dois tensores da mesma forma número por número e **conferir** o resultado contra um `for` | Exercício 23 |
| 2 | **Prever**, antes de executar, se duas formas podem ser “esticadas” para operar juntas e que forma sai | Exercícios 22 e 25 |
| 3 | **Calcular** com `@` um total por linha e **conferi-lo** de duas maneiras: com um `for` e com `sum(dim=1)` | Exercício 24 |
| 4 | **Reconhecer** quando `*` e `@` dão resultados diferentes sem dar erro | Exercício 22 |

**O que é preciso saber antes:** as sessões 1 a 5: criar um tensor, ler o `shape` e o `dtype` dele, extrair pedaços com índices, somar com `sum(dim=)` e mudar a forma com `reshape`. **O que é preciso ter:** o ambiente da sessão 2, com o PyTorch instalado, e um arquivo novo, `operar.py`, ao qual se vai acrescentando código passo a passo. Quando um passo pedir para provocar um erro, acrescente a linha, execute, interprete o erro e apague a linha para continuar.

---

## Por que importa

Mateo poderia fazer a conta como nas sessões passadas: um `for` dentro de outro `for`, célula por célula. Funciona. Mas não é assim que um modelo de inteligência artificial trabalha. Nem o sistema que suspendeu a maquininha do Norte num domingo à meia-noite.

Boa parte do trabalho de um modelo vai em duas operações: **multiplicar tensores número por número** e **multiplicar matrizes**. Quando um banco revisa uma compra, o modelo pega a lista de números da compra e a multiplica por uma tabela de “pesos”, que são números que o modelo aprendeu. Isso é uma multiplicação de matrizes. E ele não faz isso uma vez: faz milhões de vezes por segundo. Em algum lugar do banco, uma conta assim foi a que marcou as cobranças do Norte como “incomuns”.

É por isso que existem as placas de vídeo para IA. Uma analogia que ajuda: o computador comum é um punhado de chefs muito habilidosos; a placa de vídeo são milhares de ajudantes que só sabem picar, mas picam todos ao mesmo tempo. Multiplicar matrizes é “picar”: muitíssimas multiplicações e somas que podem ser repartidas entre muitos ajudantes ao mesmo tempo. A AlexNet, a rede de 2012 com a qual começou a onda atual da IA, segundo o próprio artigo dos autores “leva entre cinco e seis dias para ser treinada em duas GPUs GTX 580 de 3 GB” ([Krizhevsky, Sutskever e Hinton, 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)). Duas placas de videogame.

Hoje tudo roda no computador comum, sem placa de vídeo. As duas operações são as mesmas, e na próxima sessão elas vão ser necessárias para treinar um modelo próprio. Esta noite, Mateo precisa delas para uma pergunta mais simples: quanto dinheiro as peças dizem?

## O caso: agora dona Lupe conta peças

Até agora, os números de dona Lupe eram pesos. Desta vez Mateo pediu algo diferente: **quantas peças** o Centro vendeu de cada produto em cada dia. E, à parte, a lista de preços, que ela dita de memória: o caderno a $40, a caneta a $10 e a mochila a $350.

— Não mudaram desde a volta às aulas — diz ela. — E não vão mudar até o papel subir ou eu morrer, o que acontecer primeiro.

Com as peças e os preços, qualquer coisa de dinheiro sai de uma conta. Mateo começa pelo Centro, a filial de que ninguém suspeita: se a conta der certo ali, vai poder confiar nela quando chegar ao Norte.

### Passo 1 — As peças e os preços

Escreva isto no começo de `operar.py`. Primeiro as listas de Python, já conhecidas; depois os tensores, com o que se viu na sessão 2:

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

Saída:

```
torch.Size([7, 3]) torch.int64
torch.Size([3]) torch.int64
```

Leia como na sessão 2: `unidades` é uma tabela de 7 dias por 3 produtos; `precios` é uma lista de 3 preços, um por produto. As duas guardam inteiros. Guarde as listas de Python: elas vão servir para conferir com `for`.

Antes de continuar, uma pergunta para pensar: na segunda-feira foram vendidos 30 cadernos a $40. Quanto dinheiro entrou com cadernos na segunda? Se a resposta foi $1.200, a conta de hoje já é conhecida. A única novidade é fazê-la com os 21 números de uma vez.

### Passo 2 — Elemento a elemento: duas tabelas iguais

Agora o arquivo da Fer. Mateo o abre com cuidado: na sessão 2, um arquivo do Norte trazia uma linha torta. Este vem completo. Mesma forma que o do Centro: 7 dias, 3 produtos. Antes de tudo, Mateo quer ver as duas filiais juntas, célula com célula. Acrescente ao arquivo:

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

Saída:

```
tensor([[ 40, 100,   3],
        [ 50, 115,   3],
        [ 30,  90,   3],
        [ 45, 105,   2],
        [ 65, 150,   4],
        [ 40, 100,   1],
        [ 25,  60,   2]])
```

O `+` pegou a célula segunda-cadernos do Centro (30) e a somou com a célula segunda-cadernos do Norte (10): 40. E assim com as 21 células, cada uma com o par dela. A isso se chama operar **elemento a elemento**: cada número é operado com o que está no mesmo lugar do outro tensor.

É como sobrepor duas folhas de papel quadriculado contra a janela: cada quadradinho fica em cima do par dele. Por isso as duas folhas precisam ter a mesma forma.

Confira contra a versão com `for`. Acrescente:

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

Como na sessão 5, `tolist()` devolve o tensor a listas de Python para compará-lo com o `for`. Primeiro a forma, depois o valor.

**Atenção: as listas de Python não fazem isso.** Teste no console de Python:

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

Com listas, `+` **cola** uma lista atrás da outra e `* 2` a **repete**. Com tensores, `+` soma célula por célula e `* 2` multiplica cada número. Se um dia o resultado sair com o dobro do comprimento em vez do dobro do tamanho, eram listas que estavam sendo operadas.

Mateo fica olhando a última coluna da Fer: 1, 0, 1, 0, 1, 0, 1. Quatro mochilas na semana inteira. O mais caro da loja, quase parado, igual à sessão 3. Ele anota em um post-it. Ainda não sabe para quê.

### Passo 3 — O esticamento automático: peças vezes preços

Agora sim, a conta que importa. As peças são uma tabela `(7, 3)`; os preços, uma lista `(3,)`. Não têm a mesma forma. Mesmo assim, acrescente isto:

```python
ingresos = unidades * precios
print(ingresos)
print(ingresos.shape)
```

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

Funcionou. Segunda: 30 cadernos × $40 = $1.200; 80 canetas × $10 = $800; 2 mochilas × $350 = $700. E o mesmo para os outros seis dias.

O que aconteceu? O PyTorch viu que faltava à lista de preços a dimensão dos dias e a **esticou**: fez de conta que a lista de preços estava escrita sete vezes, uma embaixo da outra, uma por dia.

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

E, já com as duas folhas do mesmo tamanho, multiplicou elemento a elemento, como no passo 2.

Em inglês isso se chama ***broadcasting***; em português vamos chamar de **esticamento automático**. O importante: o PyTorch **não copia** os preços sete vezes na memória. Só faz a conta *como se* estivessem copiados. Com 7 dias tanto faz; com 7 milhões de compras, essa diferença é o que impede o computador de travar.

É como a cartolina de preços que dona Lupe tem colada atrás do balcão, escrita com caneta hidrográfica: há uma só, e ela serve para cobrar todas as vendas do dia. Ninguém a reescreve para cada cliente.

Confira contra o `for`:

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

Olhe o `for` com calma: em `lista_precios[p]` só aparece o produto, não o dia. É exatamente o que o esticamento fez: usar o mesmo preço para todos os dias. E o total da semana, $19.900, sai com o `sum` já conhecido da sessão 4.

$19.900 no Centro. A conta está comprovada duas vezes: com tensores e com `for`. Agora Mateo tem uma ferramenta em que pode confiar. Mas, antes de usá-la com o Norte, quer saber onde ela quebra.

### Passo 4 — A regra: quando dá para esticar?

O PyTorch não estica qualquer coisa. Segue uma regra, a mesma que o NumPy usa (a biblioteca da qual o PyTorch copiou o jeito de trabalhar) e que o TensorFlow também segue. Diz assim:

> **As formas são comparadas da direita para a esquerda. Dois tamanhos são compatíveis se são iguais ou se um dos dois é 1.**

E uma parte extra: se faltam dimensões à esquerda em uma forma, o PyTorch põe uns (1) ali, e esses uns são esticados.

Aplique-a a peças vezes preços. Escreva as formas alinhadas **à direita**, como na conta de somar armada no primário:

```
unidades   7   3
precios        3
           ↑   ↑
           │   └─ 3 contra 3: iguales, compatibles
           └───── 7 contra nada: a precios le falta, se le pone 1 y se estira a 7
resultado  7   3
```

Agora um caso que **não** funciona. Dona Lupe começou a vender lápis de cor a $25 e os acrescentou à lista de preços, mas ainda não os conta na tabela de peças.

— É que as crianças pedem — defende-se ela. — Já estão na cartolina. Na contagem, ainda não.

Acrescente isto, execute e interprete o erro:

```python
precios_con_colores = torch.tensor([40, 10, 350, 25])   # agregó los colores
print(unidades * precios_con_colores)
```

```
Traceback (most recent call last):
  ...
RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1
```

Leia palavra por palavra, como na sessão 5:

- *The size of tensor a (3)*: o tamanho do tensor a, que é o da esquerda do `*`, `unidades`, é 3…
- *must match the size of tensor b (4)*: …e tem que coincidir com o tamanho do tensor b, `precios_con_colores`, que é 4…
- *at non-singleton dimension 1*: …na dimensão 1, que não é de tamanho 1. (*Singleton* quer dizer “de um único elemento”. Se um dos dois medisse 1, daria para esticar; como nenhum mede 1, não há o que fazer.)

Com a regra:

```
unidades               7   3
precios_con_colores        4
                           ↑
                           3 contra 4: ni iguales ni alguno es 1 → error
```

O erro está dizendo algo verdadeiro sobre o negócio: há quatro preços e três colunas de peças. A que coluna cabe o preço dos lápis de cor? O PyTorch não adivinha, e ainda bem. Apague essas duas linhas para continuar.

### Passo 5 — Um escalar com todos: o IVA

O caso mais simples de esticamento já vinha sendo usado sem que se soubesse. Dona Lupe quer ver as receitas com o IVA de 16 % incluído, “para o contador não brigar comigo em janeiro”. Aqui, suponha que os preços da lista dela são antes do IVA; em um balcão de verdade, o preço que o cliente vê já deve trazer os impostos incluídos ([Ley Federal de Protección al Consumidor, artigo 7 bis](https://www.diputados.gob.mx/LeyesBiblio/pdf/LFPC.pdf)). Acrescente:

```python
con_iva = ingresos * 1.16
print(con_iva)
print(con_iva.shape, con_iva.dtype)
print(ingresos.sum(), con_iva.sum())
```

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

O 1.16 é um escalar, um único número; na sessão 1 se viu que ele não tem dimensões, a forma dele é `()`. Pela regra, faltam todas as dimensões, então se põem uns e ele é esticado para `(7, 3)`. Por isso multiplica as 21 células. A semana com IVA: $23.084.

Duas coisas para notar na saída:

- Os números agora trazem um ponto (`1392.`) e o `dtype` mudou para `float32`. Multiplicar inteiros por um número com decimais dá decimais.
- A forma não mudou. Um escalar nunca muda a forma do que multiplica.

Acrescente a conferência:

```python
assert con_iva.shape == ingresos.shape, "El IVA no cambia la forma"
assert con_iva[0][0] == 1392, "Lunes, cuadernos: 1,200 × 1.16 = 1,392"
print("IVA correcto")
```

```
IVA correcto
```

### Passo 6 — `@`: multiplicar e somar de uma vez só

— E quanto a gente vendeu na sexta? — pergunta dona Lupe. — A sexta é sempre a boa.

Ela quer a receita de cada dia. Isso já daria para fazer em dois passos: multiplicar peças por preços e depois somar cada linha com `sum(dim=1)`. Essa combinação, “multiplica pares e soma”, é tão comum que tem um símbolo próprio: `@`. Chama-se **produto de matrizes** (em inglês, *matrix multiplication*; no PyTorch também existe como `torch.matmul`).

Acrescente:

```python
por_dia = unidades @ precios
print(por_dia)
print(por_dia.shape)
```

```
tensor([2700, 3550, 2400, 3000, 4250, 2350, 1650])
torch.Size([7])
```

Sete totais, um por dia, em uma única operação. Faça a conta à mão para a segunda-feira e fica claro o que `@` fez:

```
lunes:   30  80   2
precios: 40  10  350

30 × 40  +  80 × 10  +  2 × 350
 1200    +   800     +   700     =  2700
```

`@` pega **uma linha** da tabela da esquerda e **a lista** da direita, multiplica os pares e **soma tudo**. Uma linha, um número. Sete linhas, sete números. Por isso a forma passou de `(7, 3)` para `(7,)`: a dimensão dos produtos foi “consumida” na soma.

É exatamente o que a operadora de caixa faz com o cupom: peças vezes preço, linha por linha, e no final o total. `@` faz o cupom de cada dia. E a sexta, o dia bom, foi o mais alto: $4.250.

**A regra das formas para `@`**, com tabelas e listas como as desta sessão: o último tamanho da esquerda tem que ser igual ao primeiro da direita, e esse tamanho desaparece:

```
unidades @ precios
 (7, 3)    (3,)
     └──┬──┘
   3 y 3: iguales, se consumen
resultado: (7,)
```

Agora confira de três maneiras. Esta é a autocorreção mais importante da sessão:

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

Olhe o `for`: tem `suma = suma + …` dentro. Essa soma é a diferença em relação ao `for` do passo 3, que só multiplicava. `*` multiplica; `@` multiplica **e soma**.

**A ordem importa.** Teste ao contrário:

```python
print(precios @ unidades)
```

```
Traceback (most recent call last):
  ...
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 7x3)
```

*mat1 and mat2 shapes cannot be multiplied*: as formas da matriz 1 e da matriz 2 não podem ser multiplicadas. O PyTorch tratou os preços como uma tabela de 1 × 3 e a pôs contra uma de 7 × 3: o último tamanho da esquerda (3) não é igual ao primeiro da direita (7). Com `@`, quem traz as linhas vai à esquerda.

### Passo 7 — A armadilha mais perigosa: `*` e `@` com tabelas quadradas

São 21h15 e Mateo está prestes a cometer o erro que quase todo mundo comete uma vez.

Até aqui, toda vez que se usou o símbolo errado, o PyTorch reclamou. Com tabelas quadradas, não.

Dona Lupe quer comparar duas filiais em dois produtos: peças de cadernos e canetas, no Centro e no Norte. No Norte ela vende um pouco mais caro, então tem uma tabela de preços por filial. O que ela quer é a receita de cada célula: peças vezes preço, cada uma com o par dela. Isso é elemento a elemento, `*`. Mas imagine que alguém escreve `@` por costume:

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

**Nenhum dos dois deu erro.** Os dois deram uma tabela de 2 × 2. Mas só o primeiro é o que dona Lupe queria: 30 cadernos × $40 = $1.200. O segundo diz que no Centro entraram $4.800 com cadernos, quatro vezes mais que o real. De onde saiu isso? Com duas tabelas, `@` pega uma linha da esquerda e uma **coluna** da direita, e faz o “multiplica e soma” do cupom: a linha do Centro (30, 80) contra a coluna de preços de cadernos (40, 45) dá 30 × 40 + 80 × 45 = 4.800. Misturou as peças de canetas do Centro com o preço de cadernos do Norte. Não faz nenhum sentido para o negócio.

Por que não houve erro? Porque com tabelas quadradas as duas regras se cumprem ao mesmo tempo: `(2, 2)` e `(2, 2)` são iguais, então `*` funciona; e o último tamanho da esquerda (2) é igual ao primeiro da direita (2), então `@` também funciona.

É a armadilha mais perigosa porque **um erro que quebra se conserta em cinco minutos; um que não quebra chega até o relatório do diretor.** Ou até a mesa do doutor Ortega. Mateo se imagina chegando ao banco no dia 23 com uma tabela que parece perfeita e está errada. Sente um frio na espinha.

A defesa é a de toda a série: comparar contra o `for`, nem que seja de uma linha:

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

Coincide com `*`, não com `@`. E, para escolher o símbolo antes de escrever, uma única pergunta:

| Se a ideia é… | Use | O que acontece com a forma |
|---|---|---|
| Cada número com o par dele (preço × peças da mesma célula) | `*` | Fica igual (ou a do esticamento) |
| Multiplicar pares **e somá-los** (o total do cupom) | `@` | A dimensão somada desaparece |

---

**E no TensorFlow?** O esticamento segue a mesma regra e dá o mesmo resultado. O `@` é mais rígido: com uma tabela e uma lista dá erro (*different ndims*, número diferente de dimensões), e para esse caso se usa outra função, `tf.linalg.matvec`. Mesmas ideias, modos diferentes.

---

👉 **Para quem é de Administração:** isto já se faz na planilha. Ao escrever `=B2*$F$1` e arrastar a fórmula para baixo, o preço fixo de `$F$1` é usado em todas as linhas: isso é o esticamento. E `@` é a função `SOMARPRODUTO` do Excel: multiplica pares e soma. A armadilha do passo 7 também tem versão no Excel: uma fórmula que não mostra `#VALOR!`, mas soma a coluna errada. Por isso, em qualquer relatório, vale conferir uma linha à mão antes de apresentá-lo.

---

## 22h40 — O que não fecha

Mateo faz com o Norte o mesmo que fez com o Centro: as peças da Fer pelos mesmos preços, e depois a soma. Uma linha. (Essa conta fica para o exercício 23; o número não é adiantado aqui.)

Compara com o e-mail de Ortega. Refaz. Compara contra o `for`. Mesmo resultado.

— Tia. O que o Norte vendeu na semana inteira não chega nem à metade do que a maquininha cobrou.

— Vai ver quase todo mundo pagou com cartão.

— Mesmo que *todo mundo* tivesse pagado com cartão. Mesmo que ninguém tivesse deixado um peso em dinheiro. A maquininha cobrou $16.850, e do Norte não saiu mercadoria para tanto dinheiro.

Dona Lupe larga as moedas.

— Então o que cobraram?

— Nada. Cobranças sem mercadoria por trás.

Lá fora passa um ônibus e a porta de enrolar vibra.

— A Fer — diz dona Lupe, devagar. — É ela quem cobra no Norte.

Mateo já tinha pensado nisso. Por isso, enquanto o `for` rodava, abriu o e-mail da papelaria. O que ninguém olha. O que dona Lupe diz que “é só propaganda”. Procurou “Norte”. Procurou “Fernanda”.

Lá estava. Terça-feira, 22 de setembro, 18h12. Assunto: **Faltas: a maquininha não bate com o estoque**.

“Dona Lupe: fiz o fechamento da semana e a maquininha marca mais cobranças do que as que aparecem no meu estoque. Contei duas vezes. Pelo que a maquininha diz, deveriam estar faltando peças na prateleira, e não estão. Acho que tem algo errado com a maquininha nova. Aviso o banco ou aviso a senhora? Fer.”

Sem resposta. Sem abrir. Desde setembro.

— Ela fez a mesma conta que eu — diz Mateo. — Peças vezes preços contra o que foi cobrado. Com calculadora. Há dois meses e meio.

Dona Lupe tira os óculos e esfrega os olhos.

— E eu achando que era ela.

— A gente pensava com suspeitas, tia. Os números dizem outra coisa. Se a Fer estivesse roubando, não escreveria para avisar que não bate.

— Amanhã eu ligo para ela. — Uma pausa. — E levo conchas. Das de chocolate, que são as de que ela mais gosta.

Naquela noite, Mateo aprende o que nenhum `assert` ia ensinar: **decide-se com dados, não com suspeitas.**

---

## Exercícios da sessão 6

As soluções estão no final desta edição. Tente primeiro; o `assert` diz se acertou.

**Exercício 22 — Preveja antes de executar.**

a) Para cada par, escreva no papel se dá para esticar e, se der, que forma sai. Use a regra do passo 4: alinhe à direita e compare. Depois execute:

`torch.ones(...)` cria um tensor cheio de uns com a forma indicada. Aqui ele só é usado para testar formas: os valores não importam.

```python
print((torch.ones(7, 3) * torch.ones(3)).shape)
print((torch.ones(2, 7, 3) * torch.ones(7, 1)).shape)
print((torch.ones(7, 1) * torch.ones(1, 3)).shape)
print((torch.ones(7, 3) * torch.tensor(2)).shape)
print((torch.ones(2, 7, 3) * torch.ones(2)).shape)
```

Exemplo de como fica um resolvido: `(7, 3)` e `(3,)` → alinho: 3 com 3, iguais; falta o 7 na lista, ela é esticada → sai `(7, 3)`.

b) Com esta tabela, calcule **à mão**, no papel, `m * m` e `m @ m`. Depois execute e compare.

```python
m = torch.tensor([[2, 1],
                  [0, 3]])
print(m * m)
print(m @ m)
```

Dica para `@`: a célula de cima à direita sai da linha de cima, (2, 1), contra a coluna da direita, (1, 3): 2 × 1 + 1 × 3.

**Exercício 23 — As receitas do Norte.** Preencha as lacunas. Use `unidades_norte` do passo 2 e os mesmos `precios`.

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

Antes de executar, preveja: quanto entrou com mochilas na terça no Norte? Olhe a tabela de peças.

**Exercício 24 — O Buen Fin.** No Buen Fin, dona Lupe baixa $5 no caderno, nada na caneta e $50 na mochila. Suponha que ela vende as mesmas peças desta semana.

1. Crie um tensor `descuento` com os três descontos (como números negativos) e some-o a `precios` para obter `precios_bf`.
2. Com `@`, calcule `por_dia_bf`, a receita de cada dia com os preços do Buen Fin.
3. Escreva dois `assert`: o da forma e um que compare o `@` contra a versão com `*` e `sum(dim=1)`.
4. Calcule quanto ela deixou de ganhar em cada dia por causa dos descontos, e quanto na semana inteira.

Desafio: o dinheiro que ela deixou de ganhar em cada dia pode ser obtido com um único `@`, sem usar `por_dia_bf`. Com qual?

**Exercício 25 — Escreva a regra.** Sem PyTorch, só com Python. Escreva uma função `se_pueden_estirar(forma_a, forma_b)` que receba duas tuplas e devolva `True` se elas podem ser esticadas para operar juntas e `False` se não. Ela tem que passar nestes `assert`:

```python
assert se_pueden_estirar((7, 3), (3,)) == True
assert se_pueden_estirar((7, 3), (7,)) == False
assert se_pueden_estirar((2, 7, 3), (7, 1)) == True
assert se_pueden_estirar((7, 3), (4,)) == False
assert se_pueden_estirar((7, 1), (1, 3)) == True
assert se_pueden_estirar((7, 3), ()) == True
print("Ejercicio 25 superado")
```

Dicas: `forma[-1]` é o último tamanho de uma tupla e `forma[-2]`, o penúltimo; assim se percorre da direita para a esquerda. Só é preciso comparar tantos tamanhos quantos tiver a forma mais curta.

---

## Erros comuns (e como perceber)

| Erro | Como aparece | Como perceber | O que fazer |
|---|---|---|---|
| Uma lista de preços com um produto a mais (ou a menos) | `RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1` | Os dois números entre parênteses são diferentes e nenhum é 1 | Imprima o `.shape` dos dois antes de operar; corrija os dados, não a conta |
| Dividir uma tabela `(7, 3)` por um total por dia `(7,)` | `RuntimeError: The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 1` | A regra alinha à direita: pôs os 7 dias embaixo dos 3 produtos | Dê a ele forma de coluna: `por_dia.reshape(7, 1)` |
| Usar `@` onde se queria `*` com tabelas quadradas | **Não há erro.** Sai uma tabela do tamanho esperado com números que não batem | Calcule uma célula à mão ou compare contra o `for` | A pergunta: quero pares (`*`) ou pares somados (`@`)? |
| Inverter os lados de `@` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 7x3)` | O último tamanho da esquerda não é igual ao primeiro da direita | Quem traz as linhas vai à esquerda: `unidades @ precios`. Girar com `.T` não adianta aqui: uma lista `(3,)` continua sendo `(3,)` |
| Misturar inteiros e decimais em `@` | `RuntimeError: addmv input tensors must have the same dtype, but got Float, Long, and Float` | Confira o `.dtype` dos dois | Converta o de inteiros com `.float()` |
| Esperar que uma lista de Python opere como tensor | `[1, 2, 3] * 2` dá `[1, 2, 3, 1, 2, 3]` | O resultado sai mais comprido, não maior | Converta antes com `torch.tensor(...)` |

---

## Revisão para a próxima sessão

A sessão 7 começa com estas três perguntas. Responda sem olhar a lição:

1. Quando duas formas podem ser “esticadas” para operar juntas?
2. Que diferença há entre `*` e `@`?
3. Que forma sai de uma tabela `(7, 3) @` um vetor `(3,)`?

🔁 **Na próxima semana:** Mateo já consegue provar que essas cobranças não saíram da papelaria. O que ele não sabe é de onde saíram. Para isso, vai construir um detector de fraude que aprende com os dados: recebe milhares de compras inventadas com uma regra escondida, e o modelo a descobre sozinho, só com `@` e esticamento automático, as duas operações de hoje.

*As cobranças não vinham da papelaria. Vinham de outro lugar. E o banco só via a maquininha.*

---

## Soluções (não olhe antes de tentar)

**Revisão para abrir.**

1. Quando o total de números é igual antes e depois: uma forma `(12,)` cabe em `(4, 3)` porque 12 = 4 × 3, mas não em `(5, 3)`.
2. Que o PyTorch calcula esse tamanho sozinho, com os números que sobram. Só pode haver um `-1` por `reshape`.
3. Porque `reshape` rearruma os números na ordem em que estão guardados, linha por linha; girar a tabela, fazer as linhas virarem colunas, é `.T`.

**Exercício 22.**

a) Saída real:

```
torch.Size([7, 3])
torch.Size([2, 7, 3])
torch.Size([7, 3])
torch.Size([7, 3])
Traceback (most recent call last):
  ...
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 2
```

- `(7, 3)` e `(3,)`: 3 com 3; no da direita se põe 1 e ele é esticado para 7 → `(7, 3)`.
- `(2, 7, 3)` e `(7, 1)`: 3 com 1, estica; 7 com 7; falta o 2 no da direita → `(2, 7, 3)`.
- `(7, 1)` e `(1, 3)`: os dois são esticados → `(7, 3)`.
- `(7, 3)` e um escalar `()`: faltam todas ao escalar, ele é esticado → `(7, 3)`.
- `(2, 7, 3)` e `(2,)`: o 2 se alinha com o 3 da direita, não com o 2 da esquerda. 3 contra 2: erro. Este é o que engana: o 2 “se parece” com as filiais, mas a regra compara pela direita. Repare que o erro diz *dimension 2*: em uma forma de três dimensões, a dimensão 2 é a última.

b)

```
tensor([[4, 1],
        [0, 9]])
tensor([[4, 5],
        [0, 9]])
```

`m * m`: cada número ao quadrado. `m @ m`: em cima à esquerda, 2 × 2 + 1 × 0 = 4; em cima à direita, 2 × 1 + 1 × 3 = 5; embaixo à esquerda, 0 × 2 + 3 × 0 = 0; embaixo à direita, 0 × 1 + 3 × 3 = 9. Veja como é traiçoeiro: as duas tabelas coincidem em três de quatro células. Quem conferisse só a de cima à esquerda acharia que davam o mesmo.

**Exercício 23.** As lacunas são `unidades_norte` e `precios`; `lista_norte[d][p] * lista_precios[p]`, e `ingresos_norte * 1.16`. A saída:

```
tensor(5400) tensor(6264.)
Ejercicio 23 superado
```

As mochilas de terça no Norte: 0 peças, então $0. Ao imprimir `ingresos_norte`, a terça é `[400, 250, 0]`.

**Exercício 24.**

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

`precios + descuento` é elemento a elemento: duas listas `(3,)`. Na semana, ela deixou de ganhar $1.850. O desafio: `unidades @ descuento` dá o que ela deixou de ganhar em cada dia, com sinal negativo, porque os descontos são negativos; o `-` da frente o torna positivo. É o que o último `assert` confere.

**Exercício 25.** Uma solução possível:

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

Com os `assert` do enunciado embaixo, a saída é:

```
Ejercicio 25 superado
```

Percorre da direita para a esquerda com índices negativos: `-1`, `-2`… Só compara tantos tamanhos quantos tiver a forma mais curta, porque à outra faltam dimensões, e essas sempre podem ser esticadas. Se encontra um par que não é igual e em que nenhum é 1, não há mais o que fazer.

---

## O que vem a seguir

Na sessão 7 junta-se tudo o que se viu até aqui: um tensor de compras, um `@` contra uma tabela de pesos e um esticamento para somar um ajuste, repetido até o modelo aprender. Para adiantar, vale refazer o passo 6 à mão com a terça: quem chegar a $3.550 sem olhar está pronto.

---

*Escrevo a partir do que vemos todos os dias na Hábil, onde construímos sistemas para bancos, seguradoras e empresas no México. Se no trabalho ou no negócio da família aparecer um problema assim, vamos conversar: [habil.mx](https://www.habil.mx/pt/)*
