# Tensores do zero (4): somar por aba

*Totais, médias e o melhor dia da semana, extraídos de um tensor em uma única linha, e a regra que nunca falha: o eixo que se soma desaparece.*

*Tensores do zero · lição A, sessão 4 de 8*

*Cenário: os personagens e a história são inventados. Os dados técnicos, históricos e os números reais trazem a fonte.*

**Segunda-feira, 23 de novembro, 19h40. Depósito da filial Centro.**

A porta de aço já estava abaixada até a metade. Cheirava a papelão novo e a café requentado. Mateo estava com o notebook em cima de uma caixa de cadernos pautados, e dona Lupe, de pé atrás dele, ainda não tinha tirado o suéter.

— Faltam trinta dias — disse ela. — O doutor Ortega me ligou de novo. Quer “algo concreto”.

— Está quase, tia.

— Foi o que disse na segunda passada. E na outra.

Mateo não respondeu. Fazia uma semana que fatiava o tensor da papelaria: um dia aqui, um produto ali. Tinha visto coisas estranhas. A sexta do Norte, fraca como uma terça. As mochilas, paradas. Mas eram pedaços. E com pedaços não se convence um banco.

— Preciso ver tudo junto — disse. — Não um dia. Não um produto. A semana inteira de cada filial, em um único número.

— E o computador faz isso?

— Faz em uma linha. Se eu disser direito o que juntar.

Dona Lupe se sentou numa lata de tinta virada.

— Já vou avisando, filho: o Norte fica na avenida, junto do ponto de ônibus. Por aquela porta entra o dobro de gente que por esta. Se vende menos que aqui, quero saber quanto menos.

Mateo pôs os dedos no teclado. Até hoje tinha aprendido a **extrair pedaços** do tensor. Hoje era o contrário: **resumir**. Juntar muitos números em um só, ao longo do eixo que ele escolhesse.

Lá fora passou um ônibus e a porta de aço vibrou.

— Então vai — disse dona Lupe. — Deixa esse tensor falar.

*É a última sessão da lição A. A ideia é fazer o mesmo que Mateo, passo a passo.*

## Revisão para abrir

A ideia é responder sem olhar a sessão 3. As respostas estão nas soluções, no final.

1. Que forma tem `papeleria[:, 4]` e o que ela representa?
2. Que diferença há entre `papeleria[0]` e `papeleria[0:1]` na forma?
3. Como extrair as vendas de mochilas das duas filiais em todos os dias?

> **O código desta sessão** está no repositório público do curso, na pasta `sesion-04`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-04). Vale escrevê-lo primeiro, porque é assim que se aprende; o repositório serve para comparar quando algo não sair igual. Cada programa vem com a saída que deve produzir.

## Objetivos desta sessão

Ao terminar, será possível fazer estas quatro coisas. Ao lado de cada uma está o exercício que a demonstra.

| # | Ao terminar, é possível… | Demonstrado em |
|---|---|---|
| 1 | **Prever** que forma sobra depois de somar um tensor sobre um ou vários eixos, antes de executar o código | Exercício 14 |
| 2 | **Calcular** totais e médias por eixo com `sum` e `mean`, e **corrigir** o erro de `mean` com inteiros | Exercícios 15 e 17 |
| 3 | **Encontrar** o maior ou o menor ao longo de um eixo, e **em que posição** ele está, com `max`, `min`, `argmax` e `argmin` | Exercícios 16 e 17 |
| 4 | **Conferir** qualquer resultado com tensores contra o mesmo cálculo feito com `for` | Exercícios 15, 16 e 17 |

---

## O que é preciso

- O ambiente virtual da sessão 2, com o PyTorch instalado, ativado.
- Um arquivo `papeleria.py` com os dados de dona Lupe. Se já existe da sessão 3, use-o. Se não, copie assim:

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

Cada passo de hoje é um arquivo novo na mesma pasta, que começa com `from papeleria import ...`. Assim não é preciso copiar os dados toda vez.

---

## Por que importa

Um tensor guarda o detalhe. Cada célula, cada dia, cada produto. Mas o detalhe não acusa ninguém nem salva ninguém: o que serve são **respostas**. Quanto vendeu cada filial? Em que dia se vende mais? Quanto entra, em média, por dia?

Mateo percebeu uma coisa naquela noite: as três perguntas são a mesma operação. **Juntar muitos números em um, ao longo de uma direção.** No Excel é a coluna de totais no fim de cada linha, ou a linha de totais embaixo. Em um tensor isso se chama **reduzir**, porque depois da operação sobram menos números.

E não é um truque de papelaria. Os sistemas de inteligência artificial reduzem o tempo todo. Se um banco tem as compras de 1.000 clientes em 30 dias, com 5 dados por compra (forma `(1000, 30, 5)`, como na sessão 1), e tira a média sobre os dias, obtém como cada cliente costuma comprar: uma tabela de `(1000, 5)`. Uma compra que se afasta muito dessa média é a que vale a pena revisar. Exatamente o que o doutor Ortega está fazendo, do outro lado do telefone, com a maquininha do Norte.

Nas lojas acontece o mesmo em grande escala. Em 2020, uma competição internacional de previsão chamada M5 pediu para prever as vendas diárias de 10 lojas de uma rede de supermercados americana (dados públicos da competição M5 do Kaggle) em três estados dos Estados Unidos: 42.840 séries de vendas organizadas em níveis (fonte: [M5 Forecasting - Accuracy, Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy); resultados no *International Journal of Forecasting*, [doi:10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013)). Somar ao longo de um eixo, como se faz hoje, é a forma de passar de um nível para o de cima.

## O caso: a papelaria de dona Lupe, outra vez

O tensor `papeleria` tem forma `(2, 7, 3)`: 2 filiais, 7 dias, 3 produtos. São os mesmos 42 números de sempre. Esta noite Mateo vai fazer a eles as perguntas que de fato tiram o sono de dona Lupe.

### Passo 1 — O total de tudo

Mateo começou pelo mais simples, para aquecer: quanto as duas filiais venderam juntas na semana inteira? Crie `paso1.py`:

```python
from papeleria import papeleria

total = papeleria.sum()
print(total)
print(total.shape)
print(total.item())
```

Saída:

```
tensor(23100)
torch.Size([])
23100
```

`sum()` sem nada dentro junta **os 42 números** em um só: 23.100 pesos. Repare na forma: `torch.Size([])`, vazia. É um escalar, como o `()` da sessão 1, e `.item()` o devolve como número comum de Python. Bate: 18.500 do Centro mais 4.600 do Norte.

— Vinte e três mil e cem? — dona Lupe torceu a boca. — Isso eu já sei, filho. Isso o caderno me diz.

Tinha razão. Um único número esconde tudo. Ela quer saber **quanto vendeu cada filial**, **cada dia**, **cada produto**. Para isso, diz-se a `sum` ao longo de que eixo juntar.

### Passo 2 — Uma aba: somar na horizontal e na vertical

Antes de encarar o cubo inteiro, Mateo fez o que faria qualquer detetive com pressa, mas sem vontade de errar: testar em terreno conhecido. Uma única aba, a do Centro. Uma tabela de 7 dias × 3 produtos. Com uma tabela há duas maneiras de somar, conhecidas do Excel:

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

- **Na horizontal** (a coluna da direita): juntam-se os 3 produtos de cada dia. Sobram 7 números, um por dia.
- **Na vertical** (a linha de baixo): juntam-se os 7 dias de cada produto. Sobram 3 números, um por produto.

No PyTorch, diz-se qual com `dim=`, que quer dizer **dimensão**, ou seja, eixo. Vale lembrar como se numeram os eixos da aba: o eixo 0 são os dias (as linhas) e o eixo 1 são os produtos (as colunas).

Crie `paso2.py`:

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

Saída:

```
torch.Size([7, 3])
tensor([2500, 3100, 2200, 2800, 3900, 2400, 1600])
torch.Size([7])
[2500, 3100, 2200, 2800, 3900, 2400, 1600]
Por día: cuadra
```

— Esses números eu sei de cor — disse dona Lupe. — Segunda, dois mil e quinhentos; terça, três mil e cem…

Exato. São a lista `por_dia` que ela ditou para Mateo na sessão 1. O tensor e o `for` dão o mesmo. Primeira prova superada: a ferramenta não mente em terreno conhecido.

Agora repare na forma, que é o importante de hoje. Entrou `(7, 3)`. Somou-se com `dim=1`. Saiu `(7,)`. **O 3, que era o eixo 1, desapareceu.** Lógico: os três produtos foram juntados em um, então já não há “eixo dos produtos”.

Uma ferramenta nova nesse código: `.tolist()`. É parente de `.item()`: `.item()` converte um tensor de um único número em número de Python, e `.tolist()` converte um tensor de vários números em lista de Python. Ela é usada para que o `assert` possa comparar lista com lista.

Agora na vertical. No terminal de Python, com `from papeleria import papeleria`:

```
>>> papeleria[0].sum(dim=0)
tensor([9000, 5800, 3700])
>>> papeleria[0].sum(dim=0).shape
torch.Size([3])
```

Entrou `(7, 3)`, somou-se com `dim=0` e saiu `(3,)`: **o 7, que era o eixo 0, desapareceu.** Os sete dias foram juntados e sobrou um total por produto. (No exercício 15 se confere algo parecido com um `for`.)

Mateo ficou olhando as duas formas. `(7,)`. `(3,)`. E escreveu em um post-it, que colou na tampa do notebook, a regra da sessão inteira. Vale decorá-la tal como está:

> **O eixo que se reduz desaparece da forma.** Os outros ficam, na mesma ordem.

Um truque para não confundir: `dim=` não diz **o que sobra**, diz **o que se junta**. `dim=1` não é “me dá os produtos”; é “junta os produtos”. Por isso os produtos desaparecem.

— Ou seja, o que se junta vai embora — resumiu dona Lupe. — Como o décimo terceiro.

### Passo 3 — O cubo completo: três maneiras de somar

Agora sim, o cubo completo. `(2, 7, 3)`: três eixos, então há três maneiras de somar um só. Mateo ainda não executou nada. Primeiro pegou uma caneta (das que a tia vende) e previu cada forma em um guardanapo. Vale fazer o mesmo, com a regra: riscar o número do eixo que se junta.

| Junta | Que eixo | Forma que sobra | O que significa |
|---|---|---|---|
| as filiais | `dim=0` | ~~2~~, 7, 3 → `(7, 3)` | cada dia e produto, com as duas filiais juntas |
| os dias | `dim=1` | 2, ~~7~~, 3 → `(2, 3)` | cada filial e produto, na semana inteira |
| os produtos | `dim=2` | 2, 7, ~~3~~ → `(2, 7)` | cada filial e dia, com todos os produtos |

Agora a conferência. Crie `paso3.py`:

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

Saída:

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

Leia como dona Lupe leria:

- **`dim=0`** põe uma aba em cima da outra e soma célula por célula. Na segunda, em cadernos, o Centro vendeu 1.200 e o Norte 300: juntos, 1.500. É a primeira célula. As filiais desapareceram: é como se dona Lupe tivesse uma única loja grande.
- **`dim=1`** junta a semana. A linha de cima é o Centro: 9.000 de cadernos, 5.800 de canetas, 3.700 de mochilas, os mesmos do passo 2. A de baixo é o Norte. Os dias desapareceram.
- **`dim=2`** junta os produtos. A linha de cima é o `por_dia` do Centro, já conhecido; a de baixo, o do Norte. Os produtos desapareceram.

As três previsões do guardanapo bateram. Essa é a vantagem da regra: **sabe-se a forma antes de executar o código.** E, se a forma que sai não é a prevista, sabe-se que o eixo juntado foi o errado.

Mas Mateo já não estava olhando as formas. Estava olhando a linha de baixo de `dim=1`. *2350, 1400, 850.* A semana inteira do Norte, produto por produto. Não chegava nem ao que o Centro vende só de cadernos.

Faltava uma conta. A que ele tinha vindo fazer.

### Passo 4 — Juntar dois eixos de uma vez: o total de cada filial

A pergunta de dona Lupe, a da porta por onde entra o dobro de gente: quanto vendeu cada filial? Para isso é preciso juntar **os dias e os produtos**: dois eixos de uma vez. `sum` aceita uma tupla de eixos. Crie `paso4.py`:

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

Saída:

```
torch.Size([2])
tensor([18500,  4600])
[18500, 4600]
Por sucursal: cuadra
```

18.500 o Centro e 4.600 o Norte.

Os números estavam na planilha desde a sessão 1. Mas nunca tinham sido postos assim, um ao lado do outro, com a porta do Norte na cabeça.

Dona Lupe se aproximou da tela. Ficou um bom tempo sem dizer nada.

— Quatro vezes — disse por fim. — O Centro vendeu umas quatro vezes o que o Norte vendeu; o Norte, perto de um quarto. E pela porta dele entra o dobro de gente.

— Tia, isto não diz por quê. Só diz quanto.

— Sei. — Ela já estava olhando o celular. — E quem fecha o caixa do Norte toda noite, Mateo?

Ele sabia a resposta. Todos sabiam. Fer.

A regra, enquanto isso, continua igual: 2, ~~7~~, ~~3~~ → `(2,)`. Dois números riscados, dois eixos desaparecidos.

O `for` precisou de **três laços**, um dentro do outro. O tensor, de uma linha. E não é só mais curto de escrever: com arrays grandes, o `for` de Python pode demorar muito mais que o `sum` do PyTorch. Quanto mais depende do computador, da versão e do tamanho, então vale medir na própria máquina antes de citar um número.

**Cuidado ao fazer em dois passos.** Mateo, com as mãos frias, quis refazer a conta de outro jeito, só por garantia: somar primeiro um eixo e depois o outro. Mas atenção: **depois da primeira soma, os eixos são numerados de novo.** Veja esta tentativa. Crie `paso4c.py`:

```python
from papeleria import papeleria

paso_a = papeleria.sum(dim=1)     # quito los días: queda (2, 3)
print(paso_a.shape)
paso_b = paso_a.sum(dim=2)        # ¿quito los productos?
```

Saída:

```
torch.Size([2, 3])
Traceback (most recent call last):
  File ".../paso4c.py", line 5, in <module>
    paso_b = paso_a.sum(dim=2)        # ¿quito los productos?
             ^^^^^^^^^^^^^^^^^
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

Interprete o erro palavra por palavra: “dimensão fora do intervalo: esperava algo entre -2 e 1, e recebi 2”. Depois de tirar os dias, `paso_a` tem forma `(2, 3)`: só dois eixos, o 0 e o 1. Os produtos já não são o eixo 2; agora são o eixo 1. (Os números negativos contam os eixos a partir do fim; por enquanto não são necessários.)

A solução fácil: fazer as duas coisas em uma única chamada, `dim=(1, 2)`. Assim os números se referem sempre à forma original.

Mateo voltou à chamada de uma linha. 4.600. O número não se mexeu.

No outro canto do depósito, dona Lupe discava um número. Deixou tocar. Desligou. Discou de novo.

### Passo 5 — A média diária, e o erro que com certeza vai aparecer

— Me diz quanto o Norte vende num dia normal — disse dona Lupe sem largar o telefone. — Para eu dizer isso na cara dela.

Ou seja: quanto se vende **em média** por dia em cada filial? A média se calcula com `mean` (em inglês, “média”). Funciona igual a `sum`, com `dim=`. Primeiro se calculam as vendas de cada dia (juntando produtos) e depois se tira a média dos 7 dias. Crie `paso5a.py`:

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
print(ventas_diarias.dtype)
promedio_diario = ventas_diarias.mean(dim=1)
```

Saída:

```
torch.int64
Traceback (most recent call last):
  File ".../paso5a.py", line 5, in <module>
    promedio_diario = ventas_diarias.mean(dim=1)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mean(): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long
```

Deu erro.

— E agora? — perguntou dona Lupe.

— Nada grave, tia. Está reclamando que eu dei inteiros.

— Pois como todo mundo. Eu também gosto de inteiros.

Leia devagar, traduzido:

- *mean(): could not infer output dtype*: “`mean` não soube de que tipo fazer o resultado”.
- *Input dtype must be either a floating point or complex dtype*: “o que entra precisa ser decimal” (*floating point* são os números com ponto decimal; *complex* são outro tipo de número que não será usado aqui).
- *Got: Long*: “e recebi `Long`”. `Long` é como o PyTorch chama internamente o `int64`, o tipo de inteiros visto na sessão 2. Por isso o `dtype` foi impresso antes: `torch.int64`.

Por que ele se recusa? Porque uma média quase nunca dá inteira: 18.500 dividido por 7 dá 2642.857…, e um tensor de inteiros não tem onde guardar o “.857”. Em vez de cortá-lo às escondidas, o PyTorch para e avisa. Mateo gostou disso. Nesta história já havia coisas demais acontecendo às escondidas.

Há duas maneiras de corrigir.

**Correção 1: converter para decimais com `.float()`.** Crie `paso5b.py`:

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

Saída:

```
torch.float32
tensor([[2500., 3100., 2200., 2800., 3900., 2400., 1600.],
        [ 600.,  800.,  650.,  550., 1000.,  600.,  400.]])
torch.Size([2])
tensor([2642.8572,  657.1429])
2642.8571428571427
Promedio diario: cuadra
```

`.float()` não muda os números, só a forma como são guardados: repare no ponto depois de cada um (`2500.`). Agora são `float32`, decimais. O Centro vende em média $2642.86 por dia e o Norte, $657.14.

— Seiscentos e cinquenta e sete pesos — repetiu dona Lupe. — Com aquela esquina. Isso uma barraquinha de chiclete vende.

Dois detalhes desse código:

- A forma: `(2, 7)` com `dim=1` deu `(2,)`. Os dias desapareceram, como sempre.
- O `assert` compara **arredondando para duas casas decimais**, porque o tensor diz 2642.8572 e o `for`, 2642.8571428571427. Nenhum está errado: o tensor usa `float32` e o Python divide com um `float` de 64 bits. Os dois são aproximações binárias, mas o `float32` conserva menos precisão. Sem arredondar, o `assert` falharia com o cálculo certo. Atenção: arredonda-se só para comparar ou exibir; para contabilidade de verdade, guarde os valores em centavos como inteiros ou use aritmética decimal, não `float32`.

**Correção 2: criar o tensor com decimais desde o início.** Se os dados de dona Lupe viessem com decimais, o tensor já nasceria decimal e `mean` funcionaria direto (para dinheiro exato, melhor centavos inteiros). Crie `paso5c.py`:

```python
import torch

con_centavos = torch.tensor([2500.0, 3100.5, 2200.0])
print(con_centavos.dtype)
print(con_centavos.mean())
```

Saída:

```
torch.float32
tensor(2600.1667)
```

Basta que os números tenham ponto decimal para que o PyTorch escolha `float32` sozinho. Para a papelaria, que já existe com inteiros, o mais prático é a correção 1: `.float()` logo antes do `mean`.

### Passo 6 — O melhor dia: `max` e `argmax`

Mateo tinha uma suspeita diferente da da tia, e queria testá-la antes de dizer qualquer coisa. Se alguém estivesse escondendo vendas a torto e a direito, o Norte se comportaria de um jeito estranho: dias bons onde não deviam estar, dias mortos ao acaso. Então fez a última pergunta da noite: em que dia se vende mais em cada filial? Aqui há **duas** respostas que interessam: **quanto** se vendeu nesse dia (o valor) e **que dia** foi (a posição). Crie `paso6.py`:

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)       # (2, 7): sucursal × día
resultado = ventas_diarias.max(dim=1)
print(resultado)
```

Saída:

```
torch.return_types.max(
values=tensor([3900, 1000]),
indices=tensor([4, 4]))
```

Parece estranho, mas é simples. Quando se passa `dim=` para `max`, ele não devolve um tensor, e sim **uma caixinha com duas coisas**:

- `values` (valores): a venda mais alta de cada filial. 3.900 no Centro, 1.000 no Norte.
- `indices` (índices): **em que posição** do eixo estava essa venda. 4 e 4.

E qual é a posição 4 do eixo dos dias? Segunda 0, terça 1, quarta 2, quinta 3… **sexta**. As duas filiais vendem mais na sexta.

Igual a uma tupla de Python, dá para extrair as duas coisas de uma vez, cada uma em uma variável. E, com uma lista de nomes, o índice vira palavra. Crie `paso6b.py`:

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

Saída:

```
tensor([3900, 1000])
tensor([4, 4])
Centro vendió más el viernes : 3900
Norte vendió más el viernes : 1000
```

A regra da forma também vale aqui: entrou `(2, 7)`, os dias foram juntados com `dim=1`, e tanto `valores` quanto `indices` têm forma `(2,)`: um por filial.

Mateo se recostou na cadeira. O mesmo dia para as duas. O Norte não se comportava como uma loja descontrolada. Comportava-se como o Centro, só que em miniatura. Como se, de cada dia, mais ou menos por igual, faltasse um pedaço.

Isso não parece alguém que tira dinheiro do caixa um dia sim, outro não. Parece outra coisa. Mas o quê? Mateo ainda não tinha como descobrir, então não disse nada.

No exercício 16, um resultado assim é conferido contra um `for`: é mais longo, mas já se sabe escrevê-lo.

**Os parentes.** Para ter só a posição, sem o valor, usa-se `argmax` (“o argumento do máximo”, ou seja, onde ele está). Para o mais baixo, `min` e `argmin` funcionam igual. E, sem o `dim`, `max` procura no tensor inteiro e devolve um único número. Crie `paso6c.py`:

```python
from papeleria import papeleria

ventas_diarias = papeleria.sum(dim=2)
print(ventas_diarias.argmax(dim=1))
print(ventas_diarias.max())
print(ventas_diarias.min(dim=1))
```

Saída:

```
tensor([4, 4])
tensor(3900)
torch.return_types.min(
values=tensor([1600,  400]),
indices=tensor([6, 6]))
```

A sexta é o melhor dia das duas filiais; 3.900 é a venda diária mais alta do tensor inteiro; e o dia mais fraco das duas é o 6, o domingo. O mesmo padrão, outra vez. O Norte copia o Centro até nos dias ruins.

Uma diferença que vai poupar um bom tempo: `sum` e `mean` aceitam uma tupla de eixos, como `dim=(1, 2)`; **`max` e `min` não**, só um de cada vez. Isso aparece na tabela de erros.

**Para quando for necessário: `keepdim`.** Acrescentando `keepdim=True` a qualquer uma dessas funções, o eixo que se junta não desaparece, e sim fica com tamanho 1: `papeleria.sum(dim=2, keepdim=True)` dá forma `(2, 7, 1)` em vez de `(2, 7)`. Hoje não é necessário; na lição B se vê quando convém.

### Tudo em uma tabela

Eram quase nove horas. Mateo copiou em uma única folha tudo o que tinha extraído naquela noite, para que dona Lupe pudesse levar ao doutor Ortega. Cada pergunta, uma linha:

| Pergunta de dona Lupe | Código | Forma |
|---|---|---|
| Quanto vendi no total? | `papeleria.sum()` | `()` |
| Quanto vendeu cada filial? | `papeleria.sum(dim=(1, 2))` | `(2,)` |
| Quanto vendeu cada filial em cada dia? | `papeleria.sum(dim=2)` | `(2, 7)` |
| Quanto vendeu cada filial de cada produto? | `papeleria.sum(dim=1)` | `(2, 3)` |
| Quanto vendi de cada produto em cada dia, juntando filiais? | `papeleria.sum(dim=0)` | `(7, 3)` |
| Quanto vendo em média por dia em cada filial? | `papeleria.sum(dim=2).float().mean(dim=1)` | `(2,)` |
| Em que dia vendo mais em cada filial? | `papeleria.sum(dim=2).argmax(dim=1)` | `(2,)` |

---

## PyTorch e TensorFlow: a mesma soma, com outro nome

Antes de fechar o notebook, Mateo quis ter certeza de que o 4.600 não era um capricho do PyTorch. Executou a mesma conta no TensorFlow, a outra ferramenta grande. Lá as funções se chamam `reduce_sum` e `reduce_mean` (aí aparece a palavra “reduzir”), e o eixo se chama `axis` em vez de `dim`. Este bloco é para ler, não para executar: o curso não instala o TensorFlow.

```python
import tensorflow as tf
from papeleria import centro, norte

papeleria_tf = tf.constant([centro, norte])
print(tf.reduce_sum(papeleria_tf, axis=(1, 2)))
print(tf.reduce_mean(tf.reduce_sum(papeleria_tf, axis=2), axis=1))
```

Saída (TensorFlow 2.21):

```
tf.Tensor([18500  4600], shape=(2,), dtype=int32)
tf.Tensor([2642  657], shape=(2,), dtype=int32)
```

A soma dá igual: 4.600 nas duas ferramentas. Mas repare na média: **2642 e 657, sem decimais e sem nenhum aviso**. O TensorFlow mantém o tipo inteiro e faz divisão inteira: trunca a parte fracionária; o PyTorch preferiu dar erro. Um erro pelo menos avisa; um número torto que ninguém percebe é pior. Ao usar o TensorFlow, converta para decimais antes de tirar a média:

```python
promedios_tf = tf.reduce_mean(
    tf.cast(tf.reduce_sum(papeleria_tf, axis=2), tf.float32),
    axis=1,
)
print(promedios_tf)
```

Saída (TensorFlow 2.21):

```
tf.Tensor([2642.8572  657.1429], shape=(2,), dtype=float32)
```

Mateo anotou isso com sublinhado duplo. A trinta dias de um prazo final, um número que muda em silêncio é a última coisa de que se precisa.

---

👉 **Para quem é de Administração:** o que Mateo fez naquela noite se faz com uma tabela dinâmica. Ao pôr “filial” nas linhas e “soma de vendas” nos valores, o Excel junta os dias e os produtos e deixa um número por filial: é `papeleria.sum(dim=(1, 2))`. Com “dia” nas linhas, junta filiais e produtos: `papeleria.sum(dim=(0, 2))`. A regra de hoje vale igual lá: o que não entra na tabela dinâmica desaparece do relatório. E a lição do passo 5 também: antes de apresentar uma média, vale conferir se a ferramenta não cortou os centavos. E mais uma, que a tabela dinâmica não vai dizer: um total baixo diz **quanto** falta, nunca **quem** tem a culpa.

---

## Exercícios da sessão 4

As soluções estão no final desta edição. Tente primeiro; o `assert` diz se houve acerto. Sempre na mesma ordem: **primeiro a forma, depois o valor.**

**Exercício 14 — Preveja a forma.** Escreva no papel que forma sobra em cada caso e o que ela significa para dona Lupe. Depois transforme cada previsão em um `assert` e execute. O primeiro já está resolvido:

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

Pergunta extra: de que outra forma dá para escrever `papeleria.sum(dim=(0, 1, 2))`?

**Exercício 15 — A média por produto (pela metade).** Dona Lupe quer saber quanto vende **em média por dia** de cada produto em cada filial. Preencha as duas lacunas:

O prefixo `f` antes das aspas cria uma **f-string**: permite colocar um valor entre chaves dentro de um texto. Aqui ele faz a mensagem do `assert` mostrar a forma real se algo falhar.

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

Repare na mensagem do primeiro `assert`: se o eixo estiver errado, ela diz que forma saiu. Essa forma revela que eixo foi juntado por engano.

**Exercício 16 — O melhor e o pior da semana.** O código fica por conta de quem resolve. Para cada pergunta: um `assert` da forma e depois o valor.

a) Qual é o dia mais fraco de cada filial, e quanto se vendeu nesse dia? Confira a resposta do Norte com um `for`.
b) Qual é o produto estrela (o que mais vendeu na semana) de cada filial?
c) Juntando as duas filiais, em que dia se vendeu mais e quanto? Deve dar 4.900.

Dica: em cada pergunta, primeiro decida que eixos juntar na soma e depois em que eixo procurar.

**Exercício 17 — As próprias notas.** Retome as notas do exercício 4 da sessão 1. Se as próprias não foram guardadas, use estas (aluno × matéria × prova):

```python
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]
```

Converta-as em tensor e calcule, cada uma com o `assert` da forma:

a) A média geral de cada aluno.
b) A média de cada aluno em cada matéria.
c) Que aluno tem a melhor média.
d) Qual é a matéria mais difícil da turma (a de média mais baixa, juntando alunos e provas).

Confira o item a) do primeiro aluno com um `for`. Com dados próprios em outra ordem de eixos, os `dim` vão ser diferentes dos da solução, e não há problema: o que conta é que a forma prevista seja a que sai.

---

## Erros comuns (e como perceber)

| Erro | Como aparece | Como perceber | O que fazer |
|---|---|---|---|
| Achar que `dim` diz o que sobra | Quer-se um total por dia e escreve-se `dim=1`, pensando “o eixo dos dias” | A forma sai `(2, 3)` em vez de `(2, 7)` | `dim` diz o que se **junta**. Para um total por dia, junte os outros eixos |
| Tirar a média de inteiros | `RuntimeError: mean(): could not infer output dtype. ... Got: Long` | O `dtype` diz `torch.int64` | `.float()` antes do `mean`, ou dados com ponto decimal |
| Somar em dois passos com a numeração antiga | `IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)` | Foi pedido um eixo que já não existe | Depois de reduzir, os eixos são renumerados. Melhor uma única chamada: `dim=(1, 2)` |
| Comparar um tensor com uma lista | `assert por_dia == [2500, 3100, ...]` lança `AssertionError` embora os números estejam certos | O `assert` falha com números que coincidem a olho | Converta primeiro: `por_dia.tolist() == [...]` |
| Comparar decimais sem arredondar | `2642.857177734375 == 18500 / 7` dá `False` | O `assert` falha pelo último dígito | Arredonde os dois lados: `round(x, 2)` |
| Passar uma tupla para `max` | `TypeError: max() received an invalid combination of arguments - got (dim=tuple, ), ...` | Foi escrito `max(dim=(1, 2))` | `max` e `min` aceitam um único eixo. Some ou reduza primeiro um e depois procure no outro |

---

## Autoavaliação da lição A

A primeira lição terminou. Antes de passar para a lição B, confira se é possível fazer cada coisa **sem olhar as sessões**. Se alguma estiver difícil, volte à sessão indicada e repita o exercício dela.

**Sessão 1 — Um número, uma lista, uma tabela, um cubo**
- [ ] Digo quantas dimensões tem um conjunto de dados e o que cada uma representa.
- [ ] Escrevo esses dados como listas aninhadas de Python e aponto um número com os índices dele.
- [ ] Calculo a forma à mão e digo quantos números há no total, multiplicando-a.
- [ ] Reconheço uma lista torta e explico por que ela não pode ser um tensor.
- [ ] Escrevo um `assert` com mensagem em espanhol para conferir o meu próprio trabalho.

**Sessão 2 — O primeiro tensor**
- [ ] Converto uma lista de Python em tensor com `torch.tensor`.
- [ ] Leio `shape`, `ndim`, `dtype` e `numel()` de um tensor e explico o que cada um diz.
- [ ] Sei que `dtype` o PyTorch escolhe para inteiros e para decimais.

**Sessão 3 — Os eixos têm nome**
- [ ] Extraio um número com `papeleria[1, 4, 2]` e o converto em número de Python com `.item()`.
- [ ] Extraio uma fatia completa por qualquer eixo, como `papeleria[:, 4]` ou `papeleria[:, :, 2]`, e digo a forma dela antes de executar.

**Sessão 4 — Somar por aba**
- [ ] Prevejo a forma depois de reduzir um ou vários eixos.
- [ ] Calculo totais e médias por eixo com `sum` e `mean`, e corrijo o erro de `mean` com inteiros.
- [ ] Encontro o maior, o menor e a posição deles com `max`, `min`, `argmax` e `argmin`.
- [ ] Confiro um resultado com tensores contra o mesmo cálculo com `for`.

## Revisão da lição A

Dez perguntas das quatro sessões. Responda sem olhar nada; as respostas estão nas soluções.

1. Qual é a regra para saber quantas dimensões tem um conjunto de dados?
2. Se dona Lupe guarda 4 semanas de vendas, por filial, dia e produto, que forma tem o tensor e quantos números ele guarda?
3. O que `assert` faz quando a condição é verdadeira, e o que faz quando é falsa?
4. O que acontece ao passar para `torch.tensor` a lista `[[1, 2, 3], [4, 5]]`?
5. O que dizem `papeleria.ndim`, `papeleria.numel()` e `papeleria.dtype`?
6. Que `dtype` tem `torch.tensor([1, 2, 3])`? E `torch.tensor([1.5, 2.0])`?
7. Que número dá `papeleria[1, 4, 2]` e para que serve acrescentar `.item()`?
8. Que forma tem `papeleria[:, :, 2]` e o que ela representa?
9. Que forma deixa `papeleria.sum(dim=0)` e o que ela significa para dona Lupe?
10. O que `max(dim=1)` devolve que `sum(dim=1)` não devolve?

---

## Revisão para a próxima sessão

A sessão 5 começa com estas três perguntas. Responda sem olhar a lição:

1. Ao somar `papeleria` com `dim=1`, que forma sobra e o que ela significa?
2. Por que `mean` falha com inteiros e como corrigir isso?
3. Como conferir que a soma com tensores está certa?

🔁 **Na próxima semana** começa a lição B, **Os tensores em ação**. Mateo vai precisar olhar mais para trás do que uma semana: meses inteiros. E, para vê-los, vai ter que mudar a forma do tensor sem mudar os dados: os mesmos números, arrumados em outras caixas. Também se lê, palavra por palavra, o erro que aparece quando a forma nova não cabe.

Antes de baixar a porta de aço, dona Lupe mostrou o celular a Mateo. Tinha pedido a Fer o fechamento de caixa da semana desde as quatro. Depois vieram as ligações: as da tarde, e as desta noite. Nenhuma retornada. Mateo pediu que ela esperasse, que os números diziam quanto e não quem. Ela guardou o telefone sem responder.

Fer não atendeu o telefone a tarde inteira.

---

## Soluções (não olhe antes de tentar)

**Revisão para abrir (da sessão 3).**

1. `papeleria[:, 4]` tem forma `(2, 3)`: os três produtos da sexta em cada filial.
2. `papeleria[0]` tem forma `(7, 3)`: a aba do Centro, sem o eixo das filiais. `papeleria[0:1]` tem forma `(1, 7, 3)`: uma fatia que conserva o eixo das filiais, com uma única filial dentro.
3. `papeleria[:, :, 2]`: todas as filiais, todos os dias, o produto 2 (mochilas). Forma `(2, 7)`.

**Exercício 14.**

```python
assert papeleria.sum(dim=0).shape == (7, 3)          # cada día y producto, sucursales juntas
assert papeleria.sum(dim=1).shape == (2, 3)          # cada sucursal y producto, en la semana
assert papeleria.sum(dim=2).shape == (2, 7)          # cada sucursal y día, productos juntos
assert papeleria.sum(dim=(0, 1)).shape == (3,)       # cada producto, en todo
assert papeleria.sum(dim=(0, 2)).shape == (7,)       # cada día, en todo
assert papeleria.sum(dim=(0, 1, 2)).shape == ()      # un solo número: el total
```

Imprimindo também os valores:

```
tensor([11350,  7200,  4550])
tensor([3100, 3900, 2850, 3350, 4900, 3000, 2000])
tensor(23100)
Ejercicio 14 superado
```

Pergunta extra: `papeleria.sum(dim=(0, 1, 2))` é o mesmo que `papeleria.sum()`. Juntar todos os eixos é juntar tudo: 23.100.

**Exercício 15.** As lacunas são `float` e `1`: converte-se para decimais e juntam-se os dias.

```python
promedio = papeleria.float().mean(dim=1)
```

Saída:

```
tensor([[1285.7142,  828.5714,  528.5714],
        [ 335.7143,  200.0000,  121.4286]])
Ejercicio 15 superado
```

O Norte vende em média exatamente $200 de canetas por dia (1.400 dividido por 7). Com `dim=0`, a mensagem do `assert` teria dito:

```
AssertionError: Esperaba (sucursal, producto) = (2, 3) y salió (7, 3)
```

O `(7, 3)` entrega o erro: desapareceram as filiais, não os dias.

**Exercício 16.**

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

Saída:

```
tensor([1600,  400]) tensor([6, 6])
cuadernos cuadernos
viernes 4900
Ejercicio 16 superado
```

O domingo é o dia mais fraco das duas filiais (1.600 e 400), os cadernos são o produto estrela das duas e na sexta, juntando filiais, venderam-se 4.900. No item c), `argmax()` sem `dim` funciona porque `por_dia_juntas` já tem um único eixo.

**Exercício 17.**

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

Saída:

```
tensor([8.2500, 9.1667, 7.6667])
tensor([[8.6667, 8.0000, 9.3333, 7.0000],
        [9.6667, 9.0000, 8.3333, 9.6667],
        [7.3333, 8.0000, 6.6667, 8.6667]])
tensor(1)
tensor([8.5556, 8.3333, 8.1111, 8.4444]) tensor(2)
Ejercicio 17 superado
```

O segundo aluno (índice 1) tem a melhor média, 9.17, e a matéria mais difícil é a terceira (índice 2), com 8.11. Quem esqueceu o `.float()` viu o mesmo `RuntimeError` do passo 5: as notas também são inteiros.

**Revisão da lição A.**

1. O número de dimensões é o número de dados necessários para apontar um único número.
2. `(4, 2, 7, 3)`: semana, filial, dia e produto. Guarda 4 × 2 × 7 × 3 = 168 números.
3. Se a condição é verdadeira, não faz nada e o programa continua. Se é falsa, para o programa com um `AssertionError` e mostra a mensagem escrita.
4. O PyTorch se recusa a criá-lo, porque a lista é torta (uma linha de 3 e outra de 2): `ValueError: expected sequence of length 3 at dim 1 (got 2)`.
5. `ndim` dá 3 (três eixos), `numel()` dá 42 (o total de números, 2 × 7 × 3) e `dtype` dá `torch.int64` (inteiros).
6. `torch.int64` para os inteiros e `torch.float32` para os decimais.
7. Dá 200: Norte, sexta, mochilas. `.item()` o converte de tensor de um número em número comum de Python, para imprimi-lo ou compará-lo em um `assert`.
8. `(2, 7)`: as vendas de mochilas das duas filiais nos sete dias.
9. `(7, 3)`: junta as duas filiais; sobra cada dia e cada produto, como se dona Lupe tivesse uma única loja.
10. Além do valor mais alto, devolve **em que posição** ele está (os índices). `sum` só devolve os totais.

---

## O que vem a seguir

Com isso se fecha a lição A: já se sabe o que é um tensor, como criá-lo, percorrê-lo pelos eixos e resumi-lo. Na sessão 5 começa a lição B. Antes de chegar lá, resolva a autoavaliação completa; se tudo sair sem olhar, a base está firme.

---

*Escrevo a partir do que vemos todos os dias na Hábil, onde construímos sistemas para bancos, seguradoras e empresas no México. Se no trabalho ou no negócio da família aparecer um problema assim, vamos conversar: [habil.mx](https://www.habil.mx/pt/)*
