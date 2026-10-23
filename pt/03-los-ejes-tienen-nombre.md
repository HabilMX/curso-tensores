# Tensores do zero (3): os eixos têm nome

*Como extrair um pedaço de um tensor por qualquer um dos eixos (um número, um dia, um produto, o canal vermelho de uma foto) e o que acontece com a forma quando isso é feito.*

*Tensores do zero · lição A, sessão 3 de 8*

*Cenário: os personagens e a história são inventados. Os dados técnicos, históricos e os números reais trazem a fonte.*

**Segunda-feira, 16 de novembro, 19h40. Papelaria Centro.**

A porta de aço está abaixada até a metade. Lá dentro, cheiro de papel novo e de café requentado. Dona Lupe conta notas no balcão. Mateo está com o notebook aberto ao lado da caixa registradora, entre um pote de lápis e uma pilha de cartolinas.

Faltam 37 dias para o 23 de dezembro.

Na semana passada consertaram o arquivo do Norte, o da linha cortada. Agora o PyTorch já o aceita: um tensor de forma `(2, 7, 3)`. Duas filiais, sete dias, três produtos. Quarenta e dois números.

— E aí? — pergunta dona Lupe sem levantar os olhos. — Já sabe quem está me roubando?

— Ainda não, tia. Tenho os números. Falta saber olhar para eles.

— Pois olhe. Os olhos servem para isso.

Mateo ri, mas ela tem razão. Quarenta e dois números juntos não dizem nada. São como um depósito com tudo amontoado: a resposta está ali, mas não se encontra. O que ele precisa é extrair **pedaços**. Um dia. Um produto. Uma filial. Colocá-los lado a lado e ver o que não bate.

Escreve uma linha. Aparece a sexta-feira das duas filiais.

Fica imóvel.

— Tia. Qual é o melhor dia da semana da senhora?

— A sexta, qual outro? A criançada sai da escola, vem com os pais, leva de tudo.

— No Centro, sim. Na sexta vende 2.000 de cadernos. Nenhum outro dia chega perto.

— E no Norte?

Mateo gira a tela. `500, 300, 200`.

Dona Lupe põe os óculos. Tira de novo.

— Isso o Norte vende numa terça qualquer.

Quase. A terça do Norte é 400, 250 e 150. A sexta, o dia forte, ganha dela por um triz.

E há algo pior na última coluna. As mochilas. O mais caro da loja. No Norte, quase todos os dias: 100. Como se ninguém tocasse nelas.

Mateo sente um frio na barriga. Ainda não sabe o que significa. Mas sabe que acaba de encontrar a primeira pista de verdade.

---

Na sessão anterior, o PyTorch foi instalado e a papelaria de dona Lupe virou o primeiro tensor de verdade, com forma `(2, 7, 3)`. Hoje se aprende o que Mateo fez naquela noite: extrair pedaços dele. Não só um número, mas um dia inteiro, um produto inteiro ou uma semana recortada, pelo eixo escolhido.

## Revisão para abrir

A ideia é responder sem olhar a sessão anterior. As respostas estão nas soluções, no final.

1. O que `t.shape` diz e como conferir isso com um `assert`?
2. Por que `torch.tensor([[1, 2, 3], [4, 5]])` falha?
3. Que `dtype` sai de `torch.tensor([1, 2, 3])` e qual sai de `torch.tensor([1.0, 2.0])`?

> **O código desta sessão** está no repositório público do curso, na pasta `sesion-03`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-03). Vale escrevê-lo primeiro, porque é assim que se aprende; o repositório serve para comparar quando algo não sair igual. Cada programa vem com a saída que deve produzir.

## Objetivos desta sessão

Ao terminar, será possível fazer estas quatro coisas. Ao lado de cada uma está o exercício que a demonstra.

| # | Ao terminar, é possível… | Demonstrado em |
|---|---|---|
| 1 | **Nomear** cada eixo de um tensor e **escrevê-lo** em um comentário acima do código | Exercícios 10 e 13 |
| 2 | **Extrair** um único número com um índice por eixo, `papeleria[1, 4, 2]`, e **convertê-lo** em número de Python com `.item()` | Exercício 10 |
| 3 | **Extrair** um pedaço por qualquer eixo com `:` e com um intervalo como `1:5`, e **prever** a forma dele antes de executar | Exercícios 11 e 12 |
| 4 | **Decidir** a ordem dos eixos de uma foto e **extrair** uma única cor | Exercício 13 |

**O que é preciso:** o ambiente da sessão 2 com o PyTorch instalado (ativado antes de começar) e um editor de texto. Hoje não se instala nada novo.

---

## Por que importa

Mateo não encontrou a pista somando tudo. Encontrou recortando: um dia de duas filiais, uma coluna de uma filial. É assim que trabalha quase qualquer sistema com dados.

Um programa que analisa a foto de um carro batido quase nunca a usa inteira: recorta a região do para-choque, ou fica com uma única cor. Um banco que guarda as compras de mil clientes em 30 dias, com forma `(1000, 30, 5)`, às vezes quer um cliente inteiro e às vezes só o dia de ontem de todos. É o mesmo tensor. O que muda é o eixo pelo qual ele é recortado.

E aqui vem a armadilha. Muitos dos erros que aparecem com modelos não são de matemática: são de forma. Pede-se um pedaço, acredita-se que ele tem uma forma, e ele tem outra. Nada quebra. Só sai um número errado, e a gente acredita nele. Hoje se aprende a prever essa forma antes de executar o código, para que isso não aconteça.

## Passo 1 — Dar nome a cada eixo

Antes de recortar qualquer coisa, Mateo faz algo que parece exagero e não é: escreve acima do código o que é cada eixo. Porque o PyTorch não sabe o que é “Norte”. Não sabe o que é “sexta-feira”.

Para o PyTorch, `papeleria` é uma caixa de `2 × 7 × 3` números e nada mais. Quem sabe que o primeiro eixo são as filiais é quem escreve o código. Por isso a primeira regra de hoje: **dizer em voz alta e escrever em um comentário**, como a etiqueta de uma gaveta da cozinha.

Crie um arquivo `ejes.py` e comece assim:

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

Execute com `python3 ejes.py`:

```
torch.Size([2, 7, 3])
```

Leia a forma com os nomes ao lado: **2 filiais, 7 dias, 3 produtos.** Os eixos são contados a partir de 0, como os índices de uma lista: eixo 0, eixo 1, eixo 2. Onde estiver escrito “pelo eixo 1”, leia-se “pelos dias”.

— E isso serve para quê? — diz dona Lupe, espiando. — Se já sabe qual é qual.

— Hoje sim. Daqui a três semanas, às duas da manhã, não vou lembrar.

— Igual ao tio com as chaves. Por isso pus etiqueta nelas.

## Passo 2 — Um único número: `papeleria[1, 4, 2]`

A primeira coisa que Mateo quer ver é um único número: o que o Norte vendeu de mochilas na sexta. Se houver algo estranho ali, quer tocá-lo com o dedo.

Na sessão 1, um número assim era apontado com `papeleria[1][4][2]`. Com um tensor isso continua funcionando, mas há um jeito melhor: **todos os índices dentro do mesmo colchete, separados por vírgulas.**

Daqui em diante, cada bloco vai sendo acrescentado ao final de `ejes.py`, e o arquivo é executado de novo; em cada saída aparecem só as linhas novas. Acrescente:

```python
print(papeleria[1][4][2])
print(papeleria[1, 4, 2])
```

```
tensor(200)
tensor(200)
```

As duas dão o mesmo: Norte, sexta, mochilas, $200. No melhor dia da semana. Mateo anota o número em um post-it e cola na borda da tela.

A diferença está em como trabalham por dentro:

- `papeleria[1][4][2]` são **três passos**: tira a aba do Norte, dela a sexta, da sexta as mochilas. Caixa dentro de caixa.
- `papeleria[1, 4, 2]` é **um passo só**: recebe o endereço completo e vai direto.

No passo 7 aparece um caso em que a dos colchetes separados engana. A partir de hoje, use a das vírgulas.

### O que sai não é um número: é um tensor de zero dimensões

A saída não diz `200`, diz `tensor(200)`: o número vem embrulhado. Acrescente isto:

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

Forma vazia, zero dimensões: é um **escalar**, como o total de $18.500 da sessão 1, mas em versão tensor. Uma bala que ainda está no papel. `.item()` tira o papel e devolve um `int` comum de Python, para imprimir, guardar em uma lista ou mandar para um relatório.

E a autocorreção de sempre: primeiro a forma, depois o valor.

```python
assert papeleria[1, 4, 2].shape == (), "Un solo número no tiene ejes"
assert papeleria[1, 4, 2].item() == 200, "Norte, viernes, mochilas: 200"
print("Paso 2 correcto")
```

```
Paso 2 correcto
```

`torch.Size([])` se compara sem problema com a tupla vazia `()`: as formas continuam sendo escritas como tuplas.

Um número solto não prova nada. 200 pesos de mochilas pode ser só uma sexta ruim. Mateo precisa ver a semana inteira do Norte, de uma vez.

## Passo 3 — Um índice só: o eixo desaparece

O que acontece quando se dão menos índices que eixos? Teste com um só:

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

A aba inteira do Norte. Mateo desce os olhos pela última coluna: 100, 150, 100, 100, 200, 100, 100. Plana como uma mesa.

Mas repare na forma, que hoje é a protagonista. Passou de `(2, 7, 3)` para `(7, 3)`: **o eixo das filiais desapareceu.** Uma já foi escolhida; diante da aba do Norte já não há “filiais”, só dias e produtos.

Com dois índices, a mesma coisa:

```python
print(papeleria[1, 4])
print(papeleria[1, 4].shape)
```

```
tensor([500, 300, 200])
torch.Size([3])
```

Norte, sexta, os três produtos. Foram-se dois eixos; fica o dos produtos. Primeira metade da regra de hoje:

> **Cada índice dado come o eixo dele.** Três eixos menos um índice: sobram dois. Três eixos menos três índices: não sobra nenhum, e isso é um escalar.

## Passo 4 — Os dois-pontos: “todos, por favor”

Agora vem a linha que deu o frio na barriga. Mateo não quer o Norte sozinho: quer o Norte **ao lado** do Centro, no mesmo dia, para que a diferença não tenha onde se esconder.

E para ter **a sexta das duas filiais**? É preciso pular o primeiro eixo sem escolher nada. Para isso servem os dois-pontos, `:`, que em um índice querem dizer “todos”.

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

Leia com os nomes dos eixos: **todas as filiais, o dia 4 (sexta) e, como nada foi dito sobre o produto, todos os produtos.** Linha de cima, Centro; linha de baixo, Norte.

Aí está. Em cima, 2000. Embaixo, 500. Mesmo dia, mesma saída da escola. Uma linha de código e a diferença fica à vista, sem abas nem copiar e colar.

Forma `(2, 3)`: o eixo dos dias se foi porque um foi escolhido; o das filiais ficou porque todas foram pedidas; o dos produtos, porque não foi tocado. Como pedir tacos “com tudo”: não se escolhe cebola ou coentro, leva-se o que houver.

### Conferir com a versão com `for`

Mateo não confia em uma linha só. Se vai mostrar isso ao banco, quer chegar ao resultado por dois caminhos. Com o que já se sabia, seria número por número:

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

`.tolist()` é primo de `.item()`: converte um tensor inteiro de volta em listas de Python, que se comparam com `==` como sempre. As duas dão o mesmo; vale deixar isso por escrito:

```python
assert viernes.shape == (2, 3), "Viernes: 2 sucursales × 3 productos"
assert viernes.tolist() == [[2000, 1200, 700], [500, 300, 200]]
print("Paso 4 correcto")
```

```
Paso 4 correcto
```

## Passo 5 — Um intervalo: `papeleria[0, 1:5]`

— Vamos ver — diz dona Lupe. — Mostre como é uma semana normal. A do Centro. Mas sem a segunda, que é sempre parada, nem o fim de semana.

Mateo precisa de um intervalo: de terça a sexta. Os dois-pontos também pedem um intervalo, como nas listas de Python: `lista[1:5]` dá os elementos 1, 2, 3 e 4. **Inclui o primeiro, deixa de fora o último.** Com tensores, em qualquer eixo.

Dona Lupe quer o Centro de terça a sexta:

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

Centro, dias de 1 a 4 (o 5, sábado, fica de fora), todos os produtos. Forma `(4, 3)`: o eixo das filiais se foi por causa do índice; o dos dias **ficou**, mas mede 4 em vez de 7. Um intervalo encurta o eixo, não o remove. Compare com a lista `centro` de Python:

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

Esse pedaço, o que se extrai com dois-pontos, se chama **fatia**. É a palavra que aparece na documentação em português; em inglês é *slice*.

## Passo 6 — O último eixo: `papeleria[:, :, 2]`

Dona Lupe larga as notas. Já não está contando. Agora quem pergunta é ela: **quanto cada filial vendeu de mochilas em cada dia?** As mochilas são o que mais dá lucro. Se algo está errado, ela quer ver ali.

O produto é o eixo 2, o último; para chegar até ele, passa-se pelos outros dois com “todos”:

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

Forma `(2, 7)`: foi-se o eixo dos produtos; cada linha é uma filial e cada coluna, um dia.

Dona Lupe chega tão perto da tela que Mateo precisa se afastar. Em cima, Centro: 500, 700, 500, 600… Embaixo, Norte: 100, 150, 100, 100…

— Menino, no Norte passa mais gente do que aqui. Fica junto do ponto de ônibus.

Mateo não responde. Ainda não tem o que responder.

Na planilha, seria preciso copiar a última coluna de cada aba e colá-las lado a lado. Aqui foi uma linha. Com `for`, são dois laços:

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

Sete linhas contra uma. Essa é a graça dos tensores: quando há pressa (e Mateo tem 37 dias), a pergunta e a resposta cabem na mesma linha.

## Passo 7 — Por que a das vírgulas: a armadilha de `papeleria[:][4]`

Naquela mesma noite, Mateo quase errou. Antes de chegar à linha certa do passo 4, testou a que saía naturalmente. Para pedir a sexta das duas filiais com colchetes separados, o natural seria isto:

```python
print(papeleria[:][4])
```

```
Traceback (most recent call last):
  ...
IndexError: index 4 is out of bounds for dimension 0 with size 2
```

Interprete o erro palavra por palavra: “o índice 4 está fora dos limites da dimensão 0, que mede 2”. Dimensão 0? Mas o que se queria era o dia!

Os colchetes separados trabalham em passos, e cada passo começa de novo pelo primeiro eixo **do que sobrou** do passo anterior, não pelo eixo que se tinha em mente:

1. `papeleria[:]` é “todas as filiais”: devolve o tensor inteiro, igualzinho.
2. `[4]` cai outra vez no primeiro eixo: pede a filial 4, e só há duas.

Com vírgulas, cada posição é um eixo fixo. E desta vez houve sorte: deu erro. Se dona Lupe tivesse 7 filiais, `papeleria[:][4]` daria a filial 4 sem reclamar, e pareceria ser a sexta. **O erro que não aparece é o perigoso.**

Mateo imagina a cena: ele diante do doutor Ortega, mostrando “as sextas” que na verdade são outra filial. O banco não lhe daria uma segunda chance. Desde aquela noite, tudo com vírgulas.

## Passo 8 — `papeleria[0]` versus `papeleria[0:1]`

Uma última coisa incomoda Mateo. Há duas maneiras de pedir uma única filial, e parece que dão o mesmo. Parece.

Use o índice e a fatia para pedir **a mesma coisa**, a filial Centro:

```python
print(papeleria[0].shape)
print(papeleria[0:1].shape)
```

```
torch.Size([7, 3])
torch.Size([1, 7, 3])
```

Os mesmos 21 números, forma diferente. `papeleria[0]` é um índice: **o eixo desaparece** e o que sobra é uma tabela. `papeleria[0:1]` é uma fatia de uma única filial: **o eixo fica**, mesmo medindo 1, e o que sobra é uma planilha de uma aba só:

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

Três colchetes no começo: continua tendo três dimensões. É tirar uma folha da pasta versus ficar com uma pasta de uma folha só. E isso importa para o PyTorch: um modelo que espera “uma pasta de fotos”, com forma `(N, C, H, W)`, pode rejeitar uma foto solta de forma `(C, H, W)`. A regra completa de hoje:

> **Com um índice, o eixo desaparece. Com uma fatia, o eixo fica** (mesmo medindo 1).

| O que se escreve | Que eixo se vai | Forma | O que é |
|---|---|---|---|
| `papeleria[1, 4, 2]` | os três | `()` | Norte, sexta, mochilas |
| `papeleria[1]` | filial | `(7, 3)` | a aba do Norte |
| `papeleria[1, 4]` | filial e dia | `(3,)` | Norte, sexta, os 3 produtos |
| `papeleria[:, 4]` | dia | `(2, 3)` | a sexta das duas filiais |
| `papeleria[0, 1:5]` | filial | `(4, 3)` | Centro de terça a sexta |
| `papeleria[:, :, 2]` | produto | `(2, 7)` | as mochilas, por filial e dia |
| `papeleria[0:1]` | nenhum | `(1, 7, 3)` | uma planilha só com a aba Centro |

O truque para prever a forma: **escrever a forma original, riscar cada eixo onde foi posto um índice e trocar pelo comprimento do intervalo cada eixo onde foi posta uma fatia.** Para `papeleria[0, 1:5]`: `(2, 7, 3)` → risca o 2 → troca o 7 por 4 (de 1 a 4) → `(4, 3)`.

## Passo 9 — Uma foto de 2 × 3 pontos

**Terça-feira, 17 de novembro, 1h15. Quarto de Mateo.**

Ele não consegue dormir. No celular há uma foto que dona Lupe mandou à tarde: a maquininha do Norte, com o cabo enrolado. “Para ver que está conectada, sim”, escreveu.

Mateo pensa que uma foto também é um tensor. E que, se sabe recortar a papelaria, sabe recortar uma foto.

Para conferir, começa com uma foto minúscula feita à mão: 2 pontos de altura por 3 de largura. Cada ponto traz três números, **vermelho, verde e azul**, de 0 (nada) a 255 (tudo). O amarelo, por exemplo, é vermelho e verde no máximo, sem azul.

Crie um arquivo `foto.py`:

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

A forma tem dois 3: o primeiro são os pontos de largura; o segundo, as cores. Sem o comentário, ninguém sabe qual é qual, nem quem escreveu o código daqui a duas semanas. Extraia **só o canal vermelho**, igual às mochilas:

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

Em cima, só o ponto vermelho tem vermelho; embaixo, o branco e o amarelo têm, o preto não. Essa tabela é uma foto em preto e branco de quanto vermelho há em cada ponto: isso é “separar canais”. Um ponto solto, com as três cores:

```python
print(foto[1, 2])
```

```
tensor([255, 255,   0])
```

Embaixo, à direita: o amarelo.

Funciona igual às mochilas. Os mesmos dois-pontos, as mesmas regras. Mateo sorri no escuro. O que aprendeu com a papelaria serve para qualquer caixa de números.

## Passo 10 — A ordem dos eixos é uma decisão de projeto

Mas ali, à uma da manhã, surge uma pergunta incômoda: quem decidiu que a cor seria o último eixo? Ele mesmo. A foto foi guardada como **altura × largura × cor**, mas nada obrigava a isso. Dá para guardá-la como **cor × altura × largura**: a tabela do vermelho, depois a do verde, depois a do azul.

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

Os mesmos 18 números, outra forma: `(3, 2, 3)`. E o vermelho já não é `foto[:, :, 0]`, é só `foto_cha[0]`. **O que vem primeiro é o que se extrai com mais facilidade.**

Qual é o correto? Os dois. É como arrumar o guarda-roupa por tipo de roupa ou por conjunto, ou como dona Lupe arruma o depósito: por filial ou por produto. A mercadoria é a mesma; muda o que se encontra rápido. Por isso a ordem dos eixos é uma **decisão de projeto**: é tomada pensando no que vai ser pedido com mais frequência.

Já existem costumes. No PyTorch, a convenção comum para imagens é **canais primeiro**, que na documentação aparece como **C × H × W** (*channels*, *height*, *width*: canais, altura e largura; [documentação do TorchVision](https://docs.pytorch.org/vision/stable/transforms.html)); um lote de fotos acrescenta no começo o eixo de qual foto. Outras ferramentas e muitos programas de imagem costumam usar altura × largura × cor. Nenhum está errado; o problema é não saber qual se tem em mãos. Por isso, com fotos, **o comentário com os eixos vai sempre acima do tensor.**

| Ordem | Forma da foto | O canal vermelho | Um ponto com as 3 cores |
|---|---|---|---|
| altura × largura × cor | `(2, 3, 3)` | `foto[:, :, 0]` | `foto[1, 2]` |
| cor × altura × largura | `(3, 2, 3)` | `foto_cha[0]` | `foto_cha[:, 1, 2]` |

Executando `foto_cha[:, 1, 2]`, sai `tensor([255, 255,   0])`: o mesmo amarelo.

Mateo fecha o notebook às 2h03. No post-it da tela continua o 200. Já não lhe parece um número. Parece uma pergunta.

---

👉 **Para quem é de Administração:** o que Mateo fez naquela noite é o que se faz numa planilha ao filtrar ou ocultar colunas: “só a sexta de todas as filiais” ou “só as mochilas de todas as abas”. Aqui cada uma é uma linha. E o passo 10 serve para qualquer relatório: quando pedirem “vendas por filial e por mês”, vale perguntar o que vai nas linhas e o que vai nas colunas. É uma decisão de projeto, e é tomada pensando em quem vai ler.

---

## Exercícios da sessão 3

As soluções estão no final. O trabalho é no `ejes.py`, que já tem a `papeleria` e o comentário com os eixos. No exercício 12, os `assert` já escritos dizem se houve acerto; no 10, no 11 e no 13, compare o trabalho com as soluções.

**Exercício 10 — Aponte o número, agora com vírgulas.** Com um índice por eixo e `.item()`, escreva o `assert` de cada dado:

a) O que o Centro vendeu de canetas na quarta.
b) O que o Norte vendeu de cadernos no domingo.
c) O que o Centro vendeu de mochilas no sábado.
d) Confira com um `assert` que a resposta de a) dá o mesmo com os colchetes separados da sessão 1.

Exemplo de um resolvido: `assert papeleria[0, 0, 0].item() == 1200` (Centro, segunda, cadernos).

**Exercício 11 — Preveja a forma antes de executar.** No papel, escreva a forma de cada expressão e o que ela representa, usando o truque do passo 8. Depois execute e compare.

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

Se alguma falhar, procure qual eixo parecia ficar e se foi, ou o contrário.

**Exercício 12 — Complete as lacunas.** Preencha cada `____` e confira com os `assert`:

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

Dica para a): o domingo é o dia 6, e a fatia deixa de fora o último número.

**Exercício 13 — Uma foto própria.** Crie à mão uma foto de **3 pontos de altura por 2 de largura**, guardada na ordem que o PyTorch costuma usar: **cor × altura × largura** (`C × H × W`). Escolha as cores dos 6 pontos (vermelho, verde, azul, branco, preto, amarelo ou as que quiser) e escreva-as em um comentário. Depois:

a) Escreva o `assert` da forma.
b) Extraia o canal verde. Escreva o `assert` da forma dele e o do valor contra uma lista escrita à mão.
c) Extraia a linha de cima com as três cores. Que forma ela tem? Escreva os dois `assert`.
d) Extraia o ponto de baixo à direita com as três cores. Escreva os dois `assert`.

Não há uma única resposta; o que precisa se cumprir é que os `assert` passem e que o comentário diga o que é cada eixo.

---

## Erros comuns (e como perceber)

| Erro | Como aparece | Como perceber | O que fazer |
|---|---|---|---|
| Pedir um índice que não existe | `papeleria[1, 7, 2]` dá `IndexError: index 7 is out of bounds for dimension 1 with size 7` | A mensagem diz qual eixo (*dimension 1*, os dias) e quanto ele mede (*size 7*) | Os índices vão de 0 ao tamanho menos 1: os dias, de 0 a 6 |
| Inverter a ordem dos eixos | `papeleria[4, 1, 2]` em vez de `papeleria[1, 4, 2]` dá `IndexError: index 4 is out of bounds for dimension 0 with size 2` | Dá erro na *dimension 0*, que tem só 2 filiais | Leia o comentário dos eixos: o primeiro índice é sempre a filial |
| Encadear colchetes com `:` | `papeleria[:][4]` dá o mesmo erro de *dimension 0* ou, pior, não dá erro e devolve outra coisa | O resultado não é o que foi pedido, ou dá erro em um eixo que não foi tocado | Todos os índices em um único colchete: `papeleria[:, 4]` |
| Achar que o intervalo inclui o último | Pede-se de terça a sexta com `1:4` e falta a sexta | O `assert` da forma diz 3 dias onde se esperavam 4 | O último número fica de fora: terça a sexta é `1:5` |
| Confundir índice com fatia | `papeleria[0:1]` tem forma `(1, 7, 3)` e se esperava `(7, 3)` | O `assert` da forma falha com um 1 a mais | Índice: o eixo se vai. Fatia: o eixo fica, mesmo medindo 1 |
| Usar `.item()` com vários números | `papeleria[0, 0].item()` dá `RuntimeError: a Tensor with 3 elements cannot be converted to Scalar` | A mensagem diz quantos elementos havia (3) | `.item()` é só para um número; para vários, `.tolist()` |
| Comparar um pedaço com um número em um `assert` | `assert papeleria[0, 0] == 1200` dá `RuntimeError: Boolean value of Tensor with more than one value is ambiguous` | Quem dá erro é o `assert`, não o índice | Faltou um índice (o desejado era `papeleria[0, 0, 0]`), ou compare a forma e depois `.tolist()` com uma lista |

Uma armadilha que não dá erro: `papeleria[0, 5:20]` não acusa nada, embora só existam 7 dias; devolve sábado e domingo, forma `(2, 3)`, igual fariam as listas de Python. Se um intervalo trouxer menos do que o esperado, revise os números dele.

---

## Revisão para a próxima sessão

A sessão 4 começa com estas três perguntas. Responda sem olhar a lição:

1. Que forma tem `papeleria[:, 4]` e o que ela representa?
2. Que diferença há entre `papeleria[0]` e `papeleria[0:1]` na forma?
3. Como extrair as vendas de mochilas das duas filiais em todos os dias?

🔁 **Na próxima semana:**

Terça, 8h10 da manhã. Mateo liga para dona Lupe antes de entrar na aula. Conta sobre a sexta. Sobre as mochilas que não saem.

Do outro lado da linha, um silêncio longo. Ouve-se a porta de aço do Centro subindo.

— Pois se as mochilas não estão sendo vendidas — diz enfim dona Lupe —, o que estão cobrando das pessoas?

Mateo não sabe. Mas sabe como continuar: parar de extrair pedaços e começar a **resumi-los**. Totais e médias por eixo: quanto cada filial vendeu na semana, qual foi a média de cada dia, tudo em uma linha e conferido contra um `for`. Quando somar a semana do Norte, o número vai apontar para alguém.

---

## Soluções (não olhe antes de tentar)

**Revisão para abrir.**

1. `t.shape` diz quantos elementos há em cada eixo: para a papelaria, 2 filiais, 7 dias e 3 produtos. Confere-se com `assert papeleria.shape == (2, 7, 3)`.
2. Porque não é retangular: a primeira linha tem 3 números e a segunda, 2. O PyTorch se recusa e diz isso: `ValueError: expected sequence of length 3 at dim 1 (got 2)`.
3. `torch.tensor([1, 2, 3])` sai `torch.int64` (inteiros) e `torch.tensor([1.0, 2.0])` sai `torch.float32` (decimais).

**Exercício 10.**

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

Os itens a) e b) são os mesmos dados procurados nos itens a) e c) do exercício 2 da sessão 1, agora com vírgulas.

**Exercício 11.**

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

a) a aba Centro; b) o domingo do Centro; c) o domingo das duas filiais; d) os cadernos das duas, todos os dias; e) Norte de segunda a quarta; f) as canetas do Norte de segunda a quarta; g) o domingo do Centro conservando o eixo de filial, que mede 1; h) Norte, sexta, mochilas. A traiçoeira é a g): parece a b), mas a filial foi pedida com fatia.

**Exercício 12.**

```python
fin_norte = papeleria[1, 5:7]       # a)
plumas = papeleria[0, 0:3, 1]       # b)
for d in range(3):                  # c)
    a_mano.append(papeleria[0, d, 1].item())
```

```
Ejercicio 12 superado
```

Em a), `5:7` pega o sábado (5) e o domingo (6) e deixa de fora o 7, que de qualquer forma não existe.

**Exercício 13.** Uma resposta possível:

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

Em c), a forma é `(3, 2)`, não `(2, 3)`: extrair um pedaço não reorganiza os eixos que sobram, deixa-os na ordem em que estavam, e a cor vem primeiro.

---

## O que vem a seguir

Na sessão 4, o total do Centro vai ser a soma de `papeleria[0]`. Para adiantar, vale calcular com um `for` quanto cada filial vendeu na semana. O do Centro já se conhece desde a sessão 1: $18.500. Falta o do Norte.

---

*Escrevo a partir do que vemos todos os dias na Hábil, onde construímos sistemas para bancos, seguradoras e empresas no México. Se no trabalho ou no negócio da família aparecer um problema assim, vamos conversar: [habil.mx](https://www.habil.mx/pt/)*
