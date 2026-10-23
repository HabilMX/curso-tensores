# Tensores do zero (5): mudar de forma sem mudar os dados

*Doze meses em uma fila, quatro trimestres ou dois semestres: os mesmos números arrumados de outra maneira, e como interpretar o erro que aparece quando se pede uma forma impossível.*

*Tensores do zero · lição B, sessão 5 de 8*

*Cenário: os personagens e a história são inventados. Os dados técnicos, históricos e os números reais trazem a fonte.*

**Segunda-feira, 30 de novembro, 21h15. Cozinha de dona Lupe.**

A panela de feijão ainda estava morna no fogão. Sobre a toalha de plástico havia uma pasta verde, gorda, amarrada com um elástico.

— A Fer continua sem responder — disse dona Lupe. — Uma semana, Mateo. Uma semana.

— Vai ver está doente, tia.

— Vai ver. — Não parecia convencida.

Empurrou a pasta para ele. Dentro vinha o caderno do contador: o que o Centro vendeu em cada mês do ano passado. Doze linhas à mão, de caneta azul.

— O contador quer por trimestre. E o doutor Ortega, por semestre, para “a revisão dele”. Cada um quer na própria caixinha.

— São os mesmos doze números.

— Pois vai dizer isso a eles.

Mateo digitou os números no notebook. Doze números em fila. Não precisava mudar nenhum. Só arrumá-los de outro jeito. E uma coisa não saía da cabeça dele: se visse um ano saudável arrumado por trimestres, saberia o que procurar no Norte. Que mês devia ser forte. Qual não.

— Em que mês a senhora vende mais?

— Agosto, filho. Volta às aulas. As mães fazem fila até a rua. Quase todas pagam com cartão.

Mateo anotou *agosto* em um post-it e o colou na borda da tela. Ainda não sabia por quê.

No celular de dona Lupe acendeu uma notificação. Um e-mail do banco. Ela abriu, leu com os óculos na ponta do nariz e passou para ele sem dizer nada.

*“Informamos que as primeiras cobranças incomuns registradas na maquininha da filial Norte correspondem à segunda quinzena de agosto.”*

O post-it continuava ali. Amarelo. *Agosto*.

---

Com esta sessão começa a segunda lição, **Os tensores em ação**. Até aqui, Mateo aprendeu a olhar: criar um tensor, extrair pedaços dele e resumi-lo. Foi assim que descobriu que o Centro vende umas quatro vezes o que o Norte vende: $18.500 versus $4.600 na semana. Agora os tensores começam a trabalhar, e cada ferramenta o aproxima um passo do que aconteceu no Norte: muda-se a forma de um tensor sem tocar nos dados, fazem-se contas com tensores inteiros sem um único `for`, treina-se um modelo que detecta fraudes e vê-se onde eles aparecem em um banco, uma seguradora e uma loja. Faltam 23 dias para o prazo do banco.

| Sessão | O que se vê | Prática |
|---|---|---|
| **5 — Mudar de forma sem mudar os dados** | `reshape` e `flatten`, e como interpretar o erro de forma | Exercícios 18 a 21 |
| **6 — Operar tensores inteiros** | Fazer contas com o tensor inteiro de uma vez, sem `for` | Exercícios 22 a 25 |
| **7 — Um detector de fraude que aprende com os dados** | Treinar um modelo pequeno com tensores | Exercícios 26 a 29 |
| **8 — Tensores no banco, na seguradora e na loja** | Ler um caso real e dizer que forma tem o tensor dele | Exercícios 30 a 33, autoavaliação e revisão da lição B |

Como na lição A, cada sessão dura umas 2 horas: uma para ler e outra para praticar.

Na sessão anterior, somou-se por aba com `sum(dim=)` e conferiu-se o resultado contra um `for`. Hoje nenhum número muda: muda o jeito como estão arrumados. Igual a Mateo naquela noite, com o caderno do contador aberto ao lado do notebook.

## Revisão para abrir

A ideia é responder sem olhar a sessão 4. As respostas estão nas soluções, no final.

1. Ao somar `papeleria` com `dim=1`, que forma sobra e o que ela significa?
2. Por que `mean` falha com inteiros e como corrigir isso?
3. Como conferir que a soma com tensores está certa?

> **O código desta sessão** está no repositório público do curso, na pasta `sesion-05`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-05). Vale escrevê-lo primeiro, porque é assim que se aprende; o repositório serve para comparar quando algo não sair igual. Cada programa vem com a saída que deve produzir.

## Objetivos desta sessão

Ao lado de cada objetivo está o exercício que o demonstra.

| # | Ao terminar, é possível… | Demonstrado em |
|---|---|---|
| 1 | **Dizer**, sem executar nada, se uma forma nova (com ou sem um `-1`) é válida para um tensor, e **calcular** a dimensão que falta quando há um `-1` | Exercício 18 |
| 2 | **Rearrumar** um tensor com `reshape`, usando `-1` para que o PyTorch calcule um tamanho, e **conferir** o resultado contra um `for` | Exercícios 19 e 20 |
| 3 | **Interpretar** palavra por palavra o erro de forma do PyTorch e **corrigir** o código que o provoca | Exercício 20 |
| 4 | **Distinguir** entre rearrumar uma tabela e girá-la, e **detectar** com um `assert` quando alguém confundiu as duas coisas | Exercício 21 |
| 5 | **Achatar** um tensor com `flatten` e **restaurar** a forma dele sem perder valores | Exercício 19, parte c |

**O que é preciso saber antes:** as sessões 1 a 4: criar um tensor, ler o `shape` dele, indexar, fatiar e somar com `sum(dim=)`; e, em Python, usar `for`, `range`, listas, `append`, `tolist()` e `assert` (abaixo, no passo 3, há um lembrete). **O que é preciso ter:** o ambiente da sessão 2 com o PyTorch, e o arquivo `papeleria.py` da sessão 4 (o tensor `papeleria` de forma `(2, 7, 3)`). Os arquivos de hoje vão nessa mesma pasta.

**Alcance de hoje:** a decisão nova que se pratica é se `reshape` conserva a ordem de que se precisa. `flatten()` é o caso de levar tudo para uma fila. `.T` aparece só para contrastar com `reshape`: escreva essa linha e deixe que os `assert` a revisem. O passo 10 e o quadro do TensorFlow são leitura para ver onde isto é usado em IA: dá para executar `foto.py` por curiosidade, mas nenhum exercício precisa dele, e não é preciso instalar o TensorFlow.

---

## Por que importa

— E como arrumar sem que se misturem? — perguntou dona Lupe.

Mateo apontou a cartela de ovos ao lado do fogão.

Uma dúzia cabe em uma cartela de 2 fileiras por 6 ou em uma de 3 por 4: os mesmos 12 ovos, outra cartela. E, se pedirem para colocá-la em uma de 5 por 3, já se sabe que não dá: tem 15 buracos e há só 12 ovos.

— Isso até eu sei — disse ela. — E, se sobrar um buraco, alguém comeu um ovo.

Com os tensores isso acontece o tempo todo: os dados chegam com uma forma e o passo seguinte precisa deles com outra. Dona Lupe tem vendas por mês e o contador dela quer por trimestre. Uma foto chega como grade e o modelo a quer como uma fila de números. Não se inventa nem se perde um único número: só se troca a cartela. Isso é **reshape** (“mudar de forma”, em inglês).

E uma razão prática: alguns dos erros mais frequentes com tensores são os de forma: pedir uma cartela que não combina com os ovos. Hoje se aprende a interpretá-los sem susto. E aparece também o outro erro, o perigoso: o que não quebra. A cartela parece igual, fecha a conta, ninguém reclama… e os ovos já não são os mesmos.

## O caso: as vendas do ano de dona Lupe

São as doze linhas do caderno do contador: o que o Centro vendeu em cada mês do ano passado, em pesos. Um ano saudável, sem sobressaltos. Agosto é o mês forte, pela volta às aulas, e dezembro também, pelas mochilas de presente. Mateo vai usá-lo como modelo: é assim que um ano normal se parece em uma papelaria de dona Lupe.

### Passo 1 — Os doze meses em uma fila

Crie um arquivo `cambiar_forma.py`:

```python
import torch

#        ene    feb    mar    abr    may    jun    jul    ago     sep    oct    nov    dic
meses = [60000, 55000, 60000, 50000, 65000, 50000, 70000, 100000, 70000, 60000, 55000, 90000]
ventas = torch.tensor(meses)
print(ventas.shape)
print(ventas.sum())
```

Execute com `python3 cambiar_forma.py`:

```
torch.Size([12])
tensor(785000)
```

Nada novo: 12 números que somam $785.000. Esse total é a testemunha: a forma vai mudar muitas vezes, e ele nunca deve se mexer.

### Passo 2 — Quatro trimestres de três meses

O contador pede as vendas por trimestre: janeiro a março, abril a junho, e assim por diante. Queremos uma tabela de **4 linhas (trimestres) por 3 colunas (meses)**.

Acrescente ao final do arquivo:

```python
trimestres = ventas.reshape(4, 3)
print(trimestres)
print(trimestres.shape)
print(trimestres.sum())
```

As linhas novas da saída (acima continuam saindo as de antes):

```
tensor([[ 60000,  55000,  60000],
        [ 50000,  65000,  50000],
        [ 70000, 100000,  70000],
        [ 60000,  55000,  90000]])
torch.Size([4, 3])
tensor(785000)
```

Quatro linhas, três colunas e o mesmo total. `reshape(4, 3)` quer dizer “arrume-os em 4 por 3”.

**Como o PyTorch decide que número vai em cada lugar?** Lê a fila original da esquerda para a direita, como um livro, e preenche a tabela nova linha por linha, igual a quem arruma ovos em uma cartela: primeiro a primeira fileira, depois a segunda.

Como os meses vinham em ordem, cada linha ficou exatamente como um trimestre. Não é sorte: a ordem de leitura coincide com o que se quer. Guarde essa ideia; no passo 9 ela vai falhar.

Agora já dá para usar tudo da lição A sobre a tabela nova. Acrescente:

```python
print(trimestres.sum(dim=1))   # total de cada trimestre
print(trimestres[3, 2])        # cuarto trimestre, tercer mes: diciembre
print(ventas.shape)            # ¿se descompuso la original?
```

Saída nova:

```
tensor([175000, 165000, 240000, 205000])
tensor(90000)
torch.Size([12])
```

- O terceiro trimestre (julho a setembro) é o mais forte: $240.000, pela volta às aulas.
- `trimestres[3, 2]` é dezembro, $90.000: linha 3 (quarto trimestre), coluna 2 (terceiro mês).
- `ventas` continua com forma `(12,)`. **`reshape` não muda a forma do tensor original: entrega um tensor novo com a forma nova.** Por isso ele é guardado em uma variável, como `trimestres`. Atenção: nem sempre é uma cópia separada; muitas vezes os dois compartilham os mesmos números por dentro. Hoje isso não importa, porque nenhum deles é modificado.

> Dona Lupe espiou por cima do ombro dele.
>
> — E como dá para saber que a máquina não inventou os números?
>
> — Não sei, tia. Por isso vou conferir.
>
> — Disso eu gosto. Nem no computador se deve acreditar de primeira.

### Passo 3 — Conferir contra um `for`

Não se deve acreditar no PyTorch só porque sim. Calcule os totais com um `for` e compare.

**Lembrete de Python que vai ser usado:** `range(4)` produz 0, 1, 2 e 3; `lista.append(x)` acrescenta `x` ao final de uma lista; `tensor.tolist()` converte um tensor em uma lista comum para compará-la com outra lista.

E uma ferramenta que aparece na série inteira. Um `assert` é uma revisão automática: se a condição é verdadeira, não imprime nada e o programa continua; se é falsa, o programa para e mostra a mensagem entre aspas. Aqui ele funciona como controle de erro: não é preciso acreditar no resultado, dá para conferi-lo. Para ver como é uma falha, vale testar isto em um console de Python:

```python
# Prueba opcional para reconocer un fallo:
assert 2 == 3, "2 no es igual a 3"
# Sale: AssertionError: 2 no es igual a 3
```

Agora sim, acrescente:

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

Saída nova:

```
Trimestres comprobados
```

Repare em `meses[t * 3 + m]`: é a conta que `reshape` faz por dentro. Dezembro (trimestre 3, mês 2) está na posição 3 × 3 + 2 = 11 da fila original. `reshape` poupa essa conta. E, como sempre: primeiro o `assert` da forma, depois o dos valores.

### Passo 4 — Dois semestres, e até três dimensões

O doutor Ortega pede a dona Lupe as vendas por semestre: duas metades de seis meses. Acrescente:

```python
semestres = ventas.reshape(2, 6)
print(semestres)
print(semestres.sum(dim=1))

anio = ventas.reshape(2, 2, 3)   # semestre × trimestre × mes
print(anio)
print(anio[1, 0, 2])
```

Saída nova:

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

Primeiro semestre, $340.000; segundo, $445.000. Somam os $785.000 de sempre.

A forma `(2, 2, 3)` transforma o ano em um **cubo**: a planilha com abas da sessão 1, uma aba por semestre e, em cada uma, uma tabela de trimestres por meses. `anio[1, 0, 2]` é o segundo semestre, o primeiro trimestre dele, o terceiro mês: setembro, $70.000. Doze números, quatro formas, e nada se perdeu.

> Mateo ficou olhando o segundo semestre. Ali estava o maior número do ano inteiro, sozinho, no meio do trimestre: agosto, $100.000. Em um ano saudável, agosto é o mês em que mais gente passa o cartão.
>
> Olhou o post-it. Olhou o e-mail do banco: *segunda quinzena de agosto*.
>
> Se alguém quisesse esconder cobranças falsas, que mês escolheria? O mês em que ninguém presta atenção, porque todo mundo está pagando.

### Passo 5 — A regra: multiplique a forma

> **Um `reshape` com tamanhos positivos é válido quando a forma nova, multiplicada, dá o mesmo total de números que a original.** Com `-1` há uma regra à parte, no passo 6.

Confira antes de executar:

| Forma pedida | Multiplicada | Serve para 12 meses? |
|---|---|---|
| `(4, 3)` | 4 × 3 = 12 | Sim |
| `(2, 2, 3)` | 2 × 2 × 3 = 12 | Sim |
| `(5, 3)` | 5 × 3 = 15 | Não: faltam 3 ovos |
| `(5, 2)` | 5 × 2 = 10 | Não: sobram 2 ovos |

`reshape` nunca preenche com zeros nem joga fora o que sobra. Se não fecha a conta, ele se recusa.

### Passo 6 — O `-1`, que calcula sozinho

Às vezes se sabe quanto se quer em uma dimensão e dá preguiça de dividir para achar a outra. Põe-se **`-1`** no lugar que não se quer calcular. Acrescente:

```python
print(ventas.reshape(-1, 3).shape)
print(ventas.reshape(6, -1).shape)
print(ventas.reshape(2, -1, 3).shape)
print(ventas.reshape(-1).shape)
```

Saída nova:

```
torch.Size([4, 3])
torch.Size([6, 2])
torch.Size([2, 2, 3])
torch.Size([12])
```

**`-1` quer dizer “calcule quanto vai aqui para fechar a conta”**. Em `reshape(-1, 3)`, o PyTorch pensa: “12 números em colunas de 3: são 4 linhas”. Em `reshape(2, -1, 3)`: “2 × algo × 3 = 12; algo é 2”.

E `reshape(-1)`, sozinho, quer dizer “ponha tudo em uma única dimensão”, seja qual for a forma: uma lista de números, como a `(12,)` do começo.

**A regra com `-1`:** multiplique os tamanhos que foram escritos; esse produto tem que dividir o total de números de forma exata (sem sobrar nada). O `-1` vale o total dividido por esse produto. Em `reshape(2, -1, 3)`: 2 × 3 = 6, e 12 ÷ 6 = 2.

- **Só pode haver um.** Com `ventas.reshape(-1, -1)` aparece `RuntimeError: only one dimension can be inferred` (“só é possível deduzir uma dimensão”).
- **Não faz milagre.** Com `reshape(-1, 5)` não há número que multiplicado por 5 dê 12, e quebra. O `-1` calcula; não inventa ovos.
- **É o único negativo que vale.** Com `ventas.reshape(-2, -6)` aparece `RuntimeError: invalid shape dimension -2 at index 0 of shape [-2, -6]`, embora -2 × -6 seja 12.

### Passo 7 — O erro de forma, lido palavra por palavra

Vamos provocar o erro de propósito. Um colega quis “5 grupos de 3 meses”. Acrescente ao final do arquivo:

```python
quintos = ventas.reshape(5, 3)
```

Execute (o caminho foi cortado com `...`; em cada computador é outro):

```
Traceback (most recent call last):
  File ".../cambiar_forma.py", line 42, in <module>
    quintos = ventas.reshape(5, 3)
              ^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[5, 3]' is invalid for input of size 12
```

Uma mensagem de erro se lê **de baixo para cima**: a última linha diz o que aconteceu; as de cima, onde. A última, palavra por palavra:

| Pedaço | O que quer dizer |
|---|---|
| `RuntimeError` | “Erro em tempo de execução”: o código estava bem escrito, mas falhou enquanto rodava. |
| `shape '[5, 3]'` | “A forma [5, 3]”. É a forma que **foi pedida**. |
| `is invalid` | “não é válida”. |
| `for input of size 12` | “para uma entrada de tamanho 12”. É o que **se tem**: 12 números. |

Junto: **“A forma 5 × 3 não serve para algo que tem 12 números.”** 5 × 3 = 15, e 15 não é 12. O erro não dá bronca: mostra o que foi pedido e o que se tem.

As de cima dizem onde: no arquivo, linha 42 (pode aparecer outro número se houver mais ou menos linhas em branco), e copiam a linha culpada com `^^^^` embaixo do pedaço que quebrou.

Se a ideia era grupos de 3 meses, a correção é `reshape(4, 3)` ou `reshape(-1, 3)`. Se de fato eram necessários 5 grupos, o problema não é o código: 12 não se divide em 5 partes iguais. Apague essa linha antes de continuar.

**A receita para qualquer erro de forma:** procure na última linha os dois tamanhos (o que foi pedido e o que se tem), multiplique e pergunte-se o que de fato se queria.

> — Que educado — disse dona Lupe quando Mateo leu o erro para ela. — Diz o que a gente pediu e o que a gente tem. Quem dera o banco fosse assim.
>
> Mateo riu. Depois parou de rir. *O que foi pedido e o que se tem.* O que se vendeu e o que se cobrou. Se alguém pusesse esses dois números juntos, um ao lado do outro…
>
> Anotou em outro post-it. Isso teria que esperar.

### Passo 8 — Achatar a papelaria com `flatten`

Voltemos à papelaria `(2, 7, 3)`. Às vezes se precisa dos 42 números em uma única fila, para guardá-los ou entregá-los a um modelo. Isso é **achatar**, como desmontar uma caixa de papelão: o mesmo papelão, sem volume.

No PyTorch isso se chama **`flatten`** (“achatar”). Antes de criar o arquivo, confira se o `papeleria.py` pode ser importado. Abra o Terminal na pasta onde ele foi salvo e execute:

```bash
python3 -c "from papeleria import papeleria; print(papeleria.shape)"
```

Deve imprimir `torch.Size([2, 7, 3])`. Se disser `ModuleNotFoundError`, coloque `papeleria.py` nesta mesma pasta e abra o Terminal ali. Se importar mas disser que não encontra `papeleria`, volte ao arquivo da sessão 4 e verifique se essa é a variável salva.

Agora sim, crie `aplanar.py`:

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

Salve o arquivo e execute-o na mesma pasta com `python3 aplanar.py`. Se aparecer um erro, não avance: compare primeiro a última linha com a saída esperada.

Saída:

```
torch.Size([2, 7, 3])
torch.Size([42])
tensor([1200,  800,  500, 1500,  900,  700])
tensor([300, 200, 100])
tensor(23100) tensor(23100)
Ida y vuelta sin perder nada
```

- **2 × 7 × 3 = 42** números, que somam os mesmos $23.100.
- **A ordem de leitura é a de sempre**, da caixa de fora para a de dentro: a aba Centro inteira, dia por dia e produto por produto, e depois a do Norte inteira. Os seis primeiros são Centro segunda e terça. Os 21 primeiros são do Centro (7 × 3), então as posições 21 a 23 são Norte segunda: 300, 200, 100.
- **Dá para voltar**, desde que se lembre qual era a forma.

`flatten()` faz o mesmo que `reshape(-1)`, mas o nome diz o que ele faz.

### Passo 9 — A armadilha: `reshape` não é girar a tabela

O mais importante da sessão, porque aqui o erro **não quebra**.

`papeleria.sum(dim=1)` é a tabela da semana: 2 filiais × 3 produtos. Dona Lupe a quer ao contrário para um relatório: **produtos nas linhas, filiais nas colunas**, 3 × 2. `reshape(3, 2)`?

Crie `girar.py`:

```python
from papeleria import papeleria

semana = papeleria.sum(dim=1)    # sucursal × producto
print(semana)
print(semana.shape)

mal = semana.reshape(3, 2)
print(mal)
```

Execute com `python3 girar.py`. Saída:

```
tensor([[9000, 5800, 3700],
        [2350, 1400,  850]])
torch.Size([2, 3])
tensor([[9000, 5800],
        [3700, 2350],
        [1400,  850]])
```

O PyTorch não reclamou: 3 × 2 = 2 × 3. Mas leia como relatório:

- A linha de cadernos deveria trazer Centro 9.000 e Norte 2.350. Traz **9.000 e 5.800**: cadernos e canetas do Centro.
- A de canetas deveria trazer 5.800 e 1.400. Traz **3.700 e 2.350**: mochilas do Centro e cadernos do Norte.

Uma bagunça. `reshape` fez a única coisa que sabe: ler em ordem (9000, 5800, 3700, 2350, 1400, 850) e preencher linhas de 2 em 2. **`reshape` troca a cartela; não sabe que ovo é de quem.**

O que se queria era **girar** a tabela: linhas viram colunas. No PyTorch é `.T` (de *transpor*, o nome matemático). Escreve-se `semana.T`, **sem parênteses**: não é chamado como função, descreve a tabela girada. Acrescente:

```python
bien = semana.T
print(bien)
print(bien.shape)
```

Saída nova:

```
tensor([[9000, 2350],
        [5800, 1400],
        [3700,  850]])
torch.Size([3, 2])
```

Agora sim: cadernos 9.000 e 2.350; canetas 5.800 e 1.400; mochilas 3.700 e 850. A mesma forma `(3, 2)` da bagunça: **mesmo tamanho, conteúdo diferente.** Por isso o `assert` da forma nunca vai sozinho. De `.T` não é preciso saber mais por enquanto: é o que de fato gira.

**E por que com os trimestres funcionou?** Porque ali não se girava nada: cortava-se uma fila em pedaços de 3, na mesma ordem. A pergunta antes de cada `reshape`: **quero cortar em pedaços, na mesma ordem de leitura?** Se sim, `reshape`. Se a ideia é que as linhas virem colunas, `reshape` entrega uma bagunça sem avisar.

> *Mesmo tamanho, conteúdo diferente.* Mateo disse em voz alta.
>
> — O quê? — perguntou dona Lupe, que já estava lavando a panela.
>
> — Uma coisa que parece igual por fora pode ser outra por dentro. Como a bagunça: passa na revisão da forma e ninguém percebe.
>
> Dona Lupe fechou a torneira. Secou as mãos no avental, devagar.
>
> — A maquininha do Norte foi trocada em agosto.
>
> Mateo levantou os olhos.
>
> — Quem trocou?
>
> — Um técnico. A Fer me mandou uma foto no celular: “Já veio o do banco, a maquininha nova está instalada”. Igualzinha à outra. E na borda da foto, um senhor de camiseta azul, de costas. Eu achei que o banco tinha mandado.
>
> — A senhora chamou ele?
>
> — Não.
>
> — E a Fer?
>
> Dona Lupe não respondeu. Na mesa, o post-it continuava dizendo *agosto*.

### Passo 10 — Por que a IA vive mudando de forma (leitura; executar é opcional)

**Uma foto se achata.** Uma foto em preto e branco de 28 × 28 pontos é uma tabela, e muitos modelos simples esperam uma fila de números por exemplo.

**Um lote se rearruma.** Os modelos veem fotos em lotes, por exemplo de 32: um cubo `(32, 28, 28)` que o modelo quer como 32 linhas, uma por foto.

Para ver isso rodando, crie `foto.py` e execute com `python3 foto.py`. `torch.zeros` cria um tensor da forma pedida cheio de zeros, para não escrever milhares de números:

```python
import torch

foto = torch.zeros(28, 28)           # una foto de 28 × 28 puntos, toda negra
print(foto.shape, foto.numel())
print(foto.flatten().shape)

lote = torch.zeros(32, 28, 28)       # 32 fotos
print(lote.reshape(32, -1).shape)
print(lote.flatten().shape)
```

Saída:

```
torch.Size([28, 28]) 784
torch.Size([784])
torch.Size([32, 784])
torch.Size([25088])
```

- A foto achatada: 784 números em fila (28 × 28).
- `lote.reshape(32, -1)`: 32 linhas, uma por foto; o `-1` calcula os 784.
- Cuidado com `lote.flatten()`: achata **tudo**, inclusive a separação entre fotos. Ficam 25.088 números (32 × 784) e a forma já não diz onde começa cada foto: só se sabe lembrando que eram 32 de 784. Com lotes, o certo é `reshape(32, -1)`, que guarda essa separação na forma.

---

> **Quadro: o mesmo no TensorFlow.** Só para comparar; não é para executar nem para instalar o TensorFlow nesta sessão: todos os objetivos e exercícios se resolvem com o PyTorch. A ideia é idêntica: `tf.reshape(ventas, (4, 3))` em vez de `ventas.reshape(4, 3)`, com o mesmo `-1`. A mensagem de erro inclui: `Input to reshape is a tensor with 12 values, but the requested shape has 15`. O TensorFlow faz a multiplicação (5 × 3 = 15); o PyTorch mostra a forma e a multiplicação fica por conta de quem lê.

---

👉 **Para quem é de Administração:** isto é o que o contador de dona Lupe faz a cada fim de trimestre, e o que se faz ao converter um relatório mensal em trimestral: os mesmos números agrupados de outro jeito, e o total do ano não se mexe. Essa é a prova de controle, como o `assert` do passo 3. E a armadilha do passo 9 vive no Excel: ao rearrumar uma tabela à mão em vez de usar “Colar especial → Transpor”, os números caem em células alheias sem nenhum alerta. Depois de rearrumar, confira um valor que se sabe de cor.

---

## Exercícios da sessão 5

As soluções estão no final. Tente primeiro. Os `assert` revisam as partes de código; para as previsões e explicações, compare a resposta com a saída esperada e com a solução no final.

**Exercício 18 — Preveja antes de executar.** Com os 12 meses de `ventas`, escreva no papel se cada forma é válida e que forma final daria:

```
(3, 4)    (6, 2)    (5, 2)    (2, -1)    (-1, 5)    (2, 3, 2)    (12, 1)
```

Depois confira uma por uma no console de Python com `ventas` já criada. Se ainda houver em `cambiar_forma.py` a linha `quintos = ventas.reshape(5, 3)` do passo 7, apague-a (senão o arquivo para nesse erro) e abra o console com:

```bash
python3 -i cambiar_forma.py
```

Quando aparecer `>>>`, escreva uma forma por vez. Se `>>>` não aparecer, o arquivo parou em outro erro: resolva-o antes de fazer o exercício. Para sair, escreva `exit()`. Fica assim (não copie o `>>>`: quem o põe é o Python; escreve-se só o que vem depois):

```text
>>> ventas.reshape(3, 4).shape
```

Assim fica uma resolvida: `(4, 3)` → 4 × 3 = 12 → válida → `torch.Size([4, 3])`. Nas inválidas, confira se o erro diz os dois tamanhos esperados.

**Exercício 19 — A tabela longa.** Muitas planilhas guardam os dados em uma tabela longa: uma linha por filial e dia, uma coluna por produto. **Partes a) e b):** preencha as lacunas e responda a pergunta de baixo.

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

Depois, sem executar nada, diga que linha de `larga` ocupa Norte quarta. Confira com `print(larga[9])`, copie os três números que saíram e acrescente:

```python
assert larga[9].tolist() == papeleria[1, 2].tolist(), "Ese renglón debería ser Norte miércoles"
```

**Parte c) — Achatar.** Sem olhar o passo 8, preencha as lacunas para achatar a papelaria e devolvê-la à forma original:

```python
plana = papeleria.____()
de_vuelta = plana.____(2, 7, 3)

assert plana.shape == (42,), "Deberían quedar 42 números en una fila"
assert de_vuelta.tolist() == papeleria.tolist(), "La ida y vuelta debe conservar cada dato"
print("Ejercicio 19c superado")
```

**Exercício 20 — Interprete o erro e corrija.** As vendas diárias do Centro em um fevereiro de 28 dias que começou numa segunda: 4 semanas exatas. Um colega quis arrumá-las em semanas:

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

a) Execute, copie a última linha do erro e explique estas quatro partes: `RuntimeError`, `shape '[7, 5]'`, `is invalid` e `input of size 28`. Depois explique por que 7 × 5 não pode acomodar 28 dias.

b) Corrija para que `por_semana` fique como 4 semanas × 7 dias (segunda a domingo). Escreva o `assert` da forma e compare os totais por semana contra um `for` sobre a lista `dias_febrero`, como no passo 3 (o dia `d` da semana `s` está em `dias_febrero[s * 7 + d]`).

c) Outro colega diz: “eu corrigi com `reshape(7, 4)`, que também dá 28”. Não quebra. Por que está errado para o que se queria? Para ver, acrescente:

```python
mal = febrero.reshape(7, 4)
print(mal[0])
```

Compare essa saída com os sete dias de segunda a domingo da primeira semana.

**Exercício 21 — Pegue a armadilha.** Cubra o passo 9 antes de tentar e escreva um programa que:

1. Calcule `semana = papeleria.sum(dim=1)`.
2. Construa a tabela de produtos × filiais (3 × 2) corretamente.
3. Tenha um `assert` da forma e outro que confira se a linha de cadernos traz `[9000, 2350]`.
4. Confira com dois `for` que `girada[p, s]` é igual a `semana[s, p]` para todos.

Depois, para testar a armadilha, substitua **só** a linha da tabela girada (na solução é `girada = semana.T`) por:

```python
girada = semana.reshape(3, 2)
```

Não mude os `assert` nem os dois `for`. Execute. Qual `assert` quebra? Por que o PyTorch não avisou antes?

---

## Erros comuns (e como perceber)

| Erro | Como aparece | Como perceber | O que fazer |
|---|---|---|---|
| Pedir uma forma que não fecha a conta | `RuntimeError: shape '[5, 3]' is invalid for input of size 12` | O programa quebra na linha do `reshape` | Multiplique a forma pedida e compare com o tamanho que o erro mostra |
| Achar que `reshape` mudou o original | Depois de `ventas.reshape(4, 3)`, `ventas.shape` continua `(12,)` | `ventas[3, 2]` quebra | Guarde o resultado: `trimestres = ventas.reshape(4, 3)` |
| Usar dois `-1` | `RuntimeError: only one dimension can be inferred` | Quebra embora o total fechasse | Deixe um único `-1` e escreva os outros tamanhos |
| Usar `reshape` para girar uma tabela | Números em linhas alheias, sem nenhum erro | Passa o `assert` da forma e quebra o dos valores | Para girar, `.T`; `reshape` só corta na ordem de leitura |

---

## Revisão para a próxima sessão

A sessão 6 começa com estas três perguntas. Responda sem olhar a lição:

1. Quando um `reshape` é válido?
2. O que significa o `-1` em `reshape`?
3. Por que fazer `reshape` de uma tabela não é o mesmo que girá-la?

🔁 **Na próxima semana:** operam-se tensores inteiros, sem `for`. Multiplicam-se preços por quantidades de todas as filiais e todos os dias de uma vez só, e aparece uma armadilha que, como a de hoje, não dá erro. Mateo vai finalmente pôr dois números lado a lado: o que se vendeu e o que se cobrou.

Naquela noite, antes de ir embora, ele escreveu ao doutor Ortega: *O banco mandou um técnico trocar a maquininha da filial Norte em agosto?*

A resposta chegou às 9h03 da manhã seguinte. Uma única linha:

*“O banco não enviou pessoal técnico à filial Norte neste ano.”*

Mateo ampliou a foto que a Fer tinha mandado para a tia em agosto. A maquininha nova, reluzente. E na borda, cortado pelo enquadramento, um homem de costas.

**Quem era o homem de camiseta azul?**

---

## Soluções (não olhe antes de tentar)

**Revisão para abrir.**

1. Sobra `(2, 3)`: os dias desaparecem porque foram somados. Cada número é o total da semana de uma filial em um produto (foi usada hoje no passo 9).
2. Porque `mean` só aceita tensores de decimais (ou complexos): com inteiros quebra sempre, mesmo que a média desse exata. A razão dessa regra é a da sessão 4: uma média quase nunca dá inteira e o PyTorch não corta os decimais às escondidas. Converte-se para decimais antes: `papeleria.float().mean(dim=1)`.
3. Calculando o mesmo com um `for`, que já se sabe que funciona, e comparando com um `assert`: primeiro a forma, depois os valores.

**Exercício 18.**

| Forma | Multiplicada | Resultado |
|---|---|---|
| `(3, 4)` | 12 | `torch.Size([3, 4])` |
| `(6, 2)` | 12 | `torch.Size([6, 2])` |
| `(5, 2)` | 10 | `RuntimeError: shape '[5, 2]' is invalid for input of size 12` |
| `(2, -1)` | 2 × 6 | `torch.Size([2, 6])` |
| `(-1, 5)` | nenhum número × 5 dá 12 | `RuntimeError: shape '[-1, 5]' is invalid for input of size 12` |
| `(2, 3, 2)` | 12 | `torch.Size([2, 3, 2])` |
| `(12, 1)` | 12 | `torch.Size([12, 1])` |

A última é curiosa: 12 linhas de um único número cada, uma coluna. Ao contrário de `(12,)`, que tem uma única dimensão, `(12, 1)` tem duas. Parece inútil, mas aparece com frequência em exemplos de IA.

**Exercício 19.** As lacunas são `-1` (ou `14`) e `tolist`. Este bloco não é para copiar: mostra só as duas linhas com lacunas já preenchidas, e os `...` são as linhas do arquivo que não mudam:

```text
larga = papeleria.reshape(-1, 3)
...
        a_mano.append(papeleria[s, d].tolist())
```

Norte quarta é a linha 9: as 7 primeiras linhas (0 a 6) são o Centro, Norte segunda é a 7, terça a 8 e quarta a 9. `print(larga[9])` dá:

```
tensor([350, 200, 100])
```

A conta é a mesma do passo 3: filial × 7 + dia = 1 × 7 + 2 = 9.

Parte c): as lacunas são `flatten` e `reshape`:

```python
plana = papeleria.flatten()
de_vuelta = plana.reshape(2, 7, 3)
```

Saída: `Ejercicio 19c superado`.

**Exercício 20.**

a) A última linha é:

```
RuntimeError: shape '[7, 5]' is invalid for input of size 28
```

`RuntimeError` é um erro em tempo de execução: o código estava bem escrito, mas falhou enquanto rodava. `shape '[7, 5]'` é a forma que o colega pediu. `is invalid` diz que não serve. `input of size 28` é o que se tem: os dias de fevereiro. 7 × 5 = 35, e 35 não é 28, então 28 dias não cabem em 7 linhas de 5.

b) Troque a última linha por `por_semana = febrero.reshape(4, 7)` (ou `reshape(-1, 7)`: “de 7 em 7, calcule quantas semanas”). O `for` é o do passo 3, com 4 e 7 em vez de 4 e 3, sobre a lista `dias_febrero`:

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

Saída:

```
tensor([18500, 18600, 18400, 20000])
Ejercicio 20 superado
```

c) `reshape(7, 4)` corta em pedaços de 4 dias. A primeira linha é `tensor([2500, 3100, 2200, 2800])`: segunda a quinta da primeira semana. Outra bagunça que não quebra (a linha seguinte começa na sexta). Que o total feche só diz que o `reshape` é possível, não que seja o que se queria.

**Exercício 21.**

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

Saída:

```
Ejercicio 21 superado
```

Com `semana.reshape(3, 2)` passa o `assert` da forma e quebra o dos cadernos: a primeira linha traz `[9000, 5800]`. O PyTorch não avisou porque, para ele, não há erro: foram pedidos 6 números em 3 × 2 e foi isso que ele entregou. Se os dados fazem sentido, só quem programa pode conferir, com um `assert` dos valores.

---

## O que vem a seguir

*Fontes dos dados técnicos: documentação oficial do PyTorch para [`reshape`](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html), [`flatten`](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html) e [`mean`](https://docs.pytorch.org/docs/stable/generated/torch.mean.html), [`Tensor.numel`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numel.html), [`torch.zeros`](https://docs.pytorch.org/docs/stable/generated/torch.zeros.html) e o atributo `Tensor.T` (na [página de tensores](https://docs.pytorch.org/docs/stable/tensors.html)), e do TensorFlow para [`tf.reshape`](https://www.tensorflow.org/api_docs/python/tf/reshape).*

Na sessão 6, as contas número por número ficam para trás. Para adiantar, vale escrever com um `for` como multiplicar por 2 cada um dos 42 números da papelaria. Na semana que vem isso se faz em uma linha.

---

*Escrevo a partir do que vemos todos os dias na Hábil, onde construímos sistemas para bancos, seguradoras e empresas no México. Se no trabalho ou no negócio da família aparecer um problema assim, vamos conversar: [habil.mx](https://www.habil.mx/pt/)*
