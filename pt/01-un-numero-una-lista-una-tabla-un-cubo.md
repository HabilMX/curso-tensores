# Tensores do zero (1): um número, uma lista, uma tabela, um cubo

*O que é um tensor, contado com as listas de Python que já se sabe escrever, e por que Einstein precisou pedir ajuda a um amigo para entendê-lo.*

*Tensores do zero · lição A, sessão 1 de 8*

*Cenário: os personagens e a história são inventados. Os dados técnicos, históricos e os números reais trazem a fonte.*

**Domingo, 1º de novembro, 23h52**

O celular de dona Lupe vibrou no criado-mudo. Uma vez. Ela, que nunca atende nada depois das dez, virou o aparelho por puro costume.

> *Detectamos cobranças incomuns na maquininha da filial Norte. A maquininha fica suspensa enquanto a análise é feita.*

Leu três vezes. Depois se sentou na beira da cama, com os óculos na testa, e não dormiu mais.

**Segunda-feira, 2 de novembro, 9h10**

— Senhora Herrera, bom dia. Quem fala é o doutor Ortega, do banco.

Voz formal. A pressa de quem tem outras quarenta ligações pendentes.

— A senhora tem até 23 de dezembro para nos provar quais cobranças dessa maquininha são vendas reais. Se não, nós a cancelamos.

— Dia 23? Doutor, é alta temporada! Sem maquininha no Norte, em dezembro…

— Por isso estou dando o prazo. Tenha um bom dia.

Clique.

**Segunda-feira, 2 de novembro, 19h40. Papelaria, filial Centro.**

Cheiro de papel novo e de borracha. A porta de aço meio abaixada, raspando cada vez que passa um caminhão. Mateo Ríos, 20 anos, segundo semestre de Sistemas, entra se abaixando, com a mochila no ombro e o notebook debaixo do braço.

— Tia, minha mãe disse que era urgente.

Dona Lupe não o cumprimenta. Coloca o celular na frente dele, com a mensagem do banco.

— Computador é contigo.

Mateo lê. Sente um frio no estômago.

— Tia, eu sei Python básico. Isto é… outra coisa.

— Então aprende a outra coisa. O prazo é 23 de dezembro. — Ela cruza os braços. — E nem venha dizer que não, que fui eu que paguei a matrícula.

Mateo abre o notebook sobre o balcão, entre um pote de lápis e uma pilha de cartolinas. Tem certeza de uma única coisa: o banco não viu uma papelaria. Viu números. Se quer defender a tia, precisa aprender a ver o mesmo que o banco.

E a gente vai aprender junto com ele.

---

Esta série tem duas lições de quatro sessões cada uma, uma sessão por semana. Na primeira lição, **Da lista ao cubo**, aprende-se o que é um tensor, como criá-lo, percorrê-lo e resumi-lo. Na segunda, **Os tensores em ação**, muda-se a forma dele, opera-se com ele inteiro, treina-se com ele um modelo que detecta fraudes e vê-se onde ele aparece em um banco, uma seguradora e uma loja.

| Sessão | O que se vê | Prática |
|---|---|---|
| **1 — Um número, uma lista, uma tabela, um cubo** | O que é uma dimensão, como contá-las e o que é a forma de um tensor, tudo com listas de Python | Exercícios 1 a 5 |
| **2 — O primeiro tensor** | Instalar o PyTorch, converter as listas em tensores, `shape` e `dtype` | Exercícios 6 a 9 |
| **3 — Os eixos têm nome** | Extrair um pedaço do tensor por qualquer um dos eixos | Exercícios 10 a 13 |
| **4 — Somar por aba** | Totais e médias por eixo, comparados com um `for` | Exercícios 14 a 17, autoavaliação e revisão da lição |

Cada sessão tem umas 2 horas: mais ou menos uma para ler e acompanhar o exemplo, e outra para praticar. A partir da sessão 2, cada uma começa com **3 perguntas de revisão** da anterior, para responder sem olhar.

**O que é preciso saber antes:** Python básico: variáveis, `if`, `for` e `while`, funções, listas e tuplas. Para quem vem do curso de Python da Hábil, são as lições 0 a 4. **O que é preciso ter:** Python 3 e um editor de texto. Hoje não se instala mais nada.

> **O código desta sessão** está no repositório público do curso, na pasta `sesion-01`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-01). Vale escrevê-lo primeiro, porque é assim que se aprende; o repositório serve para comparar quando algo não sair igual. Cada programa vem com a saída que deve produzir.

## Objetivos desta sessão

Ao terminar, será possível fazer estas quatro coisas. Ao lado de cada uma está o exercício que a demonstra.

| # | Ao terminar, é possível… | Demonstrado em |
|---|---|---|
| 1 | Diante de um conjunto de dados (vendas, notas, uma foto), **dizer** quantas dimensões ele tem e o que cada uma representa | Exercício 1 |
| 2 | **Escrever** esses dados como listas aninhadas de Python e **apontar** um número pelos índices | Exercícios 2 e 4 |
| 3 | **Calcular** a forma de um tensor (quantos elementos há em cada dimensão), manualmente e com uma função | Exercícios 1 e 3 |
| 4 | **Reconhecer** quando uma tabela não pode ser um tensor porque está torta, e comprovar isso com código | Exercício 5 |

---

## Por que importa

Naquela noite, enquanto dona Lupe conta o troco do caixa, Mateo pesquisa no celular como um banco decide que algo é “incomum”. A resposta lhe dá um pouco de medo.

O banco não lê uma compra como uma pessoa a lê. Ele a converte em números: o valor, a hora, a distância até a última compra, quantas compras houve na última hora. Uma compra vira uma lista de números. Mil compras, uma tabela. As compras de cada cliente nos últimos 30 dias, um cubo.

Ninguém no banco viu o rosto dos clientes do Norte. Uma máquina viu cubos de números e algo não bateu.

E não é só o banco. Quando uma seguradora recebe a foto de um carro batido, também não vê um carro. Vê uma grade de pontos, e cada ponto traz três números: quanto vermelho, quanto verde e quanto azul ele tem. De novo, um cubo de números.

Quase todos os sistemas de inteligência artificial dos quais se vai ouvir falar na carreira funcionam assim: primeiro convertem o mundo em caixas de números ordenadas, e depois fazem contas com essas caixas. Essas caixas se chamam **tensores**. Quem entende como estão ordenadas entende metade do que um modelo faz. A outra metade fica para a segunda lição.

— E isso aí é de comer ou de passar no cabelo? — pergunta dona Lupe, sem levantar os olhos das moedas.

— É de listas, tia. Listas dentro de listas.

É verdade, e é a boa notícia: escrever os dados de um tensor já é algo que se sabe fazer. São listas dentro de listas, só de números e bem alinhadas. Na sessão 2 elas vão virar tensores de verdade.

## O caso: a papelaria de dona Lupe

Dona Lupe tem uma papelaria com duas filiais, Centro e Norte. Vende três coisas que importam para ela: cadernos, canetas e mochilas. Os números desta série são de exemplo, mas a forma dos dados é a de qualquer negócio.

Mateo ainda não sabe o que procura. Então faz a única coisa sensata: começa pelo fácil e vai pedindo mais detalhe. Pergunta à tia pelas vendas da semana passada quatro vezes, cada vez com mais lupa.

### Passo 1 — Um número

Dona Lupe abaixa a porta de aço até o chão. O barulho metálico ecoa por toda a loja.

— Tia, quanto a senhora vendeu na semana passada?
— $18.500, no Centro. E nem me pergunte do Norte ainda, que minha pressão sobe.

```python
total = 18500
```

Um único número. Para apontá-lo não é preciso nenhuma informação: só existe um. Dizemos que ele tem **zero dimensões**. Na linguagem dos tensores, ele se chama **escalar**.

Um escalar diz muito e explica pouco. Com $18.500 ninguém consegue defender nada diante de um banco.

### Passo 2 — Uma lista

— E dia a dia?
— Segunda 2.500; terça 3.100; quarta 2.200; quinta 2.800; sexta 3.900; sábado 2.400; domingo 1.600. — Ela diz de cor, como quem reza.

```python
por_dia = [2500, 3100, 2200, 2800, 3900, 2400, 1600]
```

Agora sim é preciso uma informação para apontar um número: **o dia**. `por_dia[4]` é a sexta-feira, 3.900 (lembrete: os índices começam em 0). Uma direção, **uma dimensão**. Isso se chama **vetor**.

Vale conferir que nada se perdeu: `sum(por_dia)` dá 18500, o mesmo total do passo 1. O detalhe não muda o total; só o distribui.

Mateo anota algo em um post-it: *o detalhe não mente; o total esconde.* Ainda não sabe o quanto essa frase vai ser útil.

### Passo 3 — Uma tabela

— E por dia e por produto?

Dona Lupe bufa, pega o notebook velho, o do adesivo da Virgem, e abre a planilha:

| Dia | Cadernos | Canetas | Mochilas |
|---|---|---|---|
| Segunda | 1.200 | 800 | 500 |
| Terça | 1.500 | 900 | 700 |
| Quarta | 1.000 | 700 | 500 |
| Quinta | 1.300 | 900 | 600 |
| Sexta | 2.000 | 1.200 | 700 |
| Sábado | 1.200 | 800 | 400 |
| Domingo | 800 | 500 | 300 |

Em Python, cada linha é uma lista, e a tabela é uma lista de linhas:

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

Para apontar um número são necessárias **duas informações: o dia e o produto**. `por_dia_y_producto[4][2]` é a sexta-feira (linha 4), mochilas (coluna 2): 700. Duas direções, **duas dimensões**. Isso se chama **matriz**.

Atenção à ordem: primeiro a linha, depois a coluna. Essa ordem não é um capricho; é a ordem em que as listas foram escritas. A de fora são os dias; a de dentro, os produtos.

— Quer dizer que cada número tem o próprio endereço — diz dona Lupe.

— Exato, tia. Dia e produto. Rua e número.

### Passo 4 — Um cubo: a planilha com abas

Então Mateo faz a pergunta que a tia vinha evitando.

— E a filial Norte?

Silêncio. Dona Lupe clica na outra aba da mesma planilha, devagar, como quem abre um envelope com más notícias. Mesma forma (7 dias, 3 produtos), outros números. A planilha agora tem **duas abas, uma por filial**, e cada aba é uma tabela.

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

Mateo olha as duas abas lado a lado. Algo lhe dá um arrepio na nuca. Mas antes ele precisa colocar tudo em Python.

Em Python, a planilha completa é uma lista de abas:

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

Para apontar um número são necessárias **três informações: a filial, o dia e o produto**. `papeleria[1][4][2]` é o Norte (aba 1), sexta-feira (linha 4), mochilas (coluna 2): 200. Três direções, **três dimensões**. Já não tem nome especial: é um **tensor de três dimensões**.

Esta é a imagem que convém guardar de toda a série: **um tensor de três dimensões é uma planilha com abas, em que todas as abas têm as mesmas colunas e o mesmo número de linhas.**

| O que se tem | Para apontar um número é preciso | Dimensões | Nome |
|---|---|---|---|
| $18.500 | nada: só existe um | 0 | escalar |
| Vendas de 7 dias | o dia | 1 | vetor |
| Vendas por dia e produto | o dia e o produto | 2 | matriz |
| O mesmo, por filial | a filial, o dia e o produto | 3 | tensor de 3 dimensões |

**A regra que resume tudo:** o número de dimensões é o número de informações necessárias para apontar um único número. Cada uma dessas direções também se chama **eixo**; as duas palavras vão aparecer.

— E é só isso? Um cubo? — Dona Lupe o olha por cima dos óculos. — Eu achava que o banco usava coisa da NASA.

— Usa cubos, tia. Só que bem grandes.

### Passo 5 — Contar as dimensões com código

Contar colchetes no olho funciona com exemplos pequenos. Com o que o banco tem nas mãos, não. Mateo sabe disso: se vai brigar com uma máquina, a dele precisa contar sozinha. Antes, o Python que vai ser usado daqui em diante, caso não esteja fresco na memória:

- `type(x) == list` pergunta se `x` é uma lista.
- `x[0]` pega o primeiro elemento; `len(x)` conta quantos há.
- `lista.append(valor)` acrescenta um valor ao final de uma lista; `tuple(lista)` a converte em tupla.

Crie um arquivo `dimensiones.py` com as listas dos passos 1 a 4 e, acima delas, esta função:

```python
def contar_dimensiones(x):
    dimensiones = 0
    while type(x) == list:
        dimensiones = dimensiones + 1
        x = x[0]
    return dimensiones
```

Como funciona: pense nas bonecas russas. Enquanto o que está na mão for uma lista, a função abre a caixa (pega o primeiro elemento, `x[0]`) e conta um. Quando o que sai já é um número, ela para. Uma lista de listas de listas se abre três vezes: três dimensões. Um número não se abre nenhuma vez: zero.

Acrescente ao final do arquivo:

```python
if __name__ == "__main__":
    print(contar_dimensiones(total))
    print(contar_dimensiones(por_dia))
    print(contar_dimensiones(por_dia_y_producto))
    print(contar_dimensiones(papeleria))
```

Execute com `python3 dimensiones.py`:

```
0
1
2
3
```

A linha `if __name__ == "__main__":` é um detalhe do Python: faz com que o código de baixo só rode quando este arquivo é executado diretamente. Por enquanto, basta copiá-la como está.

Quatro números na tela. Dona Lupe espia.

— Pronto? Isso salva o Norte?

— Não, tia. Isso me diz que eu entendo o que tenho. O que não é pouco.

### Passo 6 — A forma: quantos há em cada direção

Saber que a papelaria tem três dimensões não basta. Um detetive não anota “há suspeitos”: anota quantos. Mateo também quer saber **quantos elementos há em cada uma**: 2 filiais, 7 dias, 3 produtos. Isso se chama a **forma** do tensor, e se escreve como uma tupla: `(2, 7, 3)`.

A forma diz muito num relance:

- Quantas dimensões há: quantos números a tupla tem (3).
- Qual o tamanho de cada uma: 2, 7 e 3.
- Quantos números há no total: multiplicam-se, 2 × 7 × 3 = 42.

Quarenta e dois números. Uma semana inteira de duas filiais cabe em quarenta e dois números.

A função é quase igual à anterior. Em vez de contar, anota o comprimento de cada caixa que abre:

```python
def forma(x):
    tamanos = []
    while type(x) == list:
        tamanos.append(len(x))
        x = x[0]
    return tuple(tamanos)
```

Troque o bloco final do arquivo por este:

```python
if __name__ == "__main__":
    print("total", contar_dimensiones(total), forma(total))
    print("por_dia", contar_dimensiones(por_dia), forma(por_dia))
    print("por_dia_y_producto", contar_dimensiones(por_dia_y_producto), forma(por_dia_y_producto))
    print("papeleria", contar_dimensiones(papeleria), forma(papeleria))

    assert forma(papeleria) == (2, 7, 3), "La papelería debería ser 2 sucursales × 7 días × 3 productos"
    print("Todo cuadra")
```

Saída:

```
total 0 ()
por_dia 1 (7,)
por_dia_y_producto 2 (7, 3)
papeleria 3 (2, 7, 3)
Todo cuadra
```

Duas coisas novas nessa saída:

- `(7,)` leva uma vírgula porque é assim que o Python escreve uma tupla de um único elemento. Sem a vírgula, `(7)` seria só o número 7.
- `()` é uma tupla vazia: o escalar não tem nenhuma direção.

E uma linha nova no código: `assert`. Quer dizer **“isto tem que ser verdade”**. É como o segurança da entrada: se for, nada acontece e o programa segue. Se não, o programa para e diz por quê. Faça o teste: troque o `(2, 7, 3)` por `(7, 2, 3)` e execute de novo:

```
Traceback (most recent call last):
  ...
AssertionError: La papelería debería ser 2 sucursales × 7 días × 3 productos
```

Volte para `(2, 7, 3)`. É assim que se trabalha em toda a série: cada exercício traz o próprio `assert`, e o programa avisa sozinho quando há um erro, sem esperar que alguém revise.

— “Todo cuadra”, tudo bate — lê dona Lupe em voz alta. — Tomara que o banco pensasse igual.

### Passo 7 — Quando uma lista de listas NÃO é um tensor

São quase dez da noite. Mateo boceja e lhe ocorre a pergunta incômoda: e se um dia alguém mandar uma aba incompleta? A função dele perceberia?

Há uma condição que não se negocia: **todas as caixas do mesmo nível têm que ter o mesmo tamanho.** Todas as abas com 7 linhas, todas as linhas com 3 colunas. Um tensor é retangular.

Veja esta lista:

```python
chueca = [[1, 2, 3], [4, 5]]
```

O Python a aceita sem problema: é uma lista de duas listas. Mas a primeira linha tem 3 elementos e a segunda, 2. Não é uma tabela, é uma tabela torta. E as nossas funções mentem descaradamente:

```
>>> contar_dimensiones(chueca)
2
>>> forma(chueca)
(2, 3)
```

`forma` diz `(2, 3)` porque só mediu a primeira linha. Como uma testemunha que viu o primeiro quarteirão e jura pela rua inteira. Quem confiar nela e pedir `chueca[1][2]` leva um erro:

```
IndexError: list index out of range
```

Por isso é preciso verificar. Para uma tabela (duas dimensões), esta função compara o comprimento de cada linha com o da primeira. Com mais dimensões a verificação é mais longa; nada de preocupação, na próxima sessão o PyTorch a faz completa:

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

As bibliotecas de IA, como o PyTorch, fazem essa verificação automaticamente: ao converter uma lista torta em um tensor normal, elas se recusam. Isso aparece na próxima sessão. (Existem tensores especiais para dados irregulares, mas nesta série eles não são usados.)

Mateo fica olhando para esse `False`. Um dado torto não avisa. Ele se esconde, até que algo quebra.

### Além de três: o endereço postal

— E se eu pedir quatro semanas, tia, em vez de uma?

— Eu te passo. Mas em troca me ajuda a arrumar as cartolinas.

Com as vendas de 4 semanas, Mateo precisaria de quatro informações para apontar um número: semana, filial, dia e produto. Quatro dimensões, forma `(4, 2, 7, 3)`.

Quatro dimensões não se desenham, mas se entendem do mesmo jeito que um endereço postal. Para chegar a um apartamento, informa-se o estado, o bairro, o prédio, o andar e o número do apartamento: cinco informações, cinco dimensões. Ninguém se assusta com isso. Cada informação escolhe uma caixa dentro da anterior, igual a cada índice de `papeleria[1][4][2]`.

Alguns tensores reais que vão aparecer pelo caminho:

| O que é | Dimensões | Forma de exemplo |
|---|---|---|
| Uma foto colorida pequena | altura × largura × cor | `(480, 640, 3)` |
| Um lote de 32 fotos | foto × altura × largura × cor | `(32, 480, 640, 3)` |
| 10 segundos de vídeo a 30 quadros por segundo | quadro × altura × largura × cor | `(300, 720, 1280, 3)` |
| Resumo diário de 1.000 clientes durante 30 dias, com 5 valores por cliente e dia | cliente × dia × valor | `(1000, 30, 5)` |

Aqui as fotos são escritas como altura × largura × cor porque é assim que as nossas listas ficam. A ordem não é universal: em muitos programas de PyTorch aparece cor × altura × largura. O importante é saber o que é cada eixo.

Esse vídeo, guardado sem compressão, são 829.440.000 números (basta multiplicar a forma). Por isso os vídeos são guardados comprimidos (um arquivo MP4 leva dentro vídeo comprimido), e por isso trabalhar com vídeo sem compressão exige muita memória.

Mateo olha a última linha da tabela: 1.000 clientes, 30 dias, 5 valores. Mais ou menos assim deve ser o que o banco tem da maquininha do Norte. E ele mal chegou a 42 números.

---

## De onde veio a palavra “tensor”?

Já passada a meia-noite, Mateo continua acordado no quarto. Não consegue parar de ler. Acontece que a ferramenta que ele vai usar contra o banco tem uma história de gente tão empacada quanto ele.

Ninguém inventou os tensores sozinho.

**1846.** O irlandês William Rowan Hamilton usa pela primeira vez a palavra *tensor*, do latim *tendere*, “esticar”. Mas para ele significava outra coisa: o tamanho de um quatérnio, um tipo de número que ele mesmo inventou. A palavra nasce, mas ainda não sabe o que vai ser.

**1898.** O alemão Woldemar Voigt lhe dá o sentido moderno. Estudava cristais, que se esticam e se deformam de modo diferente conforme a direção em que são puxados, e precisava de uma única coisa que guardasse todas essas direções ao mesmo tempo. Uma caixa para muitas direções. Soa familiar, não é?

**1900.** Os italianos Gregorio Ricci-Curbastro e Tullio Levi-Civita publicam, bem organizada, a matemática que hoje chamamos de cálculo tensorial (eles a chamavam de “cálculo diferencial absoluto”). E aqui um detalhe que parece pista de romance policial: em todo o artigo deles não aparece nem uma vez a palavra “tensor”.

**1912.** Albert Einstein volta a Zurique empacado na teoria da gravidade. As contas não fecham. Procura o amigo da universidade, o matemático Marcel Grossmann. Segundo lembrava anos depois um colega dos dois, Einstein lhe disse:

— Grossmann, me ajude, senão eu enlouqueço.

Grossmann lhe mostra o trabalho dos italianos. O de Ricci-Curbastro e Levi-Civita. O que não dizia “tensor”.

**1915.** Março. Chega uma carta para Einstein. Assinada por Levi-Civita. Diz que uma demonstração da versão da teoria que Einstein tinha então está errada.

Imagine a cena: Einstein, corrigido por carta.

As cartas se cruzam até maio. Einstein defende a prova com teimosia. As cartas vão e vêm. No fim, Einstein acaba aceitando que ela estava incompleta. Em uma carta de 2 de abril, escreveu a ele: “Nunca tive uma correspondência tão interessante”. Meses depois mudou de caminho, e em 25 de novembro apresentou as equações finais da relatividade geral, escritas com tensores.

Há uma lição aí que não é de matemática: corrigir alguém não é brigar. O “a demonstração está errada” de Levi-Civita não terminou em briga: mostrou a Einstein uma falha de verdade, meses antes das equações finais, e terminou em amizade.

*Fontes: Jeff Miller, “Earliest Known Uses of Some of the Words of Mathematics”, verbete TENSOR (Hamilton 1846, Voigt 1898, Ricci-Curbastro e Levi-Civita), no MacTutor: [mathshistory.st-andrews.ac.uk/Miller/mathword/t/](https://mathshistory.st-andrews.ac.uk/Miller/mathword/t/). O artigo de Ricci-Curbastro e Levi-Civita é “Méthodes de calcul différentiel absolu et leurs applications”, Mathematische Annalen 54, pp. 125–201; o volume tem data de 1900 e alguns catálogos o citam como de 1901. As cartas de 1915: Galina Weinstein, “Einstein the Stubborn: Correspondence between Einstein and Levi-Civita”, [arxiv.org/abs/1202.4305](https://arxiv.org/abs/1202.4305).*

Mateo apaga o abajur pensando nisso. Se a Einstein diziam “está errado”, a ele também vão dizer. Melhor que um `assert` diga hoje do que o banco em 23 de dezembro.

### Uma verdade para ninguém ser pego de surpresa

Se um dia um físico disser que um tensor “não é uma tabela de números”, ele tem razão, do lado dele. Na física e na matemática, um tensor é uma caixa de números que, além disso, obedece a regras precisas quando se giram os eixos ou se mudam as unidades. Na programação e na IA, *tensor* é só a caixa ordenada. Essa caixa pode guardar os números de um tensor da física, mas não traz por si só as regras nem o significado dele. Os programadores pegaram o nome emprestado da física porque soa melhor que “arranjo de várias dimensões”, que é o que ele é.

E uma armadilha de vocabulário: em inglês, o número de dimensões às vezes é chamado de *rank*. Não confundir com o “posto” de uma matriz, que talvez tenha aparecido em álgebra linear; são coisas diferentes.

---

👉 **Para quem é de Administração:** a parte de números de uma planilha com abas (sem cabeçalhos nem notas) já é um tensor de três dimensões, desde que todas as abas tenham as mesmas colunas e as mesmas linhas. Dona Lupe tinha isso no notebook havia anos e não sabia. Quando, em uma reunião, alguém disser “os dados entram no modelo como tensores”, pense nisso: tabelas empilhadas, todas com a mesma forma. E a regra do passo 7 vale para os relatórios: se cada filial manda a própria aba com colunas diferentes, ninguém consegue somá-las sem arrumá-las primeiro.

---

## Exercícios da sessão 1

As soluções estão no final desta edição. Tente primeiro; o `assert` diz se a resposta está certa.

**Exercício 1 — Preveja antes de executar.** Para cada dado, escreva no papel quantas dimensões ele tem, a forma e o que cada dimensão representa. Depois confira com as funções criadas.

```python
temperatura = 31
calificaciones = [[9, 8, 10], [7, 9, 8], [10, 10, 9], [6, 8, 7]]
foto = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
lote = [foto, foto]
```

Com as respostas já no papel, escreva-as nestes `assert` (troque cada `?`) e execute:

```python
assert (contar_dimensiones(temperatura), forma(temperatura)) == (?, ?)
assert (contar_dimensiones(calificaciones), forma(calificaciones)) == (?, ?)
assert (contar_dimensiones(foto), forma(foto)) == (?, ?)
assert (contar_dimensiones(lote), forma(lote)) == (?, ?)
print("Ejercicio 1 superado")
```

Dica: `calificaciones` são 4 alunos com 3 matérias cada um. `foto` é uma imagem de 2 × 2 pontos, e cada ponto traz três números: vermelho, verde e azul.

**Exercício 2 — Aponte o número.** Com a `papeleria` do passo 4, escreva o índice que dá cada valor e confira com um `assert`:

a) O que o Centro vendeu na quarta-feira em canetas.
b) O que o Norte vendeu na sexta-feira em mochilas.
c) O que o Norte vendeu no domingo em cadernos.

Exemplo de como fica um resolvido: `assert papeleria[0][0][0] == 1200` (Centro, segunda-feira, cadernos).

**Exercício 3 — Complete a função.** Esta versão de `forma` tem duas lacunas. Preencha-as sem olhar o passo 6 e confira com os `assert`:

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

**Exercício 4 — Dados próprios.** Escreva como listas aninhadas as notas de 3 alunos, em 4 matérias, em 3 provas parciais. Invente os números. Escolha a ordem das dimensões e escreva-a em um comentário. Depois:

a) Escreva o `assert` da forma.
b) Escreva o índice da segunda prova parcial da terceira matéria do primeiro aluno, e confira com um `assert` contra a nota inventada nesse lugar.

Este exercício não tem uma única resposta correta: depende da ordem escolhida. O que precisa acontecer é o `assert` passar e o comentário dizer o que é cada dimensão.

**Exercício 5 — Encontre a torta.** Uma destas tabelas não pode ser um tensor. Encontre-a no olho e confira com `es_rectangular`:

```python
a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8]]
c = [[0], [0], [0]]
```

Escreva as respostas nestes `assert` (troque cada `?` por `True` ou `False`):

```python
assert es_rectangular(a) == ?
assert es_rectangular(b) == ?
assert es_rectangular(c) == ?
print("Ejercicio 5 superado")
```

O que `forma(b)` diria? Por que não é confiável?

---

## Erros comuns (e como perceber)

| Erro | Como aparece | Como perceber | O que fazer |
|---|---|---|---|
| Contar os índices a partir de 1 | Pede-se a sexta-feira com `por_dia[5]` e vem o sábado | O número não coincide com a tabela | Os índices começam em 0: segunda é 0, sexta é 4 |
| Inverter a ordem dos índices | `papeleria[4][1][2]` em vez de `papeleria[1][4][2]` | `IndexError`, porque só há 2 filiais | A ordem dos índices é a ordem das listas: de fora para dentro |
| Confiar em `forma` com uma lista torta | `forma` diz `(2, 3)` e depois um índice dá erro | `es_rectangular` dá `False` | Corrigir os dados antes de usá-los; nunca preencher com números inventados sem avisar |
| Passar uma lista vazia para `forma` ou `contar_dimensiones` | `IndexError: list index out of range` dentro da função | Dá erro com `[]` | Uma caixa vazia não tem primeiro elemento para abrir; verificar se os dados têm algo |
| Esquecer a vírgula em `(7,)` | `assert forma(por_dia) == (7)` falha embora a forma esteja correta | O `assert` dá erro com algo que “parece certo” | Uma tupla de um elemento leva vírgula: `(7,)` |

---

## Revisão para a próxima sessão

A sessão 2 começa com estas três perguntas. Responda sem olhar a lição:

1. Como saber quantas dimensões um conjunto de dados tem, sem contar colchetes?
2. O que a forma `(2, 7, 3)` diz, e quantos números há no total?
3. Que condição uma lista de listas precisa cumprir para ser um tensor?

🔁 **Na próxima semana:** Mateo deixa as listas e pega uma ferramenta séria. É hora de instalar o PyTorch e converter a papelaria de dona Lupe no primeiro tensor de verdade. Dá para ver que o PyTorch já traz a forma embutida (lá ela se chama `shape`), e que ele se recusa a aceitar listas tortas.

Naquela noite, antes de fechar o notebook, Mateo voltou a abrir as duas abas. Centro, Norte. Centro, Norte. As mesmas colunas, os mesmos sete dias. E, no entanto, na aba do Norte havia algo que não batia. Mas para ver isso ele precisava de uma ferramenta de verdade.

---

## Soluções (não olhe antes de tentar)

**Exercício 1.**

```
temperatura      0 ()             un solo número
calificaciones   2 (4, 3)         alumno × materia
foto             3 (2, 2, 3)      renglón de puntos × punto × color
lote             4 (2, 2, 2, 3)   foto × renglón × punto × color
```

Os `assert` ficam assim: `(0, ())`, `(2, (4, 3))`, `(3, (2, 2, 3))` e `(4, (2, 2, 2, 3))`.

O `lote` acrescenta uma dimensão no início: qual foto. É assim que os modelos de imagens trabalham quase sempre: recebem lotes, e até uma única foto costuma ir como um lote de uma.

**Exercício 2.**

```python
assert papeleria[0][2][1] == 700   # a) Centro, miércoles, plumas
assert papeleria[1][4][2] == 200   # b) Norte, viernes, mochilas
assert papeleria[1][6][0] == 200   # c) Norte, domingo, cuadernos
```

**Exercício 3.** As lacunas são `len(x)` e `x[0]`: anotar o comprimento da caixa e depois abri-la.

**Exercício 4.** Uma resposta possível, com a ordem aluno × matéria × prova parcial:

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

Com a ordem matéria × aluno × prova parcial, a forma é `(4, 3, 3)` e o índice é `parciales[2][0][1]`. As duas estão certas se o comentário disser isso. Escolher a ordem das dimensões é uma decisão de design, e na sessão 3 fica claro por que ela importa.

**Exercício 5.** Os `assert` são `True`, `False` e `True`. A torta é `b`: a última linha tem 2 elementos e as demais, 3. `es_rectangular(b)` dá `False`. `forma(b)` diria `(3, 3)` porque só mede a primeira linha, e por isso não é confiável: não verifica as demais.

---

## O que vem a seguir

Na sessão 2, as listas ficam para trás e entra uma ferramenta feita para isto. Para adiantar, vale conferir se o Python instalado é o 3.11 ou mais recente com `python3 --version`.

---

*Escrevo a partir do que vemos todos os dias na Hábil, onde construímos sistemas para bancos, seguradoras e empresas no México. Se no trabalho ou no negócio da família aparecer um problema assim, vamos conversar: [habil.mx](https://www.habil.mx/pt/)*
