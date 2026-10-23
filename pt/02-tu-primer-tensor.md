# Tensores do zero (2): o primeiro tensor

*Instala-se o PyTorch, a papelaria de dona Lupe vira um tensor de verdade e descobre-se que a função programada na semana passada já vinha de fábrica.*

*Tensores do zero · lição A, sessão 2 de 8*

*Cenário: os personagens e a história são inventados. Os dados técnicos, históricos e os números reais trazem a fonte.*

**Segunda-feira, 9 de novembro, 19h40. Fundos da filial Centro.**

Cheiro de papel novo e de café requentado. Dona Lupe abaixa a porta de aço até a metade, com aquele rangido que se ouve até a esquina, e se senta ao lado de Mateo.

— Chegou o arquivo do Norte — diz, e coloca o celular na frente dele. — O da semana. É esse que me importa.

Mateo passa o arquivo para o notebook. Sete dias, três produtos: cadernos, canetas e mochilas. O de sempre. No calendário da parede, alguém circulou em vermelho o dia 23 de dezembro. Faltam quarenta e quatro dias.

— E agora, o que vai fazer com isso? De novo as listinhas?

Na semana passada, Mateo escreveu a papelaria como listas dentro de listas e programou manualmente duas funções: uma que conta dimensões e outra que calcula a forma. Funcionaram. Mas ele também descobriu algo incômodo: a função `forma` se deixa enganar por uma lista torta. E não avisa.

— As listas já não bastam, tia. Preciso de uma ferramenta de verdade.

— E quanto custa?

— Nada. É de graça.

Dona Lupe o olha por cima dos óculos.

— O barato sai caro, menino.

— Esta não. É a que o pessoal de inteligência artificial usa.

— Pior ainda.

Mateo ri e abre o terminal. Mas antes de instalar qualquer coisa, impõe a si mesmo uma regra: responder às três perguntas da semana passada sem olhar as anotações. Se não conseguir, não faz sentido seguir. Uma ferramenta nova sobre uma base frágil só faz os erros acontecerem mais rápido.

## Revisão para abrir

Antes de começar, a ideia é responder sem olhar a sessão 1. As respostas estão no final, nas soluções.

1. Como saber quantas dimensões um conjunto de dados tem, sem contar colchetes?
2. O que a forma `(2, 7, 3)` diz, e quantos números há no total?
3. Que condição uma lista de listas precisa cumprir para ser um tensor?

Se bateu dúvida em alguma, vale voltar à sessão 1 antes de seguir.

> **O código desta sessão** está no repositório público do curso, na pasta `sesion-02`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-02). Vale escrevê-lo primeiro, porque é assim que se aprende; o repositório serve para comparar quando algo não sair igual. Cada programa vem com a saída que deve produzir.

## Objetivos desta sessão

Ao terminar, será possível fazer estas cinco coisas. Ao lado de cada uma está o exercício que a demonstra.

| # | Ao terminar, é possível… | Demonstrado em |
|---|---|---|
| 1 | **Instalar** o PyTorch em um ambiente próprio e **comprovar** que funciona | Passo 1 e exercício 6 |
| 2 | **Converter** listas aninhadas em um tensor com `torch.tensor` | Exercícios 7 e 8 |
| 3 | **Ler** a forma, as dimensões e o total de números de um tensor (`.shape`, `.ndim`, `.numel()`) e **conferi-los** com um `assert` contra as funções da sessão 1 | Exercícios 6, 7 e 8 |
| 4 | **Prever** o tipo de número de um tensor (`.dtype`) e **escolhê-lo** quando for preciso | Exercícios 6 e 8 |
| 5 | **Interpretar** o erro que o PyTorch dá com uma lista torta e **dizer** onde está o problema | Exercício 9 |

**O que é preciso:** a sessão 1 terminada, com o arquivo `dimensiones.py` (as listas da papelaria e as funções `contar_dimensiones` e `forma`). Um computador com Linux Mint (ou qualquer Linux ou macOS com Python 3.11 ou mais recente; no Windows o curso não foi testado), internet para baixar o PyTorch e um terminal aberto.

---

## Por que importa

Mateo volta a olhar a função `forma`. Um `while`, um `append`, um `tuple`. Funcionou com a papelaria. Mas tem um buraco: com uma lista torta, ela mente. E mente com toda a convicção.

É isso que não o deixa tranquilo. Se o arquivo do Norte viesse errado, a função lhe daria uma forma sem pestanejar, e ele acreditaria. E se o arquivo trouxesse um milhão de números, cada conta seria mais um `for` escrito à mão.

Quem trabalha com inteligência artificial não programa essas funções toda vez. Usa uma biblioteca que já guarda os números em ordem, sabe a forma deles, verifica se são retangulares e faz as contas muito rápido. A mais usada em pesquisa se chama **PyTorch**.

Hoje se aprende a primeira coisa que qualquer pessoa que trabalha com IA faz quando chegam dados novos: **criar o tensor e perguntar a ele como ele é.** Que forma tem? Quantas dimensões? Quantos números? De que tipo? Se essas respostas não forem as esperadas, nada do que vier depois vai dar certo.

Mateo ainda não sabe, mas esta noite uma dessas perguntas vai responder “não”.

---

## Passo 1 — Instalar o PyTorch

### Um cômodo separado para as ferramentas

Antes de instalar, cria-se um **ambiente virtual**. Parece complicado e não é: é uma pasta onde ficam as bibliotecas de um projeto, separadas das do resto do computador.

Pense na caixa de ferramentas de um tio encanador: se ele guarda as chaves na gaveta da cozinha, um dia não encontra nada. Com a própria caixa, ninguém mexe em nada. O ambiente virtual é essa caixa: o que se instala ali não atrapalha o Linux Mint, que usa o próprio Python para funcionar.

No terminal, primeiro confira a versão do Python:

```bash
python3 --version
```

É preciso ter o Python 3.11 ou mais recente: é o que o PyTorch exige desde a versão 2.15. O Linux Mint 22 traz o Python 3.12, então está tudo certo.

**No Linux Mint e no Ubuntu, primeiro isto.** A ferramenta para criar ambientes vem em um pacote separado que uma instalação nova não traz. Instale-o (ele vai pedir a senha); se já estava instalado, não acontece nada:

```bash
sudo apt install python3-venv
```

Agora crie o ambiente e ative-o:

```bash
python3 -m venv ~/tensores
source ~/tensores/bin/activate
```

A primeira linha cria a caixa: uma pasta chamada `tensores` na pasta pessoal (é isso que `~` significa). A segunda a abre. Repare que o terminal agora começa com `(tensores)`. Esse é o sinal de que se está dentro.

**No Windows:** essas duas linhas são para Linux e macOS. No Windows, a partir da pasta pessoal, crie o ambiente com `python -m venv tensores` e ative-o com `.\tensores\Scripts\activate` (é o que indica a [documentação do Python](https://docs.python.org/3/library/venv.html)). Atenção: o curso não foi testado no Windows; se uma saída não coincidir, vale compará-la com a do repositório antes de supor que houve um erro.

**Se aparecer este erro:** o passo de cima foi pulado. No Ubuntu 24.04 (a mesma base do Linux Mint 22) a mensagem é assim:

```text
The virtual environment was not created successfully because ensurepip is not
available.  On Debian/Ubuntu systems, you need to install the python3-venv
package using the following command.

    apt install python3.12-venv

You may need to use sudo with that command.  After installing the python3-venv
package, recreate your virtual environment.
```

Repare que a mensagem propõe `python3.12-venv`, com o número da versão, e aqui pedimos `python3-venv`, sem número. Os dois servem: `python3-venv` instala por dentro o da versão do sistema. Preferimos o que não leva número porque continua funcionando quando o sistema mudar de versão do Python. Execute `sudo apt install python3-venv` e rode de novo as duas linhas.

Atenção: **cada terminal novo começa com o ambiente desativado.** Execute de novo `source ~/tensores/bin/activate` e procure o `(tensores)`. Sem ele, o Python não vai encontrar o PyTorch.

### O comando oficial

Com o ambiente ativado, atualize o instalador de pacotes e depois instale o PyTorch com o comando oficial da página dele para Linux, sem placa de vídeo:

```bash
pip3 install --upgrade pip
pip3 install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

O que esse comando diz, parte por parte:

- `torch==2.14.1` é o PyTorch, na versão exata com que este curso foi escrito. Ela é fixada para que as saídas coincidam com as da lição: com outra versão os números podem mudar um pouco, e não daria para saber de onde veio o erro.
- `torch` é o PyTorch. O nome vem do **Torch**, uma biblioteca anterior escrita em outra linguagem (Lua), de cuja comunidade saiu parte das pessoas que o criaram.
- `torchvision` é um pacote companheiro para imagens. Vem no comando oficial e não atrapalha.
- `--index-url https://download.pytorch.org/whl/cpu` diz ao `pip` de qual loja baixar: a da edição **só CPU**. Sem essa parte, no Linux ele baixaria a edição para placas de vídeo NVIDIA, que é muito mais pesada e não é necessária neste curso.

É um download grande; calma.

### Conferir que deu certo

Crie um arquivo `verifica.py`, na mesma pasta onde está `dimensiones.py`:

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

Execute com `python3 verifica.py`:

```
2.14.1 False
```

O primeiro valor é a versão do PyTorch. Se aparecer `2.14.1+cpu`, também está certo: o `+cpu` identifica a edição só CPU. O segundo valor pergunta se o PyTorch pode usar uma placa de vídeo da NVIDIA (o que o PyTorch chama de CUDA). Como foi instalada a edição só CPU, o `False` é exatamente o esperado, mesmo que o notebook tenha placa de vídeo.

**O que é uma GPU?** É a placa de vídeo: nasceu para renderizar milhões de pontos da tela ao mesmo tempo nos videogames. A CPU é um punhado de chefs muito habilidosos; a GPU é uma equipe de milhares de ajudantes que só sabem picar, mas picam todos ao mesmo tempo, e multiplicar tensores grandes é puro picar. Este curso inteiro roda na CPU de um notebook comum.

> Mateo vê o `False` e ri. Dona Lupe, que não tira os olhos da tela, pergunta se isso é ruim.
> — Quer dizer que o PyTorch não vai usar placa de vídeo no meu notebook, tia.
> — Ah, que bom. Assim é só estudar.

---

## Passo 2 — Do número e da lista ao tensor

> **20h25.** Mateo está a um clique do arquivo do Norte. Não o abre. Primeiro vai praticar com os dados da semana passada, os que ele já sabe que estão certos. Se algo falhar com dados bons, o erro é dele. Se falhar com dados novos, o erro vem nos dados. Misturar as duas coisas é o jeito mais rápido de não entender nada.

Há uma única função para aprender hoje para criar tensores: `torch.tensor`. Ela recebe as listas e devolve um tensor com os mesmos números.

Crie `primer_tensor.py` na mesma pasta que `dimensiones.py`. A linha `from dimensiones import ...` pega as listas escritas na semana passada, sem copiá-las de novo:

```python
import torch
from dimensiones import total, por_dia

t_total = torch.tensor(total)
t_dia = torch.tensor(por_dia)

print(t_total)
print(t_dia)
print(t_total.shape, t_dia.shape)
```

Execute com `python3 primer_tensor.py`:

```
tensor(18500)
tensor([2500, 3100, 2200, 2800, 3900, 2400, 1600])
torch.Size([]) torch.Size([7])
```

Três coisas para ler nessa saída:

- O PyTorch embrulha os números em `tensor(...)`: já não é uma lista de Python, é um tensor.
- O escalar aparece como `tensor(18500)`, sem colchetes. Zero dimensões, igual à sessão 1.
- `.shape` é a **forma**. Para o escalar sai `torch.Size([])`, vazia, porque ele não tem nenhuma direção. Para a semana sai `torch.Size([7])`: sete dias.

Repare: `.shape` não leva parênteses. Não é uma função, é um valor que o tensor já traz guardado, como a etiqueta com o tamanho de uma camisa: não se mede, se lê.

> — E daí? — diz dona Lupe, apontando o `18500`. — É o mesmo que eu já tinha anotado.
> — A graça é essa, tia. Ser o mesmo. Agora o computador também sabe.

---

## Passo 3 — A papelaria completa

Agora a planilha com as duas abas. Crie `papeleria.py` na mesma pasta; este arquivo vai guardar o tensor usado nas próximas sessões:

```python
import torch
from dimensiones import centro, norte

papeleria = torch.tensor([centro, norte])

# Esto corre al abrir papeleria.py, no cuando otra sesión lo importa.
if __name__ == "__main__":
    print(papeleria)
    print(papeleria.shape)
    print(papeleria.ndim)
    print(papeleria.numel())
    print(papeleria.dtype)
```

Saída:

```
tensor([[[1200,  800,  500],
         [1500,  900,  700],
         [1000,  700,  500],
         [1300,  900,  600],
         [2000, 1200,  700],
         [1200,  800,  400],
         [ 800,  500,  300]],

        [[ 300,  200,  100],
         [ 400,  250,  150],
         [ 350,  200,  100],
         [ 300,  150,  100],
         [ 500,  300,  200],
         [ 300,  200,  100],
         [ 200,  100,  100]]])
torch.Size([2, 7, 3])
3
42
torch.int64
```

A condição `if __name__ == "__main__":` deixa essas cinco verificações para quando este arquivo é executado diretamente. As próximas sessões poderão usar `from papeleria import papeleria` sem imprimir a tabela inteira de novo.

O PyTorch alinha as colunas e deixa uma linha em branco entre a aba Centro e a Norte: a mesma planilha com abas da semana passada. Agora, as quatro respostas:

| Pergunta | Escreve-se | Resposta | Quer dizer |
|---|---|---|---|
| Qual é a forma? | `papeleria.shape` | `torch.Size([2, 7, 3])` | 2 filiais, 7 dias, 3 produtos |
| Quantas dimensões? | `papeleria.ndim` | `3` | filial, dia e produto |
| Quantos números traz? | `papeleria.numel()` | `42` | 2 × 7 × 3 |
| De que tipo são? | `papeleria.dtype` | `torch.int64` | inteiros (visto no passo 4) |

Os nomes vêm do inglês: `shape` é “forma”; `ndim`, “número de dimensões”; `numel`, *number of elements*, “número de elementos”; `dtype`, *data type*, “tipo de dado”. Repare num detalhe: `numel()` leva parênteses, porque é uma função que conta. Os outros três não.

### `torch.Size([2, 7, 3])` é a `forma` da semana passada

A sessão 1 terminou com esta saída da função:

```
papeleria 3 (2, 7, 3)
```

E hoje o PyTorch diz `torch.Size([2, 7, 3])`. **É exatamente a mesma coisa**, com outra roupa. `torch.Size` é a maneira como o PyTorch guarda uma forma, e por dentro ela se comporta como uma tupla. Por isso dá para compará-la diretamente com `(2, 7, 3)`.

O `while` com `append` e o `.shape` fazem o mesmo trabalho. A diferença: o do PyTorch, como se vê no passo 5, não se deixa enganar por uma lista torta.

### Conferir com `assert`: primeiro a forma, depois o valor

Não se deve acreditar em ninguém, nem no PyTorch: é preciso conferir. Crie `compara.py`:

```python
import torch
from dimensiones import centro, norte, forma, contar_dimensiones

papeleria_lista = [centro, norte]
papeleria = torch.tensor([centro, norte])

# 1) Primero la forma: PyTorch contra tus funciones de la sesión 1
print(papeleria.shape, forma(papeleria_lista))
print(papeleria.ndim, contar_dimensiones(papeleria_lista))
assert papeleria.shape == (2, 7, 3), "La papelería debería ser 2 sucursales × 7 días × 3 productos"
assert papeleria.shape == forma(papeleria_lista), "PyTorch y tu función deberían dar la misma forma"
assert papeleria.ndim == contar_dimensiones(papeleria_lista), "Deberían contar las mismas dimensiones"

# 2) Después el total de números, contra un for
cuenta = 0
for pestana in papeleria_lista:
    for renglon in pestana:
        for numero in renglon:
            cuenta = cuenta + 1
print(papeleria.numel(), cuenta)
assert papeleria.numel() == cuenta, "Deberían ser los mismos números"

# 3) Al final, el valor: de regreso a listas, ¿quedó igual?
assert papeleria.tolist() == papeleria_lista, "Los datos no deberían cambiar al convertirlos"
print("Todo cuadra")
```

Saída:

```
torch.Size([2, 7, 3]) (2, 7, 3)
3 3
42 42
Todo cuadra
```

Três verificações, nesta ordem:

1. **A forma.** É o `assert` que vai ser escrito em cada exercício do curso. Se a forma está errada, todo o resto está errado.
2. **O total contra um `for`.** Os três `for` aninhados contam um por um o que `numel()` conta numa palavra. Se os dois dão 42, nada se perdeu nem sobrou.
3. **O valor.** `.tolist()` faz a viagem de volta: converte o tensor outra vez em listas. A lista se chama `papeleria_lista` para não colidir com o tensor `papeleria`; se forem iguais, os números não mudaram no caminho.

Trocando `(2, 7, 3)` por `(7, 2, 3)`, o primeiro `assert` dá erro com a mensagem dele, igual à sessão 1. A ordem importa: 2 abas de 7 linhas não é o mesmo que 7 abas de 2.

> “Todo cuadra”, tudo bate. Mateo lê duas vezes. Com os dados conhecidos, a ferramenta não erra e não o deixa errar. Já dá para confiar nela. Falta uma coisa antes do arquivo do Norte: saber que tipo de número ela vai guardar.

---

## Passo 4 — O tipo de número: `dtype`

Todos os números de um tensor são do mesmo tipo. Não vale misturar, como em uma lista de Python, um inteiro com um texto. Esse tipo único se chama **dtype**.

Por quê? Porque assim o computador sabe exatamente quanto espaço cada número ocupa e os enfileira sem conferir um por um, como caixas do mesmo tamanho em uma prateleira.

Para começar, bastam dois tipos:

| dtype | O que guarda | Exemplo |
|---|---|---|
| `torch.int64` | inteiros: peças, pesos sem centavos, contagens | 12 cadernos |
| `torch.float32` | números com decimais: medições, médias, porcentagens | média 9.7 |

O número do nome é quanto espaço cada um ocupa, em bits: 64 ou 32. Por enquanto, não é preciso mais.

Um aviso para quando houver dinheiro de verdade envolvido: `float32` guarda os decimais de forma aproximada (o 0.1 fica como 0.10000000149011612). Para valores exatos, os sistemas guardam centavos como inteiros: $35.50 é guardado como `3550`. No exemplo abaixo usamos preços com decimais só para ver qual `dtype` sai.

Crie `tipos.py`:

```python
import torch

piezas = torch.tensor([12, 30, 7])
precios = torch.tensor([35.5, 12.0, 450.0])
mezcla = torch.tensor([12, 30.5, 7])

print(piezas, piezas.dtype)
print(precios, precios.dtype)
print(mezcla, mezcla.dtype)
```

Saída:

```
tensor([12, 30,  7]) torch.int64
tensor([ 35.5000,  12.0000, 450.0000]) torch.float32
tensor([12.0000, 30.5000,  7.0000]) torch.float32
```

A regra que o PyTorch segue:

- **Se todos são inteiros, sai `int64`.** Por isso a papelaria saiu `torch.int64`: dona Lupe registra pesos sem centavos.
- **Se pelo menos um traz ponto decimal, sai `float32`,** e os inteiros também viram decimais: o 12 se converte em `12.0000`. Basta um único 30.5 para a fila inteira mudar.

> — Eu não anoto centavos — diz dona Lupe. — Centavo é coisa de banco.
> — Por isso a planilha da senhora saiu `int64`, tia. Inteira, como a senhora.
> — Está me chamando de quadrada?
> — Estou chamando de exata.

### Escolher o tipo

Às vezes se quer decimais mesmo que os dados sejam inteiros. Por exemplo, para tirar médias, como se vê na sessão 4. Para isso, `torch.tensor` aceita uma segunda informação:

```python
import torch
from dimensiones import por_dia

dia_dec = torch.tensor(por_dia, dtype=torch.float32)
print(dia_dec)
print(dia_dec.dtype)
```

Saída:

```
tensor([2500., 3100., 2200., 2800., 3900., 2400., 1600.])
torch.float32
```

O ponto depois de cada número (`2500.`) é a pista de que já são decimais. Mesmos números, outro tipo.

---

## Passo 5 — A lista torta: o PyTorch diz não

**22h15.** Agora sim. Mateo copia as sete linhas do arquivo do Norte para uma lista, escreve a linha de sempre, `torch.tensor(...)`, e aperta Enter.

O terminal se enche de letras vermelhas.

— Já quebrou? — pergunta dona Lupe.

— Não, tia. Ele se recusou.

Mateo desce pelo arquivo, linha por linha, com o dedo na tela. Segunda: três números. Terça: dois. Quarta: três. Falta um na terça.

Antes de tirar conclusões, ele reproduz o problema em miniatura. Na sessão 1 se viu que a função `forma` se deixava enganar por esta lista e era preciso conferi-la com `es_rectangular`:

```python
chueca = [[1, 2, 3], [4, 5]]
```

Vamos entregá-la ao PyTorch. Crie `chueca.py`:

```python
import torch

chueca = [[1, 2, 3], [4, 5]]
t = torch.tensor(chueca)
```

Execute:

```
Traceback (most recent call last):
  File ".../chueca.py", line 4, in <module>
    t = torch.tensor(chueca)
        ^^^^^^^^^^^^^^^^^^^^
ValueError: expected sequence of length 3 at dim 1 (got 2)
```

O mesmo vermelho do arquivo do Norte. O PyTorch se recusa a criar o tensor. O importante é a última linha, e ela se lê palavra por palavra:

| Diz | Traduzido | O que significa aqui |
|---|---|---|
| `ValueError` | erro de valor | o dado recebido não serve para isto |
| `expected sequence of length 3` | esperava uma lista de comprimento 3 | a primeira linha tinha 3, então todas deveriam ter 3 |
| `at dim 1` | na dimensão 1 | a dimensão de dentro: as linhas. A de fora é a 0 |
| `(got 2)` | recebi 2 | encontrou uma linha que tem 2 |

Repare no `dim 1`. As dimensões são numeradas a partir de 0, igual aos índices de uma lista: a de fora é a 0, a seguinte é a 1. Na papelaria, a dimensão 0 são as filiais, a 1 os dias e a 2 os produtos. Essa numeração vai ser muito usada na sessão 3.

É o `es_rectangular`, só que melhor: confere todas as caixas de todos os níveis e diz onde está o problema. Por isso o tensor criado com `torch.tensor` é sempre retangular. (O PyTorch tem tensores especiais para dados irregulares, mas neste curso eles não são usados.)

A função da semana passada teria engolido o arquivo do Norte sem dizer nada. O PyTorch não: prefere dar erro a mentir.

**Como se corrige:** não preenchendo com um número inventado. Mateo poderia pôr um zero na terça e seguir em frente. O tensor sairia perfeito, e estaria mentindo: um zero diz “não se vendeu nada”; um buraco diz “não sei”. Não é a mesma coisa. É preciso perguntar a quem mandou os dados qual valor falta. Se faltam as mochilas de terça, o tensor não se corrige com um zero; corrige-se perguntando.

> — Quem monta este arquivo, tia?
> — É a filial que manda. Toda segunda. Nunca tinha chegado errado.
> Nunca. Mateo fica olhando para a terça.

---

## Quadro: a briga de dois gigantes (PyTorch e TensorFlow)

Enquanto o PyTorch baixava, Mateo se fez uma pergunta: se esta ferramenta é tão boa, por que em todo lugar se ouve outro nome? Dois nomes vão aparecer o tempo todo, e por trás deles há uma briga entre dois gigantes. Os dados são da pesquisa do curso, com data.

**O que chegou primeiro.** O **TensorFlow** é do Google. Foi liberado como código aberto em **9 de novembro de 2015**, com permissão para qualquer pessoa usá-lo: onze anos exatos antes da noite em que Mateo instalou o rival. A versão vigente em outubro de 2026 é a **2.21.0**, publicada em 6 de março de 2026.

**O desafiante.** O **PyTorch** nasceu no Facebook (hoje Meta), no laboratório de pesquisa da empresa. As primeiras versões no GitHub são de **setembro de 2016**, dez meses depois, e o lançamento oficial, de janeiro de 2017. Desde **setembro de 2022** ele já não é de uma única empresa: quem cuida dele é a PyTorch Foundation, dentro da Linux Foundation, e os membros fundadores foram AMD, AWS (da Amazon), Google Cloud, Meta, Microsoft Azure e NVIDIA. Repare num nome dessa lista: o Google Cloud, da mesma empresa que o rival, também tem lugar ali. A versão vigente quando esta sessão foi escrita (outubro de 2026) é a **2.14.1**, publicada em 30 de setembro de 2026. Se uma mais nova foi instalada, está tudo bem.

**Quem está ganhando?** Depende do ringue.

- **Nos modelos que as pessoas publicam, o PyTorch, de longe.** No Hugging Face, o site onde as pessoas publicam modelos de IA, em 18 de outubro de 2026 havia 258.777 modelos com a tag do PyTorch versus 17.469 com a do TensorFlow: uns 15 para cada um. Isso mede o que as pessoas publicam, não o que usam no trabalho, mas a diferença é clara. E a biblioteca de modelos mais popular do Hugging Face, a Transformers, anunciou que na versão 5 deixa o TensorFlow e fica só com o PyTorch.
- **Entre programadores, quase empate.** Na pesquisa do Stack Overflow de 2024, 10.6 % de quem respondeu disse usar PyTorch e 10.1 %, TensorFlow. A pesquisa pergunta o que cada pessoa usa, não o que roda em produção. O TensorFlow chegou antes às empresas e continua vivo em muitas, e nos celulares.

*Fontes: [anúncio do Google sobre o TensorFlow (2015)](https://blog.google/innovation-and-ai/products/tensorflow-smarter-machine-learning-for/); versões no PyPI: [tensorflow](https://pypi.org/project/tensorflow/) e [torch](https://pypi.org/project/torch/); versões do PyTorch no [GitHub](https://github.com/pytorch/pytorch/releases); [Linux Foundation, 12 de setembro de 2022](https://www.linuxfoundation.org/press/press-release/meta-transitions-pytorch-to-the-linux-foundation); [modelos do PyTorch no Hugging Face](https://huggingface.co/models?library=pytorch), consultado em 18 de outubro de 2026 (o número muda todo dia); [pesquisa do Stack Overflow 2024](https://survey.stackoverflow.co/2024/technology).*

**Uma diferença que aparece para quem um dia usar os dois:** com os mesmos inteiros, o PyTorch cria `int64` e o TensorFlow, `int32`.

| | PyTorch | TensorFlow |
|---|---|---|
| Criar um tensor | `torch.tensor([1, 2, 3])` | `tf.constant([1, 2, 3])` |
| Inteiro padrão | `int64` | `int32` |
| Com decimais | `float32` | `float32` |

**Qual usar?** Para aprender e ler código de pesquisa, o PyTorch. Se um dia o trabalho usar TensorFlow, aprende-se em uma tarde: forma, dimensões e tipo funcionam igual.

---

👉 **Para quem é de Administração:** o `dtype` é o formato dado a uma coluna na planilha: “número inteiro” ou “moeda com duas casas decimais”. E o erro da lista torta é o relatório da filial que chega com uma coluna a menos: não se soma com os outros até alguém perguntar o que falta. Quando um analista disser “o modelo não rodou porque os dados não tinham a forma correta”, já se sabe do que ele está falando.

---

## Exercícios da sessão 2

As soluções estão no final desta edição. O trabalho é na mesma pasta do `dimensiones.py` e com o ambiente ativado.

**Exercício 6 — Preveja antes de executar.** Escreva no papel a forma, as dimensões, o tipo e o total de números de cada tensor. Depois execute e compare.

```python
import torch

temperatura = torch.tensor(31)
calificaciones = torch.tensor([[9, 8, 10], [7, 9, 8], [10, 10, 9], [6, 8, 7]])
foto = torch.tensor([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])
promedios = torch.tensor([9.0, 8.0, 9.7, 7.0])

for nombre, t in [("temperatura", temperatura), ("calificaciones", calificaciones),
                  ("foto", foto), ("promedios", promedios)]:
    print(nombre, t.shape, t.ndim, t.dtype, t.numel())
```

São os mesmos dados do exercício 1 da sessão 1, mais um com decimais. Dica: quantos números um escalar traz?

**Exercício 7 — Complete as lacunas.** A aba Norte sozinha, como tensor. Preencha os `____` e execute. Se o `print` final aparecer, acertou.

```python
import torch
from dimensiones import norte

norte_t = torch.____(norte)

assert norte_t.shape == ____, "Norte es una pestaña: 7 días × 3 productos"
assert norte_t.ndim == ____, "Una pestaña es una tabla"
assert norte_t.numel() == ____, "7 × 3"
assert norte_t.dtype == ____, "Son pesos enteros"
assert norte_t.tolist() == norte, "Los datos no deberían cambiar"
print("Ejercicio 7 superado")
```

**Exercício 8 — Dados próprios, agora em tensor.** Pegue as notas escritas no exercício 4 da sessão 1 (alunos, matérias e provas parciais) e:

a) Converta-as em tensor e escreva o `assert` da forma contra a função `forma`.
b) Escreva o `assert` das dimensões contra `contar_dimensiones`.
c) Conte os números com `for` aninhados e compare com um `assert` contra `numel()`.
d) Crie uma segunda versão com `dtype=torch.float32` e confira com um `assert` que o tipo mudou, mas a forma não.

Termine com `print("Ejercicio 8 superado")`.

**Exercício 9 — Encontre a torta, agora com o PyTorch.** Antes de executar, preveja quais dão erro e que número de `dim` o erro vai dizer. Depois teste cada uma separadamente com `torch.tensor(...)`.

```python
a = [[1, 2], [3, 4], [5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8]]
c = [[[1, 2], [3, 4]], [[5, 6], [7]]]
d = [[1, 2], [3, 4, 5]]
```

Para as que dão erro, responda: por que em `d` o erro diz que esperava 2 e recebeu 3, se a linha que “sobra” é a segunda? E corrija `b` acrescentando o número que falta (suponha que seja um 9).

---

## Erros comuns (e como perceber)

| Erro | Como aparece | Como perceber | O que fazer |
|---|---|---|---|
| Esquecer de ativar o ambiente | `ModuleNotFoundError: No module named 'torch'` | O terminal não começa com `(tensores)` | Execute `source ~/tensores/bin/activate` em cada terminal novo |
| Chamar o arquivo de `torch.py` | `AttributeError: partially initialized module 'torch' has no attribute 'tensor' (most likely due to a circular import)` | Dá erro na primeira linha que usa `torch.` | O Python importou o arquivo local em vez do PyTorch. Mude o nome, por exemplo para `primer_tensor.py` |
| Pôr parênteses em `.shape` | `TypeError: 'torch.Size' object is not callable` | Foi escrito `t.shape()` | `.shape`, `.ndim` e `.dtype` vão sem parênteses; só `.numel()` os leva |
| Esquecer os parênteses de `numel` | Aparece algo como `<built-in method numel of Tensor object at 0x...>` em vez de um número | O `print` não mostra um número | Escreva `t.numel()`, com parênteses |
| Escrever `torch.Tensor` com maiúscula | `torch.Tensor([1, 2, 3])` dá `tensor([1., 2., 3.])`, decimais | O `dtype` sai `torch.float32` com dados inteiros | Use `torch.tensor`, com minúscula: respeita o tipo dos dados |

---

## Revisão para a próxima sessão

A sessão 3 começa com estas três perguntas. Responda sem olhar a lição:

1. O que `t.shape` diz e como conferir isso com um `assert`?
2. Por que `torch.tensor([[1, 2, 3], [4, 5]])` falha?
3. Que `dtype` sai de `torch.tensor([1, 2, 3])` e qual sai de `torch.tensor([1.0, 2.0])`?

🔁 **Na próxima semana:** a papelaria já é um tensor. Agora é hora de extrair pedaços dela por qualquer um dos eixos: só a aba Norte, só as sextas-feiras das duas filiais, só as mochilas da semana inteira. E dá para ver que uma foto se recorta exatamente do mesmo jeito.

Já passa das onze quando Mateo fecha o notebook. Dona Lupe já contou o caixa e espera junto à porta de aço com as chaves na mão. No celular continua aberto o arquivo do Norte, com a terça de dois números. Não parece um erro de digitação qualquer: é a única linha torta em semanas de arquivos certinhos. E chegou justo na semana em que o banco suspendeu a maquininha.

Quem deixa um dia pela metade?

---

## Soluções (não olhe antes de tentar)

**Revisão para abrir.**

1. Conta-se quantas informações são necessárias para apontar um único número. Na papelaria são necessárias a filial, o dia e o produto: três informações, três dimensões. E, para não contar manualmente, `contar_dimensiones` conta (e, a partir de hoje, `.ndim`).
2. Que há 2 elementos na primeira dimensão (filiais), 7 na segunda (dias) e 3 na terceira (produtos). No total, 2 × 7 × 3 = 42 números.
3. Ser retangular: todas as caixas do mesmo nível têm que ter o mesmo tamanho. Todas as abas com 7 linhas, todas as linhas com 3 colunas.

**Exercício 6.**

```
temperatura torch.Size([]) 0 torch.int64 1
calificaciones torch.Size([4, 3]) 2 torch.int64 12
foto torch.Size([2, 2, 3]) 3 torch.int64 12
promedios torch.Size([4]) 1 torch.float32 4
```

A armadilha era o escalar: a forma dele é vazia e ele tem zero dimensões, mas traz **um** número. `promedios` sai `float32` porque traz decimais. Repare também que `calificaciones` e `foto` trazem os mesmos 12 números com formas diferentes: o total não diz a forma, mas a forma diz o total.

**Exercício 7.**

```python
norte_t = torch.tensor(norte)

assert norte_t.shape == (7, 3), "Norte es una pestaña: 7 días × 3 productos"
assert norte_t.ndim == 2, "Una pestaña es una tabla"
assert norte_t.numel() == 21, "7 × 3"
assert norte_t.dtype == torch.int64, "Son pesos enteros"
```

Saída:

```
Ejercicio 7 superado
```

**Exercício 8.** Uma resposta possível, com a lista `parciales` (aluno × matéria × prova parcial) da solução do exercício 4 da sessão 1:

```python
import torch
from dimensiones import forma, contar_dimensiones

# alumno × materia × parcial
parciales = [
    [[8, 9, 9], [7, 8, 9], [9, 9, 10], [6, 7, 8]],
    [[10, 9, 10], [9, 9, 9], [8, 8, 9], [9, 10, 10]],
    [[7, 7, 8], [8, 8, 8], [6, 7, 7], [8, 9, 9]],
]

p = torch.tensor(parciales)
print(p.shape, p.ndim, p.numel(), p.dtype)
assert p.shape == forma(parciales), "PyTorch y tu función deberían coincidir"
assert p.ndim == contar_dimensiones(parciales)

cuenta = 0
for alumno in parciales:
    for materia in alumno:
        for calificacion in materia:
            cuenta = cuenta + 1
assert p.numel() == cuenta

p_dec = torch.tensor(parciales, dtype=torch.float32)
print(p_dec.dtype)
assert p_dec.dtype == torch.float32
assert p_dec.shape == p.shape, "Cambiar el dtype no cambia la forma"
print("Ejercicio 8 superado")
```

Saída:

```
torch.Size([3, 4, 3]) 3 36 torch.int64
torch.float32
Ejercicio 8 superado
```

Se os dados tiverem outra ordem ou outros tamanhos, a forma vai ser diferente, e está tudo bem: o que importa é os `assert` passarem.

**Exercício 9.**

- `a` funciona: `torch.Size([3, 2])`.
- `b` dá erro: `ValueError: expected sequence of length 3 at dim 1 (got 2)`. A última linha tem 2 e as demais, 3.
- `c` dá erro: `ValueError: expected sequence of length 2 at dim 2 (got 1)`. Aqui o problema está mais para dentro, na dimensão 2: o último par, `[7]`, só traz um número.
- `d` dá erro: `ValueError: expected sequence of length 2 at dim 1 (got 3)`. O PyTorch toma a **primeira** linha como regra. A primeira tem 2, então a segunda, que tem 3, é a que “está errada”, mesmo que pareça que falta um na primeira. Por isso o erro diz onde a regra foi quebrada, mas decidir qual dado é o correto cabe a quem analisa, junto com quem mandou os dados.

`b` corrigido:

```python
b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(torch.tensor(b).shape)
```

```
torch.Size([3, 3])
```

---

## O que vem a seguir

Na sessão 3, deixa-se de olhar o tensor inteiro e começa-se a extrair pedaços dele. Para adiantar, vale abrir o `papeleria.py` e deixar pronto o tensor `papeleria`: é a primeira coisa que vai ser usada.

---

*Escrevo a partir do que vemos todos os dias na Hábil, onde construímos sistemas para bancos, seguradoras e empresas no México. Se no trabalho ou no negócio da família aparecer um problema assim, vamos conversar: [habil.mx](https://www.habil.mx/pt/)*
