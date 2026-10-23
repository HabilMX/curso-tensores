# Tensores do zero (7): um detector de fraude que aprende com os dados

*O programa recebe 4.000 compras com cartão e a resposta de cada uma, mas ninguém escreve a regra para ele. Em menos de um segundo, ele aprende boa parte dela. E depois fica claro por que acertar 91 % das vezes pode esconder que mais de 1 em cada 3 fraudes escapa.*

*Tensores do zero · lição B, sessão 7 de 8*

*Cenário: os personagens e a história são inventados. Os dados técnicos, históricos e os números reais trazem a fonte.*

**Segunda-feira, 14 de dezembro, 20h18. Papelaria Herrera, filial Centro.**

A porta de enrolar já está abaixada e ainda treme um pouco, como se não quisesse ficar quieta. A lâmpada do depósito zumbe. Há cheiro de papel novo, de café requentado e da canela de um pão que dona Lupe esqueceu ao lado do caixa. Lá fora passa um ônibus e o vidro da porta vibra.

Mateo está com o notebook aberto sobre uma caixa de cartolinas. Ao lado, duas folhas impressas. Em uma, as vendas que de fato saíram do Norte: cadernos, canetas, uma ou outra mochila. Na outra, as cobranças que o banco mandou. Algumas chegaram de madrugada. Outras são altas. Outras aparecem muito longe da filial, em lugares onde a papelaria não vendeu nem um clipe.

— Esta das três da manhã não pode ser uma mochila — diz Mateo, e a circula com caneta vermelha.

— Nessa hora eu só vendo sono — responde dona Lupe. — E barato, porque ninguém compra.

Uma semana atrás, multiplicaram peças por preços e a conta não fechou: há cobranças sem mercadoria por trás. A Fer não as inventou; ela tinha avisado das faltas em setembro e ninguém leu o e-mail dela. O problema não estava no estoque. Estava na maquininha que aquele homem de camiseta azul trocou. Um leitor escondido pode ter copiado cartões no Norte. Depois, alguém os usou em outros lugares.

Mas uma suspeita não basta para o banco. O doutor Ortega disse pelo telefone, sem rodeios: “Senhora, preciso saber quais cobranças são mesmo da senhora. Não um palpite. Quais”.

E aí está o nó. Uma compra cara não é uma fraude: tem gente que junta o ano inteiro para comprar a televisão. Uma compra de madrugada também não: há farmácias abertas e gente com urgências. Uma cobrança distante pode ser alguém de férias. Nenhuma pista sozinha condena ninguém. O que pesa é quando elas se juntam.

Mateo tenta revisar a folha do banco linha por linha e desiste na quadragésima. Desse jeito não termina nem em janeiro.

No calendário da parede, o que a loja de ferragens da frente deu de brinde, alguém circulou o 23 em vermelho. Faltam nove dias.

— Não vou escrever uma regra para cada caso, tia — diz Mateo. — Vou dar milhares de exemplos ao computador: quanto custou, a que horas, a que distância e se o estabelecimento era de outro país. Que ele encontre o padrão.

— E se o computador errar?

Mateo fica olhando o círculo vermelho do calendário.

— Então ele vai dizer por onde continuar procurando. Mas não vai decidir sozinho.

Na sessão anterior, os tensores foram operados inteiros de uma vez: peças por preço de uma semana inteira de uma vez, com broadcasting e com `@`. Hoje essas duas ferramentas vão trabalhar sozinhas: é hora de ver um programa **aprender**.

É a sessão mais difícil do curso, então vamos devagar. Se algo não ficar claro na primeira vez, é normal: vale ler de novo com o código rodando na frente.

## Revisão para abrir

A ideia é responder sem olhar a sessão 6. As respostas estão nas soluções, no final.

1. Quando duas formas podem ser “esticadas” para operar juntas?
2. Que diferença há entre `*` e `@`?
3. Que forma sai de uma tabela `(7, 3) @` um vetor `(3,)`?

> **O código desta sessão** está no repositório público do curso, na pasta `sesion-07`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-07). Vale escrevê-lo primeiro, porque é assim que se aprende; o repositório serve para comparar quando algo não sair igual. Cada programa vem com a saída que deve produzir.

## Objetivos desta sessão

| # | Ao terminar, é possível… | Demonstrado em |
|---|---|---|
| 1 | **Nomear** em ordem os cinco passos do ciclo de treinamento (prever, medir a perda, apagar as inclinações antigas, calcular a inclinação, girar os botões) e **dizer** qual linha de código faz cada um | Exercício 26 |
| 2 | **Prever** o que acontece com a perda ao mudar o número de épocas ou a taxa de aprendizado, e **comprovar** executando o modelo | Exercícios 26 e 27 |
| 3 | **Calcular** com tensores a acurácia, o recall e a precisão de um detector, e **explicar** por que a acurácia sozinha engana | Exercício 28 |
| 4 | **Relacionar** o que o modelo aprendeu com a regra escondida que gerou os dados | Exercício 29 |

**O que é preciso:** o mesmo ambiente da sessão 2 (Python 3 com PyTorch no venv) e tudo das sessões 1 a 6: criar tensores, `shape`, `dtype`, indexar, reduzir com `sum` e `mean`, `reshape`, broadcasting e `@`. Hoje não se instala nada.

Também é preciso reconhecer três peças de Python: `for` repete as linhas indentadas; `def nome(X):` dá nome a uma receita reutilizável; e `with ...:` agrupa linhas que são executadas sob uma mesma regra. Ainda não é preciso dominá-las: basta copiar os blocos completos e mudar só o que cada exercício indicar.

---

## Por que importa

Até hoje, a regra era escrita à mão: some este eixo, multiplique estas duas tabelas. Mateo tem outro tipo de problema. Ele vê milhares de cobranças e sabe quais acabaram sendo fraude, mas não tem uma frase curta que explique todos os casos.

Cada cobrança traz pistas. Valor. Hora. Distância. Estabelecimento estrangeiro. Uma pista sozinha não condena ninguém. Várias juntas podem pedir uma revisão. O trabalho do programa é encontrar essa combinação em exemplos antigos, não adivinhar a história de uma pessoa.

Pense em como se aprende a saber se uma feira livre é boa. Ninguém dá um manual. A gente vai muitas vezes. Vê onde há fila, qual barraca dá a fruta madura e qual a esconde embaixo. Com o tempo, sem conseguir explicar numa frase, já se sabe de qual barraca não chegar perto. Isso foi aprender com exemplos.

Ou pense na avó com o feijão. Ela não mede nada. Prova, falta sal, põe uma pitada, prova de novo. Nunca passaram a fórmula exata para ela: ela a encontrou à força de corrigir. O computador vai fazer exatamente isso, mas com números e milhares de vezes por segundo.

Então a coisa se inverte: **em vez de escrever a regra, o programa recebe exemplos e busca sozinho uma regra útil.** Isso é treinar um modelo. Está por trás de muito do que hoje chamamos de inteligência artificial.

O bom é que não aparece mágica nova. Treinar é **pura conta com tensores**. Os dados são um tensor. Os botões que o modelo aprende são tensores. A previsão usa `@`. E, no final, Mateo ainda precisa comparar, perguntar e reunir provas para que alguém revise o caso.

> **No caderno de Mateo:** um modelo não prova que uma cobrança foi fraude. Só organiza milhares de pistas e aponta quais vale a pena revisar primeiro.

---

## Passo 1 — A ideia em uma frase

Mateo começa pequeno. Se entender um botão, consegue entender quarenta e nove. Se a perda cair no exemplo pequeno, já sabe o que procurar quando treinarem as compras.

Treinar um modelo é isto, repetido muitas vezes:

1. **Prever:** o modelo dá um palpite.
2. **Medir:** compara-se o palpite com a resposta correta e tira-se um número que diz o quanto ele foi mal.
3. **Apagar as inclinações antigas:** limpa-se a conta da volta anterior (no Passo 2 fica claro por que isso é necessário).
4. **Calcular a inclinação:** descobre-se para onde mover cada botão para que ele vá um pouquinho menos mal.
5. **Girar os botões:** eles são movidos um passinho nessa direção.

E de volta ao começo: o `for` repete esses cinco passos.

Como os lances livres na quadra: arremessa, vê que a bola foi para a esquerda, corrige um pouco para a direita e arremessa de novo. Ninguém deu a fórmula do arremesso perfeito: ela foi encontrada à base de correção.

Antes de ver o detector de fraude, que tem 49 botões, vamos fazer o ciclo completo com **um só**.

> **20h24** — Cinco passos e pronto? — Mateo franze a boca. — Parece fácil demais. Se fosse assim, qualquer um faria inteligência artificial.
>
> — Qualquer um faz tortilha, filho — diz dona Lupe sem tirar os olhos das moedas. — Fazer bem é outra coisa. A receita é curta; o longo é a prática.
>
> Mateo ri, mas fica pensando. Ela tem razão. A receita do ciclo cabe num guardanapo. O difícil vai ser confiar nela quando os botões não forem um, mas quarenta e nove, e ele não puder ver com os próprios olhos para onde se movem. Por isso começa com um. Se o vir funcionar em pequeno, vai acreditar nele em grande.

## Passo 2 — Um só botão: o preço de um caderno

Dona Lupe perdeu a lista de preços. A única coisa de que ela se lembra é que um cliente levou 3 cadernos e pagou $60. Quanto custa um caderno?

De cabeça, sai 20. Mas vamos deixar o PyTorch descobrir isso “às cegas”, com o ciclo de 5 passos. Crie `cerro.py`:

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

Execute com `python3 cerro.py`:

```
paso 1: precio   0.00  pérdida  3600.00  pendiente  -360.00
paso 2: precio  18.00  pérdida    36.00  pendiente   -36.00
paso 3: precio  19.80  pérdida     0.36  pendiente    -3.60
paso 4: precio  19.98  pérdida     0.00  pendiente    -0.36
paso 5: precio  20.00  pérdida     0.00  pendiente    -0.04
precio aprendido: 20.0
```

Em cinco passos chegou a $20 sem que ninguém dissesse a resposta. Vamos linha por linha, porque aqui está a sessão inteira em miniatura.

**O botão.** `precio` é o número que o modelo pode mover. Começa em 0. O `requires_grad=True` diz ao PyTorch: “anote tudo o que for feito com este número, porque depois vou pedir a inclinação dele”. É ligar o gravador antes da reunião.

Por que “botão”? Pense no fogão. O botão não sabe cozinhar. Só aumenta ou diminui a chama. Quem decide quanto girá-lo é quem prova a comida. Aqui acontece o mesmo: `precio` não sabe nada de cadernos. É um número que pode ser movido um pouquinho para um lado ou para o outro. Quem decide quanto movê-lo é o ciclo.

**A previsão.** `cobro = precio * 3`: com o preço em que acredita agora, quanto cobraria por 3 cadernos? No passo 1, $0.

**A perda.** `(cobro - 60) ** 2`: a diferença entre o que previu e o que aconteceu de verdade, ao quadrado. O quadrado faz com que a perda nunca seja negativa (passar $10 ou ficar $10 abaixo custa igual) e com que os erros grandes pesem muito mais. No passo 1 vale 3.600: ele foi péssimo. **A perda é uma nota ao contrário: quanto mais baixa, melhor.** Zero é perfeito.

É como no golfe, onde ganha quem dá menos tacadas. Ou como as multas de trânsito: quanto menos, melhor foi o comportamento. A perda não diz *o que* foi feito errado. Só diz *quanto*. Com isso basta para começar a corrigir.

**A inclinação.** Imagine estar parado num morro **de olhos vendados**, querendo chegar ao fundo do vale. Não se vê nada, então o chão é tateado com o pé: se para a frente desce, dá-se um passo para a frente; se sobe, volta-se. Isso é a **inclinação** (nos livros, **gradiente**): para que lado o chão desce e quão íngreme ele é. O morro é a perda, o lugar onde se está parado é o valor do botão, e o fundo é a menor perda possível.

Vale fazer isso de verdade, nem que seja com a imaginação. É noite, acabou a luz no bairro inteiro e é preciso descer a rua íngreme de casa sem lanterna. Não dá para ver a esquina. Mas os pés sentem a inclinação. Onde parece mais íngreme, o passo é mais longo. Onde o chão já está quase plano, os passos são curtinhos, porque já se chegou ou quase. Ninguém viu o mapa da rua. Bastou sentir o chão, passo a passo. É isso que o computador faz: não vê a forma completa do morro, só sente a inclinação onde está parado.

`perdida.backward()` é tatear o chão. O PyTorch revisa o que o gravador anotou (multiplicou-se por 3, subtraiu-se 60, elevou-se ao quadrado) e calcula a inclinação sozinho. Isso se chama **autograd**: inclinações automáticas. O resultado fica em `precio.grad`.

Desconfiado? Para isso existe o `assert` com `a_mano`: com o cálculo do ensino médio, a inclinação é `18 × precio − 360`, e o `assert` confere em cada passo que o PyTorch diz o mesmo. Opcional, não precisa decorar: ao derivar `(3 × precio − 60)²` sai `2 × (3 × precio − 60) × 3`, que se simplifica para `18 × precio − 360`. Não é preciso saber derivar; essa linha só verifica que o PyTorch fez a mesma conta, para mostrar que não há truque.

E esse `assert` é uma revisão automática: `assert condição, "mensagem"` não imprime nada se a condição for verdadeira. Se for falsa, interrompe o programa e mostra a mensagem. Enquanto se aprende, não vale apagar os `assert`: eles são o semáforo.

**O passo.** `precio -= tasa * precio.grad`. A inclinação é −360: negativa quer dizer que o chão desce se o preço subir. Subtrair um negativo é somar: 0 − 0.05 × (−360) = 18. O preço salta para $18. A `tasa` (*learning rate*, taxa de aprendizado) é **o tamanho de cada passo**: curto demais e se leva uma eternidade; longo demais e se passa para o outro lado do vale. É como estacionar de ré numa vaga apertada: girando o volante aos trancos, sobe-se na calçada; girando milímetro por milímetro, o carro de trás começa a buzinar. O `with torch.no_grad():` desliga o gravador enquanto o botão é girado, porque ajustar não faz parte da conta.

**Apagar a inclinação.** O PyTorch **soma** cada inclinação nova à que já tinha. Se ela não for apagada com `precio.grad.zero_()`, o passo seguinte usaria a antiga mais a nova e passaria do ponto.

Repare: a perda passa de 3.600 para 36 e para 0.36. Perto do fundo, a inclinação fica pequenininha, então os passos encolhem sozinhos. Ninguém disse “freie”: o morro disse.

> **20h36** Dona Lupe espia por cima do ombro de Mateo e vê o preço chegar a 20.
>
> — Vinte pesos. Isso eu dizia sem computador.
>
> — Eu sei, tia. Mas ele não sabia. Começou do zero, sem ideia, e chegou sozinho.
>
> — Igual ao teu primo com a bicicleta. Caiu umas vinte vezes e na vigésima primeira já ia sozinho.
>
> Mateo ri e olha de novo as cinco linhas. Ele não encontrou uma resposta escondida. Corrigiu uma resposta ruim até ela deixar de ser ruim. E entende algo que não esperava: a primeira tentativa não importa. Pode começar péssima. O que importa é que cada volta seja um pouquinho menos péssima. O que vem agora é o mesmo, com 4.000 compras e 49 botões.

## Passo 3 — Os dados: 4.000 compras inventadas

As compras são **inventadas**: não há dados pessoais de ninguém, funciona sem internet e, sobretudo, **sabemos a regra que decide o que é fraude**, então dá para conferir se o modelo a descobriu. São o ensaio de Mateo antes de mexer de novo nas cobranças do Norte.

De cada compra guardamos quatro dados: **valor** (de $0 até quase $20.000), **hora** (de 0 a 23), **distância** da casa do cliente (de 0 até quase 500 km) e se o estabelecimento é **estrangeiro** (1 = sim, 0 = não; 1 em cada 10).

E esta é **a regra escondida**, que só nós conhecemos. Cada compra acende “luzes vermelhas”:

- uma luz se o valor passa de $15.000,
- uma luz se a compra é de madrugada (antes das 5),
- uma luz se é a mais de 400 km de casa,
- uma luz se o estabelecimento é estrangeiro.

**Se uma compra acende duas luzes ou mais, é fraude.** Uma só não basta: comprar uma televisão cara ao meio-dia é normal; comprá-la às 3 da manhã já cheira estranho.

O modelo **nunca** vê esta regra escrita: só recebe os quatro dados de cada compra e, enquanto estuda, a resposta correta (fraude ou não). Este arquivo faz dois trabalhos. Até aqui, **nós** fabricamos o laboratório com `focos`; a partir de `predecir(X_ent)`, o modelo só recebe `X_ent`. `focos` nunca entra em `predecir`, e `y_ent` só serve para medir o quanto ele previu mal. Em dados reais, `y` viria de casos já investigados.

Crie `detector.py`. Ele vai por partes. Copie os blocos 1 a 7 no mesmo arquivo, na ordem em que aparecem (no código eles vêm numerados com comentários). Esta é a primeira:

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

Funções novas, todas simples:

- `torch.rand(n)` dá `n` números aleatórios entre 0 e 1; vezes 20.000, valores de 0 a 20.000 (broadcasting, como na sessão 6). `torch.randint(0, 24, (n,))` dá inteiros aleatórios de 0 a 23.
- `torch.manual_seed(42)` fixa o “acaso”. O computador não joga dados: segue uma receita, e com a mesma semente dá os mesmos números toda vez que é executado no mesmo computador.
- `(monto > 15_000)` dá um `True` ou `False` por compra; `.float()` o transforma em 1.0 ou 0.0. Somar quatro desses conta as luzes das 4.000 compras de uma vez.
- `torch.stack([...], dim=1)` cola as quatro listas como **colunas** de uma tabela.

Dividir cada coluna (`monto / 20_000`, `hora / 23`…) põe tudo entre 0 e 1, para que o valor (milhares) não esmague a hora (dezenas) só por ser um número maior.

A saída desta parte:

```
X: (4000, 4) | y: (4000,) | % fraude: 15.8
```

**`X` tem forma `(4000, 4)`: compras × dados de cada compra.** Uma planilha de 4.000 linhas e 4 colunas. O primeiro eixo se chama **lote** (quantos exemplos) e o segundo, **características** (o que se sabe de cada um). Essa forma vai aparecer em todo lugar.

**`y` tem forma `(4000,)`:** uma resposta por compra. Elas se chamam **rótulos** (*labels*): a coluna de respostas corretas da prova. 15.8 % das compras são fraude.

Para ver de perto, acrescente `print(X[:3])` e `print(y[:3])` logo depois desse `print`. São uma inspeção temporária: execute-os uma vez aqui e apague-os antes de comparar a saída com a “Saída completa” do Passo 8:

```
tensor([[0.8823, 0.1304, 0.4309, 0.0000],
        [0.9150, 0.6087, 0.3986, 0.0000],
        [0.3829, 0.4348, 0.1907, 0.0000]])
tensor([1., 0., 0.])
```

A compra 0 é de $17.645 (0.8823 × 20.000) às 3 da manhã (0.1304 × 23): duas luzes, fraude. A compra 1 é mais cara, $18.300, mas às 2 da tarde: uma luz só, normal. Esse detalhe é o que o modelo tem que descobrir.

> **20h52** — Pronto — diz Mateo, muito seguro. — Tudo acima de quinze mil é fraude. Marco e acabou.
>
> Dona Lupe para de contar moedas.
>
> — E a senhora que veio em agosto pagar o material escolar dos quatro filhos de uma vez? Também é ladra?
>
> Mateo abre a boca e fecha de novo. Olha a compra 1 na tela: $18.300, às duas da tarde, normal. A regra dele a teria acusado. **A pista não mora numa coluna só.** Nem o valor sozinho, nem a hora sozinha. O estranho é quando se juntam.
>
> — Uma andorinha só não faz verão — arremata dona Lupe, e volta às moedas.
>
> Mateo risca a regra. Por isso não vai ser ele quem a escreve.

## Passo 4 — Estudar e fazer a prova

Ao avaliar um aluno com as mesmas perguntas que ele estudou, não se sabe se ele aprendeu ou decorou. Com os modelos acontece o mesmo:

```python
# 4) Estudia con 3,200 compras; el examen es con las otras 800
X_ent, y_ent = X[:3200], y[:3200]
X_exa, y_exa = X[3200:], y[3200:]
```

É fatiar pelo primeiro eixo, como na sessão 3. As últimas 800 compras ficam guardadas a sete chaves: o modelo nunca as vê enquanto aprende.

Mateo chama as primeiras 3.200 de “estudo” e as outras 800 de “prova”. Não é enfeite. Se mostrasse ao modelo as respostas da prova, o número final soaria bonito e não diria nada.

É a diferença entre decorar a apostila e entender a matéria. Se a prova traz exatamente as perguntas da apostila, tira-se dez mesmo sem saber nada. Se o professor muda os números, dá para ver quem entendeu. As 800 compras separadas são a prova com os números trocados.

Um detalhe honesto: nos exercícios 26 e 27, as épocas e as taxas vão ser comparadas olhando estas mesmas 800. Assim que se escolhe com elas, elas já não são uma prova limpa, e sim um treino para decidir. Num projeto real separa-se um terceiro grupo, a prova final, que ninguém consulta até o último dia.

> **21h05** O celular de dona Lupe vibra sobre o balcão. É a Fer, do Norte: “Já fechei. Hoje foi tudo em dinheiro. Três clientes de sempre foram embora sem comprar porque não tinham. Mando foto das notas do dia.”
>
> Dona Lupe lê a mensagem duas vezes e a mostra a Mateo.
>
> — E eu que há um mês não queria nem dizer o nome dela — diz em voz baixa. — A gente vê chifre em cabeça de cavalo.
>
> Mateo não responde. Guarda a foto das notas numa pasta nova. Se o modelo disser algo sobre as cobranças do Norte, vai precisar de algo com que comparar. Dados de verdade, separados, que o modelo nunca tenha visto. Igual à prova.

## Passo 5 — Os botões e a previsão com `@`

Agora os botões. Em `cerro.py` havia um. Aqui há quatro tensores de botões:

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

Nos livros, esses botões se chamam **pesos** (o `W`, de *weights*) e **vieses** (o `b`, de *bias*). `W1` tem 4 × 8 = 32, `b1` tem 8, `W2` tem 8 e `b2` tem 1: **49 botões**. Começam aleatórios (`torch.randn`) ou em zero, como o preço começou em $0. Dois detalhes de sintaxe: o `()` de `torch.zeros((), ...)` quer dizer “um único número”, o botão final; e o sublinhado de `requires_grad_()` liga o “gravador” nesse mesmo tensor, já criado.

Já viu a mesa de som do DJ numa festa? Uma mesa cheia de botões: graves, agudos, volume de cada microfone, eco. Nenhum botão sozinho faz soar bem. O que soa bem é a combinação. E o DJ não a calcula: mexe em um, escuta, mexe em outro, escuta de novo. Os **pesos** são isso. Cada um diz quanto um dado importa numa mistura. Cada peso de `W1` ajusta quanto um dado conta numa das suspeitas intermediárias: grande, “aumente isto”; perto de zero, “ignore isto”; negativo, “subtraia”. Sozinho, ele não mede a importância global do dado nem se ele é sinal de fraude ou de normalidade: isso depende também de `W2`, dos vieses e de `relu`. A diferença é que o nosso DJ tem 49 botões e mexe em todos ao mesmo tempo, guiado pela perda em vez do ouvido.

Leia `predecir` olhando só as formas, como na sessão 6:

- `X @ W1`: `(compras, 4) @ (4, 8)` dá `(compras, 8)`. Cada compra passa de 4 dados para **8 “suspeitas” intermediárias**, cada uma uma mistura diferente de valor, hora, distância e estrangeiro. Uma poderia acabar se concentrando em “caro e de madrugada”; outra, em “longe e estrangeiro”. Ninguém diz quais: ele as descobre ao girar os botões.
- `+ b1`: um vetor `(8,)` somado a cada linha. Broadcasting.
- `torch.relu(...)`: **se um número sai negativo, vira zero; se sai positivo, fica igual.** “Se esta suspeita não se aplica, nem a mencione”. Sem `relu`, as duas camadas se juntariam numa mistura só, uma única conta de multiplicar e somar, que não consegue imitar os cortes da regra (mais de $15.000, antes das 5, mais de 400 km) nem como eles se combinam. (Se as luzes já viessem contadas como entradas, “duas ou mais” seria sim uma mistura só.)
- `oculta @ W2 + b2`: `(compras, 8) @ (8,)` dá `(compras,)`, a conta da terceira pergunta da revisão. **Um número por compra**: a **pontuação**.

**Pontuação maior que zero: o modelo diz “fraude”; menor ou igual a zero: “normal”.** Quanto mais longe do zero, mais seguro está o modelo, o que não é uma garantia. Nos livros essa pontuação se chama **logit** (daí o nome da perda do Passo 6). Uma pontuação de 0 equivale a “meio a meio”: 50 % de probabilidade de fraude segundo o modelo. Por isso o corte está no zero, e o código desempata o caso exato como “normal” (`> 0`, não `>= 0`). Isso se chama **previsão** ou *forward* (“para a frente”): os dados entram por um lado e a pontuação sai pelo outro, como a massa na máquina de tortilha.

> **21h20** Mateo fica olhando para `W1` como se lhe devesse dinheiro.
>
> — E qual das oito é a de “caro e de madrugada”? — pergunta em voz alta, para ninguém.
>
> Imprime `W1` com `print(W1)`. Aparecem trinta e dois números aleatórios, nenhum com nome. Ele se desespera um pouquinho. Queria ver a regra escrita, bonita, numa célula. Não vai acontecer. As suspeitas não vêm com rótulo: vão se formando sozinhas enquanto os botões giram.
>
> — Já brigou com a máquina? — pergunta dona Lupe.
>
> — Ainda não. Só que ela não me diz o que está pensando.
>
> — Igual ao teu tio. Trinta anos de casada e continuo sem saber.
>
> Mateo cai na risada e decide não brigar com o que não pode ver. Basta cuidar das formas: quatro dados entram, oito sinais saem e no final fica uma pontuação por cobrança.

## Passo 6 — A perda para um “sim ou não”

Em `cerro.py`, a perda era a diferença ao quadrado. Para perguntas de “sim ou não” usa-se outra, já pronta no PyTorch:

```python
perdida_fn = torch.nn.BCEWithLogitsLoss()
optimizador = torch.optim.SGD([W1, b1, W2, b2], lr=1.0)
```

Não vale se assustar com o nome. `BCEWithLogitsLoss` faz o papel da nossa subtração ao quadrado, mas para fraude/não fraude: recebe pontuações e rótulos e devolve **um único número** com o quanto ele foi mal em todas as compras juntas. A regra dela é de bom senso: se era fraude e a pontuação foi muito positiva, perda quase zero; se era fraude e a pontuação foi muito negativa, perda enorme (estar muito seguro e errar é o pior); se a pontuação está perto de zero, perda média.

Pense no amigo que jura que o jogo já está ganho e no final o time perde de 3 a 0. Não é o mesmo que o amigo que disse “sei lá”. Os dois erraram, mas do primeiro se cobra mais caro a aposta. Esta perda cobra igual: duvidar custa pouquinho; errar com muita certeza custa muitíssimo. Assim o modelo aprende a não exibir uma segurança que não tem.

A segunda linha é o **otimizador**, o que gira os botões. `SGD` faz o mesmo que foi feito à mão em `cerro.py`, `botão -= taxa * inclinação`, mas com os 49 de uma vez. `lr=1.0` é a taxa.

> **21h41** A máquina já tem os dados, os botões e uma maneira de avaliar os próprios erros. Só falta soltá-la.
>
> Mateo está com o dedo em cima da tecla e não aperta. E se ela não aprender nada? E se a perda ficar onde está e ele perder a noite, e a seguinte, e o dia 23 chegar de mãos vazias?
>
> Dona Lupe desliga a cafeteira, que já cheira a queimado.
>
> — Quem não arrisca não petisca, filho. E quem não aperta o botão não fica sabendo.
>
> O relógio da parede marca quase dez horas. Faltam nove dias, e cada um pesa. Mateo respira. Cada volta do ciclo tem que aproximá-lo de uma pista, não só de um número bonito.

## Passo 7 — O ciclo completo e as épocas

Agora sim, o coração da sessão:

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

Compare com `cerro.py`. É o mesmo ciclo, linha por linha:

| Passo | Em `cerro.py` | Em `detector.py` |
|---|---|---|
| Prever | `cobro = precio * 3` | `puntaje = predecir(X_ent)` |
| Medir | `(cobro - 60) ** 2` | `perdida_fn(puntaje, y_ent)` |
| Apagar inclinações antigas | `precio.grad.zero_()` | `optimizador.zero_grad()` |
| Calcular inclinações | `perdida.backward()` | `perdida.backward()` |
| Girar botões | `precio -= tasa * precio.grad` | `optimizador.step()` |

Cada volta do `for` mostra ao modelo **todas** as 3.200 compras de estudo. Uma volta completa por todos os dados se chama **época**: repassar a matéria inteira uma vez. Aqui damos 300.

Por que tantas? Porque em cada volta os botões se movem só um passinho. É como aprender uma música no violão: na primeira vez que se toca inteira, erra-se em todas as trocas de acorde. Na décima, só na difícil. Na centésima, nem se pensa mais. Cada época é tocar a música inteira mais uma vez. Ninguém aprende de primeira, nem gente nem computador.

Execute o que já está escrito e olhe a perda:

```
época   1   pérdida 0.6819
época  10   pérdida 0.4098
época  50   pérdida 0.2528
época 100   pérdida 0.2070
época 200   pérdida 0.1984
época 300   pérdida 0.1910
```

**A perda de estudo cai:** o modelo está reduzindo erros nas 3.200 compras que já viu. É um bom sinal, mas ainda não prova que funcione com compras novas: isso é conferido na prova do Passo 8. Cai rápido no começo e depois cada vez mais devagar, como com o caderno: perto do fundo o chão quase não se inclina. Na minha máquina, as 300 épocas levaram menos de um décimo de segundo.

> **Por que 300 e não mais?** Nós medimos. Cada computador arredonda os decimais numa ordem um pouco diferente; a diferença é microscópica, do tamanho da sétima casa decimal. Com passos longos como `lr=1.0`, depois de umas 300 voltas essas diferenças começam a crescer: com 400, dois computadores com as mesmas versões terminaram com contagens diferentes de fraudes pegas. Com 300, o resultado não se moveu nem quando introduzimos de propósito diferenças mil vezes maiores que as que existem de um computador para outro. Por isso o curso usa 300: para que, se o número sair diferente, fique claro que o erro está no código e não na máquina.

**21h47** Mateo aperta a tecla.

O terminal cospe a primeira linha antes que ele consiga piscar: `época 1 pérdida 0.6819`. Dá um nó no estômago. Esse número ainda não diz nada a ele. É o modelo chutando às cegas, com os botões aleatórios, como um cachorro que ouve o próprio nome em outra língua.

Depois aparece a segunda: `0.4098`.

Mateo se inclina para a tela.

`0.2528`.

— Tia. Tia, vem cá.

Dona Lupe chega secando as mãos no avental.

— O que foi? Quebrou?

— Não. Olha. Está caindo.

`0.2070`. `0.1984`. `0.1910`.

Seis números, um embaixo do outro, cada um menor. Acabou. Durou menos do que a água do café leva para ferver. Mateo fica olhando a coluna como quem vê o placar no fim de um jogo que não esperava ganhar.

Ninguém escreveu para a máquina a regra “duas luzes ou mais”. Ninguém falou de madrugadas nem de estabelecimentos estrangeiros. Deram a ela, sim, 3.200 compras, os quatro dados de cada uma e a resposta fraude/não fraude de cada uma, e uma maneira de medir o quanto ela ia mal. Com esses exemplos, volta por volta, ela foi movendo os 49 botões até errar menos. Mateo sente algo estranho, entre orgulho e arrepio. Ele escreveu umas poucas linhas. O resto foi ela que fez.

— E isso é bom? — pergunta dona Lupe.

— É que ela está aprendendo, tia. Estou vendo ela aprender.

Dona Lupe olha para a tela, depois para o sobrinho, que está com os olhos brilhando como quando era menino e conseguia fazer o pião rodar.

— Pois tomara que aprenda rápido. Porque o dia 23 não espera.

Mateo concorda com a cabeça, mas já não a ouve direito. **A perda é um sinal, não uma sentença.** Diz que o modelo erra menos no estudo. Ainda falta o importante: ver o que ele faz quando as respostas são tiradas dele.

## Passo 8 — A prova: 91 %

Falta o importante: ele vai bem com compras que **nunca viu**? Acrescente a última parte:

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

Os dois últimos números que ele imprime (recall e precisão) ainda não têm nome: ignore-os até o Passo 9, onde são explicados e lidos de novo. Por ora, olhe a acurácia e as três contagens.

O truque é o `*` elemento a elemento, o da sessão 6: `dice_fraude * y_exa` só dá 1 onde **os dois** são 1 (disse fraude **e** era). `1 - y_exa` inverte uns e zeros, então `dice_fraude * (1 - y_exa)` dá 1 onde disse fraude e **não** era. Somando, tem-se quantos de cada caso.

O script completo tem 61 linhas, 41 sem comentários nem linhas em branco. Execute com `python3 detector.py`. Saída completa, idêntica nas duas vezes em que o executei:

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

**Acurácia: 0.908.** Em compras que nunca viu, acertou quase 91 de cada 100, sem que ninguém lhe dissesse a regra das luzes.

Sobre a semente: com a mesma versão do PyTorch e um computador como o meu, o normal é sair exatamente isto. Mas o PyTorch não garante números idênticos entre versões nem entre computadores diferentes ([documentação do PyTorch sobre reprodutibilidade](https://docs.pytorch.org/docs/2.14/notes/randomness.html)): as últimas casas decimais podem se mover um pouco. Com 300 épocas medimos que não se movem (ver “Por que 300 e não mais?” no Passo 7); nos exercícios com muito mais voltas, podem se mover. O que **tem** que acontecer sempre: a perda cai e a acurácia fica em torno de 91 %. Se sair muito diferente, revise a tabela de erros.

**22h12** Mateo vê o 91 e levanta os braços como se tivesse feito gol.

— Noventa e um, tia! Acertou noventa e um de cada cem!

Dona Lupe não levanta os braços. Ajeita os óculos e se aproxima da tela com a calma de quem já contou mil fechamentos de caixa.

— Então, quão bom é?

— Bom… muito bom. É um nove.

— Um nove em quê? O que diz essa linha de baixo?

Mateo a lê em voz alta, mais devagar: “se escaparon: 44”.

A empolgação murcha como balão furado. Quarenta e quatro. De 125 fraudes na prova, quarenta e quatro passaram como se nada fosse. O modelo as viu, as revisou e disse “normal”.

São compras inventadas, ele sabe. Mas não consegue evitar dar um rosto a elas. Imagina uma dessas quarenta e quatro. Uma senhora que comprou cadernos no Norte em setembro, pagou com cartão, agradeceu e foi embora. E numa manhã, na fila da padaria, abre o aplicativo do banco e vê uma cobrança de dezessete mil pesos numa loja onde nunca esteve. O décimo terceiro, ou o que ia ser o décimo terceiro. Depois imagina o rapaz que juntou dinheiro para um celular e teve a conta esvaziada numa terça às três da manhã. E outro. E outro. Quarenta e quatro vezes.

— Não é um nove, tia — diz, e a voz sai miudinha. — Para uma lição de casa, nove é nota boa. Para 125 fraudes, quarenta e quatro é uma porta aberta.

Dona Lupe põe a mão no ombro dele.

— Viu só? Nem tudo que reluz é ouro, filho. Agora descubra de que é feito.

> **Quadro de alerta:** a acurácia conta todos os acertos juntos. Não diz quantas fraudes deixou passar. Por isso um banco não pode ficar só com esse número.

91 % é bom? Soa como nota 9. Agora vem o mais importante da sessão.

## Passo 9 — A armadilha: o detector preguiçoso tira 84

Imagine um “detector” que diz a todas as compras “não é fraude”, sem olhar para elas.

Na prova há 800 compras e 125 são fraude. As outras 675 são normais, e o preguiçoso acerta todas: 675 de 800 é **84.4 %**. Isso vai ser comprovado com um tensor no exercício 28.

Um programa de uma linha que não serve para nada tira 84; o nosso, 91. A diferença é bem menor do que parecia. **A acurácia engana quando o importante é raro**, e num banco a fraude costuma ser rara: quase todas as compras são de gente honesta.

Pense num detector de metais na entrada de um show. Quase ninguém traz nada perigoso. Se o segurança deixa todo mundo passar sem revistar, “acerta” com quase todo o público. E mesmo assim é o pior segurança do mundo, porque o único caso que importava passou. Acertar o comum é fácil. O difícil, e o que vale, é pegar o raro.

> **22h31** Mateo escreve o detector preguiçoso num papel, sem computador: “para tudo, não”. Fica olhando um tempo. É uma linha. É ridícula. E tira 84.
>
> — Ou seja, o meu modelo de 49 botões ganha de não fazer nada por menos de sete pontos — murmura.
>
> — E isso é pouco? — pergunta dona Lupe.
>
> — É como se gabar de correr mais rápido que alguém sentado.
>
> Dona Lupe solta uma gargalhada que faz tilintar os copos da prateleira.
>
> — Pois com isso não ganha nem do caminhão de lixo.
>
> Mateo não se ofende. Pela primeira vez na noite sente que entende de verdade. O número que importa não é o maior. É o que conta o que escapou.

Por isso os bancos olham outros dois números. Primeiro, os quatro casos possíveis da prova:

| | De verdade era fraude | De verdade era normal |
|---|---|---|
| **O modelo disse “fraude”** | 81 pegas | 30 alarmes falsos |
| **O modelo disse “normal”** | 44 escaparam | 645 liberadas corretamente |

(As quatro células somam 800. Os acertos são 81 + 645 = 726, e 726 / 800 = 0.9075, que arredondado dá 0.908, a acurácia de cima.)

**Recall: de todas as fraudes, quantas ele pegou?** Houve 125 fraudes e ele pegou 81: 81 / 125 = **0.648**. Escaparam 44 de 125: mais de 1 em cada 3 fraudes. Cada uma dessas 44 é alguém que amanhã olha o extrato e encontra uma cobrança de $17.000 numa loja onde nunca esteve. Tem que contestar a cobrança, o banco revisa o caso e muitas vezes acaba devolvendo o dinheiro; o susto, ninguém devolve. O detector preguiçoso tem recall **zero**: não pega nenhuma.

**Precisão: de todos os alarmes que deu, quantos eram verdadeiros?** Deu 81 + 30 = 111 alarmes, e 81 eram fraude: 81 / 111 = **0.73**. Mais ou menos 1 em cada 4 alarmes foi um alarme falso. Cada um desses 30 é uma pessoa honesta que, dependendo de como o banco trabalha, teve a compra bloqueada no caixa do supermercado, com a fila atrás, ou precisou confirmar que era ela antes de ser liberada. Se isso acontece com frequência, o cliente troca de banco.

As duas coisas pesam. Se o banco quer pegar **todas** as fraudes, marca tudo como suspeito: recall perfeito e todos os clientes irritados. Se não quer incomodar **ninguém**, não marca nada: zero alarmes falsos e todas as fraudes passam, como com o preguiçoso. O trabalho de verdade está no meio, e depende de quanto custa cada erro. Por isso, para julgar um sistema antifraude, importa mais quantas fraudes ele pega e quantos alarmes falsos dá do que a “acurácia” sozinha.

**A grande lição de hoje:** acertar muito não basta. Antes de acreditar num modelo, pergunte quão raro é o que ele procura e quantos desses casos raros ele pega.

> **23h05** Mateo desenha a tabela de quatro células no verso de uma conta de luz. Em cima à esquerda, 81: os que pegou. Embaixo à esquerda, 44: os que foram embora. Em cima à direita, 30: gente honesta com o cartão recusado no caixa. Fica olhando essa célula. Se o banco fizesse isso com os clientes do Norte, a papelaria perderia os poucos que ainda voltam.
>
> — Não é um número, tia. São dois. E um puxa o outro, como cobertor curto: se cobre os pés, descobre os ombros.
>
> Dona Lupe concorda devagar. Essa ela entende perfeitamente. Mateo guarda a conta no bolso da jaqueta. Amanhã vai precisar dela.

---

👉 **Para quem é de Administração:** não é preciso treinar o modelo para fazer a pergunta certa na reunião. Quando alguém apresentar “o nosso modelo acerta 91 %”, pergunte: quanto tira um que não faz nada? Quantos dos casos que nos importam ele pega (recall)? Quantos dos alarmes dele são verdadeiros (precisão)? Depois traduza cada erro em dinheiro: quanto custa uma fraude que escapa e quanto custa um cliente irritado. Essa tabela de quatro células é uma ferramenta de negócio, e serve igual para decidir a quem dar crédito ou que sinistro de seguro revisar.

---

## Exercícios da sessão 7

As soluções estão no final. Antes de cada mudança, **escreva no papel o que deve acontecer**. Prever antes de executar é metade do aprendizado. Preencha esta tabela no papel (os exercícios 26 e 27 pedem estas quatro execuções):

| Configuração | Minha previsão | Perda final | Acurácia | Pegas | Alarmes falsos | O que aprendi? |
|---|---|---:|---:|---:|---:|---|
| 300 épocas, `lr=1.0` | | | | | | |
| 2000 épocas, `lr=1.0` | | | | | | |
| 300 épocas, `lr=0.1` | | | | | | |
| 300 épocas, `lr=10` | | | | | | |

**Exercício 26 — Mais voltas, menos voltas.** Em `detector.py`, o número de épocas está em `range(1, 301)`. Já o executei com 100 épocas (`range(1, 101)`): a perda termina em 0.2070, a acurácia em 0.902, pega 77 fraudes e dá 30 alarmes falsos. Agora é a vez de executá-lo com **2000** épocas. Acrescente `1000, 2000` à lista de épocas que são impressas para ver a perda.

a) Antes de executar: a perda final vai ser maior ou menor que 0.1910?
b) O que aconteceu com a acurácia, as fraudes pegas e os alarmes falsos?
c) Aponte no código qual linha é cada um dos cinco passos do ciclo (prever, medir, apagar inclinações, calcular inclinações, girar botões).

**Exercício 27 — O tamanho do passo.** Volte para 300 épocas. Agora mude a taxa, `lr=1.0`, primeiro para `lr=0.1` (passinhos de formiga) e depois para `lr=10` (passadas de gigante). Antes de executar cada uma, preveja: a perda cai mais ou menos? Por quê?

Aviso: com uma das duas, o programa quebra no final com um `ZeroDivisionError`. Não é erro de quem programou. Leia as linhas acima do erro e explique por que aconteceu. Dica: olhe quantos alarmes ele deu no total.

**Exercício 28 — Calcule o detector preguiçoso.** Acrescente isto no final de `detector.py` e preencha as lacunas. O detector preguiçoso é um tensor de 800 zeros: “não é fraude” para todas as compras da prova.

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

**Segunda parte.** Calcule agora o recall e a precisão do **detector treinado**, sem olhar as duas últimas linhas do Passo 8. `atrapados`, `falsas` e `escapados` já estão no arquivo:

```python
recall_modelo = ______________________________
precision_modelo = ___________________________

print("recall del modelo:", round(recall_modelo, 3))
print("precisión del modelo:", round(precision_modelo, 3))
assert round(recall_modelo, 3) == 0.648 and round(precision_modelo, 3) == 0.73, "Revisa cuál de las cuatro casillas falta en tu fracción"
```

Com a execução base (300 épocas, `lr=1.0`) devem sair `0.648` e `0.73`. Se as épocas ou a taxa foram mudadas, tire o `assert`.

Depois responda: qual é o recall do detector preguiçoso? Por que não dá para calcular a precisão dele?

**Exercício 29 — O que o modelo aprendeu?** A regra usa os quatro dados. O modelo percebeu? Pergunte a ele **embaralhando uma coluna**: ao embaralhar a hora entre as 800 compras da prova, cada compra fica com a hora de outra ao acaso. A coluna continua lá, mas já não diz nada. Se a acurácia cair muito nessa execução, é evidência de que o modelo estava usando essa coluna para prever. Isso não prova por si só que ele descobriu a regra exata nem que essa coluna cause fraude.

É preciso repetir isso para cada uma das 4 colunas: copiar a prova, embaralhar **só essa coluna**, prever de novo e mostrar a acurácia e as fraudes pegas. Duas ferramentas novas:

- `X_exa.clone()` faz uma cópia, para não estragar a prova original.
- `torch.randperm(800)` dá os números de 0 a 799 fora de ordem. `X_exa[torch.randperm(800), col]` é a coluna `col` embaralhada.

Lembre-se de prever dentro de `with torch.no_grad():`, como na prova. Um exemplo resolvido, com a coluna `monto` (a coluna 0):

```python
X_original = X_exa.clone()
X_revuelto = X_exa.clone()
X_revuelto[:, 0] = X_exa[torch.randperm(800), 0]

with torch.no_grad():
    dice = (predecir(X_revuelto) > 0).float()

print("monto: exactitud", round((dice == y_exa).float().mean().item(), 3))
```

Agora complete, com um `for`, as colunas 1, 2 e 3 (hora, distância e estrangeiro), e mostre também as fraudes pegas:

```python
for col in range(1, 4):
    X_revuelto = X_exa.________()
    X_revuelto[:, col] = X_exa[torch.________(800), col]
    with torch.no_grad():
        dice = (________(X_revuelto) > 0).float()
    # imprime la exactitud y los fraudes atrapados de esta columna
    assert torch.equal(X_exa, X_original), "No debes alterar el examen original"
```

Depois responda: qual coluna o modelo mais sentiu perder? Alguma não fez diferença? Isso bate com a regra escondida das luzes?

---

## Erros comuns (e como perceber)

| Erro | Como aparece | Como perceber | O que fazer |
|---|---|---|---|
| Esquecer `requires_grad=True` num botão | Em `cerro.py`, onde é o único botão: `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` ao chamar `backward()`. Em `detector.py`, se for esquecido só em `W1`, **não quebra**: `W1.grad` fica em `None`, `W1` nunca se move, a perda empaca em 0.3366 e pega só 13 de 125 | Em `cerro.py` quebra na linha do `backward`; em `detector.py`, a perda cai muito menos que o normal e `print(W1.grad)` diz `None` | Ligue o “gravador” ao criar cada botão: `requires_grad=True` |
| Esquecer de apagar as inclinações (`zero_grad`) | Em `cerro.py`, o `assert` salva: quebra no passo 2 com `AssertionError: PyTorch y el lápiz deberían coincidir`. Sem os `assert`, não quebra, e isso é o perigoso: o preço salta para 37.80, depois para 41.58, a perda sobe até 4191.27 (no passo 4) e, ao terminar, o preço aprendido fica em 4.95 em vez de 20 | A perda sobe e desce em vez de cair | O PyTorch soma as inclinações novas às antigas; apague-as a cada volta |
| Rótulos com forma diferente das pontuações | `ValueError: Target size (torch.Size([3200, 1])) must be the same as input size (torch.Size([3200]))` | Imprima `y.shape` e o `shape` da pontuação: têm que ser iguais | Deixe `y` com forma `(4000,)`, sem `reshape(4000, 1)` |
| Taxa grande demais | O treinamento não quebra. Com `lr=10` a perda empaca em 0.4365 e o modelo deixa de pegar fraudes; no final quebra a precisão com `ZeroDivisionError`, porque não deu nenhum alarme (exercício 27) | A perda para de cair ou salta | Baixe a taxa; teste de 10 em 10 vezes (1, 0.1, 0.01…) |
| Chamar `backward()` duas vezes sobre a mesma perda | `RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed).` | Quebra no segundo `backward` | Um `backward` por volta; a volta seguinte prevê e mede de novo |
| Acreditar só na acurácia | “91 %!” | Calcula-se o detector preguiçoso e ele tira 84 % | Olhe também recall e precisão |

---

## Revisão para a próxima sessão

A sessão 8 começa com estas três perguntas. Responda sem olhar a lição:

1. Que forma tem o tensor de entrada do detector e o que é cada eixo?
2. O que a perda mede e por que ela tem que cair?
3. Por que acertar 91 % das vezes pode não ser suficiente para um banco?

🔁 **Na próxima semana:** os tensores no banco, na seguradora e na loja. Casos reais, com a fonte, e o desafio de escrever a forma do tensor de cada um. Spoiler: o detector de hoje tem exatamente a forma dos de verdade.

---

## Soluções (não olhe antes de tentar)

**Revisão para abrir.**

1. As formas são comparadas **da direita para a esquerda**, eixo por eixo. Em cada eixo, os dois tamanhos têm que ser **iguais, ou um deles tem que ser 1** (o 1 é o que é “esticado”). Se em algum eixo isso não se cumpre, o PyTorch dá erro.
2. `*` multiplica **elemento a elemento**: célula com célula, e as formas têm que coincidir ou ser esticadas. `@` é o **produto de matrizes**: linha por coluna, multiplicando e somando. O enganoso é que às vezes os dois rodam sem erro e dão coisas diferentes.
3. `(7,)`. O 3 do meio é “consumido” (multiplicado e somado) e fica um número para cada uma das 7 linhas. Hoje isso foi usado: `(compras, 8) @ (8,)` deu `(compras,)`.

**Exercício 26.** Com 2000 épocas (as primeiras linhas são iguais às de 300; colo a partir da 300):

```
época 300   pérdida 0.1910
época 1000   pérdida 0.1641
época 2000   pérdida 0.1142
exactitud: 0.93
fraudes en el examen: 125
atrapados: 83 | se escaparon: 42 | alarmas falsas: 14
recall: 0.664 | precisión: 0.856
```

a) Menor: 0.1142. Mais voltas, mais tempo para descer o morro. As primeiras épocas dão os mesmos números que com 300 graças à semente fixa. Da 1000 em diante, em outro computador a última casa da perda e alguma contagem podem sair um pouco diferentes: são as diferenças de arredondamento do quadro “Por que 300 e não mais?”, que com tantas voltas têm tempo de crescer.

b) A acurácia sobe para 93 %, pega 83 fraudes em vez de 81 e os alarmes falsos caem de 30 para 14. Atenção: mais épocas nem sempre ajudam. Com dados reais, voltas demais podem fazer o modelo **decorar** os exemplos de estudo e ir pior na prova. Por isso a prova é feita com dados separados.

c) `puntaje = predecir(X_ent)` prevê; `perdida = perdida_fn(puntaje, y_ent)` mede; `optimizador.zero_grad()` apaga inclinações; `perdida.backward()` calcula inclinações; `optimizador.step()` gira botões.

**Exercício 27.** Com `lr=0.1` (tiro algumas épocas do meio):

```
época   1   pérdida 0.6819
época  50   pérdida 0.4238
época 300   pérdida 0.3194
exactitud: 0.859
fraudes en el examen: 125
atrapados: 12 | se escaparon: 113 | alarmas falsas: 0
recall: 0.096 | precisión: 1.0
```

A perda cai, mas muito mais devagar: em 300 épocas chega a 0.3194, pior do que a taxa 1.0 conseguia em 50. Passinhos de formiga: precisa de muito mais voltas. E pega só 12 de 125. Repare na precisão: 1.0, perfeita, porque os 12 alarmes que deu eram fraude. Um detector que quase nunca dispara quase nunca erra quando dispara, e mesmo assim deixa passar 113 fraudes: outro motivo para não julgá-lo por um único número.

Com `lr=10` (também recortado):

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

As passadas eram tão longas que na época 10 a perda **subiu** (de 0.6819 para 0.8124): saltou para o outro lado do vale. Depois empacou em 0.4365. E como terminou? Acurácia **0.844**, zero pegas, zero alarmes: **o modelo acabou se comportando como o detector preguiçoso.** Com `lr=10`, nesta execução as atualizações desligaram os oito sinais intermediários (`relu` zerou todos os 25.600 das compras de estudo) e o modelo ficou com uma pontuação constante e negativa, a mesma para todas as compras (−1.67 na prova). Com o corte em zero, nunca dá alarme. Atenção: ele continua minimizando a mesma perda de sempre; não “escolheu” a acurácia. Sair 84 % é consequência de a fraude ser rara.

O `ZeroDivisionError` aparece porque a precisão é pegas dividido por alarmes, e houve 0 alarmes: 0 / 0. Não dá para medir a pontaria de quem nunca atirou.

**Exercício 28.** As lacunas são `zeros`, `y_exa`, `y_exa` e `0`. Este bloco não é para copiar: mostra só as linhas com as lacunas já preenchidas, e os `...` são as linhas do arquivo que não mudam:

```text
siempre_no = torch.zeros(800)
exactitud_flojo = (siempre_no == y_exa).float().mean().item()
atrapados_flojo = (siempre_no * y_exa).sum().item()
...
    if y_exa[i].item() == 0:
```

Saída (as três últimas linhas, depois da saída normal do detector):

```
exactitud del flojo: 0.844
fraudes que atrapa: 0 de 125
El for dice lo mismo: 675 aciertos de 800
```

O recall é 0 / 125 = **0**: não pega nenhuma fraude. A precisão não pode ser calculada pelo mesmo motivo do exercício 27: não dá nenhum alarme, e 0 / 0 não está definido. Um detector com 84 % de acurácia e recall zero: essa é a lição inteira em dois números.

**Exercício 29.** Uma solução:

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

Saída (no final do detector, com a mesma semente):

```
revolviendo monto      exactitud 0.848   atrapados 54 de 125
revolviendo hora       exactitud 0.873   atrapados 63 de 125
revolviendo distancia  exactitud 0.870   atrapados 64 de 125
revolviendo extranjero exactitud 0.849   atrapados 57 de 125
```

Sem embaralhar, acertava 90.8 % e pegava 81. Embaralhar **qualquer uma** das quatro o derruba: a acurácia cai para entre 84.8 % e 87.3 %, mal acima do preguiçoso ou igual a ele, e ele deixa de pegar entre 17 e 27 fraudes. As que mais fizeram falta: valor e estrangeiro.

É compatível com a regra escondida: as quatro colunas acendem uma luz, e o modelo recebeu exemplos rotulados produzidos com essas quatro colunas. É evidência de que ele as usa, não uma prova de que tenha recuperado a regra exata. Se os números saíram diferentes porque o `randperm` embaralhou de outra forma, tudo bem: o que tem que acontecer é as quatro caírem.

---

## Na próxima semana

Na sessão 8, a última, Mateo sai da papelaria e entra no banco para ver como esses mesmos tensores trabalham com milhões de compras, e com pessoas que revisam a decisão.

Para adiantar, vale pensar: se o detector visse não uma compra, mas as últimas 30 compras de cada cliente, que forma teria o tensor de entrada dele?

**23h48**

O depósito está em silêncio. Dona Lupe adormeceu na cadeira de plástico, de avental e com o saquinho de moedas no colo. O pão de canela continua ao lado do caixa, já duro.

Mateo não roda o detector nas cobranças do Norte. Pensa nisso, com o dedo sobre a tecla, e não faz: o exercício aprendeu com compras e uma regra inventadas, e não serve para decidir sobre pessoas reais. Em vez disso, pega a tabela de quatro células e a põe ao lado das notas que a Fer anotou à mão. Organiza as perguntas: quais cobranças não têm mercadoria por trás, quais caem de madrugada ou longe, que evidência pode pedir ao banco.

Deveria se sentir contente, e está, pela metade. Tem um método. Mas um método não é uma prova.

Abre o e-mail para escrever ao doutor Ortega. Digita “Compartilho com o senhor um modelo que” e para. Imagina a cara do doutor. Um rapaz de vinte anos, um notebook com um adesivo descolando, um programa que acerta 91 % e deixa escapar mais de uma em cada três. E do outro lado, um banco com os próprios sistemas, os próprios analistas, as próprias regras.

Apaga a frase.

Pela janela passa a última viatura da noite. A lâmpada zumbe. No calendário, o círculo vermelho do 23 parece maior que de manhã.

Mateo fecha o notebook devagar, para não acordar a tia. Já sabe o que tem. Já sabe também o que falta: não um número mais alto, mas alguém do outro lado disposto a revisar.

Faltavam nove dias para o 23. E o banco não ia acreditar num modelo feito num notebook.

---

*Escrevo a partir do que vemos todos os dias na Hábil, onde construímos sistemas para bancos, seguradoras e empresas no México. Se no trabalho ou no negócio da família aparecer um problema assim, vamos conversar: [habil.mx](https://www.habil.mx/pt/)*
