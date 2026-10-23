# Tensores do zero (8): tensores no banco, na seguradora e na loja

*Como ler um caso real e dizer que forma tem o tensor dele, por que numa competição de previsão com dados de uma rede de supermercados norte-americana ganhou um método que não era uma rede neural, e quais cuidados não são opcionais.*

*Tensores do zero · lição B, sessão 8 de 8*

*Cenário: os personagens e a história são inventados. Os dados técnicos, históricos e os números reais trazem a fonte.*

**Quarta-feira, 23 de dezembro, 9h40. Agência do banco, segundo andar.**

O último dia do prazo.

A sala de espera cheira a café requentado e a pinheiro de plástico. Alguém pôs uma arvorezinha ao lado do bebedouro. Tem três bolas e uma está caindo.

Dona Lupe está com a bolsa no colo, agarrada com as duas mãos, como se alguém fosse tomá-la. Mateo trouxe o notebook. Ali dentro está tudo: a planilha do Norte, os totais, a troca de agosto, as cobranças sem mercadoria e o detector da semana passada.

— Se disserem que não — diz dona Lupe —, fecho o Norte no dia 2 de janeiro. Já pensei nisso.

— Não vão dizer que não.

— Isso dizia o teu tio do décimo terceiro, e olha só.

Mateo não responde. Repassa na cabeça o que o preocupa há nove dias: o detector acertava 91 % das vezes. E fraudes escapavam. Que banco vai acreditar num modelo feito num notebook, na mesa de uma papelaria?

— Senhora Herrera?

O doutor Ortega é mais jovem do que parecia pelo telefone. Gravata vermelha, crachá pendurado, pressa nos sapatos. Leva os dois a uma sala com uma janela que dá para o estacionamento.

— Tenho vinte minutos — diz. — Mostre o que a senhora trouxe.

Mateo abre o notebook. Mostra a queda do Norte em agosto. As cobranças de madrugada. Os valores que não correspondem a mercadoria nenhuma. O detector.

Ortega não diz nada durante um bom tempo. Depois vira o próprio monitor para eles.

— Sabe o que é isto?

Na tela há uma lista. Cobranças. Muitas. Algumas marcadas em vermelho.

— É o mesmo que o senhor fez — diz Ortega. — Só que nós fazemos isso com milhões de compras por dia. Sente-se, meu jovem. Vou mostrar como isto fica em grande escala.

Mateo olha o relógio da parede. Os vinte minutos acabam de começar a correr.

## Revisão para abrir

A ideia é responder sem olhar a sessão 7. As respostas estão nas soluções.

1. Que forma tem o tensor de entrada do detector e o que é cada eixo?
2. O que a perda mede e por que ela tem que cair?
3. Por que acertar 91 % das vezes pode não ser suficiente para um banco?

> **O código desta sessão** está no repositório público do curso, na pasta `sesion-08`: [github.com/HabilMX/curso-tensores](https://github.com/HabilMX/curso-tensores/tree/main/sesion-08). Vale escrevê-lo primeiro, porque é assim que se aprende; o repositório serve para comparar quando algo não sair igual. Cada programa vem com a saída que deve produzir.

## Objetivos desta sessão

| # | Ao terminar, é possível… | Demonstrado em |
|---|---|---|
| 1 | **Ler** um caso real e **escrever** a forma do tensor dele, dizendo o que é cada eixo | Exercício 30 |
| 2 | **Criar** com `torch.zeros` um tensor com a forma de um caso e **conferi-la** com `assert` | Exercícios 30 e 31 |
| 3 | **Explicar** o que é um embedding e **prever** a forma da saída dele | Exercício 32 |
| 4 | **Converter** uma sequência em tabela e **dizer** o que se perde | Exercício 33 |

**O que é preciso:** o ambiente do PyTorch da sessão 2 e as sessões 5 a 7. Hoje não se instala nada.

---

## Por que importa

Até aqui, os dados foram inventados ou saíram da planilha de dona Lupe. Hoje é a vez de ver o que Mateo viu na tela de Ortega: casos reais de bancos, seguradoras e lojas.

Ninguém os entrega com a forma escrita. Eles chegam com uma frase como “o modelo analisa o histórico do cliente”. O trabalho é traduzi-la para uma forma: quantos eixos, o que é cada um e de que tamanho. Quem sabe fazer isso não ouve mais nenhuma notícia de IA como mágica. Ouve uma caixa com direções, como a planilha da papelaria.

Ortega disse isso a Mateo com outras palavras: “Aqui ninguém se impressiona com a palavra *inteligência*. Aqui se pergunta o que entra e o que sai”.

**Dois avisos.** Primeiro: quase nenhuma empresa publica as formas exatas dos tensores dela; publica que técnica usa. Então as formas de hoje dizem **como é esse tipo de dado em geral**, não a receita secreta de cada empresa. Segundo: quando um número é informado pela própria empresa, sem revisão de fora, ele vem com um **“segundo a empresa”**. Como o primo que diz ter 1.80: pode ser verdade, mas ninguém mediu.

---

## Passo 1 — As três perguntas para ler um caso

— Antes de eu mostrar qualquer coisa — diz Ortega —, me diga uma coisa. Quando o senhor vê um sistema desses, qual é a primeira coisa que pergunta?

Mateo pensa. Em novembro teria dito “quão bom ele é”. Agora não.

— Que forma ele tem.

Ortega sorri pela primeira vez na manhã.

A habilidade central nova de hoje é traduzir um caso em eixos e forma, e ela cabe em três perguntas, nesta ordem (além disso, aparecem duas convenções novas, a ordem das cores no PyTorch e os embeddings; hoje basta reconhecer as formas delas):

1. **O que é uma coisa?** Uma compra, um cliente, uma foto. Quase sempre é o primeiro eixo: o lote.
2. **O que cada coisa traz dentro?** Uma lista de dados acrescenta um eixo. Uma história no tempo acrescenta outro antes dos dados. Uma foto traz cores, altura e largura.
3. **Que número vai em cada célula?** Um valor, uma cor, o número de um produto.

Com as respostas, a forma é “desenhada” com `torch.zeros`, que cria um tensor de zeros com a forma pedida: os valores não importam, importa a caixa. Crie `casos.py`:

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

Saída:

```
compras: torch.Size([64, 4])
historial: torch.Size([32, 30, 4])
fotos: torch.Size([8, 3, 480, 640])
números en las fotos: 7372800
Las tres formas cuadran
```

`numel()` conta quantos números cabem na caixa: 8 × 3 × 480 × 640 = 7.372.800. Oito fotos médias já são mais de sete milhões de números.

Dona Lupe espia o notebook.

— Só zeros?

— É a fôrma, tia. Como quando se compra a moldura antes da foto.

— Ah. Pois a moldura já me custou sete milhões.

---

## Passo 2 — O banco: cinco maneiras de caçar uma fraude

Ortega abre uma pasta no computador. Dentro há recortes: comunicados, notas, artigos de blogs de empresas.

— Junto isto para os treinamentos — diz. — Repare que cada um olha para uma coisa diferente.

Todos os bancos querem o mesmo: detectar a compra estranha sem incomodar o cliente honesto. O que muda é **o que olham**, e isso muda a forma. Isto é o que Ortega foi abrindo, um por um:

- **Mastercard.** O comunicado de fevereiro de 2024 diz que o Decision Intelligence ajuda a pontuar 143 bilhões de transações por ano (a versão em espanhol do mesmo comunicado diz 125 bilhões: por isso se lê a fonte), e que a versão nova avalia **a relação entre várias entidades** em torno de cada compra. Segundo a empresa, melhora a detecção em 20 % em média, e até 300 % em alguns casos.
- **Visa.** Em 2019, segundo a empresa, analisava 100 % de 127 bilhões de transações por ano, em mais ou menos um milissegundo cada uma.
- **BBVA.** Numa nota de 2018 sobre um trabalho com o MIT, o BBVA diz que gerou **mais de 200 variáveis novas por transação** e que o modelo “pode reduzir em 54 % o nível de falsos positivos” (alarmes que incomodam um cliente honesto). É o BBVA global, não o BBVA México.
- **HSBC.** O primo da fraude: a lavagem de dinheiro. Num texto de 2023 no blog do Google Cloud, o HSBC diz que revisa mais de 1.200 milhões de transações por mês procurando **movimentações rápidas entre contas e mudanças bruscas de comportamento**, e que encontra de duas a quatro vezes mais atividade suspeita que antes, com 60 % menos alertas. Segundo a empresa.
- **Nubank.** No blog de engenharia, conta que transforma as transações de cada cliente em **sequências de tokens**, como se fossem palavras, e as lê com o tipo de modelo usado para texto. Cada transação pode produzir vários tokens (valor, data, texto da descrição). A empresa informa 131 milhões de clientes no total no fim de 2025; isso não quer dizer que este modelo chegue a todos eles.

Mateo nota algo em cada recorte: quase todos dizem “segundo a empresa”. Ortega o vê notar.

— É disso que eu mais gosto no senhor, meu jovem. Que lê as letras miúdas.

Agora, as três perguntas:

| Caso | O que é uma coisa? | O que traz dentro? | Forma (tipo de dado) |
|---|---|---|---|
| Visa: uma compra de cada vez | uma compra | os dados dela | `(datos,)` |
| Mastercard, BBVA: muitas compras | uma compra | os dados dela (no BBVA, mais de 200) | `(compras, datos)` |
| HSBC: comportamento no tempo | uma conta | as movimentações em ordem, cada uma com os dados dela | `(cuentas, movimientos, datos)` |
| Nubank: histórico como texto | um cliente | as compras em ordem, em tokens | modelo didático, com um token por compra: `(clientes, compras)` de inteiros → `(clientes, compras, números)`; no caso real, como uma compra dá vários tokens: `(clientes, fichas)` → `(clientes, fichas, números)` |

No BBVA, o eixo dos dados passou de 4 (o detector da sessão 7) para mais de 200: **a forma não mudou de ideia, só de tamanho.** No HSBC, olhar o tempo acrescenta um eixo: a passagem da tabela ao cubo da sessão 1, com uma aba por conta.

Mateo fica olhando a linha do BBVA. O detector dele tinha quatro dados por cobrança: valor, hora, distância e se o estabelecimento era estrangeiro. O do banco tem mais de duzentos. E mesmo assim é a mesma caixa: `(compras, datos)`.

O notebook dele não estava brincando de outra coisa. Estava brincando da mesma coisa, em miniatura.

---

## Passo 3 — A seguradora: a batida em fotos

— O banco não é o único que faz isso — diz Ortega, e abre outro recorte. — Meu cunhado trabalha com seguros. Lá já não mandam um perito ver cada amassado. Pedem fotos.

Em março de 2020, a seguradora britânica Ageas anunciou que usava o sistema da empresa Tractable para avaliar batidas de carro. Segundo a Ageas, com fotos o sistema identifica quais peças estão danificadas e quanto, monta o orçamento completo e decide em minutos, às vezes durante a primeira ligação.

As três perguntas: uma coisa é **uma foto** (um sinistro traz várias: o lote); cada foto traz **3 cores**, e cada cor é uma tabela de **altura por largura**; em cada célula vai quanto daquela cor tem aquele ponto. Forma: `(fotos, 3, alto, ancho)`, o tensor `fotos` do passo 1.

Dona Lupe ajeita os óculos.

— Ou seja, se eu bater, tiro foto do carro e o computador me diz quanto.

— Mais ou menos, tia.

— Pois que tire foto da minha conta de luz, para ver o que acha.

**A armadilha da ordem das cores.** Na sessão 1, uma foto foi escrita como `(480, 640, 3)`: altura, largura, cor. No PyTorch o costume é pôr a cor **antes**: `(3, 480, 640)`. Mesmos números, outra arrumação. Se os costumes se misturam, o `assert` avisa:

```python
fotos = torch.zeros(8, 3, 480, 640)
assert fotos.shape == (8, 480, 640, 3), "Un lote de fotos es (fotos, alto, ancho, colores)"
```

```
Traceback (most recent call last):
  ...
AssertionError: Un lote de fotos es (fotos, alto, ancho, colores)
```

Nenhum costume está errado; o errado é supor.

E a saída? Para cada peça, quão provável é “reparar” e quão provável é “substituir”: `(piezas, 2)`. Entram quatro eixos, sai uma tabelinha.

---

## Passo 4 — A loja: a lição de humildade da previsão

Mateo pergunta o que quer perguntar a manhã inteira:

— Então tudo isso é feito com redes neurais? O mais avançado sempre ganha?

Ortega ri.

— Vou mostrar uma de que a tia do senhor vai gostar. É de uma loja.

Em 2020 aconteceu a competição de previsão M5. Estes dados saem do artigo acadêmico dos organizadores (Makridakis, Spiliotis e Assimakopoulos):

- Era preciso prever **42.840 séries hierárquicas** de vendas de **uma rede de supermercados norte-americana** (dados públicos da competição M5 do Kaggle): 10 lojas na Califórnia, no Texas e em Wisconsin, dados diários de uns 5.5 anos. No nível mais desagregado eram 30.490 séries, cada uma “quanto este produto vendeu nesta loja, dia a dia”; as demais eram somas por categoria, departamento, loja ou estado.
- Participaram **7.092 pessoas em 5.507 equipes de 101 países**.
- **O primeiro lugar usou LightGBM**, um método de árvores de decisão, não redes neurais: um ensemble grande de 220 modelos, nada trivial. O segundo lugar combinou LightGBM com uma rede (N-BEATS). O terceiro, sim, usou redes: uma combinação de 43 redes neurais (a equipe usou DeepAR, que lê sequências).

A lição de humildade: **as redes neurais não ganham sempre.** Com dados em tabela, um método que não é uma rede pode vencê-las. E mesmo assim o LightGBM recebe uma tabela `(renglones, datos)`: **o tensor é a forma de organizar os dados, não um tipo de modelo.**

| Quem | Como vê os dados | Forma |
|---|---|---|
| Uma rede que lê sequências (terceiro lugar) | cada série é uma história | `(series, días, datos)` |
| LightGBM (vencedor) | cada dia de cada série é uma linha | `(series × días, datos)` |

— Viu? — diz dona Lupe. — O simples também serve. Digo isso há trinta anos.

Passar de uma para a outra é a sessão 5. Mateo faz isso ali mesmo, na sala, com a papelaria de dona Lupe: 2 filiais × 7 dias viram 14 linhas. Crie `tienda.py`:

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

Saída:

```
papeleria: torch.Size([2, 7, 3])
tabla: torch.Size([14, 3])
Total: 23100 | con for: 23100
```

As linhas 0 a 6 são o Centro; as 7 a 13, o Norte. O total, 18.500 + 4.600 = 23.100, não mudou. O que se perdeu? Que a linha 7 seja “Norte, segunda-feira” a forma já não diz; na prática, acrescentam-se colunas como “filial” e “dia”.

Ortega olha os dois números: 18.500 e 4.600.

— Essa diferença foi o que nos fez olhar para o Norte — diz. — O nosso sistema viu cobranças estranhas na maquininha. O que ele não viu foi o que o senhor acabou de me mostrar: que não havia mercadoria por trás.

---

## Passo 5 — A loja que recomenda: o dicionário que aprende sozinho

Toca o celular de dona Lupe. Uma notificação de uma loja on-line: “Talvez também goste de: kit de geometria”.

— E esta, como sabe que eu vendo kits de geometria?

Ortega e Mateo se olham. Essa pergunta tem resposta.

Por trás de “talvez também goste de…” muitas vezes há um **embedding**. O **Facebook (hoje Meta)** publicou em 2019 um modelo de recomendação de código aberto em PyTorch (DLRM) em que “que produto é” passa por **tabelas de embeddings**. O **Alibaba** publicou naquele ano um que lê a sequência do que um usuário viu e comprou; segundo a empresa, funciona na loja Taobao.

Pense num dicionário de Python: `{"cuadernos": [1.5, -0.3, -2.2], ...}`, uma lista de números por produto. A diferença é que **esses números ninguém escreve: o modelo os aprende sozinho**, como aprendeu os pesos do detector, e eles se ajustam com os dados e a perda. Numa tarefa de recomendação, produtos com padrões de compra parecidos **podem** acabar com números próximos, mas não é uma regra automática.

**Um embedding é um dicionário que aprende sozinho.** No PyTorch a chave é um inteiro, então primeiro o catálogo é numerado. Crie `embedding.py`:

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

Saída:

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

`torch.equal(a, b)` responde `True` só se os dois tensores têm a mesma forma e os mesmos valores. Por isso os dois `assert` conferem que buscar o mesmo produto dá a mesma linha.

Leia devagar:

- `Embedding(5, 3)` é uma tabela de 5 linhas (uma por produto) e 3 colunas, no começo com números aleatórios.
- Buscar um produto é **tirar a linha dele**: cadernos aparece duas vezes, idêntico.
- `grad_fn` é o PyTorch avisando que está anotando a conta para poder ajustar esses números no treinamento. Essa é a parte do “aprende sozinho”.
- **A regra da forma:** a saída é a forma da entrada **mais um eixo no final**, do tamanho do segundo número do Embedding. `(3,)` → `(3, 3)`; `(2, 4)` → `(2, 4, 3)`.

Essa última linha é o modelo didático do Nubank: com um token por compra, `(clientes, compras)` de inteiros vira `(clientes, compras, números)`. No caso real uma compra pode dar vários tokens, e o eixo seria o de tokens. O histórico de compras é lido como um texto.

— Então a loja não sabe que eu vendo kits de geometria — diz dona Lupe. — Sabe que quem compra o que eu compro também compra isso.

— Exato, tia.

— Pois é mais fofoqueira que a minha comadre.

Ao pedir um produto que não está na tabela (o 5, quando só há do 0 ao 4), sai:

```
IndexError: index out of range in self
```

---

## Passo 6 — O México e os cuidados que não são opcionais

**10h05** Ortega olha o relógio. Passou cinco minutos dos vinte e não parece se importar.

— Agora vou dizer uma coisa que não disse pelo telefone — diz. — A mensagem que chegou para a tia do senhor no dia 1º de novembro foi mandada por uma máquina. Um sistema viu cobranças estranhas e suspendeu a maquininha. Sozinho.

Dona Lupe aperta a bolsa.

— E por que me deram prazo até hoje?

— Porque uma máquina pode marcar. Mas uma decisão dessas não deve ficar com a máquina. Alguém tem que revisar. Hoje esse alguém sou eu.

Mateo entende de repente por que está sentado ali. Não veio convencer um modelo. Veio para que uma pessoa revisasse o que o modelo decidiu.

O que vem a seguir é por que esse “alguém tem que revisar” não é cortesia do banco.

**Banxico.** O Banco do México publicou em 10 de junho de 2026 o quadro “Inteligencia artificial y estabilidad financiera: canales potenciales de transmisión de riesgos”, em que alerta para a concentração em **poucos fornecedores**, para modelos **opacos** e difíceis de supervisionar, e para a possível **exclusão não justificada de certos grupos da população**. Uma nota da Expansión de 18 de junho de 2026 também comenta o assunto.

**O viés: German Credit.** É uma base pública muito usada para praticar crédito, do repositório da Universidade da Califórnia em Irvine: 1.000 pessoas com 20 dados cada uma, de 1994. Como tabela original, `X` tem 1.000 pessoas e 20 atributos mistos (números e categorias); antes de usá-la num tensor numérico é preciso codificar as categorias, e então o eixo de dados pode deixar de medir 20. O problema está nas colunas: uma mistura **estado civil e sexo**, outra é **a idade** e outra diz se a pessoa é **“trabalhador estrangeiro”**. Treinar com isso do jeito que está é pedir ao modelo que empreste ou não conforme o sexo ou a nacionalidade. Por isso o curso não a usou na sessão 7.

E se essas colunas forem apagadas? Não basta: outras podem **delatar** o mesmo dado. É como tirar o nome de uma prova e deixar a letra: o professor sabe de quem é do mesmo jeito. É preciso medir como o modelo trata cada grupo.

**A lei europeia.** O *AI Act* da União Europeia, Anexo III, ponto 5(b), classifica como **alto risco** os sistemas que avaliam se uma pessoa é boa pagadora para um crédito ou lhe atribuem uma pontuação de crédito, **exceto os que detectam fraude financeira**. O detector da sessão 7 ficaria de fora; um modelo que pontue crédito, como os que se praticam com German Credit, dentro. Para pensar: por que a lei trata a fraude de forma diferente do crédito? Dica: se o detector erra, uma compra é bloqueada; se o de crédito erra, alguém fica sem a casa.

**A lei mexicana.** A nova Ley Federal de Protección de Datos Personales en Posesión de los Particulares foi publicada em 20 de março de 2025. O **artigo 26, inciso II**, permite que a pessoa **se oponha** a que os dados dela passem por um **tratamento automatizado** que a afete de forma importante e que a avalie **sem intervenção humana**: situação econômica, saúde, confiabilidade ou comportamento, entre outras coisas. Traduzindo: se um modelo decide sozinho se alguém recebe um crédito, nessas hipóteses é possível se opor. Isso não quer dizer que a lei determine, por si só, uma revisão humana em cada decisão, e o direito tem exceções: não procede, por exemplo, quando o tratamento é necessário para cumprir uma obrigação legal. Para quem projeta, “o modelo decide e pronto” não basta: convém um humano no caminho.

Atenção ao que esse artigo diz e ao que não diz: ele fala de **dados pessoais**, de decisões sobre **pessoas**. A maquininha de uma papelaria é outra história, e no relato o banco a revisa manualmente por regra própria. Mas a ideia é a mesma, e Mateo a viu na sala de Ortega: **a máquina marca; uma pessoa revisa; e quem é afetado tem por onde reclamar.**

> **10h20. A sala do doutor Ortega.**
>
> Ortega digita algo. Imprime uma folha. Assina.
>
> — A maquininha do Norte fica reativada a partir de hoje — diz. — Mas antes vai um técnico. Um de verdade, com crachá e ordem de serviço. Ele revisa o aparelho.
>
> Dona Lupe fica calada. Pela primeira vez em sete semanas não tem um ditado à mão.
>
> **13h15. Filial Norte.**
>
> O técnico do banco abre a maquininha sobre o balcão. A Fer está ao lado, de braços cruzados. Mateo, atrás.
>
> Dentro da fenda onde entra o cartão há algo que não deveria estar ali: uma lâmina fina, com um fio da grossura de um fio de cabelo que corre para dentro do aparelho.
>
> — Um leitor — diz o técnico. — Copia o cartão cada vez que alguém paga. As pessoas nem percebem.
>
> A Fer solta o ar devagar.
>
> — Quem pôs foi o de camiseta azul. Em agosto.
>
> Ninguém o tinha chamado. Ninguém perguntou o nome dele. As pessoas pagavam no Norte, e os cartões delas eram usados de madrugada, longe, em outro lugar. Por isso o banco só via a maquininha.
>
> O técnico põe a lâmina num saquinho plástico. Mateo pensa na tabela de cobranças da sessão 6: as que não tinham mercadoria por trás. Ali estava a mão que faltava.

---

👉 **Para quem é de Administração:** quando um fornecedor apresentar o sistema de IA dele, faça as três perguntas do passo 1; se ele não souber responder, não entende o próprio produto. Pergunte de onde sai cada número. E se o sistema decide sobre pessoas (crédito, seguros, contratação), pergunte onde está o humano que revisa (convém mesmo que a lei não exija em todos os casos).

---

## Exercícios da sessão 8

Os últimos do curso. As soluções estão no final.

**Exercício 30 — Três casos, três formas.** Responda as três perguntas no papel, escreva a forma, desenhe-a com `torch.zeros` e confira-a com `assert`. O item a) vem resolvido.

a) Um lote de 500 compras com 200 dados cada uma, como na nota do BBVA:

```python
bbva = torch.zeros(500, 200)
assert bbva.shape == (500, 200), "a) compras × datos"
```

b) O histórico de 16 clientes em sequências de 100 compras, cada compra passada por um Embedding de 64 números, como no modelo didático do Nubank. Dica: `torch.zeros(16, 100, dtype=torch.long)` e `torch.nn.Embedding(1000, 64)`.
c) Um sinistro com 5 fotos de 720 de altura por 1.280 de largura, na ordem do PyTorch.

Termine com `print("Ejercicio 30 superado")`.

**Exercício 31 — O lote de fotos.** Preencha as lacunas: um sinistro com 6 fotos de 224 × 224, e a quantidade de números calculada de duas maneiras que devem coincidir:

```python
siniestro = ____
assert siniestro.shape == ____, "6 fotos × 3 colores × 224 de alto × 224 de ancho"
cuenta = 1
for tamano in (6, 3, 224, 224):
    cuenta = ____
assert siniestro.numel() == cuenta, "numel debe dar lo mismo que multiplicar la forma con un for"
print("Ejercicio 31 superado:", siniestro.numel(), "números")
```

**Exercício 32 — O dicionário de dona Lupe.** A loja on-line dela tem 4 produtos e ela quer 2 números por produto. Preveja as formas **antes** de executar:

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

**Exercício 33 — De sequência a tabela, sem ajuda.** Uma M5 pequenininha: 100 séries, 28 dias, 3 dados por dia (preço, promoção, feriado). a) Crie com `torch.zeros` o que uma rede que lê sequências veria. b) Converta-o com `reshape` e `-1` na tabela que o LightGBM veria. c) Escreva três `assert`: as duas formas e que tenham a mesma quantidade de números. d) Num comentário, diga o que é uma linha da tabela e o que a forma já não diz.

---

## Erros comuns (e como perceber)

| Erro | Como aparece | Como perceber | O que fazer |
|---|---|---|---|
| Misturar a ordem das cores | Escreve-se `(fotos, alto, ancho, 3)` e o PyTorch espera `(fotos, 3, alto, ancho)` | Quebra o `assert` da forma | Pergunte em que ordem vêm as fotos |
| Pedir um produto fora da tabela | `IndexError: index out of range in self` | Algum número é ≥ ao primeiro tamanho do Embedding | N produtos vão de 0 a N − 1 |
| Dar decimais a um Embedding | `RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.FloatTensor instead (while checking arguments for embedding)` | Diz `FloatTensor` onde esperava inteiros | `torch.tensor([0, 2])`, sem ponto |
| Repetir um número da empresa como fato | “A Mastercard melhora a detecção em 300 %” | A fonte é um comunicado da empresa | “Segundo a empresa”, e diga se é “até” ou “em média” |

---

## Autoavaliação da lição B

Marque com honestidade. Se algo sair “ainda não”, volte a essa sessão.

| Consigo… | Sessão | Sim / Ainda não |
|---|---|---|
| Mudar de forma com `reshape`, usar `-1` e interpretar o erro quando não fecha | 5 | |
| Prever a forma que sai ao operar tensores de formas diferentes | 6 | |
| Distinguir `*` de `@` e saber qual usar | 6 | |
| Explicar perda, gradiente, otimizador e época | 7 | |
| Explicar por que a acurácia sozinha engana, e o que medem precisão e recall | 7 | |
| Ler um caso real e escrever a forma do tensor dele | 8 | |

## Revisão da lição B

Responda sem olhar. Nas de código, preveja e depois execute.

1. Se `meses = torch.arange(12)`, que forma dá `meses.reshape(4, -1)`? E `meses.reshape(-1, 6)`?
2. O que acontece com `meses.reshape(5, -1)`, e por quê?
3. Com a papelaria `(2, 7, 3)` e `precios = torch.tensor([45.0, 12.0, 350.0])`, que forma tem `papeleria * precios`?
4. `torch.zeros(2, 7, 3) * torch.zeros(7)` funciona?
5. Com `a = [[1, 2], [3, 4]]` e `b = [[10, 20], [30, 40]]` como tensores, quanto dão `a * b` e `a @ b`? Por que `*` é uma armadilha que não acusa erro?
6. Se `tabla` é `(14, 3)` e `precios` é `(3,)`, que forma tem `tabla @ precios` e o que é cada número?
7. O que fazem `perdida.backward()` e `optimizador.step()`?
8. O que é uma época?
9. De 1.000 compras, 50 são fraude. O modelo dispara 40 alertas e 30 são fraude de verdade. Calcule acurácia, precisão e recall. Qual preocupa mais o banco?
10. Escreva a forma de: (a) 64 compras com 4 dados; (b) 32 clientes com as últimas 30 compras de 4 dados; (c) 8 fotos de 480 × 640 na ordem do PyTorch.

---

🔁 **Para fechar o curso**

**Quarta-feira, 23 de dezembro, 19h30. Papelaria, filial Centro.**

Dona Lupe chega com uma caixa de confeitaria amarrada com fita. Bolo três leites. Põe-na sobre o balcão, entre os grampeadores e os rolos de papel lustroso.

— Este eu paguei com cartão — diz. — Na maquininha do Norte. Vamos ver se agora cobram o que é certo.

A Fer corta o bolo. Fica com o pedaço maior, e ninguém diz nada, porque todos sabem por quê: foi ela que em setembro escreveu que faltava mercadoria, e ninguém leu o e-mail dela.

Mateo come de pé, encostado na vitrine. Lá fora alguém baixa uma porta de enrolar. Cheira a papel novo e a canela.

— E agora, o que vai fazer? — pergunta dona Lupe. — Já não há nada para salvar.

Mateo pensou nisso a semana inteira. Em novembro queria terminar a faculdade e pronto. Agora sabe do que gosta: que uma pilha de números diga quem mente. Quer estudar isso. Dados. Modelos. E as letras miúdas.

— Isto, tia. O que a gente fez.

— Cuidar de papelarias?

— Mais ou menos.

Naquela noite abre o notebook. Uma planilha nova, em branco. Nenhuma forma ainda, nenhum eixo.

Escreve uma única linha:

`# E agora?`

Quem chegou até aqui está igual a ele. Em oito sessões, o caminho foi de um número solto a ler como trabalham um banco, uma seguradora e uma loja. Não fica uma fórmula: fica uma maneira de olhar.

| Sessão | O que se aprendeu |
|---|---|
| 1 | O que é uma dimensão e o que é a forma, com listas de Python |
| 2 | O primeiro tensor: `torch.tensor`, `shape` e `dtype` |
| 3 | Extrair um pedaço do tensor por qualquer um dos eixos |
| 4 | Somar e tirar a média por eixo, comparando com um `for` |
| 5 | Mudar de forma com `reshape` sem mudar os dados |
| 6 | Operar tensores inteiros, e a diferença entre `*` e `@` |
| 7 | Treinar um detector de fraude, e desconfiar da acurácia |
| 8 | Ler casos reais, embeddings e os cuidados do crédito automático |

---

## Soluções (não olhe antes de tentar)

### Revisão para abrir

1. `(compras, dados de cada compra)`: na sessão 7, `(4000, 4)`.
2. Mede **o quanto o modelo prevê mal**, num único número; se cai, o modelo está aprendendo.
3. Porque quase todas as compras são boas: um “detector” que nunca dispara já acerta cerca de 84 %. Com 91 %, o nosso ainda deixava passar 1 de cada 3 fraudes.

### Exercícios

**30.** b) `(16, 100, 64)`: clientes × compras × números do embedding. O 1000 é quantos produtos diferentes cabem na tabela; não muda a forma da saída. c) `(5, 3, 720, 1280)`.

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

Saída: `Ejercicio 30 superado`.

**31.** As lacunas são `torch.zeros(6, 3, 224, 224)`, `(6, 3, 224, 224)` e `cuenta * tamano`. Saída: `Ejercicio 31 superado: 903168 números`.

**32.** `Embedding(4, 2)`, `(4, 2)`, `(3, 5)` e `(3, 5, 2)`: à forma da entrada acrescenta-se no final um eixo de tamanho 2. Saída: `Ejercicio 32 superado`.

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

### Revisão da lição B

Saídas reais das perguntas 1 a 6:

```
1) torch.Size([4, 3]) torch.Size([2, 6])
2) shape '[5, -1]' is invalid for input of size 12
3) torch.Size([2, 7, 3])
4) The size of tensor a (3) must match the size of tensor b (7) at non-singleton dimension 2
5) a * b = [[10, 40], [90, 160]] | a @ b = [[70, 100], [150, 220]]
6) torch.Size([14])
```

1. `(4, 3)` e `(2, 6)`: o `-1` é “calcule por conta própria”.
2. Quebra: 12 números não se dividem em 5 linhas iguais.
3. `(2, 7, 3)`: os preços são esticados para todas as filiais e dias.
4. Não: o PyTorch compara as formas da direita para a esquerda, e o último eixo mede 3 contra 7.
5. `*` é célula por célula; `@`, linha por coluna. Com estas formas as duas funcionam: usar a errada não quebra, só dá outro número. Por isso se compara com um `for`.
6. `(14,)`: o total em pesos de cada filial em cada dia (quantidades vezes preços, somadas).
7. `perdida.backward()` calcula os gradientes da perda em relação a cada peso; `optimizador.step()` usa esses gradientes para mover os pesos um passinho e fazer a perda cair.
8. Uma passada completa por todos os dados de treinamento.
9. Compras boas sem alerta: 1.000 − 30 − 10 − 20 = 940. Acurácia = 970 ÷ 1.000 = 0.97; precisão = 30 ÷ 40 = 0.75; recall = 30 ÷ 50 = 0.6. Os 97 % soam excelentes, mas passam 4 de cada 10 fraudes.
10. `(64, 4)`, `(32, 30, 4)` e `(8, 3, 480, 640)`: os tensores do passo 1.

---

## O que vem depois

Uma ordem possível para continuar (em inglês, menos o último):

- **Para a intuição:** a série *Neural Networks* do **3Blue1Brown**, vídeos animados que mostram o que uma rede faz por dentro. Comece por “But what is a neural network?”. Servem para entender, não para praticar.
- **Para construir uma rede a partir de um único número:** *Neural Networks: Zero to Hero*, de **Andrej Karpathy**. O primeiro vídeo, *micrograd* (2 horas e 25 minutos), constrói do zero o que `perdida.backward()` faz; ele diz que basta Python básico e uma vaga lembrança do cálculo do ensino médio. Vai em ritmo de especialista: vale assistir com pausas.
- **Para redes maiores em PyTorch:** *Learn PyTorch for Deep Learning*, de Daniel Bourke (learnpytorch.io), com exercícios e soluções para comparar.
- **Em espanhol:** os vídeos do **DotCSV** para se motivar e, como meta, o curso de pós-graduação *Aprendizaje Profundo* da UNAM, com o material aberto no GitHub. Se um dia esses notebooks fizerem sentido, este curso cumpriu o papel dele.

Um último conselho: quando algo não funcionar, imprima `.shape`. Ler isso já não é problema.

---

*Escrevo a partir do que vemos todos os dias na Hábil, onde construímos sistemas para bancos, seguradoras e empresas no México. Se no trabalho ou no negócio da família aparecer um problema assim, vamos conversar: [habil.mx](https://www.habil.mx/pt/)*
