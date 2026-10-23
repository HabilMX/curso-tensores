# Tensores do zero

**Por Dorian Chávez, fundador da Hábil e arquiteto de integração.**

Da lista de Python que já se sabe escrever a um detector de fraude que aprende com os dados.

**Para quem é:** estudantes de Sistemas que já programam um pouco em Python, e pessoas de Administração que querem
entender o que há por baixo da inteligência artificial usada em bancos, seguradoras e lojas. É um curso de
nível intermediário: não explica o que é uma variável nem um laço, mas cada palavra nova é explicada quando aparece. Cada
sessão traz um quadro “Para quem é de Administração” com a pergunta que vale a pena fazer em uma reunião.

**O que se sabe ao terminar:**

- Criar um tensor do PyTorch a partir de dados reais e dizer a forma dele antes de executar o código.
- Extrair e resumir dados por qualquer um dos eixos, e conferir cada resultado com o mesmo cálculo feito com um `for`.
- Interpretar um erro de forma do PyTorch, palavra por palavra, e corrigi-lo.
- Treinar um pequeno detector de fraude e explicar por que acertar 91 % não basta.

**O que é preciso antes de começar:** Python básico (variáveis, listas, laços `for` e funções; as lições 0 a 4
do [curso de Python](https://github.com/HabilMX/curso-python) bastam), Python 3.11 ou mais recente e um terminal. A
primeira sessão não instala nada; o PyTorch é instalado na segunda. Testado no Linux (Mint 22.3 e Ubuntu 24.04) e no
macOS; no Windows não foi testado. Não é preciso placa de vídeo.

## As oito sessões

Duas lições de quatro sessões, de umas 2 horas cada: mais ou menos uma para ler e acompanhar o exemplo, e outra para
praticar. Todas seguem a mesma história: uma papelaria, uma maquininha de cartão que cobra a mais e um estudante que
tem até 23 de dezembro para entender o que aconteceu.

| Sessão | | O que se aprende |
|---|---|---|
| 1 | [Um número, uma lista, uma tabela, um cubo](01-un-numero-una-lista-una-tabla-un-cubo.md) | o que é um tensor, com as listas do Python; contar dimensões; de onde veio a palavra |
| 2 | [O primeiro tensor](02-tu-primer-tensor.md) | instalar o PyTorch; `torch.tensor`, `shape` e `dtype`; PyTorch e TensorFlow |
| 3 | [Os eixos têm nome](03-los-ejes-tienen-nombre.md) | extrair um pedaço por qualquer eixo; o que acontece com a forma |
| 4 | [Somar por aba](04-sumar-por-pestana.md) | somar e tirar a média ao longo de um eixo com `dim` |
| 5 | [Mudar de forma sem mudar os dados](05-cambiar-de-forma.md) | `reshape` e como interpretar um erro de forma |
| 6 | [Operar tensores inteiros](06-operar-tensores-completos.md) | broadcasting, e a armadilha de `*` versus `@` |
| 7 | [Um detector de fraude que aprende com os dados](07-detector-de-fraude.md) | o ciclo de treinamento; acurácia, recall e precisão |
| 8 | [Tensores no banco, na seguradora e na loja](08-banco-aseguradora-tienda.md) | ler um caso real e dizer a forma do tensor dele; os cuidados que não são opcionais |

## O código

Cada sessão tem a própria pasta neste repositório ([`sesion-01`](../sesion-01) a [`sesion-08`](../sesion-08)), com cada
programa e a saída que ele deve produzir. `verificar.py` executa todos e os compara com o que a lição diz. A
instalação passo a passo está no [README principal](../README.md).

## Onde é publicado

Uma sessão por semana no “En construcción”, o Substack da Hábil, às quartas-feiras, às 8h00, de 4 de novembro a 23 de
dezembro de 2026: https://enconstruccionmx.substack.com
