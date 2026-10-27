# Raciocínio — Exercício MLP

Como pensei o [relatório do MLP](../exercises/mlp/index.md), na ordem real das decisões —
incluindo as três vezes em que um número me obrigou a investigar antes de seguir.

## Etapa 0 — Ler o enunciado caçando o que custa nota

- **A MLP inteira é escrita à mão.** Usar biblioteca para o modelo anula o critério de
  implementação.
- **O scikit-learn só entra em quatro pontos:** gerar os dados, dividir treino/teste, montar
  a matriz de confusão e rodar os dois baselines de regressão logística, que o próprio
  enunciado autoriza.
- **O tamanho das camadas é argumento.** Uma rede de duas camadas fixas custa ponto nos
  Exercícios 3 e 4.
- **Vale +1** se o núcleo do Exercício 2 for reusado **intocado** no Exercício 3.
- **No Exercício 4, relatar com honestidade que a profundidade não ajudou vale nota cheia.**
  O que se avalia é o método, não o número.

## Etapa 1 — Decidir o design antes de escrever código

O ponto extra se decide no design. Se o backward começar de "∂L/∂ŷ", cada perda precisa de
um código diferente e o núcleo muda. A saída foi fazer **cada perda devolver ∂L/∂u**, o
gradiente na pré-ativação da saída, junto com a sua própria ativação de saída. Para
sigmoide + BCE e para softmax + entropia cruzada, esse gradiente é o mesmo, ŷ − y. Com
isso, forward, backward e update ficaram idênticos para qualquer perda, e o Exercício 3
virou só uma troca de argumentos. Bônus: a mesma classe, com a perda `MSETanh`, também
reproduz a conta do Exercício 1.

## Etapa 2 — Provar o backward antes de treinar qualquer coisa

Uma rede com backward errado ainda "aprende" alguma coisa, e o erro passa despercebido. Por
isso, antes de treinar, comparei cada gradiente com diferenças finitas em 5 redes (as três
perdas, até 3 camadas escondidas).

**Surpresa 1:** o pior erro relativo deu 1.23e-06, acima do limite de 1e-06 que eu mesmo
fixei, e o script disse "COM ERRO". Em vez de afrouxar o limite, investiguei. Repetindo com
vários ε, o erro *aumentava* quando ε diminuía (1.05e-07 com ε = 1e-5, 1.23e-06 com 1e-6,
6.83e-06 com 1e-7), sempre num gradiente minúsculo (3e-05). Essa é a assinatura do
arredondamento de ponto flutuante na subtração `L(θ+ε) − L(θ−ε)`. Uma derivada errada daria
erro de ordem 1, constante em ε. Com ε = 1e-5, o equilíbrio clássico entre o erro de
truncamento e o de arredondamento, o pior caso caiu para 1.05e-07.

## Etapa 3 — O Exercício 1 à mão, e o cruzamento com o código

Fiz a dedução inteira à mão, na regra da cadeia de trás para frente. Para conferir, rodei as
mesmas fórmulas em `numpy` e depois a **classe genérica** com os pesos do enunciado. A
diferença foi **0.00e+00**: a conta à mão valida o código, e o código valida a conta.

Dois cuidados que surgiram aqui:

- **A rede tem 9 parâmetros escalares, não 8** (4 + 2 + 2 + 1). Reportei os nove e anotei a
  contagem.
- **A "perda cai?" virou um número.** Ela cai de 0.4004 para 0.0834. A aproximação de
  primeira ordem previa −η‖∇L‖² = −0.536; a queda real foi −0.317. A diferença é a curvatura:
  a tanh satura quando ŷ se aproxima de 1.

## Etapa 4 — Hiperparâmetros só pela perda de treino

Escolher o learning rate pela acurácia de teste seria usar o teste para decidir, o mesmo
vazamento que evitei no Data. Comparei learning rates olhando só a perda de treino:

- **Ex. 2:** com 0.5, a perda na época 1000 (0.1130) já é menor que a de 0.1 na época 2000
  (0.1174). As duas ainda caem devagar no fim; 0.5 só chega antes. A comparação está na
  `saida.txt`. Fiquei com 0.5 e 2000 épocas.
- **Ex. 3:** mantive 0.5 e usei 3000 épocas, olhando a Figura 4. A perda ainda cai devagar
  depois disso, mas o que importa para o Ex. 4 é dar o mesmo orçamento às duas redes.

## Etapa 5 — Exercício 2: explicar o "pior que chutar"

Minha hipótese, antes de rodar: a distribuição dos dados é **simétrica** em relação à origem. A perda
logística é convexa, então o mínimo tem que ficar em w ≈ 0, e a regressão logística
"desiste". Os números confirmaram (‖w‖ = 0.0297, probabilidade entre 0.457 e 0.544 no teste).

O que eu não esperava era a explicação exata do 47.50%. A reta quase nula acerta um bloco
inteiro, erra o outro inteiro e corta a classe 0 ao meio. No split de teste, o bloco errado
tem 58 pontos e o certo tem 42: (53 + 42 + 0)/200 = 47.50%. O "abaixo de 50%" vem do
split: cada bloco tem 250 pontos, então o maior no treino (para onde a regressão pende) é o
menor no teste. Não é limite da reta. Para provar isso, procurei a **melhor reta possível**: ela faz
69.00% no teste. A regressão logística simplesmente não a encontra.

## Etapa 6 — Exercício 3: prever o par confundido antes de olhar a matriz

O enunciado pede para explicar a confusão pela estrutura dos clusters, mas o
`make_classification` não diz a que cluster cada ponto pertence. A saída foi gerar o
**mesmo** dataset com `shuffle=False`. O embaralhamento é o último passo do gerador, então os
pontos são os mesmos, só que em ordem. Conferi que as normas de todas as linhas coincidem
antes de confiar nisso.

Com os 6 clusters na mão, usei de novo a régua do Data (r = distância / soma dos desvios). O
par de classes 1 e 2 tinha o menor r (0.837 < 1, nuvens que se atravessam) e o menor r médio.
**Previ o par 1 e 2 antes de ver a matriz de confusão, e foi esse o par que ela deu** (20
trocas contra 17 do par 0 e 1).

**Surpresa 2:** a verificação me mostrou que eu tinha acertado por sorte. Com as seeds 42 a
46, o par 1 e 2 lidera em só 2 seeds; nas outras 3 lidera o 0 e 1 (somando, 92 contra 95).
O que se repete em todas é a classe 1 nos dois pares do topo. A régua explica isso: o cluster
1 é o único que se atravessa com clusters de **duas** outras classes. Lição: uma previsão que
acerta uma seed não é evidência; é preciso repetir.

O baseline: a regressão logística só faz regiões convexas, e a perda dela se concentra no
cluster 2 (41.3%), que a MLP recupera (84.8%). Pensei primeiro numa explicação pelo ponto
médio entre os clusters da classe 2, mas ela dependia de ruído da amostra: nos centroides
exatos, as distâncias empatam. O que pesa é a forma esticada de cada nuvem em 4D, então
fiquei com a medida direta, cluster a cluster.

## Etapa 7 — Exercício 4: um empate honesto, e o que a figura revelou

A comparação controlada muda só a profundidade (16 → 16-16) e usa 5 seeds. Para garantir que
o orçamento era o mesmo, conferi que a rede rasa com seed 42 reproduz o Exercício 3
exatamente.

O resultado foi um empate: **+0.40 p.p.**, com desvio de 2.44 p.p. nas diferenças. Mas a
perda de treino caiu 42%: a rede funda ajusta muito melhor o treino sem ganhar no teste, e a
distância entre treino e teste cresce de 7.65 para 10.68 p.p. É sobreajuste.

**Surpresa 3:** a Figura 6 mostrou picos na curva da rede funda. Antes de escrever "instável",
medi: em 4 das 5 seeds, entre 449 e 1231 épocas *aumentam* a perda, e a rede rasa nunca
piora. Com o mesmo learning rate, a descida do gradiente da rede funda está no limite da
estabilidade. É o "custo em otimização" que o enunciado pede, medido em vez de suposto.

## O que mudou no meu entendimento

- **Backprop é contabilidade da regra da cadeia.** O par softmax + entropia cruzada é especial
  porque o log desfaz a exponencial, e o gradiente vira só o erro de previsão.
- **O design do código pode refletir a matemática:** se todas as perdas entregam ∂L/∂u, o
  backward é um só.
- **"Pior que chutar" não quer dizer "impossível para uma reta".** Pode querer dizer "a perda
  convexa, por simetria, escolheu não decidir".
- **Profundidade é ferramenta para estrutura composicional.** Neste dado, com lr 0.5 e
  largura 16, ela não comprou teste distinguível; trouxe mais sobreajuste e oscilação.

## Perguntas que eu esperaria na defesa

| Pergunta provável | Minha resposta em uma linha |
|---|---|
| Por que ∂L/∂u = ŷ − y na softmax com entropia cruzada? | O jacobiano da softmax é ŷₖ(δₖⱼ − ŷⱼ); o 1/ŷₖ do log cancela o ŷₖ, e Σyₖ = 1 deixa ŷⱼ − yⱼ. |
| Como você garante que o backward está certo? | Checagem por diferenças finitas em 5 redes: o pior erro relativo é 1.05e-07; uma derivada errada daria erro de ordem 1. |
| Por que ε = 1e-5 e não 1e-7? | Com ε menor, o erro *cresce* (arredondamento na subtração); 1e-5 equilibra truncamento e arredondamento. |
| A perda sobe ou desce na segunda passada do Ex. 1? | Desce, de 0.4004 para 0.0834: o passo vai contra o gradiente, e a primeira ordem dá −η‖∇L‖² < 0. |
| Por que a regressão logística fica abaixo de 50% no Ex. 2? | Por simetria, o mínimo da perda convexa tem w ≈ 0; a reta que sobra pende para o bloco maior do treino (208 contra 192), que, como cada bloco tem 250 pontos, é o menor no teste (42 contra 58). |
| Uma reta pode passar de 50% nesses dados? | Pode: a melhor reta faz 69.00% no teste, sacrificando um bloco. Nenhuma pega os dois, porque o segmento entre eles cruza a classe 0. |
| Como o Ex. 3 reaproveita o Ex. 2 sem mudar código? | A mesma classe `MLP`; mudam só os argumentos (tamanhos e perda), porque toda perda entrega ∂L/∂u. |
| Por que as classes 1 e 2 se confundem? | Os clusters delas têm o menor r (0.837 < 1): se atravessam, e o cluster 2 fica colado nos dois clusters da classe 1. |
| O par 1 e 2 é sempre o mais confundido? | Não: com as seeds 42 a 46, lidera em 2 e o 0 e 1 em 3. O robusto é a classe 1 nos dois pares do topo, porque o cluster 1 se atravessa com clusters de duas outras classes. |
| O que a camada escondida acrescentou no Ex. 3? | Regiões não convexas: +17.3 p.p. no total; o cluster 2, onde a restrição convexa mais pesava, foi de 41.3% para 84.8%. |
| A segunda camada ajudou? | Não de forma distinguível (+0.40 p.p., desvio de 2.44); baixou a perda de treino em 42%, mas aumentou o sobreajuste e a instabilidade. |
| Que dados premiariam a profundidade? | Dados composicionais (tabuleiro com muitas casas, imagens), em que cada camada reaproveita as features da anterior. |