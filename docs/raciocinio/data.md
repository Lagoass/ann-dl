# Raciocínio — Exercício Data

Registro de como pensei cada parte do [relatório do Data](../exercises/data/index.md),
escrito para eu conseguir reconstruir tudo numa defesa, meses depois. A ordem abaixo é a
ordem real em que as decisões apareceram.

## Antes de escrever qualquer código

A primeira coisa que fiz foi ler o enunciado inteiro procurando **o que custa nota**, não o
que fazer. Achei quatro regras transversais: seed fixa (−0,5 se não reproduzir), todo plot
com título/eixos/legenda (−0,5), números citados **no texto** e não só na saída do código
(critério vale metade sem isso), e *data leakage* no Ex. 3 (−1,0). Isso definiu meu método:
calcular os números primeiro, escrever a análise já com eles, e deixar o código reproduzível
de ponta a ponta.

A segunda coisa: o fio condutor declarado é **dispersão**. Os três exercícios são o mesmo
argumento em três cenários — ela impõe limite (2D), ela engana ferramenta linear (5D), e ela
é moldada pelo pré-processamento (dado real).

## Exercise 1 — as decisões, na ordem

**Desvio por eixo, não esférico.** O enunciado dá $\sigma$ como vetor de 2 posições por
classe. A armadilha seria passar um escalar para o `rng.normal`. Conferi que
`rng.normal(mean, std, size=(100, 2))` com `mean` e `std` de shape `(2,)` faz broadcast por
coluna — cada eixo com sua média e seu desvio. É o comportamento que eu quero e sei explicar.

**Reusar a amostra do item A para $s=1$.** O item B pede "as mesmas 4 classes, 4 vezes".
Gerar um $s{=}1$ novo daria números levemente diferentes dos da Figura 1 — e aí a Figura 2 e
a mixing rate de $s{=}1$ não bateriam com o item A. Reusei a mesma amostra e **declarei** no
relatório. Alternativa defensável seria gerar de novo; escolhi consistência interna.

**$r_{ij}$ com parâmetros teóricos.** A fórmula usa $\mu$ e $\bar\sigma$ — usei os do
enunciado, não os empíricos da amostra, porque a razão é uma propriedade da *distribuição*,
não do sorteio. Sanity check que fiz na mão antes de confiar no código:
$\lVert[2,3]-[5,6]\rVert = \sqrt{18} \approx 4.243$, $\bar\sigma_0 = (0.8{+}2.5)/2 = 1.65$,
$\bar\sigma_1 = 1.55$, logo $r_{01} = 4.243/3.20 = 1.326$. O código tinha que devolver isso
— devolveu.

**A dedução de $s=2$ sem gerar dados.** O enunciado sublinha "without generating anything
new" — é um teste de entendimento, não de código. Como $s$ multiplica só os desvios,
$r_{ij} \propto 1/s$: o menor vira $1.326/2 = 0.663$. Um ponto de atenção para a defesa: se
me pedirem para provar, a prova é uma linha (numerador fixo, denominador escala linearmente
com $s$).

**Mixing rate contra as médias teóricas.** O enunciado diz "comparing each point against
the 4 means". Li "the 4 means" como as médias da definição das classes. Com as empíricas o
resultado seria quase idêntico (n=100 por classe), mas declarei a escolha. O cálculo é uma
matriz de distâncias `(400, 4)` + `argmin` — geometria pura, nada treinado, que é o espírito
do exercício.

**O "esboço" virou partição de Voronoi.** O item C pede para esboçar as fronteiras que uma
rede aprenderia. Em vez de desenhar à mão, desenhei a partição pelo centro mais próximo
(grade + `argmin`), porque ela é *exatamente o classificador cuja taxa de erro a mixing rate
mede* — o esboço e a métrica ficam sendo o mesmo objeto, e as fronteiras são retas, que é o
que uma rede pequena consegue. Numerei como "Figura 1b" para não conflitar com a numeração
1–6 do enunciado, e declarei isso.

**Por que respondi "$s = 2$".** Critério que escolhi: em $s{=}2$ o menor $r_{ij}$ cruza
abaixo de 1 (0.663) — centros mais próximos que a soma das dispersões médias, nuvens
literalmente se atravessando — e a mixing rate salta de 5% para 19.25%. Sei que é
discutível ($s{=}1$ já tem 5% de mistura); minha defesa é que em $s{=}1$ o erro se concentra
num único par e um conjunto de retas ainda resolve 95% do problema, enquanto em $s{=}2$ não
existe arranjo de retas que escape de ~1/5 de erro.

## Exercise 2 — as decisões, na ordem

**O `0.4` é desvio ou variância?** $\mathcal{N}(2.0,\ 0.4)$ é ambíguo (a notação matemática
clássica usaria variância; a assinatura do `rng.normal` usa desvio). Tratei como
**desvio-padrão** — é o que `rng.normal(2.0, 0.4)` faz — e declarei em uma linha. Numa
defesa: com variância 0.4 o desvio seria $\sqrt{0.4} \approx 0.63$, as cascas ficariam mais
largas, mas o vão entre elas continuaria (2.0 vs 5.0), então a conclusão não mudaria.

**PCA nos dados crus, sem padronizar antes.** Padronizar apagaria justamente o que o
exercício constrói de propósito ($\Sigma_B$ com variâncias 1.5 contra 1.0 de $\Sigma_A$). O
objeto de estudo é a geometria como ela é.

**O argumento central saiu da Figura 5, não da 4.** A projeção PCA do Dataset II é um borrão
— e o pulo é perceber que isso não prova nada, porque PCA é linear. O que prova é o
histograma de raios: centros a 0.2662 um do outro, raios com um **vão** entre 3.25 (máx do
núcleo) e 3.75 (mín da casca). Daí a cadeia: hiperplano separa por posição ao longo de uma
direção → por simetria esférica as duas classes têm a mesma distribuição em *qualquer*
direção → nenhum hiperplano separa, e mais dados não mudam a família de fronteiras.

**O limiar 12.25 não foi treinado.** O enunciado dá a dica ($\lVert x \rVert^2$). Escolhi o
limiar no ponto médio entre os raios nominais: $((2+5)/2)^2 = 3.5^2 = 12.25$ — uma regra
fixa derivada do enunciado, não ajustada aos dados (ajustar seria "treinar", que é proibido
aqui). A verificação de 100% de acerto é consequência do vão nos raios.

## Exercise 3 — as decisões, na ordem

**A ordem do pipeline foi decidida antes do código.** Split → imputação → `TotalSpend` →
`log1p` → escala. Cada ordem tem um porquê: o split vem primeiro porque toda estatística
posterior sai só do treino; `TotalSpend` depois da imputação porque soma com `NaN` propaga
`NaN`; `log1p` antes da escala porque logaritmo de valor negativo (pós-escala) não existe.

**Mediana e moda por causa do item A, não por convenção.** O próprio item A mostra média
458 × mediana 0 no `FoodCourt` — cauda pesada. Imputar pela média puxaria os buracos para
cima; a mediana imputa o valor típico (gasto 0, idade 27). Nas categóricas, moda não inventa
nível novo.

**Por que dropei `Cabin` (e o que eu diria se questionassem).** O enunciado manda dropar
(`Cabin`, `Name`, `PassengerId`). Mas sei que dá para extrair sinal dela — o
notebook-exemplo da disciplina divide `Cabin` em Deck/Número/Lado antes de descartar. Minha
resposta na defesa: segui o enunciado da edição, que pede o drop explícito; num projeto real
eu testaria a decomposição, porque deck provavelmente correlaciona com `HomePlanet` e gasto.

**`handle_unknown="ignore"` responde uma pergunta literal do enunciado.** "Como seu código
lida com categoria que só aparece no teste?" — vetor todo-zeros no grupo, pipeline não
quebra. Neste dataset as categorias coincidem, mas a garantia é estrutural.

**Escala $[-1,1]$ na mão, de propósito.** `MinMaxScaler` faria o mesmo, mas escrever
$2(x-\min)/(\max-\min)-1$ deixa visível *de onde vêm as estatísticas* (min/max do treino). E
o efeito colateral virou argumento: o teste estoura para **1.1383** em `ShoppingMall` e
`VRDeck` — se eu tivesse fitado no dataset inteiro, o teste caberia certinho em $[-1,1]$ e
isso seria o *sintoma* do vazamento. O estouro é a evidência de que não vazou.

**`random_state=42` inteiro no split.** Detalhe técnico que aprendi verificando: o
`train_test_split` não aceita um `Generator` do NumPy (o `check_random_state` do sklearn só
aceita `None`/int/`RandomState`). Então a regra "mesmo rng em tudo" é fisicamente impossível
ali — uso o inteiro e declaro.

## Perguntas que eu esperaria na defesa

| Pergunta provável | Minha resposta em uma linha |
|---|---|
| Por que a mixing rate usa as médias teóricas? | Porque mede uma propriedade da distribuição; com as empíricas daria ~igual (n=100), e eu declarei a escolha. |
| O que acontece com $r_{ij}$ se eu dobrar $s$? E por quê? | Cai pela metade — numerador (distância entre médias) fixo, denominador linear em $s$. |
| Uma projeção PCA misturada prova inseparabilidade? | Não; PCA é linear e o Dataset II é o contraexemplo: borrão em 2D, 100% separável por $\lVert x \rVert^2$. |
| Por que nenhum hiperplano separa as cascas? | Simetria esférica: em qualquer direção, as projeções das duas classes têm a mesma distribuição em torno de zero. |
| Por que o teste passa de 1 depois da escala? | Min/max vieram só do treino; valores novos maiores estouram — evidência de split honesto, e o `tanh` aceita qualquer real. |
| Por que `log1p` antes da escala e não depois? | Depois da escala há valores negativos — $\log(1+x)$ deixaria de ser monotônica/definida; e o log é quem doma a cauda que ditaria o min/max. |
| Por que mediana e não média na imputação? | O item A mostra média ≫ mediana (cauda pesada): a média é puxada pelos extremos, a mediana imputa o valor típico. |
| Por que você dropou `Cabin` em vez de decompor? | O enunciado manda dropar; sei da alternativa Deck/Número/Lado e a testaria num projeto real. |
