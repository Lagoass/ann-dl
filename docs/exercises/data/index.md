# Exercício 1 — Data

!!! abstract "Status"

    :material-progress-clock: **Em andamento** — prazo **27/ago (qui), 23:59**.

    [Enunciado oficial](https://insper.github.io/ann-dl/2026.2/exercises/data/){:target="_blank"}

Os três exercícios tratam do mesmo tema por ângulos diferentes: **dispersão** e o custo que
ela impõe à fronteira de decisão. O Ex. 1 mede isso em 2D, o Ex. 2 mostra que a *forma* da
distribuição importa tanto quanto a dispersão, e o Ex. 3 aplica em dado real. Nenhum modelo
é treinado.

Semente fixa em todo o relatório: `rng = np.random.default_rng(42)`.

---

## Exercise 1 — Point Clouds: Geometry and Spread in 2D

**Abordagem:** gerar as nuvens, medir separabilidade com $r_{ij}$ e mixing rate, e mostrar
como as duas métricas variam com o fator de escala $s$.

### A — Generate the clouds

??? note "Itens"

    - 400 amostras, 4 classes × 100, 2D, desvio-padrão por eixo.
    - **Figura 1**: scatter por classe, com os centros marcados.

<!-- TODO -->

### B — More or less spread out

??? note "Itens"

    - As mesmas 4 classes geradas 4 vezes, $s \in \{0.5, 1.0, 2.0, 4.0\}$; médias fixas.
    - **Figura 2**: 4 subplots com limites de eixo compartilhados.
    - Tabela dos 6 $r_{ij}$ em $s = 1$, indicando o menor; e o valor desse menor em
      $s = 2$ deduzido de $r_{ij} \propto 1/s$, sem gerar dado novo.
    - As 4 mixing rates e a **Figura 3**: mixing rate × $s$.
    - Responder: a partir de qual $s$ as nuvens deixam de ser separáveis por retas, e o que
      acontece com o menor $r_{ij}$ nesse ponto.

<!-- TODO -->

### C — Analysis

??? note "Itens"

    - Sobreposição em $s = 1$: uma única fronteira linear separa tudo? E um conjunto delas?
    - Esboçar sobre a Figura 1 as fronteiras que uma rede provavelmente aprenderia.
    - Relacionar com o item B: quanto mais espalhadas as nuvens, o que acontece com a região
      onde a rede necessariamente erra?

<!-- TODO -->

!!! success "Aprendizado"

    <!-- TODO -->
    O que mede dificuldade é a distância entre classes **relativa à dispersão**, não a
    distância absoluta — e parte do erro é dos dados, não do modelo.

---

## Exercise 2 — Non-Linearity in Higher Dimensions

**Abordagem:** construir dois datasets 5D com a mesma dimensionalidade e estruturas
diferentes, e comparar o que medidas lineares (PCA, distância entre centros) conseguem
capturar em cada um.

### A — Dataset I: shifted Gaussians

??? note "Itens"

    - 500 + 500 amostras via `rng.multivariate_normal`, com as médias e covariâncias dadas.

<!-- TODO -->

### B — Dataset II: concentric shells

??? note "Itens"

    - Direções uniformes na esfera unitária: $v \sim \mathcal{N}(0, I_5)$,
      $u = v / \lVert v \rVert$.
    - Raios $\rho \sim \mathcal{N}(2{,}0;\ 0{,}4)$ e $\rho \sim \mathcal{N}(5{,}0;\ 0{,}4)$;
      $x = \rho \cdot u$.
    - Declarar em uma linha que `0.4` foi tratado como desvio-padrão.

<!-- TODO -->

### C — Visualize and compare

??? note "Itens"

    - **Figura 4**: PCA para 2D nos dois datasets, lado a lado.
    - Variância explicada de PC1 + PC2 em cada caso, e em qual a projeção preserva melhor a
      informação de classe.
    - Em 5D: distância entre os centros das classes, nos dois datasets.
    - **Figura 5**: histograma do raio $\lVert x \rVert$, as duas classes sobrepostas.

<!-- TODO -->

### D — Analysis

??? note "Itens"

    1. Centros quase coincidentes + raios bem separados: o que isso diz sobre separar com um
       hiperplano?
    2. Por que nenhuma quantidade de dados resolve o Dataset II com fronteira linear?
    3. Uma projeção PCA bagunçada prova inseparabilidade? Justificar com os próprios
       resultados e escrever a função das entradas que separa o Dataset II.

<!-- TODO -->

!!! success "Aprendizado"

    <!-- TODO -->
    Distância entre centros não mede separabilidade, e uma ferramenta linear não diagnostica
    estrutura não-linear. O que falta é a representação, não o dado.

---

## Exercise 3 — Preparing Real-World Data for a Neural Network

**Abordagem:** pipeline fitado apenas no treino, com as transformações necessárias para
alimentar uma rede com `tanh`.

### A — Get to know the data

??? note "Itens"

    - [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic){:target="_blank"},
      `train.csv`.
    - O que é `Transported` e o balanço de classes.
    - Features numéricas × categóricas.
    - Tabela de missing values por coluna, em contagem e percentual.
    - Média, mediana e máximo das 5 colunas de gasto, com a leitura de média × mediana.

<!-- TODO -->

### B — Split before you transform

??? note "Itens"

    - Split 80/20 estratificado, semente fixa.
    - Duas ou três frases sobre por que o split vem antes da imputação e do scaling.

<!-- TODO -->

### C — Preprocess

??? note "Itens"

    - **Missing**: estratégia por tipo de coluna, imputer fitado no treino.
    - **Categóricas**: one-hot em `HomePlanet`, `CryoSleep`, `Destination`, `VIP`; explicar o
      tratamento de categoria ausente no treino.
    - **Feature engineering**: `TotalSpend`; dropar `Cabin`, `Name`, `PassengerId`.
    - **Caudas**: $\log(1+x)$ nas colunas de gasto, com histograma de uma delas antes e
      depois, e por que isso ajuda com `tanh`.
    - **Escala**: normalização para $[-1, 1]$, com min/max reportados.

    Ordem: imputação → `TotalSpend` → `log1p` → scaling.

<!-- TODO -->

### D — Verify and visualize

??? note "Itens"

    - **Figura 6**: histograma de uma feature de cauda pesada antes e depois.
    - Checagens: nenhum `NaN`, `shape` final, faixa de valores compatível com `tanh`.
    - Um parágrafo: qual decisão de pré-processamento mais afetaria o treino, e por quê.

<!-- TODO -->

!!! success "Aprendizado"

    <!-- TODO -->
    O pré-processamento define a geometria que a rede recebe, e o split fitado só no treino é
    o que faz o número reportado significar alguma coisa.

---

## Results summary

| # | Item | Valor |
|---|---|---|
| 1 | Mixing rate em $s = 0.5$ | |
| 2 | Mixing rate em $s = 1.0$ | |
| 3 | Mixing rate em $s = 2.0$ | |
| 4 | Mixing rate em $s = 4.0$ | |
| 5 | Menor $r_{ij}$ em $s = 1.0$, e qual par | |
| 6 | Distância entre centros — Dataset I | |
| 7 | Distância entre centros — Dataset II | |
| 8 | Variância explicada PC1 + PC2 — Dataset I | |
| 9 | Variância explicada PC1 + PC2 — Dataset II | |
| 10 | Proporção da classe positiva em `Transported` | |
| 11 | Média e mediana de `FoodCourt` no treino, antes de transformar | |
| 12 | `shape` final da matriz de features de treino | |
| 13 | Mínimo e máximo de treino e teste após o scaling | |
