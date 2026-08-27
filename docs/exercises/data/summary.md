# Resumo de resultados

Tabela-índice dos números que calculei nos três exercícios (última seção do relatório,
conforme o enunciado). Cada valor está reportado e discutido na seção correspondente do
notebook indicado.

| # | Item | Valor |
|---|---|---|
| 1 | Mixing rate em $s = 0.5$ | **0.00%** (0/400) |
| 2 | Mixing rate em $s = 1.0$ | **5.00%** (20/400) |
| 3 | Mixing rate em $s = 2.0$ | **19.25%** (77/400) |
| 4 | Mixing rate em $s = 4.0$ | **48.25%** (193/400) |
| 5 | Menor $r_{ij}$ em $s = 1.0$, e qual par | **1.326**, par **(0, 1)** |
| 6 | Distância entre centros — Dataset I | **3.2282** |
| 7 | Distância entre centros — Dataset II | **0.2662** |
| 8 | Variância explicada PC1 + PC2 — Dataset I | **0.6597** (0.5004 + 0.1593) |
| 9 | Variância explicada PC1 + PC2 — Dataset II | **0.4291** (0.2159 + 0.2132) |
| 10 | Proporção da classe positiva em `Transported` | **50.36%** (4378/8693) |
| 11 | Média e mediana de `FoodCourt` no treino, antes de transformar | **452.61** / **0.00** |
| 12 | `shape` final da matriz de features de treino | **(6954, 17)** |
| 13 | Mínimo e máximo de treino e teste após o scaling | treino **[−1.0000, 1.0000]** · teste **[−1.0000, 1.1383]** |

Fontes: itens 1–5 no [Ex. 1](ex1_point_clouds.ipynb), itens 6–9 no
[Ex. 2](ex2_nonlinearity.ipynb), itens 10–13 no
[Ex. 3](ex3_preprocessing.ipynb).
