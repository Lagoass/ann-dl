# Exercício 1 — Data

!!! abstract "Status"

    :material-check-circle: **Entregue** — prazo 27/ago (qui), 23:59.

    [Enunciado oficial](https://insper.github.io/ann-dl/2026.2/exercises/data/){:target="_blank"}

Atividade sobre geração, manipulação e preparação de dados para redes neurais. O fio
condutor é a **dispersão**: quanto uma nuvem de pontos se espalha, em que direção, e
como isso muda a dificuldade do problema de classificação. Nenhum modelo é treinado —
todas as medidas que uso aqui são geométricas ou estatísticas.

## Como ataquei cada parte

| Parte | Abordagem |
|---|---|
| [Ex. 1 — Nuvens de pontos](ex1_point_clouds.ipynb) | Gero 4 nuvens gaussianas 2D e meço a separabilidade com a razão $r_{ij}$ e a mixing rate, observando como as duas degradam com o fator de escala $s$. |
| [Ex. 2 — Não-linearidade](ex2_nonlinearity.ipynb) | Contrasto dois datasets 5D — gaussianas deslocadas × cascas concêntricas — e comparo o que medidas lineares (PCA, distância entre centros) enxergam em cada um. |
| [Ex. 3 — Dados reais](ex3_preprocessing.ipynb) | Pré-processo o Spaceship Titanic para uma rede com `tanh`: split estratificado antes de tudo, imputação, one-hot, `TotalSpend`, `log(1+x)` e escala $[-1,1]$, com toda estatística calculada só no treino. |
| [Resumo de resultados](summary.md) | Tabela-índice com os 13 números exigidos pelo enunciado. |

## Regras técnicas que segui

- Seed fixa `np.random.default_rng(42)` (Ex. 1 e 2) e `random_state=42` no split
  (Ex. 3); cada notebook roda de ponta a ponta e reproduz todos os números citados.
- Bibliotecas: `numpy`, `pandas`, `matplotlib` e `scikit-learn` (PCA e pré-processamento
  apenas).
- Todos os gráficos têm título, rótulos de eixo e legenda de classes; as Figuras 1–6
  seguem a numeração do enunciado. A paleta âmbar/telha/oliva/aço é a mesma em todas as
  figuras.

## Reprodução

``` shell
git clone https://github.com/Lagoass/LagoaNNPages.git
cd LagoaNNPages
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace docs/exercises/data/*.ipynb
```

O dataset do Ex. 3 (`train.csv` do
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic){:target="_blank"})
está versionado em `docs/exercises/data/spaceship-titanic/`.

!!! note "Uso de IA"

    Este trabalho foi desenvolvido com auxílio de IA (Claude), utilizada para estruturar
    o código e o site; todo o código e as análises foram revisados e compreendidos pelo
    autor, conforme a política da disciplina.
