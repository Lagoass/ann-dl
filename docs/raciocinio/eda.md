# Raciocínio — Projeto, entrega 1: EDA

Como pensamos o [EDA do Adult Income](../projects/eda/index.md), na ordem real em que as
decisões apareceram. A prova de projeto (19/nov) é nota **de equipe** e limita a nota do
projeto: os dois integrantes precisam conseguir reconstruir cada passo daqui.

## Etapa 0 — Ler o enunciado caçando o que custa nota

Antes de abrir o dataset, procuramos onde se perde ponto. Achamos quatro coisas:

- **O prazo era o próprio dia (08/out).**
- **A proposta de dataset é pré-requisito:** EDA sobre dataset não aprovado não recebe nota.
  E o formulário só oferece os 4 datasets de exemplo do enunciado (mais "Outro"). Escolher
  um dos quatro foi a forma de não depender de uma aprovação que não daria tempo de obter.
- **O peso está na etapa 4 (3.5 dos 10 pontos)**, e o critério cheio exige que cada escolha
  aponte o achado que a motiva.
- **No máximo 3 figuras por item** nas etapas 2 e 3, e cada uma termina com uma conclusão.

O starting point da disciplina (laboratório Palmer Penguins) virou referência de rigor, não
modelo. Dele trouxemos quatro hábitos: provar que nada vazou, comparar confiabilidade com
silhueta, rodar um controle com ruído e o argumento geométrico contra `drop="first"`.

## Etapa 1 — Escolher o dataset

Entre os quatro, ficamos com o **Adult Income**, por três motivos:

1. **Classificação** combina com a MLP da entrega seguinte.
2. **Tem matéria-prima para cada etapa do EDA:** tipos misturados, faltantes, teto de
   valores e alta cardinalidade.
3. **Dá para baixar da fonte original (UCI)** sem depender de login no Kaggle.

Versionamos o `adult.data` sem alteração nenhuma — reprodutibilidade começa pelo arquivo.

## Etapa 2 — Um obstáculo que não era do projeto

Na primeira execução, o Windows bloqueou as DLLs do `scipy`. O Smart App Control estava
ligado, e sem `scipy` não rodam nem `sklearn` nem UMAP. Duas saídas foram descartadas:
desligar a proteção não é uma decisão técnica do projeto, e contornar o bloqueio na própria
máquina seria burlar a segurança dela. Então movemos a **execução para o GitHub Actions**:
um workflow roda os scripts num clone limpo, em Linux, e devolve as figuras e a saída para o
repositório. O obstáculo virou evidência: a cada push, o item "code that runs from a clean
clone" da rubrica é provado de novo.

## Etapa 3 — Medir antes de decidir

Antes de qualquer decisão de pré-processamento, levantamos os fatos. Cinco deles mudaram o
plano:

- **Os faltantes andam juntos.** As 1836 linhas sem `workclass` também estão sem
  `occupation`, e nelas só 10.40% ganham `>50K` (contra 24.08% no geral). A ausência é
  informativa. Daí a categoria "Unknown" no lugar da moda.
- **A regra do IQR seria um desastre.** Em `capital-gain`, Q1 = Q3 = 0, então todo valor
  diferente de zero vira "outlier". São as pessoas onde está 21% da classe positiva. Daí
  `log1p`, e nenhuma linha removida.
- **`education` é uma bijeção com `education-num`.** Daí descartar uma das duas.
- **`fnlwgt` é peso amostral** e tem ρ = −0.011 com o alvo. Daí descartá-la.
- **`capital-gain` ≥ 7000 implica `>50K` em 98.6% dos casos.** A documentação mostra que a
  extração filtrou por *AGI*, que inclui ganhos de capital. Mantivemos a coluna, mas
  marcada como risco — não é vazamento temporal, mas é uma relação quase mecânica.

## Etapa 4 — Split antes de qualquer decisão

O split 80/20 estratificado vem no item 1D, e a regra que seguimos foi mais estrita que o
mínimo: **da etapa 2 em diante, até a análise descritiva usa só o treino**. Se a assimetria
que justifica o `log1p` fosse medida no dataset inteiro, a *decisão* já teria visto o teste.
A única limpeza antes do split foi remover as 24 duplicatas exatas. Isso não aprende nenhum
parâmetro e impede que a mesma linha caia nos dois lados.

## Etapa 5 — Olhar cada figura (e descobrir que elas mentiam)

Os scripts rodaram de primeira, mas olhar as figuras uma a uma revelou **três defeitos** que
nenhuma checagem de código pegaria:

1. **Vales falsos no histograma de idade.** 40 bins sobre 73 idades inteiras fazem alguns
   bins pegarem um ano e outros dois, e isso desenha vales que não existem. Alguém leria ali
   uma distribuição com vários picos. Correção: um bin por inteiro.
2. **A barra do teto 99999 tinha sumido.** A borda final do `logspace` saía como
   99998.999… por arredondamento de ponto flutuante, e os 124 valores no teto ficavam fora
   do último bin. O gráfico escondia justamente o achado que a seta dele anunciava.
3. **Legendas com as duas cores iguais**, porque uma barra vazia não carrega a cor para a
   legenda.

A lição: uma figura também é código e também pode estar errada. "Rodou sem erro" não quer
dizer "mostra a verdade".

## Etapa 6 — A hipótese das ilhas

No t-SNE e no UMAP apareceram regiões grandes e duas ilhas pequenas, destacadas e quase só
`>50K`. Deixamos o título da Figura 13 **neutro de propósito**, porque a primeira versão já
afirmava uma conclusão antes de ela existir. Colorindo por `relationship`, as regiões
grandes se explicaram, mas as ilhas não: elas misturam vários `relationship`.

A hipótese seguinte foi que as ilhas seriam as pessoas com ganho e com perda de capital.
São duas porque as duas condições nunca aparecem juntas. Testamos com número, não com o
olho: a silhueta por esse agrupamento deu **0.748** no UMAP, a maior de todas as
rotulagens, contra no máximo 0.253 pela renda. Hipótese confirmada.

O mecanismo também faz sentido. Depois do `log1p` e da padronização, 91.6% das pessoas
ficam no mesmo valor (zero) e as outras ficam vários desvios longe dele. Os métodos não
lineares transformam esse salto numa ilha.

## Etapa 7 — O controle com ruído

Embaralhando cada coluna de forma independente, a silhueta pela renda caiu de 0.147 para
−0.013. A estrutura de renda que o mapa mostra é real. Mas a confiabilidade continuou em
0.945 nos dados embaralhados. O t-SNE preserva fielmente até as vizinhanças do ruído, e por
isso **confiabilidade alta sozinha não prova que o mapa mostra algo com significado**.

## Etapa 8 — O pipeline como consequência dos achados

Nenhuma escolha do `pipeline.py` foi "padrão":

- **mediana** porque as numéricas têm cauda;
- **"Unknown"** porque a ausência é informativa;
- **`log1p`** porque existe o teto e a armadilha do IQR;
- **one-hot completo** porque `drop="first"` distorce distâncias;
- **balde `infrequent`** por causa das 39 categorias raras de `native-country`, e o mesmo
  balde recebe categorias novas do teste;
- **padronização** porque as escalas vão de 1–16 a 0–99999, e não min-max, por causa dos
  valores no teto.

E provamos que o ajuste foi feito só no treino: a média das numéricas transformadas é 0 no
treino e **não** é 0 no teste.

## O que mudou no nosso entendimento

- EDA não é um álbum de gráficos. Cada achado tem que virar uma linha do pré-processamento,
  ou um risco com plano.
- A pergunta "isso é outlier?" depende da distribuição. Em dados com massa em zero, a regra
  do IQR perde o sentido.
- Faltante não é só um buraco. O *padrão* da ausência pode ser a informação.
- Mapas não lineares mostram estrutura, mas a estrutura pode vir das features (one-hot,
  degraus) e não do alvo. Verificar o que forma as ilhas é parte da leitura.

## Perguntas que esperamos na prova de projeto

| Pergunta provável | Resposta em uma linha |
|---|---|
| Por que "Unknown" e não a moda nos faltantes? | Porque a ausência é informativa: 10.48% de `>50K` contra 24.09%. A moda apagaria esse sinal e contaminaria `Private`, que cairia de 21.90% para 21.03%. |
| Por que não removeram os outliers? | Porque a regra do IQR marca toda pessoa com ganho de capital (Q1 = Q3 = 0), e essas pessoas são 21.3% da classe positiva. O `log1p` resolve a escala sem perder ninguém. |
| `capital-gain` não é vazamento? | É uma relação quase mecânica (≥ 7000 → 98.6% `>50K`), mas está no mesmo registro e estaria disponível na hora de prever. Mantivemos e vamos treinar com e sem ela. |
| Por que Spearman e não Pearson? | `capital-*` têm assimetria de +12 e +4.6, e `education-num` é ordinal. Pearson subestima: em `age × hours`, dá 0.069 contra 0.145 do Spearman. |
| Por que padronização e não min-max? | Os valores no teto que mantivemos (90 anos, 99 horas) ditariam o min-max e espremeriam o miolo dos dados. |
| Por que one-hot sem `drop="first"`? | Porque a categoria descartada vira a origem e fica à distância 1 das outras, enquanto as outras ficam a √2 entre si: proximidade inventada para métodos de distância. Com one-hot completo, todas são equidistantes, e uma rede com viés não precisa do descarte. |
| Como o pipeline trata uma categoria nova no teste? | `handle_unknown="infrequent_if_exist"` a manda para o balde `infrequent`. Testamos com `occupation="Astronaut"`. |
| O que são as ilhas do t-SNE e do UMAP? | As pessoas com ganho e com perda de capital, que nunca coincidem. A silhueta por esse agrupamento é 0.748 no UMAP, contra 0.253 pela renda. |
| Como provam que o escalonador não viu o teste? | A média das numéricas transformadas é 0 no treino e diferente de 0 no teste (0.013, −0.007…). |
| Por que só 30.37% em PC1+PC2? | As 50 colunas têm 43 direções reais (7 dependências do one-hot) e a variância se espalha: são precisos 10 componentes para 80%. |
