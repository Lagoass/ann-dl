# Raciocínio — Exercício Perceptron

Desta vez planejei antes de codar: nove etapas, cada uma com "o que faço" e "o que espero
entender", e só depois fui para o código — na ordem do plano. O que segue é o plano com o
que **realmente** aconteceu em cada etapa, inclusive as duas surpresas.
O relatório está em [Perceptron](../exercises/perceptron/index.md).

## Etapa 0 — Ler o enunciado caçando o que custa nota

A regra dura desta entrega é que **o perceptron inteiro é escrito à mão** — usar
`sklearn` "voids the implementation criterion". De resto, as mesmas regras do Data: seed
fixa, números no texto, Figuras 1–6, Results summary por último. Uma novidade: o enunciado
*avisa* das pegadinhas (a forma errada da regra de update, a proibição do zero-start) —
então demonstrar que entendi cada aviso vale tanto quanto o código.

## Etapa 1 — Medir antes de treinar (a régua do Data)

Gerei as nuvens e, antes de existir perceptron, apliquei o $r$ do Data:
$4.950 / (2 \times 0.707) = 3.5$. No Data, $r \geq 2.4$ já dava mistura zero — então
**previ 100% antes de rodar**. Foi o primeiro momento em que o exercício anterior deixou de
ser passado: virou instrumento.

## Etapa 2 — O neurônio parado

Medi a acurácia da reta aleatória do init, sem treinar. **Surpresa 1**: deu 61.50%, não
os ~50% que eu esperava. Fui olhar: $\mathbf{w}_0 = [0.0099, -0.0083]$ — o sorteio caiu
numa inclinação parecida com a boa (a direção $[1,1]$ separa as nuvens), e passando pela
origem a reta já deixa boa parte da classe 0 de um lado. Lição: "reta aleatória" não é
"50%", é "depende da sorte do sorteio" — e o 50% real de chute aparece só no Ex. 2, em
outro contexto. Reportei o número verdadeiro.

## Etapa 3 — Um update na mão

Peguei o primeiro ponto errado ($\mathbf{x} = [2.541, -0.315]$, $y = 0$, previsto 1) e
apliquei a regra à mão: $\mathbf{w}\cdot\mathbf{x} + b$ foi de $+0.0278$ para $-0.0477$
em um único passo. Foi aqui que a pegadinha do enunciado fez sentido no corpo: com a forma
dos livros ($+\eta\,y\,\mathbf{x}$) e $y = 0$, esse update seria **zero** — o falso
positivo nunca seria corrigido. O termo $(y - \hat y)$ é o que dá sinal ao erro.

## Etapa 4 — O loop, no dado separável

Convergiu em 2 épocas: 48 updates, depois 0. Entendi "convergir" de forma concreta: não é
a acurácia subir, é **os erros se esgotarem** — em dado separável cada correção fica
correta, ninguém do outro lado pede o movimento contrário. Por isso pus a curva de updates
por época na Figura 3 junto com a acurácia: é ela que conta a história.

## Etapa 5 — Mexer no η

Mesma init, $\eta = 1.0$: também 2 épocas, também 100%, direção a **2.09°** da outra, norma
56× maior. A lição que guardei: **$\eta$ muda a escala do passo, não a geometria** — e só
importa em relação à init. A prova para o zero-start saiu por indução: se em $t$ os pesos
das duas execuções diferem por $c = \eta_2/\eta_1$, a predição é a mesma (mesmo sinal),
o erro é o mesmo, e em $t+1$ continuam diferindo por $c$. Fronteira invariante a escala →
mesmas épocas, mesmos erros. A init não-nula é o que quebra isso ($\mathbf{w}_0 \neq
c\,\mathbf{w}_0$), e é a única razão de os 2.09° existirem. Anotei para usar na Etapa 8.

## Etapa 6 — Prever o fracasso antes de rodar

Régua de novo: $r = 1.414 / 2.449 = 0.577 < 1$. Pelo Data, nuvens que se atravessam —
nenhuma reta separa. Previ: o loop não para, e a melhor reta erra mais de um quarto.
Calculei a mediatriz entre as médias como referência fixa (nada treinado): 72.60%. O
enunciado dizia "~73%". Bateu antes de treinar.

## Etapa 7 — Pocket

A única mudança no loop: quando um update melhora a acurácia máxima já vista, copio
$(\mathbf{w}, b)$. Decisão de implementação: só recalculo a acurácia **quando houve
update** — senão são 2000 avaliações por época à toa. Resultado: pocket **73.30%** na
época 23, e nunca superado nas 77 épocas seguintes. O iterado final: 63.90%.

## Etapa 8 — Entender o fracasso

**Surpresa 2**: o enunciado antecipa iterado final "~50%"; o meu deu 63.90%. Em vez de
forçar, fui entender. Hipótese: o ~50% vem da **ordem** — o enunciado não manda embaralhar,
então na versão "natural" cada época termina depois das 1000 amostras da classe 1 em
sequência, e a reta acaba prevendo classe 1 para tudo (= 50%). Testei: mesmo treino, sem
embaralhar → final **50.10%**, pocket 72.95%. Hipótese confirmada com número. A minha
versão embaralhada oscila entre 59.80% e 71.40% porque os empurrões se alternam — menos
ruim, igualmente instável. Nos dois casos o final é "onde a última rajada de erros deixou
a reta"; a dica do enunciado ($b$ anda $\eta$, $\mathbf{w}$ anda $5\eta$) explica *para
onde* ela é empurrada.

O teorema de convergência ficou concreto: ele garante updates finitos ($\leq (R/\gamma)^2$)
**se** existe margem $\gamma > 0$. Aqui não existe reta separadora, logo não existe
$\gamma$ — o limite não é grande, é inexistente. E as duas perguntas finais se responderam
com o que eu já tinha: mais épocas não ajudam porque os ~700 updates/época não têm
tendência de queda (evidência nos dados), e $\eta$ menor não ajuda **pela Etapa 5** — só
reescala a mesma trajetória em torno do mesmo lugar.

## Etapa 9 — Relatório

Escrevi com os números já cravados, declarei as decisões (embaralhar uma vez; mesma init
para $\eta = 1.0$; recalcular acurácia só em update), e fechei com a tabela de 8 linhas.

## O que mudou no meu entendimento

- Perceptron é uma reta que se move **só quando erra**; convergir = erros acabarem.
- $\eta$ é escala, não direção; do zero ele é literalmente irrelevante.
- Em dado não-separável o "final" não significa nada — o pocket é o algoritmo de verdade.
- A régua do Data ($r$) prevê o comportamento do perceptron antes de treinar: $r = 3.5$ →
  converge; $r = 0.58$ → nunca.
- Detalhes de implementação (ordem das amostras) mudam o número final sem mudar a lição —
  e vale mais explicar a diferença do que esconder.

## Perguntas que eu esperaria na defesa

| Pergunta provável | Minha resposta em uma linha |
|---|---|
| Por que $(y - \hat y)$ e não $y$ na regra de update? | Com rótulos 0/1, $y\,\mathbf{x}$ nunca atualiza na classe 0 — falso positivo nunca seria corrigido; $(y-\hat y)\in\{-1,0,+1\}$ dá as duas direções. |
| Por que o enunciado proíbe começar de $\mathbf{w} = 0$? | Porque do zero os pesos de duas execuções diferem só por $\eta_2/\eta_1$: mesma fronteira, mesmas épocas — o item D sobre $\eta$ não teria o que comparar. |
| O que $\eta$ controla, então? | A escala de cada passo ($\eta\,\mathbf{x}$); só importa relativo à init — por isso as direções diferem 2.09° e não zero. |
| Por que o iterado final do Ex. 2 é ruim? | Não há assentar: ~700 erros por época para sempre; a reta é o que a última rajada de erros deixou. Em ordem por classe cai a 50.10%. |
| Qual hipótese do teorema de convergência o Ex. 2 viola? | Separabilidade linear: sem reta separadora não existe margem $\gamma$, e o limite $(R/\gamma)^2$ de updates não existe. |
| Mais épocas ou $\eta$ menor resolvem? | Não: updates/época não decaem (dado), e $\eta$ só reescala a trajetória (prova do Ex. 1). |
| Por que o pocket chega em ~73% e não mais? | Porque 73% é o teto de *qualquer* reta neste dado (mediatriz: 72.60%) — o resto é erro irredutível, o $r = 0.58$ do Data. |
| Por que embaralhou só uma vez? | Reprodutibilidade: ordem fixa em todas as épocas; embaralhar por época consumiria o `rng` e o enunciado não pede. |
