"""Exercício 3 — multiclasse com a MESMA rede. Gera as Figuras 4 e 5.

A MLP é a mesma classe do Exercício 2, importada de mlp.py sem nenhuma alteração: mudam só
os argumentos — 3 saídas, a perda (softmax + entropia cruzada) e os hiperparâmetros.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import dados  # noqa: E402
from mlp import MLP, CategoricalCrossEntropy, one_hot, padronizador  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")  # o console cp1252 não imprime "σ"
FIG = BASE.parent / "figures"
FIG.mkdir(exist_ok=True)

rng = np.random.default_rng(42)  # mesma seed do relatório; aqui ela inicializa os pesos

CORES = ["#E8A13D", "#C75146", "#6B8F3D", "#33658A"]
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110,
})

ARQ, EPOCAS, LR = [4, 16, 3], 3000, 0.5  # o Exercício 4 reusa EPOCAS e LR

# ------------------------------------------------------------------ A — gerar os dados
X, y, X_tr, X_te, y_tr, y_te = dados.ex3()
print(f"dataset {X.shape} | classes {np.bincount(y)} | treino {X_tr.shape} {np.bincount(y_tr)} "
      f"| teste {X_te.shape} {np.bincount(y_te)}")

# ------------------------------------------------------------------ B e C — treinar e avaliar
P = padronizador(X_tr)
net = MLP(ARQ, CategoricalCrossEntropy, rng)
hist = net.fit(P(X_tr), one_hot(y_tr, 3), EPOCAS, LR)
pred_te = net.predict(P(X_te))
acc_tr = (net.predict(P(X_tr)) == y_tr).mean()
acc_te = (pred_te == y_te).mean()
perda_final = CategoricalCrossEntropy.value(net.forward(P(X_tr)), one_hot(y_tr, 3))
print(f"\nMLP {'-'.join(map(str, ARQ))} ({net.n_params()} parâmetros), tanh + softmax, entropia "
      f"cruzada, lr = {LR}, {EPOCAS} épocas em lote completo")
print(f"   perda de treino: época 1 = {hist[0]:.4f} | época {EPOCAS} = {hist[-1]:.4f} | final = {perda_final:.4f}")
print(f"   acurácia: treino {acc_tr:.4f} | TESTE {acc_te:.4f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(np.arange(1, EPOCAS + 1), hist, color=CORES[3], label="perda de treino (entropia cruzada)")
ax.set_title(f"Figura 4 — Perda de treino por época (MLP {'-'.join(map(str, ARQ))}, lr = {LR})")
ax.set_xlabel("época")
ax.set_ylabel("entropia cruzada categórica")
ax.legend()
fig.savefig(FIG / "fig4.png", bbox_inches="tight")
plt.close(fig)

cm = confusion_matrix(y_te, pred_te)
print("\nmatriz de confusão (linhas = verdadeira, colunas = prevista):\n" + str(cm))
pares = {(a, b): cm[a, b] + cm[b, a] for a, b in itertools.combinations(range(3), 2)}
for (a, b), n in pares.items():
    print(f"   classes {a} e {b}: {cm[a, b]} + {cm[b, a]} = {n} trocas")
par_mais = max(pares, key=pares.get)
print(f"   par mais confundido: classes {par_mais[0]} e {par_mais[1]}")

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(3), [f"classe {c}" for c in range(3)])
ax.set_yticks(range(3), [f"classe {c}" for c in range(3)])
ax.set_xlabel("classe prevista")
ax.set_ylabel("classe verdadeira")
ax.set_title(f"Figura 5 — Matriz de confusão no teste (acurácia {acc_te:.2%})")
ax.grid(False)
fig.colorbar(im, ax=ax, label="número de amostras de teste")
fig.savefig(FIG / "fig5.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ D — baseline e estrutura
base = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
acc_base = base.score(X_te, y_te)
print(f"\nbaseline (regressão logística multinomial): acurácia de teste = {acc_base:.4f} "
      f"| a camada escondida acrescentou {100 * (acc_te - acc_base):.1f} pontos percentuais")

# a geometria dos 6 clusters, medida no MESMO dataset gerado sem embaralhar (ver dados.py)
Xu, yu = dados.ex3_sem_embaralhar()
k = np.repeat(np.arange(6), 250)          # cluster de cada linha
classe_k = np.arange(6) % 3               # o make_classification dá a classe k % 3 ao cluster k
print(f"\nmesmo conjunto de pontos? {np.allclose(np.sort(np.linalg.norm(X, axis=1)), np.sort(np.linalg.norm(Xu, axis=1)))} "
      f"| rótulos trocados pelo ruído (flip_y = 0.01): {int((yu != classe_k[k]).sum())} de 1500")
C = np.array([Xu[k == i].mean(axis=0) for i in range(6)])
for i in range(6):
    print(f"   cluster {i} (classe {classe_k[i]}): centroide {C[i].round(2)}")
print("pares de clusters de classes diferentes — r = distância / (σi + σj) ao longo do eixo entre eles:")
medidas = []
for i, j in itertools.combinations(range(6), 2):
    if classe_k[i] == classe_k[j]:
        continue
    v = C[j] - C[i]
    d = np.linalg.norm(v)
    s_i, s_j = (Xu[k == i] @ (v / d)).std(), (Xu[k == j] @ (v / d)).std()
    medidas.append((d / (s_i + s_j), d, i, j))
for r, d, i, j in sorted(medidas):
    print(f"   clusters {i}-{j} (classes {classe_k[i]}-{classe_k[j]}): distância {d:.3f} | r = {r:.3f}")
for a, b in itertools.combinations(range(3), 2):
    rs = [(r, i, j) for r, d, i, j in medidas if {classe_k[i], classe_k[j]} == {a, b}]
    menor = min(rs)
    print(f"   classes {a}-{b}: menor r = {menor[0]:.3f} (clusters {menor[1]}-{menor[2]}) | "
          f"r médio = {np.mean([x[0] for x in rs]):.3f}")
for a in range(3):
    i, j = [c for c in range(6) if classe_k[c] == a]
    meio = (C[i] + C[j]) / 2
    dists = np.linalg.norm(C - meio, axis=1)
    viz = int(np.argmin(dists))
    print(f"   classe {a}: o ponto médio entre os clusters {i} e {j} fica a {dists[i]:.3f} deles e a "
          f"{dists[viz]:.3f} do cluster {viz} (classe {classe_k[viz]})")

# acerto por cluster no teste: o cluster de cada ponto vem da norma da linha (não depende da
# ordem das colunas e não se repete entre 1500 pontos contínuos)
cluster_de = {round(float(nrm), 9): int(c) for nrm, c in zip(np.linalg.norm(Xu, axis=1), k)}
c_te = np.array([cluster_de[round(float(nrm), 9)] for nrm in np.linalg.norm(X_te, axis=1)])
pred_base = base.predict(X_te)
print("acerto por cluster no teste:")
for i in range(6):
    m = c_te == i
    print(f"   cluster {i} (classe {classe_k[i]}, n={m.sum():3}): regressão logística "
          f"{(pred_base[m] == y_te[m]).mean():.3f} | MLP {(pred_te[m] == y_te[m]).mean():.3f}")
print("figuras salvas em ../figures/")
