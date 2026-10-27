"""Exercício 2 — classificação binária que nenhuma reta resolve. Gera as Figuras 1, 2 e 3."""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import LogisticRegression

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import dados  # noqa: E402
from mlp import MLP, BinaryCrossEntropy, padronizador  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")  # acentos em UTF-8 também com a saída redirecionada
FIG = BASE.parent / "figures"
FIG.mkdir(exist_ok=True)

rng = np.random.default_rng(42)  # mesma seed do relatório; aqui ela inicializa os pesos

CORES = ["#E8A13D", "#C75146", "#6B8F3D", "#33658A"]  # âmbar, telha, oliva, aço
COR = {0: CORES[3], 1: CORES[0]}
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110,
})

# ------------------------------------------------------------------ A — gerar os dados
X, y, X_tr, X_te, y_tr, y_te, c_tr, c_te = dados.ex2()
print(f"dataset {X.shape} | classe 0: {(y == 0).sum()} | classe 1: {(y == 1).sum()}")
print(f"treino {X_tr.shape} ({np.bincount(y_tr)}) | teste {X_te.shape} ({np.bincount(y_te)})")

fig, ax = plt.subplots(figsize=(7, 6))
for k in (0, 1):
    ax.scatter(X[y == k, 0], X[y == k, 1], s=10, alpha=0.55, color=COR[k], label=f"classe {k}")
ax.scatter(dados.CENTROS_EX2[:, 0], dados.CENTROS_EX2[:, 1], marker="X", s=140, color="black",
           zorder=5, label="centros dos blocos")
ax.set_title("Figura 1 — Os 1000 pontos: classe 0 no meio, classe 1 em dois blocos opostos")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend()
fig.savefig(FIG / "fig1.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------ B — baseline linear
base = LogisticRegression().fit(X_tr, y_tr)
acc_base = base.score(X_te, y_te)
w, b0 = base.coef_[0], base.intercept_[0]
p_base = base.predict_proba(X_te)[:, 1]
print(f"\nbaseline (regressão logística): acurácia de teste = {acc_base:.4f}")
print(f"   pesos w = {w.round(4)} | |w| = {np.linalg.norm(w):.4f} | viés = {b0:.4f}")
print(f"   P(classe 1) no teste vai de {p_base.min():.4f} a {p_base.max():.4f}")
for k, nome in enumerate(["bloco central (classe 0)", "bloco (3, 3) (classe 1)", "bloco (-3, -3) (classe 1)"]):
    m = c_te == k
    print(f"   acerto no {nome:26}: {base.predict(X_te[m]).tolist().count(y_te[m][0]) / m.sum():.4f} (n={m.sum()})")


def melhor_reta(X, y):
    """Busca exaustiva da melhor reta no treino: 360 direções × todos os limiares."""
    n = len(y)
    melhor = (0.0, None, None, None)
    for ang in np.arange(0.0, 180.0, 0.5):
        d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        s = X @ d
        ordem = np.argsort(s)
        ys, ss = y[ordem], s[ordem]
        zeros_ate = np.concatenate([[0], np.cumsum(ys == 0)])                    # previstos 0
        uns_depois = np.concatenate([[0], np.cumsum((ys == 1)[::-1])])[::-1]     # previstos 1
        acertos = zeros_ate + uns_depois                                          # "1 se s > t"
        for sentido, ac in ((1, acertos), (-1, n - acertos)):                    # e "1 se s < t"
            i = int(ac.argmax())
            if ac[i] / n > melhor[0]:
                t = (ss[i - 1] + ss[i]) / 2 if 0 < i < n else (ss[0] - 1 if i == 0 else ss[-1] + 1)
                melhor = (ac[i] / n, d, t, sentido)
    return melhor


acc_reta_tr, d_reta, t_reta, sentido = melhor_reta(X_tr, y_tr)
s_te = X_te @ d_reta
acc_reta_te = ((s_te > t_reta) if sentido == 1 else (s_te < t_reta)).astype(int)
acc_reta_te = (acc_reta_te == y_te).mean()
print(f"melhor reta possível (busca no treino): treino {acc_reta_tr:.4f} | teste {acc_reta_te:.4f} | "
      f"direção {d_reta.round(3)}")

# ------------------------------------------------------------------ C — a MLP
P = padronizador(X_tr)  # média e desvio só do treino
ARQ, EPOCAS, LR = [2, 8, 1], 2000, 0.5
net = MLP(ARQ, BinaryCrossEntropy, rng)
hist = net.fit(P(X_tr), y_tr[:, None].astype(float), EPOCAS, LR)
acc_tr = (net.predict(P(X_tr)) == y_tr).mean()
acc_te = (net.predict(P(X_te)) == y_te).mean()
perda_final = BinaryCrossEntropy.value(net.forward(P(X_tr)), y_tr[:, None].astype(float))
print(f"\nMLP {'-'.join(map(str, ARQ))} ({net.n_params()} parâmetros), tanh + sigmoide, BCE, "
      f"lr = {LR}, {EPOCAS} épocas em lote completo")
print(f"   perda de treino: época 1 = {hist[0]:.4f} | época {EPOCAS} = {hist[-1]:.4f} | final = {perda_final:.4f}")
print(f"   acurácia: treino {acc_tr:.4f} | TESTE {acc_te:.4f}")
for k, nome in enumerate(["bloco central", "bloco (3, 3)", "bloco (-3, -3)"]):
    m = c_te == k
    print(f"   acerto no {nome:15}: {(net.predict(P(X_te[m])) == y_te[m]).mean():.4f} (n={m.sum()})")

# o learning rate: mesma inicialização (seed 42), perda em épocas-chave
for lr_teste in (0.1, 0.5):
    rede = MLP(ARQ, BinaryCrossEntropy, np.random.default_rng(42))
    h = rede.fit(P(X_tr), y_tr[:, None].astype(float), EPOCAS, lr_teste)
    print(f"   lr = {lr_teste}: perda na época 500 = {h[499]:.4f} | época 1000 = {h[999]:.4f} "
          f"| época {EPOCAS} = {h[-1]:.4f}")

# a forma da região da classe 1: ao longo da diagonal x1 = x2 a classe prevista troca 2 vezes
t = np.linspace(-7, 7, 2801)
diag = np.column_stack([t, t])
classe = net.predict(P(diag))
trocas = t[1:][np.diff(classe) != 0]
print(f"   ao longo da diagonal x1 = x2, a classe prevista troca em x1 = {trocas.round(2).tolist()}")
anti = np.column_stack([t, -t])
print(f"   ao longo da antidiagonal x1 = -x2 (que passa pelo meio da classe 0): "
      f"classes previstas {sorted(set(net.predict(P(anti)).tolist()))}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(np.arange(1, EPOCAS + 1), hist, color=CORES[3], label="perda de treino (BCE)")
ax.set_title(f"Figura 2 — Perda de treino por época (MLP {'-'.join(map(str, ARQ))}, lr = {LR})")
ax.set_xlabel("época")
ax.set_ylabel("entropia cruzada binária")
ax.legend()
fig.savefig(FIG / "fig2.png", bbox_inches="tight")
plt.close(fig)

lim = np.abs(X_te).max() + 1.0
gx, gy = np.meshgrid(np.linspace(-lim, lim, 400), np.linspace(-lim, lim, 400))
grade = np.column_stack([gx.ravel(), gy.ravel()])
prob = net.forward(P(grade))[:, 0].reshape(gx.shape)
erros = net.predict(P(X_te)) != y_te

fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.contourf(gx, gy, prob, levels=[0, 0.5, 1], colors=[COR[0], COR[1]], alpha=0.22)
ax.contour(gx, gy, prob, levels=[0.5], colors="black", linewidths=1.5)
for k in (0, 1):
    m = y_te == k
    ax.scatter(X_te[m, 0], X_te[m, 1], s=16, color=COR[k], edgecolors="white", linewidths=0.3)
ax.scatter(X_te[erros, 0], X_te[erros, 1], s=70, facecolors="none", edgecolors="black", linewidths=1.2)
xs = np.array([-lim, lim])  # fronteira da regressão logística: w·x + b = 0
if abs(w[1]) > abs(w[0]):
    ax.plot(xs, -(w[0] * xs + b0) / w[1], color=CORES[1], ls="--", lw=2)
else:
    ax.plot(-(w[1] * xs + b0) / w[0], xs, color=CORES[1], ls="--", lw=2)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_title("Figura 3 — Fronteira de decisão da MLP sobre o conjunto de teste")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(handles=[
    Patch(color=COR[0], alpha=0.35, label="região prevista: classe 0"),
    Patch(color=COR[1], alpha=0.35, label="região prevista: classe 1"),
    Line2D([], [], color="black", lw=1.5, label="fronteira da MLP (P = 0.5)"),
    Line2D([], [], color=CORES[1], ls="--", lw=2, label="fronteira da regressão logística"),
    Line2D([], [], marker="o", ls="", color=COR[0], label="teste: classe 0"),
    Line2D([], [], marker="o", ls="", color=COR[1], label="teste: classe 1"),
    Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="black",
           label=f"erros da MLP ({erros.sum()})"),
], fontsize=8, loc="upper left")
fig.savefig(FIG / "fig3.png", bbox_inches="tight")
plt.close(fig)
print("figuras salvas em ../figures/")
