# Exercício 2 — Dados sobrepostos: o caso que o perceptron não resolve.
# Gera as Figuras 4, 5 e 6 em ../figures/ e imprime todos os números do relatório.
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from perceptron import accuracy, predict, train  # noqa: E402  (o MESMO perceptron do Ex. 1)

FIGURAS = BASE.parent / "figures"
FIGURAS.mkdir(exist_ok=True)

rng = np.random.default_rng(42)  # mesma seed em todo o exercício

# identidade visual dos meus gráficos: paleta fixa + eixos limpos
CORES = ["#E8A13D", "#C75146", "#6B8F3D", "#33658A"]  # âmbar, telha, oliva, aço
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.prop_cycle": plt.cycler(color=CORES),
    "figure.dpi": 110,
})


def fronteira(ax, w, b, xlim, **kw):
    """Reta w·x + b = 0 dentro dos limites do eixo x."""
    xs = np.array(xlim)
    ax.plot(xs, -(w[0] * xs + b) / w[1], **kw)


# ---------------------------------------------------------------- A — Gerar os dados
MEDIAS = np.array([[3.0, 3.0], [4.0, 4.0]])
COV = np.array([[1.5, 0.0], [0.0, 1.5]])
N = 1000

X0 = rng.multivariate_normal(MEDIAS[0], COV, N)
X1 = rng.multivariate_normal(MEDIAS[1], COV, N)
X = np.vstack([X0, X1])
y = np.array([0] * N + [1] * N)
ordem = rng.permutation(2 * N)  # embaralho uma vez, como no Ex. 1
X, y = X[ordem], y[ordem]
print("dataset:", X.shape, "| classes:", np.bincount(y))

# a régua do exercício Data, de novo
dist = np.linalg.norm(MEDIAS[1] - MEDIAS[0])
sigma_barra = np.sqrt(COV[0, 0])
print(f"régua do Data: ||mu1 - mu0|| = {dist:.3f}, sigma_barra = {sigma_barra:.3f} por classe, "
      f"r = {dist / (2 * sigma_barra):.3f}")
print(f"||x|| médio dos dados: {np.linalg.norm(X, axis=1).mean():.3f}")

fig, ax = plt.subplots(figsize=(7, 6))
for k, cor in enumerate(CORES[:2]):
    ax.scatter(X[y == k, 0], X[y == k, 1], s=8, alpha=0.5, color=cor, label=f"Classe {k}")
ax.set_title("Figura 4 — Duas classes sobrepostas (1000 pontos cada)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURAS / "fig4.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------- B — Treinar guardando os melhores pesos
w0 = rng.normal(0, 0.01, size=2)
b0 = 0.0
print(f"init: w0 = {w0}, b0 = {b0} | acurácia antes de treinar: {accuracy(w0, b0, X, y):.4f}")

r = train(X, y, w0, b0, eta=0.01, max_epochs=100, pocket=True)
w_fim, b_fim = r["w"], r["b"]
pk = r["pocket"]
print(f"\népocas rodadas: {r['epocas']} (o loop nunca parou de atualizar)")
print(f"FINAL : w = {w_fim}, b = {b_fim:.4f}, acurácia = {r['acc']:.4f}")
print(f"POCKET: w = {pk['w']}, b = {pk['b']:.4f}, acurácia = {pk['acc']:.4f}, "
      f"obtido na época {pk['epoca']}")
ups = r["hist"]["updates"]
print(f"updates por época — primeiras 5: {ups[:5]} | últimas 5: {ups[-5:]} | "
      f"média: {np.mean(ups):.1f}")
accs = r["hist"]["acc"]
print(f"acurácia do iterado ao fim de cada época — mín {min(accs):.4f}, máx {max(accs):.4f}, "
      f"média {np.mean(accs):.4f}")

# referência geométrica (regra fixa, nada treinado): a mediatriz entre as duas médias,
# que é a melhor reta possível para gaussianas isotrópicas de mesma covariância
w_ref = MEDIAS[1] - MEDIAS[0]
b_ref = -w_ref @ (MEDIAS[0] + MEDIAS[1]) / 2
print(f"\nmediatriz entre as médias (referência): acurácia = {accuracy(w_ref, b_ref, X, y):.4f}")

# contraprova para o item D: mesmo treino SEM embaralhar (as 1000 da classe 0 e depois as
# 1000 da classe 1, em bloco) — cada época termina após 1000 amostras da classe 1 seguidas
X_bloco = np.vstack([X0, X1])
y_bloco = np.array([0] * N + [1] * N)
rb = train(X_bloco, y_bloco, w0, b0, eta=0.01, max_epochs=100, pocket=True)
print(f"ordem em bloco (sem embaralhar): final = {rb['acc']:.4f} | pocket = {rb['pocket']['acc']:.4f} "
      f"(época {rb['pocket']['epoca']}) | updates/época médio: {np.mean(rb['hist']['updates']):.1f}")

# ------------------------------------------------------------------- C — Figuras
xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
ylim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
erros_pk = predict(pk["w"], pk["b"], X) != y  # marco os erros do pocket (a fronteira que presta)

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for k, cor in enumerate(CORES[:2]):
    ax.scatter(X[y == k, 0], X[y == k, 1], s=8, alpha=0.35, color=cor, label=f"Classe {k}")
ax.scatter(X[erros_pk, 0], X[erros_pk, 1], s=40, facecolors="none", edgecolors="black",
           linewidths=0.8, label=f"Mal classificados pelo pocket ({erros_pk.sum()})")
fronteira(ax, pk["w"], pk["b"], xlim, color="black", lw=2.2,
          label=f"pocket — {pk['acc']:.1%}")
fronteira(ax, w_fim, b_fim, xlim, color=CORES[1], lw=2.2, ls="--",
          label=f"final — {r['acc']:.1%}")
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_title("Figura 5 — Fronteira final × fronteira pocket")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend(loc="upper left")
plt.tight_layout()
fig.savefig(FIGURAS / "fig5.png", bbox_inches="tight")
plt.close(fig)

epocas = np.arange(1, r["epocas"] + 1)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(epocas, accs, color=CORES[1], alpha=0.9, label="acurácia dos pesos atuais")
ax.plot(epocas, r["hist"]["melhor_acc"], color="black", lw=2, label="melhor até agora (pocket)")
ax.set_title("Figura 6 — Acurácia por época: iterado atual × pocket (dados sobrepostos)")
ax.set_xlabel("época")
ax.set_ylabel("acurácia")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(FIGURAS / "fig6.png", bbox_inches="tight")
plt.close(fig)

print("figuras salvas em", FIGURAS)
