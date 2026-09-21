# Exercício 1 — Dados separáveis: o caso para o qual o perceptron foi feito.
# Gera as Figuras 1, 2 e 3 em ../figures/ e imprime todos os números do relatório.
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from perceptron import accuracy, predict, train  # noqa: E402  (meu perceptron, escrito à mão)

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
MEDIAS = np.array([[1.5, 1.5], [5.0, 5.0]])
COV = np.array([[0.5, 0.0], [0.0, 0.5]])
N = 1000

X0 = rng.multivariate_normal(MEDIAS[0], COV, N)
X1 = rng.multivariate_normal(MEDIAS[1], COV, N)
X = np.vstack([X0, X1])
y = np.array([0] * N + [1] * N)

# embaralho UMA vez (o perceptron vê as amostras nesta ordem em todas as épocas)
ordem = rng.permutation(2 * N)
X, y = X[ordem], y[ordem]
print("dataset:", X.shape, "| classes:", np.bincount(y))

# a régua do exercício Data: distância entre centros / soma das dispersões médias
dist = np.linalg.norm(MEDIAS[1] - MEDIAS[0])
sigma_barra = np.sqrt(COV[0, 0])  # desvio por eixo = sqrt(0.5), igual nas duas classes
print(f"régua do Data: ||mu1 - mu0|| = {dist:.3f}, sigma_barra = {sigma_barra:.3f} por classe, "
      f"r = {dist / (2 * sigma_barra):.3f}")

fig, ax = plt.subplots(figsize=(7, 6))
for k, cor in enumerate(CORES[:2]):
    ax.scatter(X[y == k, 0], X[y == k, 1], s=8, alpha=0.5, color=cor, label=f"Classe {k}")
ax.set_title("Figura 1 — Duas classes separáveis (1000 pontos cada)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURAS / "fig1.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- B — Implementar o perceptron
# (a implementação está em perceptron.py; aqui só a inicialização do enunciado)
w0 = rng.normal(0, 0.01, size=2)
b0 = 0.0
print(f"init: w0 = {w0}, b0 = {b0}")
print(f"acurácia ANTES de treinar (reta aleatória): {accuracy(w0, b0, X, y):.4f}")

# um update feito na mão, para entender a regra: pego o primeiro ponto errado
errados = np.where(predict(w0, b0, X) != y)[0]
i = errados[0]
xi, yi = X[i], y[i]
y_hat = int(w0 @ xi + b0 >= 0)
w_dep = w0 + 0.01 * (yi - y_hat) * xi
b_dep = b0 + 0.01 * (yi - y_hat)
print(f"exemplo de update: x = {xi.round(3)}, y = {yi}, y_hat = {y_hat}, erro = {yi - y_hat}")
print(f"   antes: w·x+b = {w0 @ xi + b0:+.4f}  |  depois: w·x+b = {w_dep @ xi + b_dep:+.4f}")

# ------------------------------------------------------------ C — Treinar e medir
r1 = train(X, y, w0, b0, eta=0.01, max_epochs=100)
w, b = r1["w"], r1["b"]
print(f"\nη = 0.01 → épocas: {r1['epocas']} | acurácia final: {r1['acc']:.4f}")
print(f"   w = {w}, b = {b:.4f}")
print(f"   updates por época: {r1['hist']['updates']}")
print(f"   acurácia por época: {[round(a, 4) for a in r1['hist']['acc']]}")

erros = predict(w, b, X) != y
xlim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
fig, ax = plt.subplots(figsize=(7, 6))
for k, cor in enumerate(CORES[:2]):
    ax.scatter(X[y == k, 0], X[y == k, 1], s=8, alpha=0.4, color=cor, label=f"Classe {k}")
ax.scatter(X[erros, 0], X[erros, 1], s=60, facecolors="none", edgecolors="black",
           linewidths=1.2, label=f"Mal classificados ({erros.sum()})")
fronteira(ax, w, b, xlim, color="black", lw=2, label="w·x + b = 0")
ax.set_xlim(*xlim)
ax.set_title(f"Figura 2 — Fronteira aprendida (η = 0.01, {r1['epocas']} épocas)")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURAS / "fig2.png", bbox_inches="tight")
plt.close(fig)

epocas = np.arange(1, r1["epocas"] + 1)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epocas, r1["hist"]["acc"], marker="o", color=CORES[0], label="acurácia (dataset inteiro)")
ax.set_xlabel("época")
ax.set_ylabel("acurácia")
ax.set_xticks(epocas)
ax.set_ylim(0.9, 1.01)
ax2 = ax.twinx()  # eixo secundário: quantos updates cada época gerou
ax2.plot(epocas, r1["hist"]["updates"], marker="s", ls="--", color=CORES[3],
         label="updates na época")
ax2.set_ylabel("updates na época")
ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right")
ax.set_title("Figura 3 — Acurácia e updates por época (dados separáveis)")
plt.tight_layout()
fig.savefig(FIGURAS / "fig3.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------- D — Análise
# Mesma init w0, mesmo dado, mesma ordem: só o η muda
r2 = train(X, y, w0, b0, eta=1.0, max_epochs=100)
w_1, b_1 = r2["w"], r2["b"]
dir_001 = w / np.linalg.norm(w)
dir_1 = w_1 / np.linalg.norm(w_1)
angulo = np.degrees(np.arccos(np.clip(dir_001 @ dir_1, -1, 1)))
print(f"\nη = 1.0  → épocas: {r2['epocas']} | acurácia final: {r2['acc']:.4f}")
print(f"   w = {w_1}, b = {b_1:.4f}")
print(f"   updates por época: {r2['hist']['updates']}")
print(f"   ||w|| com η=0.01: {np.linalg.norm(w):.4f} | com η=1.0: {np.linalg.norm(w_1):.4f}")
print(f"   direção w/||w|| — η=0.01: {dir_001.round(4)} | η=1.0: {dir_1.round(4)} "
      f"| ângulo entre elas: {angulo:.2f}°")
print(f"   ||x|| médio dos dados: {np.linalg.norm(X, axis=1).mean():.3f}")
print("figuras salvas em", FIGURAS)
