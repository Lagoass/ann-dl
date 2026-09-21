# Exercício 1 — Nuvens de pontos: geometria e dispersão em 2D
# Gera as Figuras 1, 2, 3 e 1b em ../figures/ e imprime todos os números do relatório.
# Roda de qualquer diretório: os caminhos são relativos a este arquivo.
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
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

# ---------------------------------------------------------------- A — Gerar as nuvens
# Parâmetros do enunciado: 4 classes, 100 pontos cada, média e desvio POR EIXO
MEDIAS = np.array([[2, 3], [5, 6], [8, 1], [15, 4]], dtype=float)
DESVIOS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]], dtype=float)

X1 = np.vstack([rng.normal(MEDIAS[k], DESVIOS[k], size=(100, 2)) for k in range(4)])
y1 = np.repeat(np.arange(4), 100)
print("dataset:", X1.shape, "| classes:", np.bincount(y1))

fig, ax = plt.subplots(figsize=(8, 6))
for k in range(4):
    ax.scatter(X1[y1 == k, 0], X1[y1 == k, 1], s=14, alpha=0.65, color=CORES[k],
               label=f"Classe {k}")
ax.scatter(MEDIAS[:, 0], MEDIAS[:, 1], marker="X", s=140, color="black", zorder=5,
           label="Centros")
for k in range(4):
    ax.annotate(f"$\\mu_{k}$", MEDIAS[k], textcoords="offset points", xytext=(8, 6))
ax.set_title("Figura 1 — Nuvens de pontos 2D (s = 1) com os centros marcados")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.legend()
plt.tight_layout()
fig.savefig(FIGURAS / "fig1.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------ B — Mais ou menos espalhadas
# As mesmas 4 classes em 4 escalas; as médias não mudam. s = 1 reutiliza o item A.
escalas = [0.5, 1.0, 2.0, 4.0]
dados = {1.0: (X1, y1)}
for s in [0.5, 2.0, 4.0]:
    X = np.vstack([rng.normal(MEDIAS[k], DESVIOS[k] * s, size=(100, 2)) for k in range(4)])
    dados[s] = (X, np.repeat(np.arange(4), 100))

# limites de eixo iguais nos 4 subplots — senão a comparação mente
todos = np.vstack([dados[s][0] for s in escalas])
xlim = (todos[:, 0].min() - 1, todos[:, 0].max() + 1)
ylim = (todos[:, 1].min() - 1, todos[:, 1].max() + 1)

fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), escalas):
    X, y = dados[s]
    for k in range(4):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=8, alpha=0.6, color=CORES[k],
                   label=f"Classe {k}" if s == 0.5 else None)
    ax.scatter(MEDIAS[:, 0], MEDIAS[:, 1], marker="X", s=80, color="black", zorder=5)
    ax.set_title(f"s = {s}")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
for ax in axes[-1]:
    ax.set_xlabel("$x_1$")
for ax in axes[:, 0]:
    ax.set_ylabel("$x_2$")
fig.suptitle("Figura 2 — As mesmas 4 classes sob 4 fatores de escala (eixos compartilhados)")
fig.legend(loc="lower center", ncol=5, frameon=False, markerscale=2.5)
plt.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(FIGURAS / "fig2.png", bbox_inches="tight")
plt.close(fig)

# Razão de separação r_ij em s = 1
sigma_barra = DESVIOS.mean(axis=1)  # dispersão média de cada classe

linhas = []
for i in range(4):
    for j in range(i + 1, 4):
        dist = np.linalg.norm(MEDIAS[i] - MEDIAS[j])
        r = dist / (sigma_barra[i] + sigma_barra[j])
        linhas.append({"par": f"({i}, {j})", "‖μi − μj‖": round(dist, 3),
                       "σ̄i + σ̄j": round(sigma_barra[i] + sigma_barra[j], 2),
                       "r_ij": round(r, 3)})
tabela_r = pd.DataFrame(linhas)
print(tabela_r.to_string(index=False))

menor = tabela_r.loc[tabela_r["r_ij"].idxmin()]
print(f"\nMenor razão: par {menor['par']} com r = {menor['r_ij']}")
print(f"Como r_ij ∝ 1/s, em s = 2 esse valor vira {menor['r_ij'] / 2:.3f}")

# Mixing rate: fração de pontos cujo centro (média teórica) mais próximo não é o seu
mistura = {}
for s in escalas:
    X, y = dados[s]
    d2 = ((X[:, None, :] - MEDIAS[None, :, :]) ** 2).sum(axis=2)  # (400, 4)
    mistura[s] = float((d2.argmin(axis=1) != y).mean())
    print(f"mixing rate (s = {s}): {mistura[s]:.4f}  ({int(mistura[s] * 400)}/400 pontos)")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(escalas, [mistura[s] for s in escalas], marker="o", color=CORES[0],
        label="mixing rate")
for s in escalas:
    ax.annotate(f"{mistura[s]:.2%}", (s, mistura[s]), textcoords="offset points",
                xytext=(6, 8))
ax.set_title("Figura 3 — Mixing rate em função do fator de escala s")
ax.set_xlabel("fator de escala $s$")
ax.set_ylabel("mixing rate")
ax.set_xticks(escalas)
ax.legend()
plt.tight_layout()
fig.savefig(FIGURAS / "fig3.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------- C — Análise
# Esboço das fronteiras: partição do plano pela regra do centro mais próximo
gx, gy = np.meshgrid(np.linspace(*xlim, 500), np.linspace(*ylim, 500))
grade = np.stack([gx.ravel(), gy.ravel()], axis=1)
Z = ((grade[:, None, :] - MEDIAS[None, :, :]) ** 2).sum(axis=2).argmin(axis=1).reshape(gx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=CORES, alpha=0.15)
ax.contour(gx, gy, Z, levels=[0.5, 1.5, 2.5], colors="black", linewidths=1)
for k in range(4):
    ax.scatter(X1[y1 == k, 0], X1[y1 == k, 1], s=14, alpha=0.65, color=CORES[k],
               label=f"Classe {k}")
ax.scatter(MEDIAS[:, 0], MEDIAS[:, 1], marker="X", s=140, color="black", zorder=5,
           label="Centros")
ax.set_title("Figura 1b — Esboço das fronteiras de decisão sobre o dataset s = 1")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim(-2, 18)
ax.set_ylim(-6, 12)
ax.legend(loc="lower left")
plt.tight_layout()
fig.savefig(FIGURAS / "fig1b.png", bbox_inches="tight")
plt.close(fig)

print("figuras salvas em", FIGURAS)
