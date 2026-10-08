"""EDA — etapa 4B: projeções 2D das features de treino já transformadas (PCA, t-SNE, UMAP).

PCA usa o treino inteiro. t-SNE e UMAP usam uma amostra estratificada de 3.000 pontos do
treino (o enunciado autoriza amostrar em datasets grandes) — a mesma amostra para os dois,
para a comparação ser justa. Salva as Figuras 11 a 13 em ../figures/.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from pipeline import SEED, build_preprocessor, load_clean, split  # noqa: E402

FIG = BASE.parent / "figures"
FIG.mkdir(exist_ok=True)
CORES = ["#E8A13D", "#C75146", "#6B8F3D", "#33658A"]
COR_CLASSE = {0: CORES[3], 1: CORES[0]}
ROTULO = {0: "≤50K", 1: ">50K"}
CORES_REL = ["#33658A", "#E8A13D", "#C75146", "#6B8F3D", "#8E6C8A", "#7A7A7A"]
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110,
})


def secao(txt):
    print("\n" + "=" * 80 + f"\n{txt}\n" + "=" * 80)


def por_classe(ax, E, y, titulo, s=6):
    for k in (0, 1):
        m = y == k
        ax.scatter(E[m, 0], E[m, 1], s=s, alpha=0.45, color=COR_CLASSE[k], label=ROTULO[k])
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("dimensão 1")
    ax.set_ylabel("dimensão 2")
    ax.legend(title="classe", markerscale=2.5, fontsize=8)


df = load_clean()
X_tr, X_te, y_tr, y_te = split(df)
prep = build_preprocessor()
Z_tr = prep.fit_transform(X_tr)
nomes = np.array(prep.get_feature_names_out())
y_arr = y_tr.to_numpy()

# ------------------------------------------------------------------------------ PCA
secao("4B — PCA (treino inteiro)")
pca = PCA().fit(Z_tr)
evr = pca.explained_variance_ratio_
acum = np.cumsum(evr)
print(f"{Z_tr.shape[1]} colunas | autovalores ~0 (< 1e-10): {int((pca.explained_variance_ < 1e-10).sum())}")
print("variância explicada, 10 primeiras:", (evr[:10] * 100).round(2).tolist())
print(f"PC1 = {evr[0] * 100:.2f}% | PC2 = {evr[1] * 100:.2f}% | PC1+PC2 = {acum[1] * 100:.2f}%")
for alvo in (0.5, 0.8, 0.9, 0.95):
    print(f"   componentes para {alvo:.0%} da variância: {int(np.searchsorted(acum, alvo) + 1)}")
pca2 = PCA(2, random_state=SEED).fit(Z_tr)
for i in range(2):
    ordem = np.argsort(-np.abs(pca2.components_[i]))[:8]
    print(f"loadings PC{i + 1} (8 maiores |peso|): " +
          ", ".join(f"{nomes[j]}={pca2.components_[i][j]:+.3f}" for j in ordem))
P_tr = pca2.transform(Z_tr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
k = np.arange(1, len(acum) + 1)
axes[0].plot(k, acum * 100, marker="o", ms=3, color=CORES[3], label="variância acumulada")
axes[0].axhline(80, color=CORES[1], ls=":", label="80%")
axes[0].axvline(2, color=CORES[0], ls="--", label=f"PC1+PC2 = {acum[1]:.1%}")
axes[0].set_xlabel("número de componentes")
axes[0].set_ylabel("variância explicada acumulada (%)")
axes[0].set_title("variância acumulada")
axes[0].legend(fontsize=8)
ordem = np.random.default_rng(SEED).permutation(len(P_tr))
por_classe(axes[1], P_tr[ordem], y_arr[ordem], f"PC1 × PC2 (treino inteiro, n={len(P_tr)})", s=3)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
fig.suptitle("Figura 11 — PCA das features de treino transformadas")
plt.tight_layout()
fig.savefig(FIG / "fig11-pca.png", bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------- t-SNE e UMAP (amostra)
secao("4B — t-SNE e UMAP (amostra estratificada do treino)")
idx = np.arange(len(Z_tr))
idx_s, _ = train_test_split(idx, train_size=3000, stratify=y_arr, random_state=SEED)
Zs, ys = Z_tr[idx_s], y_arr[idx_s]
rel_s = X_tr["relationship"].to_numpy()[idx_s]
sexo_s = X_tr["sex"].to_numpy()[idx_s]
mar_s = X_tr["marital-status"].to_numpy()[idx_s]
print(f"amostra: {Zs.shape} | >50K = {ys.mean() * 100:.2f}%")

mapas = {"PCA": pca2.transform(Zs)}
for p in (30, 50):
    mapas[f"t-SNE perplexidade {p}"] = TSNE(2, perplexity=p, init="pca",
                                            random_state=SEED).fit_transform(Zs)
for nn in (15, 50):
    mapas[f"UMAP n_neighbors {nn}"] = umap.UMAP(n_neighbors=nn, min_dist=0.1,
                                                random_state=SEED).fit_transform(Zs)

print(f"{'mapa':24} {'trust k=10':>11} {'trust k=50':>11} {'silh. income':>13} "
      f"{'silh. relationship':>19} {'silh. sex':>10} {'silh. marital':>14}")
for nome, E in mapas.items():
    print(f"{nome:24} {trustworthiness(Zs, E, n_neighbors=10):11.3f} "
          f"{trustworthiness(Zs, E, n_neighbors=50):11.3f} {silhouette_score(E, ys):13.3f} "
          f"{silhouette_score(E, rel_s):19.3f} {silhouette_score(E, sexo_s):10.3f} "
          f"{silhouette_score(E, mar_s):14.3f}")
print(f"{'(espaço original)':24} {'—':>11} {'—':>11} {silhouette_score(Zs, ys):13.3f} "
      f"{silhouette_score(Zs, rel_s):19.3f} {silhouette_score(Zs, sexo_s):10.3f} "
      f"{silhouette_score(Zs, mar_s):14.3f}")

# controle: as mesmas colunas embaralhadas independentemente (destrói a estrutura conjunta)
rng = np.random.default_rng(SEED)
Zr = np.column_stack([rng.permutation(Zs[:, j]) for j in range(Zs.shape[1])])
E_r = TSNE(2, perplexity=30, init="pca", random_state=SEED).fit_transform(Zr)
print(f"controle — t-SNE perplexidade 30 em colunas embaralhadas: silh. income {silhouette_score(E_r, ys):.3f} "
      f"| trust k=10 {trustworthiness(Zr, E_r, n_neighbors=10):.3f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
for ax, nome in zip(axes.ravel(), [n for n in mapas if n != "PCA"]):
    por_classe(ax, mapas[nome], ys, nome)
fig.suptitle("Figura 12 — t-SNE e UMAP com dois valores de parâmetro cada, coloridos pelo alvo "
             "(amostra de 3.000 do treino)")
plt.tight_layout()
fig.savefig(FIG / "fig12-tsne-umap.png", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
cats = sorted(set(rel_s))
for ax, nome in zip(axes, ["t-SNE perplexidade 30", "UMAP n_neighbors 15"]):
    E = mapas[nome]
    for cat, cor in zip(cats, CORES_REL):
        m = rel_s == cat
        ax.scatter(E[m, 0], E[m, 1], s=6, alpha=0.5, color=cor, label=cat)
    ax.set_title(nome, fontsize=10)
    ax.set_xlabel("dimensão 1")
    ax.set_ylabel("dimensão 2")
    ax.legend(title="relationship", markerscale=2.5, fontsize=7)
fig.suptitle("Figura 13 — Os mesmos mapas da Figura 12, coloridos por relationship")
plt.tight_layout()
fig.savefig(FIG / "fig13-ilhas-relationship.png", bbox_inches="tight")
plt.close(fig)
print("figuras salvas em", FIG)
