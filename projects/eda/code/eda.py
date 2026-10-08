"""EDA — etapas 1 a 3, mais as evidências que sustentam as escolhas das etapas 4A e 4C.

Imprime todos os números citados no relatório e salva as figuras em ../figures/.
A etapa 1 olha o dataset inteiro (é o que o enunciado pede antes do split); da etapa 2 em
diante, tudo é calculado SÓ no conjunto de treino, para nenhuma decisão de
pré-processamento ser informada pelo teste.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from pipeline import (CATEGORICAL, NUM_PLAIN, NUM_SKEWED, TARGET,  # noqa: E402
                      build_preprocessor, load_clean, load_raw, split)

FIG = BASE.parent / "figures"
FIG.mkdir(exist_ok=True)

CORES = ["#E8A13D", "#C75146", "#6B8F3D", "#33658A"]  # âmbar, telha, oliva, aço
COR_CLASSE = {0: CORES[3], 1: CORES[0]}
ROTULO = {0: "≤50K", 1: ">50K"}
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110,
})
NUM = NUM_PLAIN + NUM_SKEWED


def salva(fig, nome):
    fig.savefig(FIG / nome, bbox_inches="tight")
    plt.close(fig)


def secao(txt):
    print("\n" + "=" * 80 + f"\n{txt}\n" + "=" * 80)


# ============================================================ 1. INITIAL INSPECTION
raw = load_raw()
y_raw = raw[TARGET] == ">50K"

secao("1A — DICIONÁRIO")
print("shape bruto:", raw.shape)
for c in raw.columns:
    tipo = "numérica" if pd.api.types.is_numeric_dtype(raw[c]) else "categórica"
    print(f"  {c:16} {tipo:11} únicos={raw[c].nunique():6}  exemplo={raw[c].dropna().iloc[0]!r}")
print("colunas constantes:", [c for c in raw.columns if raw[c].nunique() <= 1])

secao("1B — QUALIDADE")
falt = raw.isna().sum()
print(pd.DataFrame({"faltantes": falt, "%": (falt / len(raw) * 100).round(2)}).to_string())
print("linhas com algum faltante:", int(raw.isna().any(axis=1).sum()))
wc, oc = raw["workclass"].isna(), raw["occupation"].isna()
print(f"workclass NaN={wc.sum()} | occupation NaN={oc.sum()} | ambos={(wc & oc).sum()} | "
      f"occupation NaN com workclass preenchido={(oc & ~wc).sum()} "
      f"(workclass delas: {raw.loc[oc & ~wc, 'workclass'].value_counts().to_dict()})")
print(f">50K — geral {y_raw.mean() * 100:.2f}% | entre workclass NaN {y_raw[wc].mean() * 100:.2f}% | "
      f"entre native-country NaN {y_raw[raw['native-country'].isna()].mean() * 100:.2f}%")
print("duplicatas exatas (15 colunas):", int(raw.duplicated().sum()))
print("relationship × sex:\n" + pd.crosstab(raw["relationship"], raw["sex"]).to_string())
inc = (((raw["relationship"] == "Husband") & (raw["sex"] == "Female")).sum()
       + ((raw["relationship"] == "Wife") & (raw["sex"] == "Male")).sum())
print("inconsistências (Husband+Female ou Wife+Male):", int(inc))
print(f"valores no teto: age=90 → {(raw['age'] == 90).sum()} | hours-per-week=99 → "
      f"{(raw['hours-per-week'] == 99).sum()} | capital-gain=99999 → {(raw['capital-gain'] == 99999).sum()}")
g = raw.groupby("education")["education-num"].nunique().max()
h = raw.groupby("education-num")["education"].nunique().max()
mapa = raw.groupby("education")["education-num"].first().sort_values()
print(f"education ↔ education-num: máx. de valores por nível = {g} e {h} → bijeção de "
      f"{raw['education'].nunique()} níveis: " + ", ".join(f"{k}={v}" for k, v in mapa.items()))
rho = raw["fnlwgt"].corr(y_raw.astype(int), method="spearman")
print(f"fnlwgt: média >50K {raw.loc[y_raw, 'fnlwgt'].mean():.0f} vs ≤50K "
      f"{raw.loc[~y_raw, 'fnlwgt'].mean():.0f} | Spearman com o alvo {rho:+.4f}")
print("investigação de vazamento — capital-gain:")
for t in [1, 3000, 5000, 7000, 10000, 99999]:
    s = raw["capital-gain"] >= t
    print(f"   capital-gain >= {t:>5}: n={s.sum():5} | >50K={y_raw[s].mean() * 100:5.1f}%")
s = raw["capital-loss"] > 0
print(f"   capital-loss > 0: n={s.sum()} | >50K={y_raw[s].mean() * 100:.1f}%")
print("   gain>0 e loss>0 na mesma linha:",
      int(((raw["capital-gain"] > 0) & (raw["capital-loss"] > 0)).sum()))

df = load_clean()
print(f"\napós limpeza (−duplicatas, −education, −fnlwgt): {df.shape} → "
      f"{len(NUM)} numéricas + {len(CATEGORICAL)} categóricas + alvo")

secao("1C — ALVO")
cont = df[TARGET].value_counts().sort_index()
print({ROTULO[k]: int(v) for k, v in cont.items()},
      f"| minoritária (>50K) = {df[TARGET].mean() * 100:.2f}% | razão = {cont[0] / cont[1]:.2f}:1 | "
      f"baseline 'sempre ≤50K' acerta {cont[0] / len(df) * 100:.2f}%")

fig, ax = plt.subplots(figsize=(6, 4))
barras = ax.bar([ROTULO[k] for k in cont.index], cont.values,
                color=[COR_CLASSE[k] for k in cont.index])
for b, v in zip(barras, cont.values):
    ax.annotate(f"{v} ({v / len(df):.1%})", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom")
ax.set_title("Figura 1 — Distribuição do alvo (income)")
ax.set_xlabel("classe de renda anual")
ax.set_ylabel("número de pessoas")
ax.legend(barras, [f"{ROTULO[k]}" for k in cont.index], title="classe")
salva(fig, "fig01-alvo.png")

secao("1D — TREINO E TESTE")
X_tr, X_te, y_tr, y_te = split(df)
print(f"treino {X_tr.shape} | teste {X_te.shape} | >50K treino {y_tr.mean() * 100:.2f}% "
      f"| teste {y_te.mean() * 100:.2f}%")
tr = X_tr.assign(**{TARGET: y_tr.values})

# ============================================================ 2. UNIVARIATE (treino)
secao("2A — NUMÉRICAS (treino)")
desc = tr[NUM].describe().T
desc["median"] = tr[NUM].median()
desc["skew"] = tr[NUM].skew()
print(desc[["mean", "median", "std", "min", "25%", "50%", "75%", "max", "skew"]].round(2).to_string())
print(f"hours-per-week = 40 em {(tr['hours-per-week'] == 40).mean() * 100:.1f}% do treino")
print("education-num, 3 níveis mais comuns:",
      (tr["education-num"].value_counts(normalize=True).head(3) * 100).round(1).to_dict())
for c in NUM_SKEWED:
    nz = tr[c] > 0
    print(f"{c}: zeros {(~nz).mean() * 100:.1f}% | não-zero {nz.sum()} | mediana dos não-zero "
          f"{tr.loc[nz, c].median():.0f} | máx {tr[c].max()}")
print("capital-gain = 99999 no treino:", int((tr["capital-gain"] == 99999).sum()))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, c in zip(axes, NUM_PLAIN):
    if c == "education-num":
        vc = tr[c].value_counts().sort_index()
        ax.bar(vc.index, vc.values, color=CORES[3], label="contagem por nível")
    else:
        # um bin por valor inteiro: bins que não casam com inteiros criam vales falsos
        ax.hist(tr[c], bins=np.arange(tr[c].min() - 0.5, tr[c].max() + 1.5, 1),
                color=CORES[3], label="contagem")
    ax.axvline(tr[c].mean(), color=CORES[1], ls="--", label=f"média {tr[c].mean():.1f}")
    ax.axvline(tr[c].median(), color=CORES[0], ls="-", label=f"mediana {tr[c].median():.0f}")
    ax.set_xlabel(c)
    ax.set_ylabel("número de pessoas")
    ax.legend(fontsize=8)
fig.suptitle("Figura 2 — Distribuição das numéricas sem cauda pesada (treino)")
plt.tight_layout()
salva(fig, "fig02-numericas.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, c in zip(axes, NUM_SKEWED):
    nz = tr.loc[tr[c] > 0, c]
    # a borda final vai além do máximo: com logspace exato até 99999, o arredondamento de
    # ponto flutuante deixava os valores no teto fora do último bin
    ax.hist(nz, bins=np.logspace(np.log10(nz.min()), np.log10(nz.max() * 1.05), 40),
            color=CORES[0], label=f"não-zero: {len(nz)} ({len(nz) / len(tr):.1%})")
    ax.set_xscale("log")
    ticks = [100, 1000, 10000, 99999] if c == "capital-gain" else [200, 500, 1000, 2000, 4000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.plot([], [], " ", label=f"zeros: {(tr[c] == 0).mean():.1%} (fora do gráfico)")
    ax.set_xlabel(f"{c} (escala log, só valores > 0)")
    ax.set_ylabel("número de pessoas")
    ax.legend(fontsize=8)
n_teto = int((tr["capital-gain"] == 99999).sum())
axes[0].annotate(f"{n_teto} pessoas no teto 99.999", (99999, n_teto), xytext=(9000, 200),
                 arrowprops={"arrowstyle": "->"}, fontsize=8)
fig.suptitle("Figura 3 — capital-gain e capital-loss: massa em zero e cauda longa (treino)")
plt.tight_layout()
salva(fig, "fig03-capital.png")

secao("2B — CATEGÓRICAS (treino)")
for c in CATEGORICAL:
    vc = tr[c].value_counts(dropna=False)
    pct = vc / len(tr) * 100
    raras = pct[pct < 1]
    print(f"\n{c}: {tr[c].nunique()} níveis + NaN={tr[c].isna().sum()} | "
          f"top {vc.index[0]!r} {pct.iloc[0]:.1f}% | raras (<1%): {len(raras)} níveis, {int(vc[raras.index].sum())} linhas")
    print("   " + ", ".join(f"{k}={v:.1f}%" for k, v in pct.head(8).items()) + (" ..." if len(pct) > 8 else ""))
    if len(raras):
        print("   raras: " + ", ".join(f"{k}={v:.2f}%" for k, v in raras.items()))

baixa = [c for c in CATEGORICAL if c != "native-country"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, c in zip(axes.ravel(), baixa):
    pct = (tr[c].fillna("NaN").value_counts(normalize=True) * 100).sort_values()
    cores = [CORES[1] if v < 1 else CORES[3] for v in pct.values]
    ax.barh(pct.index.astype(str), pct.values, color=cores)
    for i, v in enumerate(pct.values):
        ax.text(v + pct.max() * 0.01, i, f"{v:.2f}%" if v < 1 else f"{v:.1f}%",
                va="center", fontsize=7, color=CORES[1] if v < 1 else "black")
    ax.axvline(1, color="black", ls=":", lw=1)
    ax.set_xlim(0, pct.max() * 1.18)
    ax.set_title(c)
    ax.set_xlabel("% do treino")
    ax.legend(handles=[Patch(color=CORES[3], label="≥ 1%"), Patch(color=CORES[1], label="< 1% (rara)")],
              fontsize=7, loc="lower right")
fig.suptitle("Figura 4 — Frequência das categóricas de baixa cardinalidade (treino)")
plt.tight_layout()
salva(fig, "fig04-categoricas.png")

c = "native-country"
pct = tr[c].fillna("NaN").value_counts(normalize=True) * 100
top = pct.head(10)
resto = pct.iloc[10:]
serie = pd.concat([top, pd.Series({f"outros {len(resto)} países": resto.sum()})]).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
cores = [CORES[1] if (v < 1 or k.startswith("outros")) else CORES[3] for k, v in serie.items()]
ax.barh(serie.index.astype(str), serie.values, color=cores)
ax.set_xscale("log")
ax.axvline(1, color="black", ls=":", lw=1)
ax.legend(handles=[Patch(color=CORES[3], label="≥ 1% do treino"),
                   Patch(color=CORES[1], label="< 1% (raro, ou soma de raros)")],
          fontsize=8, loc="lower right")
ax.set_title(f"Figura 5 — native-country: {tr[c].nunique()} países, cauda longa (treino)")
ax.set_xlabel("% do treino (escala log)")
ax.set_ylabel("país de origem")
salva(fig, "fig05-native-country.png")

# ============================================================ 3. BIVARIATE (treino)
secao("3A — NUMÉRICA × NUMÉRICA (treino)")
sp = tr[NUM].corr(method="spearman")
pe = tr[NUM].corr(method="pearson")
print("Spearman:\n" + sp.round(3).to_string())
print("Pearson:\n" + pe.round(3).to_string())
pares = [(sp.loc[a, b], a, b) for i, a in enumerate(NUM) for b in NUM[i + 1:]]
r_max, a_max, b_max = max(pares, key=lambda t: abs(t[0]))
print(f"par mais correlacionado (|Spearman|): {a_max} × {b_max} = {r_max:+.3f} "
      f"(Pearson {pe.loc[a_max, b_max]:+.3f})")
print(f"Spearman × Pearson em capital-gain × hours: {sp.loc['capital-gain', 'hours-per-week']:+.3f} vs "
      f"{pe.loc['capital-gain', 'hours-per-week']:+.3f}")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(sp.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(NUM)), NUM, rotation=35, ha="right")
ax.set_yticks(range(len(NUM)), NUM)
for i in range(len(NUM)):
    for j in range(len(NUM)):
        ax.text(j, i, f"{sp.values[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(sp.values[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax, label="ρ de Spearman")
fig.suptitle("Figura 6 — Correlação de Spearman entre as numéricas (treino)")
ax.set_xlabel("feature")
ax.set_ylabel("feature")
ax.grid(False)
salva(fig, "fig06-correlacao.png")

rng = np.random.default_rng(42)
amostra = tr.sample(4000, random_state=42)
fig, ax = plt.subplots(figsize=(7, 5))
for k in (0, 1):
    s = amostra[amostra[TARGET] == k]
    ax.scatter(s[a_max] + rng.uniform(-0.3, 0.3, len(s)), s[b_max] + rng.uniform(-0.3, 0.3, len(s)),
               s=6, alpha=0.35, color=COR_CLASSE[k], label=ROTULO[k])
ax.set_title(f"Figura 7 — {a_max} × {b_max}, o par mais correlacionado (ρ = {r_max:.2f})")
ax.set_xlabel(f"{a_max} (com jitter)")
ax.set_ylabel(f"{b_max} (com jitter)")
ax.legend(title="classe", markerscale=3)
salva(fig, "fig07-par-correlacionado.png")

secao("3B — CATEGÓRICA × ALVO (treino)")
taxa_geral = tr[TARGET].mean() * 100
print(f"taxa geral >50K no treino: {taxa_geral:.2f}%")
taxas = {}
for c in CATEGORICAL:
    t = tr.fillna({c: "NaN"}).groupby(c)[TARGET].agg(["mean", "size"])
    t["mean"] *= 100
    t = t.sort_values("mean", ascending=False)
    taxas[c] = t
    print(f"\n{c}: amplitude {t['mean'].max() - t['mean'].min():.1f} p.p.")
    print("   " + ", ".join(f"{k}={r['mean']:.1f}% (n={int(r['size'])})" for k, r in t.iterrows()
                            if r["size"] >= 50))

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, c in zip(axes.ravel(), baixa):
    t = taxas[c][taxas[c]["size"] >= 50].sort_values("mean")
    ax.barh(t.index.astype(str), t["mean"], color=[CORES[0] if v > taxa_geral else CORES[3] for v in t["mean"]])
    ax.axvline(taxa_geral, color="black", ls="--", lw=1, label=f"taxa geral {taxa_geral:.1f}%")
    ax.set_title(c)
    ax.set_xlabel("% com renda >50K")
    ax.legend(fontsize=7, loc="lower right")
fig.suptitle("Figura 8 — Proporção de >50K por categoria (treino; categorias com n ≥ 50)")
plt.tight_layout()
salva(fig, "fig08-categorica-alvo.png")

secao("3C — NUMÉRICA × CATEGÓRICA (treino)")
for c in NUM_PLAIN:
    for k in (0, 1):
        v = tr.loc[tr[TARGET] == k, c]
        q1, q2, q3 = v.quantile([0.25, 0.5, 0.75])
        print(f"{c:15} {ROTULO[k]:5}: mediana {q2:5.1f} | Q1 {q1:5.1f} | Q3 {q3:5.1f} | IQR {q3 - q1:5.1f} | dp {v.std():5.2f}")
for s in ["Male", "Female"]:
    v = tr.loc[tr["sex"] == s, "hours-per-week"]
    q1, q2, q3 = v.quantile([0.25, 0.5, 0.75])
    print(f"hours-per-week  {s:6}: mediana {q2:5.1f} | Q1 {q1:5.1f} | Q3 {q3:5.1f} | IQR {q3 - q1:5.1f} | dp {v.std():5.2f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, c in zip(axes, NUM_PLAIN):
    dados = [tr.loc[tr[TARGET] == k, c] for k in (0, 1)]
    bp = ax.boxplot(dados, tick_labels=[ROTULO[0], ROTULO[1]], patch_artist=True, widths=0.6)
    for patch, k in zip(bp["boxes"], (0, 1)):
        patch.set_facecolor(COR_CLASSE[k])
        patch.set_alpha(0.7)
    ax.set_xlabel("classe de renda")
    ax.set_ylabel(c)
    ax.legend(bp["boxes"], [ROTULO[0], ROTULO[1]], fontsize=8, title="classe")
fig.suptitle("Figura 9 — Numéricas por classe de renda (treino)")
plt.tight_layout()
salva(fig, "fig09-numerica-alvo.png")

fig, ax = plt.subplots(figsize=(6, 4.5))
dados = [tr.loc[tr["sex"] == s, "hours-per-week"] for s in ["Male", "Female"]]
bp = ax.boxplot(dados, tick_labels=["Male", "Female"], patch_artist=True, widths=0.6)
for patch, cor in zip(bp["boxes"], [CORES[3], CORES[1]]):
    patch.set_facecolor(cor)
    patch.set_alpha(0.7)
ax.set_title("Figura 10 — hours-per-week por sexo (treino)")
ax.set_xlabel("sexo")
ax.set_ylabel("horas trabalhadas por semana")
ax.legend(bp["boxes"], ["Male", "Female"], fontsize=8, title="sexo")
salva(fig, "fig10-horas-sexo.png")

# ============================================================ 4A — evidências (treino)
secao("4A — EVIDÊNCIAS PARA AS ESTRATÉGIAS (treino)")
wc_tr = tr["workclass"].isna()
print(f"faltantes no treino: workclass {wc_tr.sum()} | occupation {tr['occupation'].isna().sum()} | "
      f"native-country {tr['native-country'].isna().sum()}")
print(f">50K no treino — geral {taxa_geral:.2f}% | workclass NaN {tr.loc[wc_tr, TARGET].mean() * 100:.2f}% | "
      f"native-country NaN {tr.loc[tr['native-country'].isna(), TARGET].mean() * 100:.2f}%")
moda_wc = tr["workclass"].mode()[0]
print(f"se imputasse a moda ('{moda_wc}'), a taxa >50K de '{moda_wc}' passaria de "
      f"{tr.loc[tr['workclass'] == moda_wc, TARGET].mean() * 100:.2f}% para "
      f"{tr.loc[(tr['workclass'] == moda_wc) | wc_tr, TARGET].mean() * 100:.2f}%")
q1, q3 = tr["capital-gain"].quantile([0.25, 0.75])
flag = tr["capital-gain"] > q3 + 1.5 * (q3 - q1)
print(f"regra IQR em capital-gain: Q1={q1} Q3={q3} → {flag.sum()} 'outliers' ({flag.mean() * 100:.1f}%), "
      f"com {tr.loc[flag, TARGET].mean() * 100:.1f}% de >50K; contêm {int(tr.loc[flag, TARGET].sum())} "
      f"dos {int(tr[TARGET].sum())} positivos ({tr.loc[flag, TARGET].sum() / tr[TARGET].sum() * 100:.1f}%)")
afet = (tr["capital-gain"] > 0) | (tr["capital-loss"] > 0)
print(f"linhas afetadas pelo log1p (capital-gain>0 ou capital-loss>0): {afet.sum()} ({afet.mean() * 100:.1f}% do treino); "
      f"linhas removidas: 0")
for c in NUM_SKEWED:
    z = (tr[c].max() - tr[c].mean()) / tr[c].std()
    lz = np.log1p(tr[c])
    zl = (lz.max() - lz.mean()) / lz.std()
    print(f"{c}: máximo a {z:.1f} desvios da média sem log | a {zl:.1f} desvios com log1p")
print("faixas das numéricas no treino: " + " | ".join(
    f"{c} {tr[c].min()}–{tr[c].max()}" for c in NUM))
novas = {c: sorted(set(X_te[c].dropna()) - set(X_tr[c].dropna())) for c in CATEGORICAL}
print("categorias que aparecem no teste e não no treino:", {c: v for c, v in novas.items() if v} or "nenhuma")

# ============================================================ 4C — pipeline
secao("4C — PIPELINE (ajustado só no treino)")
prep = build_preprocessor()
Z_tr = prep.fit_transform(X_tr)
Z_te = prep.transform(X_te)
nomes = list(prep.get_feature_names_out())
print(f"NaN — treino {int(np.isnan(Z_tr).sum())} | teste {int(np.isnan(Z_te).sum())}")
print(f"shape — treino {Z_tr.shape} | teste {Z_te.shape}")
print(f"{len(nomes)} features: " + ", ".join(nomes))
n_num = len(NUM)
print("média das numéricas transformadas — treino:", Z_tr[:, :n_num].mean(axis=0).round(3).tolist())
print("média das numéricas transformadas — teste :", Z_te[:, :n_num].mean(axis=0).round(3).tolist())
onehot = prep.named_transformers_["cat"].named_steps["onehot"]
print("categorias agrupadas em 'infrequent' (aprendidas no treino):")
for c, inf in zip(CATEGORICAL, onehot.infrequent_categories_):
    if inf is not None:
        print(f"   {c}: {list(inf)}")
print("medianas aprendidas no treino:",
      dict(zip(NUM_PLAIN, prep.named_transformers_["num"].named_steps["imputa"].statistics_)))
nova = X_te.iloc[[0]].copy()
nova["occupation"] = "Astronaut"
z = prep.transform(nova)[0]
ativas = [nomes[i] for i in np.flatnonzero(z) if nomes[i].startswith("cat__occupation")]
print(f"teste com categoria inédita (occupation='Astronaut'): coluna ativada → {ativas}")
print("figuras salvas em", FIG)
