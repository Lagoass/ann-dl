# Exercício 3 — Preparando dados reais (Spaceship Titanic) para uma rede com tanh
# Gera a Figura 6 e o histograma do log em ../figures/ e imprime todos os números.
# Regra que rege o script inteiro: toda estatística é calculada SÓ no treino.
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
FIGURAS = BASE.parent / "figures"
FIGURAS.mkdir(exist_ok=True)

SEED = 42

# identidade visual dos meus gráficos: paleta fixa + eixos limpos
CORES = ["#E8A13D", "#C75146", "#6B8F3D", "#33658A"]  # âmbar, telha, oliva, aço
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.prop_cycle": plt.cycler(color=CORES),
    "figure.dpi": 110,
})

df = pd.read_csv(BASE.parent / "spaceship-titanic" / "train.csv")
print("shape bruto:", df.shape)

# ------------------------------------------------------------ A — Conhecer os dados
print(df["Transported"].value_counts().to_string())
print(f"Proporção da classe positiva (True): {df['Transported'].mean():.4f}")

GASTOS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERICAS = ["Age"] + GASTOS
CATEGORICAS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]

# Tabela de valores faltantes por coluna
faltantes = pd.DataFrame({
    "faltantes": df.isna().sum(),
    "%": (df.isna().mean() * 100).round(2),
}).sort_values("faltantes", ascending=False)
print(faltantes.to_string())

# Estatísticas das colunas de gasto (só descrição — nada é ajustado aqui)
print(df[GASTOS].agg(["mean", "median", "max"]).round(2).to_string())

# ---------------------------------------------------- B — Separar antes de transformar
X = df.drop(columns=["Transported", "Cabin", "Name", "PassengerId"])
y = df["Transported"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
X_train = X_train.copy()
X_test = X_test.copy()
print(f"treino: {X_train.shape} | teste: {X_test.shape}")
print(f"proporção True — treino: {y_train.mean():.4f} | teste: {y_test.mean():.4f}")

# ------------------------------------------------------------------ C — Pré-processar
# Valores faltantes: mediana nas numéricas, moda nas categóricas (só do treino)
medianas = X_train[NUMERICAS].median()
modas = X_train[CATEGORICAS].mode().iloc[0]

X_train[NUMERICAS] = X_train[NUMERICAS].fillna(medianas)
X_test[NUMERICAS] = X_test[NUMERICAS].fillna(medianas)
X_train[CATEGORICAS] = X_train[CATEGORICAS].fillna(modas)
X_test[CATEGORICAS] = X_test[CATEGORICAS].fillna(modas)

print("medianas (treino):", medianas.to_dict())
print("modas (treino):   ", modas.to_dict())

# Feature engineering: TotalSpend (depois da imputação, senão a soma vira NaN)
X_train["TotalSpend"] = X_train[GASTOS].sum(axis=1)
X_test["TotalSpend"] = X_test[GASTOS].sum(axis=1)

# Caudas pesadas: log(1 + x) nos gastos e no TotalSpend
COLS_LOG = GASTOS + ["TotalSpend"]
spa_antes = X_train["Spa"].copy()  # guardo para o histograma
X_train[COLS_LOG] = np.log1p(X_train[COLS_LOG])
X_test[COLS_LOG] = np.log1p(X_test[COLS_LOG])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(spa_antes, bins=50, color=CORES[3])
axes[0].set_title("Spa — antes (escala original)")
axes[0].set_xlabel("gasto")
axes[1].hist(X_train["Spa"], bins=50, color=CORES[0])
axes[1].set_title("Spa — depois de log(1 + x)")
axes[1].set_xlabel("log(1 + gasto)")
for ax in axes:
    ax.set_ylabel("contagem")
fig.suptitle("Efeito da transformação log(1+x) em uma coluna de gasto (treino)")
plt.tight_layout()
fig.savefig(FIGURAS / "fig_log_spa.png", bbox_inches="tight")
plt.close(fig)

# Categóricas: one-hot (ajustado só no treino)
codificador = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
codificador.fit(X_train[CATEGORICAS])
onehot_treino = codificador.transform(X_train[CATEGORICAS])
onehot_teste = codificador.transform(X_test[CATEGORICAS])
print("categorias aprendidas no treino:")
for col, cats in zip(CATEGORICAS, codificador.categories_):
    print(f"  {col}: {list(cats)}")

# Escala: normalização para [-1, 1] nas numéricas, na mão (min/max só do treino)
NUM_FINAIS = NUMERICAS + ["TotalSpend"]
minimos = X_train[NUM_FINAIS].min()
maximos = X_train[NUM_FINAIS].max()

num_treino = 2 * (X_train[NUM_FINAIS] - minimos) / (maximos - minimos) - 1
num_teste = 2 * (X_test[NUM_FINAIS] - minimos) / (maximos - minimos) - 1

# matriz final = numéricas escaladas + one-hot (que já vive em {0, 1} ⊂ [-1, 1])
nomes_features = NUM_FINAIS + list(codificador.get_feature_names_out(CATEGORICAS))
X_train_final = np.hstack([num_treino.to_numpy(), onehot_treino])
X_test_final = np.hstack([num_teste.to_numpy(), onehot_teste])

print(f"treino — min: {X_train_final.min():.4f} | max: {X_train_final.max():.4f}")
print(f"teste  — min: {X_test_final.min():.4f} | max: {X_test_final.max():.4f}")

# ------------------------------------------------------- D — Verificar e visualizar
# Figura 6 — FoodCourt antes (bruto) e depois (pipeline completo)
idx_fc = nomes_features.index("FoodCourt")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(df["FoodCourt"].dropna(), bins=50, color=CORES[3])
axes[0].set_title("Antes — valores brutos")
axes[0].set_xlabel("FoodCourt")
axes[1].hist(X_train_final[:, idx_fc], bins=50, color=CORES[0])
axes[1].set_title("Depois — imputação + log(1+x) + escala [-1, 1]")
axes[1].set_xlabel("FoodCourt processado")
for ax in axes:
    ax.set_ylabel("contagem")
fig.suptitle("Figura 6 — FoodCourt antes e depois do pré-processamento (treino)")
plt.tight_layout()
fig.savefig(FIGURAS / "fig6.png", bbox_inches="tight")
plt.close(fig)

# Checagens finais, explícitas
print(f"NaN restantes — treino: {np.isnan(X_train_final).sum()} | "
      f"teste: {np.isnan(X_test_final).sum()}")
print(f"shape final — treino: {X_train_final.shape} | teste: {X_test_final.shape}")
print(f"faixa de valores — treino: [{X_train_final.min():.4f}, {X_train_final.max():.4f}] | "
      f"teste: [{X_test_final.min():.4f}, {X_test_final.max():.4f}]")
fc_bruto = df.loc[X_train.index, "FoodCourt"]  # FoodCourt do treino, antes de transformar
print(f"média/mediana de FoodCourt no treino ANTES de transformar: "
      f"{fc_bruto.mean():.2f} / {fc_bruto.median():.2f}")

print("figuras salvas em", FIGURAS)
