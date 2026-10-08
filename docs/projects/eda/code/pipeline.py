"""Pré-processamento do projeto (Adult Income) — arquivo importável.

Uso, a partir de outro script ou notebook nesta pasta:

    from pipeline import load_clean, split, build_preprocessor

    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)
    prep = build_preprocessor()
    Z_train = prep.fit_transform(X_train)   # aprende medianas, modas, médias e desvios SÓ no treino
    Z_test = prep.transform(X_test)         # o teste é apenas transformado

Cada escolha abaixo aponta para o achado do EDA que a motiva (seções do relatório entre
parênteses).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
DATA = Path(__file__).resolve().parents[2] / "data" / "adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
TARGET = "income"

# Colunas descartadas (1B)
DROP = {
    "education": "codifica exatamente o mesmo que education-num (bijeção de 16 níveis)",
    "fnlwgt": "peso amostral do censo, não atributo da pessoa; correlação com o alvo ≈ 0",
}

NUM_PLAIN = ["age", "education-num", "hours-per-week"]   # assimetria baixa (2A)
NUM_SKEWED = ["capital-gain", "capital-loss"]            # >90% zeros e cauda até 99999 (2A, 4A)
CATEGORICAL = [
    "workclass", "marital-status", "occupation", "relationship",
    "race", "sex", "native-country",
]


def load_raw():
    """O arquivo original da UCI, sem nenhuma alteração; '?' vira NaN."""
    return pd.read_csv(DATA, header=None, names=COLUMNS,
                       skipinitialspace=True, na_values="?")


def load_clean():
    """Remove as duplicatas exatas e as colunas de DROP; alvo vira 0/1 (1 = >50K)."""
    df = load_raw().drop_duplicates().drop(columns=list(DROP))
    df[TARGET] = (df[TARGET] == ">50K").astype(int)
    return df.reset_index(drop=True)


def split(df, test_size=0.2):
    """Split 80/20 estratificado pelo alvo (24% de positivos), com seed fixa (1D)."""
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)


def build_preprocessor(min_frequency=0.01):
    """ColumnTransformer com um tratamento por grupo de colunas (4A).

    - numéricas comuns: mediana (defensiva — não há faltantes) + padronização, porque as
      escalas vão de 1–16 (education-num) a 1–99 (hours-per-week) e a rede é treinada por
      gradiente;
    - capital-gain/loss: log1p antes de padronizar — sem isso o teto de 99999 fica a
      dezenas de desvios da média e domina as ativações; nenhuma linha é removida;
    - categóricas: o faltante vira a categoria própria "Unknown" (a ausência é
      informativa), one-hot completo (sem drop: não distorce distâncias), e categorias com
      menos de 1% do treino viram um balde "infrequent" — que também recebe qualquer
      categoria nova que apareça só no teste.
    """
    num_plain = Pipeline([
        ("imputa", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
    ])
    num_skewed = Pipeline([
        ("imputa", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escala", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputa", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=min_frequency, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_plain, NUM_PLAIN),
        ("num_log", num_skewed, NUM_SKEWED),
        ("cat", categorical, CATEGORICAL),
    ])
