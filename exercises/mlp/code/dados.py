"""Geração e split dos datasets dos Exercícios 2 a 4, exatamente com os parâmetros do enunciado.

O scikit-learn entra só onde o enunciado permite: gerar os dados e dividir treino/teste.
Os Exercícios 3 e 4 chamam a MESMA função — mesmo dado, mesmo split.
"""
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split

CENTROS_EX2 = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])

PARAMS_EX3 = dict(n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
                  n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=42)


def ex2():
    """Classe 0 no centro, classe 1 em dois blocos opostos. Devolve também o bloco de origem."""
    X, c = make_blobs(n_samples=[500, 250, 250], centers=CENTROS_EX2,
                      cluster_std=1.2, random_state=42)
    y = (c > 0).astype(int)  # classe 0: 1 bloco | classe 1: 2 blocos
    # passar c junto não muda o split: os índices dependem só de tamanho, seed e estratificação
    X_tr, X_te, y_tr, y_te, c_tr, c_te = train_test_split(
        X, y, c, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te, c_tr, c_te


def ex3():
    X, y = make_classification(**PARAMS_EX3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_tr, X_te, y_tr, y_te


def ex3_sem_embaralhar():
    """O MESMO dataset do Ex. 3 com shuffle=False — as amostras saem agrupadas por cluster.

    O embaralhamento é o último passo do make_classification: com shuffle=False o gerador é
    consumido da mesma forma, então os pontos são idênticos, só que em ordem (o cluster k
    ocupa as linhas 250k a 250k+249) e com as colunas na ordem original. Distâncias não
    dependem da ordem das colunas — dá para medir a geometria dos 6 clusters exatamente.
    """
    return make_classification(**{**PARAMS_EX3, "shuffle": False})
