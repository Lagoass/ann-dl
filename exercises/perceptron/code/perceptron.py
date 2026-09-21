# Perceptron de camada única, escrito à mão — o enunciado proíbe qualquer modelo pronto.
# Escrito uma vez aqui e reusado, sem alteração, nos dois exercícios.
import numpy as np


def step(z):
    """Função degrau: 1 se z >= 0, senão 0."""
    return (z >= 0).astype(int)


def predict(w, b, X):
    """y_hat = step(w · x + b), para todas as linhas de X de uma vez."""
    return step(X @ w + b)


def accuracy(w, b, X, y):
    """Fração de acertos no dataset inteiro."""
    return float((predict(w, b, X) == y).mean())


def train(X, y, w0, b0=0.0, eta=0.01, max_epochs=100, pocket=False):
    """Treina amostra a amostra, na ordem em que o dataset está.

    Regra de update (rótulos 0/1):  w <- w + eta * (y - y_hat) * x ;  b <- b + eta * (y - y_hat)
    Para quando uma época inteira não gera update, ou em max_epochs.

    Com pocket=True, toda vez que um update produz acurácia (no dataset inteiro) maior que a
    melhor já vista, copio (w, b) para o "bolso" — é a única coisa que o pocket acrescenta.
    """
    w = np.array(w0, dtype=float).copy()  # cópia: não mexer na init de quem chamou
    b = float(b0)

    melhor = {"w": w.copy(), "b": b, "acc": accuracy(w, b, X, y), "epoca": 0}
    hist = {"acc": [], "updates": [], "melhor_acc": []}  # um valor por época

    for epoca in range(1, max_epochs + 1):
        updates = 0
        for xi, yi in zip(X, y):
            y_hat = 1 if (w @ xi + b) >= 0 else 0
            erro = yi - y_hat  # 0 (acertou), +1 ou -1 (os dois tipos de erro)
            if erro != 0:
                w += eta * erro * xi
                b += eta * erro
                updates += 1
                if pocket:
                    acc_agora = accuracy(w, b, X, y)  # só recalculo quando houve update
                    if acc_agora > melhor["acc"]:
                        melhor = {"w": w.copy(), "b": b, "acc": acc_agora, "epoca": epoca}

        hist["acc"].append(accuracy(w, b, X, y))
        hist["updates"].append(updates)
        hist["melhor_acc"].append(melhor["acc"])
        if updates == 0:  # época inteira sem erro: convergiu
            break

    return {"w": w, "b": b, "epocas": epoca, "acc": hist["acc"][-1], "hist": hist,
            "pocket": melhor}
