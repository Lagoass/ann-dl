"""MLP escrita à mão, só com numpy: ativações, perdas, forward, backward e update.

O mesmo arquivo serve aos Exercícios 2, 3 e 4 — e confere a conta à mão do Exercício 1.
Entre um exercício e outro mudam só os ARGUMENTOS (tamanhos das camadas, perda e
hiperparâmetros); forward, backward e update são o mesmo código, sem nenhuma alteração.

Convenção do enunciado: cada camada calcula u = W a + b, com W de forma (saída, entrada).
Em lote, cada linha de A é uma amostra, então u = A Wᵀ + b.
"""
import numpy as np


# ------------------------------------------------------------------------------ perdas
# Cada perda traz junto a ativação da camada de saída e devolve dL/du, o gradiente na
# PRÉ-ativação da saída. É isso que deixa o backward idêntico para qualquer perda.

class MSETanh:
    """Erro quadrático médio com saída tanh — a rede do Exercício 1."""

    name = "MSE + tanh"

    @staticmethod
    def activate(u):
        return np.tanh(u)

    @staticmethod
    def value(y_hat, y):
        return float(np.mean(np.sum((y - y_hat) ** 2, axis=1)))

    @staticmethod
    def grad_u(y_hat, y):
        # dL/dŷ = (2/N)(ŷ − y), vezes a derivada da tanh: 1 − ŷ²
        return (2.0 / y.shape[0]) * (y_hat - y) * (1.0 - y_hat ** 2)


class BinaryCrossEntropy:
    """Entropia cruzada binária com saída sigmoide — Exercício 2."""

    name = "BCE + sigmoide"

    @staticmethod
    def activate(u):
        return 0.5 * (1.0 + np.tanh(0.5 * u))  # é a sigmoide, sem overflow para |u| grande

    @staticmethod
    def value(y_hat, y):
        p = np.clip(y_hat, 1e-12, 1.0 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    @staticmethod
    def grad_u(y_hat, y):
        # a derivada da sigmoide cancela com o 1/[ŷ(1 − ŷ)] da BCE: sobra ŷ − y
        return (y_hat - y) / y.shape[0]


class CategoricalCrossEntropy:
    """Entropia cruzada categórica com saída softmax — Exercícios 3 e 4."""

    name = "CE + softmax"

    @staticmethod
    def activate(u):
        e = np.exp(u - u.max(axis=1, keepdims=True))  # subtrair o máximo evita overflow
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def value(y_hat, y):
        p = np.clip(y_hat, 1e-12, 1.0)
        return float(-np.mean(np.sum(y * np.log(p), axis=1)))

    @staticmethod
    def grad_u(y_hat, y):
        # o jacobiano da softmax cancela com o 1/ŷ do log: sobra ŷ − y (dedução no Ex. 3B)
        return (y_hat - y) / y.shape[0]


# ------------------------------------------------------------------------------ a rede

class MLP:
    """Perceptron multicamadas: tanh nas camadas escondidas, saída definida pela perda.

    layer_sizes = [entradas, escondida_1, ..., escondida_k, saídas] — quantas camadas
    se quiser; nada aqui pressupõe uma camada escondida só.
    """

    def __init__(self, layer_sizes, loss, rng):
        self.sizes = list(layer_sizes)
        self.loss = loss
        self.W, self.b = [], []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            desvio = np.sqrt(2.0 / (fan_in + fan_out))  # inicialização de Glorot, feita para tanh
            self.W.append(rng.normal(0.0, desvio, size=(fan_out, fan_in)))
            self.b.append(np.zeros(fan_out))

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    # ---------------- núcleo: forward, backward e update (o mesmo nos Exercícios 2, 3 e 4)

    def forward(self, X):
        """Propaga o lote X camada a camada e guarda as ativações para o backward."""
        self.acts = [X]
        a = X
        ultima = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            u = a @ W.T + b
            a = self.loss.activate(u) if l == ultima else np.tanh(u)
            self.acts.append(a)
        return a

    def backward(self, y):
        """Regra da cadeia de trás para frente; devolve os gradientes de cada W e b."""
        delta = self.loss.grad_u(self.acts[-1], y)  # dL/du da camada de saída
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for l in range(len(self.W) - 1, -1, -1):
            a_ant = self.acts[l]                    # entrada da camada l
            gW[l] = delta.T @ a_ant                 # dL/dW = δᵀ · ativação anterior
            gb[l] = delta.sum(axis=0)               # dL/db = δ somado no lote
            if l > 0:
                # volta pela camada: dL/da = δ W, depois pela tanh: (1 − a²)
                delta = (delta @ self.W[l]) * (1.0 - a_ant ** 2)
        return gW, gb

    def update(self, gW, gb, lr):
        """Descida do gradiente: cada parâmetro anda −lr vezes o seu gradiente."""
        for l in range(len(self.W)):
            self.W[l] -= lr * gW[l]
            self.b[l] -= lr * gb[l]

    # ---------------- conveniências

    def fit(self, X, y, epochs, lr):
        """Gradiente em lote completo; devolve a perda de treino de cada época."""
        historico = []
        for _ in range(epochs):
            y_hat = self.forward(X)
            historico.append(self.loss.value(y_hat, y))
            gW, gb = self.backward(y)
            self.update(gW, gb, lr)
        return np.array(historico)

    def predict(self, X):
        p = self.forward(X)
        return (p[:, 0] >= 0.5).astype(int) if p.shape[1] == 1 else p.argmax(axis=1)


# ------------------------------------------------------------------------------ utilidades

def padronizador(X_treino):
    """Média e desvio aprendidos SÓ no treino; devolve a função que padroniza qualquer X."""
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0)
    return lambda X: (X - media) / desvio


def one_hot(y, k):
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1.0
    return Y
