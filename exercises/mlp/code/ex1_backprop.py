"""Exercício 1 — um passo de backpropagation, conferido.

A dedução e as contas estão à mão no relatório. Este script refaz a MESMA conta com as
fórmulas explícitas (só aritmética, em precisão completa) para conferir cada número, e depois
cruza tudo com a classe genérica de mlp.py: se a conta à mão e o código genérico dão os mesmos
gradientes, um valida o outro.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp import MLP, MSETanh  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")  # acentos em UTF-8 também com a saída redirecionada


def mostra(nome, valor):
    print(f"{nome:16} = {np.array2string(np.asarray(valor), precision=6, floatmode='fixed', suppress_small=False)}")


# dados do enunciado
x = np.array([0.5, -0.2])
y = 1.0
W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
b1 = np.array([0.1, -0.2])
W2 = np.array([0.5, -0.3])
b2 = 0.2
eta = 0.3

# ------------------------------------------------------------------ A — forward
z1 = W1 @ x + b1
h1 = np.tanh(z1)
u2 = W2 @ h1 + b2
y_hat = np.tanh(u2)
L = (y - y_hat) ** 2
print("A — forward")
for nome, v in [("z1", z1), ("h1", h1), ("u2", u2), ("y_hat", y_hat), ("L", L)]:
    mostra(nome, v)

# ------------------------------------------------------------------ B — backward
dL_dyhat = -2.0 * (y - y_hat)
dL_du2 = dL_dyhat * (1.0 - y_hat ** 2)
dL_dW2 = dL_du2 * h1
dL_db2 = dL_du2
dL_dh1 = dL_du2 * W2
dL_dz1 = dL_dh1 * (1.0 - h1 ** 2)
dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1
print("\nB — backward")
for nome, v in [("1 - y_hat^2", 1 - y_hat ** 2), ("dL/dy_hat", dL_dyhat), ("dL/du2", dL_du2),
                ("dL/dW2", dL_dW2), ("dL/db2", dL_db2), ("dL/dh1", dL_dh1),
                ("1 - h1^2", 1 - h1 ** 2), ("dL/dz1", dL_dz1), ("dL/dW1", dL_dW1), ("dL/db1", dL_db1)]:
    mostra(nome, v)

# ------------------------------------------------------------------ C — update
W2_n = W2 - eta * dL_dW2
b2_n = b2 - eta * dL_db2
W1_n = W1 - eta * dL_dW1
b1_n = b1 - eta * dL_db1
print("\nC — parâmetros depois do update (eta = 0.3)")
for nome, v in [("W2", W2_n), ("b2", b2_n), ("W1", W1_n), ("b1", b1_n)]:
    mostra(nome, v)

# segunda passada: a perda sobe ou desce?
y_hat_2 = np.tanh(W2_n @ np.tanh(W1_n @ x + b1_n) + b2_n)
L_2 = (y - y_hat_2) ** 2
grad_quadrado = (np.sum(dL_dW1 ** 2) + np.sum(dL_db1 ** 2) + np.sum(dL_dW2 ** 2) + dL_db2 ** 2)
print("\nsegunda passada com os parâmetros novos")
mostra("y_hat novo", y_hat_2)
mostra("L novo", L_2)
mostra("variação de L", L_2 - L)
mostra("||grad||^2", grad_quadrado)
mostra("-eta*||grad||^2", -eta * grad_quadrado)

# ------------------------------------------------------------------ conferência com mlp.py
net = MLP([2, 2, 1], MSETanh, np.random.default_rng(42))  # pesos sobrescritos abaixo
net.W = [W1.copy(), W2.reshape(1, 2).copy()]
net.b = [b1.copy(), np.array([b2])]
saida = net.forward(x.reshape(1, 2))
gW, gb = net.backward(np.array([[y]]))
dif = max(abs(saida[0, 0] - y_hat),
          np.abs(gW[0] - dL_dW1).max(), np.abs(gb[0] - dL_db1).max(),
          np.abs(gW[1][0] - dL_dW2).max(), abs(gb[1][0] - dL_db2))
print(f"\nconferência com a classe MLP (mlp.py): maior diferença entre conta à mão e código genérico = {dif:.2e}")
