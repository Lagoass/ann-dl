"""Checagem de gradiente: o backward escrito à mão contra diferenças finitas centradas.

Para cada parâmetro θ, compara o dL/dθ do backward com [L(θ + ε) − L(θ − ε)] / 2ε.
Se a dedução de alguma derivada estivesse errada, o erro relativo seria da ordem de 1;
com o backward certo, ele fica no nível do arredondamento numérico (< 1e-6).

ε = 1e-5 equilibra os dois erros da diferença finita: o de truncamento (cresce com ε²) e o
de arredondamento (cresce com 1e-16/ε). Com ε = 1e-6 ou 1e-7 o erro relativo AUMENTA nos
gradientes muito pequenos — sinal de arredondamento, não de derivada errada.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp import MLP, BinaryCrossEntropy, CategoricalCrossEntropy, MSETanh, one_hot  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")  # o console cp1252 não imprime "→"
EPS = 1e-5
casos = [
    ("MSE + tanh, 2-2-1 (a rede do Ex. 1)", [2, 2, 1], MSETanh),
    ("BCE + sigmoide, 2-8-1 (a rede do Ex. 2)", [2, 8, 1], BinaryCrossEntropy),
    ("CE + softmax, 4-16-3 (a rede do Ex. 3)", [4, 16, 3], CategoricalCrossEntropy),
    ("CE + softmax, 4-16-16-3 (a rede do Ex. 4)", [4, 16, 16, 3], CategoricalCrossEntropy),
    ("BCE + sigmoide, 3 camadas escondidas 2-5-4-3-1", [2, 5, 4, 3, 1], BinaryCrossEntropy),
]

rng = np.random.default_rng(42)
pior = 0.0
for nome, sizes, loss in casos:
    net = MLP(sizes, loss, rng)
    X = rng.normal(size=(9, sizes[0]))
    if loss is CategoricalCrossEntropy:
        y = one_hot(rng.integers(0, sizes[-1], 9), sizes[-1])
    elif loss is BinaryCrossEntropy:
        y = rng.integers(0, 2, (9, 1)).astype(float)
    else:
        y = rng.uniform(-1, 1, (9, 1))

    net.forward(X)
    gW, gb = net.backward(y)

    maior = 0.0
    maior_abs = 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for P, G in zip(params, grads):
            for idx in np.ndindex(P.shape):
                original = P[idx]
                P[idx] = original + EPS
                l_mais = loss.value(net.forward(X), y)
                P[idx] = original - EPS
                l_menos = loss.value(net.forward(X), y)
                P[idx] = original
                numerico = (l_mais - l_menos) / (2 * EPS)
                rel = abs(numerico - G[idx]) / max(1e-12, abs(numerico) + abs(G[idx]))
                maior = max(maior, rel)
                maior_abs = max(maior_abs, abs(numerico - G[idx]))
    pior = max(pior, maior)
    print(f"{nome:48} {net.n_params():4} parâmetros | maior erro relativo = {maior:.2e} "
          f"| maior erro absoluto = {maior_abs:.2e}")

print(f"\npior caso entre todos: {pior:.2e} → backward {'CORRETO' if pior < 1e-6 else 'COM ERRO'}")
