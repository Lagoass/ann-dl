"""Exercício 4 — uma segunda camada escondida ajuda? Gera a Figura 6.

Comparação controlada: o mesmo dado e o mesmo split do Exercício 3, as mesmas épocas, o mesmo
learning rate e a mesma largura (16). A única mudança é a profundidade. Cada arquitetura roda
com 5 seeds de inicialização (o enunciado pede pelo menos 3), para comparar modelos e não sorte.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from sklearn.metrics import confusion_matrix

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import dados  # noqa: E402
from mlp import MLP, CategoricalCrossEntropy, one_hot, padronizador  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")  # o console cp1252 não imprime "−" (U+2212)
FIG = BASE.parent / "figures"
FIG.mkdir(exist_ok=True)

CORES = ["#E8A13D", "#C75146", "#6B8F3D", "#33658A"]
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110,
})

EPOCAS, LR = 3000, 0.5                        # os mesmos do Exercício 3
ARQS = {"Ex. 3 — 1 camada escondida": [4, 16, 3],
        "Ex. 4 — 2 camadas escondidas": [4, 16, 16, 3]}
SEEDS = [42, 43, 44, 45, 46]                  # 42 é a seed do relatório; as outras dão a média

X, y, X_tr, X_te, y_tr, y_te = dados.ex3()
P = padronizador(X_tr)
Xs_tr, Xs_te, Y_tr = P(X_tr), P(X_te), one_hot(y_tr, 3)

res = {nome: [] for nome in ARQS}
for nome, arq in ARQS.items():
    for seed in SEEDS:
        net = MLP(arq, CategoricalCrossEntropy, np.random.default_rng(seed))
        # tamanho do gradiente que chega à primeira camada logo na inicialização
        net.forward(Xs_tr)
        g0 = np.linalg.norm(net.backward(Y_tr)[0][0])
        hist = net.fit(Xs_tr, Y_tr, EPOCAS, LR)
        pred_te = net.predict(Xs_te)
        cm = confusion_matrix(y_te, pred_te)
        res[nome].append(dict(
            seed=seed, hist=hist, params=net.n_params(), g0=g0,
            perda=CategoricalCrossEntropy.value(net.forward(Xs_tr), Y_tr),
            acc_tr=(net.predict(Xs_tr) == y_tr).mean(),
            acc_te=(pred_te == y_te).mean(),
            trocas=[int(cm[a, b] + cm[b, a]) for a, b in ((0, 1), (0, 2), (1, 2))],
        ))

print(f"mesmo orçamento nas duas: {EPOCAS} épocas, lr = {LR}, lote completo, mesmo split do Ex. 3")
for nome, arq in ARQS.items():
    print(f"\n{nome}: {'-'.join(map(str, arq))} — {res[nome][0]['params']} parâmetros")
    for r in res[nome]:
        print(f"   seed {r['seed']}: perda final de treino {r['perda']:.4f} | acurácia treino {r['acc_tr']:.4f} "
              f"| TESTE {r['acc_te']:.4f} | ||grad 1ª camada|| na init {r['g0']:.4f}")
    perdas = np.array([r["perda"] for r in res[nome]])
    acc_te = np.array([r["acc_te"] for r in res[nome]])
    acc_tr = np.array([r["acc_tr"] for r in res[nome]])
    g0s = np.array([r["g0"] for r in res[nome]])
    print(f"   MÉDIA ({len(SEEDS)} seeds): perda {perdas.mean():.4f} ± {perdas.std(ddof=1):.4f} | "
          f"treino {acc_tr.mean():.4f} | TESTE {acc_te.mean():.4f} ± {acc_te.std(ddof=1):.4f} | "
          f"||grad 1ª camada|| na init {g0s.mean():.4f}")

raso, fundo = (np.array([r["acc_te"] for r in res[n]]) for n in ARQS)
dif = 100 * (fundo - raso)  # pareado: mesma seed, mesmo dado
print(f"\ndiferença de acurácia de teste (Ex. 4 − Ex. 3), média das seeds: {dif.mean():+.2f} p.p.")
print(f"   por seed: {dif.round(2).tolist()} p.p. | desvio das diferenças {dif.std(ddof=1):.2f} p.p.")
print(f"seeds em que a rede mais funda ganhou: {int((fundo > raso).sum())} de {len(SEEDS)}")
for nome in ARQS:
    tr = np.mean([r["acc_tr"] for r in res[nome]])
    te = np.mean([r["acc_te"] for r in res[nome]])
    print(f"distância treino − teste, {nome}: {100 * (tr - te):.2f} p.p.")
# instabilidade: épocas em que a perda SUBIU (descida do gradiente com passo grande demais)
for nome in ARQS:
    saltos = [np.diff(r["hist"]) for r in res[nome]]
    excesso = max((r["hist"] - np.minimum.accumulate(r["hist"])).max() for r in res[nome])
    print(f"épocas com perda subindo, {nome}: {[int((s > 0).sum()) for s in saltos]} por seed "
          f"| maior subida em uma época: {max(s.max() for s in saltos):.4f} "
          f"| maior excesso sobre o mínimo anterior: {excesso:.4f}")
# o par mais confundido do Ex. 3D é robusto à seed? trocas no teste por par de classes
for nome in ARQS:
    print(f"trocas no teste (pares 0-1, 0-2, 1-2), {nome}: {[r['trocas'] for r in res[nome]]} por seed "
          f"| soma {np.sum([r['trocas'] for r in res[nome]], axis=0).tolist()}")
r42 = res["Ex. 3 — 1 camada escondida"][0]
print(f"conferência: a rede rasa com seed 42 reproduz o Ex. 3? perda final {r42['perda']:.4f} | teste {r42['acc_te']:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
for (nome, rs), cor in zip(res.items(), [CORES[3], CORES[1]]):
    for r in rs[1:]:
        ax.plot(np.arange(1, EPOCAS + 1), r["hist"], color=cor, alpha=0.25, lw=1)
    ax.plot(np.arange(1, EPOCAS + 1), rs[0]["hist"], color=cor, lw=2.2, label=f"{nome} (seed 42)")
    ax.plot([], [], color=cor, alpha=0.4, lw=1, label=f"{nome}: outras 4 seeds")
ax.set_yscale("log")
ax.set_yticks([0.15, 0.2, 0.3, 0.5, 1.0])
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_title("Figura 6 — Perda de treino: 1 × 2 camadas escondidas (mesmo dado, mesmo orçamento)")
ax.set_xlabel("época")
ax.set_ylabel("entropia cruzada (escala log)")
ax.legend()
fig.savefig(FIG / "fig6.png", bbox_inches="tight")
plt.close(fig)
print("figuras salvas em ../figures/")
