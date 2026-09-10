# ann-dl — entregas

Entregas de **Redes Neurais Artificiais & Deep Learning** — Insper 2026.2 — Gustavo Lagôa.

📄 **Site publicado:** <https://lagoass.github.io/ann-dl/>

Baseado no [template de entrega](https://github.com/hsandmann/documentation.template)
da disciplina (MkDocs + Material for MkDocs).

## Rodando localmente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt --upgrade
mkdocs serve -o
```

No Linux/macOS troque a ativação por `source ./.venv/bin/activate`.

Antes de publicar, vale checar links quebrados e páginas fora do `nav`:

```shell
mkdocs build --strict
```

## Publicação

O workflow [`.github/workflows/main.yaml`](.github/workflows/main.yaml) roda
`mkdocs gh-deploy --force` a cada `push` na branch `main`, publicando o site na branch
`gh-pages`. Não é preciso rodar nada na mão.

Configuração necessária **uma única vez** no repositório:

- *Settings → Pages* → **Source: Deploy from a branch** → `gh-pages` / `(root)`

## Estrutura

```
docs/
├── index.md                  # home + checklist de entregas
├── exercises/                # entregas individuais
│   ├── data/                 # 1 — Data
│   ├── perceptron/           # 2 — Perceptron
│   ├── mlp/                  # 3 — MLP
│   ├── transformers/         # 4 — Transformers & Attention
│   ├── vae/                  # 5 — VAE
│   └── llm-finetuning/       # 6 — LLM Fine-Tuning
├── projects/                 # entregas em equipe
│   ├── classification/       # 1 — Classification
│   ├── regression/           # 2 — Regression
│   └── generative/           # 3 — Generative
└── setup/                    # como o site funciona + cheat sheet
```

Cada nova página precisa ser registrada no `nav` do [`mkdocs.yml`](mkdocs.yml).
Notebooks `.ipynb` podem ir direto para dentro de `docs/` — o plugin `mkdocs-jupyter`
renderiza as saídas já salvas (`execute: false`).
