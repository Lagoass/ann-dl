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

Os slugs e o formato `index.md` + `code/` + `figures/` são contrato com a correção
([regras de submissão](https://insper.github.io/ann-dl/2026.2/exercises/submission/)) —
não renomear.

```
docs/
├── index.md                  # capa: entregas, prazos e pesos
├── exercises/                # 4 entregas individuais
│   ├── index.md              # índice da seção
│   ├── data/                 # relatório + code/ + figures/ + notebooks anexos
│   ├── perceptron/           # idem
│   ├── mlp/                  # a fazer (13/out)
│   └── vae/                  # a fazer (22/out)
├── projects/                 # UM projeto em três entregas
│   ├── index.md              # equipe, dataset, registro de decisões
│   ├── eda/                  # 1ª entrega (08/out)
│   ├── classification/       # 2ª entrega — escolher uma...
│   ├── regression/           # ...e apagar a outra (05/nov)
│   └── generative/           # 3ª entrega (20/nov)
├── raciocinio/               # thought process por entrega (preparo da defesa oral)
└── setup/                    # como o site funciona + cheat sheet
notebooks-src/                # fontes jupytext dos notebooks anexos do Data
```

Cada nova página precisa ser registrada no `nav` do [`mkdocs.yml`](mkdocs.yml) — e, em
seções que agrupam entregas, o `index.md` da seção tem que ser o **primeiro** filho
(o `navigation.indexes` do Material absorve no título o primeiro filho chamado
`index.md`, e sem isso a primeira entrega some do menu).

O código de cada entrega vive em `code/` como arquivo executável e é puxado para o
relatório com `--8<--`; as figuras são commitadas em `figures/`. Os notebooks `.ipynb`
anexos são renderizados pelo `mkdocs-jupyter` com `execute: false`, ou seja, exibem as
saídas já salvas.
