---
tags: [codigo, devops, bootstrap]
file: env-setup.py
loc: 50
---

# `env-setup.py`

Script multi-OS para criar o ambiente virtual e instalar dependências.

## Comportamento

- **Windows**: `python -m venv venv` + `.\venv\Scripts\pip install -r requirements.txt`.
- **Linux/macOS**: usa `sys.executable -m venv venv` + `./venv/bin/pip install -r requirements.txt`.

## Ponto de atenção

- Depende de `requirements.txt` válido. Atualmente o arquivo está em **UTF-16/BOM** ⇒ pode quebrar `pip install -r`. Ver [[07_Issues/Issue - requirements.txt UTF-16]].
