# `scripts/` — operational helpers

Stand-alone scripts that support the project but are not part of the
runtime code under `core/`. Each script is independent: it owns its
own dependencies and entry point, and is safe to delete without
affecting the application.

## Contents

| File              | Purpose                                                                                  |
|-------------------|------------------------------------------------------------------------------------------|
| `env_setup.py`    | Cross-platform venv bootstrap (`python -m venv` + activation + `pip install -r requirements.txt`). |
| `wake_up.py`      | Playwright robot that pings the deployed Streamlit app to wake it from sleep.            |
| `requirements.txt`| Extra dependencies needed by `wake_up.py` only (Playwright). Not required for `core/`.   |

## Wake-up bot

Automates clicking the wake-up button of the deployed Streamlit app.

```bash
pip install -r scripts/requirements.txt
playwright install
python scripts/wake_up.py "https://your-app-url"
```

## Environment setup

```bash
python scripts/env_setup.py
```

Creates a virtual environment under `.venv/` and installs the runtime
dependencies declared in the top-level `requirements.txt`.
