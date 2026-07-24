---
tags: [codigo, devops, automacao]
file: ops/wake_up.py
loc: 113
---

# `ops/wake_up.py`

Robô **Playwright** que abre a URL do app no Streamlit Cloud e clica no botão de "Wake Up" (idempotente).

## Configuração (`WakeUpConfig` dataclass)

| Campo | Default da dataclass | Default do CLI (`main`) |
|---|---|---|
| `url` | obrigatório | argv[1] |
| `button_text` | `"Wake Up"` | `"Wake Up"` |
| `button_selector` | `None` | `"button[data-testid='wakeup-button-viewer']"` |
| `wait_seconds_after_click` | 10 | 10 |
| `headless` | True | True |

> Note que o CLI **injeta** `button_selector` mesmo a dataclass tendo `None` como default — o uso programático sem `main` não tem o selector pronto.

## Uso

```bash
pip install -r ops/requirements.txt
playwright install
python ops/wake_up.py "https://your-app-url"
```

## Estratégia de clique

1. Tenta `page.wait_for_selector(button_selector)` — usa o data-testid do Streamlit.
2. Fallback `page.get_by_role("button", name="Wake Up")`.
3. Fallback `page.locator("text=Wake Up")`.

Se nenhum funcionar, assume que o app já está acordado e retorna 0.

## Quando rodar

Ideal em scheduler (cron / GitHub Actions) periodicamente para evitar que o app entre em sleep no Streamlit Cloud.
