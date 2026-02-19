from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


@dataclass(frozen=True)
class WakeUpConfig:
    """Parâmetros de execução do robô de *wake up*.

    Centraliza as opções necessárias para abrir a página do app, identificar o botão de
    reativação (quando existir) e encerrar o navegador após alguns segundos.
    """

    url: str
    # Texto do botão (caso o botão tenha um label visível)
    button_text: str = "Wake Up"
    # Selector CSS alternativo (caso o texto não funcione)
    button_selector: str | None = None
    # Tempo para esperar após clicar
    wait_seconds_after_click: int = 10
    # Rodar invisível (headless) por padrão
    headless: bool = True


def wake_up_via_browser(cfg: WakeUpConfig) -> int:
    """Abre a página do app, aciona o *wake up* (quando necessário) e fecha o navegador.

    O fluxo é idempotente: se a página estiver em "sleep", o botão de *wake up* existe e
    será clicado. Se a página já estiver acordada, o botão não aparece; nesse caso, o
    script considera sucesso e apenas aguarda alguns segundos antes de fechar.

    :param cfg: Configurações do robô (URL, seletor do botão, tempo de espera, etc.).
    :return: 0 em caso de sucesso; 1 em caso de falha.
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=cfg.headless)
        page = browser.new_page()

        try:
            page.goto(cfg.url, wait_until="domcontentloaded", timeout=60_000)

            # Fluxo idempotente:
            # - Se a página estiver "dormindo", o botão de wake up existe -> clicamos.
            # - Se a página já estiver acordada, o botão não existe -> consideramos sucesso.

            if cfg.button_selector:
                try:
                    # Espera pouco tempo; se não aparecer, assume que já está acordado.
                    page.wait_for_selector(cfg.button_selector, timeout=5_000)
                    page.locator(cfg.button_selector).first.click(timeout=15_000)
                    print(f"Clicou via selector: {cfg.button_selector}")
                except PlaywrightTimeoutError:
                    print("Botão de wake up não encontrado (provavelmente já está acordado).")
            else:
                # 2) Tentativa por role=button com nome (mais semântico)
                try:
                    page.get_by_role("button", name=cfg.button_text).click(timeout=5_000)
                    print(f"Clicou no botão (role=button): {cfg.button_text}")
                except PlaywrightTimeoutError:
                    # 3) Fallback por texto (menos robusto, mas ajuda)
                    try:
                        page.locator(f"text={cfg.button_text}").first.click(timeout=5_000)
                        print(f"Clicou via texto (fallback): {cfg.button_text}")
                    except PlaywrightTimeoutError:
                        print("Botão de wake up não encontrado (provavelmente já está acordado).")

            time.sleep(cfg.wait_seconds_after_click)
            return 0

        except Exception as e:
            print(f"Erro ao executar wake up: {e}")
            return 1

        finally:
            browser.close()


def main(argv: list[str]) -> int:
    """Ponto de entrada via linha de comando.

    Exemplos:
        python wake_up.py "https://seuapp.streamlit.app"
        python wake_up.py "https://seuapp.streamlit.app" "Wake Up"

    :param argv: Lista de argumentos do terminal (sys.argv).
    :return: 0 se executar com sucesso; 1 se falhar; 2 se uso inválido.
    """

    if len(argv) < 2:
        print("Uso: python wake_up.py <url_do_app> [texto_do_botao]")
        return 2

    url = argv[1].strip()
    button_text = argv[2].strip() if len(argv) >= 3 else "Wake Up"

    cfg = WakeUpConfig(
        url=url,
        button_text="Wake Up",  # não será usado
        button_selector="button[data-testid='wakeup-button-viewer']",
        wait_seconds_after_click=10,
        headless=True,
    )

    return wake_up_via_browser(cfg)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))