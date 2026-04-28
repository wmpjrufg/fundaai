#!/usr/bin/env python3
"""Cross-platform virtualenv bootstrap.

Creates ``.venv/`` at the **repository root** (not next to this
script) and installs every dependency declared in the top-level
``requirements.txt``. The venv name (``.venv``) matches the one
documented in ``README.md`` and ``scripts/README.md``, the one
ignored by ``.gitignore`` and the one used by the CI of the
project.

Run from anywhere:

    python scripts/env_setup.py
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


VENV_DIR_NAME = ".venv"


def run_command(command, shell=True):
    print(f"Executing: {command}")
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    os_name = platform.system()
    print(f"Operating System detected: {os_name}")

    # Resolve the repository root from this script's location so the
    # command can be invoked from anywhere (`python scripts/env_setup.py`
    # or `cd scripts && python env_setup.py` produce the same result).
    repo_root = Path(__file__).resolve().parent.parent
    venv_path = repo_root / VENV_DIR_NAME
    requirements = repo_root / "requirements.txt"

    print(f"Repo root  : {repo_root}")
    print(f"Venv path  : {venv_path}")
    print(f"Requirement: {requirements}")

    python_cmd = sys.executable

    if os_name == "Windows":
        run_command(f'"{python_cmd}" -m venv "{venv_path}"')
        pip_path = venv_path / "Scripts" / "pip.exe"
        if requirements.exists():
            run_command(f'"{pip_path}" install -r "{requirements}"')
        else:
            print("requirements.txt not found. Skipping package installation.")
        activate_hint = f"  {venv_path}\\Scripts\\Activate.ps1"

    elif os_name in ("Linux", "Darwin"):
        run_command(f'"{python_cmd}" -m venv "{venv_path}"')
        pip_path = venv_path / "bin" / "pip"
        if requirements.exists():
            run_command(f'"{pip_path}" install -r "{requirements}"')
        else:
            print("requirements.txt not found. Skipping package installation.")
        activate_hint = f"  source {venv_path}/bin/activate"

    else:
        print(f"OS not supported: {os_name}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Virtual environment created successfully!")
    print("To activate the virtual environment, run:")
    print(activate_hint)
    print("=" * 50)
    print("\nSetup completed!")


if __name__ == "__main__":
    main()
