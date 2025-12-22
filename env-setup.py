import os
import platform
import subprocess
import sys

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
    
    if os_name == "Windows":
        run_command("python -m venv venv")
        run_command(".\\venv\\Scripts\\pip.exe install -r requirements.txt")
        
    elif os_name == "Linux" or os_name == "Darwin":  # Adicionado Darwin para macOS
        # Primeiro verifica se o Python está disponível
        python_cmd = "python3" if sys.version_info.major == 3 else "python"
        
        # Cria o ambiente virtual
        run_command(f"{python_cmd} -m venv venv")
        
        # Instala dependências
        pip_path = "./venv/bin/pip"
        if os.path.exists("requirements.txt"):
            run_command(f"{pip_path} install -r requirements.txt")
        else:
            print("requirements.txt not found. Skipping package installation.")
        
        # Mostra instruções para ativar o ambiente
        print("\n" + "="*50)
        print("Virtual environment created successfully!")
        print("To activate the virtual environment, run:")
        print("  source venv/bin/activate")
        print("="*50)
    
    else:
        print(f"OS not supported: {os_name}")
        sys.exit(1)
    
    print("\nSetup completed!")

if __name__ == "__main__":
    main()