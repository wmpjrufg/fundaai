import os
import platform
import subprocess
import sys

def run_command(command):
    print(f"Executing: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    os_name = platform.system()
    print(f"Operating System detected: {os_name}")
    
    if os_name == "Windows":
        run_command("python -m venv venv")
        run_command(".\\venv\\Scripts\\pip.exe install -r requirements.txt")
        
    elif os_name == "Linux":
        run_command("python3 -m venv venv")
        run_command("source venv/bin/activate")
        run_command("pip install -r requirements.txt")
    
    else:
        print(f"OS not supported: {os_name}")
        sys.exit(1)
    
    print("Setup completed!")

if __name__ == "__main__":
    main()