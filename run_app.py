import subprocess
import os
import sys
import time
import webbrowser
import signal

def run_app():
    # Get the project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the virtual environment's python and uvicorn
    venv_python = os.path.join(root_dir, "venv", "Scripts", "python.exe")
    venv_uvicorn = os.path.join(root_dir, "venv", "Scripts", "uvicorn.exe")
    
    if not os.path.exists(venv_python):
        print(f"Error: Virtual environment not found at {venv_python}")
        print("Please ensure you have a 'venv' folder in the project root.")
        return

    print("--- Starting Multilingual Sign-to-Speech System ---")

    # 1. Start Backend (FastAPI)
    print("Starting Backend (FastAPI)...")
    backend_process = subprocess.Popen(
        [venv_uvicorn, "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=root_dir,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    # 2. Start Frontend (Vite/React)
    print("Starting Frontend (React)...")
    frontend_process = subprocess.Popen(
        ["npm.cmd", "run", "dev"],
        cwd=os.path.join(root_dir, "frontend"),
        shell=True,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    print("\n------------------------------------------------")
    print("Application is starting!")
    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("------------------------------------------------")
    print("\nPress Ctrl+C in this window to stop everything (if possible) or close the individual windows.")

    # Wait a bit for servers to warm up then open browser
    time.sleep(5)
    webbrowser.open("http://localhost:5173")

    try:
        # Keep the main script alive so the sub-processes stay running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        # On Windows, terminating the parent doesn't always kill the children in new consoles
        # but this is the best we can do from a single script.
        backend_process.terminate()
        frontend_process.terminate()

if __name__ == "__main__":
    run_app()
