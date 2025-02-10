import subprocess
import sys
import os
import webbrowser
import time
import signal
import psutil
import socket

processes = []
FRONTEND_PORT = 3001
BACKEND_PORT = 5001

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        try:
            for conn in proc.connections():
                if conn.laddr.port == port:
                    print(f"Killing process {proc.pid} on port {port}")
                    kill_process_and_children(proc.pid)
                    time.sleep(1)  # Wait for the port to be released
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def kill_process_and_children(proc_pid):
    try:
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        process.kill()
    except psutil.NoSuchProcess:
        pass

def start_backend():
    print("Starting backend server...")
    if is_port_in_use(BACKEND_PORT):
        print(f"Port {BACKEND_PORT} is in use. Attempting to kill the process...")
        kill_process_on_port(BACKEND_PORT)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_path = os.path.join(script_dir, 'backend')
    os.chdir(backend_path)
    
    if sys.platform == 'win32':
        proc = subprocess.Popen(['python', 'app.py'], 
                              creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        proc = subprocess.Popen(['python3', 'app.py'])
    processes.append(proc)
    return proc

def start_frontend():
    print("Starting frontend...")
    if is_port_in_use(FRONTEND_PORT):
        print(f"Port {FRONTEND_PORT} is in use. Attempting to kill the process...")
        kill_process_on_port(FRONTEND_PORT)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(script_dir, 'frontend')
    os.chdir(frontend_path)
    
    if sys.platform == 'win32':
        proc = subprocess.Popen(['npm', 'start'], 
                              creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        proc = subprocess.Popen(['npm', 'start'])
    processes.append(proc)
    return proc

def cleanup():
    print("\nCleaning up processes...")
    for proc in processes:
        kill_process_and_children(proc.pid)

def signal_handler(signum, frame):
    cleanup()
    sys.exit(0)

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Store the absolute path of the original directory
    original_dir = os.path.abspath(os.getcwd())
    
    try:
        backend_proc = start_backend()
        print("Waiting for backend to start...")
        time.sleep(5)
        
        # Change back to original directory before starting frontend
        os.chdir(original_dir)
        frontend_proc = start_frontend()
        print("Waiting for frontend to start...")
        time.sleep(3)
        
        print("Opening in browser...")
        webbrowser.open(f'http://localhost:{FRONTEND_PORT}')
        
        print("\nApplication is running!")
        print("Press Ctrl+C to stop the application...")
        
        # Monitor child processes
        while True:
            if backend_proc.poll() is not None:
                print("\nBackend server stopped unexpectedly. Shutting down...")
                break
            if frontend_proc.poll() is not None:
                print("\nFrontend server stopped unexpectedly. Shutting down...")
                break
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()
        os.chdir(original_dir)

if __name__ == "__main__":
    main()