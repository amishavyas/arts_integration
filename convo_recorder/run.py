import argparse
import subprocess
import sys
import os
import webbrowser
import time
import signal
import psutil
import socket

from preflight_dialog import show_preflight_dialog

processes = []
FRONTEND_PORT = 3001
BACKEND_PORT = 5001

# murmur_file can legitimately be None (RA chose "No murmur" in the
# dialog) - this sentinel distinguishes "not provided at all" (the --test
# CLI path, which skips the dialog) from "explicitly disabled", so the
# backend can tell the difference between "use the default" and "disable it".
_UNSET = object()

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    # psutil renamed Process.connections() -> net_connections() (the old
    # name is deprecated and, as of psutil 6.x, no longer a valid attrs key
    # for process_iter/as_dict at all - only requesting 'pid'/'name' there
    # and calling net_connections() directly avoids depending on either).
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections():
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

def start_backend(test_mode=False, devdata=False, intervention_enabled=None, add_to_database=None,
                   murmur_file=_UNSET):
    print("Starting backend server...")
    if is_port_in_use(BACKEND_PORT):
        print(f"Port {BACKEND_PORT} is in use. Attempting to kill the process...")
        kill_process_on_port(BACKEND_PORT)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    backend_path = os.path.join(script_dir, 'backend')
    os.chdir(backend_path)

    env = os.environ.copy()
    if test_mode:
        env['CONVO_RECORDER_TEST_MODE'] = '1'
    if devdata:
        env['CONVO_RECORDER_DEVDATA'] = '1'
    if intervention_enabled is not None:
        env['CONVO_RECORDER_INTERVENTION_MODE'] = '1' if intervention_enabled else '0'
    if add_to_database is not None:
        env['CONVO_RECORDER_ADD_TO_DATABASE'] = '1' if add_to_database else '0'
    if murmur_file is not _UNSET:
        env['CONVO_RECORDER_MURMUR_FILE'] = murmur_file or ''  # '' means explicitly disabled

    if sys.platform == 'win32':
        proc = subprocess.Popen(['python', 'app.py'],
                              creationflags=subprocess.CREATE_NEW_CONSOLE,
                              env=env)
    else:
        proc = subprocess.Popen([sys.executable, 'app.py'], env=env)
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

    # react-scripts opens its own browser tab by default; we open one
    # explicitly below, so disable CRA's auto-open to avoid duplicates.
    env = os.environ.copy()
    env['BROWSER'] = 'none'

    if sys.platform == 'win32':
        proc = subprocess.Popen(['npm', 'start'],
                              creationflags=subprocess.CREATE_NEW_CONSOLE,
                              env=env)
    else:
        proc = subprocess.Popen(['npm', 'start'], env=env)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true',
                         help='Reuse a single data/test/ session folder instead of '
                              'creating a new numbered one each run.')
    args = parser.parse_args()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Store the absolute path of the original directory
    original_dir = os.path.abspath(os.getcwd())

    devdata = False
    intervention_enabled = None
    add_to_database = None
    murmur_file = _UNSET
    if not args.test:
        print("Opening preflight dialog...")
        choices = show_preflight_dialog()
        if choices is None:
            print("Preflight cancelled - not starting the experiment.")
            return
        devdata = choices["devdata"]
        intervention_enabled = choices["intervention_enabled"]
        add_to_database = choices["add_to_database"]
        murmur_file = choices["murmur_file"]
        print(f"Preflight choices: devdata={devdata}, intervention_enabled={intervention_enabled}, "
              f"add_to_database={add_to_database}, murmur_file={murmur_file}")

    try:
        backend_proc = start_backend(test_mode=args.test, devdata=devdata,
                                      intervention_enabled=intervention_enabled,
                                      add_to_database=add_to_database, murmur_file=murmur_file)
        print("Waiting for backend to start...")
        time.sleep(5)
        
        # Change back to original directory before starting frontend
        os.chdir(original_dir)
        frontend_proc = start_frontend()
        print("Waiting for frontend to start...")
        time.sleep(3)
        
        print("Opening in Safari...")
        url = f'http://localhost:{FRONTEND_PORT}'
        # Explicitly Safari, not whatever webbrowser.open() picks as the
        # system default (Chrome here) - a backend crash was reliably
        # reproducible with Chrome running and never happened with the
        # backend alone, consistent with Metal/GPU contention between
        # Chrome's renderer and MLX. Safari, being Apple's own browser, is
        # the best first bet for avoiding that. Falls back to the system
        # default if Safari can't be launched for some reason.
        try:
            subprocess.run(['open', '-a', 'Safari', url], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Could not open Safari specifically - falling back to the system default browser.")
            webbrowser.open(url)

        print("\nApplication is running!")
        print("Press Ctrl+C to stop the application...")

        # The backend can occasionally crash from native audio-stack issues
        # (Metal/CoreAudio contention with Chrome, observed during testing)
        # that aren't Python bugs we can just catch. For a live installation,
        # automatically restarting it beats leaving the show dark - it
        # reloads models in a few seconds and the frontend's existing
        # device_status polling naturally recovers once it's back up. Capped
        # so a truly broken setup still surfaces instead of crash-looping
        # forever.
        MAX_BACKEND_RESTARTS = 5
        RESTART_WINDOW_SECONDS = 300
        restart_times = []

        # Monitor child processes
        while True:
            if backend_proc.poll() is not None:
                now = time.time()
                restart_times[:] = [t for t in restart_times if now - t < RESTART_WINDOW_SECONDS]
                if len(restart_times) >= MAX_BACKEND_RESTARTS:
                    print(f"\nBackend crashed {MAX_BACKEND_RESTARTS} times in "
                          f"{RESTART_WINDOW_SECONDS}s - giving up on auto-restart. "
                          "Please contact the researcher.")
                    break
                restart_times.append(now)
                print(f"\nBackend stopped unexpectedly - restarting it automatically "
                      f"({len(restart_times)}/{MAX_BACKEND_RESTARTS} restarts in this session)...")
                backend_proc = start_backend(test_mode=args.test, devdata=devdata,
                                              intervention_enabled=intervention_enabled,
                                              add_to_database=add_to_database, murmur_file=murmur_file)
                time.sleep(5)  # let it reload models before the next poll
                continue
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