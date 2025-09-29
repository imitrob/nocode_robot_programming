import os, signal
import psutil  # pip install psutil

def list_other_ipykernels():
    me = os.getpid()

    # collect our ancestor PIDs to avoid self-kill
    ancestors = set()
    try:
        p = psutil.Process(me)
        while True:
            ancestors.add(p.pid)
            if p.ppid() == 0 or p.ppid() == p.pid:
                break
            p = psutil.Process(p.ppid())
    except psutil.Error:
        pass

    victims = []
    for proc in psutil.process_iter(['pid','cmdline']):
        try:
            cmd = proc.info['cmdline'] or []
            if any('ipykernel_launcher' in part for part in cmd):
                if proc.pid not in ancestors:
                    victims.append((proc.pid, ' '.join(cmd)))
        except psutil.Error:
            continue
    return victims

def kill_other_ipykernels(force=False):
    victims = list_other_ipykernels()
    for pid, cmd in victims:
        try:
            os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
            print(f"Killed {pid}: {cmd}")
        except ProcessLookupError:
            pass
    if not victims:
        print("No other ipykernel_launcher processes found.")

# Preview what would be killed:
for pid, cmd in list_other_ipykernels():
    print(pid, cmd)