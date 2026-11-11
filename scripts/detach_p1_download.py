"""Stop only this project's old foreground downloader after tmux is confirmed."""
import json
import subprocess
from pathlib import Path
import psutil

root = Path(__file__).resolve().parents[1]
subprocess.run(["tmux", "has-session", "-t", "phdq3-p1"], check=True)
pane_pid = int(subprocess.check_output(["tmux", "display-message", "-p", "-t", "phdq3-p1", "#{pane_pid}"], text=True).strip())
stopped = []
for proc in psutil.process_iter():
    try:
        command = proc.cmdline()
        if len(command) != 2 or command[1] != "blt_hf_checks/download_originals.py":
            continue
        if Path(proc.cwd()).resolve() != root:
            continue
        if pane_pid in {p.pid for p in proc.parents()}:
            continue
        proc.terminate()
        proc.wait(timeout=20)
        stopped.append(proc.pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
print(json.dumps({"tmux_confirmed": True, "old_project_downloaders_stopped": stopped}))
