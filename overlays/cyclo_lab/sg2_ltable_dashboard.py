#!/usr/bin/env python3
"""SG2 L-table VR recording launcher dashboard (stdlib only)."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CYCLO = ROOT / "cyclo_lab"
VR_REPO = ROOT / "robotis_applications"
AI_REPO = ROOT / "ai_worker"
PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))
ROS_DOMAIN_ID = os.environ.get("ROS_DOMAIN_ID", "30")
RMW = os.environ.get("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
# Selectable tasks. Each maps a display label to its gym id, dataset file, and
# robot profile (gripper SG2 vs dexterous-hand SH5).
ROBOT_PROFILES: dict[str, dict[str, str]] = {
    "FFW_SG2": {
        "robot_type": "FFW_SG2",
        "vr_model": "sg2",
        "hand": "false",
        "urdf": (
            "/root/ros2_ws/install/ffw_description/share/ffw_description/urdf/"
            "ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
        ),
    },
    "FFW_SH5": {
        "robot_type": "FFW_SH5",
        "vr_model": "sh5",
        "hand": "true",
        "urdf": (
            "/root/ros2_ws/install/ffw_description/share/ffw_description/urdf/"
            "ffw_sh5_rev1_follower/ffw_sh5_follower.urdf"
        ),
    },
}

TASKS: dict[str, dict[str, str]] = {
    "L-Table Pick & Place (thin box)": {
        "id": "Cyclo-Real-Pick-Place-LTable-FFW-SG2-v0",
        "dataset": "ffw_sg2_l_table_raw.hdf5",
        "robot": "FFW_SG2",
    },
    "Box Stack (thick box)": {
        "id": "Cyclo-Real-Box-Stack-FFW-SG2-v0",
        "dataset": "ffw_sg2_box_stack_raw.hdf5",
        "robot": "FFW_SG2",
    },
    "Single Box Far (rear table)": {
        "id": "Cyclo-Real-Single-Box-Far-FFW-SG2-v0",
        "dataset": "ffw_sg2_single_box_far_raw.hdf5",
        "robot": "FFW_SG2",
    },
    "Single Box Far (thick box)": {
        "id": "Cyclo-Real-Single-Box-Far-Thick-FFW-SG2-v0",
        "dataset": "ffw_sg2_single_box_far_thick_raw.hdf5",
        "robot": "FFW_SG2",
    },
    "L-Table Pick & Place (thin, hands)": {
        "id": "Cyclo-Real-Pick-Place-LTable-FFW-SH5-v0",
        "dataset": "ffw_sh5_l_table_raw.hdf5",
        "robot": "FFW_SH5",
    },
    "Box Stack (thick, hands)": {
        "id": "Cyclo-Real-Box-Stack-FFW-SH5-v0",
        "dataset": "ffw_sh5_box_stack_raw.hdf5",
        "robot": "FFW_SH5",
    },
    "Single Box Far (rear, hands)": {
        "id": "Cyclo-Real-Single-Box-Far-FFW-SH5-v0",
        "dataset": "ffw_sh5_single_box_far_raw.hdf5",
        "robot": "FFW_SH5",
    },
    "Single Box Far (thick, hands)": {
        "id": "Cyclo-Real-Single-Box-Far-Thick-FFW-SH5-v0",
        "dataset": "ffw_sh5_single_box_far_thick_raw.hdf5",
        "robot": "FFW_SH5",
    },
}
TASK = os.environ.get("TASK", "Cyclo-Real-Pick-Place-LTable-FFW-SG2-v0")
# Resolve the initial selected label from TASK (fall back to first entry).
_selected_task = next(
    (label for label, t in TASKS.items() if t["id"] == TASK),
    next(iter(TASKS)),
)
ROBOT_TYPE = os.environ.get("ROBOT_TYPE", "FFW_SG2")
NUM_DEMOS = os.environ.get("NUM_DEMOS", "4")
CYCLO_C = "cyclo_lab"
VR_C = "robotis-applications"
AI_C = "ai_worker"
REPO_IN = "/workspace/cyclo_lab"
ISAAC_PY = f"{REPO_IN}/third_party/IsaacLab/_isaac_sim/python.sh"
ROS_SETUP = "source /opt/ros/jazzy/setup.bash && source /root/ros2_ws/install/setup.bash"
ENV_ROS = f"export ROS_DOMAIN_ID={ROS_DOMAIN_ID} && export RMW_IMPLEMENTATION={RMW}"

_lock = threading.Lock()
_logs: dict[str, list[str]] = {k: [] for k in ("cyclo", "vr", "ai", "recorder", "isaac")}
_status: dict[str, str] = {k: "stopped" for k in _logs}
_launch_ts: dict[str, float] = {}
STARTING_GRACE_S = 90.0


def _log(key: str, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _lock:
        _logs[key].append(line)
        if len(_logs[key]) > 400:
            _logs[key] = _logs[key][-400:]


def _set_status(key: str, status: str) -> None:
    with _lock:
        _status[key] = status


def _current_task() -> dict[str, str]:
    with _lock:
        label = _selected_task
    return {"label": label, **TASKS[label]}


def _set_task(label: str) -> bool:
    global _selected_task
    if label not in TASKS:
        return False
    with _lock:
        _selected_task = label
    _log("cyclo", f"task selected: {label} ({TASKS[label]['id']})")
    return True


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode, out.strip()
    except Exception as e:
        return 1, str(e)


def _docker_running(name: str) -> bool:
    code, out = _run(["docker", "inspect", "-f", "{{.State.Running}}", name])
    return code == 0 and out.strip() == "true"


def _start_container(script: Path) -> tuple[bool, str]:
    if not script.is_file():
        return False, f"missing {script}"
    code, out = _run(["bash", str(script), "start"], cwd=script.parent)
    return code == 0, out or ("ok" if code == 0 else "failed")


def _host_ip() -> str:
    code, out = _run(["hostname", "-I"])
    if code == 0 and out.strip():
        return out.strip().split()[0]
    return "127.0.0.1"


def _exec_detached(key: str, container: str, bash_cmd: str, log: str) -> tuple[bool, str]:
    # Run the command in the FOREGROUND of the detached exec (no trailing '&').
    # `docker exec -d` keeps the process alive as long as this bash stays in the
    # foreground; backgrounding it with nohup/& makes Docker reap the tree
    # immediately and kill the child.
    inner = f"exec > {log} 2>&1; {bash_cmd}"
    _log(key, f"start -> {log}")
    code, out = _run(["docker", "exec", "-d", container, "bash", "-lc", inner])
    if code == 0:
        _set_status(key, "starting")
        with _lock:
            _launch_ts[key] = time.time()
        return True, "started"
    _set_status(key, "failed")
    return False, out or "exec failed"


def _spawn(key: str, container: str, bash_cmd: str) -> tuple[bool, str]:
    return _exec_detached(key, container, bash_cmd, f"/tmp/sg2_ltable_{key}.log")


def launch_containers() -> dict:
    results = {}
    for name, script in [
        ("cyclo_lab", CYCLO / "docker" / "container.sh"),
        ("robotis_applications", VR_REPO / "docker" / "container.sh"),
        ("ai_worker", AI_REPO / "docker" / "container.sh"),
    ]:
        ok, msg = _start_container(script)
        results[name] = {"ok": ok, "msg": msg[-500:] if msg else ""}
        _log("cyclo", f"container {name}: {'ok' if ok else 'FAIL'}")
    return results


def _cleanup(container: str, patterns: list[str]) -> None:
    # Run pkill in a SEPARATE exec from the launch. If pkill and the launch
    # command share a shell, `pkill -f <name>` matches that shell's own command
    # line (which contains <name>) and kills it before the launch runs.
    # Bracket the first char so pkill can't match this cleanup shell either.
    parts = "; ".join(f'pkill -9 -f "[{p[0]}]{p[1:]}" 2>/dev/null' for p in patterns)
    _run(["docker", "exec", container, "bash", "-lc", f"{parts}; sleep 1; true"])


def _current_robot_profile() -> dict[str, str]:
    robot = _current_task().get("robot", "FFW_SG2")
    return ROBOT_PROFILES.get(robot, ROBOT_PROFILES["FFW_SG2"])


def launch_ai_stack() -> tuple[bool, str]:
    profile = _current_robot_profile()
    _cleanup(AI_C, ["ros2_control_node", "robot_state_publisher",
                    "ai_worker_controller", "vr_controller_node", "retargeting"])
    cmd = (
        f"{ENV_ROS} && {ROS_SETUP} && "
        f"U={profile['urdf']}; "
        "ros2 run robot_state_publisher robot_state_publisher "
        '--ros-args -p robot_description:="$(cat $U)" & '
        "sleep 4; "
        "ros2 launch cyclo_motion_controller_ros ai_worker_controller.launch.py "
        f"controller_type:=vr hand:={profile['hand']}"
    )
    return _spawn("ai", AI_C, cmd)


def launch_vr() -> tuple[bool, str]:
    profile = _current_robot_profile()
    _cleanup(VR_C, ["vr_publisher"])
    cmd = (
        f"{ENV_ROS} && {ROS_SETUP} && "
        f"ros2 launch robotis_vuer vr.launch.py model:={profile['vr_model']} enable_vr_image:=true"
    )
    return _spawn("vr", VR_C, cmd)


def launch_recorder() -> tuple[bool, str]:
    task = _current_task()
    profile = _current_robot_profile()
    ds = f"{REPO_IN}/datasets/{task['dataset']}"
    _log("recorder", f"task: {task['label']} -> {task['id']} ({profile['robot_type']})")
    cmd = (
        f"cd {REPO_IN} && export DISPLAY=:1 && {ENV_ROS} && "
        f"{ISAAC_PY} scripts/sim2real/imitation_learning/recorder/record_demos.py "
        f"--task={task['id']} --robot_type {profile['robot_type']} "
        f"--dataset_file {ds} --num_demos {NUM_DEMOS} --enable_cameras"
    )
    return _spawn("recorder", CYCLO_C, cmd)


def kill_isaac() -> tuple[bool, str]:
    # The recorder runs as: python.sh -> kit/python3 (record_demos.py) -> carb threads.
    # Match record_demos.py (hits both the python.sh wrapper and the kit python)
    # plus any leftover Isaac kit process. Bracket the first char so pkill can't
    # match the shell running this command.
    cmd = (
        'pkill -9 -f "[r]ecord_demos.py" 2>/dev/null; '
        'pkill -9 -f "[_]isaac_sim/kit" 2>/dev/null; '
        'pkill -9 -f "[i]saac-sim" 2>/dev/null; '
        "sleep 1; "
        'pgrep -f "[r]ecord_demos.py" >/dev/null && echo "still-running" || echo "killed"'
    )
    code, out = _run(["docker", "exec", CYCLO_C, "bash", "-lc", cmd])
    killed = "killed" in (out or "")
    _set_status("isaac", "stopped" if killed else "running")
    _set_status("recorder", "stopped" if killed else "running")
    _log("isaac", f"kill -> {out.strip() if out else '(no output)'}")
    return killed, out or "no output"


def launch_all() -> dict:
    out: dict = {"containers": launch_containers()}
    time.sleep(3)
    for fn, label in [(launch_vr, "vr"), (launch_ai_stack, "ai")]:
        ok, msg = fn()
        out[label] = {"ok": ok, "msg": msg}
        time.sleep(2)
    out["vuer_url"] = f"https://{_host_ip()}:8012"
    out["note"] = "Start recorder separately after Isaac is up."
    return out


def _proc_running(container: str, pattern: str) -> bool:
    code, out = _run(
        ["docker", "exec", container, "bash", "-lc", f"pgrep -f '{pattern}' >/dev/null && echo yes || echo no"]
    )
    return code == 0 and out.strip() == "yes"


def _reconcile(key: str, container_up: bool, container: str, pattern: str) -> None:
    """Make status reflect reality. 'starting' is preserved until the process
    actually appears or the container goes away, so the UI shows progress."""
    if not container_up:
        _set_status(key, "stopped")
        return
    if _proc_running(container, pattern):
        _set_status(key, "running")
        return
    with _lock:
        cur = _status.get(key)
        ts = _launch_ts.get(key, 0.0)
    if cur == "starting" and (time.time() - ts) < STARTING_GRACE_S:
        return
    _set_status(key, "stopped")


def snapshot() -> dict:
    # Bracket the first char of each pattern so `pgrep -f` does not match the
    # shell that is running pgrep (its own command line contains the pattern).
    containers = {n: _docker_running(n) for n in (CYCLO_C, VR_C, AI_C)}
    _reconcile("vr", containers.get(VR_C, False), VR_C, "[v]r_publisher")
    _reconcile("ai", containers.get(AI_C, False), AI_C, "[v]r_controller_node")
    _reconcile("recorder", containers.get(CYCLO_C, False), CYCLO_C, "[r]ecord_demos.py")
    _reconcile("isaac", containers.get(CYCLO_C, False), CYCLO_C, "[i]saac-sim|[k]it/kit")
    with _lock:
        st = dict(_status)
        logs_tail = {k: v[-30:] for k, v in _logs.items()}
    task = _current_task()
    return {
        "containers": containers,
        "status": st,
        "vuer_url": f"https://{_host_ip()}:8012",
        "vuer_ws": f"wss://{_host_ip()}:8012",
        "task": task["id"],
        "task_label": task["label"],
        "robot": task.get("robot", "FFW_SG2"),
        "tasks": list(TASKS.keys()),
        "logs": logs_tail,
    }


HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SG2 L-Table Launcher</title>
<style>
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;margin:0;background:#0f1117;color:#e8eaed}
header{padding:1rem 1.5rem;background:#1a1d27;border-bottom:1px solid #2a2f3d}
h1{margin:0;font-size:1.25rem}main{padding:1.5rem;max-width:1100px;margin:0 auto}
.row{display:flex;flex-wrap:wrap;gap:.75rem;margin-bottom:1rem}
button{padding:.6rem 1rem;border:0;border-radius:8px;cursor:pointer;font-weight:600}
.primary{background:#3b82f6;color:#fff}.danger{background:#ef4444;color:#fff}
.secondary{background:#374151;color:#fff}.ok{color:#4ade80}.bad{color:#f87171}
.card{background:#1a1d27;border:1px solid #2a2f3d;border-radius:10px;padding:1rem;margin-bottom:1rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:.5rem}
.tag{padding:.35rem .6rem;border-radius:6px;background:#252a36;font-size:.85rem}
select{background:#0a0c10;color:#e8eaed;border:1px solid #2a2f3d;border-radius:6px;padding:.3rem .5rem;margin-left:.5rem;font-size:.85rem}
pre{background:#0a0c10;padding:.75rem;border-radius:8px;overflow:auto;max-height:220px;font-size:.75rem}
a{color:#60a5fa}
</style></head><body>
<header><h1>FFW-SG2 L-Table VR + Recorder</h1></header>
<main>
<div class="row">
<label class="tag">Task:
<select id="task" onchange="setTask()"></select>
</label>
</div>
<div class="row">
<button class="primary" onclick="act('launch_all')">Launch All Servers</button>
<button class="secondary" onclick="act('launch_recorder')">Start Recorder (Isaac)</button>
<button class="danger" onclick="act('kill_isaac')">Kill Isaac</button>
<button class="secondary" onclick="refresh()">Refresh</button>
</div>
<div class="card"><div id="meta"></div><div class="grid" id="status"></div></div>
<div class="card"><h3>Logs</h3><pre id="logs"></pre></div>
<p>VR: accept cert at Vuer URL. SG2: squeeze both grips to teleop. SH5: use hand gesture to toggle (see <a href="https://docs.robotis.com/docs/systems/aiworker/quick_start_guide/operation_guide/vr_teleoperation" target="_blank">ROBOTIS VR docs</a>). B=record, L=face target table, N=save, R=reset.</p>
</main>
<script>
async function act(a){await fetch('/api/'+a,{method:'POST'});refresh()}
async function setTask(){
 const label=document.getElementById('task').value;
 await fetch('/api/set_task',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task:label})});
 refresh();
}
async function refresh(){
 const d=await(await fetch('/api/status')).json();
 const sel=document.getElementById('task');
 if(sel.options.length!==d.tasks.length){
  sel.innerHTML='';
  d.tasks.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;sel.appendChild(o);});
 }
 if(document.activeElement!==sel) sel.value=d.task_label;
 let h='<p>Task: <b>'+d.task_label+'</b> <span class="tag">'+d.task+'</span><br>Robot: <b>'+d.robot+'</b><br>Vuer: <a href="'+d.vuer_url+'" target="_blank">'+d.vuer_url+'</a> ('+d.vuer_ws+')</p>';
 document.getElementById('meta').innerHTML=h;
 let s='';
 for(const[k,v]of Object.entries(d.containers))s+='<div class="tag">'+k+': <span class="'+(v?'ok':'bad')+'">'+(v?'up':'down')+'</span></div>';
 for(const[k,v]of Object.entries(d.status))s+='<div class="tag">'+k+': '+v+'</div>';
 document.getElementById('status').innerHTML=s;
 let lg='';for(const[k,lines]of Object.entries(d.logs)){lg+='=== '+k+' ===\n'+lines.join('\n')+'\n\n'}
 document.getElementById('logs').textContent=lg;
}
refresh();setInterval(refresh,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass

    def _json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/status":
            self._json(200, snapshot())
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/launch_all":
            threading.Thread(target=launch_all, daemon=True).start()
            self._json(200, {"ok": True, "msg": "launch started"})
            return
        if path == "/api/launch_recorder":
            ok, msg = launch_recorder()
            self._json(200, {"ok": ok, "msg": msg})
            return
        if path == "/api/kill_isaac":
            ok, msg = kill_isaac()
            self._json(200, {"ok": ok, "msg": msg})
            return
        if path == "/api/set_task":
            length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}
            ok = _set_task(body.get("task", ""))
            self._json(200 if ok else 400, {"ok": ok, "task": _current_task()})
            return
        self._json(404, {"error": "not found"})


def main() -> None:
    print(f"Dashboard http://0.0.0.0:{PORT}")
    print(f"Vuer https://{_host_ip()}:8012")
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
