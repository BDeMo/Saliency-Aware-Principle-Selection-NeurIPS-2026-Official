import os
import sys
import yaml
import shlex
import signal
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _env_with_overrides(base_env: Dict[str, str], overrides: Dict[str, Any]) -> Dict[str, str]:
    env = dict(base_env)
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        env[str(k)] = str(v)
    return env


def _resolve_port_ref(ports: Dict[str, Any], port_ref: str) -> int:
    if port_ref not in ports:
        raise KeyError(f"port_ref not found in ports map: {port_ref}")
    return int(ports[port_ref])


def _spawn(name: str, cmd: List[str], env: Dict[str, str]) -> Proc:
    print(f"\n[launch] {name}")
    print("  cmd:", " ".join(shlex.quote(x) for x in cmd))
    p = subprocess.Popen(cmd, env=env)
    return Proc(name=name, popen=p)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to cfg/launcher.yaml")
    ap.add_argument("--profile", required=True, help="Profile name")
    ap.add_argument("--dry", action="store_true", help="Print commands only")
    args = ap.parse_args()

    cfg = _load_yaml(args.cfg)
    ports = cfg.get("ports", {}) or {}
    services = cfg.get("services", {}) or {}
    profiles = cfg.get("profiles", {}) or {}

    if args.profile not in profiles:
        raise KeyError(f"profile not found: {args.profile}")

    svc_names = profiles[args.profile].get("services", []) or []
    if not svc_names:
        raise ValueError(f"profile has empty services: {args.profile}")

    procs: List[Proc] = []
    base_env = dict(os.environ)

    def stop_all():
        for pr in reversed(procs):
            try:
                pr.popen.terminate()
            except Exception:
                pass
        for pr in reversed(procs):
            try:
                pr.popen.wait(timeout=10)
            except Exception:
                try:
                    pr.popen.kill()
                except Exception:
                    pass

    def handle_sig(sig, frame):
        print("\n[launcher] stopping...")
        stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    for sn in svc_names:
        if sn not in services:
            raise KeyError(f"service not found in cfg: {sn}")
        s = services[sn]
        if not s.get("enable", True):
            print(f"[skip] {sn} (enable=false)")
            continue

        kind = s.get("kind")
        env_overrides = s.get("env", {}) or {}
        env_overrides["PORTS_MAP_JSON"] = str(ports)

        if kind == "uvicorn":
            module = s["module"]
            host = s.get("host", "0.0.0.0")
            port_ref = s["port_ref"]
            port = _resolve_port_ref(ports, port_ref)
            workers = int(s.get("workers", 1))

            cmd = [
                "python", "-m", "uvicorn", module,
                "--host", host,
                "--port", str(port),
                "--workers", str(workers),
            ]
            env = _env_with_overrides(base_env, env_overrides)
            if args.dry:
                print("[dry]", sn, cmd)
            else:
                procs.append(_spawn(sn, cmd, env))

        elif kind == "vllm":
            model_path = s["model_path"]
            served_model_name = s["served_model_name"]
            port_ref = s["port_ref"]
            port = _resolve_port_ref(ports, port_ref)

            tp = int(s.get("tensor_parallel_size", 1))
            max_model_len = int(s.get("max_model_len", 8192))
            dtype = s.get("dtype", "auto")
            trust_remote_code = bool(s.get("trust_remote_code", True))
            cuda_visible_devices = s.get("cuda_visible_devices", None)

            cmd = [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--served-model-name", served_model_name,
                "--port", str(port),
                "--dtype", str(dtype),
                "--max_model_len", str(max_model_len),
                "--tensor-parallel-size", str(tp),
            ]
            if trust_remote_code:
                cmd.append("--trust-remote-code")

            env = _env_with_overrides(base_env, env_overrides)
            if cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

            if args.dry:
                print("[dry]", sn, cmd)
            else:
                procs.append(_spawn(sn, cmd, env))

        else:
            raise ValueError(f"unknown service kind: {kind} for {sn}")

    print("\n[launcher] running. Ctrl+C to stop.")
    for pr in procs:
        pr.popen.wait()


if __name__ == "__main__":
    main()
