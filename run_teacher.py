"""Create a temporary Veri teacher, distill, and always stop the deployment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


VERI = "/home/me/.local/bin/veri"
ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-type", default="L4-24GB")
    args = parser.parse_args()
    created = subprocess.run(
        [VERI, "deployments", "create", "--from-hf", "Qwen/Qwen3-1.7B",
         "--name", "phoenix-temporary-teacher", "--gpu-type", args.gpu_type,
         "--min-replicas", "0", "--max-replicas", "1",
         "--scale-to-zero-window", "300", "--no-cache", "--format", "json"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    deployment = json.loads(created.stdout)
    deployment_id = deployment["id"]
    (ROOT / "teacher_deployment_id.txt").write_text(deployment_id + "\n")
    print(f"Created temporary deployment {deployment_id}", flush=True)
    try:
        deadline = time.monotonic() + 900
        while True:
            status = subprocess.run(
                [VERI, "deployments", "get", deployment_id, "--format", "json"],
                cwd=ROOT, capture_output=True, text=True, check=True,
            )
            deployment = json.loads(status.stdout)
            if deployment["status"] == "serving":
                print("Teacher is serving; starting 500 requests", flush=True)
                break
            if deployment["status"] in ("failed", "stopped") or time.monotonic() > deadline:
                raise RuntimeError(
                    f"Teacher did not become ready: {deployment['status']} "
                    f"{deployment.get('failure_code')} {deployment.get('error')}"
                )
            time.sleep(10)
        subprocess.run([sys.executable, str(ROOT / "distill.py"), deployment_id],
                       cwd=ROOT, check=True)
    finally:
        for attempt in range(3):
            stopped = subprocess.run(
                [VERI, "deployments", "stop", deployment_id],
                cwd=ROOT, capture_output=True, text=True,
            )
            if stopped.returncode == 0:
                print(f"Stopped deployment {deployment_id}", flush=True)
                break
            print(f"Stop attempt {attempt + 1} failed: {stopped.stderr}", file=sys.stderr)
            time.sleep(3)
        else:
            raise RuntimeError(f"URGENT: stop Veri deployment {deployment_id} manually")


if __name__ == "__main__":
    main()
