from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import string
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def python_candidates() -> list[str]:
    values = [sys.executable]
    for name in ("python", "python3", "py"):
        found = shutil.which(name)
        if found:
            values.append(found)
    known = [
        PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    values.extend(str(path) for path in known if path.is_file())
    return list(dict.fromkeys(str(Path(value).resolve()) for value in values))


def isaac_locations() -> list[dict[str, Any]]:
    candidates: list[Path] = []
    configured = os.environ.get("ISAAC_SIM_PATH")
    if configured:
        candidates.append(Path(configured))
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    candidates.extend(
        [
            local_app_data / "ov" / "pkg",
            local_app_data / "NVIDIA Corporation" / "Isaac Sim",
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "NVIDIA Corporation"
            / "Isaac Sim",
        ]
    )
    candidates.extend(
        Path(f"{letter}:/isaac-sim")
        for letter in string.ascii_uppercase
        if Path(f"{letter}:/").exists()
    )
    results: list[dict[str, Any]] = []
    for candidate in dict.fromkeys(candidates):
        launchers: list[str] = []
        if candidate.exists():
            patterns = ("isaac-sim*.bat", "isaac-sim*.exe", "python.bat", "python.exe")
            for pattern in patterns:
                launchers.extend(str(path) for path in candidate.glob(pattern))
                launchers.extend(str(path) for path in candidate.glob(f"*/{pattern}"))
        results.append(
            {
                "path": str(candidate),
                "exists": candidate.exists(),
                "launchers": sorted(set(launchers)),
            }
        )
    return results


def module_status() -> dict[str, bool]:
    names = (
        "isaacsim",
        "omni",
        "pxr",
        "openpi_client",
        "websockets",
        "msgpack",
        "numpy",
        "yaml",
        "cv2",
    )
    return {name: importlib.util.find_spec(name) is not None for name in names}


def disk_status() -> list[dict[str, Any]]:
    roots = {Path(PROJECT_ROOT.anchor), Path(Path(sys.executable).anchor)}
    roots.update(
        Path(f"{letter}:/")
        for letter in string.ascii_uppercase
        if Path(f"{letter}:/").exists()
    )
    results = []
    for root in sorted(roots, key=str):
        try:
            usage = shutil.disk_usage(root)
        except OSError as exc:
            results.append({"path": str(root), "error": str(exc)})
            continue
        results.append(
            {
                "path": str(root),
                "total_gb": round(usage.total / 1024**3, 2),
                "free_gb": round(usage.free / 1024**3, 2),
            }
        )
    return results


def asset_status() -> dict[str, Any]:
    xarm_root = PROJECT_ROOT / "third_party" / "xarm_ros2" / "xarm_description"
    required = {
        "device_xacro": xarm_root / "urdf" / "xarm_device.urdf.xacro",
        "xarm6_xacro": xarm_root / "urdf" / "xarm6" / "xarm6.urdf.xacro",
        "gripper_xacro": xarm_root / "urdf" / "gripper" / "xarm_gripper.urdf.xacro",
        "kinematics": xarm_root
        / "config"
        / "kinematics"
        / "default"
        / "xarm6_default_kinematics.yaml",
        "arm_meshes": xarm_root / "meshes" / "xarm6",
        "gripper_meshes": xarm_root / "meshes" / "gripper" / "xarm",
    }
    return {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in required.items()
    }


def gpu_status() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "error": "nvidia-smi was not found on PATH"}
    result = command_output(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.get("available"):
        rows = []
        for line in result.get("stdout", "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 4:
                rows.append(
                    {
                        "name": parts[0],
                        "driver_version": parts[1],
                        "vram_total_mib": int(parts[2]),
                        "vram_free_mib": int(parts[3]),
                    }
                )
        result["gpus"] = rows
    result["executable"] = executable
    return result


def collect_report() -> dict[str, Any]:
    modules = module_status()
    locations = isaac_locations()
    git_lfs = command_output(["git", "lfs", "version"])
    ros2_executable = shutil.which("ros2")
    report = {
        "schema_version": "1.0",
        "read_only_diagnostic": True,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "active_executable": sys.executable,
            "version": sys.version,
            "candidates": python_candidates(),
        },
        "gpu": gpu_status(),
        "modules": modules,
        "isaac_locations": locations,
        "isaac_launcher_found": any(item["launchers"] for item in locations),
        "disk": disk_status(),
        "git_lfs": git_lfs,
        "ros2": {
            "installed": ros2_executable is not None,
            "executable": ros2_executable,
            "required": False,
        },
        "xarm_assets": asset_status(),
        "policy_server": {
            "host": os.environ.get("OPENPI_POLICY_HOST", "127.0.0.1"),
            "port": int(os.environ.get("OPENPI_POLICY_PORT", "18000")),
            "configured_by_environment": bool(
                os.environ.get("OPENPI_POLICY_HOST")
                and os.environ.get("OPENPI_POLICY_PORT")
            ),
        },
    }
    report["isaac_ready"] = bool(
        modules["isaacsim"] or (modules["omni"] and modules["pxr"])
    )
    report["openpi_client_ready"] = all(
        modules[name] for name in ("openpi_client", "websockets", "msgpack", "numpy")
    )
    report["assets_ready"] = all(item["exists"] for item in report["xarm_assets"].values())
    return report


def print_report(report: dict[str, Any]) -> None:
    print("Isaac Sim readiness diagnostic (read-only)")
    print("==========================================")
    print(f"Windows/platform: {report['platform']['platform']}")
    print(f"CPU architecture: {report['platform']['machine']}")
    print(f"Active Python: {report['python']['active_executable']}")
    print(f"Python version: {report['python']['version'].splitlines()[0]}")
    print("Python candidates:")
    for path in report["python"]["candidates"]:
        print(f"  - {path}")
    gpu = report["gpu"]
    if gpu.get("available"):
        for item in gpu.get("gpus", []):
            print(
                "GPU: "
                f"{item['name']} | driver {item['driver_version']} | "
                f"{item['vram_total_mib']} MiB total | {item['vram_free_mib']} MiB free"
            )
    else:
        print(f"GPU diagnostic unavailable: {gpu.get('error') or gpu.get('stderr')}")
    print("Modules:")
    for name, available in report["modules"].items():
        print(f"  {'PASS' if available else 'MISS'} {name}")
    print("Isaac locations:")
    for item in report["isaac_locations"]:
        print(f"  {'FOUND' if item['exists'] else 'MISS '} {item['path']}")
        for launcher in item["launchers"]:
            print(f"        launcher: {launcher}")
    print("Disk:")
    for item in report["disk"]:
        if "error" in item:
            print(f"  {item['path']}: unavailable ({item['error']})")
        else:
            print(
                f"  {item['path']}: {item['free_gb']} GiB free / "
                f"{item['total_gb']} GiB total"
            )
    print(
        f"Git LFS: {'PASS' if report['git_lfs'].get('available') else 'MISS'} | "
        f"ROS 2: {'present (optional)' if report['ros2']['installed'] else 'not installed (optional)'}"
    )
    print(f"xArm source assets: {'PASS' if report['assets_ready'] else 'MISS'}")
    print(f"OpenPI client dependencies: {'PASS' if report['openpi_client_ready'] else 'MISS'}")
    policy = report["policy_server"]
    print(
        f"Policy server: {policy['host']}:{policy['port']} "
        f"({'environment' if policy['configured_by_environment'] else 'repository default'})"
    )
    if report["isaac_ready"]:
        print("RESULT: Isaac modules are visible to this interpreter.")
        print("Next: run sim_isaac/scripts/inspect_xarm_asset.py with the same launcher.")
    else:
        print("RESULT: Isaac Sim is not available to this interpreter.")
        print("Next steps:")
        print("  1. Install a Windows-compatible Isaac Sim release from NVIDIA.")
        print("  2. Set ISAAC_SIM_PATH to its installation directory if not auto-detected.")
        print("  3. Re-run this script through the installation's Python launcher.")
        print("  4. Do not install Isaac into the MuJoCo environment blindly; use its supported launcher.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Isaac Sim installation diagnostic")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--output", type=Path, help="Optionally save the JSON report")
    args = parser.parse_args()
    report = collect_report()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0 if report["isaac_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
