"""Reproduce the local CPU/package matrix without models, services or hardware.

Requires uv on PATH and Python 3.11+ to run this script. uv can provision the
Python 3.11/3.12 environments under this checkout's .venv directory.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("py311-np126", "3.11", "1.26.4"),
    ("py311-np200", "3.11", "2.0.2"),
    ("py311-np246", "3.11", "2.4.6"),
    ("py312-np246", "3.12", "2.4.6"),
)


def environment_python(directory):
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def plan(output):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    wheel = output / "dist" / f"chunktrust-{version}-py3-none-any.whl"
    builder = ROOT / ".venv/build"
    build_python = str(environment_python(builder))

    def step(label, command, *, cwd=ROOT, environment=None, temporary_cwd=False):
        entry = {"label": label, "command": [str(value) for value in command], "cwd": str(cwd)}
        if environment is not None:
            entry["create_environment"] = str(environment)
        if temporary_cwd:
            entry.update(cwd=None, temporary_cwd=True)
        return entry

    build = [
        step("create-build-environment", ["uv", "venv", "--python", "3.11", builder], environment=builder),
        step("install-build-tools", ["uv", "pip", "install", "--python", build_python,
                                     "build==1.3.0", "setuptools==80.9.0", "wheel==0.45.1"]),
        step("build-distributions", [build_python, "-m", "build", "--no-isolation", "--outdir", output / "dist"]),
        step("check-distributions", [build_python, "scripts/check_distribution.py", output / "dist"]),
    ]
    cases = []
    for name, python_version, numpy in CASES:
        environment = ROOT / ".venv/matrix" / name
        python = str(environment_python(environment))
        major, minor = map(int, python_version.split("."))
        probe = (
            "import json,sys,numpy,chunktrust;"
            f"assert sys.version_info[:2]==({major},{minor});"
            f"assert numpy.__version__=={numpy!r};"
            "print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,"
            "'package':chunktrust.__file__}))"
        )
        steps = [
            step("create-environment", ["uv", "venv", "--python", python_version, environment], environment=environment),
            step("install-wheel", ["uv", "pip", "install", "--python", python,
                                   "--reinstall-package", "chunktrust", wheel,
                                   "pytest==8.3.5", f"numpy=={numpy}"]),
            step("versions", [python, "-c", probe]),
            step("tests", [python, "-m", "pytest", "-q", "-p", "no:cacheprovider"]),
            step("quickstart", [python, "examples/quickstart.py"]),
            step("recorded-decisions", [python, "examples/replay_recorded_decisions.py"]),
            step("synthetic-bridge", [python, "examples/real_robot_core_smoke.py"]),
            step("result-summary", [python, "scripts/summarize_results.py", "--output", output / f"{name}-summary.json"]),
            step("release-check", [python, "scripts/check_release.py"]),
            step("installed-wheel", [python, ROOT / "scripts/check_installed_distribution.py"], temporary_cwd=True),
            step("dependencies", ["uv", "pip", "check", "--python", python]),
        ]
        cases.append({"name": name, "python_requested": python_version,
                      "numpy_requested": numpy, "steps": steps})
    return {"build": build, "cases": cases}


def execute(steps, output, group):
    records = []
    for index, entry in enumerate(steps):
        record = dict(entry)
        environment = entry.get("create_environment")
        if environment and environment_python(Path(environment)).is_file():
            record.update(returncode=0, skipped="reuse existing environment")
            records.append(record)
            continue
        temporary = tempfile.TemporaryDirectory(prefix="chunktrust-wheel-check-") if entry.get("temporary_cwd") else None
        cwd = Path(temporary.name if temporary else entry["cwd"])
        cwd.mkdir(parents=True, exist_ok=True)
        record["cwd"] = str(cwd)
        log = output / f"{group}-{index:02d}-{entry['label']}.log"
        started = time.monotonic()
        try:
            if temporary and (cwd == ROOT or ROOT in cwd.parents):
                raise OSError("The temporary wheel-check directory must be outside the checkout")
            result = subprocess.run(entry["command"], cwd=cwd, capture_output=True, text=True)
        except OSError as error:
            result = subprocess.CompletedProcess(entry["command"], 127, "", str(error) + "\n")
        finally:
            if temporary:
                temporary.cleanup()
        log.write_text(f"cwd: {cwd}\n$ {shlex.join(entry['command'])}\n" + result.stdout + result.stderr)
        record.update(returncode=result.returncode, elapsed_s=time.monotonic() - started, log=str(log))
        if entry["label"] == "versions" and result.returncode == 0:
            record["versions"] = json.loads(result.stdout)
        records.append(record)
        print(f"{group}: {entry['label']}: {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
        if result.returncode:
            print(f"See {log}", file=sys.stderr)
            break
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for command logs and results.json")
    parser.add_argument("--dry-run", action="store_true", help="Write plan.json without running commands or creating environments")
    args = parser.parse_args()
    if not args.dry_run and shutil.which("uv") is None:
        parser.error("uv must be installed and available on PATH")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands = plan(output)
    (output / "plan.json").write_text(json.dumps(commands, indent=2) + "\n")
    if args.dry_run:
        print(f"Planned {len(commands['cases'])} CPU cases; see {output / 'plan.json'}")
        return
    results = {"scope": "CPU/package checks only; no model, services or hardware",
               "build": execute(commands["build"], output, "build"), "cases": []}
    build_ok = all(record["returncode"] == 0 for record in results["build"])
    if build_ok:
        for case in commands["cases"]:
            case = dict(case)
            case["results"] = execute(case.pop("steps"), output, case["name"])
            case["success"] = all(record["returncode"] == 0 for record in case["results"])
            results["cases"].append(case)
    results["success"] = build_ok and all(case["success"] for case in results["cases"])
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if not results["success"]:
        raise SystemExit(1)
    print(f"PASS: four CPU/package cases; see {output / 'results.json'}")


if __name__ == "__main__":
    main()
