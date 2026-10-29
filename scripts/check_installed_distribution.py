"""Check an installed wheel from a working directory outside the source checkout."""
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np

import chunktrust
from chunktrust import AHS, AHSConfig


def main():
    prefix = Path(sys.prefix).resolve()
    package = Path(chunktrust.__file__).resolve()
    if not package.is_relative_to(prefix):
        raise SystemExit(f"package was not imported from the test environment: {package}")
    notices = prefix / "share/chunktrust/third_party"
    required = (
        "NOTICE.md", "upstreams.json", "source_manifest.json", "portability_changes.json",
        "robodriver_sources.json", "agilex_openpi_sources.json",
        "licenses/openpi-LICENSE", "licenses/robotwin-LICENSE", "licenses/starvla-LICENSE",
        "licenses/gr00t_n15-LICENSE", "licenses/gr00t_n16-LICENSE",
        "licenses/xvla-LICENSE", "licenses/robodriver-LICENSE",
    )
    for relative in required:
        if not (notices / relative).is_file():
            raise SystemExit(f"installed notice/provenance missing: {relative}")
        if relative.endswith(".json"):
            json.loads((notices / relative).read_text())
    distribution = metadata.distribution("chunktrust")
    root_licenses = [entry for entry in distribution.files or ()
                     if ".dist-info/" in str(entry) and str(entry).endswith("/LICENSE")]
    if not root_licenses or not all(distribution.locate_file(entry).is_file() for entry in root_licenses):
        raise SystemExit("installed root LICENSE is missing")
    config = AHSConfig(candidates=(10, 20, 30, 40))
    k, info = AHS(config, seed=0).select(np.ones((10, 50, 14)), np.zeros((50, 14)))
    if not 1 <= k <= 50 or not np.isfinite(info["q_mix"]).all():
        raise SystemExit("installed AHS smoke failed")
    print(json.dumps({
        "status": "PASS", "python": sys.version.split()[0], "numpy": np.__version__,
        "version": distribution.version, "package": str(package),
        "notice_files": len(required), "selected_horizon": k,
    }, indent=2))


if __name__ == "__main__":
    main()
