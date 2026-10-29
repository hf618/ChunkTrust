"""Check built wheel/sdist code and accompanying notices against the source tree."""
import argparse
from pathlib import Path
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = (
    "NOTICE.md", "upstreams.json", "source_manifest.json", "portability_changes.json",
    "robodriver_sources.json", "agilex_openpi_sources.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    wheels = sorted(args.directory.glob("*.whl"))
    sources = sorted(args.directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise SystemExit("expected exactly one wheel and one source distribution")
    notice_files = [ROOT / "third_party" / name for name in PROVENANCE]
    notice_files += sorted((ROOT / "third_party/licenses").glob("*"))
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        for source in notice_files:
            relative = source.relative_to(ROOT).as_posix()
            matches = [name for name in names if name.endswith(".data/data/share/chunktrust/" + relative)]
            if len(matches) != 1 or wheel.read(matches[0]) != source.read_bytes():
                raise SystemExit(f"wheel notice/provenance differs or is missing: {relative}")
        licenses = [name for name in names if ".dist-info/" in name and name.endswith("/LICENSE")]
        if not any(wheel.read(name) == (ROOT / "LICENSE").read_bytes() for name in licenses):
            raise SystemExit("wheel does not retain the root LICENSE")
        for source in (ROOT / "src/chunktrust").glob("*.py"):
            if wheel.read("chunktrust/" + source.name) != source.read_bytes():
                raise SystemExit(f"wheel source differs: {source.name}")
        if any("/overlays/" in name for name in names):
            raise SystemExit("lightweight wheel unexpectedly contains full backend overlays")
    with tarfile.open(sources[0], "r:gz") as archive:
        members = {member.name.split("/", 1)[1]: member for member in archive.getmembers()
                   if member.isfile() and "/" in member.name}
        for source in [ROOT / "LICENSE", *notice_files]:
            relative = source.relative_to(ROOT).as_posix()
            member = members.get(relative)
            if member is None or archive.extractfile(member).read() != source.read_bytes():
                raise SystemExit(f"sdist notice/provenance differs or is missing: {relative}")
    print(f"PASS: {len(notice_files)} notice/provenance files retained in wheel and sdist; "
          "wheel code matches source and excludes backend overlays")


if __name__ == "__main__":
    main()
