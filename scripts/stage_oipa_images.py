"""Verify, or reconstruct from upstream files, the frozen OIPA-Net image tree."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import yaml


def contained(root, relative):
    if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
        raise ValueError(f"Invalid relative image path: {relative}")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes its directory: {relative}")
    return path


def archived_images(config):
    prefix = PurePosixPath(config["data"]["canonical_prefix"])
    seen = {}
    for protocol in config["protocols"].values():
        with (Path(protocol) / "valid.jsonl").open(encoding="utf-8") as rows:
            for line in rows:
                for image in json.loads(line)["images"]:
                    relative = PurePosixPath(image["path"]).relative_to(prefix)
                    fingerprint = image["sha256"]
                    if relative in seen:
                        if seen[relative] != fingerprint:
                            raise ValueError(f"Conflicting SHA-256 for {relative}")
                        continue
                    seen[relative] = fingerprint
                    yield relative, PurePosixPath(image.get("archive_member") or relative), fingerprint


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def open_source(archive, source_root, relative, member):
    if archive:
        canonical = relative.as_posix()
        return archive.open(canonical if canonical in archive.NameToInfo else member.as_posix())
    canonical = contained(source_root, relative)
    return (canonical if canonical.is_file() else contained(source_root, member)).open("rb")


def stage(args):
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(config["data"]["image_root"]).resolve()
    source_root = args.upstream_root.resolve() if args.upstream_root else None
    count = 0
    with (ZipFile(args.archive) if args.archive else nullcontext()) as archive:
        for relative, member, fingerprint in archived_images(config):
            target = contained(root, relative)
            if target.is_file() and file_hash(target) == fingerprint:
                count += 1
                continue
            if not (archive or source_root):
                raise ValueError(f"Missing or mismatched image: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = open_source(archive, source_root, relative, member)
            with source, NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                candidate = Path(temporary.name)
                try:
                    copyfileobj(source, temporary)
                    temporary.close()
                    if file_hash(candidate) != fingerprint:
                        raise ValueError(f"Upstream bytes differ: {member}")
                    candidate.replace(target)
                finally:
                    candidate.unlink(missing_ok=True)
            count += 1
    print(f"Verified {count:,} unique images in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="ZIP with canonical relative paths or archive_member paths")
    source.add_argument("--upstream-root", type=Path, help="Extracted tree with canonical or upstream paths")
    stage(parser.parse_args())
