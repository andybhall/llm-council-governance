"""Experiment manifest generation for reproducibility."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_sha() -> Optional[str]:
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def is_git_dirty() -> bool:
    """Check if the git working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    import importlib.metadata

    packages = [
        "httpx",
        "datasets",
        "pandas",
        "scipy",
        "tenacity",
        "matplotlib",
    ]

    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"

    return versions


def create_manifest(
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    random_seeds: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Create an experiment manifest for reproducibility.

    Args:
        config: Experiment configuration dictionary containing:
            - models (council, chairman)
            - benchmarks
            - n_questions
            - n_replications
            - structures
            - temperature
            - etc.
        output_dir: Optional directory to save manifest.json
        random_seeds: Optional dictionary of random seeds used

    Returns:
        Manifest dictionary with all reproducibility information
    """
    import socket

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git": {
            "sha": get_git_sha(),
            "dirty": is_git_dirty(),
        },
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "hostname": socket.gethostname(),
        },
        "packages": get_package_versions(),
        "config": config,
        "random_seeds": random_seeds or {},
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manifest_file = output_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))

    return manifest


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """Load a manifest from file."""
    return json.loads(Path(manifest_path).read_text())


def compare_manifests(
    manifest1: Dict[str, Any],
    manifest2: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare two manifests and return differences.

    Args:
        manifest1: First manifest
        manifest2: Second manifest

    Returns:
        Dictionary with differences between the two manifests
    """
    differences = {}

    # Compare git SHA
    if manifest1.get("git", {}).get("sha") != manifest2.get("git", {}).get("sha"):
        differences["git_sha"] = {
            "manifest1": manifest1.get("git", {}).get("sha"),
            "manifest2": manifest2.get("git", {}).get("sha"),
        }

    # Compare package versions
    pkg1 = manifest1.get("packages", {})
    pkg2 = manifest2.get("packages", {})
    pkg_diffs = {}
    for pkg in set(pkg1.keys()) | set(pkg2.keys()):
        if pkg1.get(pkg) != pkg2.get(pkg):
            pkg_diffs[pkg] = {
                "manifest1": pkg1.get(pkg),
                "manifest2": pkg2.get(pkg),
            }
    if pkg_diffs:
        differences["packages"] = pkg_diffs

    # Compare config
    cfg1 = manifest1.get("config", {})
    cfg2 = manifest2.get("config", {})
    cfg_diffs = {}
    for key in set(cfg1.keys()) | set(cfg2.keys()):
        if cfg1.get(key) != cfg2.get(key):
            cfg_diffs[key] = {
                "manifest1": cfg1.get(key),
                "manifest2": cfg2.get(key),
            }
    if cfg_diffs:
        differences["config"] = cfg_diffs

    return differences
