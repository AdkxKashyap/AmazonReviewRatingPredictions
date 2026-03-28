from __future__ import annotations

import site
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    site_packages_dirs = [Path(path) for path in site.getsitepackages() if path.endswith("site-packages")]
    if not site_packages_dirs:
        raise RuntimeError("No site-packages directory found for the active interpreter.")

    hook_path = site_packages_dirs[0] / "zz_tf_cuda_preload.pth"
    hook_contents = f"{repo_root}\nimport tf_cuda_preload\n"
    hook_path.write_text(hook_contents, encoding="utf-8")

    print(f"Installed TensorFlow GPU preload hook at {hook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())