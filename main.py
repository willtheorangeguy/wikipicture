#!/usr/bin/env python3
"""Standalone entry point — run directly without installing the package.

Usage:
    python main.py scan /path/to/photos --output report.html
    python main.py scan /path/to/photos  # opens report automatically
    python main.py clear-cache
"""

import sys
from pathlib import Path

# Add src/ to the path so the wikipicture package is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))

from wikipicture.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
