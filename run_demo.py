#!/usr/bin/env python3
"""
HUMOS-v2 Demo Runner -- one command to run the full action pipeline.

Usage:
    python3 run_demo.py <video>
    python3 run_demo.py <video> --vlm heuristic
    python3 run_demo.py <video> --epic-csv data/epic_kitchens/EPIC_100_train.csv --video-id P01_01
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   ██╗  ██╗ ██╗   ██╗ ███╗   ███╗  ██████╗  ███████╗     ║
║   ██║  ██║ ██║   ██║ ████╗ ████║ ██╔═══██╗ ██╔════╝     ║
║   ███████║ ██║   ██║ ██╔████╔██║ ██║   ██║ ███████╗     ║
║   ██╔══██║ ██║   ██║ ██║╚██╔╝██║ ██║   ██║ ╚════██║     ║
║   ██║  ██║ ╚██████╔╝ ██║ ╚═╝ ██║ ╚██████╔╝ ███████║     ║
║   ╚═╝  ╚═╝  ╚═════╝  ╚═╝     ╚═╝  ╚═════╝  ╚══════╝    ║
║                          v 2                             ║
║                                                          ║
║   Action-Centric Ground Truth for Humanoid Navigation    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    banner()
    from main import main
    main()
