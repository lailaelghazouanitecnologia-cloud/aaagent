"""Terminal UI helpers — dense monospace output matching ZARNETTI style."""

from __future__ import annotations

import sys
import shutil


# ── Colors (ANSI) ──

class C:
    """ANSI color codes matching the ZARNETTI palette."""
    RST = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    RED = "\033[38;5;203m"      # #e5484d
    GREEN = "\033[38;5;78m"     # #5ec46a
    YELLOW = "\033[38;5;178m"   # #b8a030
    BLUE = "\033[38;5;68m"      # #4a80dd
    CYAN = "\033[38;5;80m"      # #40c8c8
    PURPLE = "\033[38;5;140m"   # #9977dd
    ORANGE = "\033[38;5;173m"   # #e09050
    WHITE = "\033[38;5;252m"    # #c8c8d4
    GRAY = "\033[38;5;242m"     # #58586a
    DARK = "\033[38;5;238m"     # #28282f

    @staticmethod
    def strip(text: str) -> str:
        """Remove ANSI codes for length calculation."""
        import re
        return re.sub(r"\033\[[0-9;]*m", "", text)


def width() -> int:
    """Terminal width."""
    return shutil.get_terminal_size((80, 24)).columns


# ── Primitives ──

def logo():
    """Print Z86 logo header."""
    w = width()
    print(f"{C.RED}{C.BOLD}Z86.DEV{C.RST} {C.DARK}/{C.RST} {C.GRAY}HCLM-D{C.RST}")
    print(f"{C.DARK}{'─' * min(w, 70)}{C.RST}")


def ok(msg: str):
    print(f"  {C.GREEN}✓{C.RST} {C.GRAY}{msg}{C.RST}")


def warn(msg: str):
    print(f"  {C.YELLOW}⚠{C.RST} {C.YELLOW}{msg}{C.RST}")


def err(msg: str):
    print(f"  {C.RED}✗{C.RST} {C.RED}{msg}{C.RST}")


def info(msg: str):
    print(f"  {C.GRAY}{msg}{C.RST}")


def step(msg: str):
    """Print a step indicator."""
    print(f"\n{C.CYAN}●{C.RST} {C.WHITE}{msg}{C.RST}")


def header(title: str):
    """Section header."""
    w = min(width(), 70)
    print(f"\n{C.DARK}{'─' * w}{C.RST}")
    print(f"  {C.WHITE}{C.BOLD}{title}{C.RST}")
    print(f"{C.DARK}{'─' * w}{C.RST}")


def kv(key: str, value: str, key_width: int = 16):
    """Key-value pair, aligned."""
    print(f"  {C.GRAY}{key:<{key_width}}{C.RST} {C.WHITE}{value}{C.RST}")


def table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None):
    """Print a compact monospace table."""
    if not rows:
        info("(empty)")
        return

    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(C.strip(row[i])))
            col_widths.append(max_w + 2)

    # Header
    hdr = ""
    for h, w in zip(headers, col_widths):
        hdr += f"{h:<{w}}"
    print(f"  {C.DARK}{hdr}{C.RST}")

    # Rows
    for row in rows:
        line = "  "
        for cell, w in zip(row, col_widths):
            visible_len = len(C.strip(cell))
            padding = w - visible_len
            line += cell + " " * max(0, padding)
        print(line)


def bar(value: float, max_val: float = 1.0, width: int = 20, color: str = C.BLUE) -> str:
    """Inline progress bar."""
    pct = min(1.0, max(0.0, value / max_val)) if max_val > 0 else 0
    filled = int(pct * width)
    empty = width - filled
    return f"{color}{'█' * filled}{C.DARK}{'·' * empty}{C.RST}"


def delta_str(value: float, higher_is_better: bool = False) -> str:
    """Format a delta value with color."""
    if abs(value) < 0.0001:
        return f"{C.GRAY}—{C.RST}"
    sign = "+" if value > 0 else ""
    if higher_is_better:
        color = C.GREEN if value > 0 else C.RED
    else:
        color = C.GREEN if value < 0 else C.RED
    return f"{color}{sign}{value:.3f}{C.RST}"


def confirm(msg: str) -> bool:
    """Ask yes/no confirmation."""
    try:
        ans = input(f"  {C.YELLOW}?{C.RST} {msg} [y/N] ").strip().lower()
        return ans in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        print()
        return False


def abort(msg: str = "Aborted."):
    """Print abort message and exit."""
    print(f"\n  {C.RED}{msg}{C.RST}")
    sys.exit(1)
