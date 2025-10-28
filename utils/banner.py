"""
ASCII banner and branding for SpatialHero.
"""

# ANSI Shadow style ASCII art for SPATIALHERO
BANNER = """
███████╗██████╗   █████╗ ████████╗██╗ █████╗ ██╗     ██╗  ██╗███████╗██████╗  ██████╗
██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔══██╗██║     ██║  ██║██╔════╝██╔══██╗██╔═══██╗
███████╗██████╔╝███████║   ██║   ██║███████║██║     ███████║█████╗  ██████╔╝██║   ██║
╚════██║██╔═══╝ ██╔══██║   ██║   ██║██╔══██║██║     ██╔══██║██╔══╝  ██╔══██╗██║   ██║
███████║██║     ██║  ██║   ██║   ██║██║  ██║███████╗██║  ██║███████╗██║  ██║╚██████╔╝
╚══════╝╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝
"""

TAGLINE = "Making Instruction-Tuned LLMs Spatially Aware"

VERSION = "v0.1.0"

# Simple ASCII banner (Windows-compatible, no Unicode box-drawing chars)
SIMPLE_BANNER = """
 ____  ____   _  _____ ___    _    _     _   _ _____ ____   ___
/ ___||  _ \\ / \\|_   _|_ _|  / \\  | |   | | | | ____|  _ \\ / _ \\
\\___ \\| |_) / _ \\ | |  | |  / _ \\ | |   | |_| |  _| | |_) | | | |
 ___) |  __/ ___ \\| |  | | / ___ \\| |___| |_| | |___|  _ <| |_| |
|____/|_|  /_/   \\_\\_| |___/_/   \\_\\_____\\___/|_____|_| \\_\\\\___/
"""

# Compact banner for smaller displays
COMPACT_BANNER = """
███████╗██████╗  █████╗ ████████╗██╗ █████╗ ██╗
██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██║██╔══██╗██║
███████╗██████╔╝███████║   ██║   ██║███████║██║
╚════██║██╔═══╝ ██╔══██║   ██║   ██║██╔══██║██║
███████║██║     ██║  ██║   ██║   ██║██║  ██║███████╗
╚══════╝╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝╚═╝  ╚═╝╚══════╝

██╗  ██╗███████╗██████╗  ██████╗
██║  ██║██╔════╝██╔══██╗██╔═══██╗
███████║█████╗  ██████╔╝██║   ██║
██╔══██║██╔══╝  ██╔══██╗██║   ██║
██║  ██║███████╗██║  ██║╚██████╔╝
╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝
"""


def print_banner(compact=False, show_version=True, simple=None):
    """
    Print the SpatialHero ASCII banner.

    Args:
        compact: Use compact banner for smaller terminals
        show_version: Show version number
        simple: Force simple ASCII (auto-detects if None)
    """
    # Auto-detect if we should use simple banner
    if simple is None:
        import sys
        # Use simple banner on Windows by default to avoid encoding issues
        simple = sys.platform == 'win32'

    if simple:
        banner = SIMPLE_BANNER
    else:
        banner = COMPACT_BANNER if compact else BANNER

    try:
        print(banner)
    except UnicodeEncodeError:
        # Fallback to simple if Unicode fails
        print(SIMPLE_BANNER)

    print(f"  {TAGLINE}")
    if show_version:
        print(f"  {VERSION}")
    print()


def print_header(title, width=60):
    """
    Print a formatted header section.

    Args:
        title: Header title
        width: Width of the header bar
    """
    print()
    print("=" * width)
    print(title.center(width))
    print("=" * width)
    print()


def print_section(title, width=60):
    """
    Print a section divider.

    Args:
        title: Section title
        width: Width of the divider
    """
    print()
    print("-" * width)
    print(title)
    print("-" * width)


def print_result(label, value, status=None, indent=2):
    """
    Print a formatted result line.

    Args:
        label: Result label
        value: Result value
        status: Optional status (PASS, FAIL, INFO, etc.)
        indent: Number of spaces to indent
    """
    prefix = " " * indent
    if status:
        status_str = f"[{status}]"
        print(f"{prefix}{status_str:8} {label}: {value}")
    else:
        print(f"{prefix}{label}: {value}")


def print_box(text, width=60, char="="):
    """
    Print text in a box.

    Args:
        text: Text to display
        width: Box width
        char: Box character
    """
    print()
    print(char * width)
    for line in text.split('\n'):
        padding = width - len(line) - 4
        print(f"  {line}{' ' * padding}  ")
    print(char * width)
    print()


# Color codes (optional - can be disabled)
class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def disable():
        """Disable colors (for Windows or no-color environments)."""
        Colors.HEADER = ''
        Colors.BLUE = ''
        Colors.CYAN = ''
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.ENDC = ''
        Colors.BOLD = ''
        Colors.UNDERLINE = ''


def colored(text, color):
    """
    Return colored text.

    Args:
        text: Text to color
        color: Color code from Colors class

    Returns:
        Colored text string
    """
    return f"{color}{text}{Colors.ENDC}"
