"""Console output utilities for consistent formatting."""

from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

# Color scheme
COLORS = {
    "primary": "#81A1C1",
    "secondary": "#88C0D0",
    "success": "#A3BE8C",
    "warning": "#EBCB8B",
    "error": "#BF616A",
    "info": "#8FBCBB",
    "muted": "#616E88",
    "text": "#D8DEE9",
}


def create_progress() -> Progress:
    """Create a standardized progress bar."""
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style=COLORS["secondary"], finished_style=COLORS["success"]),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )


def format_score(score: float) -> str:
    """Format a similarity score with appropriate color."""
    if score >= 0.9:
        color = COLORS["success"]
    elif score >= 0.8:
        color = COLORS["warning"]
    else:
        color = COLORS["error"]
    return f"[bold {color}]{score:.2f}[/]"


def print_header(console: Console, text: str) -> None:
    """Print a formatted header."""
    console.print(f"\n[bold {COLORS['primary']}]{text}[/]")


def print_success(console: Console, text: str) -> None:
    """Print a success message."""
    console.print(f"[bold {COLORS['success']}]{text}[/]")


def print_error(console: Console, text: str) -> None:
    """Print an error message."""
    console.print(f"[bold {COLORS['error']}]{text}[/]")


def print_info(console: Console, text: str) -> None:
    """Print an info message."""
    console.print(f"[{COLORS['info']}]{text}[/]")


def print_separator(console: Console) -> None:
    """Print a separator line."""
    console.print(f"[{COLORS['muted']}]{'─' * 80}[/]")
