#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Optional


# Nord color scheme
class Colors:
    """Nord color scheme for terminal output."""

    # Polar Night (dark to light grays)
    POLAR_NIGHT = {
        "nord0": "\033[38;2;46;52;64m",  # Dark gray base
        "nord1": "\033[38;2;59;66;82m",  # Darker gray
        "nord2": "\033[38;2;67;76;94m",  # Medium gray
        "nord3": "\033[38;2;76;86;106m",  # Light gray
    }

    # Snow Storm (light grays to white)
    SNOW_STORM = {
        "nord4": "\033[38;2;216;222;233m",  # Lightest gray
        "nord5": "\033[38;2;229;233;240m",  # White-ish
        "nord6": "\033[38;2;236;239;244m",  # Pure white
    }

    # Frost (blues and mint)
    FROST = {
        "nord7": "\033[38;2;143;188;187m",  # Mint
        "nord8": "\033[38;2;136;192;208m",  # Ice blue
        "nord9": "\033[38;2;129;161;193m",  # Blue
        "nord10": "\033[38;2;94;129;172m",  # Deep blue
    }

    # Aurora (accent colors)
    AURORA = {
        "nord11": "\033[38;2;191;97;106m",  # Red
        "nord12": "\033[38;2;208;135;112m",  # Orange
        "nord13": "\033[38;2;235;203;139m",  # Yellow
        "nord14": "\033[38;2;163;190;140m",  # Green
        "nord15": "\033[38;2;180;142;173m",  # Purple
    }

    # Styles
    STYLES = {
        "bold": "\033[1m",
        "dim": "\033[2m",
        "italic": "\033[3m",
        "underline": "\033[4m",
    }

    RESET = "\033[0m"

    @classmethod
    def style(cls, text: str, color: str, *styles: str) -> str:
        """Apply color and styles to text."""
        style_codes = "".join(cls.STYLES.get(s, "") for s in styles)
        return f"{color}{style_codes}{text}{cls.RESET}"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with Nord colors."""

    def __init__(self, fmt: str, datefmt: str):
        super().__init__(fmt, datefmt)

        # Level colors with styles
        self.level_colors = {
            logging.DEBUG: (Colors.FROST["nord8"], ["dim"]),  # Ice blue, dimmed
            logging.INFO: (Colors.FROST["nord9"], []),  # Blue, normal
            logging.WARNING: (Colors.AURORA["nord13"], ["bold"]),  # Yellow, bold
            logging.ERROR: (Colors.AURORA["nord11"], ["bold"]),  # Red, bold
            logging.CRITICAL: (
                Colors.AURORA["nord11"],
                ["bold", "underline"],
            ),  # Red, bold, underlined
        }

        # Component colors
        self.component_colors = {
            "timestamp": (Colors.POLAR_NIGHT["nord3"], ["dim"]),  # Light gray, dimmed
            "name": (Colors.FROST["nord7"], ["italic"]),  # Mint, italic
            "message": (Colors.SNOW_STORM["nord4"], []),  # Light gray, normal
            "path": (Colors.FROST["nord10"], ["dim"]),  # Deep blue, dimmed
            "function": (Colors.AURORA["nord15"], ["italic"]),  # Purple, italic
            "line": (Colors.AURORA["nord12"], []),  # Orange, normal
            "process": (Colors.AURORA["nord14"], ["dim"]),  # Green, dimmed
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with Nord colors."""
        # Store original values to restore later
        orig_msg = record.msg
        orig_levelname = record.levelname
        orig_name = record.name

        try:
            # Format timestamp
            time_color, time_styles = self.component_colors["timestamp"]
            record.asctime = Colors.style(
                self.formatTime(record), time_color, *time_styles
            )

            # Format level name
            level_color, level_styles = self.level_colors.get(
                record.levelno, (Colors.RESET, [])
            )
            record.levelname = Colors.style(
                f"{record.levelname:<8}", level_color, *level_styles
            )

            # Format logger name
            name_color, name_styles = self.component_colors["name"]
            record.name = Colors.style(f"{record.name:<20}", name_color, *name_styles)

            # Format message and location
            msg_color, msg_styles = self.component_colors["message"]
            message = record.getMessage()

            # Add file location if available
            if hasattr(record, "pathname") and record.pathname:
                try:
                    path = str(Path(record.pathname).relative_to(Path.cwd()))
                    location_parts = []

                    # Add file path
                    path_color, path_styles = self.component_colors["path"]
                    location_parts.append(Colors.style(path, path_color, *path_styles))

                    # Add function name if available
                    if record.funcName:
                        func_color, func_styles = self.component_colors["function"]
                        location_parts.append(
                            Colors.style(record.funcName, func_color, *func_styles)
                        )

                    # Add line number
                    line_color, line_styles = self.component_colors["line"]
                    location_parts.append(
                        Colors.style(str(record.lineno), line_color, *line_styles)
                    )

                    # Combine location parts
                    location = f" ({' → '.join(location_parts)})"
                    message = f"{message}{location}"
                except ValueError:
                    # If relative_to fails, use absolute path
                    pass

            # Set colored message
            record.msg = Colors.style(message, msg_color, *msg_styles)

            # Format the record
            return super().format(record)
        finally:
            # Restore original values
            record.msg = orig_msg
            record.levelname = orig_levelname
            record.name = orig_name


class IndexerLogger:
    """Logger class for the indexer."""

    def __init__(self, name: str) -> None:
        """Initialize logger.

        Args:
            name: Logger name (usually __name__)
        """
        self.name = name
        self._logger: Optional[logging.Logger] = None

    def get_logger(self) -> logging.Logger:
        """Get or create logger instance.

        Returns:
            Configured logger instance
        """
        if self._logger is None:
            self._logger = logging.getLogger(self.name)
            self._logger.setLevel(logging.INFO)

            # Add console handler if none exists
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)

        return self._logger
