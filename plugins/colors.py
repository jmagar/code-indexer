"""Nord color theme for syntax highlighting."""

from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
)


class NordStyle(Style):
    """Nord color theme for Pygments."""

    background_color = "#2E3440"  # nord0
    highlight_color = "#3B4252"  # nord1

    styles = {
        Token: "#D8DEE9",  # nord4 - default text
        Comment: "italic #616E88",  # nord3 - comments
        Keyword: "bold #81A1C1",  # nord9 - keywords
        Keyword.Constant: "bold #81A1C1",
        Keyword.Declaration: "bold #81A1C1",
        Keyword.Namespace: "bold #81A1C1",
        Keyword.Type: "bold #81A1C1",
        Name: "#D8DEE9",  # nord4 - names
        Name.Builtin: "#88C0D0",  # nord8 - built-in names
        Name.Class: "bold #8FBCBB",  # nord7 - classes
        Name.Function: "#88C0D0",  # nord8 - functions
        Name.Decorator: "#B48EAD",  # nord15 - decorators
        String: "#A3BE8C",  # nord14 - strings
        String.Doc: "italic #616E88",  # nord3 - docstrings
        Number: "#B48EAD",  # nord15 - numbers
        Operator: "#81A1C1",  # nord9 - operators
        Operator.Word: "bold #81A1C1",
        Punctuation: "#ECEFF4",  # nord6 - punctuation
        Generic.Heading: "bold #88C0D0",  # nord8
        Generic.Subheading: "bold #88C0D0",
        Generic.Deleted: "#BF616A",  # nord11
        Generic.Inserted: "#A3BE8C",  # nord14
        Generic.Error: "#BF616A",  # nord11
        Generic.Emph: "italic",
        Generic.Strong: "bold",
        Generic.Prompt: "bold #616E88",  # nord3
        Error: "bg:#BF616A #2E3440",  # nord11 background, nord0 text
    }
