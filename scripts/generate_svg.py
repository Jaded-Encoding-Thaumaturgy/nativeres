import os
import shlex
import sys
import traceback

from rich.console import Console
from rich.text import Text
from typer.testing import CliRunner

from nativeres.cli import app  # type: ignore[attr-defined]

COLUMNS = 110


os.environ["FORCE_COLOR"] = "1"
os.environ["COLUMNS"] = str(COLUMNS)
os.environ["TERM"] = "xterm-256color"

font_injection = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700');
.terminal { font-family: 'JetBrains Mono', Consolas, monospace !important; }
"""


def generate_svg(command: str = "") -> None:
    runner = CliRunner()

    args = shlex.split(command) if command else ["--help"]

    result = runner.invoke(app, args, color=True)

    if result.exception:
        print("ERROR: Command raised an exception!")
        if result.stdout:
            print("\nCaptured stdout:")
            print(result.stdout)
        if result.stderr:
            print("\nCaptured stderr:")
            print(result.stderr)

        traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
        return

    output_text = result.stdout + result.stderr

    if not output_text:
        print("Failed to capture output!")
        sys.exit(1)

    # Force rounded boxes
    output_text = output_text.replace("┌", "╭").replace("┐", "╮").replace("└", "╰").replace("┘", "╯")

    if "\x1b" not in output_text:
        print("⚠️ WARNING: Colors still stripped! Check your import path.")
    else:
        print("✅ Success: ANSI Color codes captured natively from memory!")

    console = Console(record=True, width=COLUMNS, force_terminal=True, color_system="truecolor")
    text = Text.from_ansi(output_text)
    console.print(text)

    cmd_name = args[0] if args else "nativeres"
    svg_path = f"{cmd_name}_help.svg"

    console.save_svg(svg_path, title=f"{command or 'nativeres'}")

    # THE FONT FIX
    with open(svg_path, encoding="utf-8") as f:
        svg_content = f.read()

    svg_content = svg_content.replace("<style>\n", f"<style>\n{font_injection}")

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"Successfully generated {svg_path}!")


if __name__ == "__main__":
    generate_svg()
    generate_svg("getnative --help")
    generate_svg("getscaler --help")
    generate_svg("getfreq --help")

    # generate_svg(
    #     'getscaler "00007.m2ts" 800 --frame 15000'
    # )
