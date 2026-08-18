import contextlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import cyclopts
from rich.console import Console

from nativeres.cli import app as nativeres_app


def discover_commands(app_obj: cyclopts.App, prefix: Sequence[str] | None = None) -> list[list[str]]:
    prefix = prefix or []
    commands = [[*prefix, "--help"]] if prefix else [["--help"]]

    sub_keys = [k for k in app_obj if not k.startswith("-")]
    for sub in sub_keys:
        sub_app = app_obj[sub]
        sub_prefix = [*prefix, sub]
        commands.extend(discover_commands(sub_app, sub_prefix))

    return commands


def generate_svg(command_args: Sequence[str], output_dir: Path, columns: int) -> Path:
    os.environ["FORCE_COLOR"] = "1"
    os.environ["COLUMNS"] = str(columns)
    os.environ["TERM"] = "xterm-256color"

    console = Console(record=True, width=columns, force_terminal=True, legacy_windows=False, safe_box=False)

    with (
        contextlib.suppress(SystemExit),
        open(os.devnull, "w", encoding="utf-8") as devnull,
        contextlib.redirect_stderr(devnull),
        contextlib.redirect_stdout(devnull),
    ):
        nativeres_app.meta(command_args, console=console)

    if command_args == ["--help"]:
        cmd_title = "nativeres --help"
        file_name = "nativeres_help.svg"
    else:
        sub_name = "_".join(arg for arg in command_args if arg != "--help")
        cmd_title = f"nativeres {' '.join(command_args)}"
        file_name = f"{sub_name}_help.svg"

    output_path = output_dir / file_name
    console.save_svg(output_path, title=cmd_title)
    print(f"Saved: {output_path}", file=sys.stderr)
    return output_path


def main(output_dir: Path = Path("assets"), width: int = 110) -> None:
    """Generate SVG help outputs for nativeres commands.

    Args:
        output_dir: Directory to save generated SVG files.
        width: Terminal width for rendered SVG
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_commands = discover_commands(nativeres_app)

    print(f"Generating SVGs for {len(all_commands)} command(s)...", file=sys.stderr)
    for cmd in all_commands:
        generate_svg(cmd, output_dir=output_dir, columns=width)

    print("All SVGs generated successfully", file=sys.stderr)


if __name__ == "__main__":
    cyclopts.run(main)
    # generate_svg(
    #     'getscaler "00007.m2ts" 800 --frame 15000'
    # )
