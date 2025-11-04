import random
from pathlib import Path

import numpy as np
import torch
import typer

from src.inference.cli_inference import load_emotion_model, predict_from_input

model_path = "models/bert_model"
tokenizer_path = "models/bert_model/tokenizer_config.json"


# Seed for reproducibility
def set_seed(seed=42):
    """
    Set random seeds for reproducibility across torch, numpy, and random libraries.
    Also sets CUDA backend for deterministic behavior.

    Args:
        seed (int, optional): The seed value to set. Defaults to 42.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define a custom help message
HELP_MESSAGE = """
📘 Emotion Prediction CLI

Use the --text or --file option to provide input.
Examples:

    python cli.py predict --text "I am so happy today!"
    python cli.py predict --file "path/to/your/file.txt"

Options:
    --text / -t : Provide a text string to analyze.
    --file / -f : Provide a path to a .txt file.
    --verbose / -v : Enable debug logging for extra info.
"""

# Initialize the CLI app with help
app = typer.Typer(help=HELP_MESSAGE)


@app.callback()
def main(ctx: typer.Context):
    """
    📘 Emotion Prediction CLI Tool
    Run --help for usage instructions.
    """
    pass

CLASS_NAMES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]


@app.command()
def predict(
    text: str = typer.Option(None, "--text", "-t", help="Input text to analyze."),
    file: Path = typer.Option(None, "--file", "-f", help="Path to text file."),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show extra debug info."
    ),
):
    """
    This command analyzes the input text or file to predict the emotion.

    You must provide either the --text/-t option with a string to analyze,
    or the --file/-f option with a path to a text file.

    Optionally, use --verbose/-v to display extra debug information during prediction.

    :param text: Input text to analyze. Mutually exclusive with 'file'.
    :param file: Path to a text file containing input text. Mutually exclusive with 'text'.
    :param verbose: If True, displays extra debug information.
    :raises typer.Exit: If neither --text nor --file is provided.

    **Example:**

        cli.py predict --text "I am so happy today!"
        cli.py predict --file ./input.txt --verbose
    """
    set_seed(42)

    if not text and not file:
        typer.echo("❌ You must provide either --text or --file")
        raise typer.Exit(code=1)

    input_text = file.read_text(encoding="utf-8").strip() if file else text

    if verbose:
        typer.echo("📦 Loading model...")
    model, tokenizer, device = load_emotion_model(model_path, tokenizer_path)

    if verbose:
        typer.echo("📄 Analyzing text:")
        typer.echo(input_text)

    emotion = predict_from_input(input_text, model, tokenizer, device, CLASS_NAMES)

    typer.secho(
        f"\n🎭 Predicted Emotion: {emotion}", fg=typer.colors.BRIGHT_GREEN, bold=True
    )
