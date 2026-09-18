from pathlib import Path
import shutil
import json
import shlex

from colorama import init, Fore

from train import train

from dqn_agent import DQNAgent

from model_manager import (
    save_model,
    load_criteria,
    get_models_dir,
)

init(autoreset=True)

MODEL_DIR = get_models_dir()


class SearchAIFileSystem:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.current = self.root

        self.root.mkdir(
            parents=True,
            exist_ok=True
        )

    def resolve(self, target):
        target = target.strip()

        if not target:
            return self.current

        if target in ("/", "\\"):
            return self.root

        path = Path(target)

        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (
                self.current / path
            ).resolve()

        try:
            resolved.relative_to(self.root)

        except ValueError:
            raise ValueError(
                "Cannot navigate outside SearchAI."
            )

        return resolved

    def cd(self, target):
        try:
            path = self.resolve(target)

        except ValueError as error:
            print(f"cd: {error}")
            return

        if not path.exists():
            print(
                f"cd: '{target}' "
                "does not exist."
            )
            return

        if not path.is_dir():
            print(
                f"cd: '{target}' "
                "is not a directory."
            )
            return

        self.current = path

    def pwd(self):
        print(self.current_path())

    def current_path(self):
        relative = self.current.relative_to(
            self.root
        )

        if str(relative) == ".":
            return "\\"

        return "\\" + str(relative)

    def list_criteria(self, target):
        try:
            path = self.resolve(target)

        except ValueError as error:
            print(
                f"criteria: {error}"
            )
            return

        if not path.exists():
            print(
                f"criteria: '{target}' "
                "does not exist."
            )
            return

        if not self.is_model(path):
            print(
                f"criteria: '{target}' "
                "is not a model."
            )
            return

        model_name = str(
            path.relative_to(self.root)
        )

        criteria = load_criteria(
            model_name
        )

        if criteria is None:
            print(
                f"Could not load criteria "
                f"for model: {model_name}"
            )
            return

        print()
        print(
            f"Criteria for: {path.name}"
        )
        print("=" * 60)

        for index, criterion in enumerate(
            criteria,
            start=1
        ):
            print(
                f"{index}. {criterion}"
            )

        print("=" * 60)

        print(
            f"Total criteria: "
            f"{len(criteria)}"
        )

    @staticmethod
    def get_model_name():
        while True:
            name = input(
                "\nModel name: "
            ).strip()

            if not name:
                print(
                    "Model name cannot be empty."
                )
                continue

            invalid_characters = (
                '<>:"/\\|?*.'
            )

            if any(
                char in name
                for char in invalid_characters
            ):
                print(
                    'Model name contains an '
                    'invalid character. '
                    'Avoid: < > : " / \\ | ? *'
                )
                continue

            return name

    def is_model(self, path):
        if not path.is_dir():
            return False

        criteria = path / "criteria.json"

        possible_weights = (
            path / "model.pt",
            path / "model.pth",
            path / "dqn_model.pt",
            path / "weights.pt",
        )

        return (
            criteria.exists()
            and any(
                file.exists()
                for file in possible_weights
            )
        )

    def dir(self):
        items = sorted(
            self.current.iterdir(),
            key=lambda path: (
                not path.is_dir(),
                path.name.lower()
            )
        )

        if not items:
            print("(empty)")
            return

        print()

        for item in items:
            if item.is_dir():

                if self.is_model(item):
                    print(
                        Fore.CYAN + f"[MODEL 🤖] {item.name}"
                    )
                else:
                    print(
                        Fore.YELLOW + f"[DIR 📁]   {item.name}"
                    )

            else:
                print(
                    f"[FILE 📄]  {item.name}"
                )

    def mkdir(self, name):
        if not name:
            print(
                "Usage: mkdir <folder>"
            )
            return

        try:
            path = self.resolve(name)

        except ValueError as error:
            print(f"mkdir: {error}")
            return

        if path.exists():
            print(
                f"mkdir: '{name}' "
                "already exists."
            )
            return

        path.mkdir(
            parents=True,
            exist_ok=False
        )

        print(
            f"Directory '{name}' created."
        )

    def open(self, target):
        try:
            path = self.resolve(target)

        except ValueError as error:
            print(f"open: {error}")
            return

        if not path.exists():
            print(
                f"open: '{target}' "
                "does not exist."
            )
            return

        if path.is_dir():

            if self.is_model(path):
                self.open_model(path)
            else:
                self.current = path

            return

        print(
            f"open: '{target}' "
            "is not a directory or model."
        )

    # ---------------------------------------------------------
    # OPEN MODEL
    # ---------------------------------------------------------

    def open_model(self, path):
        model_name = path.name

        print()
        print(
            f"Opening model: {model_name}"
        )
        print(
            f"Path: {path}"
        )
        print()

        print("1. Train")
        print("2. Back")

        choice = input(
            "\nSelect: "
        ).strip()

        if choice == "1":

            relative_model = path.relative_to(
                self.root
            )

            train(
                str(relative_model)
            )

        elif choice == "2":
            return

        else:
            print(
                "Invalid option."
            )

    # ---------------------------------------------------------
    # CREATE MODEL
    # ---------------------------------------------------------

    def create_model(self):
        name = SearchAIFileSystem.get_model_name()

        # ---------------------------------------------------------
        # ENGINE & QUERY SELECTION
        # ---------------------------------------------------------

        while True:
            engine_input = input(
                "\nSelect search engine (y/g/yt/gg/youtube/google): "
            ).strip().lower()

            if engine_input in ("y", "yt", "youtube"):
                engine = "yt"
                engine_display = "YouTube"
                break

            elif engine_input in ("g", "gg", "google"):
                engine = "google"
                engine_display = "Google"
                break

            print(
                "Invalid choice. "
                "Valid options: y, g, yt, gg, youtube, google."
            )

        query = input(
            f"\nPermanent {engine_display} search query: "
        ).strip()

        if not query:
            print(
                "Query cannot be empty."
            )
            return

        model_dir = self.current / name

        if model_dir.exists():
            print(
                "A model with that name "
                "already exists."
            )
            return

        # ---------------------------------------------------------
        # CREATE CRITERIA
        # ---------------------------------------------------------

        print(
            "\nCreate criteria for this model."
        )

        print(
            "Enter an empty name when "
            "you're finished.\n"
        )

        criteria = []

        while True:

            criterion = input(
                f"Criterion {len(criteria) + 1}: "
            ).strip()

            if not criterion:
                break

            criteria.append(
                criterion
            )

        if not criteria:
            print(
                "A model must have at least "
                "one criterion."
            )
            return

        # ---------------------------------------------------------
        # CONFIGURE NETWORK ARCHITECTURE
        # ---------------------------------------------------------

        print()
        print(
            "Configure neural network architecture."
        )

        # ---------------------------------------------------------
        # INPUT LAYER
        # ---------------------------------------------------------

        while True:

            input_size_raw = input(
                "\nInput layer size: "
            ).strip()

            try:
                input_size = int(
                    input_size_raw
                )

            except ValueError:
                print(
                    "Invalid input size. "
                    "Enter a positive integer."
                )
                continue

            if input_size <= 0:
                print(
                    "Input layer must contain "
                    "at least one node."
                )
                continue

            break

        # ---------------------------------------------------------
        # HIDDEN LAYERS
        # ---------------------------------------------------------

        print()
        print(
            "Enter hidden-layer sizes separated "
            "by spaces."
        )

        print(
            "Example: 128 64 32"
        )

        while True:

            architecture_input = input(
                "\nHidden layers: "
            ).strip()

            if not architecture_input:
                print(
                    "You must specify at least "
                    "one hidden layer."
                )
                continue

            try:
                hidden_layers = [
                    int(size)
                    for size in
                    architecture_input.split()
                ]

            except ValueError:
                print(
                    "Invalid architecture. "
                    "Use positive integers separated "
                    "by spaces."
                )
                continue

            if any(
                size <= 0
                for size in hidden_layers
            ):
                print(
                    "Every hidden layer must contain "
                    "at least one node."
                )
                continue

            break

        # ---------------------------------------------------------
        # OUTPUT LAYER
        # ---------------------------------------------------------

        while True:

            output_size_raw = input(
                "\nOutput layer size: "
            ).strip()

            try:
                output_size = int(
                    output_size_raw
                )

            except ValueError:
                print(
                    "Invalid output size. "
                    "Enter a positive integer."
                )
                continue

            if output_size <= 0:
                print(
                    "Output layer must contain "
                    "at least one node."
                )
                continue

            break

        # ---------------------------------------------------------
        # SHOW FINAL ARCHITECTURE
        # ---------------------------------------------------------

        architecture = [
            input_size,
            *hidden_layers,
            output_size,
        ]

        print()

        print(
            "Network architecture:"
        )

        print(
            " -> ".join(
                map(
                    str,
                    architecture
                )
            )
        )

        confirm = input(
            "\nCreate this network? (y/n): "
        ).strip().lower()

        if confirm != "y":

            print(
                "Model creation cancelled."
            )
            return

        # ---------------------------------------------------------
        # CREATE MODEL DIRECTORY
        # ---------------------------------------------------------

        model_dir.mkdir()

        # ---------------------------------------------------------
        # SAVE CRITERIA
        # ---------------------------------------------------------

        with open(
            model_dir / "criteria.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                criteria,
                file,
                indent=4,
                ensure_ascii=False
            )

        # ---------------------------------------------------------
        # CREATE NEURAL NETWORK
        # ---------------------------------------------------------

        agent = DQNAgent(
            state_size=input_size,
            criterion_count=output_size,
            hidden_layers=hidden_layers,
        )

        # ---------------------------------------------------------
        # SAVE MODEL
        # ---------------------------------------------------------

        relative_model = (
            model_dir.relative_to(
                self.root
            )
        )

        save_model(
            agent,
            str(relative_model),
            criteria,
            query,
            engine=engine,
            hidden_layers=hidden_layers,
            input_size=input_size,
            output_size=output_size,
        )

        # ---------------------------------------------------------
        # RESULTS
        # ---------------------------------------------------------

        print(
            f"\nModel '{name}' created for {engine_display} search."
        )

        print("\nCriteria:")

        for index, criterion in enumerate(
            criteria,
            start=1
        ):

            print(
                f"{index}. {criterion}"
            )

        print()

        print(
            "Network architecture:"
        )

        print(
            " -> ".join(
                map(
                    str,
                    architecture
                )
            )
        )

    # ---------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------

    def delete(self, target):
        try:
            path = self.resolve(target)

        except ValueError as error:
            print(f"delete: {error}")
            return

        if not path.exists():
            print(
                f"delete: '{target}' "
                "does not exist."
            )
            return

        if path == self.root:
            print(
                "delete: Cannot delete "
                "SearchAI root."
            )
            return

        if path.is_dir() and not self.is_model(path):
            item_type = "directory"

        elif self.is_model(path):
            item_type = "model"

        else:
            item_type = "file"

        confirm = input(
            f"Delete {item_type} "
            f"'{path.name}'? (y/n): "
        ).strip().lower()

        if confirm != "y":
            print(
                "Deletion cancelled."
            )
            return

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

        print(
            f"Deleted '{path.name}'."
        )

    # ---------------------------------------------------------
    # ROUND COUNT
    # ---------------------------------------------------------

    def get_round_count(self, target):
        try:
            path = self.resolve(target)

        except ValueError as error:
            print(
                f"round: {error}"
            )
            return

        if not path.exists():
            print(
                f"round: '{target}' "
                "does not exist."
            )
            return

        if not self.is_model(path):
            print(
                f"round: '{target}' "
                "is not a model."
            )
            return

        round_file = path / "round.txt"

        if not round_file.exists():
            rounds = 0

        else:
            try:
                rounds = int(
                    round_file.read_text(
                        encoding="utf-8"
                    ).strip()
                )

            except ValueError:
                rounds = 0

        print()
        print(
            f"Training rounds: {rounds}"
        )

    def increment_round(self, model_name):
        model_path = (
            self.root / model_name
        )

        round_file = (
            model_path / "round.txt"
        )

        if round_file.exists():
            try:
                rounds = int(
                    round_file.read_text(
                        encoding="utf-8"
                    ).strip()
                )

            except ValueError:
                rounds = 0

        else:
            rounds = 0

        rounds += 1

        round_file.write_text(
            str(rounds),
            encoding="utf-8"
        )

        return rounds


# ---------------------------------------------------------
# BANNERS
# ---------------------------------------------------------

def print_searchai_logo(logo_str: str):
    """
    Prints a multi-line ASCII logo in 4 quadrants:
    Red (Top-Left), Yellow (Top-Right)
    Blue (Bottom-Left), Green (Bottom-Right)
    """

    RED = (252, 65, 61)
    YELLOW = (255, 190, 0)
    BLUE = (49, 134, 255)
    GREEN = (0, 175, 87)

    def get_color_code(r, g, b):
        return f"\033[38;2;{r};{g};{b}m"

    RESET = "\033[0m"

    lines = logo_str.strip("\n").split("\n")

    if not lines:
        return

    height = len(lines)
    width = max(len(line) for line in lines)

    mid_y = height // 2
    mid_x = width // 2

    for y, line in enumerate(lines):
        colored_line = ""
        padded_line = line.ljust(width)

        for x, char in enumerate(padded_line):

            if char == " " or char == "\xa0":
                colored_line += " "
                continue

            if y < mid_y:
                color = RED if x < mid_x else YELLOW
            else:
                color = BLUE if x < mid_x else GREEN

            colored_line += (
                f"{get_color_code(*color)}{char}"
            )

        print(colored_line + RESET)


def print_banner():
    """
    GitHub-compatible banner function.

    The four-quadrant logo remains the actual banner
    used by the TUI.
    """

    ascii_logo = r"""
       _____                     _              _____ 
      / ____|                   | |       /\   |_   _|
     | (___   ___  __ _ _ __ ___| |__    /  \    | |
      \___ \ / _ \/ _` | '__/ __| '_ \  / /\ \   | |
      ____) |  __/ (_| | | | (__| | | |/ ____ \ _| |_
     |_____/ \___|\__,_|_|  \___|_| |_/_/    \_\_____|
    """

    print_searchai_logo(ascii_logo)


# ---------------------------------------------------------
# HELP
# ---------------------------------------------------------

def print_help():
    print()
    print("SearchAI commands:")
    print()

    print(
        "  cd <folder>       "
        "Change directory"
    )

    print(
        "  cd ..             "
        "Go to parent directory"
    )

    print(
        "  cd \\              "
        "Go to SearchAI root"
    )

    print(
        "  dir               "
        "List directory"
    )

    print(
        "  ls                "
        "List directory"
    )

    print(
        "  pwd               "
        "Show current directory"
    )

    print(
        "  mkdir <folder>    "
        "Create folder"
    )

    print(
        "  open <name>       "
        "Open folder/model"
    )

    print(
        "  create            "
        "Create model here"
    )

    print(
        "  delete <name>     "
        "Delete model/folder"
    )

    print(
        "  help              "
        "Show commands"
    )

    print(
        "  exit              "
        "Exit SearchAI"
    )

    print(
        "  round <model>      "
        "Show training round"
    )

    print()


# ---------------------------------------------------------
# TUI
# ---------------------------------------------------------

def tui():
    filesystem = SearchAIFileSystem(
        MODEL_DIR
    )

    print_banner()

    print(
        "SearchAI terminal initialized."
    )

    print(
        "Type 'help' for commands."
    )

    while True:

        try:
            command = input(
                f"\nsearchai"
                f"{filesystem.current_path()}> "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):
            print(
                "\n\nGoodbye."
            )
            break

        if not command:
            continue

        try:
            parts = shlex.split(command)

        except ValueError as error:
            print(
                f"SearchAI: {error}"
            )
            continue

        if not parts:
            continue

        command_name = parts[0].lower()

        argument = ""

        if len(parts) > 1:
            argument = " ".join(parts[1:])

        # -------------------------
        # cd
        # -------------------------

        if command_name == "cd":

            if not argument:
                filesystem.pwd()
                continue

            filesystem.cd(
                argument
            )

        # -------------------------
        # dir / ls
        # -------------------------

        elif command_name in (
            "dir",
            "ls"
        ):

            filesystem.dir()

        # -------------------------
        # pwd
        # -------------------------

        elif command_name == "pwd":

            filesystem.pwd()

        # -------------------------
        # mkdir
        # -------------------------

        elif command_name == "mkdir":

            filesystem.mkdir(
                argument
            )

        # -------------------------
        # open
        # -------------------------

        elif command_name == "open":

            if not argument:
                print(
                    "Usage: open <folder/model>"
                )
                continue

            filesystem.open(
                argument
            )

        # -------------------------
        # create
        # -------------------------

        elif command_name == "create":

            filesystem.create_model()

        # -------------------------
        # delete
        # -------------------------

        elif command_name in (
            "delete",
            "del",
            "rm"
        ):

            if not argument:
                print(
                    "Usage: delete <name>"
                )
                continue

            filesystem.delete(
                argument
            )

        # -------------------------
        # help
        # -------------------------

        elif command_name == "help":

            print_help()

        # -------------------------
        # exit
        # -------------------------

        elif command_name in (
            "exit",
            "quit"
        ):

            print(
                "\nGoodbye."
            )
            break

        # -------------------------
        # criteria
        # -------------------------

        elif command_name == "criteria":

            if not argument:
                print(
                    "Usage: criteria <model>"
                )
                continue

            filesystem.list_criteria(
                argument
            )

        # -------------------------
        # round
        # -------------------------

        elif command_name == "round":

            if not argument:
                print(
                    "Usage: round <model>"
                )
                continue

            filesystem.get_round_count(
                argument
            )

        # -------------------------
        # unknown
        # -------------------------

        else:

            print(
                f"'{command_name}' "
                "is not a SearchAI command."
            )


if __name__ == "__main__":
    tui()