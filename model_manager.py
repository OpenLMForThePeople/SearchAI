from pathlib import Path
import json
import torch


import sys


def get_models_dir():

    if getattr(
        sys,
        "frozen",
        False
    ):
        base_dir = Path(
            sys.executable
        ).resolve().parent

    else:
        base_dir = Path(
            __file__
        ).resolve().parent

    models_dir = (
        base_dir / "models"
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return models_dir

def get_model_dir(model_name):
    return (
        get_models_dir()
        / model_name
    )


def get_model_path(model_name):
    return (
        get_model_dir(model_name)
        / "model.pt"
    )


def save_model(
    agent,
    model_name,
    criteria,
    query,
    hidden_layers,
    input_size,
    output_size,
    engine,
):
    model_dir = get_model_dir(
        model_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # MODEL WEIGHTS
    # ---------------------------------------------------------

    torch.save(
        agent.model.state_dict(),
        model_dir / "model.pt"
    )

    # ---------------------------------------------------------
    # CRITERIA
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
    # METADATA
    # ---------------------------------------------------------

    metadata = {
        "query": query,
        "engine": engine,
        "input_size": input_size,
        "hidden_layers": list(
            hidden_layers
        ),
        "output_size": output_size,
        "architecture": [
            input_size,
            *hidden_layers,
            output_size,
        ],
    }

    with open(
        model_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )


def load_model(
    agent,
    model_name
):
    model_path = get_model_path(
        model_name
    )

    if not model_path.exists():
        return False

    try:

        state_dict = torch.load(
            model_path,
            map_location="cpu"
        )

        agent.model.load_state_dict(
            state_dict
        )

        return True

    except (
        OSError,
        RuntimeError,
        EOFError
    ):
        return False


def load_criteria(model_name):

    criteria_path = (
        get_model_dir(model_name)
        / "criteria.json"
    )

    if not criteria_path.exists():
        return None

    try:

        with open(
            criteria_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        OSError,
        json.JSONDecodeError
    ):
        return None


def load_metadata(model_name):

    metadata_path = (
        get_model_dir(model_name)
        / "metadata.json"
    )

    if not metadata_path.exists():
        return None

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        OSError,
        json.JSONDecodeError
    ):
        return None


def load_query(model_name):

    metadata = load_metadata(
        model_name
    )

    if metadata is None:
        return None

    return metadata.get(
        "query"
    )


def get_round_count(model_name):

    round_file = (
        get_model_dir(model_name)
        / "round.txt"
    )

    if not round_file.exists():
        return 0

    try:

        return int(
            round_file.read_text(
                encoding="utf-8"
            ).strip()
        )

    except (
        OSError,
        ValueError
    ):
        return 0


def increment_round(model_name):

    model_dir = get_model_dir(
        model_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    round_file = (
        model_dir / "round.txt"
    )

    rounds = get_round_count(
        model_name
    )

    rounds += 1

    round_file.write_text(
        str(rounds),
        encoding="utf-8"
    )

    return rounds


# =============================================================
# TRAINING BACKUPS
# =============================================================

def get_training_backup_path(
    model_name
):

    return (
        get_model_dir(model_name)
        / "training_backup.json"
    )


def save_training_backup(
    model_name,
    query,
    criteria,
    videos,
    training_data,
    video_index,
    criterion_index,
):

    backup = {
        "query": query,
        "criteria": criteria,
        "videos": videos,
        "training_data": training_data,
        "video_index": video_index,
        "criterion_index": criterion_index,
    }

    backup_path = get_training_backup_path(
        model_name
    )

    backup_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        backup_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            backup,
            file,
            indent=4,
            ensure_ascii=False
        )


def load_training_backup(
    model_name
):

    backup_path = get_training_backup_path(
        model_name
    )

    if not backup_path.exists():
        return None

    try:

        with open(
            backup_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        OSError,
        json.JSONDecodeError
    ):
        return None


def delete_training_backup(
    model_name
):

    backup_path = get_training_backup_path(
        model_name
    )

    if backup_path.exists():
        backup_path.unlink()