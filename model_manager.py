from pathlib import Path
import json

import torch


MODEL_DIR = Path(__file__).resolve().parent / "models"


def get_model_dir(model_name):
    return MODEL_DIR / model_name


def get_model_path(model_name):
    return (
        get_model_dir(model_name)
        / "model.pt"
    )


def get_criteria_path(model_name):
    return (
        get_model_dir(model_name)
        / "criteria.json"
    )


def get_query_path(model_name):
    return (
        get_model_dir(model_name)
        / "query.json"
    )


def load_criteria(model_name):
    criteria_path = get_criteria_path(
        model_name
    )

    if not criteria_path.exists():
        return None

    with open(
        criteria_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_model(
    agent,
    model_name,
    criteria,
    query,
):
    model_dir = get_model_dir(
        model_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict":
                agent.model.state_dict(),

            "optimizer_state_dicts":
                [
                    optimizer.state_dict()
                    for optimizer
                    in agent.optimizers
                ],

            "state_size":
                agent.state_size,

            "criterion_count":
                agent.criterion_count,
        },
        get_model_path(model_name),
    )

    with open(
        get_criteria_path(model_name),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            criteria,
            file,
            indent=4
        )

    with open(
        get_query_path(model_name),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            query,
            file,
            indent=4
        )


def load_query(model_name):
    query_path = get_query_path(
        model_name
    )

    if not query_path.exists():
        return None

    with open(
        query_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_model(agent, model_name):
    model_path = get_model_path(
        model_name
    )

    if not model_path.exists():
        return False

    checkpoint = torch.load(
        model_path,
        map_location=agent.device,
    )

    saved_state_size = checkpoint.get(
        "state_size"
    )

    if saved_state_size != agent.state_size:
        print(
            "\nModel architecture mismatch."
        )

        print(
            f"Saved state size: "
            f"{saved_state_size}"
        )

        print(
            f"Current state size: "
            f"{agent.state_size}"
        )

        return False

    saved_criterion_count = checkpoint.get(
        "criterion_count"
    )

    if (
        saved_criterion_count
        != agent.criterion_count
    ):
        print(
            "\nCriterion count mismatch."
        )

        print(
            f"Saved criteria: "
            f"{saved_criterion_count}"
        )

        print(
            f"Current criteria: "
            f"{agent.criterion_count}"
        )

        return False

    agent.model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer_states = checkpoint.get(
        "optimizer_state_dicts"
    )

    if optimizer_states:
        if len(optimizer_states) != len(
            agent.optimizers
        ):
            print(
                "\nOptimizer count mismatch."
            )

            return False

        for optimizer, state in zip(
            agent.optimizers,
            optimizer_states
        ):
            optimizer.load_state_dict(
                state
            )

    return True

def get_round_count(model_name):
    model_dir = get_model_dir(
        model_name
    )

    round_file = (
        model_dir / "round.txt"
    )

    if not round_file.exists():
        return 0

    try:
        return int(
            round_file.read_text(
                encoding="utf-8"
            ).strip()
        )

    except ValueError:
        return 0


def increment_round(model_name):
    model_dir = get_model_dir(
        model_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    rounds = get_round_count(
        model_name
    )

    rounds += 1

    round_file = (
        model_dir / "round.txt"
    )

    round_file.write_text(
        str(rounds),
        encoding="utf-8"
    )

    return rounds

def get_training_backup_path(model_name):
    return (
        get_model_dir(model_name)
        / "training_backup.txt"
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
    model_dir = get_model_dir(
        model_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    backup_data = {
        "query": query,
        "criteria": criteria,
        "videos": videos,
        "training_data": training_data,
        "video_index": video_index,
        "criterion_index": criterion_index,
    }

    backup_file = get_training_backup_path(
        model_name
    )

    backup_file.write_text(
        json.dumps(
            backup_data,
            indent=4
        ),
        encoding="utf-8"
    )


def load_training_backup(model_name):
    backup_file = get_training_backup_path(
        model_name
    )

    if not backup_file.exists():
        return None

    try:
        return json.loads(
            backup_file.read_text(
                encoding="utf-8"
            )
        )

    except (
        json.JSONDecodeError,
        OSError
    ):
        return None


def delete_training_backup(model_name):
    backup_file = get_training_backup_path(
        model_name
    )

    if backup_file.exists():
        backup_file.unlink()