from search import search_youtube
from dqn_agent import DQNAgent
import csv

from model_manager import (
    load_model,
    load_criteria,
    load_query,
    load_metadata,
    save_model,
    get_model_dir,
    get_round_count,
    increment_round,
    save_training_backup,
    load_training_backup,
    delete_training_backup,
    get_training_backup_path,
)

from title_encoder import (
    encode_title_and_channel,
)

class TrainingInterrupted(Exception):
    pass

def export_approved_videos(
    model_name,
    training_data
):
    """
    Export videos where ALL criteria were approved with 1.0.

    CSV contains ONLY:
        Title
        Channel
        URL
    """

    model_dir = get_model_dir(
        model_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        model_dir / "approved_videos.csv"
    )

    approved_videos = []

    for item in training_data:

        scores = item["scores"]

        # ALL criteria must be exactly 1.0.
        if all(
            float(score) == 1.0
            for score in scores
        ):
            approved_videos.append(
                item["video"]
            )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Title",
            "Channel",
            "URL"
        ])

        for video in approved_videos:

            writer.writerow([
                video.get("title", ""),
                video.get("channel", ""),
                video.get("url", "")
            ])

    print()
    print(
        f"Approved videos exported: "
        f"{len(approved_videos)}"
    )

    print(
        f"CSV: {output_file}"
    )

def get_score(criterion):
    while True:
        try:
            value = input(
                f"{criterion} "
                "(0.0 (NO) - 1.0 (YES), "
                "blank = UNKNOWN, "
                "I = INTERRUPT): "
            ).strip()

        except KeyboardInterrupt:
            print()
            raise TrainingInterrupted

        if value.lower() == "i":
            raise TrainingInterrupted

        if value == "":
            return 0.5

        try:
            value = float(value)

            if 0.0 <= value <= 1.0:
                return value

            print(
                "Enter a value between "
                "0.0 and 1.0."
            )

        except ValueError:
            print("Invalid value.")

def display_table(
    training_data,
    criteria
):
    print("\n" + "=" * 120)
    print("TRAINING REVIEW")
    print("=" * 120)

    print(
        f"{'#':<4}"
        f"{'Title':<55}",
        end=""
    )

    for criterion in criteria:
        print(
            f"{criterion:<20}",
            end=""
        )

    print()
    print("-" * 120)

    for index, item in enumerate(
        training_data,
        start=1
    ):
        title = item["video"]["title"]

        if len(title) > 52:
            title = title[:49] + "..."

        print(
            f"{index:<4}"
            f"{title:<55}",
            end=""
        )

        for score in item["scores"]:
            print(
                f"{score:<20.2f}",
                end=""
            )

        print()

    print("=" * 120)


def edit_choice(
    training_data,
    criteria
):
    if not training_data:
        print(
            "\nThere are no choices to edit."
        )
        return

    choice = input(
        "\nSelect video number: "
    ).strip()

    try:
        index = int(choice) - 1

        if (
            index < 0
            or index >= len(training_data)
        ):
            raise ValueError

    except ValueError:
        print("Invalid video.")
        return

    item = training_data[index]

    print(
        f"\nTitle: "
        f"{item['video']['title']}"
    )

    print("\nCurrent scores:")

    for criterion_index, criterion in enumerate(
        criteria,
        start=1
    ):
        score = item["scores"][
            criterion_index - 1
        ]

        print(
            f"{criterion_index}. "
            f"{criterion}: {score:.2f}"
        )

    criterion_choice = input(
        "\nSelect criterion: "
    ).strip()

    try:
        criterion_index = (
            int(criterion_choice) - 1
        )

        if (
            criterion_index < 0
            or criterion_index >= len(criteria)
        ):
            raise ValueError

    except ValueError:
        print("Invalid criterion.")
        return

    item["scores"][
        criterion_index
    ] = get_score(
        criteria[criterion_index]
    )

    print("\nChoice updated.")


def clear_choices(training_data):
    confirm = input(
        "\nClear ALL training choices? "
        "(y/n): "
    ).strip().lower()

    if confirm != "y":
        print("Cancelled.")
        return False

    training_data.clear()

    print("\nAll choices cleared.")

    return True


def review_menu(
    training_data,
    criteria
):
    while True:
        display_table(
            training_data,
            criteria
        )

        print("1. Edit choice")
        print("2. Clear all choices")
        print("3. Confirm and train")
        print("4. Cancel")
        print()

        choice = input(
            "Select: "
        ).strip()

        if choice == "1":
            edit_choice(
                training_data,
                criteria
            )

        elif choice == "2":
            clear_choices(
                training_data
            )

        elif choice == "3":
            if not training_data:
                print(
                    "\nThere are no choices "
                    "to train."
                )
                continue

            return "train"

        elif choice == "4":
            return "cancel"

        else:
            print("\nInvalid option.")


def show_prediction(
    agent,
    criteria,
    state
):
    predictions = agent.predict(
        state
    )

    print("\nSearchAI prediction:")

    for criterion, prediction in zip(
        criteria,
        predictions
    ):
        print(
            f"  {criterion}: "
            f"{prediction:.4f}"
        )


def train(model_name):
    print(
        f"\nTraining model: "
        f"{model_name}"
    )

    current_round = get_round_count(
        model_name
    )

    print(
        f"Training round: "
        f"{current_round + 1}"
    )

    criteria = load_criteria(
        model_name
    )

    if criteria is None:
        print(
            "Could not load model criteria."
        )
        return

    query = load_query(
        model_name
    )

    if not query:
        print(
            "Could not load the model's "
            "permanent query."
        )
        return

    # ---------------------------------------------------------
    # LOAD USER-DEFINED NETWORK ARCHITECTURE
    # ---------------------------------------------------------

    metadata = load_metadata(
        model_name
    )

    if metadata is None:
        print(
            "Could not load model metadata."
        )
        return

    input_size = metadata.get(
        "input_size"
    )

    hidden_layers = metadata.get(
        "hidden_layers"
    )

    output_size = metadata.get(
        "output_size"
    )

    if not isinstance(
        input_size,
        int
    ) or input_size <= 0:
        print(
            "Invalid model input size."
        )
        return

    if not isinstance(
        hidden_layers,
        list
    ) or not hidden_layers:

        print(
            "Invalid model hidden-layer "
            "configuration."
        )
        return

    if not all(
        isinstance(size, int)
        and size > 0
        for size in hidden_layers
    ):
        print(
            "Invalid model hidden-layer "
            "configuration."
        )
        return

    if not isinstance(
        output_size,
        int
    ) or output_size <= 0:
        print(
            "Invalid model output size."
        )
        return

    if output_size != len(criteria):
        print(
            "\nModel architecture is "
            "inconsistent with its criteria."
        )

        print(
            f"Output nodes: {output_size}"
        )

        print(
            f"Criteria:     {len(criteria)}"
        )

        print(
            "The output layer must have "
            "one node per criterion."
        )

        return

    print()
    print(
        "Network architecture:"
    )

    print(
        " -> ".join(
            map(
                str,
                [
                    input_size,
                    *hidden_layers,
                    output_size,
                ]
            )
        )
    )

    # ---------------------------------------------------------
    # CREATE THE EXACT NETWORK THE USER CHOSE
    # ---------------------------------------------------------

    agent = DQNAgent(
        state_size=input_size,
        criterion_count=output_size,
        hidden_layers=hidden_layers,
    )

    if not load_model(
        agent,
        model_name
    ):
        print(
            "Could not load model weights."
        )
        return

    # ---------------------------------------------------------
    # CHECK FOR EXISTING BACKUP
    # ---------------------------------------------------------

    backup = load_training_backup(
        model_name
    )

    training_data = []
    videos = []
    start_video_index = 0
    start_criterion_index = 0

    if backup:
        print()
        print(
            "A previous training session "
            "was interrupted."
        )

        print(
            "1. Resume previous training"
        )
        print(
            "2. Start new training"
        )

        choice = input(
            "\nSelect: "
        ).strip()

        if choice == "1":

            training_data = backup.get(
                "training_data",
                []
            )

            videos = backup.get(
                "videos",
                []
            )

            start_video_index = backup.get(
                "video_index",
                0
            )

            start_criterion_index = backup.get(
                "criterion_index",
                0
            )

            print(
                "\nResuming previous "
                "training session..."
            )

            print(
                f"Resuming at video "
                f"{start_video_index + 1}"
            )

            print(
                f"Resuming at criterion "
                f"{start_criterion_index + 1}"
            )

        elif choice == "2":

            delete_training_backup(
                model_name
            )

            print(
                "\nStarting new training "
                "session..."
            )

        else:
            print(
                "\nInvalid option."
            )
            return

    # ---------------------------------------------------------
    # SEARCH ONLY IF WE ARE NOT RESUMING
    # ---------------------------------------------------------

    if not videos:

        print(
            f"\nPermanent query: "
            f"{query}"
        )

        print("\nCriteria:")

        for index, criterion in enumerate(
            criteria,
            start=1
        ):
            print(
                f"{index}. {criterion}"
            )

        print(
            "\nSearching YouTube..."
        )

        videos = search_youtube(
            query,
            max_results=10
        )

        if not videos:
            print(
                "\nNo videos found."
            )
            return

        print(
            f"\nFound {len(videos)} videos."
        )

        # Save the search results immediately.
        # This guarantees that a resume uses
        # the SAME videos instead of searching again.
        save_training_backup(
            model_name=model_name,
            query=query,
            criteria=criteria,
            videos=videos,
            training_data=training_data,
            video_index=0,
            criterion_index=0,
        )

    else:
        print(
            f"\nLoaded {len(videos)} videos "
            "from training backup."
        )

    # ---------------------------------------------------------
    # TRAINING / HUMAN REVIEW
    # ---------------------------------------------------------

    try:

        for index in range(
            start_video_index,
            len(videos)
        ):

            video = videos[index]

            print(
                "\n" + "=" * 60
            )

            print(
                f"VIDEO "
                f"{index + 1}/{len(videos)}"
            )

            print(
                "=" * 60
            )

            print(
                f"Title:   "
                f"{video['title']}"
            )

            print(
                f"Channel: "
                f"{video['channel']}"
            )

            print(
                f"URL:     "
                f"{video['url']}"
            )

            state = encode_title_and_channel(
                video["title"],
                video["channel"],
                state_size=input_size
            )

            show_prediction(
                agent,
                criteria,
                state
            )

            # -------------------------------------------------
            # GET OR CREATE THIS VIDEO'S TRAINING ITEM
            # -------------------------------------------------

            if index < len(training_data):

                item = training_data[index]

            else:

                item = {
                    "video": video,
                    "scores": [],
                    "state": (
                        state.tolist()
                        if hasattr(
                            state,
                            "tolist"
                        )
                        else state
                    ),
                }

                training_data.append(
                    item
                )

            # -------------------------------------------------
            # DETERMINE WHERE TO START
            # -------------------------------------------------

            if index == start_video_index:

                criterion_start = (
                    start_criterion_index
                )

            else:

                criterion_start = 0

            # -------------------------------------------------
            # ASK EACH CRITERION
            # -------------------------------------------------

            for criterion_index in range(
                criterion_start,
                len(criteria)
            ):

                criterion = criteria[
                    criterion_index
                ]

                # IMPORTANT:
                # Save BEFORE asking the question.
                #
                # Therefore:
                #
                #   I
                #   Ctrl+C
                #
                # can resume on THIS EXACT
                # criterion.
                save_training_backup(
                    model_name=model_name,
                    query=query,
                    criteria=criteria,
                    videos=videos,
                    training_data=training_data,
                    video_index=index,
                    criterion_index=criterion_index,
                )

                score = get_score(
                    criterion
                )

                if (
                    criterion_index
                    < len(item["scores"])
                ):

                    item["scores"][
                        criterion_index
                    ] = score

                else:

                    item["scores"].append(
                        score
                    )

                # Save immediately after answering too.
                save_training_backup(
                    model_name=model_name,
                    query=query,
                    criteria=criteria,
                    videos=videos,
                    training_data=training_data,
                    video_index=index,
                    criterion_index=(
                        criterion_index + 1
                    ),
                )

            print(
                "\nVideo completed."
            )

            start_criterion_index = 0

        # -----------------------------------------------------
        # ALL VIDEOS COMPLETED
        # -----------------------------------------------------

        delete_training_backup(
            model_name
        )

    except TrainingInterrupted:

        print(
            "\nTraining interrupted."
        )

        print(
            "Training session saved."
        )

        print(
            "Run training again and choose "
            "'1. Resume previous training'."
        )

        return

    # ---------------------------------------------------------
    # REVIEW TRAINING DATA
    # ---------------------------------------------------------

    result = review_menu(
        training_data,
        criteria
    )

    if result == "cancel":
        print(
            "\nTraining cancelled."
        )
        return

    # ---------------------------------------------------------
    # TRAIN THE MODEL
    # ---------------------------------------------------------

    print(
        "\nTraining model..."
    )

    losses = []

    for item in training_data:

        loss = agent.train_step(
            state=item["state"],
            targets=item["scores"]
        )

        losses.append(
            loss
        )

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    save_model(
        agent,
        model_name,
        criteria,
        query,
        hidden_layers=hidden_layers,
        input_size=input_size,
        output_size=output_size,
    )

    round_count = increment_round(
        model_name
    )

    average_loss = (
        sum(losses)
        / len(losses)
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print(
        "\nTraining complete."
    )

    print(
        f"Training round completed: "
        f"{round_count}"
    )

    export_approved_videos(
        model_name,
        training_data
    )

    print(
        f"Videos trained: "
        f"{len(training_data)}"
    )

    print(
        f"Average loss: "
        f"{average_loss:.6f}"
    )
