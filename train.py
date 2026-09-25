import csv
import json
from search_yt import search_youtube
from search_google import (
    search_google,
    GoogleBlockedError,
)

from dqn_agent import DQNAgent

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


# =====================================================================
# HISTORY (JSON MEMORY) HELPERS
# =====================================================================

def get_history_path(model_name):
    return get_model_dir(model_name) / "history.json"

def load_history(model_name):
    path = get_history_path(model_name)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}

def save_history(model_name, history_data):
    path = get_history_path(model_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4, ensure_ascii=False)


def export_approved_videos(model_name, training_data):
    model_dir = get_model_dir(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_file = model_dir / "approved_videos.csv"

    approved_videos = []
    for item in training_data:
        scores = item["scores"]
        if all(float(score) == 1.0 for score in scores):
            approved_videos.append(item["video"])

    with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Channel", "URL"])
        for video in approved_videos:
            writer.writerow([
                video.get("title", ""),
                video.get("channel", ""),
                video.get("url", "")
            ])

    print(f"\nApproved videos exported: {len(approved_videos)}")
    print(f"CSV: {output_file}")


# =====================================================================
# WEB / FASTAPI INTERFACE
# =====================================================================

def init_training_session(model_name: str):
    criteria = load_criteria(model_name)
    query = load_query(model_name)
    metadata = load_metadata(model_name)

    if criteria is None or query is None or metadata is None:
        raise ValueError(f"Could not load metadata or criteria for model '{model_name}'. Ensure metadata.json and criteria.json exist.")

    engine = metadata.get("engine", "yt")
    input_size = metadata.get("input_size", 10)
    hidden_layers = metadata.get("hidden_layers", [128, 64, 32])
    output_size = metadata.get("output_size", len(criteria))

    agent = DQNAgent(
        state_size=input_size,
        criterion_count=output_size,
        hidden_layers=hidden_layers,
    )
    
    load_model(agent, model_name)

    backup = load_training_backup(model_name)
    if backup and backup.get("videos"):
        videos = backup.get("videos")
    else:
        if engine == "google":
            google_results = search_google(query, max_results=10)
            videos = [
                {
                    "title": r.get("title", ""),
                    "channel": "",
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                }
                for r in google_results
            ]
        else:
            videos = search_youtube(query, max_results=10)

        save_training_backup(
            model_name=model_name,
            query=query,
            criteria=criteria,
            videos=videos,
            training_data=[],
            video_index=0,
            criterion_index=0,
        )

    # Pre-fill scores from history if available
    history = load_history(model_name)
    items = []
    
    for video in videos:
        state = encode_title_and_channel(video["title"], video["channel"], state_size=input_size)
        preds = agent.predict(state)
        
        video_id = video.get("url") or video.get("title")
        default_scores = history.get(video_id) if video_id in history and len(history[video_id]) == len(criteria) else [0.5] * len(criteria)
        
        items.append({
            "video": video,
            "state": state.tolist() if hasattr(state, "tolist") else state,
            "predictions": [float(p) for p in preds],
            "scores": default_scores
        })

    return {
        "model_name": model_name,
        "criteria": criteria,
        "query": query,
        "engine": engine,
        "items": items
    }


def complete_training_session(model_name: str, training_data: list):
    criteria = load_criteria(model_name)
    query = load_query(model_name)
    metadata = load_metadata(model_name)

    engine = metadata.get("engine", "yt")
    input_size = metadata.get("input_size", 10)
    hidden_layers = metadata.get("hidden_layers", [128, 64, 32])
    output_size = metadata.get("output_size", len(criteria))

    agent = DQNAgent(
        state_size=input_size,
        criterion_count=output_size,
        hidden_layers=hidden_layers,
    )
    load_model(agent, model_name)

    history = load_history(model_name)
    losses = []
    
    for item in training_data:
        # Train network
        loss = agent.train_step(
            state=item["state"],
            targets=item["scores"]
        )
        losses.append(loss)
        
        # Save to JSON memory
        video_id = item["video"].get("url") or item["video"].get("title")
        history[video_id] = item["scores"]

    save_history(model_name, history)

    save_model(
        agent,
        model_name,
        criteria,
        query,
        engine=engine,
        hidden_layers=hidden_layers,
        input_size=input_size,
        output_size=output_size,
    )

    round_count = increment_round(model_name)
    export_approved_videos(model_name, training_data)
    delete_training_backup(model_name)

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return {
        "status": "success",
        "round_count": round_count,
        "trained_count": len(training_data),
        "average_loss": avg_loss
    }


# =====================================================================
# TERMINAL CLI INTERFACE
# =====================================================================

def get_score(criterion):
    while True:
        try:
            value = input(f"{criterion} (0.0 (NO) - 1.0 (YES), blank = UNKNOWN, I = INTERRUPT): ").strip()
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
            print("Enter a value between 0.0 and 1.0.")
        except ValueError:
            print("Invalid value.")

def display_table(training_data, criteria):
    print("\n" + "=" * 120)
    print("TRAINING REVIEW")
    print("=" * 120)
    print(f"{'#':<4}{'Title':<55}", end="")
    for criterion in criteria:
        print(f"{criterion:<20}", end="")
    print("\n" + "-" * 120)

    for index, item in enumerate(training_data, start=1):
        title = item["video"]["title"]
        if len(title) > 52:
            title = title[:49] + "..."
        print(f"{index:<4}{title:<55}", end="")
        for score in item["scores"]:
            print(f"{score:<20.2f}", end="")
        print()
    print("=" * 120)

def edit_choice(training_data, criteria):
    if not training_data:
        print("\nThere are no choices to edit.")
        return
    choice = input("\nSelect video number: ").strip()
    try:
        index = int(choice) - 1
        if index < 0 or index >= len(training_data):
            raise ValueError
    except ValueError:
        print("Invalid video.")
        return

    item = training_data[index]
    print(f"\nTitle: {item['video']['title']}\nCurrent scores:")
    for criterion_index, criterion in enumerate(criteria, start=1):
        score = item["scores"][criterion_index - 1]
        print(f"{criterion_index}. {criterion}: {score:.2f}")

    criterion_choice = input("\nSelect criterion: ").strip()
    try:
        criterion_index = int(criterion_choice) - 1
        if criterion_index < 0 or criterion_index >= len(criteria):
            raise ValueError
    except ValueError:
        print("Invalid criterion.")
        return

    item["scores"][criterion_index] = get_score(criteria[criterion_index])
    print("\nChoice updated.")

def clear_choices(training_data):
    confirm = input("\nClear ALL training choices? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return False
    training_data.clear()
    print("\nAll choices cleared.")
    return True

def review_menu(training_data, criteria):
    while True:
        display_table(training_data, criteria)
        print("1. Edit choice")
        print("2. Clear all choices")
        print("3. Confirm and train")
        print("4. Cancel\n")
        choice = input("Select: ").strip()
        if choice == "1":
            edit_choice(training_data, criteria)
        elif choice == "2":
            clear_choices(training_data)
        elif choice == "3":
            if not training_data:
                print("\nThere are no choices to train.")
                continue
            return "train"
        elif choice == "4":
            return "cancel"
        else:
            print("\nInvalid option.")

def show_prediction(agent, criteria, state):
    predictions = agent.predict(state)
    print("\nSearchAI prediction:")
    for criterion, prediction in zip(criteria, predictions):
        print(f"  {criterion}: {prediction:.4f}")

def train(model_name):
    print(f"\nTraining model: {model_name}")
    current_round = get_round_count(model_name)
    print(f"Training round: {current_round + 1}")

    criteria = load_criteria(model_name)
    if criteria is None:
        print("Could not load model criteria.")
        return

    query = load_query(model_name)
    if not query:
        print("Could not load the model's permanent query.")
        return

    metadata = load_metadata(model_name)
    if metadata is None:
        print("Could not load model metadata.")
        return

    engine = metadata.get("engine", "yt")
    input_size = metadata.get("input_size")
    hidden_layers = metadata.get("hidden_layers")
    output_size = metadata.get("output_size")

    if not isinstance(input_size, int) or input_size <= 0:
        print("Invalid model input size.")
        return

    if not isinstance(hidden_layers, list) or not hidden_layers:
        print("Invalid model hidden-layer configuration.")
        return

    if not all(isinstance(size, int) and size > 0 for size in hidden_layers):
        print("Invalid model hidden-layer configuration.")
        return

    if not isinstance(output_size, int) or output_size <= 0:
        print("Invalid model output size.")
        return

    if output_size != len(criteria):
        print("\nModel architecture is inconsistent with its criteria.")
        return

    agent = DQNAgent(
        state_size=input_size,
        criterion_count=output_size,
        hidden_layers=hidden_layers,
    )

    if not load_model(agent, model_name):
        print("Could not load model weights.")
        return

    history = load_history(model_name)
    backup = load_training_backup(model_name)
    training_data = []
    videos = []
    start_video_index = 0
    start_criterion_index = 0

    if backup:
        print("\nA previous training session was interrupted.\n1. Resume\n2. Start new")
        choice = input("\nSelect: ").strip()
        if choice == "1":
            training_data = backup.get("training_data", [])
            videos = backup.get("videos", [])
            start_video_index = backup.get("video_index", 0)
            start_criterion_index = backup.get("criterion_index", 0)
        elif choice == "2":
            delete_training_backup(model_name)
        else:
            return

    # --- CLI MODE SELECTION ---
    print("\nSelect training mode:")
    print("1. Manual (Grade all videos)")
    print("2. Auto (Use JSON memory for familiar videos)")
    print("3. Abort")
    
    auto_mode = False
    while True:
        mode = input("\nSelect: ").strip()
        if mode == "1":
            auto_mode = False
            break
        elif mode == "2":
            auto_mode = True
            break
        elif mode == "3":
            print("\nTraining aborted.")
            return
        else:
            print("Invalid choice.")

    if not videos:
        if engine == "google":
            try:
                google_results = search_google(query, max_results=10)
            except GoogleBlockedError as e:
                print(f"\nGoogle search error: {e}")
                return
            videos = [{
                "title": r.get("title", ""),
                "channel": "",
                "url": r.get("url", ""),
                "description": r.get("description", ""),
            } for r in google_results]
        else:
            videos = search_youtube(query, max_results=10)

        if not videos:
            print("\nNo videos found.")
            return

        save_training_backup(
            model_name=model_name,
            query=query,
            criteria=criteria,
            videos=videos,
            training_data=training_data,
            video_index=0,
            criterion_index=0,
        )

    try:
        for index in range(start_video_index, len(videos)):
            video = videos[index]
            print(f"\n{'='*60}\nVIDEO {index + 1}/{len(videos)}\n{'='*60}")
            print(f"Title:   {video['title']}\nChannel: {video['channel']}\nURL:     {video['url']}")

            state = encode_title_and_channel(video["title"], video["channel"], state_size=input_size)
            show_prediction(agent, criteria, state)

            if index < len(training_data):
                item = training_data[index]
            else:
                item = {
                    "video": video,
                    "scores": [],
                    "state": state.tolist() if hasattr(state, "tolist") else state,
                }
                training_data.append(item)

            # --- AUTO-FILL BYPASS ---
            video_id = video.get("url") or video.get("title")
            if auto_mode and video_id in history and len(history[video_id]) == len(criteria):
                print("\n[✓] Familiar video detected. Auto-filling scores from JSON Memory.")
                item["scores"] = history[video_id].copy()
                continue
            # ------------------------

            criterion_start = start_criterion_index if index == start_video_index else 0

            for criterion_index in range(criterion_start, len(criteria)):
                criterion = criteria[criterion_index]
                save_training_backup(
                    model_name=model_name,
                    query=query,
                    criteria=criteria,
                    videos=videos,
                    training_data=training_data,
                    video_index=index,
                    criterion_index=criterion_index,
                )

                score = get_score(criterion)
                if criterion_index < len(item["scores"]):
                    item["scores"][criterion_index] = score
                else:
                    item["scores"].append(score)

                save_training_backup(
                    model_name=model_name,
                    query=query,
                    criteria=criteria,
                    videos=videos,
                    training_data=training_data,
                    video_index=index,
                    criterion_index=criterion_index + 1,
                )

            start_criterion_index = 0

        delete_training_backup(model_name)

    except TrainingInterrupted:
        print("\nTraining interrupted. Session saved.")
        return

    result = review_menu(training_data, criteria)
    if result == "cancel":
        print("\nTraining cancelled.")
        return

    # Train and Save History
    losses = []
    for item in training_data:
        loss = agent.train_step(state=item["state"], targets=item["scores"])
        losses.append(loss)
        
        # Save final assigned scores to history
        video_id = item["video"].get("url") or item["video"].get("title")
        history[video_id] = item["scores"]

    save_history(model_name, history)

    save_model(
        agent,
        model_name,
        criteria,
        query,
        engine=engine,
        hidden_layers=hidden_layers,
        input_size=input_size,
        output_size=output_size,
    )

    round_count = increment_round(model_name)
    export_approved_videos(model_name, training_data)
    print(f"\nTraining complete. Round: {round_count}, Avg Loss: {sum(losses)/len(losses):.6f}")