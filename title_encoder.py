import hashlib
import re


DEFAULT_STATE_SIZE = 256


def _hash_feature(
    feature,
    state_size
):
    digest = hashlib.sha256(
        feature.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little"
    ) % state_size


def _add_text_features(
    state,
    text,
    prefix,
    state_size
):
    text = str(text).lower()

    words = re.findall(
        r"[a-z0-9]+",
        text
    )

    for word in words:

        index = _hash_feature(
            f"{prefix}:word:{word}",
            state_size
        )

        state[index] += 1.0

    # Character n-grams
    for word in words:

        if len(word) < 3:
            continue

        for size in (3, 4, 5):

            if len(word) < size:
                continue

            for i in range(
                len(word) - size + 1
            ):

                ngram = word[
                    i:i + size
                ]

                index = _hash_feature(
                    f"{prefix}:ngram:{ngram}",
                    state_size
                )

                state[index] += 0.5


def encode_title_and_channel(
    title,
    channel,
    state_size=DEFAULT_STATE_SIZE
):
    """
    Encode both the video title and channel
    into a user-defined model input state.

    Args:
        title:
            Video title.

        channel:
            YouTube channel name.

        state_size:
            Number of input nodes/features.

    Returns:
        list[float] of length state_size
    """

    if not isinstance(
        state_size,
        int
    ) or state_size <= 0:

        raise ValueError(
            "state_size must be a "
            "positive integer."
        )

    state = [
        0.0
    ] * state_size

    _add_text_features(
        state,
        title,
        "title",
        state_size
    )

    _add_text_features(
        state,
        channel,
        "channel",
        state_size
    )

    magnitude = sum(
        value * value
        for value in state
    ) ** 0.5

    if magnitude > 0:

        state = [
            value / magnitude
            for value in state
        ]

    return state