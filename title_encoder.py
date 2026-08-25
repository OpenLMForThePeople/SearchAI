import hashlib
import re


TITLE_STATE_SIZE = 256


def _hash_feature(feature):
    digest = hashlib.sha256(
        feature.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little"
    ) % TITLE_STATE_SIZE


def encode_title(title):
    state = [0.0] * TITLE_STATE_SIZE

    title = title.lower()

    words = re.findall(
        r"[a-z0-9]+",
        title
    )

    for word in words:
        index = _hash_feature(
            f"word:{word}"
        )

        state[index] += 1.0

    # Character n-grams let related forms such as
    # crush / crushing / crushed share features.
    for word in words:
        if len(word) < 3:
            continue

        for size in (3, 4, 5):
            if len(word) < size:
                continue

            for i in range(
                len(word) - size + 1
            ):
                ngram = word[i:i + size]

                index = _hash_feature(
                    f"ngram:{ngram}"
                )

                state[index] += 0.5

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