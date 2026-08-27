def create_network_architecture(
    state_size,
    output_size
):
    print()
    print("=" * 60)
    print("NETWORK ARCHITECTURE")
    print("=" * 60)

    print(
        f"\nInput nodes: {state_size}"
    )

    print(
        f"Output nodes: {output_size}"
    )

    print(
        "\nEnter the hidden-layer sizes."
    )

    print(
        "Example:"
    )

    print(
        "  8 4 4 2"
    )

    print(
        "  64 32 16"
    )

    print(
        "  128 64 32 16 8"
    )

    print(
        "\nEnter an empty line when finished."
    )

    hidden_layers = []

    while True:

        value = input(
            f"\nHidden layer "
            f"{len(hidden_layers) + 1}: "
        ).strip()

        if not value:
            break

        try:
            size = int(value)

        except ValueError:
            print(
                "Enter a whole number."
            )
            continue

        if size <= 0:
            print(
                "Layer size must be greater "
                "than 0."
            )
            continue

        hidden_layers.append(size)

    if not hidden_layers:
        print(
            "\nNo hidden layers entered."
        )

        print(
            "Using default architecture: "
            "64 32 16"
        )

        hidden_layers = [
            64,
            32,
            16
        ]

    print()
    print("Architecture:")
    print(
        " -> ".join(
            [
                str(state_size),
                *[
                    str(size)
                    for size in hidden_layers
                ],
                str(output_size)
            ]
        )
    )

    return hidden_layers