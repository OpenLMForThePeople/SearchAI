import torch
import torch.nn as nn


class CriterionBrain(nn.Module):
    """
    One completely independent brain for one criterion.
    """

    def __init__(
        self,
        input_size,
        hidden_size=128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.network(state)


class SearchAIModel(nn.Module):
    """
    Contains one independent brain per criterion.
    """

    def __init__(
        self,
        state_size,
        criterion_count,
        hidden_size=128,
    ):
        super().__init__()

        self.brains = nn.ModuleList(
            [
                CriterionBrain(
                    state_size,
                    hidden_size,
                )
                for _ in range(criterion_count)
            ]
        )

    def forward(self, state):
        return [
            brain(state)
            for brain in self.brains
        ]