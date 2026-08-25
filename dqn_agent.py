import torch
import torch.nn as nn

from model import SearchAIModel


class DQNAgent:
    def __init__(
        self,
        state_size,
        criterion_count,
        hidden_size=128,
        learning_rate=0.001,
    ):
        self.state_size = state_size
        self.criterion_count = criterion_count

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = SearchAIModel(
            state_size=state_size,
            criterion_count=criterion_count,
            hidden_size=hidden_size,
        ).to(self.device)

        # Each criterion gets its OWN optimizer.
        self.optimizers = [
            torch.optim.Adam(
                brain.parameters(),
                lr=learning_rate,
            )
            for brain in self.model.brains
        ]

        self.loss_fn = nn.MSELoss()

    def predict(self, state):
        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(state)

        scores = []

        for output in outputs:
            # The network outputs one value.
            score = output.squeeze().item()

            # Keep the prediction between 0 and 1.
            score = max(0.0, min(1.0, score))

            scores.append(score)

        return scores

    def train_step(
        self,
        state,
        targets,
    ):
        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        total_loss = 0.0

        for criterion_index in range(
            self.criterion_count
        ):
            brain = self.model.brains[
                criterion_index
            ]

            optimizer = self.optimizers[
                criterion_index
            ]

            target = torch.tensor(
                [targets[criterion_index]],
                dtype=torch.float32,
                device=self.device,
            )

            prediction = brain(
                state
            ).squeeze(1)

            loss = self.loss_fn(
                prediction,
                target,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        return total_loss


def big_brain_decision(scores):
    """
    Every criterion must approve the video.
    """

    for score in scores:
        if score < 0.5:
            return False

    return True