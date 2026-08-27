import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):

    def __init__(
        self,
        state_size,
        criterion_count,
        hidden_layers
    ):
        super().__init__()

        layers = []

        input_size = state_size

        for hidden_size in hidden_layers:

            layers.append(
                nn.Linear(
                    input_size,
                    hidden_size
                )
            )

            layers.append(
                nn.ReLU()
            )

            input_size = hidden_size

        layers.append(
            nn.Linear(
                input_size,
                criterion_count
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:

    def __init__(
        self,
        state_size,
        criterion_count,
        hidden_layers,
        learning_rate=0.001
    ):
        if not hidden_layers:
            raise ValueError(
                "At least one hidden layer "
                "is required."
            )

        if not isinstance(
            hidden_layers,
            (list, tuple)
        ):
            raise TypeError(
                "hidden_layers must be "
                "a list or tuple."
            )

        if any(
            not isinstance(size, int)
            or size <= 0
            for size in hidden_layers
        ):
            raise ValueError(
                "Every hidden layer size "
                "must be a positive integer."
            )

        if state_size <= 0:
            raise ValueError(
                "state_size must be positive."
            )

        if criterion_count <= 0:
            raise ValueError(
                "criterion_count must be positive."
            )

        self.state_size = state_size

        self.criterion_count = (
            criterion_count
        )

        self.hidden_layers = list(
            hidden_layers
        )

        self.learning_rate = learning_rate

        self.model = DQN(
            state_size=state_size,
            criterion_count=criterion_count,
            hidden_layers=self.hidden_layers
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()

    def predict(self, state):

        if not torch.is_tensor(state):

            state = torch.tensor(
                state,
                dtype=torch.float32
            )

        with torch.no_grad():

            output = self.model(
                state
            )

        return output.tolist()

    def train_step(
        self,
        state,
        targets
    ):

        if not torch.is_tensor(state):

            state = torch.tensor(
                state,
                dtype=torch.float32
            )

        if not torch.is_tensor(targets):

            targets = torch.tensor(
                targets,
                dtype=torch.float32
            )

        prediction = self.model(
            state
        )

        loss = self.loss_function(
            prediction,
            targets
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()