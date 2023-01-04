from collections import OrderedDict

from torch import nn


hidden_layer_sizes = {
    10: [10],
    50: [30, 20],
    100: [70, 50, 20],
    200: [130, 60, 30],
    400: [250, 150, 50, 20],
    518: [300, 180, 80, 40]
}

class SimpleDenseNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()

        self.input_size = input_size
        hidden_sizes = hidden_layer_sizes[input_size]
        output_size = 16  # number of genres

        layers = [
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU()
        ]
        for i, hs in enumerate(hidden_sizes[1:]):
            layers.extend([
                nn.Linear(hidden_sizes[i], hs),
                nn.BatchNorm1d(hs),
                nn.ReLU()
            ])
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    _ = SimpleDenseNet(input_size=518)
