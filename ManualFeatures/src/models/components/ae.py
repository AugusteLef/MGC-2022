from collections import OrderedDict

from torch import nn


layer_sizes = {
    10: [518, 400, 200, 100, 50, 10],
    50: [518, 400, 200, 100, 50],
    100: [518, 400, 200, 100],
    200: [518, 400, 200],
    400: [518, 400],
    518: []
}


class AE(nn.Module):
    def __init__(self, latent_space_dim: int):
        super().__init__()

        assert latent_space_dim in layer_sizes, f'invalid latent space dimensionality: {latent_space_dim}'
        self.latent_space_dim = latent_space_dim
        sizes = layer_sizes[latent_space_dim]

        if sizes:
            layers_encode = []
            sizes_encode = sizes
            for i, hs in enumerate(sizes_encode[1:-1]):
                layers_encode.extend([
                    nn.Linear(sizes_encode[i], hs),
                    nn.BatchNorm1d(hs),
                    nn.ReLU()
                ])
            layers_encode.append(nn.Linear(sizes_encode[-2], sizes_encode[-1]))
            self.model_encode = nn.Sequential(*layers_encode)

            layers_decode = []
            sizes_decode = sizes[::-1]
            for i, hs in enumerate(sizes_decode[1:-1]):
                layers_decode.extend([
                    nn.Linear(sizes_decode[i], hs),
                    nn.BatchNorm1d(hs),
                    nn.ReLU()
                ])
            layers_decode.append(nn.Linear(sizes_decode[-2], sizes_decode[-1]))
            self.model_decode = nn.Sequential(*layers_decode)
        else:
            self.model_encode = None
            self.model_decode = None

    def forward(self, x):
        if self.latent_space_dim < 518:
            z = self.model_encode(x)
            x_hat = self.model_decode(z)

            return z, x_hat
        else:
            return x, x


if __name__ == "__main__":
    _ = AE(latent_space_dim=10)
