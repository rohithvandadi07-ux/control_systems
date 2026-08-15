import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """
    Basic Temporal Convolutional Network block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.2,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv(x)

        # Remove extra right-side samples introduced by
        # causal-style padding.
        out = out[:, :, :x.size(2)]

        out = self.relu(out)
        out = self.dropout(out)

        return out + residual


class TCNLSTM(nn.Module):
    """
    Hybrid TCN-LSTM model for battery SOC estimation.

    Input:
        [batch, sequence_length, 3]

    Features:
        voltage
        current
        temperature

    Output:
        SOC value
    """

    def __init__(
        self,
        input_size=3,
        tcn_channels=32,
        lstm_hidden_size=64,
        lstm_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        # TCN expects:
        # [batch, channels, sequence]
        self.tcn = nn.Sequential(
            TCNBlock(
                in_channels=input_size,
                out_channels=tcn_channels,
                kernel_size=3,
                dilation=1,
                dropout=dropout,
            ),
            TCNBlock(
                in_channels=tcn_channels,
                out_channels=tcn_channels,
                kernel_size=3,
                dilation=2,
                dropout=dropout,
            ),
            TCNBlock(
                in_channels=tcn_channels,
                out_channels=tcn_channels,
                kernel_size=3,
                dilation=4,
                dropout=dropout,
            ),
        )

        # LSTM expects:
        # [batch, sequence, features]
        self.lstm = nn.LSTM(
            input_size=tcn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                lstm_hidden_size,
                32,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                32,
                1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x shape:
            [batch, sequence_length, 3]

        returns:
            [batch]
        """

        # Convert:
        # [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)

        # TCN
        x = self.tcn(x)

        # Convert:
        # [B, F, T] -> [B, T, F]
        x = x.transpose(1, 2)

        # LSTM
        x, _ = self.lstm(x)

        # Use the final time step
        x = x[:, -1, :]

        # SOC regression
        x = self.regressor(x)

        return x.squeeze(-1)


if __name__ == "__main__":
    print("=" * 60)
    print("TCN-LSTM MODEL TEST")
    print("=" * 60)

    model = TCNLSTM()

    print(model)

    # Dummy input:
    # 8 samples
    # 100 time steps
    # 3 features
    x = torch.randn(8, 100, 3)

    with torch.no_grad():
        y = model(x)

    print()
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    print()
    print("Output values:")
    print(y)

    print()
    print("Total parameters:",
          sum(p.numel() for p in model.parameters()))

    print("=" * 60)

