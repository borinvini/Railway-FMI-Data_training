"""FCLL-Net: Huang et al. (2020), Transportation Research Part E 141, 102022.

Three branches: an FCNN over static non-operational features, an LSTM over the
H trains preceding the target train, and an LSTM over the Z stations preceding
the target station. Concatenated, then a final FCNN to one logit.

Deviations from the paper are deliberate and recorded in the design doc:
PyTorch rather than Keras, a binary logit rather than a delay in minutes, and
smaller default capacity because our sample count is 5.6x lower than theirs.
"""
import torch
import torch.nn as nn


class FCLLNet(nn.Module):
    def __init__(self, n_of, n_nf, h=5, z=5, lstm_hidden=64, lstm_layers=2,
                 fcnn_hidden=64, fcnn_layers=3, dropout=0.2):
        super().__init__()

        layers, width = [], max(n_nf, 1)
        for _ in range(fcnn_layers):
            layers += [nn.Linear(width, fcnn_hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = fcnn_hidden
        self.fcnn = nn.Sequential(*layers)

        self.lstm_trains = nn.LSTM(n_of, lstm_hidden, num_layers=lstm_layers,
                                   batch_first=True,
                                   dropout=dropout if lstm_layers > 1 else 0.0)
        self.lstm_stations = nn.LSTM(n_of, lstm_hidden, num_layers=lstm_layers,
                                     batch_first=True,
                                     dropout=dropout if lstm_layers > 1 else 0.0)

        self.head = nn.Sequential(
            nn.Linear(fcnn_hidden + 2 * lstm_hidden, fcnn_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(fcnn_hidden, 1),
        )

    @staticmethod
    def _last_valid(output, mask):
        """Take each sequence's last unmasked step.

        The index is the last position the mask marks real, found via cumsum so
        it is correct whichever end the padding sits at. Indexing by length
        instead would be right only for post-padding; assemble_sequences
        pre-pads, so it would read a zero row and throw away the step nearest
        the target — the most informative one in the sequence.

        A sequence with no history at all (an originating train at the target
        station) falls back to zeros, which the head can learn to read as
        'no upstream information'.
        """
        lengths = mask.sum(dim=1)
        idx = mask.cumsum(dim=1).argmax(dim=1)
        gathered = output[torch.arange(output.size(0), device=output.device), idx]
        return gathered * (lengths > 0).unsqueeze(-1).float()

    def forward(self, z_seq, h_seq, z_mask, h_mask, nf):
        if nf.size(1):
            nf_out = self.fcnn(nf)
        else:
            # No non-operational features in this scenario. The branch still runs
            # on a constant so the head keeps a fixed input width and the two
            # LSTM branches are unaffected.
            nf_out = self.fcnn(torch.zeros(nf.size(0), 1, device=nf.device))
        h_out = self._last_valid(self.lstm_trains(h_seq * h_mask.unsqueeze(-1))[0], h_mask)
        z_out = self._last_valid(self.lstm_stations(z_seq * z_mask.unsqueeze(-1))[0], z_mask)
        return self.head(torch.cat([nf_out, h_out, z_out], dim=1)).squeeze(-1)
