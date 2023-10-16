# Jacob Webber 2023
# Finder class for hider finder combiner archetecture and training code for training on GT

import lightning.pytorch as pl
import torch
import torch.optim as optim
import torch.nn as nn

import hydra

class Finder(pl.LightningModule):
    def __init__(self, hparams):
        if hparams.model.finder == 'rnn':
            self.model = RNNFinder(hparams.model.hidden_size, hparams.dataset.mel.n_mel_channels, 
                      hparams.dataset.mel.n_mel_channels, hparams.finder.n_layers)
        elif hparams.model.finder == 'transformer':
            raise NotImplementedError


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class TransformerFinder(nn.Module):
    def __init__(self):
        pass


class RNNFinder(nn.Module):
    def __init__(self, hidden_size, input_width, output_width, n_layers, dropout=0.1):
        """
        Network takes an input sequence of variable length but fixed input_width
        Returns sequence of probability dists of width num_bins.
        """
        super(Finder, self).__init__()
        # multiply dropout by this number when in rnn
        rnn_mult = 3
        self.rnn = nn.GRU(input_width, hidden_size, n_layers, dropout=rnn_mult * dropout, batch_first=True)
        self.lin = nn.Linear(hidden_size, output_width)
        self.f0_projector = nn.Linear()

    def forward(self, spectrograms):
        out, hidden = self.rnn(spectrograms)
        out = self.lin(out)
        out_global = torch.mean(out, dim=1) # use mean to reduce time dimensionality
        return out


@hydra.main(version_base=None, config_path='config', config_name="config")
def train():
    import fastspeech2_dataset
    # Trains a finder on ground truth
    ds = fastspeech2_dataset.Dataset()