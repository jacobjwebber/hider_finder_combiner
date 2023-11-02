# Jacob Webber 2023
# Finder class for hider finder combiner archetecture and training code for training on GT
import os
import json

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import hydra

from fastspeech2_dataset import HFCDataModule


class Finder(pl.LightningModule):
    def __init__(self, hparams, n_speakers):
        super().__init__()
        self.lr = hparams.training.lr
        if hparams.finder.name == 'rnn':
            self.model = RNNFinder(hparams.finder.interior_size,
                                   hparams.dataset.preprocessing.mel.n_mel_channels,
                                   hparams.training.f0_bins,
                                   n_speakers,
                                   hparams.finder.rnn.n_layers)
        elif hparams.finder.name == 'transformer':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx):
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = batch

        f0_prediction, speaker_id_prediction = self.model(mels)

        speaker_id_loss = F.cross_entropy(
            speaker_id_prediction, speakers)
        
        self.log('speaker_id', speaker_id_loss, prog_bar=True)

        loss = speaker_id_loss + 0
        self.log('loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return torch.zeros(1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'loss'}}


class TransformerFinder(nn.Module):
    def __init__(self):
        pass


class RNNFinder(nn.Module):
    def __init__(self, interior_size, input_width, f0_bins, n_speakers, n_layers, dropout=0.1):
        """
        Network takes an input sequence of variable length but fixed input_width
        Returns sequence of probability dists of width num_bins.
        """
        super().__init__()
        # multiply dropout by this number when in rnn
        rnn_mult = 3
        self.rnn = nn.GRU(input_width, interior_size, n_layers,
                          dropout=rnn_mult * dropout, batch_first=True)
        self.f0_lin = nn.Linear(interior_size, f0_bins)
        self.f0_projector = nn.Linear(interior_size, 100)
        self.speaker_id_lin = nn.Linear(interior_size, n_speakers)

    def forward(self, spectrograms):
        out, hidden = self.rnn(spectrograms)
        f0_prediction = self.f0_lin(out)
        # use mean to reduce time dimensionality
        speaker_prediction = self.speaker_id_lin(torch.mean(out, dim=1))
        return f0_prediction, speaker_prediction


@hydra.main(version_base=None, config_path='config', config_name="config")
def train(config):
    # Trains a finder on ground truth
    dm = HFCDataModule(config)

    f = os.path.join(config.dataset.path.preprocessed_path, 'speakers.json')
    with open(f, 'r') as f:
        n_speakers = len(json.load(f))

    finder = Finder(config, n_speakers)

    trainer = pl.Trainer(gradient_clip_val=0.5, detect_anomaly=True)
    # Create a Tuner
    tuner = Tuner(trainer)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    tuner.lr_find(finder, datamodule=dm)
    trainer.fit(finder, datamodule=dm)


if __name__ == '__main__':
    train()
