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
import torchmetrics
import dsp

import hydra

from new_new_dataset import HFCDataModule
from duta_vc.encoder import MelEncoder, sequence_mask
from duta_vc.postnet import PostNet


class Finder(pl.LightningModule):
    def __init__(self, hparams, n_speakers):
        super().__init__()
        #self.hparams = hparams
        self.save_hyperparameters(hparams)
        self.lr = hparams.training_finder.lr
        if hparams.finder.name == 'rnn':
            self.model = RNNFinder(hparams.finder.interior_size,
                                   hparams.num_mels,
                                   hparams.control_variables.f0_bins,
                                   n_speakers,
                                   hparams.finder.rnn.n_layers)
        elif hparams.finder.name == 'transformer':
            raise NotImplementedError
        elif hparams.finder.name == 'duta_vc':
            self.model = DuTaFinder(hparams.num_mels,
                                    hparams.finder.channels,
                                    hparams.finder.filters,
                                    hparams.finder.heads,
                                    hparams.finder.layers,
                                    hparams.finder.kernel, 
                                    hparams.finder.dropout, 
                                    hparams.finder.window_size, 
                                    hparams.finder.enc_dim,
                                    n_speakers, 
                                    hparams.control_variables.f0_bins)
        else:
            raise NotImplementedError

        self.speaker_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_speakers)
        self.f0_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=hparams.control_variables.f0_bins)
        self.f0_min = hparams.control_variables.f0_min
        self.f0_max = hparams.control_variables.f0_max
        self.f0_bins = hparams.control_variables.f0_bins
    
    def forward(self, mel):
        return self.model(mel)


    def training_step(self, batch):

        mel, f0, vuv, speaker_id = batch
        f0_prediction, speaker_id_prediction = self.model(mel)

        speaker_id_loss = F.cross_entropy(
            speaker_id_prediction, speaker_id)
        
        self.log('speaker_id', speaker_id_loss, prog_bar=True)

        f0 = dsp.bin_tensor(f0, self.f0_bins, self.f0_min, self.f0_max).squeeze(-1)
        f0_prediction = f0_prediction.transpose(1, 2)
        f0_loss = F.cross_entropy(
            f0_prediction, 
            f0,
        )
        self.log('f0_loss', f0_loss, prog_bar=True)

        loss = speaker_id_loss + f0_loss
        self.log('loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mel, f0, vuv, speaker_id = batch
        f0_prediction, speaker_id_prediction = self.model(mel)

        speaker_id_loss = F.cross_entropy(
            speaker_id_prediction, speaker_id)
        
        speaker_id_accuracy = self.speaker_accuracy(
            speaker_id_prediction, speaker_id
        )
        self.log('val/speaker_id', speaker_id_loss, prog_bar=True)
        self.log('val/speaker_id_accuracy', speaker_id_accuracy, prog_bar=True)

        f0 = dsp.bin_tensor(f0, self.f0_bins, self.f0_min, self.f0_max).squeeze(-1)
        f0_prediction = f0_prediction.transpose(1, 2)
        f0_loss = F.cross_entropy(
            f0_prediction, 
            f0,
        )
        self.log('val/f0_loss', f0_loss, prog_bar=True)

        loss = speaker_id_loss + f0_loss
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/f0_accuracy', self.f0_accuracy(f0_prediction, f0), prog_bar=True, on_step=True, on_epoch=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'loss'}}


class TransformerFinder(nn.Module):
    def __init__(self):
        pass

# "average voice" encoder as the module parameterizing the diffusion prior
class DuTaFinder(torch.nn.Module):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, dim, n_speakers, f0_bins):
        super(DuTaFinder, self).__init__()
        self.encoder = MelEncoder(n_feats, channels, filters, heads, layers, 
                                  kernel, dropout, window_size)
        self.postnet = PostNet(dim)
        self.f0_lin = nn.Linear(n_feats, f0_bins)
        self.f0_projector = nn.Linear(n_feats, 100)
        self.speaker_id_lin = nn.Linear(n_feats, n_speakers)

    def forward(self, x):
        # x: [batch, n_mels, n_frames]
        x = x.transpose(1,2)
        mask = sequence_mask(torch.Tensor([batch.shape[1] for batch in x])).unsqueeze(1).to(x.dtype).to(x.device)
        #print(batch, n_mels, n_frames)
        #mask = torch.ones(batch, 1, n_frames, device=x.device)
        z = self.encoder(x, mask)
        out = self.postnet(z, mask).transpose(1,2)

        f0_prediction = self.f0_lin(out)
        # use mean to reduce time dimensionality
        speaker_prediction = self.speaker_id_lin(torch.mean(out, dim=1))
        return f0_prediction, speaker_prediction


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
        speaker_prediction = self.speaker_id_lin(torch.mean(out, dim=1, keepdim=True))
        return f0_prediction, speaker_prediction


@hydra.main(version_base=None, config_path='config', config_name="config")
def train(config):
    # Trains a finder on ground truth
    dm = HFCDataModule(config, 'finder')
    n_speakers = dm.n_speakers

    finder = Finder(config, n_speakers)

    if config.training.wandb:
        logger = pl.loggers.wandb.WandbLogger(project="hfc_finder")
    else:
        logger = None
    trainer = pl.Trainer(logger=logger, gradient_clip_val=0.5, detect_anomaly=True,
                         strategy='ddp_find_unused_parameters_true',
    )
    # Create a Tuner
    #tuner = Tuner(trainer)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    #tuner.lr_find(finder, datamodule=dm)
    trainer.fit(finder, datamodule=dm,)


if __name__ == '__main__':
    train()
