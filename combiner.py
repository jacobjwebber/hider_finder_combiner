import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import hydra
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

import dsp
from common_nn import ParallelTransposedConvolutionalBlock, LinearNorm
from new_new_dataset import HFCDataModule

class CombinerAutoEncoder(pl.LightningModule):
    def __init__(self, hp, n_speakers):
        super(CombinerAutoEncoder, self).__init__()
        self.lr = hp.combiner.lr
        self.combiner = Combiner(hp, n_speakers)
        self.hp = hp
        self.n_speakers = n_speakers
        self.f0_bins = hp.control_variables.f0_bins
    
    def training_step(self, batch):
        mel, f0, vuv, speaker_id, spkr_emb = batch
        #mel = mel.float()
        f0_idx = dsp.bin_tensor(f0, self.f0_bins, self.hp.control_variables.f0_min, self.hp.control_variables.f0_max)
        is_voiced = vuv.float()
        mel = F.dropout(mel, p=0.2)
        out = self.combiner((mel, speaker_id, spkr_emb, f0_idx, is_voiced))
        loss = F.mse_loss(out, mel)
        self.log('loss', loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'loss'}}


class ConvCombiner(nn.Module):
    def __init__(self, hp, n_speakers):
        super(Combiner, self).__init__()
        pass

    def forward(self, x):
        pass

class TransformerCombiner(nn.Module):
    def __init__(self, hp, n_speakers):
        super(TransformerCombiner, self).__init__()
        pass

    def forward(self, x):
        pass

class Combiner(nn.Module):
    def __init__(self, hp, n_speakers):
        #hidden_size, spectral_width, output_width, n_layers, num_bins, dropout=0.1, rnn_mult=3.):
        super(Combiner, self).__init__()
        # Insert some random noise into the net -- set the width here
        # Replacing noise with f0, TODO fix
        self.noise_width = 513
        self.trans_size = 1 * hp.num_mels
        self.use_f0 = hp.use_f0

        f0_dim = hp.control_variables.f0_bins
        self.f0_dims = f0_dim
        self.speaker_emb_dim = hp.control_variables.speaker_embedding_dim

        self.f0_embedding = nn.Embedding(f0_dim, f0_dim)
        #self.speaker_embedding = nn.Embedding(n_speakers, speaker_emb_dim) # TODO replace with real speaker embedding?
        self.dropout = nn.Dropout(hp.combiner.drop)

        if self.use_f0:
            self.f0_PTCB = ParallelTransposedConvolutionalBlock(f0_dim, self.trans_size)
            self.merge_voicing = nn.Linear(2 * self.trans_size, self.trans_size)

        self.control_variable_PTCB = ParallelTransposedConvolutionalBlock(self.speaker_emb_dim, self.trans_size)
        self.spectral_PTCB = ParallelTransposedConvolutionalBlock(hp.model.hidden_size, self.trans_size)

        self.aperiodicity_lin = nn.Linear(self.noise_width, self.trans_size)
        self.control_variable_lin = nn.Linear(self.speaker_emb_dim, self.trans_size)
        self.spectral_residual = nn.Linear(hp.model.hidden_size, self.trans_size)

        self.rnn = nn.LSTM(
            hp.num_mels + self.speaker_emb_dim, 
            hp.combiner.rnn_size,
            hp.combiner.n_layers, 
            dropout=hp.combiner.drop, 
            batch_first=True, 
            bidirectional=True,
        )

        self.lin = LinearNorm(2*hp.combiner.rnn_size, hp.num_mels)

    def forward(self, features):
        spectral, speaker_id, spkr_emb, f0, is_voiced = features

        batch_size, seq_length, spectral_width = spectral.shape

        # Using f0 onehot instead of embedding. TODO use embedding instead
        #speaker_id = self.speaker_embedding(speaker_id)
        speaker_id = spkr_emb.squeeze(1)
        speaker_id = speaker_id.repeat(1, seq_length, 1)

        #speaker_id = self.control_variable_PTCB(speaker_id)
        #spectral = self.spectral_PTCB(spectral)
        x = torch.cat((spectral, speaker_id), 2)
        out, _ = self.rnn(x)

        # Apply linear layer to RNN output
        output = self.lin(out)

        return output



    

@hydra.main(version_base=None, config_path='config', config_name="config")
def train(config):
    # Trains a finder on ground truth
    dm = HFCDataModule(config)
    n_speakers = dm.n_speakers

    finder = CombinerAutoEncoder(config, n_speakers)

    if torch.cuda.device_count() > 1:
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'auto'

    if config.training.wandb:
        logger = pl.loggers.wandb.WandbLogger(project="hfc_combiner")
    else:
        logger = None
    trainer = pl.Trainer(logger=logger, gradient_clip_val=0.5, detect_anomaly=True,
                         strategy=strategy,
    )
    # Create a Tuner
    tuner = Tuner(trainer)

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    tuner.lr_find(finder, datamodule=dm)
    trainer.fit(finder, datamodule=dm,)


if __name__ == '__main__':
    train()
