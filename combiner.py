import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import hydra
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

import dsp
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
        mel = mel.float().transpose(1,2)
        f0_idx = dsp.bin_tensor(f0, self.f0_bins, self.hp.control_variables.f0_min, self.hp.control_variables.f0_max)
        is_voiced = vuv.float().unsqueeze(2)
        out = self.combiner((mel, speaker_id, spkr_emb, f0_idx, is_voiced))
        loss = F.mse_loss(out, mel)
        self.log('loss', loss)
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
        self.hidden_size = hp.combiner.rnn_size
        self.n_layers = hp.combiner.n_layers
        self.trans_size = 3 * hp.num_mels
        self.use_f0 = hp.use_f0

        f0_dim = hp.control_variables.f0_bins
        self.f0_dims = f0_dim
        speaker_emb_dim = hp.control_variables.speaker_embedding_dim

        self.f0_embedding = nn.Embedding(f0_dim, f0_dim)
        #self.speaker_embedding = nn.Embedding(n_speakers, speaker_emb_dim) # TODO replace with real speaker embedding?
        self.dropout = nn.Dropout(hp.combiner.drop)

        if self.use_f0:
            self.f0_PTCB = ParallelTransposedConvolutionalBlock(f0_dim, self.trans_size)
            self.merge_voicing = nn.Linear(2 * self.trans_size, self.trans_size)

        self.control_variable_PTCB = ParallelTransposedConvolutionalBlock(speaker_emb_dim, self.trans_size)
        self.spectral_PTCB = ParallelTransposedConvolutionalBlock(hp.model.hidden_size, self.trans_size)

        self.aperiodicity_lin = nn.Linear(self.noise_width, self.trans_size)
        self.control_variable_lin = nn.Linear(speaker_emb_dim, self.trans_size)
        self.spectral_residual = nn.Linear(hp.model.hidden_size, self.trans_size)

        if self.use_f0:
            self.rnn = nn.GRU(4 * self.trans_size, self.hidden_size, hp.combiner.n_layers, dropout=3. * hp.combiner.drop, batch_first=True)
        else:
            self.rnn = nn.GRU(3 * self.trans_size, self.hidden_size, hp.combiner.n_layers, dropout=3. * hp.combiner.drop, batch_first=True)
        self.lin = nn.Linear(self.hidden_size, hp.num_mels)

    def forward(self, features):
        spectral, speaker_id, spkr_emb, f0, is_voiced = features

        batch_size = spectral.shape[0]
        seq_length = spectral.shape[1]
        spectral_width = spectral.shape[2]

        #print(f'seq_length = {seq_length}')

        # Using f0 onehot instead of embedding. TODO use embedding instead
        #print(f0.max())
        #speaker_id = self.speaker_embedding(speaker_id)
        speaker_id = spkr_emb.squeeze(1)
        speaker_id = speaker_id.repeat(1, seq_length, 1)

        noise = torch.rand((batch_size, seq_length, self.noise_width), device=spectral.device)

        noise = self.dropout(F.relu(self.aperiodicity_lin(noise)))

        speaker_id = self.control_variable_PTCB(speaker_id)

        spectral = self.spectral_PTCB(spectral)
        if self.use_f0:
            #f0 = self.f0_embedding(f0)
            f0 = nn.functional.one_hot(f0, self.f0_dims).float()
            f0 = self.f0_PTCB(f0)
            unvoiced_f0 = is_voiced * f0
            f0 = self.merge_voicing(torch.cat((unvoiced_f0, f0), 2))

        # SECTION: Combine F0 and spectral features
        if self.use_f0:
            x = torch.cat((spectral, f0, speaker_id, noise), 2)
        else:
            x = torch.cat((spectral, speaker_id, noise), 2)
        out, _ = self.rnn(x)

        # Apply linear layer to RNN output
        output = self.lin(out)

        return output



class ParallelTransposedConvolutionalBlock(nn.Module):
    """A block that applies a block of transposed convolutions in parallel and sums the result"""

    def __init__(self, input_width, output_width):
        super(ParallelTransposedConvolutionalBlock, self).__init__()

        self.kernel_size = 50
        padding = 45
        out_padding = 0
        stride = 1

        self.residual = nn.Sequential(
            nn.Linear(input_width, output_width),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.parallel_components = nn.ModuleList()

        for dilation in range(1, 20, 2):
            conv = nn.ConvTranspose1d(1, 1, self.kernel_size,
                                      stride=stride, padding=padding,
                                      output_padding=out_padding, groups=1, bias=True,
                                      dilation=dilation)
            # Docs do not mention dilation wrt output width
            # Everyone hates convolution maths
            conv_size = (input_width - 1) * stride - 2 * padding + self.kernel_size + out_padding + \
                        (self.kernel_size - 1) * (dilation - 1)
            resize = nn.Linear(conv_size, output_width)
            component = nn.Sequential(conv, resize, nn.ReLU())
            self.parallel_components.append(component)

    def forward(self, x):
        #print(x.shape)
        batch_size, seq_length, width = x.shape
        x = x.reshape(batch_size * seq_length, 1, -1)
        xs = [c(x) for c in self.parallel_components]
        output = sum(xs)
        output += self.residual(x)
        output = output.view(batch_size, seq_length, -1)
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
    #tuner.lr_find(finder, datamodule=dm)
    trainer.fit(finder, datamodule=dm,)


if __name__ == '__main__':
    train()
