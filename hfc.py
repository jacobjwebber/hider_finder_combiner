import hydra

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# Local imports
import dsp
from hider import Hider
from finder import Finder
from combiner import Combiner
from new_new_dataset import HFCDataModule

class HFC(pl.LightningModule):
    """ An implementation of the Hider Finder Combiner architecture for
    arbitrary control of speech signals """
    def __init__(self, hp, n_speakers):
        super(HFC, self).__init__()

        self.hp = hp
        self.n_speakers = n_speakers
        self.f0_bins = hp.control_variables.f0_bins
        self.automatic_optimization=False # required for manual optimization scheduling in lightning

        # Component networks
        # TODO rename width/size dichotomy also input/output mean same thing for hfc
        # TODO pass hp to get_[finder,hider,combiner] method and let it sort everything
        self.hider = Hider(hp)
        self.finder = Finder(hp, n_speakers)
        self.combiner = Combiner(hp, n_speakers)

        # The 'combiner loss' criterion
        self.g_criterion = nn.MSELoss()
        # Ignore (for now) index associated with unvoiced regions
        self.f_criterion = nn.CrossEntropyLoss(ignore_index=0)

    def configure_optimizers(self):
        # Define params that generate during synthesis as 'generator' (g)
        self.generator_params = list(self.hider.parameters()) + list(self.combiner.parameters())
        self.g_opt = Adam(self.generator_params, lr=self.hp.training.lr_g, weight_decay=0.00)
        self.f_opt = Adam(self.finder.parameters(), lr=self.hp.training.lr_f, weight_decay=0.00)
        return [self.g_opt, self.f_opt]


    def set_input(self, mel, f0, is_voiced, speaker_id):

        # Find the shape of the onehots for the cv
        f0 = f0.squeeze()
        is_voiced = is_voiced.squeeze()
        batch_size, seq_len = f0.shape
        #print(f'f0 shape: {f0.shape}')
        #print(f'is_voiced shape: {is_voiced.shape}')
        #print(f'mel shape: {mel.shape}')
        #print(f'speaker_id shape: {speaker_id.shape}')

        # Two versions --- one matrix onehot and one array of indices
        #self.cv_index = dsp.onehot_index(control_variable, self.hp.cv_bins, self.hp.cv_range)
        self.speaker_id = speaker_id

        self.f0_idx = dsp.bin_tensor(f0, self.f0_bins, self.hp.control_variables.f0_min, self.hp.control_variables.f0_max)
        self.is_voiced = is_voiced.float().unsqueeze(2)
        self.mel = mel.float().transpose(1,2)


    def forward(self):
        """Uses hider network to generate hidden representation"""
        self.hidden = self.hider(self.mel)
        self.controlled = self.combiner((self.hidden, self.speaker_id, self.f0_idx, self.is_voiced))

    def backward_F(self):
        # Attempt to predict the control variable from the hidden repr
        pred_f0, pred_speaker_id = self.finder(self.hidden.detach())
        batch_size, n_speakers = pred_speaker_id.shape
        # Need to combine batch and sequence for CE loss
        self.finder_loss = self.f_criterion(pred_speaker_id, self.speaker_id)

    def backward_G(self):
        # G is hider and combiner together

        # newly updated finder can generate a better training signal
        pred_f0, pred_speaker_id = self.finder(self.hidden)
        # Softmax converts to probabilities, which we can use for leakage loss
        #print(pred_speaker_id.shape)
        self.pred_speaker_id = F.softmax(pred_speaker_id, dim=1)

        # Use output from forward earlier to do mse between resynthed, controlled output and
        # ground truth input
        #print(self.controlled.shape, self.mel.shape)
        self.combiner_loss = self.g_criterion(self.controlled, self.mel.transpose(1,2))

        self.leakage_loss_id = torch.var(self.pred_speaker_id, 1)
        # Use mean to reduce along all timesteps
        self.leakage_loss = self.leakage_loss_id.mean() # + leakage los for f0
        self.g_losses = self.combiner_loss + self.hp.model.beta * self.leakage_loss

    def optimize_parameters(self):
        g_opt, f_opt = self.optimizers()
        # Generate hidden repr and output
        self.forward()

        # Train finder
        self.toggle_optimizer(f_opt)
        self.backward_F()
        #nn.utils.clip_grad_norm_(self.finder.parameters(), self.hp.clip)
        f_opt.zero_grad() # TODO: check this
        self.manual_backward(self.finder_loss)
        self.clip_gradients(f_opt, gradient_clip_val=self.hp.training.clip, gradient_clip_algorithm="norm")
        f_opt.step()
        self.untoggle_optimizer(f_opt)

        # Train hider and combiner with updated Finder for leakage loss
        self.toggle_optimizer(g_opt)
        self.backward_G()
        self.manual_backward(self.g_losses)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        self.clip_gradients(g_opt, gradient_clip_val=self.hp.training.clip, gradient_clip_algorithm="norm")
        g_opt.step()
        self.untoggle_optimizer(g_opt)

    def training_step(self, batch):
        self.set_input(*batch)
        self.optimize_parameters()
        self.log('leakage_loss', self.leakage_loss, prog_bar=True)
        self.log('combiner_loss', self.combiner_loss, prog_bar=True)
        self.log('finder_loss', self.finder_loss, prog_bar=True)
        self.log('g_losses', self.g_losses, prog_bar=True)

    """
    def validation_step(self, batch):
        self.set_input(*batch)
        self.backward_G()
        self.backward_F()
    """


    def anneal(self):
        # TODO replace with LR scheduler
        self.hp.g_lr *= self.hp.annealing_rate
        self.hp.f_lr *= self.hp.annealing_rate
        print('Annealing', self.hp.f_lr, self.hp.g_lr)
        generator_params = list(self.hider.parameters()) + list(self.combiner.parameters())
        self.g_opt = Adam(generator_params, lr=self.hp.g_lr, weight_decay=0.00)
        self.f_opt = Adam(self.finder.parameters(), lr=self.hp.f_lr, weight_decay=0.00)



@hydra.main(version_base=None, config_path='config', config_name="config")
def train(config):

    dm = HFCDataModule(config, 'hfc')
    n_speakers = dm.n_speakers
    hfc = HFC(config, n_speakers)

    logger = pl.loggers.wandb.WandbLogger(project="hfc_main")
    trainer = pl.Trainer(logger=logger) #logger=logger, gradient_clip_val=0.5, detect_anomaly=True)
    # Create a Tuner

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    trainer.fit(hfc, datamodule=dm)

if __name__ == '__main__':
    train()