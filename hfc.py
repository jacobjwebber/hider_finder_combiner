import torch
import torchaudio
import lightning as L

import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


# Local imports
import dsp
from hider import Hider
from finder import Finder
from combiner import Combiner

class HFC(L.LightningModule):
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
        self.hider = Hider(hp.h_rnn_size, hp.input_width, hp.hidden_size, hp.h_layers, dropout=hp.h_drop, denoising=hp.denoising)
        self.finder = Finder(hp.f_rnn_size, hp.hidden_size, n_speakers, hp.f_layers, dropout = hp.f_drop)
        self.combiner = Combiner(hp.c_rnn_size, hp.hidden_size, hp.input_width, hp.c_layers, n_speakers, dropout=hp.c_drop)

        # The 'combiner loss' criterion
        self.g_criterion = nn.MSELoss()
        # Ignore (for now) index associated with unvoiced regions
        self.f_criterion = nn.CrossEntropyLoss(ignore_index=0)

    def configure_optimizers(self):
        # Define params that generate during synthesis as 'generator' (g)
        self.generator_params = list(self.hider.parameters()) + list(self.combiner.parameters())
        self.g_opt = Adam(self.generator_params, lr=self.hp.g_lr, weight_decay=0.00)
        self.f_opt = Adam(self.finder.parameters(), lr=self.hp.f_lr, weight_decay=0.00)
        return [self.g_opt, self.f_opt]


    def set_input(self, speaker_id, f0, is_voiced, mel):

        # Find the shape of the onehots for the cv
        batch_size, seq_len = f0.shape
        hot_shape = (batch_size, seq_len, self.f0_bins)

        # Two versions --- one matrix onehot and one array of indices
        #self.cv_index = dsp.onehot_index(control_variable, self.hp.cv_bins, self.hp.cv_range)
        #self.cv = dsp.onehotify(self.cv_index, hot_shape, self.hp.use_cuda)
        print(speaker_id.shape)
        print(speaker_id.type())
        self.speaker_id = speaker_id

        self.f0_index = dsp.onehot_index(f0, self.f0_bins, self.hp.f0_range)

        self.f0 = dsp.onehotify(self.f0_index, hot_shape, self.hp.use_cuda)
        self.is_voiced = is_voiced.float().unsqueeze(2)
        # Call the input spectral features simply x --- dont want to tie features to name e.g. mel etc
        self.mel = mel.float().transpose(1,2)


    def forward(self):
        """Uses hider network to generate hidden representation"""
        self.hidden = self.hider(self.x)
        self.controlled = self.combiner((self.hidden, self.speaker_id, self.f0, self.is_voiced))

    def backward_F(self):
        # Attempt to predict the control variable from the hidden repr
        pred_speaker_id = self.finder(self.hidden.detach())
        batch_size, seq_len, width = pred_speaker_id.shape
        # Need to combine batch and sequence for CE loss
        pred_speaker_id = pred_speaker_id.view(batch_size * seq_len, -1)
        real_speaker_id = self.speaker_id.view(batch_size * seq_len).long()
        self.finder_loss = self.f_criterion(pred_speaker_id, real_speaker_id)
        self.log('finder_loss', self.finder_loss, )

    def backward_G(self):
        # G is hider and combiner together

        # newly updated finder can generate a better training signal
        # Softmax converts to probabilities, which we can use for leakage loss
        self.pred_speaker_id = F.softmax(self.finder(self.hidden), dim=2)

        # Use output from forward earlier to do mse between resynthed, controlled output and
        # ground truth input
        self.combiner_loss = self.g_criterion(self.controlled, self.x)

        self.leakage_loss = torch.var(self.pred_speaker_id, 2)
        # Use mean to reduce along all timesteps
        self.leakage_loss = self.leakage_loss.mean()
        self.g_losses = self.combiner_loss + self.hp.beta * self.leakage_loss

    def optimize_parameters(self):
        g_opt, f_opt = self.optimizers()
        # Generate hidden repr and output
        self.forward()

        # Train finder
        self.toggle_optimizer(f_opt)
        self.backward_F()
        #nn.utils.clip_grad_norm_(self.finder.parameters(), self.hp.clip)
        f_opt.zero_grad()
        self.manual_backward(self.finder_loss)
        self.clip_gradients(f_opt, gradient_clip_val=self.hp.clip, gradient_clip_algorithm="norm")
        f_opt.step()
        self.untoggle_optimizer(f_opt)

        # Train hider and combiner with updated Finder for leakage loss
        self.toggle_optimizer(g_opt)
        self.backward_G()
        self.manual_backward(self.g_losses)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        self.clip_gradients(g_opt, gradient_clip_val=self.hp.clip, gradient_clip_algorithm="norm")
        g_opt.step()
        self.untoggle_optimizer(g_opt)

    def training_step(self, batch):
        self.set_input(*batch)
        self.optimize_parameters()

    def validation_step(self, batch):
        self.set_input(*batch)
        self.backward_G()
        self.backward_F()


    def anneal(self):
        # TODO replace with LR scheduler
        self.hp.g_lr *= self.hp.annealing_rate
        self.hp.f_lr *= self.hp.annealing_rate
        print('Annealing', self.hp.f_lr, self.hp.g_lr)
        generator_params = list(self.hider.parameters()) + list(self.combiner.parameters())
        self.g_opt = Adam(generator_params, lr=self.hp.g_lr, weight_decay=0.00)
        self.f_opt = Adam(self.finder.parameters(), lr=self.hp.f_lr, weight_decay=0.00)



