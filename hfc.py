import torch
import torchaudio
import lightning as L

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Local imports
import network
import dsp

class HFC(L.LightningModule):
    """ An implementation of the Hider Finder Combiner architecture for
    arbitrary control of speech signals """
    def __init__(self, hp):
        super(HFC, self).__init__()

        self.hp = hp
        self.automatic_optimization=False # required for manual optimization scheduling in lightning

        # Component networks
        # TODO rename width/size dichotomy also input/output mean same thing for hfc
        # TODO pass hp to get_[finder,hider,combiner] method and let it sort everything
        self.hider = network.Hider(hp.h_rnn_size, hp.input_width, hp.hidden_size, hp.h_layers, dropout=hp.h_drop, denoising=hp.denoising)
        self.finder = network.Finder(hp.f_rnn_size, hp.hidden_size, hp.cv_bins, hp.f_layers, dropout = hp.f_drop)
        self.combiner = network.Combiner(hp.c_rnn_size, hp.hidden_size, hp.input_width, hp.c_layers, hp.cv_bins, dropout=hp.c_drop)

        # The 'combiner loss' criterion
        self.g_criterion = nn.MSELoss()
        # Ignore (for now) index associated with unvoiced regions
        self.f_criterion = nn.CrossEntropyLoss(ignore_index=0)

    def configure_optimizers(self):
        # Define params that generate during synthesis as 'generator' (g)
        self.generator_params = list(self.hider.parameters()) + list(self.combiner.parameters())
        self.g_opt = torch.optim.Adam(self.generator_params, lr=self.hp.g_lr, weight_decay=0.00)
        self.f_opt = torch.optim.Adam(self.finder.parameters(), lr=self.hp.f_lr, weight_decay=0.00)
        return [self.g_opt, self.f_opt]


    def set_input(self, control_variable, f0, is_voiced, x):

        # Find the shape of the onehots for the cv
        batch_size, seq_len = f0.shape
        hot_shape = (batch_size, seq_len, self.hp.cv_bins)

        # Two versions --- one matrix onehot and one array of indices
        #self.cv_index = dsp.onehot_index(control_variable, self.hp.cv_bins, self.hp.cv_range)
        #self.cv = dsp.onehotify(self.cv_index, hot_shape, self.hp.use_cuda)
        print(control_variable.shape)
        print(control_variable.type())
        self.cv = control_variable

        self.f0_index = dsp.onehot_index(f0, self.hp.cv_bins, self.hp.f0_range)

        self.f0 = dsp.onehotify(self.f0_index, hot_shape, self.hp.use_cuda)
        self.is_voiced = is_voiced.float().unsqueeze(2)
        # Call the input spectral features simply x --- dont want to tie features to name e.g. mel etc
        self.x = x.float().transpose(1,2)


    def forward(self):
        """Uses hider network to generate hidden representation"""
        self.hidden = self.hider(self.x)
        self.controlled = self.combiner((self.hidden, self.cv, self.f0, self.is_voiced))

    def backward_F(self):
        # Attempt to predict the control variable from the hidden repr
        pred_cv = self.finder(self.hidden.detach())
        batch_size, seq_len, width = pred_cv.shape
        # Need to combine batch and sequence for CE loss
        pred_cv = pred_cv.view(batch_size * seq_len, -1)
        real_cv = self.cv_index.view(batch_size * seq_len).long()
        self.finder_loss = self.f_criterion(pred_cv, real_cv)
        self.log('finder_loss', self.finder_loss, )

    def backward_G(self):
        # G is hider and combiner together

        # newly updated finder can generate a better training signal
        # Softmax converts to probabilities, which we can use for leakage loss
        self.pred_cv = F.softmax(self.finder(self.hidden), dim=2)

        # Use output from forward earlier to do mse between resynthed, controlled output and
        # ground truth input
        self.combiner_loss = self.g_criterion(self.controlled, self.x)

        self.leakage_loss = torch.var(self.pred_cv, 2)
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
        self.hp.g_lr *= self.hp.annealing_rate
        self.hp.f_lr *= self.hp.annealing_rate
        print('Annealing', self.hp.f_lr, self.hp.g_lr)
        generator_params = list(self.hider.parameters()) + list(self.combiner.parameters())
        self.g_opt = torch.optim.Adam(generator_params, lr=self.hp.g_lr, weight_decay=0.00)
        self.f_opt = torch.optim.Adam(self.finder.parameters(), lr=self.hp.f_lr, weight_decay=0.00)

    def save_nets(self):
        # Save network params to disk
        pass


