import hydra
import os

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import wandb
from speechbrain.pretrained import HIFIGAN
import sysrsync
from pathlib import Path as P


# Local imports
import dsp
from hider import Hider
from finder import Finder
from combiner import Combiner
from new_new_dataset import HFCDataModule
import plottage

os.environ['TRANSFORMERS_CACHE'] = './cache/transformers'
os.environ['WANDB_CACHE_DIR'] = './cache/wand'

class HFC(pl.LightningModule):
    """ An implementation of the Hider Finder Combiner architecture for
    arbitrary control of speech signals """
    def __init__(self, hp, n_speakers):
        super(HFC, self).__init__()

        self.hp = hp
        self.save_hyperparameters(hp)
        self.n_speakers = n_speakers
        self.f0_bins = hp.control_variables.f0_bins
        self.automatic_optimization=False # required for manual optimization scheduling in lightning

        # Component networks
        self.hider = Hider(hp)
        self.finder = Finder(hp, n_speakers)
        self.combiner = Combiner(hp, n_speakers)
        self.hifigan = None

        # The 'combiner loss' criterion
        self.g_criterion = nn.MSELoss()
        # Ignore (for now) index associated with unvoiced regions
        self.f_criterion = nn.CrossEntropyLoss()
        print('n_speakers = ', n_speakers)

    def configure_optimizers(self):
        # Define params that generate during synthesis as 'generator' (g)
        self.generator_params = list(self.hider.parameters()) + list(self.combiner.parameters())
        self.g_opt = Adam(self.generator_params, lr=self.hp.training.lr_g, weight_decay=0.00)
        self.f_opt = Adam(self.finder.parameters(), lr=self.hp.training.lr_f)
        #self.lr_schedulers = {'g_opt': self.g_opt, 'f_opt': self.f_opt} TODO
        #self.g_schedj = torch.optim.lr_scheduler.StepLR(self.g_opt, step_size=50, gamma=0.5) # TODO add this 
        self.g_schedj = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_opt, verbose=True)
        self.f_schedj = torch.optim.lr_scheduler.ReduceLROnPlateau(self.f_opt, verbose=True)
        return [self.g_opt, self.f_opt]


    def set_input(self, mel, f0, is_voiced, speaker_id, spkr_emb):

        # Find the shape of the onehots for the cv
        f0 = f0.squeeze(-1)
        is_voiced = is_voiced.squeeze(-1)
        batch_size, seq_len = f0.shape

        # Two versions --- one matrix onehot and one array of indices
        #self.cv_index = dsp.onehot_index(control_variable, self.hp.cv_bins, self.hp.cv_range)
        self.speaker_id = speaker_id

        self.f0_idx = dsp.bin_tensor(f0, self.f0_bins, self.hp.control_variables.f0_min, self.hp.control_variables.f0_max)
        self.is_voiced = is_voiced.float().unsqueeze(2)
        self.mel = mel.float().transpose(1,2)
        self.spkr_emb = spkr_emb


    def forward(self):
        """Uses hider network to generate hidden representation"""
        self.hidden = self.hider(self.mel)
        self.controlled = self.combiner((self.hidden, self.speaker_id, self.spkr_emb, self.f0_idx, self.is_voiced))

    def backward_F(self):
        # Attempt to predict the control variable from the hidden repr
        pred_f0, pred_speaker_id = self.finder(self.hidden.detach())

        self.finder_loss_id = self.f_criterion(pred_speaker_id, self.speaker_id)
        # pred_f0 = (Batch x seq_len x f0_bins) -- need to put seq_len at the end for loss
        self.finder_loss_f0 = self.f_criterion(pred_f0.transpose(1,2), self.f0_idx)
        self.finder_loss = 0. * self.finder_loss_f0 + self.finder_loss_id

    def backward_G(self, adversarial=False):
        # G is hider and combiner together

        if adversarial:
            # newly updated finder can generate a better training signal
            pred_f0, pred_speaker_id = self.finder(self.hidden)
            # Softmax converts to probabilities, which we can use for leakage loss
            #print(pred_speaker_id.shape)
            self.pred_speaker_id = F.softmax(pred_speaker_id, dim=1)
            self.leakage_loss_id = torch.var(self.pred_speaker_id, 1).mean() * self.n_speakers
        else:
            self.leakage_loss_id = 0.

        # Use output from forward earlier to do mse between resynthed, controlled output and
        # ground truth input
        #print(self.controlled.shape, self.mel.shape)
        self.combiner_loss = self.g_criterion(self.controlled, self.mel.transpose(1,2))

        # Use mean to reduce along all timesteps
        self.leakage_loss = self.leakage_loss_id # + leakage los for f0
        self.g_losses = self.combiner_loss + self.hp.model.beta * self.leakage_loss

    def optimize_parameters(self):
        g_opt, f_opt = self.optimizers()
        self.log('learning_rate_g', g_opt.param_groups[0]['lr'])
        self.log('learning_rate_f', f_opt.param_groups[0]['lr'])
        # Generate hidden repr and output
        # Train finder
        self.forward()
        self.backward_F()
        f_opt.zero_grad() # TODO: check this
        self.manual_backward(self.finder_loss)
        self.clip_gradients(f_opt, gradient_clip_val=self.hp.training.clip, gradient_clip_algorithm="norm")
        f_opt.step()


        # Train hider and combiner with updated Finder for leakage loss
        self.backward_G(adversarial=self.global_step > self.hp.training.adversarial_start)
        g_opt.zero_grad()
        self.manual_backward(self.g_losses)
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        self.clip_gradients(g_opt, gradient_clip_val=self.hp.training.clip, gradient_clip_algorithm="norm")
        g_opt.step()

        if self.global_step % self.hp.training.scheduler_step == 0:
            self.g_schedj.step(self.combiner_loss)
            self.f_schedj.step(self.finder_loss)


    def training_step(self, batch):
        self.set_input(*batch)
        self.optimize_parameters()
        self.log('leakage_loss', self.leakage_loss, prog_bar=True)
        self.log('combiner_loss', self.combiner_loss, prog_bar=True)
        self.log('finder_loss', self.finder_loss, prog_bar=True)
        self.log('g_losses', self.g_losses, prog_bar=True)
        return self.g_losses

    def validation_step(self, batch, batch_idx):
        self.set_input(*batch)
        self.new_speaker = torch.load(
            P(os.path.expanduser(self.hp.dataset.root)) / P(self.hp.dataset.save_as) / P('spkr_emb/254/12312/254_12312_000004_000000.pt')
        ).unsqueeze(0).to(self.mel.device)
        self.forward()
        self.backward_G()
        self.backward_F()
        self.log('val/leakage_loss', self.leakage_loss, prog_bar=True)
        self.log('val/combiner_loss', self.combiner_loss, prog_bar=True)
        self.log('val/finder_loss', self.finder_loss, prog_bar=True)
        self.log('val/g_losses', self.g_losses, prog_bar=True)
        if batch_idx < self.hp.training.log_n_audios and self.hp.training.wandb:
            if not self.hifigan:
                self.hifigan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="hifigan_checkpoints", run_opts={"device": self.mel.device})
            self.new_controlled = self.combiner((self.hidden, 0, self.new_speaker, self.f0_idx, self.is_voiced))
            self.log_audio(batch_idx)
    
    def on_train_epoch_end(self):
        # Step learning rate schedulers
        self.g_schedj.step(self.trainer.callback_metrics['combiner_loss'])
        self.f_schedj.step(self.trainer.callback_metrics['finder_loss'])
    
    def log_audio(self, n):
        # Depends on wandb -- TODO make an option for local or whatever
        controlled = self.controlled.transpose(1,2)
        new_controlled = self.new_controlled.transpose(1,2)
        audio = self.hifigan.decode_batch(controlled)
        audio_changed = self.hifigan.decode_batch(new_controlled)
        audio_vc_copy = self.hifigan.decode_batch(self.mel)


        #self.logger.log_image('gt_spect', [self.mel.squeeze().cpu().numpy()], step=self.global_step)
        #self.logger.log_image('copy_spect', [controlled.squeeze().cpu().numpy()], step=self.global_step)
        #self.logger.log_image('modified_spect', [new_controlled.squeeze().cpu().numpy()], step=self.global_step)


        #self.log('gt_audio': wandb.Audio(self.mel)) TODO
        metrics = {
            'audios': [
                wandb.Audio(audio.squeeze().cpu().numpy(), sample_rate=self.hp.sr, caption='unmodified'), 
                wandb.Audio(audio_changed.squeeze().cpu().numpy(), sample_rate=self.hp.sr, caption='modified'),
                wandb.Audio(audio_vc_copy.squeeze().cpu().numpy(), sample_rate=self.hp.sr, caption='vc_copy'),
            ],
            'spectrograms': [
                wandb.Image(self.mel.squeeze().cpu().numpy(), caption='unmodified'),
                wandb.Image(controlled.squeeze().cpu().numpy(), caption='copy'),
                wandb.Image(new_controlled.squeeze().cpu().numpy(), caption='modified'),
                ]
        }
        self.logger.log_metrics(metrics, step=self.global_step)
        #self.logger.log_audio('copy_audio', audio.squeeze().cpu().numpy(), sample_rate=self.hp.sr)
        #self.logger.log_audio('modified_audio', audio_changed.squeeze().cpu().numpy(), sample_rate=self.hp.sr)
        '''
        html_file_name = f'outputs/audio_{int(self.speaker_id)}_{n}.html'
        html_file_name_changed = f'outputs/audio_changed_{int(self.speaker_id)}_{n}.html'
        plottage.save_audio_with_bokeh_plot_to_html(audio, self.hp.sr, html_file_name)
        plottage.save_audio_with_bokeh_plot_to_html(audio_changed, self.hp.sr, html_file_name)
        html = wandb.Html(html_file_name)
        html_c = wandb.Html(html_file_name_changed)

        my_table = wandb.Table(columns=["audio_unchanged"], data=[[html], [html]])
        my_table = wandb.Table(columns=["audio_changed"], data=[[html_c], [html_c]])

        self.logger.log_table('val/audio', columns=['audio_unchanged'], data=[[html], [html]])
        self.logger.log_table('val/audio_changed', columns=['audio_changed'], data=[[html_c], [html_c]])
        '''



@hydra.main(version_base=None, config_path='config', config_name="config")
def train(config):

    print('Setting up data module')
    # TODO move the below to data module
    if False: #config.dataset.rsync:
        os.makedirs(config.dataset.root, exist_ok=True)
        print('rsyncing from ', config.dataset.copy_from)
        sysrsync.run(
            source=os.path.join(os.path.expanduser(config.dataset.copy_from), config.dataset.save_as),
            destination=os.path.join(config.dataset.root, config.dataset.save_as),
            options=['-a']
        )
        print('done')

    dm = HFCDataModule(config, 'hfc', download=config.dataset.download)
    n_speakers = dm.n_speakers
    hfc = HFC(config, n_speakers)
    print('Done')
    if torch.cuda.device_count() > 1:
        strategy = 'ddp_find_unused_parameters_true'
    else:
        strategy = 'auto'

    if config.training.wandb:
        logger = pl.loggers.WandbLogger(project="hfc_main")
    else:
        logger = None
    
    checkpointers = [
        ModelCheckpoint(monitor='val/g_losses', filename='checkpoint-{epoch:02d}-{val_g_losses:.2f}', save_top_k=10),
        #ModelCheckpoint(save_top_k=10, save_last=True)
    ]
    
    trainer = pl.Trainer(logger=logger,
                         check_val_every_n_epoch=config.training.val_check_interval,
                         max_epochs=config.training.epochs,
                         strategy=strategy,
                         callbacks=checkpointers,
                        ) 

    # finds learning rate automatically
    # sets hparams.lr or hparams.learning_rate to that learning rate
    trainer.fit(hfc, datamodule=dm)

    return trainer.callback_metrics[config.training.metric]

if __name__ == '__main__':
    train()