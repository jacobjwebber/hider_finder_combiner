# Required Libraries
import torch

# Local imports
import hparams
from hfc import HFC
import dataset
# For plots --- put somewhere else TODO
import dsp
from torch.nn import functional as F
import numpy as np
import os
from matplotlib import pyplot as plt
import lightning as pl

def train_lightning():
    hp = hparams.get_hparams()
    d = dataset.get_dataset(hp)

    hfc = HFC(hp)

    dataloader, valid_dataloader = dataset.get_training_dataloaders(
            d, hp.train_batch_size, hp.valid_batch_size, hp.validation_split)

    trainer = pl.Trainer(devices=1, limit_train_batches=100, max_epochs=1)
    trainer.fit(model=hfc, train_dataloaders=dataloader)



def train():

    hp = hparams.get_hparams()
    d = dataset.get_dataset(hp)
    fe = dsp.FeatureEngineer(hp)

    print(hp)
    hfc = HFC(hp)
    dataloader, valid_dataloader = dataset.get_training_dataloaders(
            d, hp.train_batch_size, hp.valid_batch_size, hp.validation_split)

    if hp.use_cuda:
        hfc.cuda()

    if hp.save_path == '':
        save_path = os.path.join('outputs', hp.cv)
    else:
        save_path = os.path.join('outputs', hp.save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logfile = os.path.join(save_path, 'log.txt')
    l = open(logfile, 'w')

    best_loss = float('inf')

    for e in range(hp.epochs):
        for i, datum in enumerate(dataloader):
            hfc.set_input(*datum)
            hfc.optimize_parameters()
            #print('step = {}, g_losses = {}, leakage = {}'.format(i, hfc.g_losses.item(), hfc.leakage_loss.item()))

            if i % 200 == 0:
                hfc.eval()
                val_combiner_losses = []
                val_leakage_losses = []
                val_finder_losses = []
                fidel_losses = []
                for valid_datum in valid_dataloader:
                    cv, f0, is_voiced, x = valid_datum
                    hfc.set_input(*valid_datum)
                    hfc()
                    # Calculate losses using backward methods (which don't actually backprop) rename later?
                    hfc.backward_G()
                    hfc.backward_F()
                    val_finder_losses.append(hfc.finder_loss.item())
                    val_combiner_losses.append(hfc.combiner_loss.item())
                    val_leakage_losses.append(hfc.leakage_loss.item())

                    if d.fidel is not None:
                        out_cv = d.fidel(fe.mel_to_s(hfc.controlled.transpose(1,2).cpu().detach()))
                        fidel_losses.append(F.mse_loss(out_cv, cv).item())

                plot_spects(hp, hfc.cv, hfc.pred_cv, hfc.f0, hfc.hidden, hfc.controlled, hfc.x,
                        'plotty_{}_{}'.format(e, i), save_path)
                v_f_loss = np.mean(val_finder_losses)
                v_c_loss = np.mean(val_combiner_losses)
                v_l_loss = np.mean(val_leakage_losses)

                if d.fidel is not None:
                    fidel_loss = np.mean(fidel_losses)
                else:
                    fidel_loss = 'na'

                if v_c_loss > best_loss:
                    hfc.anneal()
                else:
                    best_loss = v_c_loss
                    torch.save(hfc.state_dict(), os.path.join(save_path, 'hfc.pt'))
                print('step = {}, c = {}, l = {}, f = {}, fidelity = {}'.format(i, v_c_loss,  v_l_loss, v_f_loss, fidel_loss))
                l.write('step = {}, c = {}, l = {}, f = {}, fidelity = {}\n'.format(i, v_c_loss,  v_l_loss, v_f_loss, fidel_loss))
                l.flush()
                hfc.train()



def plot_spects(hp, cv, est_control_var, f0, source, output, target, savename='', directory='.'):
    nsource = source.cpu().detach().numpy().squeeze().T
    noutput = output.cpu().detach().numpy().squeeze().T
    ncv = cv.cpu().detach().numpy().squeeze().T
    nf0 = f0.cpu().detach().numpy().squeeze().T
    n_est_control = est_control_var.cpu().detach().numpy().squeeze().T
    ntarget = target.cpu().detach().numpy().squeeze().T
    num_freq_bins, numframes = nsource.shape

    extent = None #  (0.0, numframes / hop_length, 0.0, num_freq_bins - 1.)
    cv_extent = None

    plt.rcParams.update({'font.size': 5})
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(6, 1)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Time step')
        ax.set_ylabel('Bin no.')

    aspect = None
    cv_aspect = None

    ax0.set_title('Interpolated control variable onehot encoding')
    cvplot = ax0.imshow(ncv, origin='lower', extent=cv_extent, interpolation="none", aspect=cv_aspect)

    ax1.set_title('Interpolated control variable estimated')
    cv_plot = ax1.imshow(n_est_control, origin='lower', extent=cv_extent, interpolation="none", aspect=cv_aspect)

    ax2.set_title('Hidden Encoding')
    inplot = ax2.imshow(nsource, origin='lower', extent=extent, interpolation="none", aspect=aspect)

    ax3.set_title('Output mel spectrogram')
    outplot = ax3.imshow(noutput, origin='lower', extent=extent, interpolation="none", aspect=aspect)

    ax4.set_title('Target mel spectrogram')
    ax4.imshow(ntarget, origin='lower', extent=extent, interpolation="none", aspect=aspect)

    ax5.set_title('input f0')
    ax5.imshow(nf0, origin='lower', extent=extent, interpolation="none", aspect=aspect)
    plt.tight_layout()
    if savename == '':
        plt.show()
    else:
        plt.savefig(os.path.join(directory, savename), dpi=1000)
    plt.close()


if __name__ == '__main__':
    train_lightning()

