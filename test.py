import os
import torch
from matplotlib import pyplot as plt
import glob
import numpy as np
import librosa

import hfc
import dsp
import hparams
import train

models_dir = '/home/jjw/trained_models/outputs/'
test_dir = '/home/jjw/datasets/lombard/test/'


def main():
    hp = hparams.get_hparams()

    model = hfc.HFC(hp)
    model.load_state_dict(torch.load(os.path.join(models_dir, hp.cv, 'hfc.pt')))
    if hp.use_cuda:
        model.cuda()
    print(next(model.parameters()).device)
    model.eval()
    fe = dsp.FeatureEngineer(hp)
    
    for utt in os.listdir(test_dir):
        y = fe.load_wav(os.path.join(test_dir,utt))
        s = fe.wav_to_s(y)
        mel = fe.s_to_mel(s)
        f0, vuv = fe.f0(y)
        st = fe.st(s)
        i = fe.intensity(s)
        cv = f0

        cv, f0, vuv, mel = fe.crop_to_shortest(cv, f0, vuv, mel)

        model.set_input(cv, f0, vuv, mel)
        model.forward()

        plt.imshow(torch.log(model.controlled.cpu().detach().squeeze()), origin='lower')
        plt.show()
        print(model.controlled.shape)
        s_out = fe.mel_to_s(model.controlled.transpose(1,2).cpu())
        plt.imshow(torch.log(s_out.detach().squeeze()), origin='lower')
        plt.show()
        print(s_out.shape)
        exit()

def test_simple():

    hp = hparams.get_hparams()
    ranges = {
        'f0' : hp.f0_range,
        'CoG' : hp.CoG_range,
        'ST' : hp.ST_range,
        'I' : hp.I_range
    }

    model = hfc.HFC(hp)
    model_file = '/home/jjw/trained_models/best/{}/hfc.pt'

    test_list = glob.glob('/home/jjw/datasets/lombard/test/*')
    if hp.use_cuda:
        model.cuda()
    model.eval()
    fe = dsp.FeatureEngineer(hp)
    for variable in ['f0', 'CoG', 'I', 'ST']:
        hp.cv_range = ranges[variable]
        utt = test_list[0]
        model.load_state_dict(torch.load(model_file.format(variable)))

        y = fe.load_wav(utt)
        s = fe.wav_to_s(y)
        mel = fe.s_to_mel(s)
        s_ = fe.mel_to_s(mel)
        #wav = fe.griffin_lim(s_)
        wav = librosa.griffinlim(s_.squeeze().numpy(), win_length=hp.win_length,
                hop_length=hp.hop_length)
        wav = torch.from_numpy(wav).unsqueeze(0)
        fe.save_wav(100*wav, 'gl_copy.wav')

        f0, vuv = fe.f0(y)
        st = fe.st(s)
        i = fe.intensity(s)
        cog = fe.cog(s)
        cv = {'f0' : f0, 'CoG' : cog, 'I' : i, 'ST' : st}[variable]
        cv = cv.view(1, cv.nelement())

        cvs = [cv, cv *1.2, cv*0.8, torch.ones_like(cv)*cv.mean()]
        f0s = cvs if variable == 'f0' else [f0] * 4
        names = ['copy', 'raise', 'lower', 'mono']

        output_mels = [mel]
        print(mel.min(), mel.max())
        for c, f, name in zip(cvs, f0s, names):
            c, f, vuv, mel = fe.crop_to_shortest(c, f, vuv, mel)
            model.set_input(c, f, vuv, mel)
            model.forward()
            out = model.controlled
            output_mels += out
            print(out.min(), out.max())
            _s = fe.mel_to_s(out.detach().cpu().transpose(1,2))
            wav = librosa.griffinlim(_s.squeeze().numpy(), win_length=hp.win_length,
                    hop_length=hp.hop_length)
            st = fe.st(_s)
            i = fe.intensity(_s)
            cog = fe.cog(_s)

            train.plot_spects(hp, model.cv, model.cv, model.f0, model.hidden, out, model.x, 
                    'output.png')

            base = os.path.basename(utt)
            utt_name = os.path.splitext(base)[0]
            save(out, 'test_results/' + variable + f'_{name}_' + utt_name + '.npy')
            wav = torch.from_numpy(wav).unsqueeze(0)
            fe.save_wav(50*wav, 'test_results/' + variable + f'_{name}_' + utt_name + '.wav')

        plot(*output_mels)




def test_lombard():

    hp = hparams.get_hparams()

    model = hfc.HFC(hp)
    model_file = '/home/jjw/trained_models/best/{}/hfc.pt'

    test_list = glob.glob('/home/jjw/datasets/lombard/test/*')
    if hp.use_cuda:
        model.cuda()
    model.eval()
    fe = dsp.FeatureEngineer(hp)
    
    for variable in ['CoG', 'I', 'ST', 'f0']:
        model.load_state_dict(torch.load(model_file.format(variable)))
        for utt in test_list[:1]:
            print(utt)
            base = os.path.basename(utt)
            utt_name = os.path.splitext(base)[0]
            y = fe.load_wav(utt)
            s = fe.wav_to_s(y)
            mel = fe.s_to_mel(s)
            print(mel.shape)
            f0, vuv = fe.f0(y)
            st = fe.st(s)
            i = fe.intensity(s)
            cog = fe.cog(s)
            cv = {'f0' : f0, 'CoG' : cog, 'I' : i, 'ST' : st}[variable]
            cv = cv.view(1, cv.nelement())
            # NOTE for spectral tilt add, for others miltiply because they are log

            cv, f0, vuv, mel = fe.crop_to_shortest(cv, f0, vuv, mel)
            model.set_input(cv, f0, vuv, mel)
            model.forward()
            copy = model.controlled
            save(copy, 'test_results/' + variable + '_copy_' + utt_name + '.npy')

            cv_m = torch.ones_like(cv) * cv.mean()

            if variable == 'f0':
                f0_ = torch.ones_like(f0) * f0.mean()
            else:
                f0_ = f0
            model.set_input(cv_m, f0_, vuv, mel)
            model.forward()
            mono = model.controlled
            save(mono, 'test_results/' + variable + '_mono_' + utt_name + '.npy')

            cv_r = cv * 1.2
            if variable == 'f0':
                f0_ = f0 * 1.2
            else:
                f0_ = f0
            model.set_input(cv_r, f0_, vuv, mel)
            model.forward()
            r = model.controlled
            save(r, 'test_results/' + variable + '_r_' + utt_name + '.npy')

            if variable == 'f0':
                f0_ = f0 * 0.8
            else:
                f0_ = f0
            cv_l = cv * 0.8
            model.set_input(cv_l, f0_, vuv, mel)
            model.forward()
            l = model.controlled
            save(l, 'test_results/' + variable + '_l_' + utt_name + '.npy')

            print(copy.shape)
            copy_s = fe.mel_to_s(copy.cpu().transpose(1,2))
            mono_s = fe.mel_to_s(mono.cpu().transpose(1,2))
            r_s = fe.mel_to_s(r.cpu().transpose(1,2))
            l_s = fe.mel_to_s(l.cpu().transpose(1,2))
            fig, axes = plt.subplots(3,1)
            plt.title(variable)
            axes[0].set_title('Intensity')
            axes[1].set_title('Spectral Tilt')
            axes[2].set_title('Centre of Gravity')
            for s_, label in zip([r_s, l_s, copy_s, mono_s], ['raise', 'lower', 'copy', 'mono']):
                axes[0].plot(fe.intensity(s_.transpose(1,2).detach().squeeze()), label=label)
                axes[1].plot(fe.st(s_.detach().squeeze()), label=label)
                axes[2].plot(fe.cog(s_.detach()).squeeze(), label=label)

            plt.legend()
            plt.savefig(f'controlling_{variable}.png')
            plt.clf()
            plot(mel, copy, mono, r, l, saveas=f'controlling_{variable}_spects.png')
            #s_out = fe.mel_to_s(model.controlled.transpose(1,2).cpu())
            #plt.imshow(torch.log(s_out.detach().squeeze()), origin='lower')
            #plt.show()
            #print(s_out.shape)

def save(mel, location):
    mel = mel.cpu().detach().squeeze().numpy()
    np.save(location, mel.T)

def plot(mel, copy, mono, r, l, saveas='plots.png'):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1)
    ax1.imshow(torch.log(mel.cpu().detach().squeeze()), origin='lower')
    ax2.imshow(torch.log(copy.cpu().detach().squeeze().transpose(0,1)), origin='lower')
    ax3.imshow(torch.log(mono.cpu().detach().squeeze().transpose(0,1)), origin='lower')
    ax4.imshow(torch.log(r.cpu().detach().squeeze().transpose(0,1)), origin='lower')
    ax5.imshow(torch.log(l.cpu().detach().squeeze().transpose(0,1)), origin='lower')
    ax1.set_title('orginial')
    ax2.set_title('copy')
    ax3.set_title('mono')
    ax4.set_title('lower')
    ax5.set_title('raise')
    plt.savefig(saveas)
    #plt.show()
    plt.clf()

def test_lj():
    hp = hparams.get_hparams()

    model = hfc.HFC(hp)
    model_files = '/home/jjw/trained_models/outputs/lj_f0_10/hfc.pt'
    test_list = glob.glob('/home/jjw/datasets/lombard/test/*')
    model.load_state_dict(torch.load(model_file))
    if hp.use_cuda:
        model.cuda()
    model.eval()
    fe = dsp.FeatureEngineer(hp)
    
    for utt in test_list:
        print(utt)
        y = fe.load_wav(utt)
        s = fe.wav_to_s(y)
        mel = fe.s_to_mel(s)
        f0, vuv = fe.f0(y)
        st = fe.st(s)
        i = fe.intensity(s)
        cv = f0

        cv, f0, vuv, mel = fe.crop_to_shortest(cv, f0, vuv, mel)
        model.set_input(cv, f0, vuv, mel)
        model.forward()
        copy = model.controlled

        cv_m = torch.ones_like(cv) * cv.mean()
        f0_m = torch.ones_like(f0) * f0.mean()
        model.set_input(cv_m, f0_m, vuv, mel)
        model.forward()
        mono = model.controlled

        cv_r = cv * 1.2
        model.set_input(cv_r, cv_r, vuv, mel)
        model.forward()
        r = model.controlled
        cv_l = cv * 0.8

        model.set_input(cv_l, cv_l, vuv, mel)
        model.forward()
        l = model.controlled

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1)
        ax1.imshow(torch.log(mel.cpu().detach().squeeze()), origin='lower')
        ax2.imshow(torch.log(copy.cpu().detach().squeeze().transpose(0,1)), origin='lower')
        ax3.imshow(torch.log(mono.cpu().detach().squeeze().transpose(0,1)), origin='lower')
        ax4.imshow(torch.log(r.cpu().detach().squeeze().transpose(0,1)), origin='lower')
        ax5.imshow(torch.log(l.cpu().detach().squeeze().transpose(0,1)), origin='lower')
        ax1.set_title('orginial')
        ax2.set_title('copy1')
        ax3.set_title('mono')
        ax4.set_title('lower')
        ax5.set_title('raise')
        plt.show()
        print(model.controlled.shape)
        #s_out = fe.mel_to_s(model.controlled.transpose(1,2).cpu())
        #plt.imshow(torch.log(s_out.detach().squeeze()), origin='lower')
        #plt.show()
        print(s_out.shape)
        exit()

if __name__ == '__main__':
    test_simple()
