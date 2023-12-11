import torchaudio

from torch.utils.data import Dataset, DataLoader
import dsp
from until_patched import InverseMelScale
#import pyreaper
from matplotlib import pyplot as plt

# Should only need these with legacy datasets?
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import os
import pyworld
import librosa


# Control variables

def f0_cv(utt, y, voiced, f0, s, mel, fe):
    return f0

def st_cv(utt, y, voiced, f0, s, mel, fe):
    s = fe.mel_to_s(fe.s_to_mel(s))
    return fe.st(s)

def i_cv(utt, y, voiced, f0, s, mel, fe):
    return fe.intensity(s)

def cog_cv(utt, y, voiced, f0, s, mel, fe):
    return fe.cog(s)


# TODO THIS SOURCE FILE IS GARBAaaaGE


class MultiSpeakerDataset(torchaudio.datasets.VCTK_092):

    def __init__(self, hp):
        super().__init__(hp.dataset_dir, download=True)

        self.hp = hp
        # Feature engineer is needed for providing mel spectrograms
        self.fe = dsp.FeatureEngineer(hp)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        waveform, sample_rate, transcript, speaker_id, utterance_id = item
        y = self.fe.resample(waveform, sample_rate)
        mel = self.fe.hifigan_mel_spectrogram(y)
        f0, vuv = self.fe.f0(y)
        f0 = dsp.onehot_index(f0, self.hp.cv_bins, self.hp.f0_range)
        speaker_id = int(speaker_id[1:])
        return speaker_id, f0.squeeze(), vuv.squeeze(), mel.squeeze()



class HFCDataset(Dataset):

    def __init__(self, hp):
        self.wav_dir = hp.wav_dir
        self.num_bins = hp.cv_bins
        self.utterances = self.utterance_list(self.wav_dir)
        self.use_mean = hp.use_mean
        self.hp = hp
        if hp.tiny_dataset:
            self.utterances = self.utterances[:100]
        # Give control var index rather than onehot tensor
        self.datalen = len(self.utterances)
        self.fe = dsp.FeatureEngineer(hp)

        cv_dict = {
            'f0' : (f0_cv, None),
            'ST' : (st_cv, self.fe.st),
            'I' : (i_cv, self.fe.intensity),
            'CoG' : (cog_cv, self.fe.cog)
        }
        self.get_control_var, self.fidel = cv_dict[hp.cv]

    def __len__(self):
        return self.datalen

    def __getitem__(self, index):
        utt = self.utterances[index]
        wav_name = wav_format.format(self.wav_dir, utt)
        y = self.fe.load_wav(wav_name)
        s = self.fe.wav_to_s(y)
        mel = self.fe.s_to_mel(s)
        f0, vuv = self.fe.f0(y)
        control_variable = self.get_control_var(utt, y, vuv, f0, s, mel, self.fe)

        cv, f0, vuv, mel = self.fe.crop_to_shortest(control_variable, f0, vuv, mel)

        if self.hp.cv_twice:
            print("CV TWICE")
            return cv.squeeze(), cv.squeeze(), vuv.squeeze(), mel.squeeze()
        else:
            return cv.squeeze(), f0.squeeze(), vuv.squeeze(), mel.squeeze()

    @staticmethod
    def utterance_list(wav_dir):
        l = os.listdir(wav_dir)
        li = [x.split('.')[0] for x in l]
        return li



class MyDataset(Dataset):
    def __init__(self, hp):
        # target dir is directory contining target spectrograms
        self.wav_dir = hp.wav_dir
        self.num_bins = hp.cv_bins
        self.utterances = self.utterance_list(self.wav_dir)
        self.use_mean = hp.use_mean
        self.hp = hp
        if hp.tiny_dataset:
            self.utterances = self.utterances[:100]
        # Give control var index rather than onehot tensor
        self.datalen = len(self.utterances)

        # Transforms for generating spectral features.
        hop = int(hp.hop_length * hp.sr * 10**-3)
        self.spectrogram = torchaudio.transforms.Spectrogram(hp.n_fft, hop_length=hop, power=2., normalized=True)
        self.mel_scale = torchaudio.transforms.MelScale(hp.num_mels, hp.sr, n_stft=hp.n_fft // 2 + 1)
        self.i_mel_scale = InverseMelScale(hp.n_fft // 2 + 1, n_mels=hp.num_mels, sample_rate=hp.sr)
        self.a_to_DB = torchaudio.transforms.AmplitudeToDB()
        self.norm = dsp.normalize
        self.denorm = dsp.denormalize

    def __getitem__(self, index):
        utt = self.utterances[index]
        wav_name = wav_format.format(self.wav_dir, utt)
        y, sr = torchaudio.load(wav_name)
        if self.hp.sr != sr:
            rs = torchaudio.transforms.Resample(sr, self.hp.sr)
            y = rs(y)

        s = self.spectrogram(y).squeeze().transpose(0,1)
        # This exists outside of the torch world (for now)
        f0, voiced = get_f0(wav_name, self.num_bins, self.hp, f0_as_index=True)
        f0 = torch.from_numpy(f0)
        voiced = torch.from_numpy(voiced)
        #pm_times, voiced, f0_times, f0, corr = pyreaper.reaper(y.numpy(), self.hp.sr, frame_period=self.hp.hop_length)
        #f0 = dsp.interpolate(f0, voiced)

        mel = self.mel_scale(s.transpose(0,1))
        mel = self.a_to_DB(mel)
        mel = self.norm(mel, self.hp).transpose(0,1)
        control_variable = self.get_control_var(utt, y, voiced, f0, s, self.hp)

        if self.use_mean:
            ones = np.ones_like(control_variable)
            control_variable[control_variable == 0] = np.nan
            mean = np.nanmean(control_variable)
            control_variable = np.around(ones * mean)

        seq_len = len(f0)
        mel = mel[:seq_len, :]
        control_variable = control_variable[:seq_len]

        item = (control_variable, f0, voiced, mel)

        return item

    def __len__(self):
        return self.datalen

    def get_control_var(self, utt, y, voiced, f0, s, hp):
        return f0

    @staticmethod
    def utterance_list(wav_dir):
        l = os.listdir(wav_dir)
        li = [x.split('.')[0] for x in l]
        return li


class MyCoG(MyDataset):
    def get_control_var(self, utt, y, voiced, f0, s, hp):
        CoG = dsp.lin2cog(s, hp.sr, hp.n_fft)
        CoG = to_onehot(CoG, num_bins=self.hp.cv_bins, return_index=True, frange=(100, 1500))
        CoG = torch.from_numpy(CoG)
        return CoG

class LJDataset(torchaudio.datasets.LJSPEECH):
    """ A base dataset that returns waveforms and mel spects.
    Based on VCTK but easy to change to another torchaudio dataset.
    """
    def __init__(self, hparams):
        self.hp = hparams
        super().__init__(hp.dataset_dir, download=True)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        waveform, sample_rate, transcript, normalized_transcript = item
        return waveform, sample_rate


def get_dataset(hp):
    DS = HFCDataset
    DS = MultiSpeakerDataset
    return DS(hp)


# TODO
# From here are old dataset classes --- intend to replace using torchaudio datasets and new versions of mel spects
wav_format = '{}/{}.wav'



class Spec2SpecDataset(Dataset):
    """Generalised parent dataset class, only thing that changes should be control param"""
    def __init__(self, hp):
        # target dir is directory contining target spectrograms
        self.wav_dir = hp.wav_dir
        self.target_dir = hp.mel_dir
        self.num_bins = hp.cv_bins
        self.utterances = self.utterance_list(self.wav_dir)
        self.use_mean = hp.use_mean
        self.hp = hp
        if hp.tiny_dataset:
            self.utterances = self.utterances[:100]
        # Give control var index rather than onehot tensor
        self.datalen = len(self.utterances)

    def __getitem__(self, index):
        utt = self.utterances[index]
        wav_name = wav_format.format(self.wav_dir, utt)
        target = self.target_arr(utt)
        f0, voiced = get_f0(wav_name, self.num_bins, self.hp, f0_as_index=True)
        control_variable = self.get_control_var(utt, voiced, f0)
        source = (control_variable, f0, voiced)

        diff = target.shape[0] - source[0].shape[0]
        target = target[diff:]

        diff = f0.shape[0] - source[0].shape[0]
        voiced = voiced[diff:]
        f0 = f0[diff:]

        if self.use_mean:
            ones = np.ones_like(control_variable)
            control_variable[control_variable == 0] = np.nan
            mean = np.nanmean(control_variable)
            control_variable = np.around(ones * mean)

        item = (control_variable, f0, voiced, target)
        return item

    def __len__(self):
        return self.datalen

    def get_control_var(self, utt, voiced, f0):
        return f0

    def target_arr(self, utt, use_amazon=False):
        filename = os.path.join(self.target_dir, utt + '.npy')
        data = np.load(filename)
        data = data.T
        return data

    @staticmethod
    def utterance_list(wav_dir):
        l = os.listdir(wav_dir)
        li = [x.split('.')[0] for x in l]
        return li


class CoGDataset(Spec2SpecDataset):
    def get_control_var(self, utt, voiced, f0):
        mvf, CoG, st = param_arr_new(utt, voiced, self.num_bins, return_index=self.control_var_as_index)
        return CoG

def get_training_dataloaders(dataset, train_batch_size, val_batch_size, validation_split, shuffle_dataset=False):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    print("There are {} utterances in the validation set.".format(split))
    random_seed = 42
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_indices = train_indices[:len(train_indices) - (len(train_indices) % train_batch_size)]
    val_indices = val_indices[:len(val_indices) - (len(val_indices) % val_batch_size)]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    dataloader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler, pin_memory=True, num_workers=2)
    valid_dataloader = DataLoader(dataset, batch_size=val_batch_size, sampler=valid_sampler, pin_memory=True,
                                  num_workers=2)

    return dataloader, valid_dataloader

# TODO rationalise all below into one sensible function?


def to_onehot(f0s, frange=(60, 500), num_bins=256, hot_val=100., return_index=False, shmir=False, unvoiced_val=0.):
    # Takes array of f0s and range of freqs to encode.
    # Smoosh means shmir the hot val across all bins for unvoiced sections, Default behaviour is to
    # receive interpolated signal for unvoiced regions.
    # todo make num bins hyperparam

    if return_index:
        output = np.zeros(len(f0s))
    else:
        output = np.zeros((len(f0s), num_bins))

    for i, f0 in enumerate(f0s):
        index = num2bin(f0, num_bins, frange)
        # Supports 3 behaviours at this stage, appending index, making onehot or shmiring for unvoiced regions
        if return_index:
            output[i] = int(index)
        elif shmir and f0 == unvoiced_val:
            output[i] = hot_val / num_bins
        else:
            output[i][int(index)] = 1.

    return output


def num2bin(x, num_bins, frange):
    if x < frange[0]:
        index = 0
    elif x > frange[1]:
        index = num_bins - 1
    else:
        index = (x - frange[0]) * (num_bins / (frange[1] - frange[0]))


    if int(index) >= num_bins:
        index = num_bins-1
    elif int(index) < 0:
        index = 0
    return int(index)

def test_scale():
    """ Test the maxes from the specs """
    import hparams
    hp = hparams.get_hparams()
    ds = HFCDataset(hp)
    for i in ds:
        cv, _, _, mel = i
        plt.plot(cv.squeeze())
        plt.show()
        exit()

if __name__ == '__main__':
    test_scale()
