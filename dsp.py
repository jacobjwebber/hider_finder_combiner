# Set of dsp functions for hfc
import torch
from torchaudio import functional as fa
import torchaudio
import math

import numpy as np
import pyworld
from librosa.filters import mel as librosa_mel_fn

from until_patched import InverseMelScale
import hparams


test_plots = False
if test_plots:
    from matplotlib import pyplot as plt

demo_sample = 'l_hvd_177.wav'
csv_file = 'l_hvd_177.csv'

# Used for reading in Oliviers values
col_names = ['t', 'f0', 'I',' Rd', 'CoG', 'ST',
             'MVF_HNM', 'MFV_AS_IHPC', 'MVF_AS_IHPC_ICPC']


def normalise_mel(mel):
    mel = mel + 11.5129
    mel = mel / (11.5129 + 2)
    return mel

def denormalise_mel(mel):
    mel = mel * (11.5129 + 2)
    mel = mel - 11.5129
    return mel


def plotspect(spect, name=''):
    plt.imshow(spect.numpy(), aspect='auto')
    plt.title(name)
    plt.gca().invert_yaxis()
    if name == '':
        plt.show()
    else:
        plt.savefig(name + '.png')
    plt.clf()


# Hifigan helpers
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


class FeatureEngineer:
    """ Class with methods for returning HFC features """

    def __init__(self, hp):
        """
        Takes an object containing all the necessary hyper params and instantiates
        a list of feature generating methods
        """
        self.hp = hp

        #self.spectrogram = torchaudio.transforms.Spectrogram(hp.n_fft, win_length=hp.win_length,
        #        hop_length=hp.hop_length, power=2., normalized=True)
        #self.mel_scale = torchaudio.transforms.MelScale(hp.num_mels, hp.sr, n_stft=hp.n_fft // 2 + 1)
        #self.i_mel_scale = InverseMelScale(hp.n_fft // 2 + 1, n_mels=hp.num_mels, sample_rate=hp.sr)
        #self.a_to_DB = torchaudio.transforms.AmplitudeToDB()
        #self.g_l = torchaudio.transforms.GriffinLim(n_fft=hp.n_fft, win_length=hp.win_length,
        #        hop_length=hp.hop_length,
        #        power=2.)

        # for hifigan mel
        self.mel_basis = {}
        self.hann_window = {}

    def griffin_lim(self, s):
        return self.g_l(s)

    def DB_to_a(self, s):
        return fa.DB_to_amplitude(s, 1., 1.)

    def load_wav(self, wav_name):
        y, sr = torchaudio.load(wav_name)
        y = self.resample(y, sr)
        return y

    def resample(self, wav, sr):
        if self.hp.sr != sr:
            rs = torchaudio.transforms.Resample(sr, self.hp.sr)
            wav = rs(wav)
        return wav

    def save_wav(self, wav, wav_name):
        torchaudio.save(wav_name, wav, self.hp.sr)

    def wav_to_s(self, y):
        s = self.spectrogram(y)
        return s

    def s_to_mel(self, s):
        mel = self.mel_scale(s)
        mel = self.a_to_DB(mel)
        mel = self.normalize(mel)
        return mel

    def hifigan_mel_spectrogram(self, y):
        #, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
        # taken from hifigan repo
        if torch.min(y) < -1.:
            pass #print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            pass #print('max value is ', torch.max(y))

        if self.hp.fmax not in self.mel_basis:
            mel = librosa_mel_fn(sr=self.hp.sr, n_fft=self.hp.n_fft, n_mels=self.hp.num_mels, fmin=self.hp.fmin, fmax=self.hp.fmax)
            self.mel_basis[str(self.hp.fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
            self.hann_window[str(y.device)] = torch.hann_window(self.hp.win_length).to(y.device)

        y = torch.nn.functional.pad(
                    y.unsqueeze(1), (int((self.hp.n_fft-self.hp.hop_length)/2), int((self.hp.n_fft-self.hp.hop_length)/2)), mode='reflect')

        y = y.squeeze(1)
        spec = torch.stft(y, self.hp.n_fft, hop_length=self.hp.hop_length, win_length=self.hp.win_length, window=self.hann_window[str(y.device)],
                          center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
        spec = torch.matmul(self.mel_basis[str(self.hp.fmax)+'_'+str(y.device)], spec)
        spec = spectral_normalize_torch(spec)
        return spec

    def mel_to_s(self, mel):
        mel = self.denormalize(mel)
        mel = self.DB_to_a(mel)
        s = self.i_mel_scale(mel)
        return s

    def normalize(self, S):
        scale = self.hp.max_level_db - self.hp.min_level_db
        return torch.clip((S - self.hp.min_level_db) / scale, 0, 1)

    def denormalize(self, S):
        scale = self.hp.max_level_db - self.hp.min_level_db
        return S * scale + self.hp.min_level_db

    def cog(self, s):
        min_freq = 80.
        max_freq = 5000.
        bins = self.hp.n_fft // 2 + 1
        nyquist = self.hp.sr / 2
        min_index = int((min_freq * bins) / nyquist)
        max_index = int((max_freq * bins) / nyquist)
        freqs = torch.linspace(0, nyquist, steps=bins,
                               device=s.device).reshape((-1, 1))
        freq_dim = -2
        freqs = freqs[min_index:max_index, :]
        s = s[:, min_index:max_index, :]
        return (freqs * s).sum(dim=freq_dim) / s.sum(dim=freq_dim)


    def intensity(self, s):
        return torch.sum(s, dim=1)


    def st(self, s):
        """ This function depends on numpy until I can think of some better way
        of doing the linear fit without np.polyfit """
        s = s.squeeze().numpy()
        lin_freqs = np.linspace(0, self.hp.sr/2, num=int(self.hp.n_fft/2+1))

        min_freq = 80.
        max_freq = 5000.
        bins = self.hp.n_fft // 2 + 1
        nyquist = self.hp.sr / 2
        min_index = int((min_freq * bins) / nyquist)
        max_index = int((max_freq * bins) / nyquist)

        lin_freqs = lin_freqs[min_index:max_index]
        # Crop lin spec in frequency axis
        s = s[min_index:max_index,:]

        # The order of the poly -- 0 is offset, 1 is slope
        order = 1
        # equiv to 10*log10(s) Check for negative numbers in spect
        db = 5*np.log10(s**2)
        st = np.polyfit(np.log10(lin_freqs), db, order)

        debug = False
        if debug:
            i = 50

            print((lin_freqs * st[0][i]).shape)
            line = st[1][i] + (lin_freqs * st[0][i])
            print(st[:,i])
            plt.plot(line, label='line')
            plt.plot(lin_spec[:,i], label='spec')
            plt.show()
            exit()

        return torch.from_numpy(st[0])

    def f0(self, y):
        unvoiced_value = 0.0
        hop_ms = self.hp.hop_length * 1000 / self.hp.sr
        x = y.squeeze().numpy()
        x = x.astype(np.float64)
        f0, _, aperiodicity = pyworld.wav2world(x, self.hp.sr, frame_period=hop_ms)
        is_voiced = np.logical_not(np.isclose(f0, unvoiced_value * np.ones_like(f0), atol=1e-6))
        f0 = np.expand_dims(f0, 1)
        f0 = self.interpolate(f0, is_voiced)
        is_voiced = np.expand_dims(is_voiced, 1)


        #if not return_values:
        #    f0 = to_onehot(f0, num_bins=num_bins, return_index=f0_as_index)
        source = (torch.from_numpy(f0.T), torch.from_numpy(is_voiced.T.astype(float)))
        return source

    def crop_to_shortest(self, *args):
        """ Different features may be slightly different in length, due to
        design choices arrounding padding etc
        sequence length must be the LAST parameter"""
        shortest = min([f.shape[-1] for f in args])
        # crop from start --- could crop from end instead?
        return (f[...,f.shape[-1] - shortest:] for f in args)


    def interpolate(self, signal, is_voiced):
        """Linearly interpolates the signal in unvoiced regions such that there are no discontinuities.
        Args:
            signal (tensor [n_frames, feat_dim]): Temporal signal.
            is_voiced (tensor [n_frames]<bool>): Boolean array indicating if each frame is voiced.
        Returns:
            (tensor [n_frames, feat_dim]): Interpolated signal, same shape as signal.
        """
        n_frames = signal.shape[0]
        feat_dim = signal.shape[1]

        # Initialise whether we are starting the search in voice/unvoiced.
        in_voiced_region = is_voiced[0]

        last_voiced_frame_i = None
        for i in range(n_frames):
            if is_voiced[i]:
                if not in_voiced_region:
                    # Current frame is voiced, but last frame was unvoiced.
                    # This is the first voiced frame after an unvoiced sequence, interpolate the unvoiced region.

                    # If the signal starts with an unvoiced region then `last_voiced_frame_i` will be None.
                    # Bypass interpolation and just set this first unvoiced region to the current voiced frame value.
                    if last_voiced_frame_i is None:
                        signal[:i + 1] = signal[i]

                    # Use `np.linspace` to create a interpolate a region that includes the bordering voiced frames.
                    else:
                        start_voiced_value = signal[last_voiced_frame_i]
                        end_voiced_value = signal[i]

                        unvoiced_region_length = (i + 1) - last_voiced_frame_i
                        interpolated_region = np.linspace(start_voiced_value, end_voiced_value, unvoiced_region_length)
                        interpolated_region = interpolated_region.reshape((unvoiced_region_length, feat_dim))

                        signal[last_voiced_frame_i:i + 1] = interpolated_region

                # Move pointers forward, we are waiting to find another unvoiced section.
                last_voiced_frame_i = i

            in_voiced_region = is_voiced[i]

        # If the signal ends with an unvoiced region then it would not have been caught in the loop.
        # Similar to the case with an unvoiced region at the start we can bypass the interpolation.
        if not in_voiced_region:
            signal[last_voiced_frame_i:] = signal[last_voiced_frame_i]

        return signal

def bin_tensor(tensor, num_bins, min_val, max_val):
    min_val = math.log(min_val)
    max_val = math.log(max_val)

    zeros_mask = torch.logical_not(torch.isclose(tensor, torch.zeros_like(tensor))).int()
    tensor = torch.log(tensor) * zeros_mask
    bins = torch.linspace(min_val, max_val, num_bins-1, device=tensor.device)
    bin_indices = torch.bucketize(tensor, bins)
    return bin_indices

def onehot_index(xs, num_bins, x_range):
    xs = xs.squeeze()
    print(xs.shape)
    indexs = torch.zeros_like(xs)

    for i, x in enumerate(xs):
        if x < x_range[0]:
            indexs[i] = 0
        elif x > x_range[1]:
            indexs[i] = num_bins - 1
        else:
            indexs[i] = int((x - x_range[0]) * (num_bins / (x_range[1] - x_range[0])))


        if int(indexs[i]) >= num_bins:
            indexs[i] = num_bins-1
        elif int(indexs[i]) < 0:
            indexs[i] = 0
    return indexs


def test_get_f0(y, fe):
    f0, vuv = fe.f0(y)
    plt.plot(f0.squeeze())
    plt.clf()


def test_intensity_mel(s, fe):
    i = fe.intensity(s)
    mel = fe.s_to_mel(s)
    s_new = fe.mel_to_s(mel)
    i_new = fe.intensity(s_new)
    mse = torch.nn.MSELoss()
    print(mse(i / i.max(), i_new / i.max()))
    if test_plots:
        plt.plot(i.squeeze(), label='full')
        plt.plot(i_new.squeeze(), label='mel')
        plt.legend()
        plt.savefig('test_i_mel.png')
        plt.clf()


def test_st_mel(s, fe):
    st = fe.st(s)
    mel = fe.s_to_mel(s)
    s_new = fe.mel_to_s(mel)
    st_new = fe.st(s_new)
    if test_plots:
        plt.plot(st.squeeze(), label='full')
        plt.plot(st_new.squeeze(), label='mel')
        plt.legend()
        plt.savefig('test_st_mel.png')
        plt.clf()

    mse = torch.nn.MSELoss()
    print(mse(st / st.max(), st_new / st.max()))

def test_st(s, fe):
    params = pd.read_csv(csv_file, names=col_names)
    ost = params['ST'].to_numpy()
    ost = torch.from_numpy(ost)
    st = fe.st(s)
    if test_plots:
        plt.plot(st.squeeze(), label='python')
        plt.plot(ost.squeeze(), label='matlab')
        plt.legend()
        plt.savefig('test_st.png')
        plt.clf()


def test_cog(s, fe):
    params = pd.read_csv(csv_file, names=col_names)
    oc = params['CoG'].to_numpy()
    oc = torch.from_numpy(oc)
    c = fe.cog(s)
    c, oc = fe.crop_to_shortest(c, oc)
    mse = torch.nn.MSELoss()
    print(mse(oc, c.squeeze()))


def test_cog_mel(s, fe):
    c = fe.cog(s)
    mel = fe.s_to_mel(s)
    s_new = fe.mel_to_s(mel)
    c_new = fe.cog(s_new)
    mse = torch.nn.MSELoss()
    if test_plots:
        plt.plot(c.squeeze(), label='full')
        plt.plot(c_new.squeeze(), label='mel')
        plt.legend()
        plt.savefig('test_cog_mel.png')
        plt.clf()
    print(mse(c / c.max(), c_new / c.max()))


def test_mel(s, fe):
    mel = fe.s_to_mel(s)
    s_new = fe.mel_to_s(mel)
    mse = torch.nn.MSELoss()
    print(mse(s, s_new))

def test_scale(s, fe):
    m = fe.mel_scale(s)
    n = fe.i_mel_scale(m)
    mse = torch.nn.MSELoss()
    print(mse(s, n))

def test_norm_denorm(s, fe):
    n = fe.normalize(s)
    dn = fe.denormalize(n)
    mse = torch.nn.MSELoss()
    print(mse(s, dn))

def test_a_db(s, fe):
    db = fe.a_to_DB(s)
    a = fe.DB_to_a(db)
    mse = torch.nn.MSELoss()
    print(mse(s, a))

def test_feature_engineer():
    hp = hparams.get_hparams()
    fe = FeatureEngineer(hp)
    y = fe.load_wav(demo_sample)
    s = fe.wav_to_s(y)
    mel = fe.s_to_mel(s)
    s_est = fe.mel_to_s(mel)

    test_norm_denorm(s, fe)
    test_a_db(s, fe)
    test_mel(s, fe)
    test_scale(s, fe)
    test_cog(s, fe)
    test_cog_mel(s, fe)
    test_st(s, fe)
    test_st_mel(s, fe)
    test_intensity_mel(s, fe)
    test_get_f0(y, fe)
    mse = torch.nn.MSELoss()
    print(s_est.mean(), s.mean())


def mel_to_cog(mel, hp):
    i_mel_scale = InverseMelScale(hp.n_fft // 2 + 1, n_mels=hp.num_mels, sample_rate=hp.sr)
    mel = denormalize(mel, hp)
    mel = fa.DB_to_amplitude(mel, 1., 10.)
    lin = i_mel_scale(mel)
    cog = lin2cog(lin.transpose(0,1), hp.sr, hp.n_fft)
    return cog


def main():
    hp = hparams.get_hparams()
    params = pd.read_csv(csv_file, names=col_names)
    orig_full_cog = params['CoG'].to_numpy()
    hp = hparams.get_hparams()
    y, sr = torchaudio.load(demo_sample)
    spectrogram = torchaudio.transforms.Spectrogram(hp.n_fft,
            hop_length=hp.hop_length, power=2., normalized=False)
    s = spectrogram(y)
    cog = lin2cog(s, sr, hp.n_fft)
    """
    plt.plot(cog[0])
    plt.plot(orig_full_cog)
    plt.show()
    """

    mel_scale = torchaudio.transforms.MelScale(hp.num_mels, hp.sr, n_stft=hp.n_fft // 2 + 1)
    mel = mel_scale(s)
    F = mel_scale.fb.transpose(0,1)
    F_plus = F.pinverse()
    s_est = torch.matmul(F_plus, mel)

    """
    cog2 = lin2cog(s_est, sr, hp.n_fft)
    plt.plot(cog[0], label='torch audio')
    plt.plot(cog2[0], label='inverse mel')
    plt.plot(orig_full_cog, label='MATLAB')
    plt.legend()
    plt.savefig('cog_works.png')
    """
    o_st = params['ST'].to_numpy()
    print(o_st)
    st = lin2spectral_tilt(s[0], sr, hp.n_fft)
    plt.plot(st)
    plt.plot(o_st)
    plt.show()


def main_spec_invert():
    hp = hparams.get_hparams()

    #plotspect(mel_scale.fb)
    inverse_mel_scale = torchaudio.transforms.InverseMelScale(hp.n_fft // 2 + 1, hp.num_mels, hp.sr) #, tolerance_loss=1.) #max_iter=1000)
    a_to_DB = torchaudio.transforms.AmplitudeToDB()
    gl = torchaudio.transforms.GriffinLim(hp.n_fft, hop_length=hp.hop_length)

    y, sr = torchaudio.load(demo_sample)
    print(sr)
    spectrogram = torchaudio.transforms.Spectrogram(hp.n_fft, hop_length=hp.hop_length, power=2., normalized=True)
    s = a_to_DB(spectrogram(y))
    mel_scale = torchaudio.transforms.MelScale(hp.num_mels, hp.sr, n_stft=hp.n_fft // 2 + 1)
    mel = mel_scale(s)
    F = mel_scale.fb.transpose(0,1)
    F_plus = F.pinverse()
    mel2 = torch.matmul(F, s).squeeze()

    plotspect(F, 'F')
    plotspect(F_plus, 'F_plus')

    s_est2 = torch.matmul(F_plus, mel)

    s_est, other = torch.lstsq(mel2, F)
    #plt.plot(s_est[:,140])
    #plt.plot(s[0,:,140])
    #plt.show()

    x = gl(s_est)
    exit()

    mse = torch.nn.MSELoss()

    print(mse(s_est, s.squeeze()))
    print(mse(s_est2, s.squeeze()))
    print(mse(s_est2, s_est))

    estimated_spect = inverse_mel_scale(mel)
    print(mse(estimated_spect, s))

    plotspect(s_est, 'approx_s')
    plotspect(estimated_spect.squeeze())
    plotspect(s.squeeze(), 'real_s')


def interpolate(signal, is_voiced):
    """Linearly interpolates the signal in unvoiced regions such that there are no discontinuities.
    Args:
	signal (np.ndarray[n_frames, feat_dim]): Temporal signal.
	is_voiced (np.ndarray[n_frames]<bool>): Boolean array indicating if each frame is voiced.
    Returns:
	(np.ndarray[n_frames, feat_dim]): Interpolated signal, same shape as signal.
    """
    n_frames = signal.shape[0]
    feat_dim = signal.shape[1]

    # Initialise whether we are starting the search in voice/unvoiced.
    in_voiced_region = is_voiced[0]

    last_voiced_frame_i = None
    for i in range(n_frames):
        if is_voiced[i]:
            if not in_voiced_region:
		# Current frame is voiced, but last frame was unvoiced.
                # This is the first voiced frame after an unvoiced sequence, interpolate the unvoiced region.

                # If the signal starts with an unvoiced region then `last_voiced_frame_i` will be None.
                # Bypass interpolation and just set this first unvoiced region to the current voiced frame value.
                if last_voiced_frame_i is None:
                    signal[:i + 1] = signal[i]

		# Use `np.linspace` to create a interpolate a region that includes the bordering voiced frames.
                else:
                    start_voiced_value = signal[last_voiced_frame_i]
                    end_voiced_value = signal[i]

                    unvoiced_region_length = (i + 1) - last_voiced_frame_i
                    interpolated_region = np.linspace(start_voiced_value, end_voiced_value, unvoiced_region_length)
                    interpolated_region = interpolated_region.reshape((unvoiced_region_length, feat_dim))

                    signal[last_voiced_frame_i:i + 1] = interpolated_region

	    # Move pointers forward, we are waiting to find another unvoiced section.
            last_voiced_frame_i = i

        in_voiced_region = is_voiced[i]

    # If the signal ends with an unvoiced region then it would not have been caught in the loop.
    # Similar to the case with an unvoiced region at the start we can bypass the interpolation.
    if not in_voiced_region:
        signal[last_voiced_frame_i:] = signal[last_voiced_frame_i]

    return signal

def onehotify(hot_bin, shape):
    batch_size, seq_len, num_bins = shape
    #print(f0)
    hot_bin = hot_bin.view(batch_size, seq_len, 1).long()
    onehot = torch.FloatTensor(batch_size, seq_len, num_bins, device=hot_bin.device)
    onehot.zero_()
    onehot.scatter_(2, hot_bin, 1)
    return onehot


# Funtions for extracting vocal force parameters
#####################################################

# TODO delete this and replace with torch
def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))

def lin2cog(lin_spec, sr, n_fft):
    lin_spec = lin_spec.numpy()
    lin_freqs = np.linspace(0, sr/2, num=int(n_fft/2+1))
    cogs = []
    mindex, maxdex = indices(lin_freqs, 80, 5000)
    for lin_spe in lin_spec:
        lin_spe = lin_spe[mindex:maxdex]
        cogs.append(np.dot(lin_freqs[mindex:maxdex], lin_spe) / sum(lin_spe))

    return np.array(cogs)


def lin2spectral_tilt(lin_spec, sr, n_fft):
    print(lin_spec.shape)
    lin_spec = lin_spec.numpy()
    lin_freqs = np.linspace(0, sr/2, num=int(n_fft/2+1))
    mindex, maxdex = indices(lin_freqs, 80, 5000)

    lin_freqs = lin_freqs[mindex:maxdex]
    # Crop lin spec in frequency axis
    lin_spec = lin_spec[mindex:maxdex,:]

    # The order of the poly -- 0 is offset, 1 is slope
    order = 1
    st = np.polyfit(np.log10(lin_freqs), _amp_to_db(lin_spec), order)
    print(st.shape)

    debug = False
    if debug:
        i = 50

        print((lin_freqs * st[0][i]).shape)
        line = st[1][i] + (lin_freqs * st[0][i])
        print(st[:,i])
        plt.plot(line, label='line')
        plt.plot(lin_spec[:,i], label='spec')
        plt.show()
        exit()

    return st[0]

# TODO should not need this function
def indices(arr, fmin, fmax):
    mindex = None
    for i, val in enumerate(arr):
        if val > fmin and mindex is None:
            mindex = i

        if val > fmax:
            maxdex = i
            return mindex, maxdex

if __name__ == '__main__':
    test_feature_engineer()
