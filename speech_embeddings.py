from speechbrain.pretrained import MelSpectrogramEncoder
from speechbrain.pretrained import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
from new_new_dataset import MyLibri
from new_new_dataset import melspect as melspect4
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Subset

from torch.nn.functional import mse_loss
import torchaudio
import hydra
import torch
import numpy as np

from matplotlib import pyplot as ppt

def melspect(waveform):
    """use these exact values for pretrained spect"""
    spectrogram, _ = mel_spectogram(
        audio=waveform.squeeze(),
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )
    return spectrogram

def melspect(waveform):
    """use these exact values for pretrained spect"""
    #waveform = torchaudio.functional.resample(waveform, 22050, 16000)
    spectrogram, _ = mel_spectogram(
        audio=waveform.squeeze(),
        sample_rate=22050, #16000,#22050,#16000,#22050,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        n_fft=1024,
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )
    return spectrogram

def melspect2(waveform, spk_emb_encoder):
    waveform = torchaudio.functional.resample(waveform, 22050, 16000)
    return spk_emb_encoder.mel_spectogram(waveform)

def melspect3(waveform, spk_emb_encoder):

    return spk_emb_encoder.mel_spectogram(waveform)



@hydra.main(version_base=None, config_path='config', config_name="config")
def main(hp):
    
    dataset = MyLibri(hp, download=True)
    # load speaker2ix from json file if it exists
    dataset.populate_speaker_idx()

    spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb-mel-spec", savedir="spk_emb_encoder_checkpoints", run_opts={"device": "cuda"})

    hifigan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-22050Hz", savedir="hifigan_checkpoints", run_opts={"device": 'cuda'})

    # Create DataLoader
    last_emb = None
    last_emb2 = None
    coses = []
    coses3 = []
    coses4 = []
    for i, data in enumerate(tqdm(dataset)):
        waveform = data['waveform']
        mel1 = data['mel']

        #mel2 = melspect2(waveform, spk_emb_encoder)
        mel2 = melspect(waveform)
        mel3 = melspect4(waveform)
        print(mel1.shape, mel2.shape, mel3.shape)
        #waveform = hifigan.decode_batch(mel)
        #waveform = hifigan.decode_batch(mel)
        fig, axes = ppt.subplots(1, 3)
        axes[0].imshow(mel1.squeeze().numpy())

        axes[1].imshow(mel3.squeeze().numpy())
        axes[2].imshow(mel2.squeeze().numpy())
        ppt.show()
        wav1 = hifigan.decode_batch(mel1)
        wav2 = hifigan.decode_batch(mel2)
        torchaudio.save('waveform.wav', waveform.cpu(), 22050)
        torchaudio.save('waveform1.wav', wav1.cpu(), 22050)
        torchaudio.save('waveform2.wav', wav2.cpu(), 22050)
        if i == 2:
            exit()
        break
        mel4 = torch.nn.functional.interpolate(mel.unsqueeze(0), size=mel2.shape[2])
        mel3 = torch.nn.functional.interpolate(mel.unsqueeze(0), scale_factor=16000/22050)
        print(mel4.shape, data['mel'].shape, mel3.shape, mel2.shape)

        emb = spk_emb_encoder.encode_mel_spectrogram(mel)
        emb2 = spk_emb_encoder.encode_mel_spectrogram(mel2.squeeze())
        emb3 = spk_emb_encoder.encode_mel_spectrogram(mel3.squeeze())
        emb4 = spk_emb_encoder.encode_mel_spectrogram(mel4.squeeze())
        #print(mel.max(), mel2.max())
        #print(mel2.shape, mel.shape)
        #print(emb - emb2)
        #print(mse_loss(emb , emb2))
        cos = torch.nn.functional.cosine_similarity(emb, emb2, dim=2)
        cos3 = torch.nn.functional.cosine_similarity(emb, emb3, dim=2)
        cos4 = torch.nn.functional.cosine_similarity(emb, emb4, dim=2)

        if last_emb is not None:
            coslast = torch.nn.functional.cosine_similarity(emb, last_emb, dim=2)
            coslast2 = torch.nn.functional.cosine_similarity(emb2, last_emb2, dim=2)
        last_emb = emb
        last_emb2 = emb2
        coses.append(cos.item())
        coses3.append(cos3.item())
        coses4.append(cos4.item())
        #print(cos)
        #print(cos4)
        if i > 500:
            break
        fig, axes = ppt.subplots(1, 2)
        #axes[0].imshow(mel.squeeze().numpy())
        #print((mel-mel2).max())
        #print(mel.mean())
        #print(mel2.mean())

        #axes[1].imshow((mel-mel2).squeeze().numpy())
        #ppt.show()

        #exit()
    print(np.mean(coses))
    print(np.mean(coses3))
    print(np.mean(coses4))


        

if __name__ == '__main__':
    main()