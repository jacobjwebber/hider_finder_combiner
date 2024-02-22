import random
import torch
from torchaudio import datasets
from torch.utils.data import DataLoader, random_split, Subset
import json
import os
from tqdm import tqdm
import dsp
import hydra
import lightning.pytorch as pl
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
#from speech_embeddings import melspect
from speechbrain.pretrained import MelSpectrogramEncoder
from pathlib import Path as P

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

def melspect_no(waveform):
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

def load_split_from_json(train_json_path, valid_json_path, test_json_path, dataset):
    # Downloaded splits from google drive linked to at the bottom of this page https://huggingface.co/speechbrain/tts-hifigan-libritts-22050Hz
    # Need to match training split of hifigan for any audio domain validation results to be accurate
    with open(train_json_path, 'r') as f:
        train = json.load(f)
    with open(valid_json_path, 'r') as f:
        valid = json.load(f)
    with open(test_json_path, 'r') as f:
        test = json.load(f)
    
    split_indices = {'train': [], 'valid': [], 'test': []}
    for i, datum in enumerate(tqdm(dataset)):
        utterance_id = datum['utterance']
        if utterance_id in train:
            split_indices['train'].append(i)
        elif utterance_id in valid:
            split_indices['valid'].append(i)
        elif utterance_id in test:
            split_indices['test'].append(i)
        else:
            print('oh fuck')
            exit()
    
    json.dump(split_indices, open('split_indices.json', 'w'))

    return train, valid, test

class HFCDataModule(pl.LightningDataModule):
    def __init__(self, config, model='finder', download=True):
        super().__init__()
        self.config = config
        self.dataset = MyLibri(self.config, download=download)
        self.dataset.populate_speaker_idx()
        self.n_speakers = self.dataset.n_speakers
        #self.valid_split = self.config.dataset.validation_split
        if model == 'finder':
            self.batch_size = self.config.training_finder.batch_size
        elif model == 'hfc':
            self.batch_size = self.config.training.batch_size

        # TODO fix the below so that it reads from a file or else random splits
        splits = json.load(open('split_indices.json', 'r'))
        self.valid_set = Subset(self.dataset, splits['valid'][:config.dataset.valid_size])
        self.train_set = Subset(self.dataset, splits['train'])
        #self.valid_set, self.train_set = random_split(self.dataset, [self.valid_split, 1. - self.valid_split], generator=torch.Generator().manual_seed(42))

    
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=8,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn_val,
            num_workers=16,
        )
    
    
def collate_fn_val(batch):
    return collate_fn(batch) #, max_len=10000)

def collate_fn(batch, max_len=400):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    mels, f0s, ids, vuvs, spkr_embs = [], [], [], [], []

    # Gather in lists, and encode labels as indices
    for sample in batch:
        mels += [sample['mel'].squeeze()]
        f0s += [sample['f0']]
        vuvs += [sample['vuv']]
        ids += [sample['speaker']]
        spkr_embs += [sample['spkr_emb']]

    # Group the list of tensors into a batched tensor
    start_indices, end_indices = crop_indices(mels, max_len=max_len)
    new_mels = []
    new_f0s = []
    new_vuvs = [] 
    for mel, f0, vuv, start_index, end_index in zip(mels, f0s, vuvs, start_indices, end_indices):
        assert (end_index - start_index) % 4 == 0, f'{end_index - start_index}'
        new_mels.append(mel[:,start_index:end_index].transpose(0,1))
        new_f0s.append(f0[:,start_index:end_index].transpose(0,1))
        new_vuvs.append(vuv[:,start_index:end_index].transpose(0,1))
    
    mels = torch.nn.utils.rnn.pad_sequence(new_mels, batch_first=True)
    f0s = torch.nn.utils.rnn.pad_sequence(new_f0s, batch_first=True)
    vuvs = torch.nn.utils.rnn.pad_sequence(new_vuvs, batch_first=True)

    ids = torch.LongTensor(ids)
    spkr_embs = torch.stack(spkr_embs)
    return mels, f0s, vuvs, ids, spkr_embs
    #out = self.combiner((mel, speaker_id, spkr_emb, f0_idx, is_voiced))

def crop_indices(batch, max_len=200):
    batch = [item.t() for item in batch]
    start_indices = []
    end_indices = []
    for item in batch:
        seq_len = item.shape[0]
        last_poss_index = seq_len - max_len
        # Select a random section of the waveform
        if last_poss_index > 0:
            start_index = random.randrange(last_poss_index)
        else:
            start_index = 0
        # Prevent out range error by using the min function
        end_index = min([start_index + max_len, seq_len])
        end_index = end_index - ((end_index - start_index) % 4) # Pad to 4
        start_indices.append(start_index)
        end_indices.append(end_index)

    return start_indices, end_indices



class MyLibri(datasets.LIBRITTS):
    def __init__(self, hp, download=False, device='cpu'):
        self.hp = hp
        #super().__init__(hp.dataset.root, hp.dataset.subset, hp.dataset.save_as, download)
        self.root = os.path.expanduser(hp.dataset.root) # torchaudio breaks unless this
        super().__init__(self.root, hp.dataset.subset, hp.dataset.save_as, download=download)
        self.fe = dsp.FeatureEngineer(self.hp)
        self.spkr_emb_encoder = None
        self.device = device
    
    
    def populate_speaker_idx(self):

        # Create map from speaker string to an int index
        speaker_idx_path = os.path.join(self.root, self.hp.dataset.save_as, 'speaker2idx.json')
        if os.path.exists(speaker_idx_path):
            with open(speaker_idx_path, 'r') as f:
                self.speaker2idx = json.load(f)
        else:
            self.speaker2idx = {}
            for n in tqdm(range(len(self)), desc="Populating speaker (this should only happen once)"):
                (_, _, _, _, speaker_id, _, _) = super().__getitem__(n)


                if speaker_id not in self.speaker2idx:
                    self.speaker2idx[speaker_id] = len(self.speaker2idx)

            with open(speaker_idx_path, 'w') as f:
                json.dump(self.speaker2idx, f)
            
        self.n_speakers = len(self.speaker2idx)
        return self.speaker2idx


    def __getitem__(self, n: int):
        (waveform,
        sample_rate,
        original_text,
        normalized_text,
        speaker_id,
        chapter_id,
        utterance_id) = super().__getitem__(n)

        file = os.path.join(self.root, self.hp.dataset.save_as, '{}', f'{speaker_id}', f'{chapter_id}', f'{utterance_id}.pt')

        try:
            mel = torch.load(file.format('mel'), map_location='cpu')
            f0 = torch.load(file.format('f0'), map_location='cpu')
            vuv = torch.load(file.format('vuv'), map_location='cpu')
            spkr_emb = torch.load(file.format('spkr_emb'), map_location='cpu')
        except FileNotFoundError:
            waveform = self.fe.resample(waveform, sample_rate)
            f0, vuv = self.fe.f0(waveform)
            # mel = self.fe.hifigan_mel_spectrogram(waveform)
            mel = melspect(waveform.to(self.device))
            if not self.spkr_emb_encoder:
                self.spkr_emb_encoder = MelSpectrogramEncoder.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb-mel-spec", 
                    savedir="spk_emb_encoder_checkpoints", 
                    run_opts={"device": "cpu"})
            #mel16 = torch.nn.functional.interpolate(mel.unsqueeze(0), scale_factor=16000/22050)
            spkr_emb = self.spkr_emb_encoder.encode_mel_spectrogram(torch.clone(mel))
            def save(obj, file):
                os.makedirs(os.path.dirname(file), exist_ok=True) # need to create dir if doesn't exist
                torch.save(obj, file)
            save(mel, file.format('mel'))
            save(f0, file.format('f0'))
            save(vuv, file.format('vuv'))
            save(spkr_emb, file.format('spkr_emb'))

        f0, vuv, mel = self.fe.crop_to_shortest(f0, vuv, mel)
        speaker_id = self.speaker2idx[str(speaker_id)]
        # TODO crude norm

        mel = dsp.normalise_mel(mel)

        return {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'original_text': original_text,
            'normalized_text': normalized_text,
            'speaker': speaker_id,
            'chapter': chapter_id,
            'utterance': utterance_id,
            'f0': f0,
            'mel': mel,
            'vuv': vuv,
            'spkr_emb': spkr_emb,
        }




@hydra.main(version_base=None, config_path='config', config_name="config")
def main(hp):
    # Create speaker2idx and phone2idx
    
    dataset = MyLibri(hp, download=True, device=torch.device('cuda:3'))
    dataset.populate_speaker_idx()

    # The below was only done once -- moving from hifigan split json to my own
    load_split_from_json(hp.dataset.train_json, hp.dataset.valid_json, hp.dataset.test_json, dataset)

    # load speaker2ix from json file if it exists

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collate_fn,
        num_workers=60,
    )
    for data in tqdm(dataset):
        pass
        #mel, f0, vuv, speaker_id, spkr_emb = data


if __name__ == '__main__':
    main()
