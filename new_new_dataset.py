import random
import torch
from torchaudio import datasets
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import dsp
import hydra
import lightning.pytorch as pl


class HFCDataModule(pl.LightningDataModule):
    def __init__(self, config, model='finder'):
        super().__init__()
        self.config = config
        self.dataset = MyLibri(self.config, download=True)
        self.dataset.populate_speaker_idx()
        self.n_speakers = self.dataset.n_speakers
        if model == 'finder':
            self.batch_size = self.config.training_finder.batch_size
        elif model == 'hfc':
            self.batch_size = self.config.training.batch_size
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=8,
        )
    
    
    

def collate_fn(batch, max_len=200):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    mels, f0s, ids, vuvs = [], [], [], []

    # Gather in lists, and encode labels as indices
    for sample in batch:
        mels += [sample['mel'].squeeze()]
        f0s += [sample['f0']]
        vuvs += [sample['vuv']]
        ids += [sample['speaker']]

    # Group the list of tensors into a batched tensor
    mels = pad_sequence(mels, max_len=max_len)
    f0s = pad_sequence(f0s, max_len=max_len)
    vuvs = pad_sequence(vuvs, max_len=max_len)
    ids = torch.LongTensor(ids)
    

    return mels, f0s, vuvs, ids

def pad_sequence(batch, max_len=200):
    batch = [item.t() for item in batch]
    new_batch = []
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
        cropped_item = item[start_index:end_index]
        new_batch.append(cropped_item)

    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch[:,:max_len]
    return batch



class MyLibri(datasets.LIBRITTS):
    def __init__(self, hp, download=False):
        self.hp = hp
        super().__init__(hp.dataset.root, hp.dataset.subset, hp.dataset.save_as, download)
        self.root = hp.dataset.root
        self.fe = dsp.FeatureEngineer(self.hp)
    
    def populate_speaker_idx(self):

        # Create map from speaker string to an int index
        speaker_idx_path = os.path.join(self.root, 'speaker2idx.json')
        if os.path.exists(speaker_idx_path):
            with open(speaker_idx_path, 'r') as f:
                self.speaker2idx = json.load(f)
        else:
            self.speaker2idx = {}
            for n in tqdm(range(len(self)), desc="Populating speaker"):
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
            mel = torch.load(file.format('mel'))
            f0 = torch.load(file.format('f0'))
            vuv = torch.load(file.format('vuv'))
        except FileNotFoundError:
            waveform = self.fe.resample(waveform, sample_rate)
            f0, vuv = self.fe.f0(waveform)
            mel = self.fe.hifigan_mel_spectrogram(waveform)
            def save(obj, file):
                os.makedirs(os.path.dirname(file), exist_ok=True) # need to create dir if doesn't exist
                torch.save(obj, file)
            save(mel, file.format('mel'))
            save(f0, file.format('f0'))
            save(vuv, file.format('vuv'))

        f0, vuv, mel = self.fe.crop_to_shortest(f0, vuv, mel)
        speaker_id = self.speaker2idx[str(speaker_id)]

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
        }




@hydra.main(version_base=None, config_path='config', config_name="config")
def main(hp):
    # Create speaker2idx and phone2idx
    
    dataset = MyLibri(hp, download=True)
    dataset.populate_speaker_idx()
    # load speaker2ix from json file if it exists

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collate_fn,
        num_workers=10,
    )
    for data in tqdm(dataloader):
        pass #print([datum.shape for datum in data])


if __name__ == '__main__':
    main()
