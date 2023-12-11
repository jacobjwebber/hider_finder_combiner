from pathlib import Path
from typing import Tuple, Union
import random
import torch
from torchaudio import datasets
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm
import dsp
import hydra


def pad_sequence(batch, max_len=88200):
    """
    Pad a batch of sequences to a maximum length.

    Args:
        batch (List[Tensor]): The batch of sequences to be padded.
        max_len (int, optional): The maximum length to pad the sequences to. Defaults to 88200.

    Returns:
        Tensor: The padded batch of sequences.
    """
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


def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets



class MyLibri(datasets.LIBRITTS):
    def __init__(self, hp, download=False):
        self.hp = hp
        super().__init__(hp.dataset.root, hp.dataset.subset, hp.dataset.save_as, download)
        self.root = hp.dataset.root
        self.fe = dsp.FeatureEngineer(self.hp)
        self.populate_speaker_idx()
    
    def populate_speaker_idx(self):

        # Create map from speaker string to an int index
        speaker_idx_path = os.path.join(self.root, 'speaker2idx.json')
        if os.path.exists(speaker_idx_path):
            with open(speaker_idx_path, 'r') as f:
                print('hellpo')
                speaker2idx = json.load(f)
        else:
            speaker2idx = {}
            for row in tqdm(self):
                if row["speaker"] not in speaker2idx:
                    speaker2idx[row["speaker"]] = len(speaker2idx)

            with open(speaker_idx_path, 'w') as f:
                json.dump(speaker2idx, f)

        self.speaker2idx = speaker2idx
        return speaker2idx


    def __getitem__(self, n: int):
        """
        Retrieve the item at the given index.

        Args:
            n (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the following attributes:
                - waveform: The waveform of the item.
                - sample_rate: The sample rate of the waveform.
                - original_text: The original text of the item.
                - normalized_text: The normalized text of the item.
                - speaker_id: The ID of the speaker.
                - chapter_id: The ID of the chapter.
                - utterance_id: The ID of the utterance.
        """
        (waveform,
        sample_rate,
        original_text,
        normalized_text,
        speaker_id,
        chapter_id,
        utterance_id) = super().__getitem__(n)

        waveform = self.fe.resample(waveform, sample_rate)
        f0 = self.fe.f0(waveform)
        mel = self.fe.hifigan_mel_spectrogram(waveform)
        speaker_id = self.speaker2idx[speaker_id]


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
        }



@hydra.main(version_base=None, config_path='config', config_name="config")
def main(hp):
    # Create speaker2idx and phone2idx
    
    dataset = MyLibri(hp, download=True)
    # load speaker2ix from json file if it exists
    for item in tqdm(dataset):
        pass

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collate_fn,
    )


if __name__ == '__main__':
    main()

"""
class HFCDataModule(pl.LightningDataModule):
    def __init__(self, config) -> None:
        Initializes a new instance of the class.

        Args:
            config (Any): The configuration object.

        Returns:
            None
        super().__init__()
        self.config = config
    
    def setup(self, stage: str) -> None:
        print('Setting up dataset')

        self.ds = MyLibri('train.txt', self.config, sort=True, drop_last=True)
        self.val_ds = MyLibri('val.txt', self.config, sort=False, drop_last=False)
        self.group_size = 1  # Number of groups for sorting, it;s 4 in the original codebase

    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.config.training.batch_size * self.group_size,
            shuffle=True,
            collate_fn=None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.training.batch_size * self.group_size,
            shuffle=False,
            collate_fn=None,
        )

"""