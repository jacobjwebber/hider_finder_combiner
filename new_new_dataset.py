from typing import Tuple
from torch import Tensor
from torchaudio import datasets
from torch.utils.data import DataLoader
from speech_collator import SpeechCollator, create_speaker2idx, create_phone2idx
from speech_collator.measures import PitchMeasure, EnergyMeasure

dataset = datasets.LIBRITTS('/disk/scratch2/s1116548/data', download=True)

class MyLibri(datasets.LIBRITTS):
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

        return {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'original_text': original_text,
            'normalized_text': normalized_text,
            'speaker_id': speaker_id,
            'chapter_id': chapter_id,
            'utterance_id': utterance_id,} 

# Create speaker2idx and phone2idx
speaker2idx = create_speaker2idx(dataset)
phone2idx = create_phone2idx(dataset)

speech_collator = SpeechCollator(
    speaker2idx=speaker2idx,
    phone2idx=phone2idx,
    measures=[PitchMeasure(), EnergyMeasure()],
    return_keys=["measures"]
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=speech_collator.collate_fn,
)

for item in dataloader:
    print(item)

exit
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