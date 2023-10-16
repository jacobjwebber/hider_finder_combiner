# -*- coding: utf-8 -*-
# Until my implementation for InverseMelScale gets patched to torchaudio, keep it here

import math
from typing import Callable, Optional
from warnings import warn

import torch
from torch import Tensor
from torchaudio import functional as F
from torchaudio.compliance import kaldi


class InverseMelScale(torch.nn.Module):
    r"""Solve for a normal STFT from a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    It uses a pseudo inverse of the FB matrix to retrieve STFT from mel frequency STFT

    Args:
        n_stft (int): Number of bins in STFT. See ``n_fft`` in :class:`Spectrogram`.
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
    """
    __constants__ = ['n_stft', 'n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self,
                 n_stft: int,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.,
                 f_max: Optional[float] = None):
        super(InverseMelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max or float(sample_rate // 2)
        self.f_min = f_min

        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(f_min, self.f_max)

        fb = F.create_fb_matrix(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate)
        # Create Moore-Penrose Pseudo inverse
        fb_plus = fb.transpose(0,1).pinverse()
        self.register_buffer('fb', fb)
        self.register_buffer('fb_plus', fb_plus)

    def forward(self, melspec: Tensor) -> Tensor:
        r"""
        Args:
            melspec (Tensor): A Mel frequency spectrogram of dimension (..., ``n_mels``, time)

        Returns:
            Tensor: Linear scale spectrogram of size (..., freq, time)
        """

        return torch.matmul(self.fb_plus, melspec)

