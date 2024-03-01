import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from speechbrain.nnet import CNN, linear
from speechbrain.nnet.embedding import Embedding
from speechbrain.lobes.models.FastSpeech2 import PostNet

from speechbrain.lobes.models.transformer.Transformer import (
    TransformerEncoder,
    PositionalEncoding,
)

class FastSpeech2Combiner(nn.Module):
    def __init__(
        self,
        h, 
        n_mels,
        n_hidden,
        n_speaker,
    ):
        super().__init__()
        self.n_hidden = n_hidden

        self.dec_num_head = h.dec_num_head
        self.sinusoidal_positional_embed_decoder = PositionalEncoding(
            h.dec_d_model
        )
        # This replaces the customary transformer encoder encoder
        self.encoder_linear = linear.Linear(
            n_neurons=h.enc_d_model, input_size=n_hidden,
        )

        if h.use_pretrained_spkr_emb: 
            self.speakerEmbed = linear.Linear(n_neurons=h.enc_d_model, input_size=192)
        else:
            self.speakerEmbed = Embedding(
                n_speaker, h.enc_d_model
            )

        self.pitchEmbed = CNN.Conv1d(
            in_channels=1,
            out_channels=h.enc_d_model,
            kernel_size=h.pitch_pred_kernel_size,
            padding="same",
            skip_transpose=True,
        )

        self.energyEmbed = CNN.Conv1d(
            in_channels=1,
            out_channels=h.enc_d_model,
            kernel_size=h.energy_pred_kernel_size,
            padding="same",
            skip_transpose=True,
        )

        self.decoder = TransformerEncoder(
            num_layers=h.dec_num_layers,
            nhead=h.dec_num_head,
            d_ffn=h.dec_ffn_dim,
            d_model=h.dec_d_model,
            kdim=h.dec_k_dim,
            vdim=h.dec_v_dim,
            dropout=h.dec_dropout,
            activation=nn.ReLU,
            normalize_before=h.normalize_before,
            ffn_type=h.ffn_type,
            ffn_cnn_kernel_size_list=h.ffn_cnn_kernel_size_list,
        )

        self.linear = linear.Linear(n_neurons=n_mels, input_size=h.dec_d_model)
        self.postnet = PostNet(
            n_mel_channels=n_mels,
            postnet_embedding_dim=h.postnet_embedding_dim,
            postnet_kernel_size=h.postnet_kernel_size,
            postnet_n_convolutions=h.postnet_n_convolutions,
            postnet_dropout=h.postnet_dropout,
        )


    def forward(
        self,
        hidden,
        speaker_id=None,
        pitch=None,
        energy=None,
    ):
        B, n_frame, n_hidden_ = hidden.shape
        assert self.n_hidden == n_hidden_, f"{self.n_hidden} != {n_hidden_} {hidden.shape}"

        hidden = self.encoder_linear(hidden)

        if speaker_id is not None:
            speaker_id = speaker_id.squeeze(1)
            speaker_id = self.speakerEmbed(speaker_id)
            # repeat speaker embedding for each decoder time step
            speaker_id = speaker_id.repeat(1, n_frame, 1)
            hidden = hidden.add(speaker_id)

        if pitch is not None:
            # Mean and Variance Normalization
            mean = 256.1732939688805
            std = 328.319759158607
            pitch = (pitch - mean) / std
            pitch = self.pitchEmbed(pitch.permute(0, 2, 1))
            pitch = pitch.permute(0, 2, 1)
            hidden = hidden.add(pitch)

        if energy is not None:
            energy = self.energyEmbed(energy)
            energy = energy.permute(0, 2, 1)
            hidden = hidden.add(energy)

        # decoder
        pos = self.sinusoidal_positional_embed_decoder(hidden)
        hidden = torch.add(hidden, pos)

        output_mel_feats, memory, *_ = self.decoder(
            hidden,
        )

        # postnet
        mel_post = self.linear(output_mel_feats)
        postnet_output = self.postnet(mel_post) + mel_post
        return (
            mel_post,
            postnet_output,
        )
