import torch
import torch.nn as nn
import torch.nn.functional as F

class Combiner(nn.Module):
    def __init__(self, hp, n_speakers):
        #hidden_size, spectral_width, output_width, n_layers, num_bins, dropout=0.1, rnn_mult=3.):
        super(Combiner, self).__init__()
        # Insert some random noise into the net -- set the width here
        # Replacing noise with f0, TODO fix
        self.noise_width = 513
        self.hidden_size = hp.combiner.rnn_size
        self.n_layers = hp.combiner.n_layers
        self.trans_size = 3 * hp.num_mels
        self.use_f0 = hp.use_f0

        f0_dim = hp.control_variables.f0_bins
        speaker_emb_dim = hp.control_variables.speaker_embedding_dim

        self.f0_embedding = nn.Embedding(f0_dim, f0_dim)
        #self.speaker_embedding = nn.Embedding(n_speakers, speaker_emb_dim) # TODO replace with real speaker embedding?
        self.dropout = nn.Dropout(hp.combiner.drop)

        if self.use_f0:
            self.f0_PTCB = ParallelTransposedConvolutionalBlock(f0_dim, self.trans_size)
            self.merge_voicing = nn.Linear(2 * self.trans_size, self.trans_size)

        self.control_variable_PTCB = ParallelTransposedConvolutionalBlock(speaker_emb_dim, self.trans_size)
        self.spectral_PTCB = ParallelTransposedConvolutionalBlock(hp.model.hidden_size, self.trans_size)

        self.aperiodicity_lin = nn.Linear(self.noise_width, self.trans_size)
        self.control_variable_lin = nn.Linear(speaker_emb_dim, self.trans_size)
        self.spectral_residual = nn.Linear(hp.model.hidden_size, self.trans_size)

        if self.use_f0:
            self.rnn = nn.GRU(4 * self.trans_size, self.hidden_size, hp.combiner.n_layers, dropout=3. * hp.combiner.drop, batch_first=True)
        else:
            self.rnn = nn.GRU(3 * self.trans_size, self.hidden_size, hp.combiner.n_layers, dropout=3. * hp.combiner.drop, batch_first=True)
        self.lin = nn.Linear(self.hidden_size, hp.num_mels)

    def forward(self, features):
        spectral, speaker_id, spkr_emb, f0, is_voiced = features

        batch_size = spectral.shape[0]
        seq_length = spectral.shape[1]
        spectral_width = spectral.shape[2]

        #print(f'seq_length = {seq_length}')

        # Using f0 onehot instead of embedding. TODO use embedding instead
        #print(f0.max())
        #speaker_id = self.speaker_embedding(speaker_id)
        speaker_id = spkr_emb.squeeze(1)
        speaker_id = speaker_id.repeat(1, seq_length, 1)

        noise = torch.rand((batch_size, seq_length, self.noise_width), device=spectral.device)

        noise = self.dropout(F.relu(self.aperiodicity_lin(noise)))

        speaker_id = self.control_variable_PTCB(speaker_id)

        spectral = self.spectral_PTCB(spectral)
        if self.use_f0:
            f0 = self.f0_embedding(f0)
            f0 = self.f0_PTCB(f0)
            unvoiced_f0 = is_voiced * f0
            f0 = self.merge_voicing(torch.cat((unvoiced_f0, f0), 2))

        # SECTION: Combine F0 and spectral features
        if self.use_f0:
            x = torch.cat((spectral, f0, speaker_id, noise), 2)
        else:
            x = torch.cat((spectral, speaker_id, noise), 2)
        out, _ = self.rnn(x)

        # Apply linear layer to RNN output
        output = self.lin(out)

        return output



class ParallelTransposedConvolutionalBlock(nn.Module):
    """A block that applies a block of transposed convolutions in parallel and sums the result"""

    def __init__(self, input_width, output_width):
        super(ParallelTransposedConvolutionalBlock, self).__init__()

        self.kernel_size = 50
        padding = 45
        out_padding = 0
        stride = 1

        self.residual = nn.Sequential(
            nn.Linear(input_width, output_width),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.parallel_components = nn.ModuleList()

        for dilation in range(1, 20, 2):
            conv = nn.ConvTranspose1d(1, 1, self.kernel_size,
                                      stride=stride, padding=padding,
                                      output_padding=out_padding, groups=1, bias=True,
                                      dilation=dilation)
            # Docs do not mention dilation wrt output width
            # Everyone hates convolution maths
            conv_size = (input_width - 1) * stride - 2 * padding + self.kernel_size + out_padding + \
                        (self.kernel_size - 1) * (dilation - 1)
            resize = nn.Linear(conv_size, output_width)
            component = nn.Sequential(conv, resize, nn.ReLU())
            self.parallel_components.append(component)

    def forward(self, x):
        #print(x.shape)
        batch_size, seq_length, width = x.shape
        x = x.view(batch_size * seq_length, 1, -1)
        xs = [c(x) for c in self.parallel_components]
        output = sum(xs)
        output += self.residual(x)
        output = output.view(batch_size, seq_length, -1)
        return output
    
