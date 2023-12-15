import torch
import torch.nn as nn

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
        batch_size, seq_length, width = x.shape
        x = x.view(batch_size * seq_length, 1, -1)
        xs = [c(x) for c in self.parallel_components]
        output = sum(xs)
        output += self.residual(x)
        output = output.view(batch_size, seq_length, -1)
        return output


class Combiner(nn.Module):
    def __init__(self, hidden_size, spectral_width, output_width, n_layers, num_bins, dropout=0.1, rnn_mult=3.):
        super(Combiner, self).__init__()
        # Insert some random noise into the net -- set the width here
        # Replacing noise with f0, TODO fix
        self.noise_width = 513
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.trans_size = 3 * spectral_width

        f0_width = num_bins
        control_variable_width = num_bins

        self.f0_embedding = nn.Embedding(num_bins, num_bins)
        self.cv_embedding = nn.Embedding(num_bins, num_bins)

        self.dropout = nn.Dropout(dropout)

        self.f0_PTCB = ParallelTransposedConvolutionalBlock(f0_width, self.trans_size)
        self.control_variable_PTCB = ParallelTransposedConvolutionalBlock(control_variable_width, self.trans_size)
        self.spectral_PTCB = ParallelTransposedConvolutionalBlock(spectral_width, self.trans_size)

        self.merge_voicing = nn.Linear(2 * self.trans_size, self.trans_size)
        self.aperiodicity_lin = nn.Linear(self.noise_width, self.trans_size)
        self.control_variable_lin = nn.Linear(control_variable_width, self.trans_size)
        self.spectral_residual = nn.Linear(spectral_width, self.trans_size)

        self.rnn = nn.GRU(4 * self.trans_size, hidden_size, n_layers, dropout=rnn_mult * dropout, batch_first=True)
        self.lin = nn.Linear(hidden_size, output_width)

    def forward(self, features):
        spectral, control_variable, f0, is_voiced = features

        batch_size = spectral.shape[0]
        seq_length = spectral.shape[1]
        spectral_width = spectral.shape[2]

        # Using f0 onehot instead of embedding. TODO use embedding instead
        #f0 = self.f0_embedding(f0)
        print(control_variable)
        cv = self.cv_embedding(control_variable)
        print(cv.shape)
        f0_width = f0.shape[2]

        noise = torch.rand((batch_size, seq_length, self.noise_width))
        if spectral.is_cuda:
            noise = noise.cuda()

        noise = self.dropout(F.relu(self.aperiodicity_lin(noise)))

        control_variable = self.control_variable_PTCB(control_variable)

        spectral = self.spectral_PTCB(spectral)

        f0 = self.f0_PTCB(f0)

        unvoiced_f0 = is_voiced * f0
        f0 = self.merge_voicing(torch.cat((unvoiced_f0, f0), 2))

        # SECTION: Combine F0 and spectral features
        x = torch.cat((spectral, f0, control_variable, noise), 2)
        out, _ = self.rnn(x)

        # Apply linear layer to RNN output
        output = self.lin(out)

        return output
