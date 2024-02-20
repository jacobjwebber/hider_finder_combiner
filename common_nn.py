import torch
import torch.nn as nn
import torch.nn.functional as F

# Contains component modules that are used in more than one module

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


    
class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


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
        x = x.reshape(batch_size * seq_length, 1, -1)
        xs = [c(x) for c in self.parallel_components]
        output = sum(xs)
        output += self.residual(x)
        output = output.view(batch_size, seq_length, -1)
        return output