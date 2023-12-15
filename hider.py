import torch.nn as nn
import torch.nn.functional as F

class Hider(nn.Module):
    def __init__(self, hidden_size, input_width, output_width, n_layers, dropout=0.1, denoising=0.1):
        """
        Network takes an input sequence of variable length but fixed input_width
        Returns sequence of probability dists of width num_bins.
        """
        super(Hider, self).__init__()
        self.input_width = input_width
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        kernel_size = 10
        rnn_mult = 3  # multiply dropout by this number when in rnn
        padding = 0
        stride = 1
        dilation = 1
        self.conv_out_width = int(
            1 + ((input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride)
        )
        self.dropout = nn.Dropout(dropout)
        # dropout for the output layer to for denoising autoencoder
        self.denoising = nn.Dropout(denoising)
        self.residual = nn.Linear(self.input_width, self.conv_out_width)
        self.fc1 = nn.Linear(self.conv_out_width, self.conv_out_width)
        self.conv = nn.Conv1d(1, 1, kernel_size, dilation=dilation, stride=stride, padding=padding)
        self.rnn = nn.GRU(self.conv_out_width, hidden_size, n_layers, dropout=rnn_mult * dropout, batch_first=True)
        self.lin = nn.Linear(hidden_size, output_width)

    def forward(self, spectrograms):
        shape = spectrograms.shape
        batch_size, length, width = shape
        # Arrange shape so that middle dimension is 1 -- number of conv channels
        # batch and length are squished to one dimension
        x = spectrograms.view(batch_size * length, 1, -1)
        # Pass through x as only modified by linear layer -- possibly bypassing conv
        x = F.relu(self.conv(x)) + F.relu(self.residual(x))
        x = self.dropout(F.relu(x))
        x = x.view((batch_size, length, self.conv_out_width))
        x = self.fc1(x)
        x = self.dropout(F.relu(x))
        x, hidden = self.rnn(x)
        x = self.dropout(F.relu(x))
        x = self.lin(x)
        x = self.denoising(x)
        return x

    def init_hidden(self, batch_size, use_cuda=False):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        if use_cuda:
            hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda()
        else:
            hidden = weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
        return hidden
