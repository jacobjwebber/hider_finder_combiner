import torch.nn as nn
import torch.nn.functional as F
from common_nn import LinearNorm

class Hider(nn.Module):
    def __init__(self, hp):
        #hidden_size, input_width, output_width, n_layers, dropout=0.1, denoising=0.1):
        """
        Network takes an input sequence of variable length but fixed input_width
        Returns sequence of probability dists of width num_bins.
        """
        super(Hider, self).__init__()
        self.input_width = hp.num_mels
        self.output_width = hp.model.hidden_size
        self.n_layers = hp.hider.n_layers
        self.hidden_size = hp.hider.rnn_size
        self.dropout = hp.hider.drop
        self.denoising = hp.hider.denoising
        kernel_size = 9
        rnn_mult = hp.hider.rnn_mult
        padding = 4
        stride = 1
        dilation = 1
        self.conv_out_width = int(
            1 + ((hp.num_mels + 2 * padding - dilation * (kernel_size - 1) - 1) / stride)
        )
        self.dropout = nn.Dropout(self.dropout)
        # dropout for the output layer to for denoising autoencoder
        self.denoising = nn.Dropout(self.denoising)
        self.fc1 = LinearNorm(self.conv_out_width, self.conv_out_width)
        self.conv = nn.Conv2d(1, 1, kernel_size, dilation=dilation, stride=stride, padding=padding) 
        self.rnn = nn.GRU(self.conv_out_width, self.hidden_size, self.n_layers, dropout=rnn_mult * hp.hider.drop, batch_first=True)
        self.lin = LinearNorm(self.hidden_size, self.output_width)

    def forward(self, spectrograms):
        shape = spectrograms.shape
        batch_size, width, length = shape
        #print(f'length = {length}')
        # Arrange shape so that middle dimension is 1 -- number of conv channels
        x = spectrograms.transpose(1,2).unsqueeze(1)
        # Pass through x as only modified by linear layer -- possibly bypassing conv
        x = F.relu(self.conv(x)) 
        x = self.dropout(F.relu(x))
        x = x.view((batch_size, length, self.conv_out_width)) # TODO replace with squeeze?
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
