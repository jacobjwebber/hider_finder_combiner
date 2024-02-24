import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class Generator(torch.nn.Module):
    def __init__(self, h, input_size, output_size, n_spkrs=None):
        # input_size, kernel_sizes, dilation_sizes, mid_channel):
        super(Generator, self).__init__()
        self.num_kernels = len(h.kernel_sizes)
        resblock = ResBlock1
        if n_spkrs is not None and h.use_pretrained_spkr_emb:
            raise NotImplementedError
        elif n_spkrs:
            self.spkr_layer = nn.Embedding(n_spkrs, h.speaker_chans)
            input_size += h.speaker_chans

        self.conv_pre = weight_norm(Conv1d(input_size, h.mid_channel, 7, 1, padding=3))

        # Have removed the upsampling feature
        self.resblocks = nn.ModuleList()
        ch = h.mid_channel
        for j, (k, d) in enumerate(zip(h.kernel_sizes, h.dilation_sizes)):
            self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, output_size, 7, 1, padding=3))
        self.conv_post.apply(init_weights)

    def forward(self, x, spkr=None):
        
        B, n_mels, n_frame_ = x.shape
        if spkr is not None:
            spkr = self.spkr_layer(spkr)
            spkr = spkr.unsqueeze(2).repeat(1, 1, n_frame_)
            x = torch.cat([x, spkr], dim=1)

        x = self.conv_pre(x)

        x = F.leaky_relu(x, LRELU_SLOPE)
        xs = None
        for j in range(self.num_kernels):
            if xs is None:
                xs = self.resblocks[j](x)
            else:
                xs += self.resblocks[j](x)
        x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)