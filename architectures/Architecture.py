import torch.nn as nn
from torchvision.models import resnet50
import torch
import torch.nn.functional as F

class MyArchitecture(nn.Module):
  def __init__(self, num_classes):
    super(MyArchitecture, self).__init__()
    self.model = resnet50()
    self.drop1 = nn.Dropout(p=0.25)
    self.lin1 = nn.Linear(in_features=1000, out_features=256)
    self.drop2 = nn.Dropout(p=0.5)
    self.lin2 = nn.Linear(in_features=256, out_features=num_classes)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.model(x)
    x = self.drop1(x)
    x = x.reshape(x.shape[0], -1)
    x = self.relu(self.lin1(x))
    x = self.drop2(x)
    x = self.lin2(x)
    return x





factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]


class WSConv2d(nn.Module):

  def __init__(
          self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2
  ):
    super(WSConv2d, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.scale = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
    self.bias = self.conv.bias
    self.conv.bias = None

    # initialize conv layer
    nn.init.normal_(self.conv.weight)
    nn.init.zeros_(self.bias)

  def forward(self, x):
    return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
  def __init__(self):
    super(PixelNorm, self).__init__()
    self.epsilon = 1e-8

  def forward(self, x):
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, use_pixelnorm=True):
    super(ConvBlock, self).__init__()
    self.use_pn = use_pixelnorm
    self.conv1 = WSConv2d(in_channels, out_channels)
    self.conv2 = WSConv2d(out_channels, out_channels)
    self.leaky = nn.LeakyReLU(0.2)
    self.pn = PixelNorm()

  def forward(self, x):
    x = self.leaky(self.conv1(x))
    x = self.pn(x) if self.use_pn else x
    x = self.leaky(self.conv2(x))
    x = self.pn(x) if self.use_pn else x
    return x


class Generator(nn.Module):
  def __init__(self, z_dim, in_channels, img_channels=3):
    super(Generator, self).__init__()

    # initial takes 1x1 -> 4x4
    self.initial = nn.Sequential(
      PixelNorm(),
      nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),
      nn.LeakyReLU(0.2),
      WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
      nn.LeakyReLU(0.2),
      PixelNorm(),
    )

    self.initial_rgb = WSConv2d(
      in_channels, img_channels, kernel_size=1, stride=1, padding=0
    )
    self.prog_blocks, self.rgb_layers = (
      nn.ModuleList([]),
      nn.ModuleList([self.initial_rgb]),
    )

    for i in range(
            len(factors) - 1
    ):
      conv_in_c = int(in_channels * factors[i])
      conv_out_c = int(in_channels * factors[i + 1])
      self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
      self.rgb_layers.append(
        WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
      )

  def fade_in(self, alpha, upscaled, generated):

    return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

  def forward(self, x, alpha, steps):
    out = self.initial(x)

    if steps == 0:
      return self.initial_rgb(out)

    for step in range(steps):
      upscaled = F.interpolate(out, scale_factor=2, mode="nearest")
      out = self.prog_blocks[step](upscaled)

    final_upscaled = self.rgb_layers[steps - 1](upscaled)
    final_out = self.rgb_layers[steps](out)
    return self.fade_in(alpha, final_upscaled, final_out)


