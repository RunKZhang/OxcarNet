import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union

# class SincConv2D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', sampling_rate=256):
#         super(SincConv2D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels 
#         self.kernel_size = kernel_size # Expected to be: (1, L)
#         self.stride = stride 
#         self.padding =  padding
#         self.sampling_rate = sampling_rate

#         # set the low and band of filter
#         self.low_hz = nn.Parameter(torch.rand(out_channels) * 50)
#         self.band_hz = nn.Parameter(torch.rand(out_channels) * (self.sampling_rate / 2 -50) + 50)

#         # use hamming window
#         n = torch.arange(self.kernel_size[1] // 2)
#         self.window = 0.54 - 0.46 * torch.cos(2 * np.pi * n/ self.kernel_size[1]//2)
#         self.window = self.window.to(torch.get_default_dtype())

#     def forward(self, x):
#         low = self.low_hz.abs()
#         high = (self.low_hz + self.band_hz).abs()

#         # Make filters
#         t_right = torch.linspace(1, (self.kernel_size[1] - 1) // 2, steps=(self.kernel_size[1] - 1) // 2, device=x.device) / self.sampling_rate
#         filters = torch.zeros((self.out_channels, self.kernel_size[1]), device=x.device)
#         filters[:, 0] = 2 * (high-low)
#         filters[:, 1:] = (torch.sin(2 * np.pi * high.unsqueeze(1) * t_right) - torch.sin(2 * np.pi * low.unsqueeze(1) * t_right)) / (np.pi * t_right)
#         filters = filters * self.window
#         filters = filters / (2 * high[:, None] - 2 * low[:, None])

#         filters = filters.view(self.out_channels, 1, 1, self.kernel_size[1]).to(x.dtype)
#         return F.conv2d(x, filters, stride=1, pooling=(0, self.kernel_size[1]//2))


class SincConv2D(nn.Module):
    """Sinc Convolution.

    This module performs a convolution using Sinc filters in time domain as kernel.
    Sinc filters function as band passes in spectral domain.
    The filtering is done as a convolution in time domain, and no transformation
    to spectral domain is necessary.

    This implementation of the Sinc convolution is heavily inspired
    by Ravanelli et al. https://github.com/mravanelli/SincNet,
    and adapted for the ESpnet toolkit.
    Combine Sinc convolutions with a log compression activation function, as in:
    https://arxiv.org/abs/2010.07597

    Notes:
    Currently, the same filters are applied to all input channels.
    The windowing function is applied on the kernel to obtained a smoother filter,
    and not on the input values, which is different to traditional ASR.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        window_func: str = "hamming",
        scale_type: str = "linear",
        fs: Union[int, float] = 16000,
    ):
        """Initialize Sinc convolutions.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Sinc filter kernel size (needs to be an odd number).
            stride: See torch.nn.functional.conv1d.
            padding: See torch.nn.functional.conv1d.
            dilation: See torch.nn.functional.conv1d.
            window_func: Window function on the filter, one of ["hamming", "none"].
            fs (str, int, float): Sample rate of the input data
        """
        # assert check_argument_types()
        super().__init__()
        window_funcs = {
            "none": self.none_window,
            "hamming": self.hamming_window,
        }
        if window_func not in window_funcs:
            raise NotImplementedError(
                f"Window function has to be one of {list(window_funcs.keys())}",
            )
        self.window_func = window_funcs[window_func]
        scale_choices = {
            "mel": MelScale,
            "bark": BarkScale,
            "linear": LinearScale
        }
        if scale_type not in scale_choices:
            raise NotImplementedError(
                f"Scale has to be one of {list(scale_choices.keys())}",
            )
        self.scale = scale_choices[scale_type]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.fs = float(fs)
        if self.kernel_size % 2 == 0:
            raise ValueError("SincConv: Kernel size must be odd.")
        self.f = None
        N = self.kernel_size // 2
        self._x = 2 * np.pi * torch.linspace(1, N, N)
        self._window = self.window_func(torch.linspace(1, N, N))
        # init may get overwritten by E2E network,
        # but is still required to calculate output dim
        self.init_filters()

    @staticmethod
    def sinc(x: torch.Tensor) -> torch.Tensor:
        """Sinc function."""
        x2 = x + 1e-6
        return torch.sin(x2) / x2

    @staticmethod
    def none_window(x: torch.Tensor) -> torch.Tensor:
        """Identity-like windowing function."""
        return torch.ones_like(x)

    @staticmethod
    def hamming_window(x: torch.Tensor) -> torch.Tensor:
        """Hamming Windowing function."""
        L = 2 * x.size(0) + 1
        x = x.flip(0)
        return 0.54 - 0.46 * torch.cos(2.0 * np.pi * x / L)

    def init_filters(self):
        """Initialize filters with filterbank values."""
        f = self.scale.bank(self.out_channels, self.fs)
        # print(f'f in freq.shape: {f}')
        f = torch.div(f, self.fs)
        # print(f'f in time.shape: {f}')
        self.f = torch.nn.Parameter(f, requires_grad=True)
        # print(f'f.shape: {f}')

    def _create_filters(self, device: str):
        """Calculate coefficients.

        This function (re-)calculates the filter convolutions coefficients.
        """
        f_mins = torch.abs(self.f[:, 0])
        f_maxs = torch.abs(self.f[:, 0]) + torch.abs(self.f[:, 1] - self.f[:, 0])

        self._x = self._x.to(device)
        self._window = self._window.to(device)

        f_mins_x = torch.matmul(f_mins.view(-1, 1), self._x.view(1, -1))
        f_maxs_x = torch.matmul(f_maxs.view(-1, 1), self._x.view(1, -1))

        kernel = (torch.sin(f_maxs_x) - torch.sin(f_mins_x)) / (0.5 * self._x)
        kernel = kernel * self._window

        kernel_left = kernel.flip(1)
        kernel_center = (2 * f_maxs - 2 * f_mins).unsqueeze(1)
        filters = torch.cat([kernel_left, kernel_center, kernel], dim=1)
        
        # filters = filters.view(filters.size(0), 1, filters.size(1))
        filters = filters.view(filters.size(0), 1, 1, filters.size(1))
        self.sinc_filters = filters
        # print(f'self.sinc_filters.size(): {self.sinc_filters.size()}')

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Sinc convolution forward function.

        Args:
            xs: Batch in form of torch.Tensor (B, C_in, D_in).

        Returns:
            xs: Batch in form of torch.Tensor (B, C_out, D_out).
        """
        self._create_filters(xs.device)
        # xs = torch.nn.functional.conv1d(
        #     xs,
        #     self.sinc_filters,
        #     padding=self.padding,
        #     stride=self.stride,
        #     dilation=self.dilation,
        #     groups=self.in_channels,
        # )
        xs = torch.nn.functional.conv2d(
            xs,
            self.sinc_filters,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.in_channels,
        )
        return xs

    def get_odim(self, idim: int) -> int:
        """Obtain the output dimension of the filter."""
        D_out = idim + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        D_out = (D_out // self.stride) + 1
        return D_out


class MelScale:
    """Mel frequency scale."""

    @staticmethod
    def convert(f):
        """Convert Hz to mel."""
        return 1125.0 * torch.log(torch.div(f, 700.0) + 1.0)
    # @staticmethod
    # def convert(f):
    #     """Convert Hz to mel."""
    #     return f

    @staticmethod
    def invert(x):
        """Convert mel to Hz."""
        return 700.0 * (torch.exp(torch.div(x, 1125.0)) - 1.0)
    # def invert(x):
    #     """Convert mel to Hz."""
    #     return x

    @classmethod
    def bank(cls, channels: int, fs: float) -> torch.Tensor:
        """Obtain initialization values for the mel scale.

        Args:
            channels: Number of channels.
            fs: Sample rate.

        Returns:
            torch.Tensor: Filter start frequencíes.
            torch.Tensor: Filter stop frequencies.
        """
        # assert check_argument_types()
        # min and max bandpass edge frequencies
        min_frequency = torch.tensor(30.0)
        max_frequency = torch.tensor(fs * 0.5)
        print(f'max_frequency: {max_frequency}')
        frequencies = torch.linspace(
            cls.convert(min_frequency), cls.convert(max_frequency), channels + 2
        )
        print(f'frequencies: {frequencies}')
        frequencies = cls.invert(frequencies)
        print(f'frequencies: {frequencies}')
        f1, f2 = frequencies[:-2], frequencies[2:]
        print(f'f1:{f1}')
        return torch.stack([f1, f2], dim=1)


class BarkScale:
    """Bark frequency scale.

    Has wider bandwidths at lower frequencies, see:
    Critical bandwidth: BARK
    Zwicker and Terhardt, 1980
    """

    @staticmethod
    def convert(f):
        """Convert Hz to Bark."""
        b = torch.div(f, 1000.0)
        b = torch.pow(b, 2.0) * 1.4
        b = torch.pow(b + 1.0, 0.69)
        return b * 75.0 + 25.0

    @staticmethod
    def invert(x):
        """Convert Bark to Hz."""
        f = torch.div(x - 25.0, 75.0)
        f = torch.pow(f, (1.0 / 0.69))
        f = torch.div(f - 1.0, 1.4)
        f = torch.pow(f, 0.5)
        return f * 1000.0

    @classmethod
    def bank(cls, channels: int, fs: float) -> torch.Tensor:
        """Obtain initialization values for the Bark scale.

        Args:
            channels: Number of channels.
            fs: Sample rate.

        Returns:
            torch.Tensor: Filter start frequencíes.
            torch.Tensor: Filter stop frequencíes.
        """
        # assert check_argument_types()
        # min and max BARK center frequencies by approximation
        min_center_frequency = torch.tensor(70.0)
        max_center_frequency = torch.tensor(fs * 0.45)
        center_frequencies = torch.linspace(
            cls.convert(min_center_frequency),
            cls.convert(max_center_frequency),
            channels,
        )
        center_frequencies = cls.invert(center_frequencies)

        f1 = center_frequencies - torch.div(cls.convert(center_frequencies), 2)
        f2 = center_frequencies + torch.div(cls.convert(center_frequencies), 2)
        return torch.stack([f1, f2], dim=1)

class LinearScale:
    """Linear frequency scale."""

    @staticmethod
    def convert(f):
        """No Conversion."""
        return f

    @staticmethod
    def invert(x):
        """No inversion."""
        return x

    @classmethod
    def bank(cls, channels: int, fs: float) -> torch.Tensor:
        """Obtain initialization values for the mel scale.

        Args:
            channels: Number of channels.
            fs: Sample rate.

        Returns:
            torch.Tensor: Filter start frequencíes.
            torch.Tensor: Filter stop frequencies.
        """
        # assert check_argument_types()
        # min and max bandpass edge frequencies
        min_frequency = torch.tensor(0.5)
        max_frequency = torch.tensor(fs * 0.5)
        # print(f'max_frequency: {max_frequency}')
        frequencies = torch.linspace(
            cls.convert(min_frequency), cls.convert(max_frequency), channels + 2
        )
        # print(f'frequencies: {frequencies}')
        frequencies = cls.invert(frequencies)
        # print(f'frequencies: {frequencies}')
        f1, f2 = frequencies[:-2], frequencies[2:]
        # print(f'f1:{f1}')
        return torch.stack([f1, f2], dim=1)

if __name__ == "__main__":
    sincconv2d = SincConv2D(in_channels=2, out_channels=16, kernel_size=127, scale_type='bark', fs=400)
    # print(sincconv2d.low_hz.size())
    # print(sincconv2d.window.size())
    x = torch.randn([50, 2, 16, 400], requires_grad=True)
    y = sincconv2d(x)
    print(y.size())