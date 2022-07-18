import torch
import torch.nn as nn
import torch.nn.grad
import torch.nn.functional as F
import numpy as np
import cpp_layers
import cuda_matmul


# To deal with padding and stride needing to be tuples instead of ints.
def make_tuple(param):
    if isinstance(param, int):
        param_tuple = (param, param)
        return param_tuple
    else:
        return param


class quantConvFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, bias, in_scale, in_zero_point, wt_scale, wt_zero_point, appx_mode,
                padding=(1, 1), stride=(1, 1)
                ):
        confs = torch.from_numpy(np.array([stride[0], padding[0]]))
        ctx.save_for_backward(X, weight, bias, confs)

        # First convert input and weight to integer representation (but store them as floats)
        # so they can be passed to the GPU and the GPU operates directly on the integer value.

        X_scale = in_scale.item()
        X_zero_point = int(in_zero_point.item())
        X_quantized = torch.quantize_per_tensor(X, X_scale, X_zero_point, dtype=torch.quint8)
        X_float = torch.int_repr(X_quantized).float()

        wt_scale = wt_scale.item()
        wt_zero_point = int(wt_zero_point.item())
        weight_int8 = torch.quantize_per_tensor(weight, wt_scale, wt_zero_point, dtype=torch.qint8)
        weight = torch.int_repr(weight_int8).float()

        # print(f"QUANT: X:\n{X_float}\n W:\n{weight}\n")

        (b, n_C_prev, n_H_prev, n_W_prev) = X.shape
        (n_oC, n_iC, f, f) = weight.shape

        n_H = ((n_H_prev - f + (2 * padding[0])) // stride[0]) + 1
        n_W = ((n_W_prev - f + (2 * padding[0])) // stride[0]) + 1

        X_pad = F.pad(X_float, (padding[0], padding[0], padding[0], padding[0]))

        if X.is_cuda:
            inp_unf = torch.nn.functional.unfold(X_pad, (f, f))
            inp_unf = inp_unf.transpose(1, 2)
            weight = weight.view(weight.size(0), -1).t()
            inp_unf_flat = inp_unf.flatten()
            weight_flat = weight.flatten()

            # print(f'Appx Mode:{appx_mode}, type(appx_mode):{type(appx_mode)}')

            # print(f"Values into CUDA. m:{inp_unf.size(1)}, n:{inp_unf.size(2)}, k:{weight.size(1)}")
            out_unf = cuda_matmul.conv_forward_int8(inp_unf_flat, in_scale, in_zero_point,
                                                    weight_flat, wt_scale, wt_zero_point,
                                                    bias, inp_unf.size(1), inp_unf.size(2),
                                                    weight.size(1), b, appx_mode)
            out_unf = out_unf.transpose(1, 2)
            out_unf = out_unf.view(b, n_oC, n_H, n_W)

        else:
            out_unf = cpp_layers.conv_forward_int8(X_pad, X_scale, X_zero_point,
                                                   weight, wt_scale, wt_zero_point,
                                                   bias, padding[0], stride[0], appx_mode)

        # m, n, k = inp_unf.size(1), inp_unf.size(2), weight.size(1)
        # out_unf = torch.zeros(b, m, k)
        # for l in range(b):
        #     for i in range(m):
        #         for j in range(k):
        #             temp = 0
        #             for h in range(n):
        #                 A = (inp_unf[l, i, h].item() - X_zero_point)
        #                 B = (weight[h, j].item() - wt_zero_point)
        #                 # A = inp_unf[l, i, h].item()
        #                 # B = weight[h, j].item()
        #                 temp += A * B
        #                 # print(f"({round(float(A),3)}*{round(float(B),3)}) ", end="")
        #             value = (temp*X_scale*wt_scale) + bias[j]
        #             # Perform RELU here as well.
        #             if value < 0.0:
        #                 out_unf[l, i, j] = 0.0
        #             else:
        #                 out_unf[l, i, j] = value
        #             # out_unf[l, i, j] = temp + bias[j]
        #             # print(f"out = {out_unf[l, i, j].item()}")

        # The outputs are returned as float values. So they just need to be quantized for the next layer.
        # out_unf = torch.quantize_per_tensor(out_unf, out_scale.item(), int(out_zero_point.item()), dtype=torch.quint8)

        # print("CONV output from quant:")
        # print(out_unf)
        # pdb.set_trace()

        # cloning the output to support backward pass.
        return out_unf.clone()

    @staticmethod
    def backward(ctx, grad_output):
        X, weight, bias, confs = ctx.saved_tensors
        confs = confs.numpy()
        stride, padding = confs[0], confs[1]
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(X.shape, weight, grad_output, stride, padding)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(X, weight.shape, grad_output, stride, padding)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3)).squeeze(0)

        if bias is not None:
            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None, None, None, None


class QuantConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(QuantConv2D, self).__init__()

        self.kernel_size = make_tuple(kernel_size)
        self.dilation = make_tuple(dilation)
        self.padding = make_tuple(padding)
        self.stride = make_tuple(stride)
        self.out_channels = out_channels
        self.n_channels = in_channels
        self.appx_mode = torch.tensor([0], requires_grad=False)
        self.weight = nn.Parameter(torch.rand(self.out_channels,
                                              self.n_channels,
                                              self.kernel_size[0],
                                              self.kernel_size[1]))
        self.bias = nn.Parameter(torch.rand(self.out_channels))

        # Custom values which should be buffers so they don't need to be tracked during back-prop
        self.register_buffer('in_scale', torch.tensor(1.))
        self.register_buffer('in_zero_point', torch.tensor(0))
        self.register_buffer('wt_scale', torch.tensor(1.))
        self.register_buffer('wt_zero_point', torch.tensor(0))

    def forward(self, x):
        res = quantConvFunc.apply(x, self.weight, self.bias,
                                  self.in_scale, self.in_zero_point,
                                  self.wt_scale, self.wt_zero_point,
                                  self.appx_mode,
                                  self.padding, self.stride  # optional args
                                  )
        return res


'''
def appx_mul(A,B):
    window = np.zeros((A.shape))
    for l in range(A.shape[0]):
      for j in range(A.shape[1]):
        for k in range(A.shape[2]):
          #window[l,j,k] = FP_appx_mul(A[l,j,k],B[l,j,k])
          window[l,j,k] = A[l,j,k]*B[l,j,k]
    return np.sum(window)

def mul_channel( weight,bias, x_pad, n_H, n_W,f):
      Z = np.zeros(( n_H, n_W ))
      for h in range(n_H):
          for w in range(n_W):
              vert_start = h
              vert_end = vert_start + f
              horiz_start = w
              horiz_end = horiz_start + f

              x_slice = x_pad[:, vert_start:vert_end, horiz_start:horiz_end]
              Z[ h, w] = appx_mul(x_slice, weight)  #torch.matmul(A,B)
              Z[ h, w] += bias
      return Z

X_pad = F.pad(X, (padding[0],padding[0],padding[0],padding[0]))
weight = weight.data.numpy()
bias = bias.data.numpy()
X_pad = X_pad.data.numpy()

Z = np.zeros((m, n_C, n_H, n_W ))

for i in range(m):
    for c in range(n_C):
        Z[i,c] = mul_channel( weight[c, :, :, :],bias[c], X_pad[i], n_H, n_W, f)
'''
