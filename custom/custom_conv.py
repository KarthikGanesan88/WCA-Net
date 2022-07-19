import torch
import torch.nn as nn
import torch.nn.grad
import torch.nn.functional as F
import numpy as np
import cuda_matmul

# To deal with padding and stride needing to be tuples instead of ints.
def make_tuple(param):
    if isinstance(param, int):
        param_tuple = (param, param)
        return param_tuple
    else:
        return param

class convAppx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, bias, appx_mode, padding=(1, 1), stride=(1, 1), dilation=(1, 1)):
        # confs = torch.from_numpy(np.array([stride[0], padding[0]]))
        confs = torch.Tensor([stride[0], padding[0]])
        ctx.save_for_backward(X, weight, bias, confs)

        # print(f"Input:: GPU: {X.is_cuda}, Shape: {X.shape}")
        # print(f"Weight:: GPU: {weight.is_cuda}, Shape: {weight.shape}")
        # print(f"Bias:: GPU: {bias.is_cuda}, Shape: {bias.shape}")

        # print(f"Appx Mode inside pytorch conv:{appx_mode}")

        (batch_size, n_C_prev, n_H_prev, n_W_prev) = X.shape
        (n_oC, n_iC, f_H, f_W) = weight.shape

        n_H = ((n_H_prev + (2 * padding[0]) - (dilation[0] * (f_H - 1)) - 1) // stride[0]) + 1
        n_W = ((n_W_prev + (2 * padding[1]) - (dilation[1] * (f_W - 1)) - 1) // stride[1]) + 1

        # pdb.set_trace()

        # print(f"APPX:\n X:\n{X} W:\n{weight}")

        # Left, right, top, bottom -- No need to pad separately
        # X_pad = F.pad(X, (padding[0], padding[0], padding[0], padding[0]))
        # print(f"X_pad: {X_pad.size()}, is_cuda:{X_pad.is_cuda}")

        # pdb.set_trace()

        if X.is_cuda:
            # print("Using GPU for CONV")
            # print("Input")
            # print(X_pad)

            inp_unf = torch.nn.functional.unfold(X,
                                                 kernel_size=(f_H, f_W),
                                                 padding=padding,
                                                 stride=stride,
                                                 dilation=dilation
                                                 )
            # print(f"Inp_unf: {inp_unf.size()}")
            # print(inp_unf)

            inp_unf = inp_unf.transpose(1, 2)
            # print(f"Inp_unf transposed: {inp_unf.size()}")
            # print(inp_unf)

            weight = weight.view(weight.size(0), -1).t()
            # print(f"kernel folded: {weight.size()}")
            # print(w)

            inp_unf_flat = inp_unf.flatten()
            weight_flat = weight.flatten()
            # print(f"Values into CUDA. m:{inp_unf.size(1)}, n:{inp_unf.size(2)}, k:{weight.size(1)}")
            # - input, weight, bias, m, n, k, b

            use_bias = 0 if bias is None else 1
            bias = torch.Tensor([0.]) if bias is None else bias

            out_unf = cuda_matmul.conv_forward(inp_unf_flat, weight_flat, bias,
                                               inp_unf.size(1), inp_unf.size(2), weight.size(1),
                                               batch_size, int(appx_mode.item()), use_bias)

            # pdb.set_trace()

            # out_unf = torch.matmul(inp_unf, weight)
            # print(f"out_unf : {out_unf.size()}")

            # Works OK.
            # m = inp_unf.size(1)
            # n = inp_unf.size(2)
            # k = weight.size(1)
            #
            # print(f"m:{m}, n:{n}, k:{k}")
            # out_unf = torch.zeros(b, m, k, device='cuda')
            # for l in range(b):
            #     for i in range(m):
            #         for j in range(k):
            #             temp = 0
            #             for h in range(n):
            #                 A = inp_unf[l, i, h].item()
            #                 B = weight[h, j].item()
            #                 temp += A * B
            #                 print(f"({A}*{B}) ", end="")
            #             out_unf[l, i, j] = temp + bias[j]
            #             print()

            # print(f"out_unf: {out_unf.size()}")

            out_unf = out_unf.transpose(1, 2)
            # print(f"out_unf transposed: {out_unf.size()}")

            out_unf = out_unf.view(batch_size, n_oC, n_H, n_W)
            # print(f"out_unf reshaped: {out_unf.size()}")

            # print("CONV output from appx:")
            # print(out_unf)

            # need to return a clone to support backward pass.
            return out_unf.clone()
        else:
            # Doing this because CPP layers isnt working.
            return X
            # Z = torch.empty(b, n_oC, n_H, n_W, device='cpu')
            # print("Using CPU for CONV")

            # output = cpp_layers.conv_forward(X_pad, weight, bias, padding[0], stride[0], appx_mode)
            # print(f"output shape: {output.size()}")
            # return output
            # for i in range(b):
            #     # print(f"Image#:{m}")
            #     for c in range(n_oC):
            #         for h in range(n_H):
            #             for w in range(n_W):
            #                 accumulation = 0.0
            #                 for l in range(n_iC):
            #                     # print(f"ic:{l}: ", end="")
            #                     for j in range(f):
            #                         for k in range(f):
            #                             # print(f"({i},{l},{j+h},{k+w})*({c},{l},{j},{k})", end=" ")
            #                             A = X_pad[i, l, (j + h), (k + w)]
            #                             B = weight[c, l, j, k]
            #                             # A_print = round(A.item(), 2)
            #                             # B_print = round(B.item(), 2)
            #                             # print(f"({A_print}*{B_print})", end="")
            #                             accumulation += A * B
            #                     # print("")
            #                 # print(f"=({i},{c},{h},{w})")
            #                 Z[i, c, h, w] = accumulation + bias[c]
            # return Z

    @staticmethod
    def backward(ctx, grad_output):
        X, weight, bias, confs = ctx.saved_tensors
        stride, padding = confs[0], confs[1]
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(X.shape, weight, grad_output, stride, padding)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(X, weight.shape, grad_output, stride, padding)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3)).squeeze(0)

        if bias is not None:
            return grad_input, grad_weight, grad_bias, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None

class CustomConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super().__init__()

        self.kernel_size = make_tuple(kernel_size)
        self.dilation = make_tuple(dilation)
        self.padding = make_tuple(padding)
        self.stride = make_tuple(stride)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.appx_mode = torch.tensor([0], requires_grad=False)
        self.weight = nn.Parameter(torch.rand(self.out_channels,
                                              self.in_channels,
                                              self.kernel_size[0],
                                              self.kernel_size[1]))
        self.bias = nn.Parameter(torch.rand(self.out_channels))

    def forward(self, x):
        res = convAppx.apply(x, self.weight, self.bias, self.appx_mode, self.padding, self.stride)
        return res
