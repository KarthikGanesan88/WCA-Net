import parser

import torch
import torch.nn as nn
import cpp_layers
import cuda_matmul
import pdb


class QuantLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, in_scale, in_zero_point, wt_scale, wt_zero_point, appx_mode):
        ctx.save_for_backward(X, weight, bias)

        # No need to quantize these again. They are already quint8. But they have to be converted to int8.
        X_scale = in_scale.item()
        X_zero_point = int(in_zero_point.item())
        X_quantized = torch.quantize_per_tensor(X, X_scale, X_zero_point, dtype=torch.quint8)
        X_float = torch.int_repr(X_quantized).float()

        wt_scale = wt_scale.item()
        wt_zero_point = int(wt_zero_point.item())
        weight_int8 = torch.quantize_per_tensor(weight, wt_scale, wt_zero_point, dtype=torch.qint8)
        weight = torch.int_repr(weight_int8).float()

        # X_recovered = (X_float - X_zero_point) * X_scale
        # weight_recovered = (weight - wt_zero_point) * wt_scale
        # print(f"QUANT: X:\n{X_recovered}\n W:\n{weight_recovered}\n")

        if X.is_cuda:

            (m, n) = X.shape
            (k, _) = weight.shape
            # Transpose the weights and flatten
            weight = torch.transpose(weight, 0, 1).flatten()

            out = cuda_matmul.linear_forward_int8(X_float, X_scale, X_zero_point,
                                                  weight, wt_scale, wt_zero_point,
                                                  bias, m, n, k, appx_mode)

        else:
            out = cpp_layers.linear_forward_int8(X_float, X_scale, X_zero_point,
                                                 weight, wt_scale, wt_zero_point,
                                                 bias, appx_mode)

        # weight = weight.transpose(0, 1)
        # m, n, k = X.size(0), X.size(1), weight.size(1)
        # out_unf = torch.zeros(m, k)
        # #for l in range(b):
        # for i in range(m):
        #     for j in range(k):
        #         temp = 0
        #         for h in range(n):
        #             A = (X_float[i, h].item() - X_zero_point)
        #             B = (weight[h, j].item() - wt_zero_point)
        #             # A = inp_unf[l, i, h].item()
        #             # B = weight[h, j].item()
        #             temp += A * B
        #             # print(f"({round(float(A),3)}*{round(float(B),3)}) ", end="")
        #         value = (temp * X_scale * wt_scale) + bias[j]
        #         # Perform RELU here as well.
        #         if value < 0.0:
        #             out_unf[i, j] = 0.0
        #         else:
        #             out_unf[i, j] = value
        #         # out_unf[l, i, j] = temp + bias[j]
        #         # print(f"out = {out_unf[i, j].item()}")

        # print(f"FC out:\n{out_unf}")

        # out_unf = torch.quantize_per_tensor(out, out_scale.item(), int(out_zero_point.item()), dtype=torch.quint8)

        return out.clone()

    @staticmethod
    def backward(ctx, grad_output):
        X, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(X)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.rand(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.appx_mode = torch.tensor([0], requires_grad=False)
        # Custom values which should be buffers so they don't need to be tracked during back-prop
        self.register_buffer('in_scale', torch.tensor(1.))
        self.register_buffer('in_zero_point', torch.tensor(0))
        self.register_buffer('wt_scale', torch.tensor(1.))
        self.register_buffer('wt_zero_point', torch.tensor(0))

    def forward(self, x):
        x = QuantLinearFunc.apply(x, self.weight, self.bias, self.in_scale, self.in_zero_point,
                                  self.wt_scale, self.wt_zero_point, self.appx_mode)
        return x
