import torch
import torch.nn as nn
import cuda_matmul
# import cpp_layers

class linear_appx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, appx_mode):
        ctx.save_for_backward(X, weight, bias)

        if X.is_cuda:

            (m, n) = X.shape
            (k, _) = weight.shape

            weight = torch.transpose(weight, 0, 1).flatten()

            out = cuda_matmul.linear_forward(X, weight, bias, m, n, k, appx_mode)

            return out
        else:
            raise NotImplementedError("Only CUDA is supported for custom layers.")

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        # return an extra none for appx_mode?
        return grad_input, grad_weight, grad_bias, None

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fn = linear_appx.apply
        self.in_features = in_features
        self.out_features = out_features
        self.appx_mode = torch.tensor([0], requires_grad=False)
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = self.fn(x, self.weight, self.bias, self.appx_mode)
        return x
