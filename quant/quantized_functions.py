import torch
import torch.nn.functional as nnf


# All this layer needs to do is add the quant parameters to the input so it matches all the other inputs
# that the first layer type will receive
def quant_layer(input_matrix, quant_scale, quant_zero_point):

    if input_matrix.is_cuda:
        quant_scale = quant_scale.cuda()
        quant_zero_point = quant_zero_point.cuda()
    return torch.quantize_per_tensor(input_matrix, quant_scale, quant_zero_point, dtype=torch.quint8)


def dequant_layer(input_matrix):
    return torch.dequantize(input_matrix)


def quant_maxpool2d(input_matrix, kernel_size=2, stride=2):
    scale = input_matrix.q_scale()
    zero_point = input_matrix.q_zero_point()
    input_matrix = torch.dequantize(input_matrix)

    output = nnf.max_pool2d(input_matrix, kernel_size, stride)

    output_quantized = torch.quantize_per_tensor(output, scale, zero_point, dtype=torch.quint8)

    return output_quantized
