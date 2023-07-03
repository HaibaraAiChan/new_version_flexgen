
"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """
import importlib
import numbers

import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import sys
sys.path.insert(0,'..')
# sys.path.insert(0,'../mpu/')
sys.path.insert(0,'/home/cc/FlexGen/new_flexgen/mpu')
from mpu_random import make_viewless_tensor

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
    HAVE_PERSIST_LAYER_NORM = True
except:
    HAVE_PERSIST_LAYER_NORM = False


from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction

# global fused_mix_prec_layer_norm_cuda
# fused_mix_prec_layer_norm_cuda = None


# class FusedLayerNormAffineFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, normalized_shape, eps):
#         ctx.normalized_shape = normalized_shape
#         ctx.eps = eps
#         input_ = input.contiguous()
#         weight_ = weight.contiguous()
#         bias_ = bias.contiguous()
#         output, mean, invvar = fused_mix_prec_layer_norm_cuda.forward_affine(
#             input_, ctx.normalized_shape, weight_, bias_, ctx.eps
#         )
#         ctx.save_for_backward(input_, weight_, bias_, mean, invvar)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input_, weight_, bias_, mean, invvar = ctx.saved_tensors
#         grad_input = grad_weight = grad_bias = None
#         grad_input, grad_weight, grad_bias = fused_mix_prec_layer_norm_cuda.backward_affine(
#             grad_output.contiguous(), mean, invvar, input_, ctx.normalized_shape, weight_, bias_, ctx.eps
#         )
#         return grad_input, grad_weight, grad_bias, None, None



class FusedLayerNorm(torch.nn.Module):

    def __init__(self, normalized_shape,
                 eps=1e-5, 
                 no_persist_layer_norm=True,
                 sequence_parallel=False,
                 apply_layernorm_1p=False):
        super(FusedLayerNorm, self).__init__()
        
        self.apply_layernorm_1p = apply_layernorm_1p
        
        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")


        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.bias = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

        self.sequence_parallel = sequence_parallel
        self.no_persist_layer_norm = no_persist_layer_norm
        # set sequence parallelism flag on weight and bias parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
        setattr(self.bias, 'sequence_parallel', self.sequence_parallel)


    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)
        
    def forward(self, input):
        
        weight = self.weight + 1 if self.apply_layernorm_1p else self.weight
        
        if self.no_persist_layer_norm:
            print('enter FusedLayerNormAffineFunction ------')
            print('self.normalized_shape ', self.normalized_shape)
            # print('self.normalized_shape device ', torch.empty(self.normalized_shape).device.type)
            # print('input ', input.device.type)
            # print('weight ', self.weight.device.type)
            # print('bias ', self.bias.device.type)
            # print('eps ', self.eps.device.type)
            return FusedLayerNormAffineFunction.apply(
                input, self.weight.cuda().half(), self.bias.cuda().half(), self.normalized_shape, self.eps)
            return FusedLayerNormAffineFunction.apply(
                input, self.weight.cuda(), self.bias.cuda(), self.normalized_shape, self.eps)
            print('')
        else:
            output = FastLayerNormFN.apply(
            input, self.weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            output = make_viewless_tensor(inp = output,
                                        requires_grad = input.requires_grad,
                                        keep_graph = True)

        return output