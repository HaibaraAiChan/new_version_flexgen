
import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from initialize import get_tensor_model_parallel_rank
from initialize import get_tensor_model_parallel_world_size
from initialize import get_tensor_model_parallel_group
from mappings import copy_to_tensor_model_parallel_region
from mappings import gather_from_tensor_model_parallel_region
from mappings import gather_from_sequence_parallel_region
from mappings import reduce_from_tensor_model_parallel_region
from mappings import scatter_to_tensor_model_parallel_region
from mappings import reduce_scatter_to_sequence_parallel_region

from mpu_random import get_cuda_rng_tracker
from utils import divide
from utils import split_tensor_along_last_dim
from utils import VocabUtility
from global_vars import get_args, get_global_memory_buffer

sys.path.insert(0,'../mpu')
from layers import LinearWithGradAccumulationAndAsyncCommunication


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}

def _args_to_kwargs():
    args = get_args()
    args.use_cpu_initialization = False
    args.perform_initialization = True
    args.gradient_accumulation_fusion = False
    args.sequence_parallel = False
    
    common_kwargs = {
        "params_dtype": args.params_dtype,
        "use_cpu_initialization": args.use_cpu_initialization,
        "perform_initialization": args.perform_initialization,
        "gradient_accumulation_fusion": args.gradient_accumulation_fusion,
        "sequence_parallel_enabled": args.sequence_parallel,
    }
    return common_kwargs

def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False,
                                  *, params_dtype=torch.float32):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    # init_method(master_weight) # we can use our weights (init from files and load from cache)
    
    master_weight = master_weight.to(dtype=params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization: # False
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=args.params_dtype))
            if args.perform_initialization: # False
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.output_size_per_partition, 0, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test)
        # else: 
        #     self.weight = Parameter(torch.empty(
        #         self.output_size_per_partition, self.input_size,
        #         device=torch.cuda.current_device(), dtype=args.params_dtype))
        #     if args.perform_initialization: # False
        #         _initialize_affine_weight_gpu(self.weight, init_method,
        #                                       partition_dim=0, stride=stride)

        if bias:
            if args.use_cpu_initialization: # False
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            # else:
            #     self.bias = Parameter(torch.empty(
            #         self.output_size_per_partition,
            #         device=torch.cuda.current_device(),
            #         dtype=args.params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.async_tensor_model_parallel_allreduce = (
                args.async_tensor_model_parallel_allreduce and
                world_size > 1)
        self.sequence_parallel = (
                args.sequence_parallel and
                world_size > 1)
        assert not self.async_tensor_model_parallel_allreduce or \
            not self.sequence_parallel
        self.gradient_accumulation_fusion = args.gradient_accumulation_fusion

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or \
                self.sequence_parallel:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel, self.weight, bias, self.gradient_accumulation_fusion,
            self.async_tensor_model_parallel_allreduce, self.sequence_parallel)
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

