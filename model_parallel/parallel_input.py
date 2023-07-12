
import os
import numpy as np
import torch
import sys
# sys.path.insert(0,'/home/cc/FlexGen/new_flexgen/flexgen_additional')
# sys.path.insert(0,'/home/cc/FlexGen/new_flexgen/mpu')
sys.path.insert(0,'../flexgen_additional')
sys.path.insert(0,'../mpu')
from flexgen_utils import init_weight_list,init_weight_list_mp
import torch.nn.init as init
from initialize import get_tensor_model_parallel_world_size
from utils import VocabUtility
from mappings import reduce_from_tensor_model_parallel_region


# class VocabParallelEmbedding(torch.nn.Module):
#     """Embedding parallelized in the vocabulary dimension.

#     This is mainly adapted from torch.nn.Embedding and all the default
#     values are kept.
#     Arguments:
#         num_embeddings: vocabulary size.
#         embedding_dim: size of hidden state.

#     Keyword Arguments:
#         init_method: method to initialize weights.
#         params_dtype
#         use_cpu_initialization
#         perform_initialization
#     """
#     def __init__(self, num_embeddings: int, embedding_dim: int, *,
#                  init_method=init.xavier_normal_,
#                  params_dtype: torch.dtype=torch.float32,
#                  use_cpu_initialization: bool=False,
#                  perform_initialization: bool=True):
#         super(VocabParallelEmbedding, self).__init__()
#         # Keep the input dimensions.
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         # Set the detauls for compatibility.
#         self.padding_idx = None
#         self.max_norm = None
#         self.norm_type = 2.
#         self.scale_grad_by_freq = False
#         self.sparse = False
#         self._weight = None
#         self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
#         # Divide the weight matrix along the vocaburaly dimension.
#         self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
#                 self.num_embeddings, get_tensor_model_parallel_rank(),
#                 self.tensor_model_parallel_size)
#         self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

#         # Allocate weights and initialize.
#         if use_cpu_initialization:
#             self.weight = Parameter(torch.empty(
#                 self.num_embeddings_per_partition, self.embedding_dim,
#                 dtype=params_dtype))
#             if perform_initialization:
#                 _initialize_affine_weight_cpu(
#                     self.weight, self.num_embeddings, self.embedding_dim,
#                     self.num_embeddings_per_partition, 0, init_method,
#                     params_dtype=params_dtype)
#         else:
#             self.weight = Parameter(torch.empty(
#                 self.num_embeddings_per_partition, self.embedding_dim,
#                 device=torch.cuda.current_device(), dtype=params_dtype))
#             if perform_initialization:
#                 _initialize_affine_weight_gpu(self.weight, init_method,
#                                               partition_dim=0, stride=1)

#     def forward(self, input_):
#         if self.tensor_model_parallel_size > 1:
#             # Build the mask.
#             input_mask = (input_ < self.vocab_start_index) | \
#                          (input_ >= self.vocab_end_index)
#             # Mask the input.
#             masked_input = input_.clone() - self.vocab_start_index
#             masked_input[input_mask] = 0
#         else:
#             masked_input = input_
#             # Get the embeddings.
#         output_parallel = F.embedding(masked_input, self.weight,
#                                       self.padding_idx, self.max_norm,
#                                       self.norm_type, self.scale_grad_by_freq,
#                                       self.sparse)
#         # Mask the output embedding.
#         if self.tensor_model_parallel_size > 1:
#             output_parallel[input_mask, :] = 0.0
#         # Reduce across all the model parallel GPUs.
#         output = reduce_from_tensor_model_parallel_region(output_parallel)
#         return output




class InputEmbed:
    def __init__(self, config, env, policy):
        self.name = "InputEmbed"
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None
        self.tensor_model_parallel_size = policy.tensor_model_parallel_size
        # self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        # self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
        #         self.num_embeddings, get_tensor_model_parallel_rank(),
        #         self.tensor_model_parallel_size)
        # self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        # self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

            
    def set_task(self, task):
        self.task = task

    

    # tensor model parallel version, the m-th part weights initialization
    def init_weight_mp(self, weight_home_current_layer, path, mp):
        # if mp ==1:
            v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
                self.config.max_seq_len, self.config.dtype)
            path = os.path.join(path, "")
            weight_specs = [
                # w_token
                ((v, h), dtype, path + "decoder.embed_tokens.weight"),
                # w_pos
                ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
            ]
        # else:
            
        #     weights = init_weight_list(weight_specs, self.policy, self.env)

        #     weight_home.store(weights)
            
        #     v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
        #         self.config.max_seq_len, self.config.dtype)
        #     path = os.path.join(path, "")
        #     weight_specs = [
        #         # w_token
        #         ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        #         # w_pos
        #         ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        #     ]
        #     weights = init_weight_list_mp(weight_specs, m, self.policy, self.env)
        #     weight_home_current_layer[m].store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token, w_pos = weight_home.val
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store((w_token.smart_copy(dst), w_pos.smart_copy(dst)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]), (w_pos, donate[3]) = weight_read_buf.pop()
        else:
            (w_token, _), (w_pos, _) = weight_read_buf.val

        h = self.compute.opt_input_embed(h, mask, w_token, w_pos, self.config.pad_token_id, donate)
        hidden.val = h



    # def parallel_embedding_init(tensor_model_parallel_size):
    #     tensor_model_parallel_size = get_tensor_model_parallel_world_size()
    #     tensor_model_parallel_size_ = min(tensor_model_parallel_size,
    #                             torch.distributed.get_world_size())
    

    #     embedding_vocab_parallel = layers.VocabParallelEmbedding(
    #         vocab_size, hidden_size, init_method=init.normal_).cuda()
    #     output = embedding_vocab_parallel(input_data)
    #     loss_vocab_parallel = torch.mul(output, loss_weight).sum()
    

    


