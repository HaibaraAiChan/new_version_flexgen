# from flexgen.pytorch_backend import (TorchDevice, TorchDisk, TorchLink,
#     TorchMixedDevice, DeviceType, general_copy, fix_recursive_import)
# import dataclasses
import os
import numpy as np
import sys
# sys.path.insert(0,'../flexgen_additional/')
sys.path.insert(0,'/home/cc/FlexGen/new_flexgen/flexgen_additional')
from flexgen_utils import torch_dtype_to_np_dtype, init_weight_list , init_weight_list_mp 
from pytorch_backend import TorchDevice, TorchDisk, TorchLink,TorchMixedDevice, DeviceType, general_copy, fix_recursive_import
DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

from module import MegatronModule
sys.path.insert(0,'../mpu')
from initialize import get_tensor_model_parallel_world_size
from layers import ColumnParallelLinear, RowParallelLinear
import utils 
# class ParallelAttention(MegatronModule):
#     """Parallel self-attention layer abstract class.

#     Self-attention layer takes input with size [s, b, h][sequence length, batch size, hidden]
#     and returns output of the same size.
#     """

#     def __init__(self, init_method,
#                  output_layer_init_method, layer_number,
#                  attention_type=AttnType.self_attn,
#                  attn_mask_type=AttnMaskType.padding):
#         super(ParallelAttention, self).__init__()
#         args = get_args()
#         self.layer_number = max(1, layer_number)
#         self.attention_type = attention_type
#         self.attn_mask_type = attn_mask_type
#         self.params_dtype = args.params_dtype
#         self.sequence_parallel = args.sequence_parallel

#         self.use_flash_attn = args.use_flash_attn \
#             and attention_type == AttnType.self_attn \
#             and self.attn_mask_type == AttnMaskType.causal
#         if self.use_flash_attn:
#             if flash_attn_unpadded_func is None:
#                 raise ImportError('FlashAttention is not installed, please install with '
#                                   'pip install flash-attn')
#             assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
#                                                           'self-attention for now')
#             assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
#                                                                 'supports causal mask for now')
#             if rearrange is None:
#                 raise ImportError('einops is not installed, please install with pip install einops')

#         projection_size = args.kv_channels * args.num_attention_heads

#         # Per attention head and per partition values.
#         world_size = mpu.get_tensor_model_parallel_world_size()
#         self.hidden_size_per_attention_head = core.utils.divide(
#             projection_size, args.num_attention_heads)
#         self.num_attention_heads_per_partition = core.utils.divide(
#             args.num_attention_heads, world_size)

#         # Strided linear layer.
#         if attention_type == AttnType.self_attn:
#             self.query_key_value = tensor_parallel.ColumnParallelLinear(
#                 args.hidden_size,
#                 3 * projection_size,
#                 bias=args.add_bias_linear,
#                 gather_output=False,
#                 init_method=init_method,
#                 async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
#                 **_args_to_kwargs())
#         else:
#             assert attention_type == AttnType.cross_attn
#             self.query = tensor_parallel.ColumnParallelLinear(
#                 args.hidden_size,
#                 projection_size,
#                 bias=args.add_bias_linear,
#                 gather_output=False,
#                 init_method=init_method,
#                 async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
#                 **_args_to_kwargs())


#             self.key_value = tensor_parallel.ColumnParallelLinear(
#                 args.hidden_size,
#                 2 * projection_size,
#                 bias=args.add_bias_linear,
#                 gather_output=False,
#                 init_method=init_method,
#                 async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
#                 **_args_to_kwargs())

#         self.core_attention = CoreAttention(self.layer_number,
#                                             self.attn_mask_type)
#         self.checkpoint_core_attention = args.recompute_granularity == 'selective'

#         if self.use_flash_attn:
#             self.core_attention_flash = FlashSelfAttention(
#                 causal=True, attention_dropout=args.attention_dropout
#             )

#         # Output.
#         self.dense = tensor_parallel.RowParallelLinear(
#             projection_size,
#             args.hidden_size,
#             bias=args.add_bias_linear,
#             input_is_parallel=True,
#             init_method=output_layer_init_method,
#             skip_bias_add=True,
#             **_args_to_kwargs())

#     def _checkpointed_attention_forward(self, query_layer, key_layer,
#                                         value_layer, attention_mask,
#                                         rotary_pos_emb=None):
#         """Forward method with activation checkpointing."""
#         def custom_forward(*inputs):
#             query_layer = inputs[0]
#             key_layer = inputs[1]
#             value_layer = inputs[2]
#             attention_mask = inputs[3]
#             output_ = self.core_attention(query_layer, key_layer,
#                                           value_layer, attention_mask)
#             return output_

#         q_pos_emb, k_pos_emb = (None, None) if rotary_pos_emb is None \
#             else rotary_pos_emb

#         hidden_states = tensor_parallel.checkpoint(
#             custom_forward,
#             False, query_layer, key_layer, value_layer, attention_mask,
#             q_pos_emb, k_pos_emb)

#         return hidden_states

#     def _allocate_memory(self, inference_max_sequence_len, batch_size):
#         return torch.empty(
#             inference_max_sequence_len,
#             batch_size,
#             self.num_attention_heads_per_partition,
#             self.hidden_size_per_attention_head,
#             dtype=self.params_dtype,
#             device=torch.cuda.current_device())

#     def forward(self, hidden_states, attention_mask,
#                 encoder_output=None, inference_params=None,
#                 rotary_pos_emb=None):
#         # hidden_states: [sq, b, h]

#         # =================================================
#         # Pre-allocate memory for key-values for inference.
#         # =================================================
#         is_first_step = False
#         if inference_params:
#             if self.layer_number not in inference_params.key_value_memory_dict:
#                 inf_max_seq_len = inference_params.max_sequence_len
#                 inf_max_batch_size = inference_params.max_batch_size
#                 inference_key_memory = self._allocate_memory(
#                     inf_max_seq_len, inf_max_batch_size)
#                 inference_value_memory = self._allocate_memory(
#                     inf_max_seq_len, inf_max_batch_size)
#                 inference_params.key_value_memory_dict[self.layer_number] = (
#                     inference_key_memory, inference_value_memory)
#                 is_first_step = True
#             else:
#                 inference_key_memory, inference_value_memory = \
#                     inference_params.key_value_memory_dict[self.layer_number]

#         # =====================
#         # Query, Key, and Value
#         # =====================

#         if self.attention_type == AttnType.self_attn:
#             # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
#             mixed_x_layer, _ = self.query_key_value(hidden_states)

#             # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
#             new_tensor_shape = mixed_x_layer.size()[:-1] + \
#                 (self.num_attention_heads_per_partition,
#                  3 * self.hidden_size_per_attention_head)
#             mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

#             # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
#             (query_layer,
#              key_layer,
#              value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)
#         else:
#             # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
#             mixed_kv_layer, _ = self.key_value(encoder_output)

#             # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
#             new_tensor_shape = mixed_kv_layer.size()[:-1] + \
#                 (self.num_attention_heads_per_partition,
#                  2 * self.hidden_size_per_attention_head)
#             mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

#             # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
#             (key_layer,
#              value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

#             # Attention head [sq, b, h] --> [sq, b, hp]
#             query_layer, _ = self.query(hidden_states)
#             # [sq, b, hp] --> [sq, b, np, hn]
#             new_tensor_shape = query_layer.size()[:-1] + \
#                 (self.num_attention_heads_per_partition,
#                  self.hidden_size_per_attention_head)
#             query_layer = query_layer.view(*new_tensor_shape)

#         # ==================================
#         # Adjust key and value for inference
#         # ==================================

#         # duplicate the pos_emb for self attention
#         if rotary_pos_emb is not None:
#             if isinstance(rotary_pos_emb, tuple):
#                 rotary_pos_emb = rotary_pos_emb
#             else:
#                 rotary_pos_emb = ((rotary_pos_emb,) * 2)

#         if inference_params:
#             batch_start = inference_params.batch_size_offset
#             batch_end = batch_start + key_layer.size(1)
#             assert batch_end <= inference_key_memory.size(1)
#             sequence_start = inference_params.sequence_len_offset
#             sequence_end = sequence_start + key_layer.size(0)
#             assert sequence_end <= inference_key_memory.size(0)
#             # Copy key and values.
#             inference_key_memory[sequence_start:sequence_end,
#                                  batch_start:batch_end, ...] = key_layer
#             inference_value_memory[sequence_start:sequence_end,
#                                    batch_start:batch_end, ...] = value_layer
#             key_layer = inference_key_memory[
#                 :sequence_end, batch_start:batch_end, ...]
#             value_layer = inference_value_memory[
#                 :sequence_end, batch_start:batch_end, ...]


#             # adjust the key rotary positional embedding
#             if rotary_pos_emb is not None:
#                 q_pos_emb, k_pos_emb = rotary_pos_emb
#                 # need to cross check this condition during inference
#                 # if not set_inference_key_value_memory:
#                 if not is_first_step:
#                     # In inference, we compute one token at a time.
#                     # Select the correct positional embedding
#                     # (only the last token in the sequence)
#                     q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
#                 else:
#                     # In the first forward pass of inference,
#                     # we use the entire provided prefix.
#                     # q_pos_emb here has the rope embeddings of the entire
#                     # prefix + to-be-generated output so
#                     # we slice to just the prefix.
#                     q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
#                 k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
#                 rotary_pos_emb = (q_pos_emb, k_pos_emb)


#         # ==================================
#         # core attention computation
#         # ==================================

#         # apply relative positional encoding (rotary embedding)
#         if rotary_pos_emb is not None:
#             q_pos_emb, k_pos_emb = rotary_pos_emb
#             query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
#             key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
#             # TODO, can apply positional embedding to value_layer so it has
#             # absolute positional embedding.
#             # otherwise, only relative positional embedding takes effect
#             # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

#         if not self.use_flash_attn:
#             if self.checkpoint_core_attention:
#                 context_layer = self._checkpointed_attention_forward(
#                     query_layer, key_layer, value_layer, attention_mask)
#             else:
#                 context_layer = self.core_attention(
#                     query_layer, key_layer, value_layer, attention_mask)
#         else:
#             q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
#                        for x in (query_layer, key_layer, value_layer)]
#             if not self.sequence_parallel:
#                 with tensor_parallel.get_cuda_rng_tracker().fork():
#                     context_layer = self.core_attention_flash(q, k, v)
#             else:
#                 context_layer = self.core_attention_flash(q, k, v)
#             context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

#         # =================
#         # Output. [sq, b, h]
#         # =================

#         output, bias = self.dense(context_layer)

#         return output, bias

###=========================================================================================
class SelfAttention:
    def __init__(self, config, env, policy, layer_id):
        self.name = 'SelfAttention' ####
        self.prefill = None         ####
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)

        self.task = None
        #self.sequence_parallel = policy.sequence_parallel#
        self.num_attention_heads = config.n_head
        projection_size = config.hidden_size
        # world_size = get_tensor_model_parallel_world_size()
        world_size = 4
        self.world_size = world_size
        self.hidden_size_per_attention_head = utils.divide(
            projection_size, self.config.n_head)
        self.num_attention_heads_per_partition = utils.divide(
            self.config.n_head, world_size)

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def init_weight_mp(self, weight_home, path, mp):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # b_q
            ((h,), dtype, path + ".q_proj.bias"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # b_k
            ((h,), dtype, path + ".k_proj.bias"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # b_v
            ((h,), dtype, path + ".v_proj.bias"),
            # w_out
            ((h, h), dtype, path + ".out_proj.weight"),
            # b_out
            ((h,), dtype, path + ".out_proj.bias"),
            # w_ln
            ((h,), dtype, path + "_layer_norm.weight"),
            # b_ln
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list_mp(weight_specs, self.policy, self.env, mp, self.world_size)
        weight_home.store(weights, mp)


    def load_weight(self, weight_home, weight_read_buf, k):
        w_q, b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1), b_q.smart_copy(dst2),
                w_k.smart_copy(dst1), b_k.smart_copy(dst2),
                w_v.smart_copy(dst1), b_v.smart_copy(dst2),
                w_out.smart_copy(dst1), b_out.smart_copy(dst2),
                w_ln.smart_copy(dst2), b_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        if self.policy.cache_gpu_percent == 100:
            device = self.env.gpu
        elif self.policy.cache_cpu_percent == 100:
            device = self.env.cpu
        elif self.policy.cache_disk_percent == 100:
            device = self.env.disk
        else:
            device = self.env.mixed

        if self.policy.compress_cache:
            assert device.device_type != DeviceType.MIXED
            device = device.compressed_device

        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)

    def load_cache(self, cache_home, cache_read_buf, i):
        if i == 0:  # prefill, no cache
            return

        k_home, v_home = cache_home.val

        # Pick code path
        if self.policy.compress_cache:
            path = 0
            dst = self.attention_compute.compressed_device
        else:
            if self.policy.cpu_cache_compute:
                if (k_home.device.device_type == DeviceType.MIXED and
                    k_home.data[0][0] is not None):
                    path = 2
                else:
                    path = 1
            else:
                path = 0
            dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))

            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        elif path == 2:  # Copy to both GPU and CPU
            # The caches are stored on both GPU and other devices.
            # Compute attention on gpu for caches stored on gpu.
            # Compute attention on cpu for caches stored on cpu/disk.
            gpu_k_buf = k_home.data[0][0]
            gpu_v_buf = v_home.data[0][0]

            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(gpu_k_buf.shape[1], k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)
            general_copy(v_buf, indices, v_home, indices)
            cache_read_buf.store((((gpu_k_buf, k_buf,), False),
                                  ((gpu_v_buf, v_buf,), False)))
            assert self.policy.attn_sparsity >= 1.0
        else:
            raise ValueError(f"Invalid path: {path}")

    def store_cache(self, cache_home, cache_write_buf, i):
        # shape: (s, b * n_head, head_dim)
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        n_head = self.config.n_head
        print('------------------************   number of head', n_head)

        donate = [False] * 14
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (b_q, donate[3]), (w_k, donate[4]), (b_k, donate[5]),
             (w_v, donate[6]), (b_v, donate[7]), (w_out, donate[8]), (b_out, donate[9]),
             (w_ln, donate[10]), (b_ln, donate[11])) = weight_read_buf.pop()
        else:
            ((w_q, _), (b_q, _), (w_k, _), (b_k, _),
             (w_v, _), (b_v, _), (w_out, _), (b_out, _),
             (w_ln, _), (b_ln, _)) = weight_read_buf.val

        if i == 0:  # prefill
            print('self attention prefill--------')
            self.prefill = True
            
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            
            h, new_k_cache, new_v_cache = self.compute.mha(h, mask, w_q, b_q,
                w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head, donate,
                self.policy.compress_cache, self.policy.comp_cache_config)
            
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            print('self attention decode =======')
            self.prefill = False
            
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            
            (k_cache, donate[12]), (v_cache, donate[13]) = cache_read_buf.pop()
            
            h, new_k_cache, new_v_cache = self.compute.mha_gen(h, mask, w_q,
                b_q, w_k, b_k, w_v, b_v, w_out, b_out, w_ln, b_ln, n_head,
                k_cache, v_cache, donate, self.policy.attn_sparsity,
                self.policy.compress_cache, self.policy.comp_cache_config)
            
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h


        self.query_key_value = ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                bias=args.add_bias_linear,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())