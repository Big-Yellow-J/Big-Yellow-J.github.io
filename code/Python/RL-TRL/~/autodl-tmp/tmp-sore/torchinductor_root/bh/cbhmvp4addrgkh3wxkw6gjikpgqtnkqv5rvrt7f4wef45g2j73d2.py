# AOT ID: ['21_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/s7/cs77bgzcq7s27bujjcwzzgxyhhkyq4k75bptuakx5ymphic6gwuv.py
# Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_19, position_ids_expanded], Original ATen: [aten.arange, aten.unsqueeze, aten._to_copy]
# Source node to ATen node mapping:
#   cache_position => iota
#   getitem_19 => unsqueeze_3
#   position_ids => unsqueeze
#   position_ids_expanded => convert_element_type_1
# Graph fragment:
#   %iota : Tensor "i64[s27][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (%primals_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %unsqueeze : Tensor "i64[1, s27][s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%iota, 0), kwargs = {})
#   %unsqueeze_3 : Tensor "i64[1, 1, s27][s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%unsqueeze, 1), kwargs = {})
#   %convert_element_type_1 : Tensor "f32[1, 1, s27][s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%unsqueeze_3, torch.float32), kwargs = {})
#   return %expand_3
triton_poi_fused__to_copy_arange_unsqueeze_0 = async_compile.triton('triton_poi_fused__to_copy_arange_unsqueeze_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_unsqueeze_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/z3/cz3wgb5fsn36sk4l2hzjoiyzi5vvi26lo6cg5sxjs66oaphg3mu4.py
# Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_2, rsqrt, hidden_states_1, to_7, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_2 => add_96
#   hidden_states => convert_element_type_4
#   hidden_states_1 => mul_89
#   hidden_states_2 => mul_96
#   inputs_embeds => embedding
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to_7 => convert_element_type_5
#   variance => mean
# Graph fragment:
#   %primals_2 : Tensor "i64[1, s27][s27, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[151936, 896][896, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %primals_7 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=primals_7]
#   %buf2 : Tensor "f32[1, s27, 1][s27, 1, s27]cuda:0" = PlaceHolder[target=buf2]
#   %embedding : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_3, %primals_2), kwargs = {})
#   %convert_element_type_4 : Tensor "f32[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%embedding, torch.float32), kwargs = {})
#   %pow_1 : Tensor "f32[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_4, 2), kwargs = {})
#   %mean : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_96 : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_96,), kwargs = {})
#   %mul_89 : Tensor "f32[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %rsqrt), kwargs = {})
#   %convert_element_type_5 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_89, torch.bfloat16), kwargs = {})
#   %mul_96 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_7, %convert_element_type_5), kwargs = {})
#   return %buf2,%mul_96
triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_1 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 896
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tl.full([1, 1], 151936, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tl.device_assert(((0 <= tmp4) & (tmp4 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 151936")
        tmp6 = tl.load(in_ptr1 + (r0_1 + 896*tmp4), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp12 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.full([1, 1], 151936, tl.int32)
        tmp14 = tmp0 + tmp13
        tmp15 = tmp0 < 0
        tmp16 = tl.where(tmp15, tmp14, tmp0)
        tl.device_assert(((0 <= tmp16) & (tmp16 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp16 < 151936")
        tmp18 = tl.load(in_ptr1 + (r0_1 + 896*tmp16), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = 896.0
        tmp21 = (tmp10 / tmp20)
        tmp22 = 1e-06
        tmp23 = tmp21 + tmp22
        tmp24 = libdevice.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp12 * tmp26
        tl.store(out_ptr1 + (r0_1 + 896*x0), tmp27, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/eo/ceol3tsim4mk4uqjopf6whuelyzlbqnmnx3j4gwvillolo77ewkk.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_10
# Graph fragment:
#   %primals_10 : Tensor "f32[32, 896][896, 1]cuda:0" = PlaceHolder[target=primals_10]
#   %convert_element_type_10 : Tensor "bf16[32, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_10, torch.bfloat16), kwargs = {})
#   return %convert_element_type_10
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 229376}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/2p/c2pfj7ta56ruphfszivdxmbrx7skzodw5vndzf4k7pdffijbfamk.py
# Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_5 => convert_element_type_25, permute_7
# Graph fragment:
#   %primals_15 : Tensor "f32[128, 32][32, 1]cuda:0" = PlaceHolder[target=primals_15]
#   %convert_element_type_25 : Tensor "bf16[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_15, torch.bfloat16), kwargs = {})
#   %permute_7 : Tensor "bf16[32, 128][1, 32]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_25, [1, 0]), kwargs = {})
#   return %permute_7
triton_poi_fused__to_copy_t_3 = async_compile.triton('triton_poi_fused__to_copy_t_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 32768}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/rg/crgnggscjelogsy2nlkg26ea2yfmbx7r5gr37sleiipyghya2iwg.py
# Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, result_3, linear_2, mul_4, result_4, view, query_states, sin_3, x1, x2, neg, cat_1, mul_8], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.slice, aten.neg]
# Source node to ATen node mapping:
#   cat_1 => cat
#   emb => clone, expand_4, unsqueeze_4, view_8
#   freqs => permute
#   linear_2 => view_14
#   matmul => unsqueeze_default
#   mul_4 => mul_146
#   mul_8 => mul_296
#   neg => neg
#   query_states => permute_4
#   result_3 => add_tensor_71, view_10
#   result_4 => add_148
#   sin => sin
#   sin_1 => mul_67
#   sin_2 => convert_element_type_3
#   sin_3 => unsqueeze_6
#   view => view_15
#   x1 => slice_2
#   x2 => slice_3
# Graph fragment:
#   %primals_9 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=primals_9]
#   %mm_default_72 : Tensor "bf16[s27, 896][896, 1]cuda:0" = PlaceHolder[target=mm_default_72]
#   %mm_1 : Tensor "bf16[s27, 896][896, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %mm_default : Tensor "f32[32, s27][s27, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %unsqueeze_default : Tensor "f32[1, 32, s27][32*s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, s27, 32][32*s27, 1, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, s27, 1, 32][32*s27, 1, 32*s27, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, s27, 2, 32][32*s27, 1, 0, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, %primals_1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, s27, 2, 32][64*s27, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, %primals_1, 64]), kwargs = {})
#   %sin : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_8,), kwargs = {})
#   %mul_67 : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_67, torch.bfloat16), kwargs = {})
#   %add_tensor_71 : Tensor "bf16[s27, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_9, %mm_default_72), kwargs = {})
#   %view_10 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_71, [1, %primals_1, 896]), kwargs = {})
#   %view_14 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, %primals_1, 896]), kwargs = {})
#   %mul_146 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 1.0), kwargs = {})
#   %add_148 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_10, %mul_146), kwargs = {})
#   %view_15 : Tensor "bf16[1, s27, 14, 64][896*s27, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_148, [1, %primals_1, -1, 64]), kwargs = {})
#   %permute_4 : Tensor "bf16[1, 14, s27, 64][896*s27, 64, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_15, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, s27, 64][64*s27, 64*s27, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 1), kwargs = {})
#   %slice_2 : Tensor "bf16[1, 14, s27, 32][896*s27, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 0, 32), kwargs = {})
#   %slice_3 : Tensor "bf16[1, 14, s27, 32][896*s27, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_4, 3, 32, 9223372036854775807), kwargs = {})
#   %neg : Tensor "bf16[1, 14, s27, 32][448*s27, 32, 448, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_3,), kwargs = {})
#   %cat : Tensor "bf16[1, 14, s27, 64][896*s27, 64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg, %slice_2], -1), kwargs = {})
#   %mul_296 : Tensor "bf16[1, 14, s27, 64][896*s27, 64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat, %unsqueeze_6), kwargs = {})
#   return %mul_296
triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // ks0
    x0 = (xindex % 14)
    x3 = (xindex % ks0)
    x1 = ((xindex // 14) % ks1)
    x4 = xindex
    tmp28 = tl.load(in_ptr3 + (x1 + ks1*((x2 % 32))), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32 + 64*x0 + (x2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (32 + 64*x3 + (x2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (32 + 64*x3 + (x2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = -tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 64, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr0 + (64*x0 + ((-32) + x2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr1 + (64*x3 + ((-32) + x2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr2 + (64*x3 + ((-32) + x2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp15, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp14, tmp26)
    tmp29 = tl_math.sin(tmp28)
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tl.store(out_ptr0 + (x4), tmp33, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/wm/cwm2ep6xuzxlzmzsmyqqbhjyrt3ivfb44rdnpe25w36ome2cf47r.py
# Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, result_3, linear_2, mul_4, result_4, view, query_states, cos_3, mul_7, q_embed], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze]
# Source node to ATen node mapping:
#   cos => cos
#   cos_1 => mul_60
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_8
#   freqs => permute
#   linear_2 => view_14
#   matmul => unsqueeze_default
#   mul_4 => mul_146
#   mul_7 => mul_279
#   q_embed => add_279
#   query_states => permute_4
#   result_3 => add_tensor_71, view_10
#   result_4 => add_148
#   view => view_15
# Graph fragment:
#   %primals_9 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=primals_9]
#   %mm_default_72 : Tensor "bf16[s27, 896][896, 1]cuda:0" = PlaceHolder[target=mm_default_72]
#   %mm_1 : Tensor "bf16[s27, 896][896, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %mm_default : Tensor "f32[32, s27][s27, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %mul_296 : Tensor "bf16[1, 14, s27, 64][896*s27, 1, 14, 14*s27]cuda:0" = PlaceHolder[target=mul_296]
#   %unsqueeze_default : Tensor "f32[1, 32, s27][32*s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, s27, 32][32*s27, 1, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, s27, 1, 32][32*s27, 1, 32*s27, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, s27, 2, 32][32*s27, 1, 0, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, %primals_1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, s27, 2, 32][64*s27, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, %primals_1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_8,), kwargs = {})
#   %mul_60 : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_60, torch.bfloat16), kwargs = {})
#   %add_tensor_71 : Tensor "bf16[s27, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_9, %mm_default_72), kwargs = {})
#   %view_10 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_71, [1, %primals_1, 896]), kwargs = {})
#   %view_14 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_1, [1, %primals_1, 896]), kwargs = {})
#   %mul_146 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, 1.0), kwargs = {})
#   %add_148 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_10, %mul_146), kwargs = {})
#   %view_15 : Tensor "bf16[1, s27, 14, 64][896*s27, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_148, [1, %primals_1, -1, 64]), kwargs = {})
#   %permute_4 : Tensor "bf16[1, 14, s27, 64][896*s27, 64, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_15, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, s27, 64][64*s27, 64*s27, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_279 : Tensor "bf16[1, 14, s27, 64][896*s27, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_4, %unsqueeze_5), kwargs = {})
#   %add_279 : Tensor "bf16[1, 14, s27, 64][896*s27, 64, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_279, %mul_296), kwargs = {})
#   return %add_279
triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'ks0': 'i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 14)
    y3 = yindex
    y1 = yindex // 14
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (y1 + ks0*((x2 % 32))), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y3 + 14*ks0*x2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tl_math.cos(tmp7)
    tmp9 = tmp8 * tmp4
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp6 * tmp10
    tmp13 = tmp11 + tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 64*y3), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/hi/chigzvz64shuuci6prkhgekvcrfqidnmt3np5ilrhx2rde424d2y.py
# Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, result_6, linear_5, mul_5, result_7, view_1, key_states, sin_3, x1_1, x2_1, neg_1, cat_2, mul_10], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.slice, aten.neg]
# Source node to ATen node mapping:
#   cat_2 => cat_1
#   emb => clone, expand_4, unsqueeze_4, view_8
#   freqs => permute
#   key_states => permute_8
#   linear_5 => view_21
#   matmul => unsqueeze_default
#   mul_10 => mul_320
#   mul_5 => mul_202
#   neg_1 => neg_1
#   result_6 => add_tensor_70, view_17
#   result_7 => add_195
#   sin => sin
#   sin_1 => mul_67
#   sin_2 => convert_element_type_3
#   sin_3 => unsqueeze_6
#   view_1 => view_22
#   x1_1 => slice_4
#   x2_1 => slice_5
# Graph fragment:
#   %primals_13 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=primals_13]
#   %mm_default_71 : Tensor "bf16[s27, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_71]
#   %mm_3 : Tensor "bf16[s27, 128][128, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_default : Tensor "f32[32, s27][s27, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %unsqueeze_default : Tensor "f32[1, 32, s27][32*s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, s27, 32][32*s27, 1, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, s27, 1, 32][32*s27, 1, 32*s27, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, s27, 2, 32][32*s27, 1, 0, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, %primals_1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, s27, 2, 32][64*s27, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, %primals_1, 64]), kwargs = {})
#   %sin : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_8,), kwargs = {})
#   %mul_67 : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_67, torch.bfloat16), kwargs = {})
#   %add_tensor_70 : Tensor "bf16[s27, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_13, %mm_default_71), kwargs = {})
#   %view_17 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_70, [1, %primals_1, 128]), kwargs = {})
#   %view_21 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, %primals_1, 128]), kwargs = {})
#   %mul_202 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 1.0), kwargs = {})
#   %add_195 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %mul_202), kwargs = {})
#   %view_22 : Tensor "bf16[1, s27, 2, 64][128*s27, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_195, [1, %primals_1, -1, 64]), kwargs = {})
#   %permute_8 : Tensor "bf16[1, 2, s27, 64][128*s27, 64, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_22, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, s27, 64][64*s27, 64*s27, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 1), kwargs = {})
#   %slice_4 : Tensor "bf16[1, 2, s27, 32][128*s27, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_8, 3, 0, 32), kwargs = {})
#   %slice_5 : Tensor "bf16[1, 2, s27, 32][128*s27, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%permute_8, 3, 32, 9223372036854775807), kwargs = {})
#   %neg_1 : Tensor "bf16[1, 2, s27, 32][64*s27, 32, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_5,), kwargs = {})
#   %cat_1 : Tensor "bf16[1, 2, s27, 64][128*s27, 64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%neg_1, %slice_4], -1), kwargs = {})
#   %mul_320 : Tensor "bf16[1, 2, s27, 64][128*s27, 64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cat_1, %unsqueeze_6), kwargs = {})
#   return %mul_320
triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // ks0
    x0 = (xindex % 2)
    x3 = (xindex % ks0)
    x1 = ((xindex // 2) % ks1)
    x4 = xindex
    tmp28 = tl.load(in_ptr3 + (x1 + ks1*((x2 % 32))), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (32 + 64*x0 + (x2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (32 + 64*x3 + (x2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (32 + 64*x3 + (x2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = -tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 64, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr0 + (64*x0 + ((-32) + x2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr1 + (64*x3 + ((-32) + x2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr2 + (64*x3 + ((-32) + x2)), tmp15 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp15, tmp24, tmp25)
    tmp27 = tl.where(tmp4, tmp14, tmp26)
    tmp29 = tl_math.sin(tmp28)
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tl.store(out_ptr0 + (x4), tmp33, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/jx/cjxgocquumhcfvehlobtnla6reramxk34r2k4gce5v7ta3rasurg.py
# Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, result_6, linear_5, mul_5, result_7, view_1, key_states, cos_3, mul_9, k_embed], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze]
# Source node to ATen node mapping:
#   cos => cos
#   cos_1 => mul_60
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_8
#   freqs => permute
#   k_embed => add_303
#   key_states => permute_8
#   linear_5 => view_21
#   matmul => unsqueeze_default
#   mul_5 => mul_202
#   mul_9 => mul_304
#   result_6 => add_tensor_70, view_17
#   result_7 => add_195
#   view_1 => view_22
# Graph fragment:
#   %primals_13 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=primals_13]
#   %mm_default_71 : Tensor "bf16[s27, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_71]
#   %mm_3 : Tensor "bf16[s27, 128][128, 1]cuda:0" = PlaceHolder[target=mm_3]
#   %mm_default : Tensor "f32[32, s27][s27, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %mul_320 : Tensor "bf16[1, 2, s27, 64][128*s27, 1, 2, 2*s27]cuda:0" = PlaceHolder[target=mul_320]
#   %unsqueeze_default : Tensor "f32[1, 32, s27][32*s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, s27, 32][32*s27, 1, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, s27, 1, 32][32*s27, 1, 32*s27, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, s27, 2, 32][32*s27, 1, 0, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, %primals_1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, s27, 2, 32][64*s27, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, %primals_1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_8,), kwargs = {})
#   %mul_60 : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_60, torch.bfloat16), kwargs = {})
#   %add_tensor_70 : Tensor "bf16[s27, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_13, %mm_default_71), kwargs = {})
#   %view_17 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_70, [1, %primals_1, 128]), kwargs = {})
#   %view_21 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, %primals_1, 128]), kwargs = {})
#   %mul_202 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 1.0), kwargs = {})
#   %add_195 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %mul_202), kwargs = {})
#   %view_22 : Tensor "bf16[1, s27, 2, 64][128*s27, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_195, [1, %primals_1, -1, 64]), kwargs = {})
#   %permute_8 : Tensor "bf16[1, 2, s27, 64][128*s27, 64, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_22, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, s27, 64][64*s27, 64*s27, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_304 : Tensor "bf16[1, 2, s27, 64][128*s27, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_8, %unsqueeze_5), kwargs = {})
#   %add_303 : Tensor "bf16[1, 2, s27, 64][128*s27, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_304, %mul_320), kwargs = {})
#   return %add_303
triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'ks0': 'i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 2)
    y3 = yindex
    y1 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (y1 + ks0*((x2 % 32))), xmask & ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y3 + 2*ks0*x2), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tl_math.cos(tmp7)
    tmp9 = tmp8 * tmp4
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp6 * tmp10
    tmp13 = tmp11 + tmp12
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 64*y3), tmp13, xmask & ymask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/ox/coxi6lrv33g56za3vw5eiwftlyryvezsdtcbm5xdd4522ual3bjc.py
# Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, result_6, linear_5, mul_5, result_7, view_1, key_states, cos_3, mul_9, k_embed, getitem_47, hidden_states_3, key], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   cos => cos
#   cos_1 => mul_60
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_8
#   freqs => permute
#   getitem_47 => unsqueeze_7
#   hidden_states_3 => expand_5
#   k_embed => add_303
#   key => clone_2
#   key_states => permute_8
#   linear_5 => view_21
#   matmul => unsqueeze_default
#   mul_5 => mul_202
#   mul_9 => mul_304
#   result_6 => add_tensor_70, view_17
#   result_7 => add_195
#   view_1 => view_22
# Graph fragment:
#   %add_303 : Tensor "bf16[1, 2, s27, 64][128*s27, 64, 128, 1]cuda:0" = PlaceHolder[target=add_303]
#   %unsqueeze_default : Tensor "f32[1, 32, s27][32*s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, s27, 32][32*s27, 1, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, s27, 1, 32][32*s27, 1, 32*s27, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, s27, 2, 32][32*s27, 1, 0, s27]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, %primals_1, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, s27, 2, 32][64*s27, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, %primals_1, 64]), kwargs = {})
#   %cos : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_8,), kwargs = {})
#   %mul_60 : Tensor "f32[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, s27, 64][64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_60, torch.bfloat16), kwargs = {})
#   %add_tensor_70 : Tensor "bf16[s27, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_13, %mm_default_71), kwargs = {})
#   %view_17 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_70, [1, %primals_1, 128]), kwargs = {})
#   %view_21 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_3, [1, %primals_1, 128]), kwargs = {})
#   %mul_202 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 1.0), kwargs = {})
#   %add_195 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %mul_202), kwargs = {})
#   %view_22 : Tensor "bf16[1, s27, 2, 64][128*s27, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_195, [1, %primals_1, -1, 64]), kwargs = {})
#   %permute_8 : Tensor "bf16[1, 2, s27, 64][128*s27, 64, 128, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.permute.default](args = (%view_22, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, s27, 64][64*s27, 64*s27, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_304 : Tensor "bf16[1, 2, s27, 64][128*s27, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_8, %unsqueeze_5), kwargs = {})
#   %add_303 : Tensor "bf16[1, 2, s27, 64][128*s27, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_304, %mul_320), kwargs = {})
#   %unsqueeze_7 : Tensor "bf16[1, 2, 1, s27, 64][128*s27, 64, 128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%add_303, 2), kwargs = {})
#   %expand_5 : Tensor "bf16[1, 2, 7, s27, 64][128*s27, 64, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_7, [1, 2, 7, %primals_1, 64]), kwargs = {})
#   %clone_2 : Tensor "bf16[1, 2, 7, s27, 64][896*s27, 448*s27, 64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_5,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_2
triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8(in_ptr0, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % ks0)
    x3 = xindex // ks1
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x3 + 128*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/zb/czbhlknscphpyq5e7jdsif5pgmq6vmcfpsnar6dk256zk3uesleo.py
# Topologically Sorted Source Nodes: [result_9, linear_8, mul_6, result_10, view_2, value_states, getitem_52, hidden_states_4, value], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
# Source node to ATen node mapping:
#   getitem_52 => unsqueeze_8
#   hidden_states_4 => expand_6
#   linear_8 => view_28
#   mul_6 => mul_258
#   result_10 => add_242
#   result_9 => add_tensor_69, view_24
#   value => clone_3
#   value_states => permute_12
#   view_2 => view_29
# Graph fragment:
#   %primals_17 : Tensor "bf16[128][1]cuda:0" = PlaceHolder[target=primals_17]
#   %mm_default_70 : Tensor "bf16[s27, 128][128, 1]cuda:0" = PlaceHolder[target=mm_default_70]
#   %mm_5 : Tensor "bf16[s27, 128][128, 1]cuda:0" = PlaceHolder[target=mm_5]
#   %add_tensor_69 : Tensor "bf16[s27, 128][128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_17, %mm_default_70), kwargs = {})
#   %view_24 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_tensor_69, [1, %primals_1, 128]), kwargs = {})
#   %view_28 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_5, [1, %primals_1, 128]), kwargs = {})
#   %mul_258 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_28, 1.0), kwargs = {})
#   %add_242 : Tensor "bf16[1, s27, 128][128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_24, %mul_258), kwargs = {})
#   %view_29 : Tensor "bf16[1, s27, 2, 64][128*s27, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_242, [1, %primals_1, -1, 64]), kwargs = {})
#   %permute_12 : Tensor "bf16[1, 2, s27, 64][128*s27, 64, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_29, [0, 2, 1, 3]), kwargs = {})
#   %unsqueeze_8 : Tensor "bf16[1, 2, 1, s27, 64][128*s27, 64, 128*s27, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute_12, 2), kwargs = {})
#   %expand_6 : Tensor "bf16[1, 2, 7, s27, 64][128*s27, 64, 0, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_8, [1, 2, 7, %primals_1, 64]), kwargs = {})
#   %clone_3 : Tensor "bf16[1, 2, 7, s27, 64][896*s27, 448*s27, 64*s27, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_6,), kwargs = {memory_format: torch.contiguous_format})
#   return %clone_3
triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9 = async_compile.triton('triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x3 = xindex // ks0
    x1 = ((xindex // 64) % ks1)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 64*x3 + 128*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 64*x3 + 128*x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/uh/cuhs5zmhodfwxy5yuirxes32ytmcxoqxag2zf6n274yyn3jz3d2q.py
# Topologically Sorted Source Nodes: [cache_position, attention_mask, kv_arange_1, batch_arange, le, result_1, index, result_2, batched_outputs_2, attn_output], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.scalar_tensor, aten.where, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   attention_mask => convert_element_type
#   attn_output => constant_pad_nd, full_default_1, full_default_2, where
#   batch_arange => iota_2
#   batched_outputs_2 => view_4
#   cache_position => iota
#   index => index, view_2
#   kv_arange_1 => add_15
#   le => le, view
#   result_1 => bitwise_and, full_default
#   result_2 => bitwise_and_1, view_3
# Graph fragment:
#   %primals_5 : Tensor "i64[1, s16][s16, 1]cuda:0" = PlaceHolder[target=primals_5]
#   %iota : Tensor "i64[s27][1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.iota.default](args = (%primals_1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : Tensor "b8[1, s16][s16, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_5, torch.bool), kwargs = {})
#   %add_15 : Tensor "i64[s27][1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%iota, 0), kwargs = {})
#   %iota_2 : Tensor "i64[1][1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (1,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %view : Tensor "i64[s27, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota, [%primals_1, 1]), kwargs = {})
#   %le : Tensor "b8[s27, s27][s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.le.Tensor](args = (%add_15, %view), kwargs = {})
#   %full_default : Tensor "b8[s27, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([%primals_1, 1], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bitwise_and : Tensor "b8[s27, s27][s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%full_default, %le), kwargs = {})
#   %view_2 : Tensor "i64[1, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%iota_2, [1, 1]), kwargs = {})
#   %index : Tensor "b8[1, s27][s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%convert_element_type, [%view_2, %add_15]), kwargs = {})
#   %view_3 : Tensor "b8[1, 1, s27][s27, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%index, [1, 1, %primals_1]), kwargs = {})
#   %bitwise_and_1 : Tensor "b8[1, s27, s27][s27**2, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.bitwise_and.Tensor](args = (%bitwise_and, %view_3), kwargs = {})
#   %view_4 : Tensor "b8[1, 1, s27, s27][s27**2, s27**2, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%bitwise_and_1, [1, 1, %primals_1, %primals_1]), kwargs = {})
#   %full_default_1 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : Tensor "bf16[][]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : Tensor "bf16[1, 1, s27, s27][s27**2, s27**2, s27, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%expand, %full_default_2, %full_default_1), kwargs = {})
#   %constant_pad_nd : Tensor "bf16[1, 1, s27, s27 - (Mod(s27, 8)) + 8][s27*Max(1, s27 - (Mod(s27, 8)) + 8), s27*Max(1, s27 - (Mod(s27, 8)) + 8), Max(1, s27 - (Mod(s27, 8)) + 8), 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%where, [0, %sub_125], 0.0), kwargs = {})
#   return %constant_pad_nd
triton_poi_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_index_le_scalar_tensor_view_where_10 = async_compile.triton('triton_poi_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_index_le_scalar_tensor_view_where_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*bf16', 'ks0': 'i64', 'ks1': 'i64', 'ks2': 'i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_index_le_scalar_tensor_view_where_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_index_le_scalar_tensor_view_where_10(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    tmp0 = x0
    tmp1 = ks1
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = x1
    tmp5 = tmp3 <= tmp4
    tmp6 = tl.full([1], True, tl.int1)
    tmp7 = tmp6 & tmp5
    tl.device_assert((x0 < ks2) | ~(tmp2 & xmask), "index out of bounds: x0 < ks2")
    tmp9 = tl.load(in_ptr0 + (x0), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = (tmp9 != 0)
    tmp11 = tmp7 & tmp10
    tmp12 = 0.0
    tmp13 = float("-inf")
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp2, tmp14, tmp15)
    tl.store(out_ptr0 + (x0 + x1*((1) * ((1) >= (ks0)) + (ks0) * ((ks0) > (1)))), tmp16, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/ci/cci7iwqnis25akm6cfdysj5pnopvwi4f7utnczyf4diioxthal5j.py
# Topologically Sorted Source Nodes: [transpose_4, reshape_2, linear_10], Original ATen: [aten.transpose, aten.view, aten._to_copy]
# Source node to ATen node mapping:
#   linear_10 => convert_element_type_default_329
#   reshape_2 => view_32
#   transpose_4 => permute_13
# Graph fragment:
#   %getitem : Tensor "bf16[1, 14, s27, 64][896*s27, 64, 896, 1]cuda:0" = PlaceHolder[target=getitem]
#   %permute_13 : Tensor "bf16[1, s27, 14, 64][896*s27, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%getitem, [0, 2, 1, 3]), kwargs = {})
#   %view_32 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_13, [1, %primals_1, -1]), kwargs = {})
#   %convert_element_type_default_329 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_32, torch.bfloat16), kwargs = {})
#   return %convert_element_type_default_329
triton_poi_fused__to_copy_transpose_view_11 = async_compile.triton('triton_poi_fused__to_copy_transpose_view_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_transpose_view_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_transpose_view_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/at/cat4l22vjwd2akiljl7m5bhs2cozzzt4yaicg5vd5eluu3esn725.py
# Topologically Sorted Source Nodes: [inputs_embeds, result_12, linear_11, mul_11, result_13, hidden_states_5, hidden_states_6, pow_2, variance_1, add_10, rsqrt_1, hidden_states_7, to_17, hidden_states_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   add_10 => add_427
#   hidden_states_5 => add_414
#   hidden_states_6 => convert_element_type_49
#   hidden_states_7 => mul_521
#   hidden_states_8 => mul_528
#   inputs_embeds => embedding
#   linear_11 => view_38
#   mul_11 => mul_496
#   pow_2 => pow_2
#   result_12 => view_34
#   result_13 => add_410
#   rsqrt_1 => rsqrt_1
#   to_17 => convert_element_type_50
#   variance_1 => mean_1
# Graph fragment:
#   %primals_2 : Tensor "i64[1, s27][s27, 1]cuda:0" = PlaceHolder[target=primals_2]
#   %primals_3 : Tensor "bf16[151936, 896][896, 1]cuda:0" = PlaceHolder[target=primals_3]
#   %mm_6 : Tensor "bf16[s27, 896][896, 1]cuda:0" = PlaceHolder[target=mm_6]
#   %mm_8 : Tensor "bf16[s27, 896][896, 1]cuda:0" = PlaceHolder[target=mm_8]
#   %add_414 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0" = PlaceHolder[target=add_414]
#   %buf38 : Tensor "f32[1, s27, 1][s27, 1, s27]cuda:0" = PlaceHolder[target=buf38]
#   %primals_23 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=primals_23]
#   %rsqrt_1 : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_1]
#   %embedding : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_3, %primals_2), kwargs = {})
#   %view_34 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_6, [1, %primals_1, 896]), kwargs = {})
#   %view_38 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_8, [1, %primals_1, 896]), kwargs = {})
#   %mul_496 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_38, 1.0), kwargs = {})
#   %add_410 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_34, %mul_496), kwargs = {})
#   %add_414 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, %add_410), kwargs = {})
#   %convert_element_type_49 : Tensor "f32[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_414, torch.float32), kwargs = {})
#   %pow_2 : Tensor "f32[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_49, 2), kwargs = {})
#   %mean_1 : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_427 : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_427,), kwargs = {})
#   %mul_521 : Tensor "f32[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_49, %rsqrt_1), kwargs = {})
#   %convert_element_type_50 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_521, torch.bfloat16), kwargs = {})
#   %mul_528 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_23, %convert_element_type_50), kwargs = {})
#   return %add_414,%buf38,%rsqrt_1,%mul_528
triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_12 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_12', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_12(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 896
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tl.full([1, 1], 151936, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 151936)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 151936")
    tmp6 = tl.load(in_ptr1 + (r0_1 + 896*tmp4), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp19 = 896.0
    tmp20 = (tmp18 / tmp19)
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp25 = tmp13 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tl.store(in_out_ptr0 + (r0_1 + 896*x0), tmp12, r0_mask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, xmask)
    tl.store(out_ptr0 + (r0_1 + 896*x0), tmp27, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/4z/c4zvvbuhgguglhoc5rj3pmejjkp7lnvnmqkpjzlnlpy5ocqgvyp3.py
# Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_14 => convert_element_type_58, permute_19
# Graph fragment:
#   %primals_26 : Tensor "f32[4864, 32][32, 1]cuda:0" = PlaceHolder[target=primals_26]
#   %convert_element_type_58 : Tensor "bf16[4864, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_26, torch.bfloat16), kwargs = {})
#   %permute_19 : Tensor "bf16[32, 4864][1, 32]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_58, [1, 0]), kwargs = {})
#   return %permute_19
triton_poi_fused__to_copy_t_13 = async_compile.triton('triton_poi_fused__to_copy_t_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1245184}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 155648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/ij/cijvxbvo733dt6mfpifgwfsfsw56spvsmgilepxm74ghyk44zlry.py
# Topologically Sorted Source Nodes: [result_15, linear_14, mul_14, result_16, silu, result_18, linear_17, mul_15, result_19, mul_16], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
# Source node to ATen node mapping:
#   linear_14 => view_44
#   linear_17 => view_50
#   mul_14 => mul_584
#   mul_15 => mul_644
#   mul_16 => mul_651
#   result_15 => view_40
#   result_16 => add_482
#   result_18 => view_46
#   result_19 => add_525
#   silu => convert_element_type_61, convert_element_type_62, mul_591, sigmoid
# Graph fragment:
#   %mm_9 : Tensor "bf16[s27, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_9]
#   %mm_11 : Tensor "bf16[s27, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_11]
#   %mm_12 : Tensor "bf16[s27, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_12]
#   %mm_14 : Tensor "bf16[s27, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_14]
#   %add_482 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0" = PlaceHolder[target=add_482]
#   %add_525 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0" = PlaceHolder[target=add_525]
#   %view_40 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_9, [1, %primals_1, 4864]), kwargs = {})
#   %view_44 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_11, [1, %primals_1, 4864]), kwargs = {})
#   %mul_584 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_44, 1.0), kwargs = {})
#   %add_482 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_40, %mul_584), kwargs = {})
#   %convert_element_type_61 : Tensor "f32[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_482, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_61,), kwargs = {})
#   %mul_591 : Tensor "f32[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_61, %sigmoid), kwargs = {})
#   %convert_element_type_62 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_591, torch.bfloat16), kwargs = {})
#   %view_46 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_12, [1, %primals_1, 4864]), kwargs = {})
#   %view_50 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_14, [1, %primals_1, 4864]), kwargs = {})
#   %mul_644 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_50, 1.0), kwargs = {})
#   %add_525 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_46, %mul_644), kwargs = {})
#   %mul_651 : Tensor "bf16[1, s27, 4864][4864*s27, 4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_62, %add_525), kwargs = {})
#   return %add_482,%add_525,%mul_651
triton_poi_fused__unsafe_view_add_mul_silu_14 = async_compile.triton('triton_poi_fused__unsafe_view_add_mul_silu_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_mul_silu_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 3, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_add_mul_silu_14(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_out_ptr1 + (x0), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp6 * tmp2
    tmp8 = tmp5 + tmp7
    tmp9 = tmp4.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp8
    tl.store(in_out_ptr0 + (x0), tmp4, xmask)
    tl.store(in_out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/2m/c2mmvqmzg73alixqmupyqkmyk4ne3wixppnpwsyfkpsj3i3ztde5.py
# Topologically Sorted Source Nodes: [result_21, linear_20, mul_17, result_22, hidden_states_9, hidden_states_10, pow_3, variance_2, add_15, rsqrt_2, hidden_states_11, to_25, hidden_states_12], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
# Source node to ATen node mapping:
#   add_15 => add_585
#   hidden_states_10 => convert_element_type_83
#   hidden_states_11 => mul_732
#   hidden_states_12 => mul_739
#   hidden_states_9 => add_572
#   linear_20 => view_56
#   mul_17 => mul_707
#   pow_3 => pow_3
#   result_21 => view_52
#   result_22 => add_568
#   rsqrt_2 => rsqrt_2
#   to_25 => convert_element_type_84
#   variance_2 => mean_2
# Graph fragment:
#   %add_414 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0" = PlaceHolder[target=add_414]
#   %mm_15 : Tensor "bf16[s27, 896][896, 1]cuda:0" = PlaceHolder[target=mm_15]
#   %mm_17 : Tensor "bf16[s27, 896][896, 1]cuda:0" = PlaceHolder[target=mm_17]
#   %add_572 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0" = PlaceHolder[target=add_572]
#   %buf60 : Tensor "f32[1, s27, 1][s27, 1, s27]cuda:0" = PlaceHolder[target=buf60]
#   %primals_33 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=primals_33]
#   %rsqrt_2 : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_2]
#   %view_52 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_15, [1, %primals_1, 896]), kwargs = {})
#   %view_56 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_17, [1, %primals_1, 896]), kwargs = {})
#   %mul_707 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_56, 1.0), kwargs = {})
#   %add_568 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_52, %mul_707), kwargs = {})
#   %add_572 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_414, %add_568), kwargs = {})
#   %convert_element_type_83 : Tensor "f32[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_572, torch.float32), kwargs = {})
#   %pow_3 : Tensor "f32[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_83, 2), kwargs = {})
#   %mean_2 : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %add_585 : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-06), kwargs = {})
#   %rsqrt_2 : Tensor "f32[1, s27, 1][s27, 1, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_585,), kwargs = {})
#   %mul_732 : Tensor "f32[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_83, %rsqrt_2), kwargs = {})
#   %convert_element_type_84 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_732, torch.bfloat16), kwargs = {})
#   %mul_739 : Tensor "bf16[1, s27, 896][896*s27, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_33, %convert_element_type_84), kwargs = {})
#   return %add_572,%buf60,%rsqrt_2,%mul_739
triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15 = async_compile.triton('triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_out_ptr1': '*fp32', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 3, 'num_reduction': 1, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    r0_numel = 896
    R0_BLOCK: tl.constexpr = 1024
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tmp5 = tmp1 + tmp4
    tmp6 = tmp0 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = 896.0
    tmp14 = (tmp12 / tmp13)
    tmp15 = 1e-06
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.rsqrt(tmp16)
    tmp19 = tmp7 * tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tl.store(in_out_ptr0 + (r0_1 + 896*x0), tmp6, r0_mask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp17, xmask)
    tl.store(out_ptr0 + (r0_1 + 896*x0), tmp21, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/l3/cl3ual4zhx64pwwqfc2pa6ig7iifoipc6otsprw6pagjs5j2v2af.py
# Topologically Sorted Source Nodes: [logits, float_5], Original ATen: [aten._unsafe_view, aten._to_copy]
# Source node to ATen node mapping:
#   float_5 => convert_element_type_1904
#   logits => view_1162
# Graph fragment:
#   %mm_432 : Tensor "bf16[s35, 151936][151936, 1]cuda:0" = PlaceHolder[target=mm_432]
#   %view_1162 : Tensor "bf16[1, s35, 151936][151936*s35, 151936, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_432, [1, %primals_632, 151936]), kwargs = {})
#   %convert_element_type_1904 : Tensor "f32[1, s35, 151936][151936*s35, 151936, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1162, torch.float32), kwargs = {})
#   return %convert_element_type_1904
triton_poi_fused__to_copy__unsafe_view_16 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633 = args
        args.clear()
        s27 = primals_1
        s16 = primals_4
        s35 = primals_632
        assert_size_stride(primals_2, (1, s27), (s27, 1))
        assert_size_stride(primals_3, (151936, 896), (896, 1))
        assert_size_stride(primals_5, (1, s16), (s16, 1))
        assert_size_stride(primals_6, (32, ), (1, ))
        assert_size_stride(primals_7, (896, ), (1, ))
        assert_size_stride(primals_8, (896, 896), (896, 1))
        assert_size_stride(primals_9, (896, ), (1, ))
        assert_size_stride(primals_10, (32, 896), (896, 1))
        assert_size_stride(primals_11, (896, 32), (32, 1))
        assert_size_stride(primals_12, (128, 896), (896, 1))
        assert_size_stride(primals_13, (128, ), (1, ))
        assert_size_stride(primals_14, (32, 896), (896, 1))
        assert_size_stride(primals_15, (128, 32), (32, 1))
        assert_size_stride(primals_16, (128, 896), (896, 1))
        assert_size_stride(primals_17, (128, ), (1, ))
        assert_size_stride(primals_18, (32, 896), (896, 1))
        assert_size_stride(primals_19, (128, 32), (32, 1))
        assert_size_stride(primals_20, (896, 896), (896, 1))
        assert_size_stride(primals_21, (32, 896), (896, 1))
        assert_size_stride(primals_22, (896, 32), (32, 1))
        assert_size_stride(primals_23, (896, ), (1, ))
        assert_size_stride(primals_24, (4864, 896), (896, 1))
        assert_size_stride(primals_25, (32, 896), (896, 1))
        assert_size_stride(primals_26, (4864, 32), (32, 1))
        assert_size_stride(primals_27, (4864, 896), (896, 1))
        assert_size_stride(primals_28, (32, 896), (896, 1))
        assert_size_stride(primals_29, (4864, 32), (32, 1))
        assert_size_stride(primals_30, (896, 4864), (4864, 1))
        assert_size_stride(primals_31, (32, 4864), (4864, 1))
        assert_size_stride(primals_32, (896, 32), (32, 1))
        assert_size_stride(primals_33, (896, ), (1, ))
        assert_size_stride(primals_34, (896, 896), (896, 1))
        assert_size_stride(primals_35, (896, ), (1, ))
        assert_size_stride(primals_36, (32, 896), (896, 1))
        assert_size_stride(primals_37, (896, 32), (32, 1))
        assert_size_stride(primals_38, (128, 896), (896, 1))
        assert_size_stride(primals_39, (128, ), (1, ))
        assert_size_stride(primals_40, (32, 896), (896, 1))
        assert_size_stride(primals_41, (128, 32), (32, 1))
        assert_size_stride(primals_42, (128, 896), (896, 1))
        assert_size_stride(primals_43, (128, ), (1, ))
        assert_size_stride(primals_44, (32, 896), (896, 1))
        assert_size_stride(primals_45, (128, 32), (32, 1))
        assert_size_stride(primals_46, (896, 896), (896, 1))
        assert_size_stride(primals_47, (32, 896), (896, 1))
        assert_size_stride(primals_48, (896, 32), (32, 1))
        assert_size_stride(primals_49, (896, ), (1, ))
        assert_size_stride(primals_50, (4864, 896), (896, 1))
        assert_size_stride(primals_51, (32, 896), (896, 1))
        assert_size_stride(primals_52, (4864, 32), (32, 1))
        assert_size_stride(primals_53, (4864, 896), (896, 1))
        assert_size_stride(primals_54, (32, 896), (896, 1))
        assert_size_stride(primals_55, (4864, 32), (32, 1))
        assert_size_stride(primals_56, (896, 4864), (4864, 1))
        assert_size_stride(primals_57, (32, 4864), (4864, 1))
        assert_size_stride(primals_58, (896, 32), (32, 1))
        assert_size_stride(primals_59, (896, ), (1, ))
        assert_size_stride(primals_60, (896, 896), (896, 1))
        assert_size_stride(primals_61, (896, ), (1, ))
        assert_size_stride(primals_62, (32, 896), (896, 1))
        assert_size_stride(primals_63, (896, 32), (32, 1))
        assert_size_stride(primals_64, (128, 896), (896, 1))
        assert_size_stride(primals_65, (128, ), (1, ))
        assert_size_stride(primals_66, (32, 896), (896, 1))
        assert_size_stride(primals_67, (128, 32), (32, 1))
        assert_size_stride(primals_68, (128, 896), (896, 1))
        assert_size_stride(primals_69, (128, ), (1, ))
        assert_size_stride(primals_70, (32, 896), (896, 1))
        assert_size_stride(primals_71, (128, 32), (32, 1))
        assert_size_stride(primals_72, (896, 896), (896, 1))
        assert_size_stride(primals_73, (32, 896), (896, 1))
        assert_size_stride(primals_74, (896, 32), (32, 1))
        assert_size_stride(primals_75, (896, ), (1, ))
        assert_size_stride(primals_76, (4864, 896), (896, 1))
        assert_size_stride(primals_77, (32, 896), (896, 1))
        assert_size_stride(primals_78, (4864, 32), (32, 1))
        assert_size_stride(primals_79, (4864, 896), (896, 1))
        assert_size_stride(primals_80, (32, 896), (896, 1))
        assert_size_stride(primals_81, (4864, 32), (32, 1))
        assert_size_stride(primals_82, (896, 4864), (4864, 1))
        assert_size_stride(primals_83, (32, 4864), (4864, 1))
        assert_size_stride(primals_84, (896, 32), (32, 1))
        assert_size_stride(primals_85, (896, ), (1, ))
        assert_size_stride(primals_86, (896, 896), (896, 1))
        assert_size_stride(primals_87, (896, ), (1, ))
        assert_size_stride(primals_88, (32, 896), (896, 1))
        assert_size_stride(primals_89, (896, 32), (32, 1))
        assert_size_stride(primals_90, (128, 896), (896, 1))
        assert_size_stride(primals_91, (128, ), (1, ))
        assert_size_stride(primals_92, (32, 896), (896, 1))
        assert_size_stride(primals_93, (128, 32), (32, 1))
        assert_size_stride(primals_94, (128, 896), (896, 1))
        assert_size_stride(primals_95, (128, ), (1, ))
        assert_size_stride(primals_96, (32, 896), (896, 1))
        assert_size_stride(primals_97, (128, 32), (32, 1))
        assert_size_stride(primals_98, (896, 896), (896, 1))
        assert_size_stride(primals_99, (32, 896), (896, 1))
        assert_size_stride(primals_100, (896, 32), (32, 1))
        assert_size_stride(primals_101, (896, ), (1, ))
        assert_size_stride(primals_102, (4864, 896), (896, 1))
        assert_size_stride(primals_103, (32, 896), (896, 1))
        assert_size_stride(primals_104, (4864, 32), (32, 1))
        assert_size_stride(primals_105, (4864, 896), (896, 1))
        assert_size_stride(primals_106, (32, 896), (896, 1))
        assert_size_stride(primals_107, (4864, 32), (32, 1))
        assert_size_stride(primals_108, (896, 4864), (4864, 1))
        assert_size_stride(primals_109, (32, 4864), (4864, 1))
        assert_size_stride(primals_110, (896, 32), (32, 1))
        assert_size_stride(primals_111, (896, ), (1, ))
        assert_size_stride(primals_112, (896, 896), (896, 1))
        assert_size_stride(primals_113, (896, ), (1, ))
        assert_size_stride(primals_114, (32, 896), (896, 1))
        assert_size_stride(primals_115, (896, 32), (32, 1))
        assert_size_stride(primals_116, (128, 896), (896, 1))
        assert_size_stride(primals_117, (128, ), (1, ))
        assert_size_stride(primals_118, (32, 896), (896, 1))
        assert_size_stride(primals_119, (128, 32), (32, 1))
        assert_size_stride(primals_120, (128, 896), (896, 1))
        assert_size_stride(primals_121, (128, ), (1, ))
        assert_size_stride(primals_122, (32, 896), (896, 1))
        assert_size_stride(primals_123, (128, 32), (32, 1))
        assert_size_stride(primals_124, (896, 896), (896, 1))
        assert_size_stride(primals_125, (32, 896), (896, 1))
        assert_size_stride(primals_126, (896, 32), (32, 1))
        assert_size_stride(primals_127, (896, ), (1, ))
        assert_size_stride(primals_128, (4864, 896), (896, 1))
        assert_size_stride(primals_129, (32, 896), (896, 1))
        assert_size_stride(primals_130, (4864, 32), (32, 1))
        assert_size_stride(primals_131, (4864, 896), (896, 1))
        assert_size_stride(primals_132, (32, 896), (896, 1))
        assert_size_stride(primals_133, (4864, 32), (32, 1))
        assert_size_stride(primals_134, (896, 4864), (4864, 1))
        assert_size_stride(primals_135, (32, 4864), (4864, 1))
        assert_size_stride(primals_136, (896, 32), (32, 1))
        assert_size_stride(primals_137, (896, ), (1, ))
        assert_size_stride(primals_138, (896, 896), (896, 1))
        assert_size_stride(primals_139, (896, ), (1, ))
        assert_size_stride(primals_140, (32, 896), (896, 1))
        assert_size_stride(primals_141, (896, 32), (32, 1))
        assert_size_stride(primals_142, (128, 896), (896, 1))
        assert_size_stride(primals_143, (128, ), (1, ))
        assert_size_stride(primals_144, (32, 896), (896, 1))
        assert_size_stride(primals_145, (128, 32), (32, 1))
        assert_size_stride(primals_146, (128, 896), (896, 1))
        assert_size_stride(primals_147, (128, ), (1, ))
        assert_size_stride(primals_148, (32, 896), (896, 1))
        assert_size_stride(primals_149, (128, 32), (32, 1))
        assert_size_stride(primals_150, (896, 896), (896, 1))
        assert_size_stride(primals_151, (32, 896), (896, 1))
        assert_size_stride(primals_152, (896, 32), (32, 1))
        assert_size_stride(primals_153, (896, ), (1, ))
        assert_size_stride(primals_154, (4864, 896), (896, 1))
        assert_size_stride(primals_155, (32, 896), (896, 1))
        assert_size_stride(primals_156, (4864, 32), (32, 1))
        assert_size_stride(primals_157, (4864, 896), (896, 1))
        assert_size_stride(primals_158, (32, 896), (896, 1))
        assert_size_stride(primals_159, (4864, 32), (32, 1))
        assert_size_stride(primals_160, (896, 4864), (4864, 1))
        assert_size_stride(primals_161, (32, 4864), (4864, 1))
        assert_size_stride(primals_162, (896, 32), (32, 1))
        assert_size_stride(primals_163, (896, ), (1, ))
        assert_size_stride(primals_164, (896, 896), (896, 1))
        assert_size_stride(primals_165, (896, ), (1, ))
        assert_size_stride(primals_166, (32, 896), (896, 1))
        assert_size_stride(primals_167, (896, 32), (32, 1))
        assert_size_stride(primals_168, (128, 896), (896, 1))
        assert_size_stride(primals_169, (128, ), (1, ))
        assert_size_stride(primals_170, (32, 896), (896, 1))
        assert_size_stride(primals_171, (128, 32), (32, 1))
        assert_size_stride(primals_172, (128, 896), (896, 1))
        assert_size_stride(primals_173, (128, ), (1, ))
        assert_size_stride(primals_174, (32, 896), (896, 1))
        assert_size_stride(primals_175, (128, 32), (32, 1))
        assert_size_stride(primals_176, (896, 896), (896, 1))
        assert_size_stride(primals_177, (32, 896), (896, 1))
        assert_size_stride(primals_178, (896, 32), (32, 1))
        assert_size_stride(primals_179, (896, ), (1, ))
        assert_size_stride(primals_180, (4864, 896), (896, 1))
        assert_size_stride(primals_181, (32, 896), (896, 1))
        assert_size_stride(primals_182, (4864, 32), (32, 1))
        assert_size_stride(primals_183, (4864, 896), (896, 1))
        assert_size_stride(primals_184, (32, 896), (896, 1))
        assert_size_stride(primals_185, (4864, 32), (32, 1))
        assert_size_stride(primals_186, (896, 4864), (4864, 1))
        assert_size_stride(primals_187, (32, 4864), (4864, 1))
        assert_size_stride(primals_188, (896, 32), (32, 1))
        assert_size_stride(primals_189, (896, ), (1, ))
        assert_size_stride(primals_190, (896, 896), (896, 1))
        assert_size_stride(primals_191, (896, ), (1, ))
        assert_size_stride(primals_192, (32, 896), (896, 1))
        assert_size_stride(primals_193, (896, 32), (32, 1))
        assert_size_stride(primals_194, (128, 896), (896, 1))
        assert_size_stride(primals_195, (128, ), (1, ))
        assert_size_stride(primals_196, (32, 896), (896, 1))
        assert_size_stride(primals_197, (128, 32), (32, 1))
        assert_size_stride(primals_198, (128, 896), (896, 1))
        assert_size_stride(primals_199, (128, ), (1, ))
        assert_size_stride(primals_200, (32, 896), (896, 1))
        assert_size_stride(primals_201, (128, 32), (32, 1))
        assert_size_stride(primals_202, (896, 896), (896, 1))
        assert_size_stride(primals_203, (32, 896), (896, 1))
        assert_size_stride(primals_204, (896, 32), (32, 1))
        assert_size_stride(primals_205, (896, ), (1, ))
        assert_size_stride(primals_206, (4864, 896), (896, 1))
        assert_size_stride(primals_207, (32, 896), (896, 1))
        assert_size_stride(primals_208, (4864, 32), (32, 1))
        assert_size_stride(primals_209, (4864, 896), (896, 1))
        assert_size_stride(primals_210, (32, 896), (896, 1))
        assert_size_stride(primals_211, (4864, 32), (32, 1))
        assert_size_stride(primals_212, (896, 4864), (4864, 1))
        assert_size_stride(primals_213, (32, 4864), (4864, 1))
        assert_size_stride(primals_214, (896, 32), (32, 1))
        assert_size_stride(primals_215, (896, ), (1, ))
        assert_size_stride(primals_216, (896, 896), (896, 1))
        assert_size_stride(primals_217, (896, ), (1, ))
        assert_size_stride(primals_218, (32, 896), (896, 1))
        assert_size_stride(primals_219, (896, 32), (32, 1))
        assert_size_stride(primals_220, (128, 896), (896, 1))
        assert_size_stride(primals_221, (128, ), (1, ))
        assert_size_stride(primals_222, (32, 896), (896, 1))
        assert_size_stride(primals_223, (128, 32), (32, 1))
        assert_size_stride(primals_224, (128, 896), (896, 1))
        assert_size_stride(primals_225, (128, ), (1, ))
        assert_size_stride(primals_226, (32, 896), (896, 1))
        assert_size_stride(primals_227, (128, 32), (32, 1))
        assert_size_stride(primals_228, (896, 896), (896, 1))
        assert_size_stride(primals_229, (32, 896), (896, 1))
        assert_size_stride(primals_230, (896, 32), (32, 1))
        assert_size_stride(primals_231, (896, ), (1, ))
        assert_size_stride(primals_232, (4864, 896), (896, 1))
        assert_size_stride(primals_233, (32, 896), (896, 1))
        assert_size_stride(primals_234, (4864, 32), (32, 1))
        assert_size_stride(primals_235, (4864, 896), (896, 1))
        assert_size_stride(primals_236, (32, 896), (896, 1))
        assert_size_stride(primals_237, (4864, 32), (32, 1))
        assert_size_stride(primals_238, (896, 4864), (4864, 1))
        assert_size_stride(primals_239, (32, 4864), (4864, 1))
        assert_size_stride(primals_240, (896, 32), (32, 1))
        assert_size_stride(primals_241, (896, ), (1, ))
        assert_size_stride(primals_242, (896, 896), (896, 1))
        assert_size_stride(primals_243, (896, ), (1, ))
        assert_size_stride(primals_244, (32, 896), (896, 1))
        assert_size_stride(primals_245, (896, 32), (32, 1))
        assert_size_stride(primals_246, (128, 896), (896, 1))
        assert_size_stride(primals_247, (128, ), (1, ))
        assert_size_stride(primals_248, (32, 896), (896, 1))
        assert_size_stride(primals_249, (128, 32), (32, 1))
        assert_size_stride(primals_250, (128, 896), (896, 1))
        assert_size_stride(primals_251, (128, ), (1, ))
        assert_size_stride(primals_252, (32, 896), (896, 1))
        assert_size_stride(primals_253, (128, 32), (32, 1))
        assert_size_stride(primals_254, (896, 896), (896, 1))
        assert_size_stride(primals_255, (32, 896), (896, 1))
        assert_size_stride(primals_256, (896, 32), (32, 1))
        assert_size_stride(primals_257, (896, ), (1, ))
        assert_size_stride(primals_258, (4864, 896), (896, 1))
        assert_size_stride(primals_259, (32, 896), (896, 1))
        assert_size_stride(primals_260, (4864, 32), (32, 1))
        assert_size_stride(primals_261, (4864, 896), (896, 1))
        assert_size_stride(primals_262, (32, 896), (896, 1))
        assert_size_stride(primals_263, (4864, 32), (32, 1))
        assert_size_stride(primals_264, (896, 4864), (4864, 1))
        assert_size_stride(primals_265, (32, 4864), (4864, 1))
        assert_size_stride(primals_266, (896, 32), (32, 1))
        assert_size_stride(primals_267, (896, ), (1, ))
        assert_size_stride(primals_268, (896, 896), (896, 1))
        assert_size_stride(primals_269, (896, ), (1, ))
        assert_size_stride(primals_270, (32, 896), (896, 1))
        assert_size_stride(primals_271, (896, 32), (32, 1))
        assert_size_stride(primals_272, (128, 896), (896, 1))
        assert_size_stride(primals_273, (128, ), (1, ))
        assert_size_stride(primals_274, (32, 896), (896, 1))
        assert_size_stride(primals_275, (128, 32), (32, 1))
        assert_size_stride(primals_276, (128, 896), (896, 1))
        assert_size_stride(primals_277, (128, ), (1, ))
        assert_size_stride(primals_278, (32, 896), (896, 1))
        assert_size_stride(primals_279, (128, 32), (32, 1))
        assert_size_stride(primals_280, (896, 896), (896, 1))
        assert_size_stride(primals_281, (32, 896), (896, 1))
        assert_size_stride(primals_282, (896, 32), (32, 1))
        assert_size_stride(primals_283, (896, ), (1, ))
        assert_size_stride(primals_284, (4864, 896), (896, 1))
        assert_size_stride(primals_285, (32, 896), (896, 1))
        assert_size_stride(primals_286, (4864, 32), (32, 1))
        assert_size_stride(primals_287, (4864, 896), (896, 1))
        assert_size_stride(primals_288, (32, 896), (896, 1))
        assert_size_stride(primals_289, (4864, 32), (32, 1))
        assert_size_stride(primals_290, (896, 4864), (4864, 1))
        assert_size_stride(primals_291, (32, 4864), (4864, 1))
        assert_size_stride(primals_292, (896, 32), (32, 1))
        assert_size_stride(primals_293, (896, ), (1, ))
        assert_size_stride(primals_294, (896, 896), (896, 1))
        assert_size_stride(primals_295, (896, ), (1, ))
        assert_size_stride(primals_296, (32, 896), (896, 1))
        assert_size_stride(primals_297, (896, 32), (32, 1))
        assert_size_stride(primals_298, (128, 896), (896, 1))
        assert_size_stride(primals_299, (128, ), (1, ))
        assert_size_stride(primals_300, (32, 896), (896, 1))
        assert_size_stride(primals_301, (128, 32), (32, 1))
        assert_size_stride(primals_302, (128, 896), (896, 1))
        assert_size_stride(primals_303, (128, ), (1, ))
        assert_size_stride(primals_304, (32, 896), (896, 1))
        assert_size_stride(primals_305, (128, 32), (32, 1))
        assert_size_stride(primals_306, (896, 896), (896, 1))
        assert_size_stride(primals_307, (32, 896), (896, 1))
        assert_size_stride(primals_308, (896, 32), (32, 1))
        assert_size_stride(primals_309, (896, ), (1, ))
        assert_size_stride(primals_310, (4864, 896), (896, 1))
        assert_size_stride(primals_311, (32, 896), (896, 1))
        assert_size_stride(primals_312, (4864, 32), (32, 1))
        assert_size_stride(primals_313, (4864, 896), (896, 1))
        assert_size_stride(primals_314, (32, 896), (896, 1))
        assert_size_stride(primals_315, (4864, 32), (32, 1))
        assert_size_stride(primals_316, (896, 4864), (4864, 1))
        assert_size_stride(primals_317, (32, 4864), (4864, 1))
        assert_size_stride(primals_318, (896, 32), (32, 1))
        assert_size_stride(primals_319, (896, ), (1, ))
        assert_size_stride(primals_320, (896, 896), (896, 1))
        assert_size_stride(primals_321, (896, ), (1, ))
        assert_size_stride(primals_322, (32, 896), (896, 1))
        assert_size_stride(primals_323, (896, 32), (32, 1))
        assert_size_stride(primals_324, (128, 896), (896, 1))
        assert_size_stride(primals_325, (128, ), (1, ))
        assert_size_stride(primals_326, (32, 896), (896, 1))
        assert_size_stride(primals_327, (128, 32), (32, 1))
        assert_size_stride(primals_328, (128, 896), (896, 1))
        assert_size_stride(primals_329, (128, ), (1, ))
        assert_size_stride(primals_330, (32, 896), (896, 1))
        assert_size_stride(primals_331, (128, 32), (32, 1))
        assert_size_stride(primals_332, (896, 896), (896, 1))
        assert_size_stride(primals_333, (32, 896), (896, 1))
        assert_size_stride(primals_334, (896, 32), (32, 1))
        assert_size_stride(primals_335, (896, ), (1, ))
        assert_size_stride(primals_336, (4864, 896), (896, 1))
        assert_size_stride(primals_337, (32, 896), (896, 1))
        assert_size_stride(primals_338, (4864, 32), (32, 1))
        assert_size_stride(primals_339, (4864, 896), (896, 1))
        assert_size_stride(primals_340, (32, 896), (896, 1))
        assert_size_stride(primals_341, (4864, 32), (32, 1))
        assert_size_stride(primals_342, (896, 4864), (4864, 1))
        assert_size_stride(primals_343, (32, 4864), (4864, 1))
        assert_size_stride(primals_344, (896, 32), (32, 1))
        assert_size_stride(primals_345, (896, ), (1, ))
        assert_size_stride(primals_346, (896, 896), (896, 1))
        assert_size_stride(primals_347, (896, ), (1, ))
        assert_size_stride(primals_348, (32, 896), (896, 1))
        assert_size_stride(primals_349, (896, 32), (32, 1))
        assert_size_stride(primals_350, (128, 896), (896, 1))
        assert_size_stride(primals_351, (128, ), (1, ))
        assert_size_stride(primals_352, (32, 896), (896, 1))
        assert_size_stride(primals_353, (128, 32), (32, 1))
        assert_size_stride(primals_354, (128, 896), (896, 1))
        assert_size_stride(primals_355, (128, ), (1, ))
        assert_size_stride(primals_356, (32, 896), (896, 1))
        assert_size_stride(primals_357, (128, 32), (32, 1))
        assert_size_stride(primals_358, (896, 896), (896, 1))
        assert_size_stride(primals_359, (32, 896), (896, 1))
        assert_size_stride(primals_360, (896, 32), (32, 1))
        assert_size_stride(primals_361, (896, ), (1, ))
        assert_size_stride(primals_362, (4864, 896), (896, 1))
        assert_size_stride(primals_363, (32, 896), (896, 1))
        assert_size_stride(primals_364, (4864, 32), (32, 1))
        assert_size_stride(primals_365, (4864, 896), (896, 1))
        assert_size_stride(primals_366, (32, 896), (896, 1))
        assert_size_stride(primals_367, (4864, 32), (32, 1))
        assert_size_stride(primals_368, (896, 4864), (4864, 1))
        assert_size_stride(primals_369, (32, 4864), (4864, 1))
        assert_size_stride(primals_370, (896, 32), (32, 1))
        assert_size_stride(primals_371, (896, ), (1, ))
        assert_size_stride(primals_372, (896, 896), (896, 1))
        assert_size_stride(primals_373, (896, ), (1, ))
        assert_size_stride(primals_374, (32, 896), (896, 1))
        assert_size_stride(primals_375, (896, 32), (32, 1))
        assert_size_stride(primals_376, (128, 896), (896, 1))
        assert_size_stride(primals_377, (128, ), (1, ))
        assert_size_stride(primals_378, (32, 896), (896, 1))
        assert_size_stride(primals_379, (128, 32), (32, 1))
        assert_size_stride(primals_380, (128, 896), (896, 1))
        assert_size_stride(primals_381, (128, ), (1, ))
        assert_size_stride(primals_382, (32, 896), (896, 1))
        assert_size_stride(primals_383, (128, 32), (32, 1))
        assert_size_stride(primals_384, (896, 896), (896, 1))
        assert_size_stride(primals_385, (32, 896), (896, 1))
        assert_size_stride(primals_386, (896, 32), (32, 1))
        assert_size_stride(primals_387, (896, ), (1, ))
        assert_size_stride(primals_388, (4864, 896), (896, 1))
        assert_size_stride(primals_389, (32, 896), (896, 1))
        assert_size_stride(primals_390, (4864, 32), (32, 1))
        assert_size_stride(primals_391, (4864, 896), (896, 1))
        assert_size_stride(primals_392, (32, 896), (896, 1))
        assert_size_stride(primals_393, (4864, 32), (32, 1))
        assert_size_stride(primals_394, (896, 4864), (4864, 1))
        assert_size_stride(primals_395, (32, 4864), (4864, 1))
        assert_size_stride(primals_396, (896, 32), (32, 1))
        assert_size_stride(primals_397, (896, ), (1, ))
        assert_size_stride(primals_398, (896, 896), (896, 1))
        assert_size_stride(primals_399, (896, ), (1, ))
        assert_size_stride(primals_400, (32, 896), (896, 1))
        assert_size_stride(primals_401, (896, 32), (32, 1))
        assert_size_stride(primals_402, (128, 896), (896, 1))
        assert_size_stride(primals_403, (128, ), (1, ))
        assert_size_stride(primals_404, (32, 896), (896, 1))
        assert_size_stride(primals_405, (128, 32), (32, 1))
        assert_size_stride(primals_406, (128, 896), (896, 1))
        assert_size_stride(primals_407, (128, ), (1, ))
        assert_size_stride(primals_408, (32, 896), (896, 1))
        assert_size_stride(primals_409, (128, 32), (32, 1))
        assert_size_stride(primals_410, (896, 896), (896, 1))
        assert_size_stride(primals_411, (32, 896), (896, 1))
        assert_size_stride(primals_412, (896, 32), (32, 1))
        assert_size_stride(primals_413, (896, ), (1, ))
        assert_size_stride(primals_414, (4864, 896), (896, 1))
        assert_size_stride(primals_415, (32, 896), (896, 1))
        assert_size_stride(primals_416, (4864, 32), (32, 1))
        assert_size_stride(primals_417, (4864, 896), (896, 1))
        assert_size_stride(primals_418, (32, 896), (896, 1))
        assert_size_stride(primals_419, (4864, 32), (32, 1))
        assert_size_stride(primals_420, (896, 4864), (4864, 1))
        assert_size_stride(primals_421, (32, 4864), (4864, 1))
        assert_size_stride(primals_422, (896, 32), (32, 1))
        assert_size_stride(primals_423, (896, ), (1, ))
        assert_size_stride(primals_424, (896, 896), (896, 1))
        assert_size_stride(primals_425, (896, ), (1, ))
        assert_size_stride(primals_426, (32, 896), (896, 1))
        assert_size_stride(primals_427, (896, 32), (32, 1))
        assert_size_stride(primals_428, (128, 896), (896, 1))
        assert_size_stride(primals_429, (128, ), (1, ))
        assert_size_stride(primals_430, (32, 896), (896, 1))
        assert_size_stride(primals_431, (128, 32), (32, 1))
        assert_size_stride(primals_432, (128, 896), (896, 1))
        assert_size_stride(primals_433, (128, ), (1, ))
        assert_size_stride(primals_434, (32, 896), (896, 1))
        assert_size_stride(primals_435, (128, 32), (32, 1))
        assert_size_stride(primals_436, (896, 896), (896, 1))
        assert_size_stride(primals_437, (32, 896), (896, 1))
        assert_size_stride(primals_438, (896, 32), (32, 1))
        assert_size_stride(primals_439, (896, ), (1, ))
        assert_size_stride(primals_440, (4864, 896), (896, 1))
        assert_size_stride(primals_441, (32, 896), (896, 1))
        assert_size_stride(primals_442, (4864, 32), (32, 1))
        assert_size_stride(primals_443, (4864, 896), (896, 1))
        assert_size_stride(primals_444, (32, 896), (896, 1))
        assert_size_stride(primals_445, (4864, 32), (32, 1))
        assert_size_stride(primals_446, (896, 4864), (4864, 1))
        assert_size_stride(primals_447, (32, 4864), (4864, 1))
        assert_size_stride(primals_448, (896, 32), (32, 1))
        assert_size_stride(primals_449, (896, ), (1, ))
        assert_size_stride(primals_450, (896, 896), (896, 1))
        assert_size_stride(primals_451, (896, ), (1, ))
        assert_size_stride(primals_452, (32, 896), (896, 1))
        assert_size_stride(primals_453, (896, 32), (32, 1))
        assert_size_stride(primals_454, (128, 896), (896, 1))
        assert_size_stride(primals_455, (128, ), (1, ))
        assert_size_stride(primals_456, (32, 896), (896, 1))
        assert_size_stride(primals_457, (128, 32), (32, 1))
        assert_size_stride(primals_458, (128, 896), (896, 1))
        assert_size_stride(primals_459, (128, ), (1, ))
        assert_size_stride(primals_460, (32, 896), (896, 1))
        assert_size_stride(primals_461, (128, 32), (32, 1))
        assert_size_stride(primals_462, (896, 896), (896, 1))
        assert_size_stride(primals_463, (32, 896), (896, 1))
        assert_size_stride(primals_464, (896, 32), (32, 1))
        assert_size_stride(primals_465, (896, ), (1, ))
        assert_size_stride(primals_466, (4864, 896), (896, 1))
        assert_size_stride(primals_467, (32, 896), (896, 1))
        assert_size_stride(primals_468, (4864, 32), (32, 1))
        assert_size_stride(primals_469, (4864, 896), (896, 1))
        assert_size_stride(primals_470, (32, 896), (896, 1))
        assert_size_stride(primals_471, (4864, 32), (32, 1))
        assert_size_stride(primals_472, (896, 4864), (4864, 1))
        assert_size_stride(primals_473, (32, 4864), (4864, 1))
        assert_size_stride(primals_474, (896, 32), (32, 1))
        assert_size_stride(primals_475, (896, ), (1, ))
        assert_size_stride(primals_476, (896, 896), (896, 1))
        assert_size_stride(primals_477, (896, ), (1, ))
        assert_size_stride(primals_478, (32, 896), (896, 1))
        assert_size_stride(primals_479, (896, 32), (32, 1))
        assert_size_stride(primals_480, (128, 896), (896, 1))
        assert_size_stride(primals_481, (128, ), (1, ))
        assert_size_stride(primals_482, (32, 896), (896, 1))
        assert_size_stride(primals_483, (128, 32), (32, 1))
        assert_size_stride(primals_484, (128, 896), (896, 1))
        assert_size_stride(primals_485, (128, ), (1, ))
        assert_size_stride(primals_486, (32, 896), (896, 1))
        assert_size_stride(primals_487, (128, 32), (32, 1))
        assert_size_stride(primals_488, (896, 896), (896, 1))
        assert_size_stride(primals_489, (32, 896), (896, 1))
        assert_size_stride(primals_490, (896, 32), (32, 1))
        assert_size_stride(primals_491, (896, ), (1, ))
        assert_size_stride(primals_492, (4864, 896), (896, 1))
        assert_size_stride(primals_493, (32, 896), (896, 1))
        assert_size_stride(primals_494, (4864, 32), (32, 1))
        assert_size_stride(primals_495, (4864, 896), (896, 1))
        assert_size_stride(primals_496, (32, 896), (896, 1))
        assert_size_stride(primals_497, (4864, 32), (32, 1))
        assert_size_stride(primals_498, (896, 4864), (4864, 1))
        assert_size_stride(primals_499, (32, 4864), (4864, 1))
        assert_size_stride(primals_500, (896, 32), (32, 1))
        assert_size_stride(primals_501, (896, ), (1, ))
        assert_size_stride(primals_502, (896, 896), (896, 1))
        assert_size_stride(primals_503, (896, ), (1, ))
        assert_size_stride(primals_504, (32, 896), (896, 1))
        assert_size_stride(primals_505, (896, 32), (32, 1))
        assert_size_stride(primals_506, (128, 896), (896, 1))
        assert_size_stride(primals_507, (128, ), (1, ))
        assert_size_stride(primals_508, (32, 896), (896, 1))
        assert_size_stride(primals_509, (128, 32), (32, 1))
        assert_size_stride(primals_510, (128, 896), (896, 1))
        assert_size_stride(primals_511, (128, ), (1, ))
        assert_size_stride(primals_512, (32, 896), (896, 1))
        assert_size_stride(primals_513, (128, 32), (32, 1))
        assert_size_stride(primals_514, (896, 896), (896, 1))
        assert_size_stride(primals_515, (32, 896), (896, 1))
        assert_size_stride(primals_516, (896, 32), (32, 1))
        assert_size_stride(primals_517, (896, ), (1, ))
        assert_size_stride(primals_518, (4864, 896), (896, 1))
        assert_size_stride(primals_519, (32, 896), (896, 1))
        assert_size_stride(primals_520, (4864, 32), (32, 1))
        assert_size_stride(primals_521, (4864, 896), (896, 1))
        assert_size_stride(primals_522, (32, 896), (896, 1))
        assert_size_stride(primals_523, (4864, 32), (32, 1))
        assert_size_stride(primals_524, (896, 4864), (4864, 1))
        assert_size_stride(primals_525, (32, 4864), (4864, 1))
        assert_size_stride(primals_526, (896, 32), (32, 1))
        assert_size_stride(primals_527, (896, ), (1, ))
        assert_size_stride(primals_528, (896, 896), (896, 1))
        assert_size_stride(primals_529, (896, ), (1, ))
        assert_size_stride(primals_530, (32, 896), (896, 1))
        assert_size_stride(primals_531, (896, 32), (32, 1))
        assert_size_stride(primals_532, (128, 896), (896, 1))
        assert_size_stride(primals_533, (128, ), (1, ))
        assert_size_stride(primals_534, (32, 896), (896, 1))
        assert_size_stride(primals_535, (128, 32), (32, 1))
        assert_size_stride(primals_536, (128, 896), (896, 1))
        assert_size_stride(primals_537, (128, ), (1, ))
        assert_size_stride(primals_538, (32, 896), (896, 1))
        assert_size_stride(primals_539, (128, 32), (32, 1))
        assert_size_stride(primals_540, (896, 896), (896, 1))
        assert_size_stride(primals_541, (32, 896), (896, 1))
        assert_size_stride(primals_542, (896, 32), (32, 1))
        assert_size_stride(primals_543, (896, ), (1, ))
        assert_size_stride(primals_544, (4864, 896), (896, 1))
        assert_size_stride(primals_545, (32, 896), (896, 1))
        assert_size_stride(primals_546, (4864, 32), (32, 1))
        assert_size_stride(primals_547, (4864, 896), (896, 1))
        assert_size_stride(primals_548, (32, 896), (896, 1))
        assert_size_stride(primals_549, (4864, 32), (32, 1))
        assert_size_stride(primals_550, (896, 4864), (4864, 1))
        assert_size_stride(primals_551, (32, 4864), (4864, 1))
        assert_size_stride(primals_552, (896, 32), (32, 1))
        assert_size_stride(primals_553, (896, ), (1, ))
        assert_size_stride(primals_554, (896, 896), (896, 1))
        assert_size_stride(primals_555, (896, ), (1, ))
        assert_size_stride(primals_556, (32, 896), (896, 1))
        assert_size_stride(primals_557, (896, 32), (32, 1))
        assert_size_stride(primals_558, (128, 896), (896, 1))
        assert_size_stride(primals_559, (128, ), (1, ))
        assert_size_stride(primals_560, (32, 896), (896, 1))
        assert_size_stride(primals_561, (128, 32), (32, 1))
        assert_size_stride(primals_562, (128, 896), (896, 1))
        assert_size_stride(primals_563, (128, ), (1, ))
        assert_size_stride(primals_564, (32, 896), (896, 1))
        assert_size_stride(primals_565, (128, 32), (32, 1))
        assert_size_stride(primals_566, (896, 896), (896, 1))
        assert_size_stride(primals_567, (32, 896), (896, 1))
        assert_size_stride(primals_568, (896, 32), (32, 1))
        assert_size_stride(primals_569, (896, ), (1, ))
        assert_size_stride(primals_570, (4864, 896), (896, 1))
        assert_size_stride(primals_571, (32, 896), (896, 1))
        assert_size_stride(primals_572, (4864, 32), (32, 1))
        assert_size_stride(primals_573, (4864, 896), (896, 1))
        assert_size_stride(primals_574, (32, 896), (896, 1))
        assert_size_stride(primals_575, (4864, 32), (32, 1))
        assert_size_stride(primals_576, (896, 4864), (4864, 1))
        assert_size_stride(primals_577, (32, 4864), (4864, 1))
        assert_size_stride(primals_578, (896, 32), (32, 1))
        assert_size_stride(primals_579, (896, ), (1, ))
        assert_size_stride(primals_580, (896, 896), (896, 1))
        assert_size_stride(primals_581, (896, ), (1, ))
        assert_size_stride(primals_582, (32, 896), (896, 1))
        assert_size_stride(primals_583, (896, 32), (32, 1))
        assert_size_stride(primals_584, (128, 896), (896, 1))
        assert_size_stride(primals_585, (128, ), (1, ))
        assert_size_stride(primals_586, (32, 896), (896, 1))
        assert_size_stride(primals_587, (128, 32), (32, 1))
        assert_size_stride(primals_588, (128, 896), (896, 1))
        assert_size_stride(primals_589, (128, ), (1, ))
        assert_size_stride(primals_590, (32, 896), (896, 1))
        assert_size_stride(primals_591, (128, 32), (32, 1))
        assert_size_stride(primals_592, (896, 896), (896, 1))
        assert_size_stride(primals_593, (32, 896), (896, 1))
        assert_size_stride(primals_594, (896, 32), (32, 1))
        assert_size_stride(primals_595, (896, ), (1, ))
        assert_size_stride(primals_596, (4864, 896), (896, 1))
        assert_size_stride(primals_597, (32, 896), (896, 1))
        assert_size_stride(primals_598, (4864, 32), (32, 1))
        assert_size_stride(primals_599, (4864, 896), (896, 1))
        assert_size_stride(primals_600, (32, 896), (896, 1))
        assert_size_stride(primals_601, (4864, 32), (32, 1))
        assert_size_stride(primals_602, (896, 4864), (4864, 1))
        assert_size_stride(primals_603, (32, 4864), (4864, 1))
        assert_size_stride(primals_604, (896, 32), (32, 1))
        assert_size_stride(primals_605, (896, ), (1, ))
        assert_size_stride(primals_606, (896, 896), (896, 1))
        assert_size_stride(primals_607, (896, ), (1, ))
        assert_size_stride(primals_608, (32, 896), (896, 1))
        assert_size_stride(primals_609, (896, 32), (32, 1))
        assert_size_stride(primals_610, (128, 896), (896, 1))
        assert_size_stride(primals_611, (128, ), (1, ))
        assert_size_stride(primals_612, (32, 896), (896, 1))
        assert_size_stride(primals_613, (128, 32), (32, 1))
        assert_size_stride(primals_614, (128, 896), (896, 1))
        assert_size_stride(primals_615, (128, ), (1, ))
        assert_size_stride(primals_616, (32, 896), (896, 1))
        assert_size_stride(primals_617, (128, 32), (32, 1))
        assert_size_stride(primals_618, (896, 896), (896, 1))
        assert_size_stride(primals_619, (32, 896), (896, 1))
        assert_size_stride(primals_620, (896, 32), (32, 1))
        assert_size_stride(primals_621, (896, ), (1, ))
        assert_size_stride(primals_622, (4864, 896), (896, 1))
        assert_size_stride(primals_623, (32, 896), (896, 1))
        assert_size_stride(primals_624, (4864, 32), (32, 1))
        assert_size_stride(primals_625, (4864, 896), (896, 1))
        assert_size_stride(primals_626, (32, 896), (896, 1))
        assert_size_stride(primals_627, (4864, 32), (32, 1))
        assert_size_stride(primals_628, (896, 4864), (4864, 1))
        assert_size_stride(primals_629, (32, 4864), (4864, 1))
        assert_size_stride(primals_630, (896, 32), (32, 1))
        assert_size_stride(primals_631, (896, ), (1, ))
        assert_size_stride(primals_633, (151936, 896), (896, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1, 1, s27), (s27, s27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_19, position_ids_expanded], Original ATen: [aten.arange, aten.unsqueeze, aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_arange_unsqueeze_0.run(buf0, s27, stream=stream0)
            buf1 = empty_strided_cuda((32, s27), (s27, 1), torch.float32)
            # Topologically Sorted Source Nodes: [cache_position, position_ids, getitem_16, expand, getitem_19, position_ids_expanded, matmul], Original ATen: [aten.arange, aten.unsqueeze, aten.expand, aten._to_copy, aten.bmm]
            extern_kernels.mm(reinterpret_tensor(primals_6, (32, 1), (1, 1), 0), reinterpret_tensor(buf0, (1, s27), (0, 1), 0), out=buf1)
            del primals_6
            buf3 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_2, rsqrt, hidden_states_1, to_7, hidden_states_2], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_red_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_1.run(primals_2, primals_3, primals_7, buf3, s27, 896, stream=stream0)
            del primals_7
            buf4 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_2, rsqrt, hidden_states_1, to_7, hidden_states_2, result_3], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf3, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_8, (896, 896), (1, 896), 0), out=buf4)
            del primals_8
            buf5 = empty_strided_cuda((32, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_10, buf5, 28672, stream=stream0)
            del primals_10
            buf6 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (s27, 896), (896, 1), 0), reinterpret_tensor(buf5, (896, 32), (1, 896), 0), out=buf6)
            buf7 = reinterpret_tensor(buf5, (32, 896), (1, 32), 0); del buf5  # reuse
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_11, buf7, 28672, stream=stream0)
            del primals_11
            buf8 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
            extern_kernels.mm(buf6, buf7, out=buf8)
            buf9 = empty_strided_cuda((s27, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_2, rsqrt, hidden_states_1, to_7, hidden_states_2, result_3, result_6], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf3, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_12, (896, 128), (1, 896), 0), out=buf9)
            del primals_12
            buf10 = empty_strided_cuda((32, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_14, buf10, 28672, stream=stream0)
            del primals_14
            buf11 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1, linear_4], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (s27, 896), (896, 1), 0), reinterpret_tensor(buf10, (896, 32), (1, 896), 0), out=buf11)
            buf12 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_15, buf12, 4096, stream=stream0)
            del primals_15
            buf13 = empty_strided_cuda((s27, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
            extern_kernels.mm(buf11, buf12, out=buf13)
            buf14 = empty_strided_cuda((s27, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, hidden_states, pow_1, variance, add_2, rsqrt, hidden_states_1, to_7, hidden_states_2, result_3, result_9], Original ATen: [aten.embedding, aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf3, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_16, (896, 128), (1, 896), 0), out=buf14)
            del primals_16
            buf15 = buf10; del buf10  # reuse
            # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_18, buf15, 28672, stream=stream0)
            del primals_18
            buf16 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_1, linear_7], Original ATen: [aten.view, aten._to_copy, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf3, (s27, 896), (896, 1), 0), reinterpret_tensor(buf15, (896, 32), (1, 896), 0), out=buf16)
            buf17 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_19, buf17, 4096, stream=stream0)
            del primals_19
            buf18 = empty_strided_cuda((s27, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
            extern_kernels.mm(buf16, buf17, out=buf18)
            ps0 = 14*s27
            buf19 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, result_3, linear_2, mul_4, result_4, view, query_states, sin_3, x1, x2, neg, cat_1, mul_8], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_9, buf4, buf8, buf1, buf19, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf20 = reinterpret_tensor(buf4, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf4  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, result_3, linear_2, mul_4, result_4, view, query_states, cos_3, mul_7, q_embed], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf20, primals_9, buf8, buf1, buf19, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_9
            ps1 = 2*s27
            buf21 = empty_strided_cuda((1, 2, s27, 64), (128*s27, 1, 2, 2*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, result_6, linear_5, mul_5, result_7, view_1, key_states, sin_3, x1_1, x2_1, neg_1, cat_2, mul_10], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_13, buf9, buf13, buf1, buf21, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf22 = reinterpret_tensor(buf9, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf9  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, result_6, linear_5, mul_5, result_7, view_1, key_states, cos_3, mul_9, k_embed], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf22, primals_13, buf13, buf1, buf21, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_13
            ps2 = 448*s27
            buf23 = reinterpret_tensor(buf8, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf8  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, result_6, linear_5, mul_5, result_7, view_1, key_states, cos_3, mul_9, k_embed, getitem_47, hidden_states_3, key], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf22, buf23, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf24 = reinterpret_tensor(buf19, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf19  # reuse
            # Topologically Sorted Source Nodes: [result_9, linear_8, mul_6, result_10, view_2, value_states, getitem_52, hidden_states_4, value], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_17, buf14, buf18, buf24, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_17
            ps3 = 8 + s27 + (-1)*(s27 % 8)
            buf25 = empty_strided_cuda((1, 1, s27, 8 + s27 + (-1)*(s27 % 8)), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), s27*max(1, 8 + s27 + (-1)*(s27 % 8)), max(1, 8 + s27 + (-1)*(s27 % 8)), 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [cache_position, attention_mask, kv_arange_1, batch_arange, le, result_1, index, result_2, batched_outputs_2, attn_output], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.view, aten.le, aten.bitwise_and, aten.index, aten.scalar_tensor, aten.where, aten.constant_pad_nd]
            triton_poi_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_index_le_scalar_tensor_view_where_10_xnumel = s27*s27 + 8*s27 + (-1)*s27*(s27 % 8)
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_index_le_scalar_tensor_view_where_10.run(primals_5, buf25, ps3, s27, s16, triton_poi_fused__to_copy_add_arange_bitwise_and_constant_pad_nd_index_le_scalar_tensor_view_where_10_xnumel, stream=stream0)
            del primals_5
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, result_6, linear_5, mul_5, result_7, view_1, key_states, result_9, linear_8, mul_6, result_10, view_2, value_states, cos_3, mul_9, k_embed, getitem_47, hidden_states_3, key, getitem_52, hidden_states_4, value, attn_output], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.unsqueeze, aten.expand, aten.clone, aten.slice, aten._scaled_dot_product_efficient_attention]
            buf26 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf20, reinterpret_tensor(buf23, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf24, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf27 = buf26[0]
            assert_size_stride(buf27, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf27, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf28 = buf26[1]
            assert_size_stride(buf28, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf28, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf29 = buf26[2]
            assert_size_stride(buf29, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf29, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf30 = buf26[3]
            assert_size_stride(buf30, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf30, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf26
            buf31 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_4, reshape_2, result_12], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf27, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_20, (896, 896), (1, 896), 0), out=buf31)
            buf32 = reinterpret_tensor(buf15, (896, 32), (1, 896), 0); del buf15  # reuse
            # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_21, buf32, 28672, stream=stream0)
            del primals_21
            buf33 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_4, reshape_2, linear_10], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf27, buf33, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf34 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_4, reshape_2, linear_10], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf33, (s27, 896), (896, 1), 0), buf32, out=buf34)
            buf35 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_22, buf35, 28672, stream=stream0)
            del primals_22
            buf36 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
            extern_kernels.mm(buf34, buf35, out=buf36)
            buf37 = reinterpret_tensor(buf31, (1, s27, 896), (896*s27, 896, 1), 0); del buf31  # reuse
            buf38 = reinterpret_tensor(buf0, (1, s27, 1), (s27, 1, s27), 0); del buf0  # reuse
            buf39 = reinterpret_tensor(buf38, (1, s27, 1), (s27, 1, 1), 0); del buf38  # reuse
            buf40 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [inputs_embeds, result_12, linear_11, mul_11, result_13, hidden_states_5, hidden_states_6, pow_2, variance_1, add_10, rsqrt_1, hidden_states_7, to_17, hidden_states_8], Original ATen: [aten.embedding, aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_embedding_mean_mul_pow_rsqrt_12.run(buf37, buf39, primals_2, primals_3, buf36, primals_23, buf40, s27, 896, stream=stream0)
            del primals_2
            del primals_3
            buf41 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_6, hidden_states_7, to_17, hidden_states_8, result_15], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_24, (896, 4864), (1, 896), 0), out=buf41)
            buf42 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_25, buf42, 28672, stream=stream0)
            del primals_25
            buf43 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (s27, 896), (896, 1), 0), buf42, out=buf43)
            buf44 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_26, buf44, 155648, stream=stream0)
            del primals_26
            buf45 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.mm]
            extern_kernels.mm(buf43, buf44, out=buf45)
            buf47 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_6, hidden_states_7, to_17, hidden_states_8, result_15, result_18], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_27, (896, 4864), (1, 896), 0), out=buf47)
            buf48 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_28, buf48, 28672, stream=stream0)
            del primals_28
            buf49 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_13, linear_16], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf40, (s27, 896), (896, 1), 0), buf48, out=buf49)
            buf50 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_29, buf50, 155648, stream=stream0)
            del primals_29
            buf51 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.mm]
            extern_kernels.mm(buf49, buf50, out=buf51)
            buf46 = reinterpret_tensor(buf41, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf41  # reuse
            buf52 = reinterpret_tensor(buf47, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf47  # reuse
            buf53 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_15, linear_14, mul_14, result_16, silu, result_18, linear_17, mul_15, result_19, mul_16], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf46, buf52, buf45, buf51, buf53, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf54 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [silu, mul_16, result_21], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf53, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_30, (4864, 896), (1, 4864), 0), out=buf54)
            buf55 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_31, buf55, 155648, stream=stream0)
            del primals_31
            buf56 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf53, (s27, 4864), (4864, 1), 0), buf55, out=buf56)
            buf57 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_32, buf57, 28672, stream=stream0)
            del primals_32
            buf58 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
            extern_kernels.mm(buf56, buf57, out=buf58)
            buf59 = reinterpret_tensor(buf54, (1, s27, 896), (896*s27, 896, 1), 0); del buf54  # reuse
            buf60 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf61 = reinterpret_tensor(buf60, (1, s27, 1), (s27, 1, 1), 0); del buf60  # reuse
            buf62 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_21, linear_20, mul_17, result_22, hidden_states_9, hidden_states_10, pow_3, variance_2, add_15, rsqrt_2, hidden_states_11, to_25, hidden_states_12], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf59, buf61, buf37, buf58, primals_33, buf62, s27, 896, stream=stream0)
            buf63 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11, to_25, hidden_states_12, result_24], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf62, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_34, (896, 896), (1, 896), 0), out=buf63)
            buf64 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_36, buf64, 28672, stream=stream0)
            del primals_36
            buf65 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf62, (s27, 896), (896, 1), 0), buf64, out=buf65)
            buf66 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_37, buf66, 28672, stream=stream0)
            del primals_37
            buf67 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.mm]
            extern_kernels.mm(buf65, buf66, out=buf67)
            buf68 = buf18; del buf18  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11, to_25, hidden_states_12, result_24, result_27], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf62, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_38, (896, 128), (1, 896), 0), out=buf68)
            buf69 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_40, buf69, 28672, stream=stream0)
            del primals_40
            buf70 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_22, linear_25], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf62, (s27, 896), (896, 1), 0), buf69, out=buf70)
            buf71 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_41, buf71, 4096, stream=stream0)
            del primals_41
            buf72 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.mm]
            extern_kernels.mm(buf70, buf71, out=buf72)
            buf73 = reinterpret_tensor(buf22, (s27, 128), (128, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11, to_25, hidden_states_12, result_24, result_30], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf62, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_42, (896, 128), (1, 896), 0), out=buf73)
            buf74 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_44, buf74, 28672, stream=stream0)
            del primals_44
            buf75 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_22, linear_28], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf62, (s27, 896), (896, 1), 0), buf74, out=buf75)
            buf76 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_45, buf76, 4096, stream=stream0)
            del primals_45
            buf77 = reinterpret_tensor(buf21, (s27, 128), (128, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.mm]
            extern_kernels.mm(buf75, buf76, out=buf77)
            buf78 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_24, linear_23, mul_20, result_25, view_3, query_states_1, x1_2, x2_2, neg_2, cat_3, mul_24], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_35, buf63, buf67, buf1, buf78, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf79 = reinterpret_tensor(buf63, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf63  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_24, linear_23, mul_20, result_25, view_3, query_states_1, mul_23, q_embed_1], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf79, primals_35, buf67, buf1, buf78, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_35
            buf80 = reinterpret_tensor(buf13, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_27, linear_26, mul_21, result_28, view_4, key_states_1, x1_3, x2_3, neg_3, cat_4, mul_26], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_39, buf68, buf72, buf1, buf80, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf81 = reinterpret_tensor(buf68, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf68  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_27, linear_26, mul_21, result_28, view_4, key_states_1, mul_25, k_embed_1], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf81, primals_39, buf72, buf1, buf80, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_39
            buf82 = reinterpret_tensor(buf78, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf78  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_27, linear_26, mul_21, result_28, view_4, key_states_1, mul_25, k_embed_1, getitem_89, hidden_states_13, key_1], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf81, buf82, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf83 = reinterpret_tensor(buf67, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf67  # reuse
            # Topologically Sorted Source Nodes: [result_30, linear_29, mul_22, result_31, view_5, value_states_1, getitem_94, hidden_states_14, value_1], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_43, buf73, buf77, buf83, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_43
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_27, linear_26, mul_21, result_28, view_4, key_states_1, result_30, linear_29, mul_22, result_31, view_5, value_states_1, mul_25, k_embed_1, getitem_89, hidden_states_13, key_1, getitem_94, hidden_states_14, value_1, attn_output_3], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf84 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf79, reinterpret_tensor(buf82, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf83, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf85 = buf84[0]
            assert_size_stride(buf85, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf85, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf86 = buf84[1]
            assert_size_stride(buf86, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf86, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf87 = buf84[2]
            assert_size_stride(buf87, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf87, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf88 = buf84[3]
            assert_size_stride(buf88, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf88, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf84
            buf89 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_8, reshape_5, result_33], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf85, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_46, (896, 896), (1, 896), 0), out=buf89)
            buf90 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_47, buf90, 28672, stream=stream0)
            del primals_47
            buf91 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_8, reshape_5, linear_31], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf85, buf91, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf92 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_8, reshape_5, linear_31], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf91, (s27, 896), (896, 1), 0), buf90, out=buf92)
            buf93 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_48, buf93, 28672, stream=stream0)
            del primals_48
            buf94 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
            extern_kernels.mm(buf92, buf93, out=buf94)
            buf95 = reinterpret_tensor(buf89, (1, s27, 896), (896*s27, 896, 1), 0); del buf89  # reuse
            buf96 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf97 = reinterpret_tensor(buf96, (1, s27, 1), (s27, 1, 1), 0); del buf96  # reuse
            buf98 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_33, linear_32, mul_27, result_34, hidden_states_15, hidden_states_16, pow_4, variance_3, add_23, rsqrt_3, hidden_states_17, to_35, hidden_states_18], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf95, buf97, buf59, buf94, primals_49, buf98, s27, 896, stream=stream0)
            buf99 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_17, to_35, hidden_states_18, result_36], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf98, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_50, (896, 4864), (1, 896), 0), out=buf99)
            buf100 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_51, buf100, 28672, stream=stream0)
            del primals_51
            buf101 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf98, (s27, 896), (896, 1), 0), buf100, out=buf101)
            buf102 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_52, buf102, 155648, stream=stream0)
            del primals_52
            buf103 = buf45; del buf45  # reuse
            # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.mm]
            extern_kernels.mm(buf101, buf102, out=buf103)
            buf105 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_16, hidden_states_17, to_35, hidden_states_18, result_36, result_39], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf98, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_53, (896, 4864), (1, 896), 0), out=buf105)
            buf106 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_54, buf106, 28672, stream=stream0)
            del primals_54
            buf107 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_34, linear_37], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf98, (s27, 896), (896, 1), 0), buf106, out=buf107)
            buf108 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_55, buf108, 155648, stream=stream0)
            del primals_55
            buf109 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.mm]
            extern_kernels.mm(buf107, buf108, out=buf109)
            buf104 = reinterpret_tensor(buf99, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf99  # reuse
            buf110 = reinterpret_tensor(buf105, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf105  # reuse
            buf111 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_36, linear_35, mul_30, result_37, silu_1, result_39, linear_38, mul_31, result_40, mul_32], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf104, buf110, buf103, buf109, buf111, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf112 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [silu_1, mul_32, result_42], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf111, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_56, (4864, 896), (1, 4864), 0), out=buf112)
            buf113 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_57, buf113, 155648, stream=stream0)
            del primals_57
            buf114 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf111, (s27, 4864), (4864, 1), 0), buf113, out=buf114)
            buf115 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_58, buf115, 28672, stream=stream0)
            del primals_58
            buf116 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.mm]
            extern_kernels.mm(buf114, buf115, out=buf116)
            buf117 = reinterpret_tensor(buf112, (1, s27, 896), (896*s27, 896, 1), 0); del buf112  # reuse
            buf118 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf119 = reinterpret_tensor(buf118, (1, s27, 1), (s27, 1, 1), 0); del buf118  # reuse
            buf120 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_42, linear_41, mul_33, result_43, hidden_states_19, hidden_states_20, pow_5, variance_4, add_28, rsqrt_4, hidden_states_21, to_43, hidden_states_22], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf117, buf119, buf95, buf116, primals_59, buf120, s27, 896, stream=stream0)
            buf121 = buf116; del buf116  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_21, to_43, hidden_states_22, result_45], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf120, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_60, (896, 896), (1, 896), 0), out=buf121)
            buf122 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_62, buf122, 28672, stream=stream0)
            del primals_62
            buf123 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf120, (s27, 896), (896, 1), 0), buf122, out=buf123)
            buf124 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_63, buf124, 28672, stream=stream0)
            del primals_63
            buf125 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.mm]
            extern_kernels.mm(buf123, buf124, out=buf125)
            buf126 = buf77; del buf77  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_21, to_43, hidden_states_22, result_45, result_48], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf120, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_64, (896, 128), (1, 896), 0), out=buf126)
            buf127 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_66, buf127, 28672, stream=stream0)
            del primals_66
            buf128 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_43, linear_46], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf120, (s27, 896), (896, 1), 0), buf127, out=buf128)
            buf129 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_67, buf129, 4096, stream=stream0)
            del primals_67
            buf130 = buf73; del buf73  # reuse
            # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.mm]
            extern_kernels.mm(buf128, buf129, out=buf130)
            buf131 = reinterpret_tensor(buf81, (s27, 128), (128, 1), 0); del buf81  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_21, to_43, hidden_states_22, result_45, result_51], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf120, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_68, (896, 128), (1, 896), 0), out=buf131)
            buf132 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_70, buf132, 28672, stream=stream0)
            del primals_70
            buf133 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_43, linear_49], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf120, (s27, 896), (896, 1), 0), buf132, out=buf133)
            buf134 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_71, buf134, 4096, stream=stream0)
            del primals_71
            buf135 = reinterpret_tensor(buf80, (s27, 128), (128, 1), 0); del buf80  # reuse
            # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.mm]
            extern_kernels.mm(buf133, buf134, out=buf135)
            buf136 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_45, linear_44, mul_36, result_46, view_6, query_states_2, x1_4, x2_4, neg_4, cat_5, mul_40], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_61, buf121, buf125, buf1, buf136, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf137 = reinterpret_tensor(buf121, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf121  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_45, linear_44, mul_36, result_46, view_6, query_states_2, mul_39, q_embed_2], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf137, primals_61, buf125, buf1, buf136, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_61
            buf138 = reinterpret_tensor(buf72, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf72  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_48, linear_47, mul_37, result_49, view_7, key_states_2, x1_5, x2_5, neg_5, cat_6, mul_42], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_65, buf126, buf130, buf1, buf138, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf139 = reinterpret_tensor(buf126, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf126  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_48, linear_47, mul_37, result_49, view_7, key_states_2, mul_41, k_embed_2], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf139, primals_65, buf130, buf1, buf138, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_65
            buf140 = reinterpret_tensor(buf136, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf136  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_48, linear_47, mul_37, result_49, view_7, key_states_2, mul_41, k_embed_2, getitem_131, hidden_states_23, key_2], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf139, buf140, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf141 = reinterpret_tensor(buf125, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf125  # reuse
            # Topologically Sorted Source Nodes: [result_51, linear_50, mul_38, result_52, view_8, value_states_2, getitem_136, hidden_states_24, value_2], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_69, buf131, buf135, buf141, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_69
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_48, linear_47, mul_37, result_49, view_7, key_states_2, result_51, linear_50, mul_38, result_52, view_8, value_states_2, mul_41, k_embed_2, getitem_131, hidden_states_23, key_2, getitem_136, hidden_states_24, value_2, attn_output_6], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf142 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf137, reinterpret_tensor(buf140, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf141, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf143 = buf142[0]
            assert_size_stride(buf143, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf143, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf144 = buf142[1]
            assert_size_stride(buf144, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf144, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf145 = buf142[2]
            assert_size_stride(buf145, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf145, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf146 = buf142[3]
            assert_size_stride(buf146, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf146, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf142
            buf147 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_12, reshape_8, result_54], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf143, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_72, (896, 896), (1, 896), 0), out=buf147)
            buf148 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_52], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_73, buf148, 28672, stream=stream0)
            del primals_73
            buf149 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_12, reshape_8, linear_52], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf143, buf149, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf150 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_12, reshape_8, linear_52], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf149, (s27, 896), (896, 1), 0), buf148, out=buf150)
            buf151 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_74, buf151, 28672, stream=stream0)
            del primals_74
            buf152 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.mm]
            extern_kernels.mm(buf150, buf151, out=buf152)
            buf153 = reinterpret_tensor(buf147, (1, s27, 896), (896*s27, 896, 1), 0); del buf147  # reuse
            buf154 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf155 = reinterpret_tensor(buf154, (1, s27, 1), (s27, 1, 1), 0); del buf154  # reuse
            buf156 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_54, linear_53, mul_43, result_55, hidden_states_25, hidden_states_26, pow_6, variance_5, add_36, rsqrt_5, hidden_states_27, to_53, hidden_states_28], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf153, buf155, buf117, buf152, primals_75, buf156, s27, 896, stream=stream0)
            buf157 = buf109; del buf109  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_26, hidden_states_27, to_53, hidden_states_28, result_57], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf156, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_76, (896, 4864), (1, 896), 0), out=buf157)
            buf158 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_77, buf158, 28672, stream=stream0)
            del primals_77
            buf159 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf156, (s27, 896), (896, 1), 0), buf158, out=buf159)
            buf160 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_78, buf160, 155648, stream=stream0)
            del primals_78
            buf161 = buf103; del buf103  # reuse
            # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.mm]
            extern_kernels.mm(buf159, buf160, out=buf161)
            buf163 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_26, hidden_states_27, to_53, hidden_states_28, result_57, result_60], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf156, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_79, (896, 4864), (1, 896), 0), out=buf163)
            buf164 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_80, buf164, 28672, stream=stream0)
            del primals_80
            buf165 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_55, linear_58], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf156, (s27, 896), (896, 1), 0), buf164, out=buf165)
            buf166 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_59], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_81, buf166, 155648, stream=stream0)
            del primals_81
            buf167 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_59], Original ATen: [aten.mm]
            extern_kernels.mm(buf165, buf166, out=buf167)
            buf162 = reinterpret_tensor(buf157, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf157  # reuse
            buf168 = reinterpret_tensor(buf163, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf163  # reuse
            buf169 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_57, linear_56, mul_46, result_58, silu_2, result_60, linear_59, mul_47, result_61, mul_48], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf162, buf168, buf161, buf167, buf169, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf170 = buf152; del buf152  # reuse
            # Topologically Sorted Source Nodes: [silu_2, mul_48, result_63], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf169, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_82, (4864, 896), (1, 4864), 0), out=buf170)
            buf171 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_83, buf171, 155648, stream=stream0)
            del primals_83
            buf172 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf169, (s27, 4864), (4864, 1), 0), buf171, out=buf172)
            buf173 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_84, buf173, 28672, stream=stream0)
            del primals_84
            buf174 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.mm]
            extern_kernels.mm(buf172, buf173, out=buf174)
            buf175 = reinterpret_tensor(buf170, (1, s27, 896), (896*s27, 896, 1), 0); del buf170  # reuse
            buf176 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf177 = reinterpret_tensor(buf176, (1, s27, 1), (s27, 1, 1), 0); del buf176  # reuse
            buf178 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_63, linear_62, mul_49, result_64, hidden_states_29, hidden_states_30, pow_7, variance_6, add_41, rsqrt_6, hidden_states_31, to_61, hidden_states_32], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf175, buf177, buf153, buf174, primals_85, buf178, s27, 896, stream=stream0)
            buf179 = buf174; del buf174  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_30, hidden_states_31, to_61, hidden_states_32, result_66], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf178, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_86, (896, 896), (1, 896), 0), out=buf179)
            buf180 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_88, buf180, 28672, stream=stream0)
            del primals_88
            buf181 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf178, (s27, 896), (896, 1), 0), buf180, out=buf181)
            buf182 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_89, buf182, 28672, stream=stream0)
            del primals_89
            buf183 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.mm]
            extern_kernels.mm(buf181, buf182, out=buf183)
            buf184 = buf135; del buf135  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_30, hidden_states_31, to_61, hidden_states_32, result_66, result_69], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf178, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_90, (896, 128), (1, 896), 0), out=buf184)
            buf185 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_67], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_92, buf185, 28672, stream=stream0)
            del primals_92
            buf186 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_64, linear_67], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf178, (s27, 896), (896, 1), 0), buf185, out=buf186)
            buf187 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_93, buf187, 4096, stream=stream0)
            del primals_93
            buf188 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.mm]
            extern_kernels.mm(buf186, buf187, out=buf188)
            buf189 = reinterpret_tensor(buf139, (s27, 128), (128, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_30, hidden_states_31, to_61, hidden_states_32, result_66, result_72], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf178, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_94, (896, 128), (1, 896), 0), out=buf189)
            buf190 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_96, buf190, 28672, stream=stream0)
            del primals_96
            buf191 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_64, linear_70], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf178, (s27, 896), (896, 1), 0), buf190, out=buf191)
            buf192 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_97, buf192, 4096, stream=stream0)
            del primals_97
            buf193 = reinterpret_tensor(buf138, (s27, 128), (128, 1), 0); del buf138  # reuse
            # Topologically Sorted Source Nodes: [linear_71], Original ATen: [aten.mm]
            extern_kernels.mm(buf191, buf192, out=buf193)
            buf194 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_66, linear_65, mul_52, result_67, view_9, query_states_3, x1_6, x2_6, neg_6, cat_7, mul_56], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_87, buf179, buf183, buf1, buf194, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf195 = reinterpret_tensor(buf179, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf179  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_66, linear_65, mul_52, result_67, view_9, query_states_3, mul_55, q_embed_3], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf195, primals_87, buf183, buf1, buf194, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_87
            buf196 = reinterpret_tensor(buf130, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_69, linear_68, mul_53, result_70, view_10, key_states_3, x1_7, x2_7, neg_7, cat_8, mul_58], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_91, buf184, buf188, buf1, buf196, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf197 = reinterpret_tensor(buf184, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf184  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_69, linear_68, mul_53, result_70, view_10, key_states_3, mul_57, k_embed_3], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf197, primals_91, buf188, buf1, buf196, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_91
            buf198 = reinterpret_tensor(buf194, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf194  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_69, linear_68, mul_53, result_70, view_10, key_states_3, mul_57, k_embed_3, getitem_173, hidden_states_33, key_3], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf197, buf198, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf199 = reinterpret_tensor(buf183, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf183  # reuse
            # Topologically Sorted Source Nodes: [result_72, linear_71, mul_54, result_73, view_11, value_states_3, getitem_178, hidden_states_34, value_3], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_95, buf189, buf193, buf199, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_95
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_69, linear_68, mul_53, result_70, view_10, key_states_3, result_72, linear_71, mul_54, result_73, view_11, value_states_3, mul_57, k_embed_3, getitem_173, hidden_states_33, key_3, getitem_178, hidden_states_34, value_3, attn_output_9], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf200 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf195, reinterpret_tensor(buf198, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf199, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf201 = buf200[0]
            assert_size_stride(buf201, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf201, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf202 = buf200[1]
            assert_size_stride(buf202, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf202, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf203 = buf200[2]
            assert_size_stride(buf203, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf203, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf204 = buf200[3]
            assert_size_stride(buf204, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf204, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf200
            buf205 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_16, reshape_11, result_75], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf201, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_98, (896, 896), (1, 896), 0), out=buf205)
            buf206 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_73], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_99, buf206, 28672, stream=stream0)
            del primals_99
            buf207 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_16, reshape_11, linear_73], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf201, buf207, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf208 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_16, reshape_11, linear_73], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf207, (s27, 896), (896, 1), 0), buf206, out=buf208)
            buf209 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_100, buf209, 28672, stream=stream0)
            del primals_100
            buf210 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten.mm]
            extern_kernels.mm(buf208, buf209, out=buf210)
            buf211 = reinterpret_tensor(buf205, (1, s27, 896), (896*s27, 896, 1), 0); del buf205  # reuse
            buf212 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf213 = reinterpret_tensor(buf212, (1, s27, 1), (s27, 1, 1), 0); del buf212  # reuse
            buf214 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_75, linear_74, mul_59, result_76, hidden_states_35, hidden_states_36, pow_8, variance_7, add_49, rsqrt_7, hidden_states_37, to_71, hidden_states_38], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf211, buf213, buf175, buf210, primals_101, buf214, s27, 896, stream=stream0)
            buf215 = buf167; del buf167  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_37, to_71, hidden_states_38, result_78], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_102, (896, 4864), (1, 896), 0), out=buf215)
            buf216 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_103, buf216, 28672, stream=stream0)
            del primals_103
            buf217 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (s27, 896), (896, 1), 0), buf216, out=buf217)
            buf218 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_104, buf218, 155648, stream=stream0)
            del primals_104
            buf219 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.mm]
            extern_kernels.mm(buf217, buf218, out=buf219)
            buf221 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_36, hidden_states_37, to_71, hidden_states_38, result_78, result_81], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_105, (896, 4864), (1, 896), 0), out=buf221)
            buf222 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_79], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_106, buf222, 28672, stream=stream0)
            del primals_106
            buf223 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_76, linear_79], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (s27, 896), (896, 1), 0), buf222, out=buf223)
            buf224 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_80], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_107, buf224, 155648, stream=stream0)
            del primals_107
            buf225 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_80], Original ATen: [aten.mm]
            extern_kernels.mm(buf223, buf224, out=buf225)
            buf220 = reinterpret_tensor(buf215, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf215  # reuse
            buf226 = reinterpret_tensor(buf221, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf221  # reuse
            buf227 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_78, linear_77, mul_62, result_79, silu_3, result_81, linear_80, mul_63, result_82, mul_64], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf220, buf226, buf219, buf225, buf227, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf228 = buf210; del buf210  # reuse
            # Topologically Sorted Source Nodes: [silu_3, mul_64, result_84], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf227, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_108, (4864, 896), (1, 4864), 0), out=buf228)
            buf229 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_109, buf229, 155648, stream=stream0)
            del primals_109
            buf230 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf227, (s27, 4864), (4864, 1), 0), buf229, out=buf230)
            buf231 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_83], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_110, buf231, 28672, stream=stream0)
            del primals_110
            buf232 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_83], Original ATen: [aten.mm]
            extern_kernels.mm(buf230, buf231, out=buf232)
            buf233 = reinterpret_tensor(buf228, (1, s27, 896), (896*s27, 896, 1), 0); del buf228  # reuse
            buf234 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf235 = reinterpret_tensor(buf234, (1, s27, 1), (s27, 1, 1), 0); del buf234  # reuse
            buf236 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_84, linear_83, mul_65, result_85, hidden_states_39, hidden_states_40, pow_9, variance_8, add_54, rsqrt_8, hidden_states_41, to_79, hidden_states_42], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf233, buf235, buf211, buf232, primals_111, buf236, s27, 896, stream=stream0)
            buf237 = buf232; del buf232  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41, to_79, hidden_states_42, result_87], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf236, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_112, (896, 896), (1, 896), 0), out=buf237)
            buf238 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_114, buf238, 28672, stream=stream0)
            del primals_114
            buf239 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf236, (s27, 896), (896, 1), 0), buf238, out=buf239)
            buf240 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_115, buf240, 28672, stream=stream0)
            del primals_115
            buf241 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.mm]
            extern_kernels.mm(buf239, buf240, out=buf241)
            buf242 = buf193; del buf193  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41, to_79, hidden_states_42, result_87, result_90], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf236, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_116, (896, 128), (1, 896), 0), out=buf242)
            buf243 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_88], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_118, buf243, 28672, stream=stream0)
            del primals_118
            buf244 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_85, linear_88], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf236, (s27, 896), (896, 1), 0), buf243, out=buf244)
            buf245 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_119, buf245, 4096, stream=stream0)
            del primals_119
            buf246 = buf189; del buf189  # reuse
            # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.mm]
            extern_kernels.mm(buf244, buf245, out=buf246)
            buf247 = reinterpret_tensor(buf197, (s27, 128), (128, 1), 0); del buf197  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_40, hidden_states_41, to_79, hidden_states_42, result_87, result_93], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf236, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_120, (896, 128), (1, 896), 0), out=buf247)
            buf248 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_122, buf248, 28672, stream=stream0)
            del primals_122
            buf249 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_85, linear_91], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf236, (s27, 896), (896, 1), 0), buf248, out=buf249)
            buf250 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_123, buf250, 4096, stream=stream0)
            del primals_123
            buf251 = reinterpret_tensor(buf196, (s27, 128), (128, 1), 0); del buf196  # reuse
            # Topologically Sorted Source Nodes: [linear_92], Original ATen: [aten.mm]
            extern_kernels.mm(buf249, buf250, out=buf251)
            buf252 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_87, linear_86, mul_68, result_88, view_12, query_states_4, x1_8, x2_8, neg_8, cat_9, mul_72], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_113, buf237, buf241, buf1, buf252, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf253 = reinterpret_tensor(buf237, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf237  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_87, linear_86, mul_68, result_88, view_12, query_states_4, mul_71, q_embed_4], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf253, primals_113, buf241, buf1, buf252, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_113
            buf254 = reinterpret_tensor(buf188, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_90, linear_89, mul_69, result_91, view_13, key_states_4, x1_9, x2_9, neg_9, cat_10, mul_74], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_117, buf242, buf246, buf1, buf254, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf255 = reinterpret_tensor(buf242, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_90, linear_89, mul_69, result_91, view_13, key_states_4, mul_73, k_embed_4], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf255, primals_117, buf246, buf1, buf254, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_117
            buf256 = reinterpret_tensor(buf252, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf252  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_90, linear_89, mul_69, result_91, view_13, key_states_4, mul_73, k_embed_4, getitem_215, hidden_states_43, key_4], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf255, buf256, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf257 = reinterpret_tensor(buf241, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf241  # reuse
            # Topologically Sorted Source Nodes: [result_93, linear_92, mul_70, result_94, view_14, value_states_4, getitem_220, hidden_states_44, value_4], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_121, buf247, buf251, buf257, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_121
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_90, linear_89, mul_69, result_91, view_13, key_states_4, result_93, linear_92, mul_70, result_94, view_14, value_states_4, mul_73, k_embed_4, getitem_215, hidden_states_43, key_4, getitem_220, hidden_states_44, value_4, attn_output_12], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf258 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf253, reinterpret_tensor(buf256, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf257, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf259 = buf258[0]
            assert_size_stride(buf259, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf259, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf260 = buf258[1]
            assert_size_stride(buf260, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf260, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf261 = buf258[2]
            assert_size_stride(buf261, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf261, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf262 = buf258[3]
            assert_size_stride(buf262, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf262, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf258
            buf263 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_20, reshape_14, result_96], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf259, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_124, (896, 896), (1, 896), 0), out=buf263)
            buf264 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_94], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_125, buf264, 28672, stream=stream0)
            del primals_125
            buf265 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_20, reshape_14, linear_94], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf259, buf265, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf266 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_20, reshape_14, linear_94], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf265, (s27, 896), (896, 1), 0), buf264, out=buf266)
            buf267 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_95], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_126, buf267, 28672, stream=stream0)
            del primals_126
            buf268 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_95], Original ATen: [aten.mm]
            extern_kernels.mm(buf266, buf267, out=buf268)
            buf269 = reinterpret_tensor(buf263, (1, s27, 896), (896*s27, 896, 1), 0); del buf263  # reuse
            buf270 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf271 = reinterpret_tensor(buf270, (1, s27, 1), (s27, 1, 1), 0); del buf270  # reuse
            buf272 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_96, linear_95, mul_75, result_97, hidden_states_45, hidden_states_46, pow_10, variance_9, add_62, rsqrt_9, hidden_states_47, to_89, hidden_states_48], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf269, buf271, buf233, buf268, primals_127, buf272, s27, 896, stream=stream0)
            buf273 = buf225; del buf225  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47, to_89, hidden_states_48, result_99], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf272, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_128, (896, 4864), (1, 896), 0), out=buf273)
            buf274 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_97], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_129, buf274, 28672, stream=stream0)
            del primals_129
            buf275 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_97], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf272, (s27, 896), (896, 1), 0), buf274, out=buf275)
            buf276 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_98], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_130, buf276, 155648, stream=stream0)
            del primals_130
            buf277 = buf219; del buf219  # reuse
            # Topologically Sorted Source Nodes: [linear_98], Original ATen: [aten.mm]
            extern_kernels.mm(buf275, buf276, out=buf277)
            buf279 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_46, hidden_states_47, to_89, hidden_states_48, result_99, result_102], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf272, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_131, (896, 4864), (1, 896), 0), out=buf279)
            buf280 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_100], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_132, buf280, 28672, stream=stream0)
            del primals_132
            buf281 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_97, linear_100], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf272, (s27, 896), (896, 1), 0), buf280, out=buf281)
            buf282 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_101], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_133, buf282, 155648, stream=stream0)
            del primals_133
            buf283 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_101], Original ATen: [aten.mm]
            extern_kernels.mm(buf281, buf282, out=buf283)
            buf278 = reinterpret_tensor(buf273, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf273  # reuse
            buf284 = reinterpret_tensor(buf279, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf279  # reuse
            buf285 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_99, linear_98, mul_78, result_100, silu_4, result_102, linear_101, mul_79, result_103, mul_80], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf278, buf284, buf277, buf283, buf285, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf286 = buf268; del buf268  # reuse
            # Topologically Sorted Source Nodes: [silu_4, mul_80, result_105], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf285, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_134, (4864, 896), (1, 4864), 0), out=buf286)
            buf287 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_135, buf287, 155648, stream=stream0)
            del primals_135
            buf288 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_103], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf285, (s27, 4864), (4864, 1), 0), buf287, out=buf288)
            buf289 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_104], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_136, buf289, 28672, stream=stream0)
            del primals_136
            buf290 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_104], Original ATen: [aten.mm]
            extern_kernels.mm(buf288, buf289, out=buf290)
            buf291 = reinterpret_tensor(buf286, (1, s27, 896), (896*s27, 896, 1), 0); del buf286  # reuse
            buf292 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf293 = reinterpret_tensor(buf292, (1, s27, 1), (s27, 1, 1), 0); del buf292  # reuse
            buf294 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_105, linear_104, mul_81, result_106, hidden_states_49, hidden_states_50, pow_11, variance_10, add_67, rsqrt_10, hidden_states_51, to_97, hidden_states_52], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf291, buf293, buf269, buf290, primals_137, buf294, s27, 896, stream=stream0)
            buf295 = buf290; del buf290  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51, to_97, hidden_states_52, result_108], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf294, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_138, (896, 896), (1, 896), 0), out=buf295)
            buf296 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_140, buf296, 28672, stream=stream0)
            del primals_140
            buf297 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_106], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf294, (s27, 896), (896, 1), 0), buf296, out=buf297)
            buf298 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_107], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_141, buf298, 28672, stream=stream0)
            del primals_141
            buf299 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_107], Original ATen: [aten.mm]
            extern_kernels.mm(buf297, buf298, out=buf299)
            buf300 = buf251; del buf251  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51, to_97, hidden_states_52, result_108, result_111], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf294, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_142, (896, 128), (1, 896), 0), out=buf300)
            buf301 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_109], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_144, buf301, 28672, stream=stream0)
            del primals_144
            buf302 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_106, linear_109], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf294, (s27, 896), (896, 1), 0), buf301, out=buf302)
            buf303 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_145, buf303, 4096, stream=stream0)
            del primals_145
            buf304 = buf247; del buf247  # reuse
            # Topologically Sorted Source Nodes: [linear_110], Original ATen: [aten.mm]
            extern_kernels.mm(buf302, buf303, out=buf304)
            buf305 = reinterpret_tensor(buf255, (s27, 128), (128, 1), 0); del buf255  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51, to_97, hidden_states_52, result_108, result_114], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf294, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_146, (896, 128), (1, 896), 0), out=buf305)
            buf306 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_112], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_148, buf306, 28672, stream=stream0)
            del primals_148
            buf307 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_106, linear_112], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf294, (s27, 896), (896, 1), 0), buf306, out=buf307)
            buf308 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_149, buf308, 4096, stream=stream0)
            del primals_149
            buf309 = reinterpret_tensor(buf254, (s27, 128), (128, 1), 0); del buf254  # reuse
            # Topologically Sorted Source Nodes: [linear_113], Original ATen: [aten.mm]
            extern_kernels.mm(buf307, buf308, out=buf309)
            buf310 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_108, linear_107, mul_84, result_109, view_15, query_states_5, x1_10, x2_10, neg_10, cat_11, mul_88], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_139, buf295, buf299, buf1, buf310, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf311 = reinterpret_tensor(buf295, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf295  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_108, linear_107, mul_84, result_109, view_15, query_states_5, mul_87, q_embed_5], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf311, primals_139, buf299, buf1, buf310, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_139
            buf312 = reinterpret_tensor(buf246, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf246  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_111, linear_110, mul_85, result_112, view_16, key_states_5, x1_11, x2_11, neg_11, cat_12, mul_90], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_143, buf300, buf304, buf1, buf312, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf313 = reinterpret_tensor(buf300, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf300  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_111, linear_110, mul_85, result_112, view_16, key_states_5, mul_89, k_embed_5], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf313, primals_143, buf304, buf1, buf312, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_143
            buf314 = reinterpret_tensor(buf310, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf310  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_111, linear_110, mul_85, result_112, view_16, key_states_5, mul_89, k_embed_5, getitem_257, hidden_states_53, key_5], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf313, buf314, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf315 = reinterpret_tensor(buf299, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf299  # reuse
            # Topologically Sorted Source Nodes: [result_114, linear_113, mul_86, result_115, view_17, value_states_5, getitem_262, hidden_states_54, value_5], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_147, buf305, buf309, buf315, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_147
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_111, linear_110, mul_85, result_112, view_16, key_states_5, result_114, linear_113, mul_86, result_115, view_17, value_states_5, mul_89, k_embed_5, getitem_257, hidden_states_53, key_5, getitem_262, hidden_states_54, value_5, attn_output_15], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf316 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf311, reinterpret_tensor(buf314, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf315, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf317 = buf316[0]
            assert_size_stride(buf317, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf317, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf318 = buf316[1]
            assert_size_stride(buf318, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf318, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf319 = buf316[2]
            assert_size_stride(buf319, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf319, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf320 = buf316[3]
            assert_size_stride(buf320, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf320, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf316
            buf321 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_24, reshape_17, result_117], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf317, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_150, (896, 896), (1, 896), 0), out=buf321)
            buf322 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_115], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_151, buf322, 28672, stream=stream0)
            del primals_151
            buf323 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_24, reshape_17, linear_115], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf317, buf323, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf324 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_24, reshape_17, linear_115], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf323, (s27, 896), (896, 1), 0), buf322, out=buf324)
            buf325 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_152, buf325, 28672, stream=stream0)
            del primals_152
            buf326 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_116], Original ATen: [aten.mm]
            extern_kernels.mm(buf324, buf325, out=buf326)
            buf327 = reinterpret_tensor(buf321, (1, s27, 896), (896*s27, 896, 1), 0); del buf321  # reuse
            buf328 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf329 = reinterpret_tensor(buf328, (1, s27, 1), (s27, 1, 1), 0); del buf328  # reuse
            buf330 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_117, linear_116, mul_91, result_118, hidden_states_55, hidden_states_56, pow_12, variance_11, add_75, rsqrt_11, hidden_states_57, to_107, hidden_states_58], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf327, buf329, buf291, buf326, primals_153, buf330, s27, 896, stream=stream0)
            buf331 = buf283; del buf283  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_56, hidden_states_57, to_107, hidden_states_58, result_120], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf330, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_154, (896, 4864), (1, 896), 0), out=buf331)
            buf332 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_118], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_155, buf332, 28672, stream=stream0)
            del primals_155
            buf333 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_118], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf330, (s27, 896), (896, 1), 0), buf332, out=buf333)
            buf334 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_156, buf334, 155648, stream=stream0)
            del primals_156
            buf335 = buf277; del buf277  # reuse
            # Topologically Sorted Source Nodes: [linear_119], Original ATen: [aten.mm]
            extern_kernels.mm(buf333, buf334, out=buf335)
            buf337 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_56, hidden_states_57, to_107, hidden_states_58, result_120, result_123], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf330, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_157, (896, 4864), (1, 896), 0), out=buf337)
            buf338 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_121], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_158, buf338, 28672, stream=stream0)
            del primals_158
            buf339 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_118, linear_121], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf330, (s27, 896), (896, 1), 0), buf338, out=buf339)
            buf340 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_122], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_159, buf340, 155648, stream=stream0)
            del primals_159
            buf341 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_122], Original ATen: [aten.mm]
            extern_kernels.mm(buf339, buf340, out=buf341)
            buf336 = reinterpret_tensor(buf331, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf331  # reuse
            buf342 = reinterpret_tensor(buf337, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf337  # reuse
            buf343 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_120, linear_119, mul_94, result_121, silu_5, result_123, linear_122, mul_95, result_124, mul_96], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf336, buf342, buf335, buf341, buf343, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf344 = buf326; del buf326  # reuse
            # Topologically Sorted Source Nodes: [silu_5, mul_96, result_126], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf343, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_160, (4864, 896), (1, 4864), 0), out=buf344)
            buf345 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_161, buf345, 155648, stream=stream0)
            del primals_161
            buf346 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_124], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf343, (s27, 4864), (4864, 1), 0), buf345, out=buf346)
            buf347 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_125], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_162, buf347, 28672, stream=stream0)
            del primals_162
            buf348 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_125], Original ATen: [aten.mm]
            extern_kernels.mm(buf346, buf347, out=buf348)
            buf349 = reinterpret_tensor(buf344, (1, s27, 896), (896*s27, 896, 1), 0); del buf344  # reuse
            buf350 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf351 = reinterpret_tensor(buf350, (1, s27, 1), (s27, 1, 1), 0); del buf350  # reuse
            buf352 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_126, linear_125, mul_97, result_127, hidden_states_59, hidden_states_60, pow_13, variance_12, add_80, rsqrt_12, hidden_states_61, to_115, hidden_states_62], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf349, buf351, buf327, buf348, primals_163, buf352, s27, 896, stream=stream0)
            buf353 = buf348; del buf348  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_60, hidden_states_61, to_115, hidden_states_62, result_129], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf352, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_164, (896, 896), (1, 896), 0), out=buf353)
            buf354 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_166, buf354, 28672, stream=stream0)
            del primals_166
            buf355 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_127], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf352, (s27, 896), (896, 1), 0), buf354, out=buf355)
            buf356 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_167, buf356, 28672, stream=stream0)
            del primals_167
            buf357 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_128], Original ATen: [aten.mm]
            extern_kernels.mm(buf355, buf356, out=buf357)
            buf358 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_60, hidden_states_61, to_115, hidden_states_62, result_129, result_132], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf352, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_168, (896, 128), (1, 896), 0), out=buf358)
            buf359 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_130], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_170, buf359, 28672, stream=stream0)
            del primals_170
            buf360 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_127, linear_130], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf352, (s27, 896), (896, 1), 0), buf359, out=buf360)
            buf361 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_131], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_171, buf361, 4096, stream=stream0)
            del primals_171
            buf362 = buf305; del buf305  # reuse
            # Topologically Sorted Source Nodes: [linear_131], Original ATen: [aten.mm]
            extern_kernels.mm(buf360, buf361, out=buf362)
            buf363 = reinterpret_tensor(buf313, (s27, 128), (128, 1), 0); del buf313  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_60, hidden_states_61, to_115, hidden_states_62, result_129, result_135], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf352, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_172, (896, 128), (1, 896), 0), out=buf363)
            buf364 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_133], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_174, buf364, 28672, stream=stream0)
            del primals_174
            buf365 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_127, linear_133], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf352, (s27, 896), (896, 1), 0), buf364, out=buf365)
            buf366 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_175, buf366, 4096, stream=stream0)
            del primals_175
            buf367 = reinterpret_tensor(buf312, (s27, 128), (128, 1), 0); del buf312  # reuse
            # Topologically Sorted Source Nodes: [linear_134], Original ATen: [aten.mm]
            extern_kernels.mm(buf365, buf366, out=buf367)
            buf368 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_129, linear_128, mul_100, result_130, view_18, query_states_6, x1_12, x2_12, neg_12, cat_13, mul_104], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_165, buf353, buf357, buf1, buf368, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf369 = reinterpret_tensor(buf353, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf353  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_129, linear_128, mul_100, result_130, view_18, query_states_6, mul_103, q_embed_6], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf369, primals_165, buf357, buf1, buf368, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_165
            buf370 = reinterpret_tensor(buf304, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf304  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_132, linear_131, mul_101, result_133, view_19, key_states_6, x1_13, x2_13, neg_13, cat_14, mul_106], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_169, buf358, buf362, buf1, buf370, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf371 = reinterpret_tensor(buf358, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf358  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_132, linear_131, mul_101, result_133, view_19, key_states_6, mul_105, k_embed_6], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf371, primals_169, buf362, buf1, buf370, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_169
            buf372 = reinterpret_tensor(buf368, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf368  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_132, linear_131, mul_101, result_133, view_19, key_states_6, mul_105, k_embed_6, getitem_299, hidden_states_63, key_6], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf371, buf372, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf373 = reinterpret_tensor(buf357, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf357  # reuse
            # Topologically Sorted Source Nodes: [result_135, linear_134, mul_102, result_136, view_20, value_states_6, getitem_304, hidden_states_64, value_6], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_173, buf363, buf367, buf373, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_173
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_132, linear_131, mul_101, result_133, view_19, key_states_6, result_135, linear_134, mul_102, result_136, view_20, value_states_6, mul_105, k_embed_6, getitem_299, hidden_states_63, key_6, getitem_304, hidden_states_64, value_6, attn_output_18], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf374 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf369, reinterpret_tensor(buf372, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf373, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf375 = buf374[0]
            assert_size_stride(buf375, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf375, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf376 = buf374[1]
            assert_size_stride(buf376, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf376, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf377 = buf374[2]
            assert_size_stride(buf377, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf377, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf378 = buf374[3]
            assert_size_stride(buf378, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf378, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf374
            buf379 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_28, reshape_20, result_138], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf375, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_176, (896, 896), (1, 896), 0), out=buf379)
            buf380 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_136], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_177, buf380, 28672, stream=stream0)
            del primals_177
            buf381 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_28, reshape_20, linear_136], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf375, buf381, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf382 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_28, reshape_20, linear_136], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf381, (s27, 896), (896, 1), 0), buf380, out=buf382)
            buf383 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_137], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_178, buf383, 28672, stream=stream0)
            del primals_178
            buf384 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_137], Original ATen: [aten.mm]
            extern_kernels.mm(buf382, buf383, out=buf384)
            buf385 = reinterpret_tensor(buf379, (1, s27, 896), (896*s27, 896, 1), 0); del buf379  # reuse
            buf386 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf387 = reinterpret_tensor(buf386, (1, s27, 1), (s27, 1, 1), 0); del buf386  # reuse
            buf388 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_138, linear_137, mul_107, result_139, hidden_states_65, hidden_states_66, pow_14, variance_13, add_88, rsqrt_13, hidden_states_67, to_125, hidden_states_68], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf385, buf387, buf349, buf384, primals_179, buf388, s27, 896, stream=stream0)
            buf389 = buf341; del buf341  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67, to_125, hidden_states_68, result_141], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf388, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_180, (896, 4864), (1, 896), 0), out=buf389)
            buf390 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_139], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_181, buf390, 28672, stream=stream0)
            del primals_181
            buf391 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_139], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf388, (s27, 896), (896, 1), 0), buf390, out=buf391)
            buf392 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_140], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_182, buf392, 155648, stream=stream0)
            del primals_182
            buf393 = buf335; del buf335  # reuse
            # Topologically Sorted Source Nodes: [linear_140], Original ATen: [aten.mm]
            extern_kernels.mm(buf391, buf392, out=buf393)
            buf395 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67, to_125, hidden_states_68, result_141, result_144], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf388, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_183, (896, 4864), (1, 896), 0), out=buf395)
            buf396 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_142], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_184, buf396, 28672, stream=stream0)
            del primals_184
            buf397 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_139, linear_142], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf388, (s27, 896), (896, 1), 0), buf396, out=buf397)
            buf398 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_143], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_185, buf398, 155648, stream=stream0)
            del primals_185
            buf399 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_143], Original ATen: [aten.mm]
            extern_kernels.mm(buf397, buf398, out=buf399)
            buf394 = reinterpret_tensor(buf389, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf389  # reuse
            buf400 = reinterpret_tensor(buf395, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf395  # reuse
            buf401 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_141, linear_140, mul_110, result_142, silu_6, result_144, linear_143, mul_111, result_145, mul_112], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf394, buf400, buf393, buf399, buf401, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf402 = buf384; del buf384  # reuse
            # Topologically Sorted Source Nodes: [silu_6, mul_112, result_147], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf401, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_186, (4864, 896), (1, 4864), 0), out=buf402)
            buf403 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_145], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_187, buf403, 155648, stream=stream0)
            del primals_187
            buf404 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_145], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf401, (s27, 4864), (4864, 1), 0), buf403, out=buf404)
            buf405 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_146], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_188, buf405, 28672, stream=stream0)
            del primals_188
            buf406 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_146], Original ATen: [aten.mm]
            extern_kernels.mm(buf404, buf405, out=buf406)
            buf407 = reinterpret_tensor(buf402, (1, s27, 896), (896*s27, 896, 1), 0); del buf402  # reuse
            buf408 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf409 = reinterpret_tensor(buf408, (1, s27, 1), (s27, 1, 1), 0); del buf408  # reuse
            buf410 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_147, linear_146, mul_113, result_148, hidden_states_69, hidden_states_70, pow_15, variance_14, add_93, rsqrt_14, hidden_states_71, to_133, hidden_states_72], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf407, buf409, buf385, buf406, primals_189, buf410, s27, 896, stream=stream0)
            buf411 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_70, hidden_states_71, to_133, hidden_states_72, result_150], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf410, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_190, (896, 896), (1, 896), 0), out=buf411)
            buf412 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_148], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_192, buf412, 28672, stream=stream0)
            del primals_192
            buf413 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_148], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf410, (s27, 896), (896, 1), 0), buf412, out=buf413)
            buf414 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_149], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_193, buf414, 28672, stream=stream0)
            del primals_193
            buf415 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_149], Original ATen: [aten.mm]
            extern_kernels.mm(buf413, buf414, out=buf415)
            buf416 = buf367; del buf367  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_70, hidden_states_71, to_133, hidden_states_72, result_150, result_153], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf410, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_194, (896, 128), (1, 896), 0), out=buf416)
            buf417 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_151], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_196, buf417, 28672, stream=stream0)
            del primals_196
            buf418 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_148, linear_151], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf410, (s27, 896), (896, 1), 0), buf417, out=buf418)
            buf419 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_197, buf419, 4096, stream=stream0)
            del primals_197
            buf420 = buf363; del buf363  # reuse
            # Topologically Sorted Source Nodes: [linear_152], Original ATen: [aten.mm]
            extern_kernels.mm(buf418, buf419, out=buf420)
            buf421 = reinterpret_tensor(buf371, (s27, 128), (128, 1), 0); del buf371  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_70, hidden_states_71, to_133, hidden_states_72, result_150, result_156], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf410, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_198, (896, 128), (1, 896), 0), out=buf421)
            buf422 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_154], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_200, buf422, 28672, stream=stream0)
            del primals_200
            buf423 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_148, linear_154], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf410, (s27, 896), (896, 1), 0), buf422, out=buf423)
            buf424 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_155], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_201, buf424, 4096, stream=stream0)
            del primals_201
            buf425 = reinterpret_tensor(buf370, (s27, 128), (128, 1), 0); del buf370  # reuse
            # Topologically Sorted Source Nodes: [linear_155], Original ATen: [aten.mm]
            extern_kernels.mm(buf423, buf424, out=buf425)
            buf426 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_150, linear_149, mul_116, result_151, view_21, query_states_7, x1_14, x2_14, neg_14, cat_15, mul_120], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_191, buf411, buf415, buf1, buf426, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf427 = reinterpret_tensor(buf411, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf411  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_150, linear_149, mul_116, result_151, view_21, query_states_7, mul_119, q_embed_7], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf427, primals_191, buf415, buf1, buf426, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_191
            buf428 = reinterpret_tensor(buf362, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_153, linear_152, mul_117, result_154, view_22, key_states_7, x1_15, x2_15, neg_15, cat_16, mul_122], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_195, buf416, buf420, buf1, buf428, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf429 = reinterpret_tensor(buf416, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf416  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_153, linear_152, mul_117, result_154, view_22, key_states_7, mul_121, k_embed_7], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf429, primals_195, buf420, buf1, buf428, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_195
            buf430 = reinterpret_tensor(buf426, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf426  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_153, linear_152, mul_117, result_154, view_22, key_states_7, mul_121, k_embed_7, getitem_341, hidden_states_73, key_7], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf429, buf430, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf431 = reinterpret_tensor(buf415, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf415  # reuse
            # Topologically Sorted Source Nodes: [result_156, linear_155, mul_118, result_157, view_23, value_states_7, getitem_346, hidden_states_74, value_7], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_199, buf421, buf425, buf431, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_199
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_153, linear_152, mul_117, result_154, view_22, key_states_7, result_156, linear_155, mul_118, result_157, view_23, value_states_7, mul_121, k_embed_7, getitem_341, hidden_states_73, key_7, getitem_346, hidden_states_74, value_7, attn_output_21], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf432 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf427, reinterpret_tensor(buf430, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf431, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf433 = buf432[0]
            assert_size_stride(buf433, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf433, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf434 = buf432[1]
            assert_size_stride(buf434, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf434, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf435 = buf432[2]
            assert_size_stride(buf435, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf435, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf436 = buf432[3]
            assert_size_stride(buf436, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf436, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf432
            buf437 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_32, reshape_23, result_159], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf433, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_202, (896, 896), (1, 896), 0), out=buf437)
            buf438 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_157], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_203, buf438, 28672, stream=stream0)
            del primals_203
            buf439 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_32, reshape_23, linear_157], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf433, buf439, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf440 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_32, reshape_23, linear_157], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf439, (s27, 896), (896, 1), 0), buf438, out=buf440)
            buf441 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_158], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_204, buf441, 28672, stream=stream0)
            del primals_204
            buf442 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_158], Original ATen: [aten.mm]
            extern_kernels.mm(buf440, buf441, out=buf442)
            buf443 = reinterpret_tensor(buf437, (1, s27, 896), (896*s27, 896, 1), 0); del buf437  # reuse
            buf444 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf445 = reinterpret_tensor(buf444, (1, s27, 1), (s27, 1, 1), 0); del buf444  # reuse
            buf446 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_159, linear_158, mul_123, result_160, hidden_states_75, hidden_states_76, pow_16, variance_15, add_101, rsqrt_15, hidden_states_77, to_143, hidden_states_78], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf443, buf445, buf407, buf442, primals_205, buf446, s27, 896, stream=stream0)
            buf447 = buf399; del buf399  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_77, to_143, hidden_states_78, result_162], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf446, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_206, (896, 4864), (1, 896), 0), out=buf447)
            buf448 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_160], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_207, buf448, 28672, stream=stream0)
            del primals_207
            buf449 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_160], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf446, (s27, 896), (896, 1), 0), buf448, out=buf449)
            buf450 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_161], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_208, buf450, 155648, stream=stream0)
            del primals_208
            buf451 = buf393; del buf393  # reuse
            # Topologically Sorted Source Nodes: [linear_161], Original ATen: [aten.mm]
            extern_kernels.mm(buf449, buf450, out=buf451)
            buf453 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_76, hidden_states_77, to_143, hidden_states_78, result_162, result_165], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf446, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_209, (896, 4864), (1, 896), 0), out=buf453)
            buf454 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_163], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_210, buf454, 28672, stream=stream0)
            del primals_210
            buf455 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_160, linear_163], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf446, (s27, 896), (896, 1), 0), buf454, out=buf455)
            buf456 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_164], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_211, buf456, 155648, stream=stream0)
            del primals_211
            buf457 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_164], Original ATen: [aten.mm]
            extern_kernels.mm(buf455, buf456, out=buf457)
            buf452 = reinterpret_tensor(buf447, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf447  # reuse
            buf458 = reinterpret_tensor(buf453, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf453  # reuse
            buf459 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_162, linear_161, mul_126, result_163, silu_7, result_165, linear_164, mul_127, result_166, mul_128], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf452, buf458, buf451, buf457, buf459, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf460 = buf442; del buf442  # reuse
            # Topologically Sorted Source Nodes: [silu_7, mul_128, result_168], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf459, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_212, (4864, 896), (1, 4864), 0), out=buf460)
            buf461 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_166], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_213, buf461, 155648, stream=stream0)
            del primals_213
            buf462 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_166], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf459, (s27, 4864), (4864, 1), 0), buf461, out=buf462)
            buf463 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_167], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_214, buf463, 28672, stream=stream0)
            del primals_214
            buf464 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_167], Original ATen: [aten.mm]
            extern_kernels.mm(buf462, buf463, out=buf464)
            buf465 = reinterpret_tensor(buf460, (1, s27, 896), (896*s27, 896, 1), 0); del buf460  # reuse
            buf466 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf467 = reinterpret_tensor(buf466, (1, s27, 1), (s27, 1, 1), 0); del buf466  # reuse
            buf468 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_168, linear_167, mul_129, result_169, hidden_states_79, hidden_states_80, pow_17, variance_16, add_106, rsqrt_16, hidden_states_81, to_151, hidden_states_82], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf465, buf467, buf443, buf464, primals_215, buf468, s27, 896, stream=stream0)
            buf469 = buf464; del buf464  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81, to_151, hidden_states_82, result_171], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf468, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_216, (896, 896), (1, 896), 0), out=buf469)
            buf470 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_169], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_218, buf470, 28672, stream=stream0)
            del primals_218
            buf471 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_169], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf468, (s27, 896), (896, 1), 0), buf470, out=buf471)
            buf472 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_170], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_219, buf472, 28672, stream=stream0)
            del primals_219
            buf473 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_170], Original ATen: [aten.mm]
            extern_kernels.mm(buf471, buf472, out=buf473)
            buf474 = buf425; del buf425  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81, to_151, hidden_states_82, result_171, result_174], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf468, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_220, (896, 128), (1, 896), 0), out=buf474)
            buf475 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_172], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_222, buf475, 28672, stream=stream0)
            del primals_222
            buf476 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_169, linear_172], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf468, (s27, 896), (896, 1), 0), buf475, out=buf476)
            buf477 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_173], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_223, buf477, 4096, stream=stream0)
            del primals_223
            buf478 = buf421; del buf421  # reuse
            # Topologically Sorted Source Nodes: [linear_173], Original ATen: [aten.mm]
            extern_kernels.mm(buf476, buf477, out=buf478)
            buf479 = reinterpret_tensor(buf429, (s27, 128), (128, 1), 0); del buf429  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_80, hidden_states_81, to_151, hidden_states_82, result_171, result_177], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf468, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_224, (896, 128), (1, 896), 0), out=buf479)
            buf480 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_175], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_226, buf480, 28672, stream=stream0)
            del primals_226
            buf481 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_169, linear_175], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf468, (s27, 896), (896, 1), 0), buf480, out=buf481)
            buf482 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_176], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_227, buf482, 4096, stream=stream0)
            del primals_227
            buf483 = reinterpret_tensor(buf428, (s27, 128), (128, 1), 0); del buf428  # reuse
            # Topologically Sorted Source Nodes: [linear_176], Original ATen: [aten.mm]
            extern_kernels.mm(buf481, buf482, out=buf483)
            buf484 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_171, linear_170, mul_132, result_172, view_24, query_states_8, x1_16, x2_16, neg_16, cat_17, mul_136], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_217, buf469, buf473, buf1, buf484, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf485 = reinterpret_tensor(buf469, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf469  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_171, linear_170, mul_132, result_172, view_24, query_states_8, mul_135, q_embed_8], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf485, primals_217, buf473, buf1, buf484, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_217
            buf486 = reinterpret_tensor(buf420, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf420  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_174, linear_173, mul_133, result_175, view_25, key_states_8, x1_17, x2_17, neg_17, cat_18, mul_138], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_221, buf474, buf478, buf1, buf486, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf487 = reinterpret_tensor(buf474, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf474  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_174, linear_173, mul_133, result_175, view_25, key_states_8, mul_137, k_embed_8], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf487, primals_221, buf478, buf1, buf486, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_221
            buf488 = reinterpret_tensor(buf484, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf484  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_174, linear_173, mul_133, result_175, view_25, key_states_8, mul_137, k_embed_8, getitem_383, hidden_states_83, key_8], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf487, buf488, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf489 = reinterpret_tensor(buf473, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf473  # reuse
            # Topologically Sorted Source Nodes: [result_177, linear_176, mul_134, result_178, view_26, value_states_8, getitem_388, hidden_states_84, value_8], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_225, buf479, buf483, buf489, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_225
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_174, linear_173, mul_133, result_175, view_25, key_states_8, result_177, linear_176, mul_134, result_178, view_26, value_states_8, mul_137, k_embed_8, getitem_383, hidden_states_83, key_8, getitem_388, hidden_states_84, value_8, attn_output_24], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf490 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf485, reinterpret_tensor(buf488, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf489, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf491 = buf490[0]
            assert_size_stride(buf491, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf491, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf492 = buf490[1]
            assert_size_stride(buf492, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf492, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf493 = buf490[2]
            assert_size_stride(buf493, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf493, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf494 = buf490[3]
            assert_size_stride(buf494, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf494, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf490
            buf495 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_36, reshape_26, result_180], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf491, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_228, (896, 896), (1, 896), 0), out=buf495)
            buf496 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_178], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_229, buf496, 28672, stream=stream0)
            del primals_229
            buf497 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_36, reshape_26, linear_178], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf491, buf497, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf498 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_36, reshape_26, linear_178], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf497, (s27, 896), (896, 1), 0), buf496, out=buf498)
            buf499 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_179], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_230, buf499, 28672, stream=stream0)
            del primals_230
            buf500 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_179], Original ATen: [aten.mm]
            extern_kernels.mm(buf498, buf499, out=buf500)
            buf501 = reinterpret_tensor(buf495, (1, s27, 896), (896*s27, 896, 1), 0); del buf495  # reuse
            buf502 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf503 = reinterpret_tensor(buf502, (1, s27, 1), (s27, 1, 1), 0); del buf502  # reuse
            buf504 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_180, linear_179, mul_139, result_181, hidden_states_85, hidden_states_86, pow_18, variance_17, add_114, rsqrt_17, hidden_states_87, to_161, hidden_states_88], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf501, buf503, buf465, buf500, primals_231, buf504, s27, 896, stream=stream0)
            buf505 = buf457; del buf457  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_86, hidden_states_87, to_161, hidden_states_88, result_183], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf504, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_232, (896, 4864), (1, 896), 0), out=buf505)
            buf506 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_181], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_233, buf506, 28672, stream=stream0)
            del primals_233
            buf507 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_181], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf504, (s27, 896), (896, 1), 0), buf506, out=buf507)
            buf508 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_182], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_234, buf508, 155648, stream=stream0)
            del primals_234
            buf509 = buf451; del buf451  # reuse
            # Topologically Sorted Source Nodes: [linear_182], Original ATen: [aten.mm]
            extern_kernels.mm(buf507, buf508, out=buf509)
            buf511 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_86, hidden_states_87, to_161, hidden_states_88, result_183, result_186], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf504, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_235, (896, 4864), (1, 896), 0), out=buf511)
            buf512 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_184], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_236, buf512, 28672, stream=stream0)
            del primals_236
            buf513 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_181, linear_184], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf504, (s27, 896), (896, 1), 0), buf512, out=buf513)
            buf514 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_185], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_237, buf514, 155648, stream=stream0)
            del primals_237
            buf515 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_185], Original ATen: [aten.mm]
            extern_kernels.mm(buf513, buf514, out=buf515)
            buf510 = reinterpret_tensor(buf505, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf505  # reuse
            buf516 = reinterpret_tensor(buf511, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf511  # reuse
            buf517 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_183, linear_182, mul_142, result_184, silu_8, result_186, linear_185, mul_143, result_187, mul_144], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf510, buf516, buf509, buf515, buf517, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf518 = buf500; del buf500  # reuse
            # Topologically Sorted Source Nodes: [silu_8, mul_144, result_189], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf517, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_238, (4864, 896), (1, 4864), 0), out=buf518)
            buf519 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_187], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_239, buf519, 155648, stream=stream0)
            del primals_239
            buf520 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_187], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf517, (s27, 4864), (4864, 1), 0), buf519, out=buf520)
            buf521 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_188], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_240, buf521, 28672, stream=stream0)
            del primals_240
            buf522 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_188], Original ATen: [aten.mm]
            extern_kernels.mm(buf520, buf521, out=buf522)
            buf523 = reinterpret_tensor(buf518, (1, s27, 896), (896*s27, 896, 1), 0); del buf518  # reuse
            buf524 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf525 = reinterpret_tensor(buf524, (1, s27, 1), (s27, 1, 1), 0); del buf524  # reuse
            buf526 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_189, linear_188, mul_145, result_190, hidden_states_89, hidden_states_90, pow_19, variance_18, add_119, rsqrt_18, hidden_states_91, to_169, hidden_states_92], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf523, buf525, buf501, buf522, primals_241, buf526, s27, 896, stream=stream0)
            buf527 = buf522; del buf522  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_90, hidden_states_91, to_169, hidden_states_92, result_192], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf526, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_242, (896, 896), (1, 896), 0), out=buf527)
            buf528 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_190], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_244, buf528, 28672, stream=stream0)
            del primals_244
            buf529 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_190], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf526, (s27, 896), (896, 1), 0), buf528, out=buf529)
            buf530 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_191], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_245, buf530, 28672, stream=stream0)
            del primals_245
            buf531 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_191], Original ATen: [aten.mm]
            extern_kernels.mm(buf529, buf530, out=buf531)
            buf532 = buf483; del buf483  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_90, hidden_states_91, to_169, hidden_states_92, result_192, result_195], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf526, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_246, (896, 128), (1, 896), 0), out=buf532)
            buf533 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_193], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_248, buf533, 28672, stream=stream0)
            del primals_248
            buf534 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_190, linear_193], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf526, (s27, 896), (896, 1), 0), buf533, out=buf534)
            buf535 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_194], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_249, buf535, 4096, stream=stream0)
            del primals_249
            buf536 = buf479; del buf479  # reuse
            # Topologically Sorted Source Nodes: [linear_194], Original ATen: [aten.mm]
            extern_kernels.mm(buf534, buf535, out=buf536)
            buf537 = reinterpret_tensor(buf487, (s27, 128), (128, 1), 0); del buf487  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_90, hidden_states_91, to_169, hidden_states_92, result_192, result_198], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf526, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_250, (896, 128), (1, 896), 0), out=buf537)
            buf538 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_196], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_252, buf538, 28672, stream=stream0)
            del primals_252
            buf539 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_190, linear_196], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf526, (s27, 896), (896, 1), 0), buf538, out=buf539)
            buf540 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_197], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_253, buf540, 4096, stream=stream0)
            del primals_253
            buf541 = reinterpret_tensor(buf486, (s27, 128), (128, 1), 0); del buf486  # reuse
            # Topologically Sorted Source Nodes: [linear_197], Original ATen: [aten.mm]
            extern_kernels.mm(buf539, buf540, out=buf541)
            buf542 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_192, linear_191, mul_148, result_193, view_27, query_states_9, x1_18, x2_18, neg_18, cat_19, mul_152], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_243, buf527, buf531, buf1, buf542, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf543 = reinterpret_tensor(buf527, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf527  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_192, linear_191, mul_148, result_193, view_27, query_states_9, mul_151, q_embed_9], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf543, primals_243, buf531, buf1, buf542, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_243
            buf544 = reinterpret_tensor(buf478, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf478  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_195, linear_194, mul_149, result_196, view_28, key_states_9, x1_19, x2_19, neg_19, cat_20, mul_154], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_247, buf532, buf536, buf1, buf544, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf545 = reinterpret_tensor(buf532, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf532  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_195, linear_194, mul_149, result_196, view_28, key_states_9, mul_153, k_embed_9], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf545, primals_247, buf536, buf1, buf544, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_247
            buf546 = reinterpret_tensor(buf542, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf542  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_195, linear_194, mul_149, result_196, view_28, key_states_9, mul_153, k_embed_9, getitem_425, hidden_states_93, key_9], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf545, buf546, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf547 = reinterpret_tensor(buf531, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf531  # reuse
            # Topologically Sorted Source Nodes: [result_198, linear_197, mul_150, result_199, view_29, value_states_9, getitem_430, hidden_states_94, value_9], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_251, buf537, buf541, buf547, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_251
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_195, linear_194, mul_149, result_196, view_28, key_states_9, result_198, linear_197, mul_150, result_199, view_29, value_states_9, mul_153, k_embed_9, getitem_425, hidden_states_93, key_9, getitem_430, hidden_states_94, value_9, attn_output_27], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf548 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf543, reinterpret_tensor(buf546, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf547, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf549 = buf548[0]
            assert_size_stride(buf549, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf549, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf550 = buf548[1]
            assert_size_stride(buf550, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf550, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf551 = buf548[2]
            assert_size_stride(buf551, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf551, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf552 = buf548[3]
            assert_size_stride(buf552, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf552, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf548
            buf553 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_40, reshape_29, result_201], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf549, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_254, (896, 896), (1, 896), 0), out=buf553)
            buf554 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_199], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_255, buf554, 28672, stream=stream0)
            del primals_255
            buf555 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_40, reshape_29, linear_199], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf549, buf555, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf556 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_40, reshape_29, linear_199], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf555, (s27, 896), (896, 1), 0), buf554, out=buf556)
            buf557 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_200], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_256, buf557, 28672, stream=stream0)
            del primals_256
            buf558 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_200], Original ATen: [aten.mm]
            extern_kernels.mm(buf556, buf557, out=buf558)
            buf559 = reinterpret_tensor(buf553, (1, s27, 896), (896*s27, 896, 1), 0); del buf553  # reuse
            buf560 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf561 = reinterpret_tensor(buf560, (1, s27, 1), (s27, 1, 1), 0); del buf560  # reuse
            buf562 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_201, linear_200, mul_155, result_202, hidden_states_95, hidden_states_96, pow_20, variance_19, add_127, rsqrt_19, hidden_states_97, to_179, hidden_states_98], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf559, buf561, buf523, buf558, primals_257, buf562, s27, 896, stream=stream0)
            buf563 = buf515; del buf515  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_96, hidden_states_97, to_179, hidden_states_98, result_204], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf562, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_258, (896, 4864), (1, 896), 0), out=buf563)
            buf564 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_202], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_259, buf564, 28672, stream=stream0)
            del primals_259
            buf565 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_202], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf562, (s27, 896), (896, 1), 0), buf564, out=buf565)
            buf566 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_203], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_260, buf566, 155648, stream=stream0)
            del primals_260
            buf567 = buf509; del buf509  # reuse
            # Topologically Sorted Source Nodes: [linear_203], Original ATen: [aten.mm]
            extern_kernels.mm(buf565, buf566, out=buf567)
            buf569 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_96, hidden_states_97, to_179, hidden_states_98, result_204, result_207], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf562, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_261, (896, 4864), (1, 896), 0), out=buf569)
            buf570 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_205], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_262, buf570, 28672, stream=stream0)
            del primals_262
            buf571 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_202, linear_205], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf562, (s27, 896), (896, 1), 0), buf570, out=buf571)
            buf572 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_206], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_263, buf572, 155648, stream=stream0)
            del primals_263
            buf573 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_206], Original ATen: [aten.mm]
            extern_kernels.mm(buf571, buf572, out=buf573)
            buf568 = reinterpret_tensor(buf563, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf563  # reuse
            buf574 = reinterpret_tensor(buf569, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf569  # reuse
            buf575 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_204, linear_203, mul_158, result_205, silu_9, result_207, linear_206, mul_159, result_208, mul_160], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf568, buf574, buf567, buf573, buf575, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf576 = buf558; del buf558  # reuse
            # Topologically Sorted Source Nodes: [silu_9, mul_160, result_210], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf575, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_264, (4864, 896), (1, 4864), 0), out=buf576)
            buf577 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_208], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_265, buf577, 155648, stream=stream0)
            del primals_265
            buf578 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_208], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf575, (s27, 4864), (4864, 1), 0), buf577, out=buf578)
            buf579 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_209], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_266, buf579, 28672, stream=stream0)
            del primals_266
            buf580 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_209], Original ATen: [aten.mm]
            extern_kernels.mm(buf578, buf579, out=buf580)
            buf581 = reinterpret_tensor(buf576, (1, s27, 896), (896*s27, 896, 1), 0); del buf576  # reuse
            buf582 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf583 = reinterpret_tensor(buf582, (1, s27, 1), (s27, 1, 1), 0); del buf582  # reuse
            buf584 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_210, linear_209, mul_161, result_211, hidden_states_99, hidden_states_100, pow_21, variance_20, add_132, rsqrt_20, hidden_states_101, to_187, hidden_states_102], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf581, buf583, buf559, buf580, primals_267, buf584, s27, 896, stream=stream0)
            buf585 = buf580; del buf580  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101, to_187, hidden_states_102, result_213], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf584, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_268, (896, 896), (1, 896), 0), out=buf585)
            buf586 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_211], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_270, buf586, 28672, stream=stream0)
            del primals_270
            buf587 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_211], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf584, (s27, 896), (896, 1), 0), buf586, out=buf587)
            buf588 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_212], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_271, buf588, 28672, stream=stream0)
            del primals_271
            buf589 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_212], Original ATen: [aten.mm]
            extern_kernels.mm(buf587, buf588, out=buf589)
            buf590 = buf541; del buf541  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101, to_187, hidden_states_102, result_213, result_216], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf584, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_272, (896, 128), (1, 896), 0), out=buf590)
            buf591 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_214], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_274, buf591, 28672, stream=stream0)
            del primals_274
            buf592 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_211, linear_214], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf584, (s27, 896), (896, 1), 0), buf591, out=buf592)
            buf593 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_215], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_275, buf593, 4096, stream=stream0)
            del primals_275
            buf594 = buf537; del buf537  # reuse
            # Topologically Sorted Source Nodes: [linear_215], Original ATen: [aten.mm]
            extern_kernels.mm(buf592, buf593, out=buf594)
            buf595 = reinterpret_tensor(buf545, (s27, 128), (128, 1), 0); del buf545  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_100, hidden_states_101, to_187, hidden_states_102, result_213, result_219], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf584, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_276, (896, 128), (1, 896), 0), out=buf595)
            buf596 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_217], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_278, buf596, 28672, stream=stream0)
            del primals_278
            buf597 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_211, linear_217], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf584, (s27, 896), (896, 1), 0), buf596, out=buf597)
            buf598 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_218], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_279, buf598, 4096, stream=stream0)
            del primals_279
            buf599 = reinterpret_tensor(buf544, (s27, 128), (128, 1), 0); del buf544  # reuse
            # Topologically Sorted Source Nodes: [linear_218], Original ATen: [aten.mm]
            extern_kernels.mm(buf597, buf598, out=buf599)
            buf600 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_213, linear_212, mul_164, result_214, view_30, query_states_10, x1_20, x2_20, neg_20, cat_21, mul_168], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_269, buf585, buf589, buf1, buf600, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf601 = reinterpret_tensor(buf585, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf585  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_213, linear_212, mul_164, result_214, view_30, query_states_10, mul_167, q_embed_10], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf601, primals_269, buf589, buf1, buf600, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_269
            buf602 = reinterpret_tensor(buf536, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf536  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_216, linear_215, mul_165, result_217, view_31, key_states_10, x1_21, x2_21, neg_21, cat_22, mul_170], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_273, buf590, buf594, buf1, buf602, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf603 = reinterpret_tensor(buf590, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf590  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_216, linear_215, mul_165, result_217, view_31, key_states_10, mul_169, k_embed_10], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf603, primals_273, buf594, buf1, buf602, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_273
            buf604 = reinterpret_tensor(buf600, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf600  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_216, linear_215, mul_165, result_217, view_31, key_states_10, mul_169, k_embed_10, getitem_467, hidden_states_103, key_10], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf603, buf604, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf605 = reinterpret_tensor(buf589, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf589  # reuse
            # Topologically Sorted Source Nodes: [result_219, linear_218, mul_166, result_220, view_32, value_states_10, getitem_472, hidden_states_104, value_10], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_277, buf595, buf599, buf605, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_277
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_216, linear_215, mul_165, result_217, view_31, key_states_10, result_219, linear_218, mul_166, result_220, view_32, value_states_10, mul_169, k_embed_10, getitem_467, hidden_states_103, key_10, getitem_472, hidden_states_104, value_10, attn_output_30], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf606 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf601, reinterpret_tensor(buf604, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf605, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf607 = buf606[0]
            assert_size_stride(buf607, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf607, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf608 = buf606[1]
            assert_size_stride(buf608, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf608, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf609 = buf606[2]
            assert_size_stride(buf609, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf609, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf610 = buf606[3]
            assert_size_stride(buf610, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf610, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf606
            buf611 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_44, reshape_32, result_222], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf607, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_280, (896, 896), (1, 896), 0), out=buf611)
            buf612 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_220], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_281, buf612, 28672, stream=stream0)
            del primals_281
            buf613 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_44, reshape_32, linear_220], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf607, buf613, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf614 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_44, reshape_32, linear_220], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf613, (s27, 896), (896, 1), 0), buf612, out=buf614)
            buf615 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_221], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_282, buf615, 28672, stream=stream0)
            del primals_282
            buf616 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_221], Original ATen: [aten.mm]
            extern_kernels.mm(buf614, buf615, out=buf616)
            buf617 = reinterpret_tensor(buf611, (1, s27, 896), (896*s27, 896, 1), 0); del buf611  # reuse
            buf618 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf619 = reinterpret_tensor(buf618, (1, s27, 1), (s27, 1, 1), 0); del buf618  # reuse
            buf620 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_222, linear_221, mul_171, result_223, hidden_states_105, hidden_states_106, pow_22, variance_21, add_140, rsqrt_21, hidden_states_107, to_197, hidden_states_108], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf617, buf619, buf581, buf616, primals_283, buf620, s27, 896, stream=stream0)
            buf621 = buf573; del buf573  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_106, hidden_states_107, to_197, hidden_states_108, result_225], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf620, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_284, (896, 4864), (1, 896), 0), out=buf621)
            buf622 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_223], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_285, buf622, 28672, stream=stream0)
            del primals_285
            buf623 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_223], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf620, (s27, 896), (896, 1), 0), buf622, out=buf623)
            buf624 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_224], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_286, buf624, 155648, stream=stream0)
            del primals_286
            buf625 = buf567; del buf567  # reuse
            # Topologically Sorted Source Nodes: [linear_224], Original ATen: [aten.mm]
            extern_kernels.mm(buf623, buf624, out=buf625)
            buf627 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_106, hidden_states_107, to_197, hidden_states_108, result_225, result_228], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf620, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_287, (896, 4864), (1, 896), 0), out=buf627)
            buf628 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_226], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_288, buf628, 28672, stream=stream0)
            del primals_288
            buf629 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_223, linear_226], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf620, (s27, 896), (896, 1), 0), buf628, out=buf629)
            buf630 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_227], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_289, buf630, 155648, stream=stream0)
            del primals_289
            buf631 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_227], Original ATen: [aten.mm]
            extern_kernels.mm(buf629, buf630, out=buf631)
            buf626 = reinterpret_tensor(buf621, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf621  # reuse
            buf632 = reinterpret_tensor(buf627, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf627  # reuse
            buf633 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_225, linear_224, mul_174, result_226, silu_10, result_228, linear_227, mul_175, result_229, mul_176], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf626, buf632, buf625, buf631, buf633, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf634 = buf616; del buf616  # reuse
            # Topologically Sorted Source Nodes: [silu_10, mul_176, result_231], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf633, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_290, (4864, 896), (1, 4864), 0), out=buf634)
            buf635 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_229], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_291, buf635, 155648, stream=stream0)
            del primals_291
            buf636 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_229], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf633, (s27, 4864), (4864, 1), 0), buf635, out=buf636)
            buf637 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_230], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_292, buf637, 28672, stream=stream0)
            del primals_292
            buf638 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_230], Original ATen: [aten.mm]
            extern_kernels.mm(buf636, buf637, out=buf638)
            buf639 = reinterpret_tensor(buf634, (1, s27, 896), (896*s27, 896, 1), 0); del buf634  # reuse
            buf640 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf641 = reinterpret_tensor(buf640, (1, s27, 1), (s27, 1, 1), 0); del buf640  # reuse
            buf642 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_231, linear_230, mul_177, result_232, hidden_states_109, hidden_states_110, pow_23, variance_22, add_145, rsqrt_22, hidden_states_111, to_205, hidden_states_112], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf639, buf641, buf617, buf638, primals_293, buf642, s27, 896, stream=stream0)
            buf643 = buf638; del buf638  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_111, to_205, hidden_states_112, result_234], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf642, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_294, (896, 896), (1, 896), 0), out=buf643)
            buf644 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_232], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_296, buf644, 28672, stream=stream0)
            del primals_296
            buf645 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_232], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf642, (s27, 896), (896, 1), 0), buf644, out=buf645)
            buf646 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_233], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_297, buf646, 28672, stream=stream0)
            del primals_297
            buf647 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_233], Original ATen: [aten.mm]
            extern_kernels.mm(buf645, buf646, out=buf647)
            buf648 = buf599; del buf599  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_111, to_205, hidden_states_112, result_234, result_237], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf642, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_298, (896, 128), (1, 896), 0), out=buf648)
            buf649 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_235], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_300, buf649, 28672, stream=stream0)
            del primals_300
            buf650 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_232, linear_235], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf642, (s27, 896), (896, 1), 0), buf649, out=buf650)
            buf651 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_236], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_301, buf651, 4096, stream=stream0)
            del primals_301
            buf652 = buf595; del buf595  # reuse
            # Topologically Sorted Source Nodes: [linear_236], Original ATen: [aten.mm]
            extern_kernels.mm(buf650, buf651, out=buf652)
            buf653 = reinterpret_tensor(buf603, (s27, 128), (128, 1), 0); del buf603  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_111, to_205, hidden_states_112, result_234, result_240], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf642, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_302, (896, 128), (1, 896), 0), out=buf653)
            buf654 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_238], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_304, buf654, 28672, stream=stream0)
            del primals_304
            buf655 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_232, linear_238], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf642, (s27, 896), (896, 1), 0), buf654, out=buf655)
            buf656 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_239], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_305, buf656, 4096, stream=stream0)
            del primals_305
            buf657 = reinterpret_tensor(buf602, (s27, 128), (128, 1), 0); del buf602  # reuse
            # Topologically Sorted Source Nodes: [linear_239], Original ATen: [aten.mm]
            extern_kernels.mm(buf655, buf656, out=buf657)
            buf658 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_234, linear_233, mul_180, result_235, view_33, query_states_11, x1_22, x2_22, neg_22, cat_23, mul_184], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_295, buf643, buf647, buf1, buf658, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf659 = reinterpret_tensor(buf643, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf643  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_234, linear_233, mul_180, result_235, view_33, query_states_11, mul_183, q_embed_11], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf659, primals_295, buf647, buf1, buf658, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_295
            buf660 = reinterpret_tensor(buf594, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf594  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_237, linear_236, mul_181, result_238, view_34, key_states_11, x1_23, x2_23, neg_23, cat_24, mul_186], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_299, buf648, buf652, buf1, buf660, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf661 = reinterpret_tensor(buf648, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf648  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_237, linear_236, mul_181, result_238, view_34, key_states_11, mul_185, k_embed_11], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf661, primals_299, buf652, buf1, buf660, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_299
            buf662 = reinterpret_tensor(buf658, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf658  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_237, linear_236, mul_181, result_238, view_34, key_states_11, mul_185, k_embed_11, getitem_509, hidden_states_113, key_11], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf661, buf662, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf663 = reinterpret_tensor(buf647, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf647  # reuse
            # Topologically Sorted Source Nodes: [result_240, linear_239, mul_182, result_241, view_35, value_states_11, getitem_514, hidden_states_114, value_11], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_303, buf653, buf657, buf663, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_303
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_237, linear_236, mul_181, result_238, view_34, key_states_11, result_240, linear_239, mul_182, result_241, view_35, value_states_11, mul_185, k_embed_11, getitem_509, hidden_states_113, key_11, getitem_514, hidden_states_114, value_11, attn_output_33], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf664 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf659, reinterpret_tensor(buf662, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf663, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf665 = buf664[0]
            assert_size_stride(buf665, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf665, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf666 = buf664[1]
            assert_size_stride(buf666, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf666, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf667 = buf664[2]
            assert_size_stride(buf667, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf667, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf668 = buf664[3]
            assert_size_stride(buf668, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf668, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf664
            buf669 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_48, reshape_35, result_243], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf665, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_306, (896, 896), (1, 896), 0), out=buf669)
            buf670 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_241], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_307, buf670, 28672, stream=stream0)
            del primals_307
            buf671 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_48, reshape_35, linear_241], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf665, buf671, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf672 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_48, reshape_35, linear_241], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf671, (s27, 896), (896, 1), 0), buf670, out=buf672)
            buf673 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_242], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_308, buf673, 28672, stream=stream0)
            del primals_308
            buf674 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_242], Original ATen: [aten.mm]
            extern_kernels.mm(buf672, buf673, out=buf674)
            buf675 = reinterpret_tensor(buf669, (1, s27, 896), (896*s27, 896, 1), 0); del buf669  # reuse
            buf676 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf677 = reinterpret_tensor(buf676, (1, s27, 1), (s27, 1, 1), 0); del buf676  # reuse
            buf678 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_243, linear_242, mul_187, result_244, hidden_states_115, hidden_states_116, pow_24, variance_23, add_153, rsqrt_23, hidden_states_117, to_215, hidden_states_118], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf675, buf677, buf639, buf674, primals_309, buf678, s27, 896, stream=stream0)
            buf679 = buf631; del buf631  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_116, hidden_states_117, to_215, hidden_states_118, result_246], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf678, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_310, (896, 4864), (1, 896), 0), out=buf679)
            buf680 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_244], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_311, buf680, 28672, stream=stream0)
            del primals_311
            buf681 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_244], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf678, (s27, 896), (896, 1), 0), buf680, out=buf681)
            buf682 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_245], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_312, buf682, 155648, stream=stream0)
            del primals_312
            buf683 = buf625; del buf625  # reuse
            # Topologically Sorted Source Nodes: [linear_245], Original ATen: [aten.mm]
            extern_kernels.mm(buf681, buf682, out=buf683)
            buf685 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_116, hidden_states_117, to_215, hidden_states_118, result_246, result_249], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf678, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_313, (896, 4864), (1, 896), 0), out=buf685)
            buf686 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_247], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_314, buf686, 28672, stream=stream0)
            del primals_314
            buf687 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_244, linear_247], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf678, (s27, 896), (896, 1), 0), buf686, out=buf687)
            buf688 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_248], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_315, buf688, 155648, stream=stream0)
            del primals_315
            buf689 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_248], Original ATen: [aten.mm]
            extern_kernels.mm(buf687, buf688, out=buf689)
            buf684 = reinterpret_tensor(buf679, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf679  # reuse
            buf690 = reinterpret_tensor(buf685, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf685  # reuse
            buf691 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_246, linear_245, mul_190, result_247, silu_11, result_249, linear_248, mul_191, result_250, mul_192], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf684, buf690, buf683, buf689, buf691, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf692 = buf674; del buf674  # reuse
            # Topologically Sorted Source Nodes: [silu_11, mul_192, result_252], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf691, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_316, (4864, 896), (1, 4864), 0), out=buf692)
            buf693 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_250], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_317, buf693, 155648, stream=stream0)
            del primals_317
            buf694 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_250], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf691, (s27, 4864), (4864, 1), 0), buf693, out=buf694)
            buf695 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_251], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_318, buf695, 28672, stream=stream0)
            del primals_318
            buf696 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_251], Original ATen: [aten.mm]
            extern_kernels.mm(buf694, buf695, out=buf696)
            buf697 = reinterpret_tensor(buf692, (1, s27, 896), (896*s27, 896, 1), 0); del buf692  # reuse
            buf698 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf699 = reinterpret_tensor(buf698, (1, s27, 1), (s27, 1, 1), 0); del buf698  # reuse
            buf700 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_252, linear_251, mul_193, result_253, hidden_states_119, hidden_states_120, pow_25, variance_24, add_158, rsqrt_24, hidden_states_121, to_223, hidden_states_122], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf697, buf699, buf675, buf696, primals_319, buf700, s27, 896, stream=stream0)
            buf701 = buf696; del buf696  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_120, hidden_states_121, to_223, hidden_states_122, result_255], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf700, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_320, (896, 896), (1, 896), 0), out=buf701)
            buf702 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_253], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_322, buf702, 28672, stream=stream0)
            del primals_322
            buf703 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_253], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf700, (s27, 896), (896, 1), 0), buf702, out=buf703)
            buf704 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_254], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_323, buf704, 28672, stream=stream0)
            del primals_323
            buf705 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_254], Original ATen: [aten.mm]
            extern_kernels.mm(buf703, buf704, out=buf705)
            buf706 = buf657; del buf657  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_120, hidden_states_121, to_223, hidden_states_122, result_255, result_258], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf700, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_324, (896, 128), (1, 896), 0), out=buf706)
            buf707 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_256], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_326, buf707, 28672, stream=stream0)
            del primals_326
            buf708 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_253, linear_256], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf700, (s27, 896), (896, 1), 0), buf707, out=buf708)
            buf709 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_257], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_327, buf709, 4096, stream=stream0)
            del primals_327
            buf710 = buf653; del buf653  # reuse
            # Topologically Sorted Source Nodes: [linear_257], Original ATen: [aten.mm]
            extern_kernels.mm(buf708, buf709, out=buf710)
            buf711 = reinterpret_tensor(buf661, (s27, 128), (128, 1), 0); del buf661  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_120, hidden_states_121, to_223, hidden_states_122, result_255, result_261], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf700, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_328, (896, 128), (1, 896), 0), out=buf711)
            buf712 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_259], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_330, buf712, 28672, stream=stream0)
            del primals_330
            buf713 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_253, linear_259], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf700, (s27, 896), (896, 1), 0), buf712, out=buf713)
            buf714 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_260], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_331, buf714, 4096, stream=stream0)
            del primals_331
            buf715 = reinterpret_tensor(buf660, (s27, 128), (128, 1), 0); del buf660  # reuse
            # Topologically Sorted Source Nodes: [linear_260], Original ATen: [aten.mm]
            extern_kernels.mm(buf713, buf714, out=buf715)
            buf716 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_255, linear_254, mul_196, result_256, view_36, query_states_12, x1_24, x2_24, neg_24, cat_25, mul_200], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_321, buf701, buf705, buf1, buf716, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf717 = reinterpret_tensor(buf701, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf701  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_255, linear_254, mul_196, result_256, view_36, query_states_12, mul_199, q_embed_12], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf717, primals_321, buf705, buf1, buf716, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_321
            buf718 = reinterpret_tensor(buf652, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf652  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_258, linear_257, mul_197, result_259, view_37, key_states_12, x1_25, x2_25, neg_25, cat_26, mul_202], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_325, buf706, buf710, buf1, buf718, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf719 = reinterpret_tensor(buf706, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf706  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_258, linear_257, mul_197, result_259, view_37, key_states_12, mul_201, k_embed_12], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf719, primals_325, buf710, buf1, buf718, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_325
            buf720 = reinterpret_tensor(buf716, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf716  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_258, linear_257, mul_197, result_259, view_37, key_states_12, mul_201, k_embed_12, getitem_551, hidden_states_123, key_12], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf719, buf720, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf721 = reinterpret_tensor(buf705, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf705  # reuse
            # Topologically Sorted Source Nodes: [result_261, linear_260, mul_198, result_262, view_38, value_states_12, getitem_556, hidden_states_124, value_12], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_329, buf711, buf715, buf721, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_329
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_258, linear_257, mul_197, result_259, view_37, key_states_12, result_261, linear_260, mul_198, result_262, view_38, value_states_12, mul_201, k_embed_12, getitem_551, hidden_states_123, key_12, getitem_556, hidden_states_124, value_12, attn_output_36], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf722 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf717, reinterpret_tensor(buf720, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf721, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf723 = buf722[0]
            assert_size_stride(buf723, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf723, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf724 = buf722[1]
            assert_size_stride(buf724, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf724, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf725 = buf722[2]
            assert_size_stride(buf725, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf725, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf726 = buf722[3]
            assert_size_stride(buf726, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf726, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf722
            buf727 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_52, reshape_38, result_264], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf723, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_332, (896, 896), (1, 896), 0), out=buf727)
            buf728 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_262], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_333, buf728, 28672, stream=stream0)
            del primals_333
            buf729 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_52, reshape_38, linear_262], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf723, buf729, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf730 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_52, reshape_38, linear_262], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf729, (s27, 896), (896, 1), 0), buf728, out=buf730)
            buf731 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_263], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_334, buf731, 28672, stream=stream0)
            del primals_334
            buf732 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_263], Original ATen: [aten.mm]
            extern_kernels.mm(buf730, buf731, out=buf732)
            buf733 = reinterpret_tensor(buf727, (1, s27, 896), (896*s27, 896, 1), 0); del buf727  # reuse
            buf734 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf735 = reinterpret_tensor(buf734, (1, s27, 1), (s27, 1, 1), 0); del buf734  # reuse
            buf736 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_264, linear_263, mul_203, result_265, hidden_states_125, hidden_states_126, pow_26, variance_25, add_166, rsqrt_25, hidden_states_127, to_233, hidden_states_128], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf733, buf735, buf697, buf732, primals_335, buf736, s27, 896, stream=stream0)
            buf737 = buf689; del buf689  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_126, hidden_states_127, to_233, hidden_states_128, result_267], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf736, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_336, (896, 4864), (1, 896), 0), out=buf737)
            buf738 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_265], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_337, buf738, 28672, stream=stream0)
            del primals_337
            buf739 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_265], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf736, (s27, 896), (896, 1), 0), buf738, out=buf739)
            buf740 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_266], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_338, buf740, 155648, stream=stream0)
            del primals_338
            buf741 = buf683; del buf683  # reuse
            # Topologically Sorted Source Nodes: [linear_266], Original ATen: [aten.mm]
            extern_kernels.mm(buf739, buf740, out=buf741)
            buf743 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_126, hidden_states_127, to_233, hidden_states_128, result_267, result_270], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf736, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_339, (896, 4864), (1, 896), 0), out=buf743)
            buf744 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_268], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_340, buf744, 28672, stream=stream0)
            del primals_340
            buf745 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_265, linear_268], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf736, (s27, 896), (896, 1), 0), buf744, out=buf745)
            buf746 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_269], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_341, buf746, 155648, stream=stream0)
            del primals_341
            buf747 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_269], Original ATen: [aten.mm]
            extern_kernels.mm(buf745, buf746, out=buf747)
            buf742 = reinterpret_tensor(buf737, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf737  # reuse
            buf748 = reinterpret_tensor(buf743, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf743  # reuse
            buf749 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_267, linear_266, mul_206, result_268, silu_12, result_270, linear_269, mul_207, result_271, mul_208], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf742, buf748, buf741, buf747, buf749, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf750 = buf732; del buf732  # reuse
            # Topologically Sorted Source Nodes: [silu_12, mul_208, result_273], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf749, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_342, (4864, 896), (1, 4864), 0), out=buf750)
            buf751 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_271], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_343, buf751, 155648, stream=stream0)
            del primals_343
            buf752 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_271], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf749, (s27, 4864), (4864, 1), 0), buf751, out=buf752)
            buf753 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_272], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_344, buf753, 28672, stream=stream0)
            del primals_344
            buf754 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_272], Original ATen: [aten.mm]
            extern_kernels.mm(buf752, buf753, out=buf754)
            buf755 = reinterpret_tensor(buf750, (1, s27, 896), (896*s27, 896, 1), 0); del buf750  # reuse
            buf756 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf757 = reinterpret_tensor(buf756, (1, s27, 1), (s27, 1, 1), 0); del buf756  # reuse
            buf758 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_273, linear_272, mul_209, result_274, hidden_states_129, hidden_states_130, pow_27, variance_26, add_171, rsqrt_26, hidden_states_131, to_241, hidden_states_132], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf755, buf757, buf733, buf754, primals_345, buf758, s27, 896, stream=stream0)
            buf759 = buf754; del buf754  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_131, to_241, hidden_states_132, result_276], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf758, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_346, (896, 896), (1, 896), 0), out=buf759)
            buf760 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_274], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_348, buf760, 28672, stream=stream0)
            del primals_348
            buf761 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_274], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf758, (s27, 896), (896, 1), 0), buf760, out=buf761)
            buf762 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_275], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_349, buf762, 28672, stream=stream0)
            del primals_349
            buf763 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_275], Original ATen: [aten.mm]
            extern_kernels.mm(buf761, buf762, out=buf763)
            buf764 = buf715; del buf715  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_131, to_241, hidden_states_132, result_276, result_279], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf758, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_350, (896, 128), (1, 896), 0), out=buf764)
            buf765 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_277], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_352, buf765, 28672, stream=stream0)
            del primals_352
            buf766 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_274, linear_277], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf758, (s27, 896), (896, 1), 0), buf765, out=buf766)
            buf767 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_278], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_353, buf767, 4096, stream=stream0)
            del primals_353
            buf768 = buf711; del buf711  # reuse
            # Topologically Sorted Source Nodes: [linear_278], Original ATen: [aten.mm]
            extern_kernels.mm(buf766, buf767, out=buf768)
            buf769 = reinterpret_tensor(buf719, (s27, 128), (128, 1), 0); del buf719  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_130, hidden_states_131, to_241, hidden_states_132, result_276, result_282], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf758, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_354, (896, 128), (1, 896), 0), out=buf769)
            buf770 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_280], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_356, buf770, 28672, stream=stream0)
            del primals_356
            buf771 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_274, linear_280], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf758, (s27, 896), (896, 1), 0), buf770, out=buf771)
            buf772 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_281], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_357, buf772, 4096, stream=stream0)
            del primals_357
            buf773 = reinterpret_tensor(buf718, (s27, 128), (128, 1), 0); del buf718  # reuse
            # Topologically Sorted Source Nodes: [linear_281], Original ATen: [aten.mm]
            extern_kernels.mm(buf771, buf772, out=buf773)
            buf774 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_276, linear_275, mul_212, result_277, view_39, query_states_13, x1_26, x2_26, neg_26, cat_27, mul_216], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_347, buf759, buf763, buf1, buf774, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf775 = reinterpret_tensor(buf759, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf759  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_276, linear_275, mul_212, result_277, view_39, query_states_13, mul_215, q_embed_13], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf775, primals_347, buf763, buf1, buf774, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_347
            buf776 = reinterpret_tensor(buf710, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf710  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_279, linear_278, mul_213, result_280, view_40, key_states_13, x1_27, x2_27, neg_27, cat_28, mul_218], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_351, buf764, buf768, buf1, buf776, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf777 = reinterpret_tensor(buf764, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf764  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_279, linear_278, mul_213, result_280, view_40, key_states_13, mul_217, k_embed_13], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf777, primals_351, buf768, buf1, buf776, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_351
            buf778 = reinterpret_tensor(buf774, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf774  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_279, linear_278, mul_213, result_280, view_40, key_states_13, mul_217, k_embed_13, getitem_593, hidden_states_133, key_13], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf777, buf778, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf779 = reinterpret_tensor(buf763, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf763  # reuse
            # Topologically Sorted Source Nodes: [result_282, linear_281, mul_214, result_283, view_41, value_states_13, getitem_598, hidden_states_134, value_13], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_355, buf769, buf773, buf779, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_355
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_279, linear_278, mul_213, result_280, view_40, key_states_13, result_282, linear_281, mul_214, result_283, view_41, value_states_13, mul_217, k_embed_13, getitem_593, hidden_states_133, key_13, getitem_598, hidden_states_134, value_13, attn_output_39], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf780 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf775, reinterpret_tensor(buf778, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf779, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf781 = buf780[0]
            assert_size_stride(buf781, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf781, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf782 = buf780[1]
            assert_size_stride(buf782, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf782, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf783 = buf780[2]
            assert_size_stride(buf783, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf783, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf784 = buf780[3]
            assert_size_stride(buf784, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf784, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf780
            buf785 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_56, reshape_41, result_285], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf781, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_358, (896, 896), (1, 896), 0), out=buf785)
            buf786 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_283], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_359, buf786, 28672, stream=stream0)
            del primals_359
            buf787 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_56, reshape_41, linear_283], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf781, buf787, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf788 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_56, reshape_41, linear_283], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf787, (s27, 896), (896, 1), 0), buf786, out=buf788)
            buf789 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_284], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_360, buf789, 28672, stream=stream0)
            del primals_360
            buf790 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_284], Original ATen: [aten.mm]
            extern_kernels.mm(buf788, buf789, out=buf790)
            buf791 = reinterpret_tensor(buf785, (1, s27, 896), (896*s27, 896, 1), 0); del buf785  # reuse
            buf792 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf793 = reinterpret_tensor(buf792, (1, s27, 1), (s27, 1, 1), 0); del buf792  # reuse
            buf794 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_285, linear_284, mul_219, result_286, hidden_states_135, hidden_states_136, pow_28, variance_27, add_179, rsqrt_27, hidden_states_137, to_251, hidden_states_138], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf791, buf793, buf755, buf790, primals_361, buf794, s27, 896, stream=stream0)
            buf795 = buf747; del buf747  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137, to_251, hidden_states_138, result_288], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf794, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_362, (896, 4864), (1, 896), 0), out=buf795)
            buf796 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_286], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_363, buf796, 28672, stream=stream0)
            del primals_363
            buf797 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_286], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf794, (s27, 896), (896, 1), 0), buf796, out=buf797)
            buf798 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_287], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_364, buf798, 155648, stream=stream0)
            del primals_364
            buf799 = buf741; del buf741  # reuse
            # Topologically Sorted Source Nodes: [linear_287], Original ATen: [aten.mm]
            extern_kernels.mm(buf797, buf798, out=buf799)
            buf801 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_136, hidden_states_137, to_251, hidden_states_138, result_288, result_291], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf794, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_365, (896, 4864), (1, 896), 0), out=buf801)
            buf802 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_289], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_366, buf802, 28672, stream=stream0)
            del primals_366
            buf803 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_286, linear_289], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf794, (s27, 896), (896, 1), 0), buf802, out=buf803)
            buf804 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_290], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_367, buf804, 155648, stream=stream0)
            del primals_367
            buf805 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_290], Original ATen: [aten.mm]
            extern_kernels.mm(buf803, buf804, out=buf805)
            buf800 = reinterpret_tensor(buf795, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf795  # reuse
            buf806 = reinterpret_tensor(buf801, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf801  # reuse
            buf807 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_288, linear_287, mul_222, result_289, silu_13, result_291, linear_290, mul_223, result_292, mul_224], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf800, buf806, buf799, buf805, buf807, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf808 = buf790; del buf790  # reuse
            # Topologically Sorted Source Nodes: [silu_13, mul_224, result_294], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf807, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_368, (4864, 896), (1, 4864), 0), out=buf808)
            buf809 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_292], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_369, buf809, 155648, stream=stream0)
            del primals_369
            buf810 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_292], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf807, (s27, 4864), (4864, 1), 0), buf809, out=buf810)
            buf811 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_293], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_370, buf811, 28672, stream=stream0)
            del primals_370
            buf812 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_293], Original ATen: [aten.mm]
            extern_kernels.mm(buf810, buf811, out=buf812)
            buf813 = reinterpret_tensor(buf808, (1, s27, 896), (896*s27, 896, 1), 0); del buf808  # reuse
            buf814 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf815 = reinterpret_tensor(buf814, (1, s27, 1), (s27, 1, 1), 0); del buf814  # reuse
            buf816 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_294, linear_293, mul_225, result_295, hidden_states_139, hidden_states_140, pow_29, variance_28, add_184, rsqrt_28, hidden_states_141, to_259, hidden_states_142], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf813, buf815, buf791, buf812, primals_371, buf816, s27, 896, stream=stream0)
            buf817 = buf812; del buf812  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_140, hidden_states_141, to_259, hidden_states_142, result_297], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf816, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_372, (896, 896), (1, 896), 0), out=buf817)
            buf818 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_295], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_374, buf818, 28672, stream=stream0)
            del primals_374
            buf819 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_295], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf816, (s27, 896), (896, 1), 0), buf818, out=buf819)
            buf820 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_296], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_375, buf820, 28672, stream=stream0)
            del primals_375
            buf821 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_296], Original ATen: [aten.mm]
            extern_kernels.mm(buf819, buf820, out=buf821)
            buf822 = buf773; del buf773  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_140, hidden_states_141, to_259, hidden_states_142, result_297, result_300], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf816, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_376, (896, 128), (1, 896), 0), out=buf822)
            buf823 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_298], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_378, buf823, 28672, stream=stream0)
            del primals_378
            buf824 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_295, linear_298], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf816, (s27, 896), (896, 1), 0), buf823, out=buf824)
            buf825 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_299], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_379, buf825, 4096, stream=stream0)
            del primals_379
            buf826 = buf769; del buf769  # reuse
            # Topologically Sorted Source Nodes: [linear_299], Original ATen: [aten.mm]
            extern_kernels.mm(buf824, buf825, out=buf826)
            buf827 = reinterpret_tensor(buf777, (s27, 128), (128, 1), 0); del buf777  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_140, hidden_states_141, to_259, hidden_states_142, result_297, result_303], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf816, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_380, (896, 128), (1, 896), 0), out=buf827)
            buf828 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_301], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_382, buf828, 28672, stream=stream0)
            del primals_382
            buf829 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_295, linear_301], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf816, (s27, 896), (896, 1), 0), buf828, out=buf829)
            buf830 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_302], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_383, buf830, 4096, stream=stream0)
            del primals_383
            buf831 = reinterpret_tensor(buf776, (s27, 128), (128, 1), 0); del buf776  # reuse
            # Topologically Sorted Source Nodes: [linear_302], Original ATen: [aten.mm]
            extern_kernels.mm(buf829, buf830, out=buf831)
            buf832 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_297, linear_296, mul_228, result_298, view_42, query_states_14, x1_28, x2_28, neg_28, cat_29, mul_232], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_373, buf817, buf821, buf1, buf832, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf833 = reinterpret_tensor(buf817, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf817  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_297, linear_296, mul_228, result_298, view_42, query_states_14, mul_231, q_embed_14], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf833, primals_373, buf821, buf1, buf832, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_373
            buf834 = reinterpret_tensor(buf768, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf768  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_300, linear_299, mul_229, result_301, view_43, key_states_14, x1_29, x2_29, neg_29, cat_30, mul_234], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_377, buf822, buf826, buf1, buf834, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf835 = reinterpret_tensor(buf822, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf822  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_300, linear_299, mul_229, result_301, view_43, key_states_14, mul_233, k_embed_14], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf835, primals_377, buf826, buf1, buf834, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_377
            buf836 = reinterpret_tensor(buf832, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf832  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_300, linear_299, mul_229, result_301, view_43, key_states_14, mul_233, k_embed_14, getitem_635, hidden_states_143, key_14], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf835, buf836, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf837 = reinterpret_tensor(buf821, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf821  # reuse
            # Topologically Sorted Source Nodes: [result_303, linear_302, mul_230, result_304, view_44, value_states_14, getitem_640, hidden_states_144, value_14], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_381, buf827, buf831, buf837, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_381
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_300, linear_299, mul_229, result_301, view_43, key_states_14, result_303, linear_302, mul_230, result_304, view_44, value_states_14, mul_233, k_embed_14, getitem_635, hidden_states_143, key_14, getitem_640, hidden_states_144, value_14, attn_output_42], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf838 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf833, reinterpret_tensor(buf836, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf837, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf839 = buf838[0]
            assert_size_stride(buf839, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf839, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf840 = buf838[1]
            assert_size_stride(buf840, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf840, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf841 = buf838[2]
            assert_size_stride(buf841, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf841, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf842 = buf838[3]
            assert_size_stride(buf842, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf842, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf838
            buf843 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_60, reshape_44, result_306], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf839, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_384, (896, 896), (1, 896), 0), out=buf843)
            buf844 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_304], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_385, buf844, 28672, stream=stream0)
            del primals_385
            buf845 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_60, reshape_44, linear_304], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf839, buf845, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf846 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_60, reshape_44, linear_304], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf845, (s27, 896), (896, 1), 0), buf844, out=buf846)
            buf847 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_305], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_386, buf847, 28672, stream=stream0)
            del primals_386
            buf848 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_305], Original ATen: [aten.mm]
            extern_kernels.mm(buf846, buf847, out=buf848)
            buf849 = reinterpret_tensor(buf843, (1, s27, 896), (896*s27, 896, 1), 0); del buf843  # reuse
            buf850 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf851 = reinterpret_tensor(buf850, (1, s27, 1), (s27, 1, 1), 0); del buf850  # reuse
            buf852 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_306, linear_305, mul_235, result_307, hidden_states_145, hidden_states_146, pow_30, variance_29, add_192, rsqrt_29, hidden_states_147, to_269, hidden_states_148], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf849, buf851, buf813, buf848, primals_387, buf852, s27, 896, stream=stream0)
            buf853 = buf805; del buf805  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_146, hidden_states_147, to_269, hidden_states_148, result_309], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf852, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_388, (896, 4864), (1, 896), 0), out=buf853)
            buf854 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_307], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_389, buf854, 28672, stream=stream0)
            del primals_389
            buf855 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_307], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf852, (s27, 896), (896, 1), 0), buf854, out=buf855)
            buf856 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_308], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_390, buf856, 155648, stream=stream0)
            del primals_390
            buf857 = buf799; del buf799  # reuse
            # Topologically Sorted Source Nodes: [linear_308], Original ATen: [aten.mm]
            extern_kernels.mm(buf855, buf856, out=buf857)
            buf859 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_146, hidden_states_147, to_269, hidden_states_148, result_309, result_312], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf852, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_391, (896, 4864), (1, 896), 0), out=buf859)
            buf860 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_310], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_392, buf860, 28672, stream=stream0)
            del primals_392
            buf861 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_307, linear_310], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf852, (s27, 896), (896, 1), 0), buf860, out=buf861)
            buf862 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_311], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_393, buf862, 155648, stream=stream0)
            del primals_393
            buf863 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_311], Original ATen: [aten.mm]
            extern_kernels.mm(buf861, buf862, out=buf863)
            buf858 = reinterpret_tensor(buf853, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf853  # reuse
            buf864 = reinterpret_tensor(buf859, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf859  # reuse
            buf865 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_309, linear_308, mul_238, result_310, silu_14, result_312, linear_311, mul_239, result_313, mul_240], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf858, buf864, buf857, buf863, buf865, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf866 = buf848; del buf848  # reuse
            # Topologically Sorted Source Nodes: [silu_14, mul_240, result_315], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf865, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_394, (4864, 896), (1, 4864), 0), out=buf866)
            buf867 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_313], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_395, buf867, 155648, stream=stream0)
            del primals_395
            buf868 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_313], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf865, (s27, 4864), (4864, 1), 0), buf867, out=buf868)
            buf869 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_314], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_396, buf869, 28672, stream=stream0)
            del primals_396
            buf870 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_314], Original ATen: [aten.mm]
            extern_kernels.mm(buf868, buf869, out=buf870)
            buf871 = reinterpret_tensor(buf866, (1, s27, 896), (896*s27, 896, 1), 0); del buf866  # reuse
            buf872 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf873 = reinterpret_tensor(buf872, (1, s27, 1), (s27, 1, 1), 0); del buf872  # reuse
            buf874 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_315, linear_314, mul_241, result_316, hidden_states_149, hidden_states_150, pow_31, variance_30, add_197, rsqrt_30, hidden_states_151, to_277, hidden_states_152], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf871, buf873, buf849, buf870, primals_397, buf874, s27, 896, stream=stream0)
            buf875 = buf870; del buf870  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151, to_277, hidden_states_152, result_318], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf874, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_398, (896, 896), (1, 896), 0), out=buf875)
            buf876 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_316], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_400, buf876, 28672, stream=stream0)
            del primals_400
            buf877 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_316], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf874, (s27, 896), (896, 1), 0), buf876, out=buf877)
            buf878 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_317], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_401, buf878, 28672, stream=stream0)
            del primals_401
            buf879 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_317], Original ATen: [aten.mm]
            extern_kernels.mm(buf877, buf878, out=buf879)
            buf880 = buf831; del buf831  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151, to_277, hidden_states_152, result_318, result_321], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf874, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_402, (896, 128), (1, 896), 0), out=buf880)
            buf881 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_319], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_404, buf881, 28672, stream=stream0)
            del primals_404
            buf882 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_316, linear_319], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf874, (s27, 896), (896, 1), 0), buf881, out=buf882)
            buf883 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_320], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_405, buf883, 4096, stream=stream0)
            del primals_405
            buf884 = buf827; del buf827  # reuse
            # Topologically Sorted Source Nodes: [linear_320], Original ATen: [aten.mm]
            extern_kernels.mm(buf882, buf883, out=buf884)
            buf885 = reinterpret_tensor(buf835, (s27, 128), (128, 1), 0); del buf835  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_150, hidden_states_151, to_277, hidden_states_152, result_318, result_324], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf874, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_406, (896, 128), (1, 896), 0), out=buf885)
            buf886 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_322], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_408, buf886, 28672, stream=stream0)
            del primals_408
            buf887 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_316, linear_322], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf874, (s27, 896), (896, 1), 0), buf886, out=buf887)
            buf888 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_323], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_409, buf888, 4096, stream=stream0)
            del primals_409
            buf889 = reinterpret_tensor(buf834, (s27, 128), (128, 1), 0); del buf834  # reuse
            # Topologically Sorted Source Nodes: [linear_323], Original ATen: [aten.mm]
            extern_kernels.mm(buf887, buf888, out=buf889)
            buf890 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_318, linear_317, mul_244, result_319, view_45, query_states_15, x1_30, x2_30, neg_30, cat_31, mul_248], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_399, buf875, buf879, buf1, buf890, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf891 = reinterpret_tensor(buf875, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf875  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_318, linear_317, mul_244, result_319, view_45, query_states_15, mul_247, q_embed_15], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf891, primals_399, buf879, buf1, buf890, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_399
            buf892 = reinterpret_tensor(buf826, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf826  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_321, linear_320, mul_245, result_322, view_46, key_states_15, x1_31, x2_31, neg_31, cat_32, mul_250], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_403, buf880, buf884, buf1, buf892, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf893 = reinterpret_tensor(buf880, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf880  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_321, linear_320, mul_245, result_322, view_46, key_states_15, mul_249, k_embed_15], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf893, primals_403, buf884, buf1, buf892, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_403
            buf894 = reinterpret_tensor(buf890, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf890  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_321, linear_320, mul_245, result_322, view_46, key_states_15, mul_249, k_embed_15, getitem_677, hidden_states_153, key_15], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf893, buf894, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf895 = reinterpret_tensor(buf879, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf879  # reuse
            # Topologically Sorted Source Nodes: [result_324, linear_323, mul_246, result_325, view_47, value_states_15, getitem_682, hidden_states_154, value_15], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_407, buf885, buf889, buf895, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_407
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_321, linear_320, mul_245, result_322, view_46, key_states_15, result_324, linear_323, mul_246, result_325, view_47, value_states_15, mul_249, k_embed_15, getitem_677, hidden_states_153, key_15, getitem_682, hidden_states_154, value_15, attn_output_45], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf896 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf891, reinterpret_tensor(buf894, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf895, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf897 = buf896[0]
            assert_size_stride(buf897, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf897, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf898 = buf896[1]
            assert_size_stride(buf898, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf898, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf899 = buf896[2]
            assert_size_stride(buf899, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf899, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf900 = buf896[3]
            assert_size_stride(buf900, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf900, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf896
            buf901 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_64, reshape_47, result_327], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf897, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_410, (896, 896), (1, 896), 0), out=buf901)
            buf902 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_325], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_411, buf902, 28672, stream=stream0)
            del primals_411
            buf903 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_64, reshape_47, linear_325], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf897, buf903, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf904 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_64, reshape_47, linear_325], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf903, (s27, 896), (896, 1), 0), buf902, out=buf904)
            buf905 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_326], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_412, buf905, 28672, stream=stream0)
            del primals_412
            buf906 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_326], Original ATen: [aten.mm]
            extern_kernels.mm(buf904, buf905, out=buf906)
            buf907 = reinterpret_tensor(buf901, (1, s27, 896), (896*s27, 896, 1), 0); del buf901  # reuse
            buf908 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf909 = reinterpret_tensor(buf908, (1, s27, 1), (s27, 1, 1), 0); del buf908  # reuse
            buf910 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_327, linear_326, mul_251, result_328, hidden_states_155, hidden_states_156, pow_32, variance_31, add_205, rsqrt_31, hidden_states_157, to_287, hidden_states_158], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf907, buf909, buf871, buf906, primals_413, buf910, s27, 896, stream=stream0)
            buf911 = buf863; del buf863  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_156, hidden_states_157, to_287, hidden_states_158, result_330], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf910, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_414, (896, 4864), (1, 896), 0), out=buf911)
            buf912 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_328], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_415, buf912, 28672, stream=stream0)
            del primals_415
            buf913 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_328], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf910, (s27, 896), (896, 1), 0), buf912, out=buf913)
            buf914 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_329], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_416, buf914, 155648, stream=stream0)
            del primals_416
            buf915 = buf857; del buf857  # reuse
            # Topologically Sorted Source Nodes: [linear_329], Original ATen: [aten.mm]
            extern_kernels.mm(buf913, buf914, out=buf915)
            buf917 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_156, hidden_states_157, to_287, hidden_states_158, result_330, result_333], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf910, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_417, (896, 4864), (1, 896), 0), out=buf917)
            buf918 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_331], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_418, buf918, 28672, stream=stream0)
            del primals_418
            buf919 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_328, linear_331], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf910, (s27, 896), (896, 1), 0), buf918, out=buf919)
            buf920 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_332], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_419, buf920, 155648, stream=stream0)
            del primals_419
            buf921 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_332], Original ATen: [aten.mm]
            extern_kernels.mm(buf919, buf920, out=buf921)
            buf916 = reinterpret_tensor(buf911, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf911  # reuse
            buf922 = reinterpret_tensor(buf917, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf917  # reuse
            buf923 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_330, linear_329, mul_254, result_331, silu_15, result_333, linear_332, mul_255, result_334, mul_256], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf916, buf922, buf915, buf921, buf923, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf924 = buf906; del buf906  # reuse
            # Topologically Sorted Source Nodes: [silu_15, mul_256, result_336], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf923, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_420, (4864, 896), (1, 4864), 0), out=buf924)
            buf925 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_334], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_421, buf925, 155648, stream=stream0)
            del primals_421
            buf926 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_334], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf923, (s27, 4864), (4864, 1), 0), buf925, out=buf926)
            buf927 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_335], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_422, buf927, 28672, stream=stream0)
            del primals_422
            buf928 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_335], Original ATen: [aten.mm]
            extern_kernels.mm(buf926, buf927, out=buf928)
            buf929 = reinterpret_tensor(buf924, (1, s27, 896), (896*s27, 896, 1), 0); del buf924  # reuse
            buf930 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf931 = reinterpret_tensor(buf930, (1, s27, 1), (s27, 1, 1), 0); del buf930  # reuse
            buf932 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_336, linear_335, mul_257, result_337, hidden_states_159, hidden_states_160, pow_33, variance_32, add_210, rsqrt_32, hidden_states_161, to_295, hidden_states_162], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf929, buf931, buf907, buf928, primals_423, buf932, s27, 896, stream=stream0)
            buf933 = buf928; del buf928  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_160, hidden_states_161, to_295, hidden_states_162, result_339], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf932, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_424, (896, 896), (1, 896), 0), out=buf933)
            buf934 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_337], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_426, buf934, 28672, stream=stream0)
            del primals_426
            buf935 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_337], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf932, (s27, 896), (896, 1), 0), buf934, out=buf935)
            buf936 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_338], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_427, buf936, 28672, stream=stream0)
            del primals_427
            buf937 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_338], Original ATen: [aten.mm]
            extern_kernels.mm(buf935, buf936, out=buf937)
            buf938 = buf889; del buf889  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_160, hidden_states_161, to_295, hidden_states_162, result_339, result_342], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf932, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_428, (896, 128), (1, 896), 0), out=buf938)
            buf939 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_340], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_430, buf939, 28672, stream=stream0)
            del primals_430
            buf940 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_337, linear_340], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf932, (s27, 896), (896, 1), 0), buf939, out=buf940)
            buf941 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_341], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_431, buf941, 4096, stream=stream0)
            del primals_431
            buf942 = buf885; del buf885  # reuse
            # Topologically Sorted Source Nodes: [linear_341], Original ATen: [aten.mm]
            extern_kernels.mm(buf940, buf941, out=buf942)
            buf943 = reinterpret_tensor(buf893, (s27, 128), (128, 1), 0); del buf893  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_160, hidden_states_161, to_295, hidden_states_162, result_339, result_345], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf932, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_432, (896, 128), (1, 896), 0), out=buf943)
            buf944 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_343], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_434, buf944, 28672, stream=stream0)
            del primals_434
            buf945 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_337, linear_343], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf932, (s27, 896), (896, 1), 0), buf944, out=buf945)
            buf946 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_344], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_435, buf946, 4096, stream=stream0)
            del primals_435
            buf947 = reinterpret_tensor(buf892, (s27, 128), (128, 1), 0); del buf892  # reuse
            # Topologically Sorted Source Nodes: [linear_344], Original ATen: [aten.mm]
            extern_kernels.mm(buf945, buf946, out=buf947)
            buf948 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_339, linear_338, mul_260, result_340, view_48, query_states_16, x1_32, x2_32, neg_32, cat_33, mul_264], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_425, buf933, buf937, buf1, buf948, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf949 = reinterpret_tensor(buf933, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf933  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_339, linear_338, mul_260, result_340, view_48, query_states_16, mul_263, q_embed_16], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf949, primals_425, buf937, buf1, buf948, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_425
            buf950 = reinterpret_tensor(buf884, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf884  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_342, linear_341, mul_261, result_343, view_49, key_states_16, x1_33, x2_33, neg_33, cat_34, mul_266], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_429, buf938, buf942, buf1, buf950, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf951 = reinterpret_tensor(buf938, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf938  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_342, linear_341, mul_261, result_343, view_49, key_states_16, mul_265, k_embed_16], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf951, primals_429, buf942, buf1, buf950, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_429
            buf952 = reinterpret_tensor(buf948, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf948  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_342, linear_341, mul_261, result_343, view_49, key_states_16, mul_265, k_embed_16, getitem_719, hidden_states_163, key_16], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf951, buf952, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf953 = reinterpret_tensor(buf937, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf937  # reuse
            # Topologically Sorted Source Nodes: [result_345, linear_344, mul_262, result_346, view_50, value_states_16, getitem_724, hidden_states_164, value_16], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_433, buf943, buf947, buf953, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_433
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_342, linear_341, mul_261, result_343, view_49, key_states_16, result_345, linear_344, mul_262, result_346, view_50, value_states_16, mul_265, k_embed_16, getitem_719, hidden_states_163, key_16, getitem_724, hidden_states_164, value_16, attn_output_48], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf954 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf949, reinterpret_tensor(buf952, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf953, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf955 = buf954[0]
            assert_size_stride(buf955, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf955, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf956 = buf954[1]
            assert_size_stride(buf956, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf956, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf957 = buf954[2]
            assert_size_stride(buf957, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf957, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf958 = buf954[3]
            assert_size_stride(buf958, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf958, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf954
            buf959 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_68, reshape_50, result_348], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf955, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_436, (896, 896), (1, 896), 0), out=buf959)
            buf960 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_346], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_437, buf960, 28672, stream=stream0)
            del primals_437
            buf961 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_68, reshape_50, linear_346], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf955, buf961, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf962 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_68, reshape_50, linear_346], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf961, (s27, 896), (896, 1), 0), buf960, out=buf962)
            buf963 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_347], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_438, buf963, 28672, stream=stream0)
            del primals_438
            buf964 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_347], Original ATen: [aten.mm]
            extern_kernels.mm(buf962, buf963, out=buf964)
            buf965 = reinterpret_tensor(buf959, (1, s27, 896), (896*s27, 896, 1), 0); del buf959  # reuse
            buf966 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf967 = reinterpret_tensor(buf966, (1, s27, 1), (s27, 1, 1), 0); del buf966  # reuse
            buf968 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_348, linear_347, mul_267, result_349, hidden_states_165, hidden_states_166, pow_34, variance_33, add_218, rsqrt_33, hidden_states_167, to_305, hidden_states_168], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf965, buf967, buf929, buf964, primals_439, buf968, s27, 896, stream=stream0)
            buf969 = buf921; del buf921  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_166, hidden_states_167, to_305, hidden_states_168, result_351], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf968, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_440, (896, 4864), (1, 896), 0), out=buf969)
            buf970 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_349], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_441, buf970, 28672, stream=stream0)
            del primals_441
            buf971 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_349], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf968, (s27, 896), (896, 1), 0), buf970, out=buf971)
            buf972 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_350], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_442, buf972, 155648, stream=stream0)
            del primals_442
            buf973 = buf915; del buf915  # reuse
            # Topologically Sorted Source Nodes: [linear_350], Original ATen: [aten.mm]
            extern_kernels.mm(buf971, buf972, out=buf973)
            buf975 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_166, hidden_states_167, to_305, hidden_states_168, result_351, result_354], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf968, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_443, (896, 4864), (1, 896), 0), out=buf975)
            buf976 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_352], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_444, buf976, 28672, stream=stream0)
            del primals_444
            buf977 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_349, linear_352], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf968, (s27, 896), (896, 1), 0), buf976, out=buf977)
            buf978 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_353], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_445, buf978, 155648, stream=stream0)
            del primals_445
            buf979 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_353], Original ATen: [aten.mm]
            extern_kernels.mm(buf977, buf978, out=buf979)
            buf974 = reinterpret_tensor(buf969, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf969  # reuse
            buf980 = reinterpret_tensor(buf975, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf975  # reuse
            buf981 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_351, linear_350, mul_270, result_352, silu_16, result_354, linear_353, mul_271, result_355, mul_272], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf974, buf980, buf973, buf979, buf981, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf982 = buf964; del buf964  # reuse
            # Topologically Sorted Source Nodes: [silu_16, mul_272, result_357], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf981, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_446, (4864, 896), (1, 4864), 0), out=buf982)
            buf983 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_355], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_447, buf983, 155648, stream=stream0)
            del primals_447
            buf984 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_355], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf981, (s27, 4864), (4864, 1), 0), buf983, out=buf984)
            buf985 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_356], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_448, buf985, 28672, stream=stream0)
            del primals_448
            buf986 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_356], Original ATen: [aten.mm]
            extern_kernels.mm(buf984, buf985, out=buf986)
            buf987 = reinterpret_tensor(buf982, (1, s27, 896), (896*s27, 896, 1), 0); del buf982  # reuse
            buf988 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf989 = reinterpret_tensor(buf988, (1, s27, 1), (s27, 1, 1), 0); del buf988  # reuse
            buf990 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_357, linear_356, mul_273, result_358, hidden_states_169, hidden_states_170, pow_35, variance_34, add_223, rsqrt_34, hidden_states_171, to_313, hidden_states_172], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf987, buf989, buf965, buf986, primals_449, buf990, s27, 896, stream=stream0)
            buf991 = buf986; del buf986  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_170, hidden_states_171, to_313, hidden_states_172, result_360], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf990, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_450, (896, 896), (1, 896), 0), out=buf991)
            buf992 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_358], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_452, buf992, 28672, stream=stream0)
            del primals_452
            buf993 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_358], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf990, (s27, 896), (896, 1), 0), buf992, out=buf993)
            buf994 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_359], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_453, buf994, 28672, stream=stream0)
            del primals_453
            buf995 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_359], Original ATen: [aten.mm]
            extern_kernels.mm(buf993, buf994, out=buf995)
            buf996 = buf947; del buf947  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_170, hidden_states_171, to_313, hidden_states_172, result_360, result_363], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf990, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_454, (896, 128), (1, 896), 0), out=buf996)
            buf997 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_361], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_456, buf997, 28672, stream=stream0)
            del primals_456
            buf998 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_358, linear_361], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf990, (s27, 896), (896, 1), 0), buf997, out=buf998)
            buf999 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_362], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_457, buf999, 4096, stream=stream0)
            del primals_457
            buf1000 = buf943; del buf943  # reuse
            # Topologically Sorted Source Nodes: [linear_362], Original ATen: [aten.mm]
            extern_kernels.mm(buf998, buf999, out=buf1000)
            buf1001 = reinterpret_tensor(buf951, (s27, 128), (128, 1), 0); del buf951  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_170, hidden_states_171, to_313, hidden_states_172, result_360, result_366], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf990, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_458, (896, 128), (1, 896), 0), out=buf1001)
            buf1002 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_364], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_460, buf1002, 28672, stream=stream0)
            del primals_460
            buf1003 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_358, linear_364], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf990, (s27, 896), (896, 1), 0), buf1002, out=buf1003)
            buf1004 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_365], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_461, buf1004, 4096, stream=stream0)
            del primals_461
            buf1005 = reinterpret_tensor(buf950, (s27, 128), (128, 1), 0); del buf950  # reuse
            # Topologically Sorted Source Nodes: [linear_365], Original ATen: [aten.mm]
            extern_kernels.mm(buf1003, buf1004, out=buf1005)
            buf1006 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_360, linear_359, mul_276, result_361, view_51, query_states_17, x1_34, x2_34, neg_34, cat_35, mul_280], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_451, buf991, buf995, buf1, buf1006, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf1007 = reinterpret_tensor(buf991, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf991  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_360, linear_359, mul_276, result_361, view_51, query_states_17, mul_279, q_embed_17], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf1007, primals_451, buf995, buf1, buf1006, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_451
            buf1008 = reinterpret_tensor(buf942, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf942  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_363, linear_362, mul_277, result_364, view_52, key_states_17, x1_35, x2_35, neg_35, cat_36, mul_282], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_455, buf996, buf1000, buf1, buf1008, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf1009 = reinterpret_tensor(buf996, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf996  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_363, linear_362, mul_277, result_364, view_52, key_states_17, mul_281, k_embed_17], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf1009, primals_455, buf1000, buf1, buf1008, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_455
            buf1010 = reinterpret_tensor(buf995, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf995  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_363, linear_362, mul_277, result_364, view_52, key_states_17, mul_281, k_embed_17, getitem_761, hidden_states_173, key_17], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf1009, buf1010, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf1011 = reinterpret_tensor(buf1006, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1006  # reuse
            # Topologically Sorted Source Nodes: [result_366, linear_365, mul_278, result_367, view_53, value_states_17, getitem_766, hidden_states_174, value_17], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_459, buf1001, buf1005, buf1011, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_459
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_363, linear_362, mul_277, result_364, view_52, key_states_17, result_366, linear_365, mul_278, result_367, view_53, value_states_17, mul_281, k_embed_17, getitem_761, hidden_states_173, key_17, getitem_766, hidden_states_174, value_17, attn_output_51], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf1012 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf1007, reinterpret_tensor(buf1010, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1011, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf1013 = buf1012[0]
            assert_size_stride(buf1013, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1013, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1014 = buf1012[1]
            assert_size_stride(buf1014, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1014, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1015 = buf1012[2]
            assert_size_stride(buf1015, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1015, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1016 = buf1012[3]
            assert_size_stride(buf1016, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1016, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf1012
            buf1017 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_72, reshape_53, result_369], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1013, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_462, (896, 896), (1, 896), 0), out=buf1017)
            buf1018 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_367], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_463, buf1018, 28672, stream=stream0)
            del primals_463
            buf1019 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_72, reshape_53, linear_367], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf1013, buf1019, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf1020 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_72, reshape_53, linear_367], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1019, (s27, 896), (896, 1), 0), buf1018, out=buf1020)
            buf1021 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_368], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_464, buf1021, 28672, stream=stream0)
            del primals_464
            buf1022 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_368], Original ATen: [aten.mm]
            extern_kernels.mm(buf1020, buf1021, out=buf1022)
            buf1023 = reinterpret_tensor(buf1017, (1, s27, 896), (896*s27, 896, 1), 0); del buf1017  # reuse
            buf1024 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1025 = reinterpret_tensor(buf1024, (1, s27, 1), (s27, 1, 1), 0); del buf1024  # reuse
            buf1026 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_369, linear_368, mul_283, result_370, hidden_states_175, hidden_states_176, pow_36, variance_35, add_231, rsqrt_35, hidden_states_177, to_323, hidden_states_178], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1023, buf1025, buf987, buf1022, primals_465, buf1026, s27, 896, stream=stream0)
            buf1027 = buf979; del buf979  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_176, hidden_states_177, to_323, hidden_states_178, result_372], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1026, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_466, (896, 4864), (1, 896), 0), out=buf1027)
            buf1028 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_370], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_467, buf1028, 28672, stream=stream0)
            del primals_467
            buf1029 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_370], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1026, (s27, 896), (896, 1), 0), buf1028, out=buf1029)
            buf1030 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_371], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_468, buf1030, 155648, stream=stream0)
            del primals_468
            buf1031 = buf973; del buf973  # reuse
            # Topologically Sorted Source Nodes: [linear_371], Original ATen: [aten.mm]
            extern_kernels.mm(buf1029, buf1030, out=buf1031)
            buf1033 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_176, hidden_states_177, to_323, hidden_states_178, result_372, result_375], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1026, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_469, (896, 4864), (1, 896), 0), out=buf1033)
            buf1034 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_373], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_470, buf1034, 28672, stream=stream0)
            del primals_470
            buf1035 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_370, linear_373], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1026, (s27, 896), (896, 1), 0), buf1034, out=buf1035)
            buf1036 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_374], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_471, buf1036, 155648, stream=stream0)
            del primals_471
            buf1037 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_374], Original ATen: [aten.mm]
            extern_kernels.mm(buf1035, buf1036, out=buf1037)
            buf1032 = reinterpret_tensor(buf1027, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1027  # reuse
            buf1038 = reinterpret_tensor(buf1033, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1033  # reuse
            buf1039 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_372, linear_371, mul_286, result_373, silu_17, result_375, linear_374, mul_287, result_376, mul_288], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf1032, buf1038, buf1031, buf1037, buf1039, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf1040 = buf1022; del buf1022  # reuse
            # Topologically Sorted Source Nodes: [silu_17, mul_288, result_378], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1039, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_472, (4864, 896), (1, 4864), 0), out=buf1040)
            buf1041 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_376], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_473, buf1041, 155648, stream=stream0)
            del primals_473
            buf1042 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_376], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1039, (s27, 4864), (4864, 1), 0), buf1041, out=buf1042)
            buf1043 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_377], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_474, buf1043, 28672, stream=stream0)
            del primals_474
            buf1044 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_377], Original ATen: [aten.mm]
            extern_kernels.mm(buf1042, buf1043, out=buf1044)
            buf1045 = reinterpret_tensor(buf1040, (1, s27, 896), (896*s27, 896, 1), 0); del buf1040  # reuse
            buf1046 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1047 = reinterpret_tensor(buf1046, (1, s27, 1), (s27, 1, 1), 0); del buf1046  # reuse
            buf1048 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_378, linear_377, mul_289, result_379, hidden_states_179, hidden_states_180, pow_37, variance_36, add_236, rsqrt_36, hidden_states_181, to_331, hidden_states_182], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1045, buf1047, buf1023, buf1044, primals_475, buf1048, s27, 896, stream=stream0)
            buf1049 = buf1044; del buf1044  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_180, hidden_states_181, to_331, hidden_states_182, result_381], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1048, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_476, (896, 896), (1, 896), 0), out=buf1049)
            buf1050 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_379], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_478, buf1050, 28672, stream=stream0)
            del primals_478
            buf1051 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_379], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1048, (s27, 896), (896, 1), 0), buf1050, out=buf1051)
            buf1052 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_380], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_479, buf1052, 28672, stream=stream0)
            del primals_479
            buf1053 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_380], Original ATen: [aten.mm]
            extern_kernels.mm(buf1051, buf1052, out=buf1053)
            buf1054 = buf1005; del buf1005  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_180, hidden_states_181, to_331, hidden_states_182, result_381, result_384], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1048, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_480, (896, 128), (1, 896), 0), out=buf1054)
            buf1055 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_382], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_482, buf1055, 28672, stream=stream0)
            del primals_482
            buf1056 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_379, linear_382], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1048, (s27, 896), (896, 1), 0), buf1055, out=buf1056)
            buf1057 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_383], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_483, buf1057, 4096, stream=stream0)
            del primals_483
            buf1058 = buf1001; del buf1001  # reuse
            # Topologically Sorted Source Nodes: [linear_383], Original ATen: [aten.mm]
            extern_kernels.mm(buf1056, buf1057, out=buf1058)
            buf1059 = reinterpret_tensor(buf1009, (s27, 128), (128, 1), 0); del buf1009  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_180, hidden_states_181, to_331, hidden_states_182, result_381, result_387], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1048, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_484, (896, 128), (1, 896), 0), out=buf1059)
            buf1060 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_385], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_486, buf1060, 28672, stream=stream0)
            del primals_486
            buf1061 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_379, linear_385], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1048, (s27, 896), (896, 1), 0), buf1060, out=buf1061)
            buf1062 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_386], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_487, buf1062, 4096, stream=stream0)
            del primals_487
            buf1063 = reinterpret_tensor(buf1008, (s27, 128), (128, 1), 0); del buf1008  # reuse
            # Topologically Sorted Source Nodes: [linear_386], Original ATen: [aten.mm]
            extern_kernels.mm(buf1061, buf1062, out=buf1063)
            buf1064 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_381, linear_380, mul_292, result_382, view_54, query_states_18, x1_36, x2_36, neg_36, cat_37, mul_296], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_477, buf1049, buf1053, buf1, buf1064, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf1065 = reinterpret_tensor(buf1049, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf1049  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_381, linear_380, mul_292, result_382, view_54, query_states_18, mul_295, q_embed_18], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf1065, primals_477, buf1053, buf1, buf1064, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_477
            buf1066 = reinterpret_tensor(buf1000, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf1000  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_384, linear_383, mul_293, result_385, view_55, key_states_18, x1_37, x2_37, neg_37, cat_38, mul_298], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_481, buf1054, buf1058, buf1, buf1066, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf1067 = reinterpret_tensor(buf1054, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf1054  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_384, linear_383, mul_293, result_385, view_55, key_states_18, mul_297, k_embed_18], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf1067, primals_481, buf1058, buf1, buf1066, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_481
            buf1068 = reinterpret_tensor(buf1064, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1064  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_384, linear_383, mul_293, result_385, view_55, key_states_18, mul_297, k_embed_18, getitem_803, hidden_states_183, key_18], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf1067, buf1068, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf1069 = reinterpret_tensor(buf1053, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1053  # reuse
            # Topologically Sorted Source Nodes: [result_387, linear_386, mul_294, result_388, view_56, value_states_18, getitem_808, hidden_states_184, value_18], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_485, buf1059, buf1063, buf1069, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_485
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_384, linear_383, mul_293, result_385, view_55, key_states_18, result_387, linear_386, mul_294, result_388, view_56, value_states_18, mul_297, k_embed_18, getitem_803, hidden_states_183, key_18, getitem_808, hidden_states_184, value_18, attn_output_54], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf1070 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf1065, reinterpret_tensor(buf1068, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1069, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf1071 = buf1070[0]
            assert_size_stride(buf1071, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1071, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1072 = buf1070[1]
            assert_size_stride(buf1072, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1072, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1073 = buf1070[2]
            assert_size_stride(buf1073, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1073, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1074 = buf1070[3]
            assert_size_stride(buf1074, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1074, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf1070
            buf1075 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_76, reshape_56, result_390], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1071, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_488, (896, 896), (1, 896), 0), out=buf1075)
            buf1076 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_388], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_489, buf1076, 28672, stream=stream0)
            del primals_489
            buf1077 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_76, reshape_56, linear_388], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf1071, buf1077, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf1078 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_76, reshape_56, linear_388], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1077, (s27, 896), (896, 1), 0), buf1076, out=buf1078)
            buf1079 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_389], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_490, buf1079, 28672, stream=stream0)
            del primals_490
            buf1080 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_389], Original ATen: [aten.mm]
            extern_kernels.mm(buf1078, buf1079, out=buf1080)
            buf1081 = reinterpret_tensor(buf1075, (1, s27, 896), (896*s27, 896, 1), 0); del buf1075  # reuse
            buf1082 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1083 = reinterpret_tensor(buf1082, (1, s27, 1), (s27, 1, 1), 0); del buf1082  # reuse
            buf1084 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_390, linear_389, mul_299, result_391, hidden_states_185, hidden_states_186, pow_38, variance_37, add_244, rsqrt_37, hidden_states_187, to_341, hidden_states_188], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1081, buf1083, buf1045, buf1080, primals_491, buf1084, s27, 896, stream=stream0)
            buf1085 = buf1037; del buf1037  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_186, hidden_states_187, to_341, hidden_states_188, result_393], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1084, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_492, (896, 4864), (1, 896), 0), out=buf1085)
            buf1086 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_391], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_493, buf1086, 28672, stream=stream0)
            del primals_493
            buf1087 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_391], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1084, (s27, 896), (896, 1), 0), buf1086, out=buf1087)
            buf1088 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_392], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_494, buf1088, 155648, stream=stream0)
            del primals_494
            buf1089 = buf1031; del buf1031  # reuse
            # Topologically Sorted Source Nodes: [linear_392], Original ATen: [aten.mm]
            extern_kernels.mm(buf1087, buf1088, out=buf1089)
            buf1091 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_186, hidden_states_187, to_341, hidden_states_188, result_393, result_396], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1084, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_495, (896, 4864), (1, 896), 0), out=buf1091)
            buf1092 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_394], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_496, buf1092, 28672, stream=stream0)
            del primals_496
            buf1093 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_391, linear_394], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1084, (s27, 896), (896, 1), 0), buf1092, out=buf1093)
            buf1094 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_395], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_497, buf1094, 155648, stream=stream0)
            del primals_497
            buf1095 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_395], Original ATen: [aten.mm]
            extern_kernels.mm(buf1093, buf1094, out=buf1095)
            buf1090 = reinterpret_tensor(buf1085, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1085  # reuse
            buf1096 = reinterpret_tensor(buf1091, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1091  # reuse
            buf1097 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_393, linear_392, mul_302, result_394, silu_18, result_396, linear_395, mul_303, result_397, mul_304], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf1090, buf1096, buf1089, buf1095, buf1097, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf1098 = buf1080; del buf1080  # reuse
            # Topologically Sorted Source Nodes: [silu_18, mul_304, result_399], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1097, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_498, (4864, 896), (1, 4864), 0), out=buf1098)
            buf1099 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_397], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_499, buf1099, 155648, stream=stream0)
            del primals_499
            buf1100 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_397], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1097, (s27, 4864), (4864, 1), 0), buf1099, out=buf1100)
            buf1101 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_398], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_500, buf1101, 28672, stream=stream0)
            del primals_500
            buf1102 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_398], Original ATen: [aten.mm]
            extern_kernels.mm(buf1100, buf1101, out=buf1102)
            buf1103 = reinterpret_tensor(buf1098, (1, s27, 896), (896*s27, 896, 1), 0); del buf1098  # reuse
            buf1104 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1105 = reinterpret_tensor(buf1104, (1, s27, 1), (s27, 1, 1), 0); del buf1104  # reuse
            buf1106 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_399, linear_398, mul_305, result_400, hidden_states_189, hidden_states_190, pow_39, variance_38, add_249, rsqrt_38, hidden_states_191, to_349, hidden_states_192], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1103, buf1105, buf1081, buf1102, primals_501, buf1106, s27, 896, stream=stream0)
            buf1107 = buf1102; del buf1102  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_191, to_349, hidden_states_192, result_402], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1106, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_502, (896, 896), (1, 896), 0), out=buf1107)
            buf1108 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_400], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_504, buf1108, 28672, stream=stream0)
            del primals_504
            buf1109 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_400], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1106, (s27, 896), (896, 1), 0), buf1108, out=buf1109)
            buf1110 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_401], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_505, buf1110, 28672, stream=stream0)
            del primals_505
            buf1111 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_401], Original ATen: [aten.mm]
            extern_kernels.mm(buf1109, buf1110, out=buf1111)
            buf1112 = buf1063; del buf1063  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_191, to_349, hidden_states_192, result_402, result_405], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1106, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_506, (896, 128), (1, 896), 0), out=buf1112)
            buf1113 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_403], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_508, buf1113, 28672, stream=stream0)
            del primals_508
            buf1114 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_400, linear_403], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1106, (s27, 896), (896, 1), 0), buf1113, out=buf1114)
            buf1115 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_404], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_509, buf1115, 4096, stream=stream0)
            del primals_509
            buf1116 = buf1059; del buf1059  # reuse
            # Topologically Sorted Source Nodes: [linear_404], Original ATen: [aten.mm]
            extern_kernels.mm(buf1114, buf1115, out=buf1116)
            buf1117 = reinterpret_tensor(buf1067, (s27, 128), (128, 1), 0); del buf1067  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_190, hidden_states_191, to_349, hidden_states_192, result_402, result_408], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1106, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_510, (896, 128), (1, 896), 0), out=buf1117)
            buf1118 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_406], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_512, buf1118, 28672, stream=stream0)
            del primals_512
            buf1119 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_400, linear_406], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1106, (s27, 896), (896, 1), 0), buf1118, out=buf1119)
            buf1120 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_407], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_513, buf1120, 4096, stream=stream0)
            del primals_513
            buf1121 = reinterpret_tensor(buf1066, (s27, 128), (128, 1), 0); del buf1066  # reuse
            # Topologically Sorted Source Nodes: [linear_407], Original ATen: [aten.mm]
            extern_kernels.mm(buf1119, buf1120, out=buf1121)
            buf1122 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_402, linear_401, mul_308, result_403, view_57, query_states_19, x1_38, x2_38, neg_38, cat_39, mul_312], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_503, buf1107, buf1111, buf1, buf1122, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf1123 = reinterpret_tensor(buf1107, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf1107  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_402, linear_401, mul_308, result_403, view_57, query_states_19, mul_311, q_embed_19], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf1123, primals_503, buf1111, buf1, buf1122, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_503
            buf1124 = reinterpret_tensor(buf1058, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf1058  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_405, linear_404, mul_309, result_406, view_58, key_states_19, x1_39, x2_39, neg_39, cat_40, mul_314], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_507, buf1112, buf1116, buf1, buf1124, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf1125 = reinterpret_tensor(buf1112, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf1112  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_405, linear_404, mul_309, result_406, view_58, key_states_19, mul_313, k_embed_19], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf1125, primals_507, buf1116, buf1, buf1124, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_507
            buf1126 = reinterpret_tensor(buf1122, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1122  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_405, linear_404, mul_309, result_406, view_58, key_states_19, mul_313, k_embed_19, getitem_845, hidden_states_193, key_19], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf1125, buf1126, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf1127 = reinterpret_tensor(buf1111, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1111  # reuse
            # Topologically Sorted Source Nodes: [result_408, linear_407, mul_310, result_409, view_59, value_states_19, getitem_850, hidden_states_194, value_19], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_511, buf1117, buf1121, buf1127, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_511
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_405, linear_404, mul_309, result_406, view_58, key_states_19, result_408, linear_407, mul_310, result_409, view_59, value_states_19, mul_313, k_embed_19, getitem_845, hidden_states_193, key_19, getitem_850, hidden_states_194, value_19, attn_output_57], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf1128 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf1123, reinterpret_tensor(buf1126, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1127, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf1129 = buf1128[0]
            assert_size_stride(buf1129, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1129, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1130 = buf1128[1]
            assert_size_stride(buf1130, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1130, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1131 = buf1128[2]
            assert_size_stride(buf1131, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1131, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1132 = buf1128[3]
            assert_size_stride(buf1132, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1132, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf1128
            buf1133 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_80, reshape_59, result_411], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1129, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_514, (896, 896), (1, 896), 0), out=buf1133)
            buf1134 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_409], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_515, buf1134, 28672, stream=stream0)
            del primals_515
            buf1135 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_80, reshape_59, linear_409], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf1129, buf1135, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf1136 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_80, reshape_59, linear_409], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1135, (s27, 896), (896, 1), 0), buf1134, out=buf1136)
            buf1137 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_410], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_516, buf1137, 28672, stream=stream0)
            del primals_516
            buf1138 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_410], Original ATen: [aten.mm]
            extern_kernels.mm(buf1136, buf1137, out=buf1138)
            buf1139 = reinterpret_tensor(buf1133, (1, s27, 896), (896*s27, 896, 1), 0); del buf1133  # reuse
            buf1140 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1141 = reinterpret_tensor(buf1140, (1, s27, 1), (s27, 1, 1), 0); del buf1140  # reuse
            buf1142 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_411, linear_410, mul_315, result_412, hidden_states_195, hidden_states_196, pow_40, variance_39, add_257, rsqrt_39, hidden_states_197, to_359, hidden_states_198], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1139, buf1141, buf1103, buf1138, primals_517, buf1142, s27, 896, stream=stream0)
            buf1143 = buf1095; del buf1095  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_196, hidden_states_197, to_359, hidden_states_198, result_414], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1142, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_518, (896, 4864), (1, 896), 0), out=buf1143)
            buf1144 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_412], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_519, buf1144, 28672, stream=stream0)
            del primals_519
            buf1145 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_412], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1142, (s27, 896), (896, 1), 0), buf1144, out=buf1145)
            buf1146 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_413], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_520, buf1146, 155648, stream=stream0)
            del primals_520
            buf1147 = buf1089; del buf1089  # reuse
            # Topologically Sorted Source Nodes: [linear_413], Original ATen: [aten.mm]
            extern_kernels.mm(buf1145, buf1146, out=buf1147)
            buf1149 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_196, hidden_states_197, to_359, hidden_states_198, result_414, result_417], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1142, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_521, (896, 4864), (1, 896), 0), out=buf1149)
            buf1150 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_415], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_522, buf1150, 28672, stream=stream0)
            del primals_522
            buf1151 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_412, linear_415], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1142, (s27, 896), (896, 1), 0), buf1150, out=buf1151)
            buf1152 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_416], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_523, buf1152, 155648, stream=stream0)
            del primals_523
            buf1153 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_416], Original ATen: [aten.mm]
            extern_kernels.mm(buf1151, buf1152, out=buf1153)
            buf1148 = reinterpret_tensor(buf1143, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1143  # reuse
            buf1154 = reinterpret_tensor(buf1149, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1149  # reuse
            buf1155 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_414, linear_413, mul_318, result_415, silu_19, result_417, linear_416, mul_319, result_418, mul_320], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf1148, buf1154, buf1147, buf1153, buf1155, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf1156 = buf1138; del buf1138  # reuse
            # Topologically Sorted Source Nodes: [silu_19, mul_320, result_420], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1155, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_524, (4864, 896), (1, 4864), 0), out=buf1156)
            buf1157 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_418], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_525, buf1157, 155648, stream=stream0)
            del primals_525
            buf1158 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_418], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1155, (s27, 4864), (4864, 1), 0), buf1157, out=buf1158)
            buf1159 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_419], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_526, buf1159, 28672, stream=stream0)
            del primals_526
            buf1160 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_419], Original ATen: [aten.mm]
            extern_kernels.mm(buf1158, buf1159, out=buf1160)
            buf1161 = reinterpret_tensor(buf1156, (1, s27, 896), (896*s27, 896, 1), 0); del buf1156  # reuse
            buf1162 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1163 = reinterpret_tensor(buf1162, (1, s27, 1), (s27, 1, 1), 0); del buf1162  # reuse
            buf1164 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_420, linear_419, mul_321, result_421, hidden_states_199, hidden_states_200, pow_41, variance_40, add_262, rsqrt_40, hidden_states_201, to_367, hidden_states_202], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1161, buf1163, buf1139, buf1160, primals_527, buf1164, s27, 896, stream=stream0)
            buf1165 = buf1160; del buf1160  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_200, hidden_states_201, to_367, hidden_states_202, result_423], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1164, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_528, (896, 896), (1, 896), 0), out=buf1165)
            buf1166 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_421], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_530, buf1166, 28672, stream=stream0)
            del primals_530
            buf1167 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_421], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1164, (s27, 896), (896, 1), 0), buf1166, out=buf1167)
            buf1168 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_422], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_531, buf1168, 28672, stream=stream0)
            del primals_531
            buf1169 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_422], Original ATen: [aten.mm]
            extern_kernels.mm(buf1167, buf1168, out=buf1169)
            buf1170 = buf1121; del buf1121  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_200, hidden_states_201, to_367, hidden_states_202, result_423, result_426], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1164, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_532, (896, 128), (1, 896), 0), out=buf1170)
            buf1171 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_424], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_534, buf1171, 28672, stream=stream0)
            del primals_534
            buf1172 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_421, linear_424], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1164, (s27, 896), (896, 1), 0), buf1171, out=buf1172)
            buf1173 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_425], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_535, buf1173, 4096, stream=stream0)
            del primals_535
            buf1174 = buf1117; del buf1117  # reuse
            # Topologically Sorted Source Nodes: [linear_425], Original ATen: [aten.mm]
            extern_kernels.mm(buf1172, buf1173, out=buf1174)
            buf1175 = reinterpret_tensor(buf1125, (s27, 128), (128, 1), 0); del buf1125  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_200, hidden_states_201, to_367, hidden_states_202, result_423, result_429], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1164, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_536, (896, 128), (1, 896), 0), out=buf1175)
            buf1176 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_427], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_538, buf1176, 28672, stream=stream0)
            del primals_538
            buf1177 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_421, linear_427], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1164, (s27, 896), (896, 1), 0), buf1176, out=buf1177)
            buf1178 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_428], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_539, buf1178, 4096, stream=stream0)
            del primals_539
            buf1179 = reinterpret_tensor(buf1124, (s27, 128), (128, 1), 0); del buf1124  # reuse
            # Topologically Sorted Source Nodes: [linear_428], Original ATen: [aten.mm]
            extern_kernels.mm(buf1177, buf1178, out=buf1179)
            buf1180 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_423, linear_422, mul_324, result_424, view_60, query_states_20, x1_40, x2_40, neg_40, cat_41, mul_328], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_529, buf1165, buf1169, buf1, buf1180, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf1181 = reinterpret_tensor(buf1165, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf1165  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_423, linear_422, mul_324, result_424, view_60, query_states_20, mul_327, q_embed_20], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf1181, primals_529, buf1169, buf1, buf1180, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_529
            buf1182 = reinterpret_tensor(buf1116, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf1116  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_426, linear_425, mul_325, result_427, view_61, key_states_20, x1_41, x2_41, neg_41, cat_42, mul_330], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_533, buf1170, buf1174, buf1, buf1182, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf1183 = reinterpret_tensor(buf1170, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf1170  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_426, linear_425, mul_325, result_427, view_61, key_states_20, mul_329, k_embed_20], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf1183, primals_533, buf1174, buf1, buf1182, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_533
            buf1184 = reinterpret_tensor(buf1180, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1180  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_426, linear_425, mul_325, result_427, view_61, key_states_20, mul_329, k_embed_20, getitem_887, hidden_states_203, key_20], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf1183, buf1184, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf1185 = reinterpret_tensor(buf1169, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1169  # reuse
            # Topologically Sorted Source Nodes: [result_429, linear_428, mul_326, result_430, view_62, value_states_20, getitem_892, hidden_states_204, value_20], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_537, buf1175, buf1179, buf1185, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_537
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_426, linear_425, mul_325, result_427, view_61, key_states_20, result_429, linear_428, mul_326, result_430, view_62, value_states_20, mul_329, k_embed_20, getitem_887, hidden_states_203, key_20, getitem_892, hidden_states_204, value_20, attn_output_60], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf1186 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf1181, reinterpret_tensor(buf1184, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1185, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf1187 = buf1186[0]
            assert_size_stride(buf1187, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1187, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1188 = buf1186[1]
            assert_size_stride(buf1188, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1188, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1189 = buf1186[2]
            assert_size_stride(buf1189, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1189, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1190 = buf1186[3]
            assert_size_stride(buf1190, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1190, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf1186
            buf1191 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_84, reshape_62, result_432], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1187, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_540, (896, 896), (1, 896), 0), out=buf1191)
            buf1192 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_430], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_541, buf1192, 28672, stream=stream0)
            del primals_541
            buf1193 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_84, reshape_62, linear_430], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf1187, buf1193, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf1194 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_84, reshape_62, linear_430], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1193, (s27, 896), (896, 1), 0), buf1192, out=buf1194)
            buf1195 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_431], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_542, buf1195, 28672, stream=stream0)
            del primals_542
            buf1196 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_431], Original ATen: [aten.mm]
            extern_kernels.mm(buf1194, buf1195, out=buf1196)
            buf1197 = reinterpret_tensor(buf1191, (1, s27, 896), (896*s27, 896, 1), 0); del buf1191  # reuse
            buf1198 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1199 = reinterpret_tensor(buf1198, (1, s27, 1), (s27, 1, 1), 0); del buf1198  # reuse
            buf1200 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_432, linear_431, mul_331, result_433, hidden_states_205, hidden_states_206, pow_42, variance_41, add_270, rsqrt_41, hidden_states_207, to_377, hidden_states_208], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1197, buf1199, buf1161, buf1196, primals_543, buf1200, s27, 896, stream=stream0)
            buf1201 = buf1153; del buf1153  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_206, hidden_states_207, to_377, hidden_states_208, result_435], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1200, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_544, (896, 4864), (1, 896), 0), out=buf1201)
            buf1202 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_433], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_545, buf1202, 28672, stream=stream0)
            del primals_545
            buf1203 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_433], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1200, (s27, 896), (896, 1), 0), buf1202, out=buf1203)
            buf1204 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_434], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_546, buf1204, 155648, stream=stream0)
            del primals_546
            buf1205 = buf1147; del buf1147  # reuse
            # Topologically Sorted Source Nodes: [linear_434], Original ATen: [aten.mm]
            extern_kernels.mm(buf1203, buf1204, out=buf1205)
            buf1207 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_206, hidden_states_207, to_377, hidden_states_208, result_435, result_438], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1200, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_547, (896, 4864), (1, 896), 0), out=buf1207)
            buf1208 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_436], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_548, buf1208, 28672, stream=stream0)
            del primals_548
            buf1209 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_433, linear_436], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1200, (s27, 896), (896, 1), 0), buf1208, out=buf1209)
            buf1210 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_437], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_549, buf1210, 155648, stream=stream0)
            del primals_549
            buf1211 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_437], Original ATen: [aten.mm]
            extern_kernels.mm(buf1209, buf1210, out=buf1211)
            buf1206 = reinterpret_tensor(buf1201, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1201  # reuse
            buf1212 = reinterpret_tensor(buf1207, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1207  # reuse
            buf1213 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_435, linear_434, mul_334, result_436, silu_20, result_438, linear_437, mul_335, result_439, mul_336], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf1206, buf1212, buf1205, buf1211, buf1213, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf1214 = buf1196; del buf1196  # reuse
            # Topologically Sorted Source Nodes: [silu_20, mul_336, result_441], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1213, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_550, (4864, 896), (1, 4864), 0), out=buf1214)
            buf1215 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_439], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_551, buf1215, 155648, stream=stream0)
            del primals_551
            buf1216 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_439], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1213, (s27, 4864), (4864, 1), 0), buf1215, out=buf1216)
            buf1217 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_440], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_552, buf1217, 28672, stream=stream0)
            del primals_552
            buf1218 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_440], Original ATen: [aten.mm]
            extern_kernels.mm(buf1216, buf1217, out=buf1218)
            buf1219 = reinterpret_tensor(buf1214, (1, s27, 896), (896*s27, 896, 1), 0); del buf1214  # reuse
            buf1220 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1221 = reinterpret_tensor(buf1220, (1, s27, 1), (s27, 1, 1), 0); del buf1220  # reuse
            buf1222 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_441, linear_440, mul_337, result_442, hidden_states_209, hidden_states_210, pow_43, variance_42, add_275, rsqrt_42, hidden_states_211, to_385, hidden_states_212], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1219, buf1221, buf1197, buf1218, primals_553, buf1222, s27, 896, stream=stream0)
            buf1223 = buf1218; del buf1218  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_210, hidden_states_211, to_385, hidden_states_212, result_444], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1222, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_554, (896, 896), (1, 896), 0), out=buf1223)
            buf1224 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_442], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_556, buf1224, 28672, stream=stream0)
            del primals_556
            buf1225 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_442], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1222, (s27, 896), (896, 1), 0), buf1224, out=buf1225)
            buf1226 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_443], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_557, buf1226, 28672, stream=stream0)
            del primals_557
            buf1227 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_443], Original ATen: [aten.mm]
            extern_kernels.mm(buf1225, buf1226, out=buf1227)
            buf1228 = buf1179; del buf1179  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_210, hidden_states_211, to_385, hidden_states_212, result_444, result_447], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1222, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_558, (896, 128), (1, 896), 0), out=buf1228)
            buf1229 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_445], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_560, buf1229, 28672, stream=stream0)
            del primals_560
            buf1230 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_442, linear_445], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1222, (s27, 896), (896, 1), 0), buf1229, out=buf1230)
            buf1231 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_446], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_561, buf1231, 4096, stream=stream0)
            del primals_561
            buf1232 = buf1175; del buf1175  # reuse
            # Topologically Sorted Source Nodes: [linear_446], Original ATen: [aten.mm]
            extern_kernels.mm(buf1230, buf1231, out=buf1232)
            buf1233 = reinterpret_tensor(buf1183, (s27, 128), (128, 1), 0); del buf1183  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_210, hidden_states_211, to_385, hidden_states_212, result_444, result_450], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1222, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_562, (896, 128), (1, 896), 0), out=buf1233)
            buf1234 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_448], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_564, buf1234, 28672, stream=stream0)
            del primals_564
            buf1235 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_442, linear_448], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1222, (s27, 896), (896, 1), 0), buf1234, out=buf1235)
            buf1236 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_449], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_565, buf1236, 4096, stream=stream0)
            del primals_565
            buf1237 = reinterpret_tensor(buf1182, (s27, 128), (128, 1), 0); del buf1182  # reuse
            # Topologically Sorted Source Nodes: [linear_449], Original ATen: [aten.mm]
            extern_kernels.mm(buf1235, buf1236, out=buf1237)
            buf1238 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_444, linear_443, mul_340, result_445, view_63, query_states_21, x1_42, x2_42, neg_42, cat_43, mul_344], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_555, buf1223, buf1227, buf1, buf1238, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf1239 = reinterpret_tensor(buf1223, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf1223  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_444, linear_443, mul_340, result_445, view_63, query_states_21, mul_343, q_embed_21], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf1239, primals_555, buf1227, buf1, buf1238, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_555
            buf1240 = reinterpret_tensor(buf1174, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf1174  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_447, linear_446, mul_341, result_448, view_64, key_states_21, x1_43, x2_43, neg_43, cat_44, mul_346], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_559, buf1228, buf1232, buf1, buf1240, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf1241 = reinterpret_tensor(buf1228, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf1228  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_447, linear_446, mul_341, result_448, view_64, key_states_21, mul_345, k_embed_21], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf1241, primals_559, buf1232, buf1, buf1240, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_559
            buf1242 = reinterpret_tensor(buf1238, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1238  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_447, linear_446, mul_341, result_448, view_64, key_states_21, mul_345, k_embed_21, getitem_929, hidden_states_213, key_21], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf1241, buf1242, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf1243 = reinterpret_tensor(buf1227, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1227  # reuse
            # Topologically Sorted Source Nodes: [result_450, linear_449, mul_342, result_451, view_65, value_states_21, getitem_934, hidden_states_214, value_21], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_563, buf1233, buf1237, buf1243, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_563
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_447, linear_446, mul_341, result_448, view_64, key_states_21, result_450, linear_449, mul_342, result_451, view_65, value_states_21, mul_345, k_embed_21, getitem_929, hidden_states_213, key_21, getitem_934, hidden_states_214, value_21, attn_output_63], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf1244 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf1239, reinterpret_tensor(buf1242, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1243, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf1245 = buf1244[0]
            assert_size_stride(buf1245, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1245, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1246 = buf1244[1]
            assert_size_stride(buf1246, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1246, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1247 = buf1244[2]
            assert_size_stride(buf1247, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1247, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1248 = buf1244[3]
            assert_size_stride(buf1248, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1248, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf1244
            buf1249 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_88, reshape_65, result_453], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1245, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_566, (896, 896), (1, 896), 0), out=buf1249)
            buf1250 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_451], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_567, buf1250, 28672, stream=stream0)
            del primals_567
            buf1251 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_88, reshape_65, linear_451], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf1245, buf1251, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf1252 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_88, reshape_65, linear_451], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1251, (s27, 896), (896, 1), 0), buf1250, out=buf1252)
            buf1253 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_452], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_568, buf1253, 28672, stream=stream0)
            del primals_568
            buf1254 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_452], Original ATen: [aten.mm]
            extern_kernels.mm(buf1252, buf1253, out=buf1254)
            buf1255 = reinterpret_tensor(buf1249, (1, s27, 896), (896*s27, 896, 1), 0); del buf1249  # reuse
            buf1256 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1257 = reinterpret_tensor(buf1256, (1, s27, 1), (s27, 1, 1), 0); del buf1256  # reuse
            buf1258 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_453, linear_452, mul_347, result_454, hidden_states_215, hidden_states_216, pow_44, variance_43, add_283, rsqrt_43, hidden_states_217, to_395, hidden_states_218], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1255, buf1257, buf1219, buf1254, primals_569, buf1258, s27, 896, stream=stream0)
            buf1259 = buf1211; del buf1211  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_216, hidden_states_217, to_395, hidden_states_218, result_456], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1258, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_570, (896, 4864), (1, 896), 0), out=buf1259)
            buf1260 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_454], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_571, buf1260, 28672, stream=stream0)
            del primals_571
            buf1261 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_454], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1258, (s27, 896), (896, 1), 0), buf1260, out=buf1261)
            buf1262 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_455], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_572, buf1262, 155648, stream=stream0)
            del primals_572
            buf1263 = buf1205; del buf1205  # reuse
            # Topologically Sorted Source Nodes: [linear_455], Original ATen: [aten.mm]
            extern_kernels.mm(buf1261, buf1262, out=buf1263)
            buf1265 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_216, hidden_states_217, to_395, hidden_states_218, result_456, result_459], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1258, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_573, (896, 4864), (1, 896), 0), out=buf1265)
            buf1266 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_457], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_574, buf1266, 28672, stream=stream0)
            del primals_574
            buf1267 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_454, linear_457], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1258, (s27, 896), (896, 1), 0), buf1266, out=buf1267)
            buf1268 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_458], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_575, buf1268, 155648, stream=stream0)
            del primals_575
            buf1269 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_458], Original ATen: [aten.mm]
            extern_kernels.mm(buf1267, buf1268, out=buf1269)
            buf1264 = reinterpret_tensor(buf1259, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1259  # reuse
            buf1270 = reinterpret_tensor(buf1265, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1265  # reuse
            buf1271 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_456, linear_455, mul_350, result_457, silu_21, result_459, linear_458, mul_351, result_460, mul_352], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf1264, buf1270, buf1263, buf1269, buf1271, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf1272 = buf1254; del buf1254  # reuse
            # Topologically Sorted Source Nodes: [silu_21, mul_352, result_462], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1271, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_576, (4864, 896), (1, 4864), 0), out=buf1272)
            buf1273 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_460], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_577, buf1273, 155648, stream=stream0)
            del primals_577
            buf1274 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_460], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1271, (s27, 4864), (4864, 1), 0), buf1273, out=buf1274)
            buf1275 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_461], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_578, buf1275, 28672, stream=stream0)
            del primals_578
            buf1276 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_461], Original ATen: [aten.mm]
            extern_kernels.mm(buf1274, buf1275, out=buf1276)
            buf1277 = reinterpret_tensor(buf1272, (1, s27, 896), (896*s27, 896, 1), 0); del buf1272  # reuse
            buf1278 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1279 = reinterpret_tensor(buf1278, (1, s27, 1), (s27, 1, 1), 0); del buf1278  # reuse
            buf1280 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_462, linear_461, mul_353, result_463, hidden_states_219, hidden_states_220, pow_45, variance_44, add_288, rsqrt_44, hidden_states_221, to_403, hidden_states_222], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1277, buf1279, buf1255, buf1276, primals_579, buf1280, s27, 896, stream=stream0)
            buf1281 = buf1276; del buf1276  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_220, hidden_states_221, to_403, hidden_states_222, result_465], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1280, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_580, (896, 896), (1, 896), 0), out=buf1281)
            buf1282 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_463], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_582, buf1282, 28672, stream=stream0)
            del primals_582
            buf1283 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_463], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1280, (s27, 896), (896, 1), 0), buf1282, out=buf1283)
            buf1284 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_464], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_583, buf1284, 28672, stream=stream0)
            del primals_583
            buf1285 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_464], Original ATen: [aten.mm]
            extern_kernels.mm(buf1283, buf1284, out=buf1285)
            buf1286 = buf1237; del buf1237  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_220, hidden_states_221, to_403, hidden_states_222, result_465, result_468], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1280, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_584, (896, 128), (1, 896), 0), out=buf1286)
            buf1287 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_466], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_586, buf1287, 28672, stream=stream0)
            del primals_586
            buf1288 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_463, linear_466], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1280, (s27, 896), (896, 1), 0), buf1287, out=buf1288)
            buf1289 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_467], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_587, buf1289, 4096, stream=stream0)
            del primals_587
            buf1290 = buf1233; del buf1233  # reuse
            # Topologically Sorted Source Nodes: [linear_467], Original ATen: [aten.mm]
            extern_kernels.mm(buf1288, buf1289, out=buf1290)
            buf1291 = reinterpret_tensor(buf1241, (s27, 128), (128, 1), 0); del buf1241  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_220, hidden_states_221, to_403, hidden_states_222, result_465, result_471], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1280, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_588, (896, 128), (1, 896), 0), out=buf1291)
            buf1292 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_469], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_590, buf1292, 28672, stream=stream0)
            del primals_590
            buf1293 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_463, linear_469], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1280, (s27, 896), (896, 1), 0), buf1292, out=buf1293)
            buf1294 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_470], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_591, buf1294, 4096, stream=stream0)
            del primals_591
            buf1295 = reinterpret_tensor(buf1240, (s27, 128), (128, 1), 0); del buf1240  # reuse
            # Topologically Sorted Source Nodes: [linear_470], Original ATen: [aten.mm]
            extern_kernels.mm(buf1293, buf1294, out=buf1295)
            buf1296 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_465, linear_464, mul_356, result_466, view_66, query_states_22, x1_44, x2_44, neg_44, cat_45, mul_360], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_581, buf1281, buf1285, buf1, buf1296, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf1297 = reinterpret_tensor(buf1281, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf1281  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_465, linear_464, mul_356, result_466, view_66, query_states_22, mul_359, q_embed_22], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf1297, primals_581, buf1285, buf1, buf1296, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_581
            buf1298 = reinterpret_tensor(buf1232, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf1232  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_468, linear_467, mul_357, result_469, view_67, key_states_22, x1_45, x2_45, neg_45, cat_46, mul_362], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_585, buf1286, buf1290, buf1, buf1298, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf1299 = reinterpret_tensor(buf1286, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf1286  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_468, linear_467, mul_357, result_469, view_67, key_states_22, mul_361, k_embed_22], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf1299, primals_585, buf1290, buf1, buf1298, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del primals_585
            buf1300 = reinterpret_tensor(buf1296, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1296  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_468, linear_467, mul_357, result_469, view_67, key_states_22, mul_361, k_embed_22, getitem_971, hidden_states_223, key_22], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf1299, buf1300, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            buf1301 = reinterpret_tensor(buf1285, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1285  # reuse
            # Topologically Sorted Source Nodes: [result_471, linear_470, mul_358, result_472, view_68, value_states_22, getitem_976, hidden_states_224, value_22], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_589, buf1291, buf1295, buf1301, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del primals_589
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_468, linear_467, mul_357, result_469, view_67, key_states_22, result_471, linear_470, mul_358, result_472, view_68, value_states_22, mul_361, k_embed_22, getitem_971, hidden_states_223, key_22, getitem_976, hidden_states_224, value_22, attn_output_66], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf1302 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf1297, reinterpret_tensor(buf1300, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1301, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf1303 = buf1302[0]
            assert_size_stride(buf1303, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1303, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1304 = buf1302[1]
            assert_size_stride(buf1304, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1304, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1305 = buf1302[2]
            assert_size_stride(buf1305, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1305, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1306 = buf1302[3]
            assert_size_stride(buf1306, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1306, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf1302
            buf1307 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_92, reshape_68, result_474], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1303, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_592, (896, 896), (1, 896), 0), out=buf1307)
            buf1308 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_472], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_593, buf1308, 28672, stream=stream0)
            del primals_593
            buf1309 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_92, reshape_68, linear_472], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf1303, buf1309, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf1310 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_92, reshape_68, linear_472], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1309, (s27, 896), (896, 1), 0), buf1308, out=buf1310)
            buf1311 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_473], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_594, buf1311, 28672, stream=stream0)
            del primals_594
            buf1312 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_473], Original ATen: [aten.mm]
            extern_kernels.mm(buf1310, buf1311, out=buf1312)
            buf1313 = reinterpret_tensor(buf1307, (1, s27, 896), (896*s27, 896, 1), 0); del buf1307  # reuse
            buf1314 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1315 = reinterpret_tensor(buf1314, (1, s27, 1), (s27, 1, 1), 0); del buf1314  # reuse
            buf1316 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_474, linear_473, mul_363, result_475, hidden_states_225, hidden_states_226, pow_46, variance_45, add_296, rsqrt_45, hidden_states_227, to_413, hidden_states_228], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1313, buf1315, buf1277, buf1312, primals_595, buf1316, s27, 896, stream=stream0)
            buf1317 = buf1269; del buf1269  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_226, hidden_states_227, to_413, hidden_states_228, result_477], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1316, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_596, (896, 4864), (1, 896), 0), out=buf1317)
            buf1318 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_475], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_597, buf1318, 28672, stream=stream0)
            del primals_597
            buf1319 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_475], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1316, (s27, 896), (896, 1), 0), buf1318, out=buf1319)
            buf1320 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_476], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_598, buf1320, 155648, stream=stream0)
            del primals_598
            buf1321 = buf1263; del buf1263  # reuse
            # Topologically Sorted Source Nodes: [linear_476], Original ATen: [aten.mm]
            extern_kernels.mm(buf1319, buf1320, out=buf1321)
            buf1323 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_226, hidden_states_227, to_413, hidden_states_228, result_477, result_480], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1316, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_599, (896, 4864), (1, 896), 0), out=buf1323)
            buf1324 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_478], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_600, buf1324, 28672, stream=stream0)
            del primals_600
            buf1325 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_475, linear_478], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1316, (s27, 896), (896, 1), 0), buf1324, out=buf1325)
            buf1326 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_479], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_601, buf1326, 155648, stream=stream0)
            del primals_601
            buf1327 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_479], Original ATen: [aten.mm]
            extern_kernels.mm(buf1325, buf1326, out=buf1327)
            buf1322 = reinterpret_tensor(buf1317, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1317  # reuse
            buf1328 = reinterpret_tensor(buf1323, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1323  # reuse
            buf1329 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_477, linear_476, mul_366, result_478, silu_22, result_480, linear_479, mul_367, result_481, mul_368], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf1322, buf1328, buf1321, buf1327, buf1329, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            buf1330 = buf1312; del buf1312  # reuse
            # Topologically Sorted Source Nodes: [silu_22, mul_368, result_483], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1329, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_602, (4864, 896), (1, 4864), 0), out=buf1330)
            buf1331 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_481], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_603, buf1331, 155648, stream=stream0)
            del primals_603
            buf1332 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_481], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1329, (s27, 4864), (4864, 1), 0), buf1331, out=buf1332)
            buf1333 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_482], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_604, buf1333, 28672, stream=stream0)
            del primals_604
            buf1334 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_482], Original ATen: [aten.mm]
            extern_kernels.mm(buf1332, buf1333, out=buf1334)
            buf1335 = reinterpret_tensor(buf1330, (1, s27, 896), (896*s27, 896, 1), 0); del buf1330  # reuse
            buf1336 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1337 = reinterpret_tensor(buf1336, (1, s27, 1), (s27, 1, 1), 0); del buf1336  # reuse
            buf1338 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_483, linear_482, mul_369, result_484, hidden_states_229, hidden_states_230, pow_47, variance_46, add_301, rsqrt_46, hidden_states_231, to_421, hidden_states_232], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1335, buf1337, buf1313, buf1334, primals_605, buf1338, s27, 896, stream=stream0)
            buf1339 = buf1334; del buf1334  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_230, hidden_states_231, to_421, hidden_states_232, result_486], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1338, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_606, (896, 896), (1, 896), 0), out=buf1339)
            buf1340 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_484], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_608, buf1340, 28672, stream=stream0)
            del primals_608
            buf1341 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_484], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1338, (s27, 896), (896, 1), 0), buf1340, out=buf1341)
            buf1342 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_485], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_609, buf1342, 28672, stream=stream0)
            del primals_609
            buf1343 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_485], Original ATen: [aten.mm]
            extern_kernels.mm(buf1341, buf1342, out=buf1343)
            buf1344 = buf1295; del buf1295  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_230, hidden_states_231, to_421, hidden_states_232, result_486, result_489], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1338, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_610, (896, 128), (1, 896), 0), out=buf1344)
            buf1345 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_487], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_612, buf1345, 28672, stream=stream0)
            del primals_612
            buf1346 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_484, linear_487], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1338, (s27, 896), (896, 1), 0), buf1345, out=buf1346)
            buf1347 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_488], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_613, buf1347, 4096, stream=stream0)
            del primals_613
            buf1348 = buf1291; del buf1291  # reuse
            # Topologically Sorted Source Nodes: [linear_488], Original ATen: [aten.mm]
            extern_kernels.mm(buf1346, buf1347, out=buf1348)
            buf1349 = reinterpret_tensor(buf1299, (s27, 128), (128, 1), 0); del buf1299  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_230, hidden_states_231, to_421, hidden_states_232, result_486, result_492], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.addmm]
            extern_kernels.mm(reinterpret_tensor(buf1338, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_614, (896, 128), (1, 896), 0), out=buf1349)
            buf1350 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_490], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_616, buf1350, 28672, stream=stream0)
            del primals_616
            buf1351 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_484, linear_490], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1338, (s27, 896), (896, 1), 0), buf1350, out=buf1351)
            buf1352 = empty_strided_cuda((32, 128), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_491], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_3.run(primals_617, buf1352, 4096, stream=stream0)
            del primals_617
            buf1353 = reinterpret_tensor(buf1298, (s27, 128), (128, 1), 0); del buf1298  # reuse
            # Topologically Sorted Source Nodes: [linear_491], Original ATen: [aten.mm]
            extern_kernels.mm(buf1351, buf1352, out=buf1353)
            buf1354 = empty_strided_cuda((1, 14, s27, 64), (896*s27, 1, 14, 14*s27), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_486, linear_485, mul_372, result_487, view_69, query_states_23, x1_46, x2_46, neg_46, cat_47, mul_376], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4.run(primals_607, buf1339, buf1343, buf1, buf1354, ps0, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_4_xnumel, stream=stream0)
            buf1355 = reinterpret_tensor(buf1339, (1, 14, s27, 64), (896*s27, 64, 896, 1), 0); del buf1339  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_486, linear_485, mul_372, result_487, view_69, query_states_23, mul_375, q_embed_23], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel = 14*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5.run(buf1355, primals_607, buf1343, buf1, buf1354, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_5_ynumel, 64, stream=stream0)
            del primals_607
            buf1356 = reinterpret_tensor(buf1290, (1, 2, s27, 64), (128*s27, 1, 2, 2*s27), 0); del buf1290  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, result_489, linear_488, mul_373, result_490, view_70, key_states_23, x1_47, x2_47, neg_47, cat_48, mul_378], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.slice, aten.neg]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel = 128*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6.run(primals_611, buf1344, buf1348, buf1, buf1356, ps1, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_mul_neg_sin_slice_transpose_unsqueeze_view_6_xnumel, stream=stream0)
            buf1357 = reinterpret_tensor(buf1344, (1, 2, s27, 64), (128*s27, 64, 128, 1), 0); del buf1344  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_489, linear_488, mul_373, result_490, view_70, key_states_23, mul_377, k_embed_23], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel = 2*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7.run(buf1357, primals_611, buf1348, buf1, buf1356, s27, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_transpose_unsqueeze_view_7_ynumel, 64, stream=stream0)
            del buf1348
            del buf1356
            del primals_611
            buf1358 = reinterpret_tensor(buf1354, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1354  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, result_489, linear_488, mul_373, result_490, view_70, key_states_23, mul_377, k_embed_23, getitem_1013, hidden_states_233, key_23], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.expand, aten.clone]
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8.run(buf1357, buf1358, s27, ps2, triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_clone_cos_expand_mul_transpose_unsqueeze_view_8_xnumel, stream=stream0)
            del buf1357
            buf1359 = reinterpret_tensor(buf1343, (1, 2, 7, s27, 64), (896*s27, 448*s27, 64*s27, 64, 1), 0); del buf1343  # reuse
            # Topologically Sorted Source Nodes: [result_492, linear_491, mul_374, result_493, view_71, value_states_23, getitem_1018, hidden_states_234, value_23], Original ATen: [aten.addmm, aten.view, aten._unsafe_view, aten.mul, aten.add, aten.transpose, aten.unsqueeze, aten.expand, aten.clone]
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9.run(primals_615, buf1349, buf1353, buf1359, ps2, s27, triton_poi_fused__unsafe_view_add_addmm_clone_expand_mul_transpose_unsqueeze_view_9_xnumel, stream=stream0)
            del buf1349
            del buf1353
            del primals_615
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, cos, cos_1, cos_2, cos_3, attn_output, result_489, linear_488, mul_373, result_490, view_70, key_states_23, result_492, linear_491, mul_374, result_493, view_71, value_states_23, mul_377, k_embed_23, getitem_1013, hidden_states_233, key_23, getitem_1018, hidden_states_234, value_23, attn_output_69], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.cos, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.expand, aten.addmm, aten.view, aten._unsafe_view, aten.add, aten.clone, aten._scaled_dot_product_efficient_attention]
            buf1360 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf1355, reinterpret_tensor(buf1358, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1359, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf25, (1, 14, s27, s27), (s27*max(1, 8 + s27 + (-1)*(s27 % 8)), 0, max(1, 8 + s27 + (-1)*(s27 % 8)), 1), 0), True, scale=0.125)
            buf1361 = buf1360[0]
            assert_size_stride(buf1361, (1, 14, s27, 64), (896*s27, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1361, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1362 = buf1360[1]
            assert_size_stride(buf1362, (1, 14, 32*math.ceil(s27 / 32)), (14*max(1, 32*math.ceil(s27 / 32)), max(1, 32*math.ceil(s27 / 32)), 1), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1362, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1363 = buf1360[2]
            assert_size_stride(buf1363, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1363, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            buf1364 = buf1360[3]
            assert_size_stride(buf1364, (), (), 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            assert_alignment(buf1364, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention.default')
            del buf1360
            buf1365 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_96, reshape_71, result_495], Original ATen: [aten.transpose, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1361, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_618, (896, 896), (1, 896), 0), out=buf1365)
            buf1366 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_493], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_619, buf1366, 28672, stream=stream0)
            del primals_619
            buf1367 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_96, reshape_71, linear_493], Original ATen: [aten.transpose, aten.view, aten._to_copy]
            triton_poi_fused__to_copy_transpose_view_11_xnumel = 896*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_transpose_view_11.run(buf1361, buf1367, triton_poi_fused__to_copy_transpose_view_11_xnumel, stream=stream0)
            buf1368 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [transpose_96, reshape_71, linear_493], Original ATen: [aten.transpose, aten.view, aten._to_copy, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1367, (s27, 896), (896, 1), 0), buf1366, out=buf1368)
            buf1369 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_494], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_620, buf1369, 28672, stream=stream0)
            del primals_620
            buf1370 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_494], Original ATen: [aten.mm]
            extern_kernels.mm(buf1368, buf1369, out=buf1370)
            buf1371 = reinterpret_tensor(buf1365, (1, s27, 896), (896*s27, 896, 1), 0); del buf1365  # reuse
            buf1372 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1373 = reinterpret_tensor(buf1372, (1, s27, 1), (s27, 1, 1), 0); del buf1372  # reuse
            buf1374 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_495, linear_494, mul_379, result_496, hidden_states_235, hidden_states_236, pow_48, variance_47, add_309, rsqrt_47, hidden_states_237, to_431, hidden_states_238], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1371, buf1373, buf1335, buf1370, primals_621, buf1374, s27, 896, stream=stream0)
            buf1375 = buf1327; del buf1327  # reuse
            # Topologically Sorted Source Nodes: [hidden_states_236, hidden_states_237, to_431, hidden_states_238, result_498], Original ATen: [aten._to_copy, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1374, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_622, (896, 4864), (1, 896), 0), out=buf1375)
            buf1376 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_496], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_623, buf1376, 28672, stream=stream0)
            del primals_623
            buf1377 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_496], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1374, (s27, 896), (896, 1), 0), buf1376, out=buf1377)
            buf1378 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_497], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_624, buf1378, 155648, stream=stream0)
            del primals_624
            buf1379 = buf1321; del buf1321  # reuse
            # Topologically Sorted Source Nodes: [linear_497], Original ATen: [aten.mm]
            extern_kernels.mm(buf1377, buf1378, out=buf1379)
            buf1381 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_236, hidden_states_237, to_431, hidden_states_238, result_498, result_501], Original ATen: [aten._to_copy, aten.mul, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1374, (s27, 896), (896, 1), 0), reinterpret_tensor(primals_625, (896, 4864), (1, 896), 0), out=buf1381)
            buf1382 = empty_strided_cuda((896, 32), (1, 896), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_499], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_626, buf1382, 28672, stream=stream0)
            del primals_626
            buf1383 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_496, linear_499], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1374, (s27, 896), (896, 1), 0), buf1382, out=buf1383)
            buf1384 = empty_strided_cuda((32, 4864), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_500], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_627, buf1384, 155648, stream=stream0)
            del primals_627
            buf1385 = empty_strided_cuda((s27, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_500], Original ATen: [aten.mm]
            extern_kernels.mm(buf1383, buf1384, out=buf1385)
            buf1380 = reinterpret_tensor(buf1375, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1375  # reuse
            buf1386 = reinterpret_tensor(buf1381, (1, s27, 4864), (4864*s27, 4864, 1), 0); del buf1381  # reuse
            buf1387 = empty_strided_cuda((1, s27, 4864), (4864*s27, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_498, linear_497, mul_382, result_499, silu_23, result_501, linear_500, mul_383, result_502, mul_384], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten.silu]
            triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel = 4864*s27
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_add_mul_silu_14.run(buf1380, buf1386, buf1379, buf1385, buf1387, triton_poi_fused__unsafe_view_add_mul_silu_14_xnumel, stream=stream0)
            del buf1379
            del buf1385
            buf1388 = buf1370; del buf1370  # reuse
            # Topologically Sorted Source Nodes: [silu_23, mul_384, result_504], Original ATen: [aten.silu, aten.mul, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1387, (s27, 4864), (4864, 1), 0), reinterpret_tensor(primals_628, (4864, 896), (1, 4864), 0), out=buf1388)
            buf1389 = empty_strided_cuda((4864, 32), (1, 4864), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_502], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_t_13.run(primals_629, buf1389, 155648, stream=stream0)
            del primals_629
            buf1390 = empty_strided_cuda((s27, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_502], Original ATen: [aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1387, (s27, 4864), (4864, 1), 0), buf1389, out=buf1390)
            buf1391 = empty_strided_cuda((32, 896), (1, 32), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_503], Original ATen: [aten._to_copy, aten.t]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(primals_630, buf1391, 28672, stream=stream0)
            del primals_630
            buf1392 = empty_strided_cuda((s27, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [linear_503], Original ATen: [aten.mm]
            extern_kernels.mm(buf1390, buf1391, out=buf1392)
            buf1393 = reinterpret_tensor(buf1388, (1, s27, 896), (896*s27, 896, 1), 0); del buf1388  # reuse
            buf1394 = empty_strided_cuda((1, s27, 1), (s27, 1, s27), torch.float32)
            buf1395 = reinterpret_tensor(buf1394, (1, s27, 1), (s27, 1, 1), 0); del buf1394  # reuse
            buf1396 = empty_strided_cuda((1, s27, 896), (896*s27, 896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [result_504, linear_503, mul_385, result_505, hidden_states_239, hidden_states_240, pow_49, variance_48, add_314, rsqrt_48, hidden_states_241, to_439, hidden_states_242], Original ATen: [aten._unsafe_view, aten.mul, aten.add, aten._to_copy, aten.pow, aten.mean, aten.rsqrt]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15.run(buf1393, buf1395, buf1371, buf1392, primals_631, buf1396, s27, 896, stream=stream0)
            del buf1392
            buf1397 = empty_strided_cuda((s35, 151936), (151936, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [hidden_states_240, hidden_states_241, to_439, hidden_states_242, getitem_1028, logits], Original ATen: [aten._to_copy, aten.mul, aten.slice, aten.t, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1396, (s35, 896), (896, 1), ((-896)*s35) + 896*s27), reinterpret_tensor(primals_633, (896, 151936), (1, 896), 0), out=buf1397)
            buf1398 = empty_strided_cuda((1, s35, 151936), (151936*s35, 151936, 1), torch.float32)
            # Topologically Sorted Source Nodes: [logits, float_5], Original ATen: [aten._unsafe_view, aten._to_copy]
            triton_poi_fused__to_copy__unsafe_view_16_xnumel = 151936*s35
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_16.run(buf1397, buf1398, triton_poi_fused__to_copy__unsafe_view_16_xnumel, stream=stream0)
            del buf1397
        return (buf1398, primals_20, primals_23, primals_24, primals_27, primals_30, primals_33, primals_34, primals_38, primals_42, primals_46, primals_49, primals_50, primals_53, primals_56, primals_59, primals_60, primals_64, primals_68, primals_72, primals_75, primals_76, primals_79, primals_82, primals_85, primals_86, primals_90, primals_94, primals_98, primals_101, primals_102, primals_105, primals_108, primals_111, primals_112, primals_116, primals_120, primals_124, primals_127, primals_128, primals_131, primals_134, primals_137, primals_138, primals_142, primals_146, primals_150, primals_153, primals_154, primals_157, primals_160, primals_163, primals_164, primals_168, primals_172, primals_176, primals_179, primals_180, primals_183, primals_186, primals_189, primals_190, primals_194, primals_198, primals_202, primals_205, primals_206, primals_209, primals_212, primals_215, primals_216, primals_220, primals_224, primals_228, primals_231, primals_232, primals_235, primals_238, primals_241, primals_242, primals_246, primals_250, primals_254, primals_257, primals_258, primals_261, primals_264, primals_267, primals_268, primals_272, primals_276, primals_280, primals_283, primals_284, primals_287, primals_290, primals_293, primals_294, primals_298, primals_302, primals_306, primals_309, primals_310, primals_313, primals_316, primals_319, primals_320, primals_324, primals_328, primals_332, primals_335, primals_336, primals_339, primals_342, primals_345, primals_346, primals_350, primals_354, primals_358, primals_361, primals_362, primals_365, primals_368, primals_371, primals_372, primals_376, primals_380, primals_384, primals_387, primals_388, primals_391, primals_394, primals_397, primals_398, primals_402, primals_406, primals_410, primals_413, primals_414, primals_417, primals_420, primals_423, primals_424, primals_428, primals_432, primals_436, primals_439, primals_440, primals_443, primals_446, primals_449, primals_450, primals_454, primals_458, primals_462, primals_465, primals_466, primals_469, primals_472, primals_475, primals_476, primals_480, primals_484, primals_488, primals_491, primals_492, primals_495, primals_498, primals_501, primals_502, primals_506, primals_510, primals_514, primals_517, primals_518, primals_521, primals_524, primals_527, primals_528, primals_532, primals_536, primals_540, primals_543, primals_544, primals_547, primals_550, primals_553, primals_554, primals_558, primals_562, primals_566, primals_569, primals_570, primals_573, primals_576, primals_579, primals_580, primals_584, primals_588, primals_592, primals_595, primals_596, primals_599, primals_602, primals_605, primals_606, primals_610, primals_614, primals_618, primals_621, primals_622, primals_625, primals_628, primals_631, primals_633, buf1, reinterpret_tensor(buf3, (s27, 896), (896, 1), 0), buf6, buf11, buf16, buf20, reinterpret_tensor(buf23, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf24, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf25, buf27, buf28, buf29, buf30, reinterpret_tensor(buf33, (s27, 896), (896, 1), 0), buf34, buf37, buf39, reinterpret_tensor(buf40, (s27, 896), (896, 1), 0), buf43, buf46, buf49, buf52, reinterpret_tensor(buf53, (s27, 4864), (4864, 1), 0), buf56, buf59, buf61, reinterpret_tensor(buf62, (s27, 896), (896, 1), 0), buf65, buf70, buf75, buf79, reinterpret_tensor(buf82, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf83, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf85, buf86, buf87, buf88, reinterpret_tensor(buf91, (s27, 896), (896, 1), 0), buf92, buf95, buf97, reinterpret_tensor(buf98, (s27, 896), (896, 1), 0), buf101, buf104, buf107, buf110, reinterpret_tensor(buf111, (s27, 4864), (4864, 1), 0), buf114, buf117, buf119, reinterpret_tensor(buf120, (s27, 896), (896, 1), 0), buf123, buf128, buf133, buf137, reinterpret_tensor(buf140, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf141, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf143, buf144, buf145, buf146, reinterpret_tensor(buf149, (s27, 896), (896, 1), 0), buf150, buf153, buf155, reinterpret_tensor(buf156, (s27, 896), (896, 1), 0), buf159, buf162, buf165, buf168, reinterpret_tensor(buf169, (s27, 4864), (4864, 1), 0), buf172, buf175, buf177, reinterpret_tensor(buf178, (s27, 896), (896, 1), 0), buf181, buf186, buf191, buf195, reinterpret_tensor(buf198, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf199, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf201, buf202, buf203, buf204, reinterpret_tensor(buf207, (s27, 896), (896, 1), 0), buf208, buf211, buf213, reinterpret_tensor(buf214, (s27, 896), (896, 1), 0), buf217, buf220, buf223, buf226, reinterpret_tensor(buf227, (s27, 4864), (4864, 1), 0), buf230, buf233, buf235, reinterpret_tensor(buf236, (s27, 896), (896, 1), 0), buf239, buf244, buf249, buf253, reinterpret_tensor(buf256, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf257, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf259, buf260, buf261, buf262, reinterpret_tensor(buf265, (s27, 896), (896, 1), 0), buf266, buf269, buf271, reinterpret_tensor(buf272, (s27, 896), (896, 1), 0), buf275, buf278, buf281, buf284, reinterpret_tensor(buf285, (s27, 4864), (4864, 1), 0), buf288, buf291, buf293, reinterpret_tensor(buf294, (s27, 896), (896, 1), 0), buf297, buf302, buf307, buf311, reinterpret_tensor(buf314, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf315, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf317, buf318, buf319, buf320, reinterpret_tensor(buf323, (s27, 896), (896, 1), 0), buf324, buf327, buf329, reinterpret_tensor(buf330, (s27, 896), (896, 1), 0), buf333, buf336, buf339, buf342, reinterpret_tensor(buf343, (s27, 4864), (4864, 1), 0), buf346, buf349, buf351, reinterpret_tensor(buf352, (s27, 896), (896, 1), 0), buf355, buf360, buf365, buf369, reinterpret_tensor(buf372, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf373, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf375, buf376, buf377, buf378, reinterpret_tensor(buf381, (s27, 896), (896, 1), 0), buf382, buf385, buf387, reinterpret_tensor(buf388, (s27, 896), (896, 1), 0), buf391, buf394, buf397, buf400, reinterpret_tensor(buf401, (s27, 4864), (4864, 1), 0), buf404, buf407, buf409, reinterpret_tensor(buf410, (s27, 896), (896, 1), 0), buf413, buf418, buf423, buf427, reinterpret_tensor(buf430, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf431, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf433, buf434, buf435, buf436, reinterpret_tensor(buf439, (s27, 896), (896, 1), 0), buf440, buf443, buf445, reinterpret_tensor(buf446, (s27, 896), (896, 1), 0), buf449, buf452, buf455, buf458, reinterpret_tensor(buf459, (s27, 4864), (4864, 1), 0), buf462, buf465, buf467, reinterpret_tensor(buf468, (s27, 896), (896, 1), 0), buf471, buf476, buf481, buf485, reinterpret_tensor(buf488, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf489, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf491, buf492, buf493, buf494, reinterpret_tensor(buf497, (s27, 896), (896, 1), 0), buf498, buf501, buf503, reinterpret_tensor(buf504, (s27, 896), (896, 1), 0), buf507, buf510, buf513, buf516, reinterpret_tensor(buf517, (s27, 4864), (4864, 1), 0), buf520, buf523, buf525, reinterpret_tensor(buf526, (s27, 896), (896, 1), 0), buf529, buf534, buf539, buf543, reinterpret_tensor(buf546, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf547, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf549, buf550, buf551, buf552, reinterpret_tensor(buf555, (s27, 896), (896, 1), 0), buf556, buf559, buf561, reinterpret_tensor(buf562, (s27, 896), (896, 1), 0), buf565, buf568, buf571, buf574, reinterpret_tensor(buf575, (s27, 4864), (4864, 1), 0), buf578, buf581, buf583, reinterpret_tensor(buf584, (s27, 896), (896, 1), 0), buf587, buf592, buf597, buf601, reinterpret_tensor(buf604, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf605, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf607, buf608, buf609, buf610, reinterpret_tensor(buf613, (s27, 896), (896, 1), 0), buf614, buf617, buf619, reinterpret_tensor(buf620, (s27, 896), (896, 1), 0), buf623, buf626, buf629, buf632, reinterpret_tensor(buf633, (s27, 4864), (4864, 1), 0), buf636, buf639, buf641, reinterpret_tensor(buf642, (s27, 896), (896, 1), 0), buf645, buf650, buf655, buf659, reinterpret_tensor(buf662, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf663, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf665, buf666, buf667, buf668, reinterpret_tensor(buf671, (s27, 896), (896, 1), 0), buf672, buf675, buf677, reinterpret_tensor(buf678, (s27, 896), (896, 1), 0), buf681, buf684, buf687, buf690, reinterpret_tensor(buf691, (s27, 4864), (4864, 1), 0), buf694, buf697, buf699, reinterpret_tensor(buf700, (s27, 896), (896, 1), 0), buf703, buf708, buf713, buf717, reinterpret_tensor(buf720, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf721, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf723, buf724, buf725, buf726, reinterpret_tensor(buf729, (s27, 896), (896, 1), 0), buf730, buf733, buf735, reinterpret_tensor(buf736, (s27, 896), (896, 1), 0), buf739, buf742, buf745, buf748, reinterpret_tensor(buf749, (s27, 4864), (4864, 1), 0), buf752, buf755, buf757, reinterpret_tensor(buf758, (s27, 896), (896, 1), 0), buf761, buf766, buf771, buf775, reinterpret_tensor(buf778, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf779, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf781, buf782, buf783, buf784, reinterpret_tensor(buf787, (s27, 896), (896, 1), 0), buf788, buf791, buf793, reinterpret_tensor(buf794, (s27, 896), (896, 1), 0), buf797, buf800, buf803, buf806, reinterpret_tensor(buf807, (s27, 4864), (4864, 1), 0), buf810, buf813, buf815, reinterpret_tensor(buf816, (s27, 896), (896, 1), 0), buf819, buf824, buf829, buf833, reinterpret_tensor(buf836, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf837, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf839, buf840, buf841, buf842, reinterpret_tensor(buf845, (s27, 896), (896, 1), 0), buf846, buf849, buf851, reinterpret_tensor(buf852, (s27, 896), (896, 1), 0), buf855, buf858, buf861, buf864, reinterpret_tensor(buf865, (s27, 4864), (4864, 1), 0), buf868, buf871, buf873, reinterpret_tensor(buf874, (s27, 896), (896, 1), 0), buf877, buf882, buf887, buf891, reinterpret_tensor(buf894, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf895, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf897, buf898, buf899, buf900, reinterpret_tensor(buf903, (s27, 896), (896, 1), 0), buf904, buf907, buf909, reinterpret_tensor(buf910, (s27, 896), (896, 1), 0), buf913, buf916, buf919, buf922, reinterpret_tensor(buf923, (s27, 4864), (4864, 1), 0), buf926, buf929, buf931, reinterpret_tensor(buf932, (s27, 896), (896, 1), 0), buf935, buf940, buf945, buf949, reinterpret_tensor(buf952, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf953, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf955, buf956, buf957, buf958, reinterpret_tensor(buf961, (s27, 896), (896, 1), 0), buf962, buf965, buf967, reinterpret_tensor(buf968, (s27, 896), (896, 1), 0), buf971, buf974, buf977, buf980, reinterpret_tensor(buf981, (s27, 4864), (4864, 1), 0), buf984, buf987, buf989, reinterpret_tensor(buf990, (s27, 896), (896, 1), 0), buf993, buf998, buf1003, buf1007, reinterpret_tensor(buf1010, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1011, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf1013, buf1014, buf1015, buf1016, reinterpret_tensor(buf1019, (s27, 896), (896, 1), 0), buf1020, buf1023, buf1025, reinterpret_tensor(buf1026, (s27, 896), (896, 1), 0), buf1029, buf1032, buf1035, buf1038, reinterpret_tensor(buf1039, (s27, 4864), (4864, 1), 0), buf1042, buf1045, buf1047, reinterpret_tensor(buf1048, (s27, 896), (896, 1), 0), buf1051, buf1056, buf1061, buf1065, reinterpret_tensor(buf1068, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1069, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf1071, buf1072, buf1073, buf1074, reinterpret_tensor(buf1077, (s27, 896), (896, 1), 0), buf1078, buf1081, buf1083, reinterpret_tensor(buf1084, (s27, 896), (896, 1), 0), buf1087, buf1090, buf1093, buf1096, reinterpret_tensor(buf1097, (s27, 4864), (4864, 1), 0), buf1100, buf1103, buf1105, reinterpret_tensor(buf1106, (s27, 896), (896, 1), 0), buf1109, buf1114, buf1119, buf1123, reinterpret_tensor(buf1126, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1127, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf1129, buf1130, buf1131, buf1132, reinterpret_tensor(buf1135, (s27, 896), (896, 1), 0), buf1136, buf1139, buf1141, reinterpret_tensor(buf1142, (s27, 896), (896, 1), 0), buf1145, buf1148, buf1151, buf1154, reinterpret_tensor(buf1155, (s27, 4864), (4864, 1), 0), buf1158, buf1161, buf1163, reinterpret_tensor(buf1164, (s27, 896), (896, 1), 0), buf1167, buf1172, buf1177, buf1181, reinterpret_tensor(buf1184, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1185, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf1187, buf1188, buf1189, buf1190, reinterpret_tensor(buf1193, (s27, 896), (896, 1), 0), buf1194, buf1197, buf1199, reinterpret_tensor(buf1200, (s27, 896), (896, 1), 0), buf1203, buf1206, buf1209, buf1212, reinterpret_tensor(buf1213, (s27, 4864), (4864, 1), 0), buf1216, buf1219, buf1221, reinterpret_tensor(buf1222, (s27, 896), (896, 1), 0), buf1225, buf1230, buf1235, buf1239, reinterpret_tensor(buf1242, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1243, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf1245, buf1246, buf1247, buf1248, reinterpret_tensor(buf1251, (s27, 896), (896, 1), 0), buf1252, buf1255, buf1257, reinterpret_tensor(buf1258, (s27, 896), (896, 1), 0), buf1261, buf1264, buf1267, buf1270, reinterpret_tensor(buf1271, (s27, 4864), (4864, 1), 0), buf1274, buf1277, buf1279, reinterpret_tensor(buf1280, (s27, 896), (896, 1), 0), buf1283, buf1288, buf1293, buf1297, reinterpret_tensor(buf1300, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1301, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf1303, buf1304, buf1305, buf1306, reinterpret_tensor(buf1309, (s27, 896), (896, 1), 0), buf1310, buf1313, buf1315, reinterpret_tensor(buf1316, (s27, 896), (896, 1), 0), buf1319, buf1322, buf1325, buf1328, reinterpret_tensor(buf1329, (s27, 4864), (4864, 1), 0), buf1332, buf1335, buf1337, reinterpret_tensor(buf1338, (s27, 896), (896, 1), 0), buf1341, buf1346, buf1351, buf1355, reinterpret_tensor(buf1358, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), reinterpret_tensor(buf1359, (1, 14, s27, 64), (896*s27, 64*s27, 64, 1), 0), buf1361, buf1362, buf1363, buf1364, reinterpret_tensor(buf1367, (s27, 896), (896, 1), 0), buf1368, buf1371, buf1373, reinterpret_tensor(buf1374, (s27, 896), (896, 1), 0), buf1377, buf1380, buf1383, buf1386, reinterpret_tensor(buf1387, (s27, 4864), (4864, 1), 0), buf1390, buf1393, buf1395, reinterpret_tensor(buf1396, (s35, 896), (896, 1), ((-896)*s35) + 896*s27), reinterpret_tensor(buf1391, (896, 32), (32, 1), 0), reinterpret_tensor(buf1389, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf1384, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1382, (32, 896), (896, 1), 0), reinterpret_tensor(buf1378, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1376, (32, 896), (896, 1), 0), reinterpret_tensor(buf1369, (896, 32), (32, 1), 0), reinterpret_tensor(buf1366, (32, 896), (896, 1), 0), reinterpret_tensor(buf1352, (128, 32), (32, 1), 0), reinterpret_tensor(buf1350, (32, 896), (896, 1), 0), reinterpret_tensor(buf1347, (128, 32), (32, 1), 0), reinterpret_tensor(buf1345, (32, 896), (896, 1), 0), reinterpret_tensor(buf1342, (896, 32), (32, 1), 0), reinterpret_tensor(buf1340, (32, 896), (896, 1), 0), reinterpret_tensor(buf1333, (896, 32), (32, 1), 0), reinterpret_tensor(buf1331, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf1326, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1324, (32, 896), (896, 1), 0), reinterpret_tensor(buf1320, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1318, (32, 896), (896, 1), 0), reinterpret_tensor(buf1311, (896, 32), (32, 1), 0), reinterpret_tensor(buf1308, (32, 896), (896, 1), 0), reinterpret_tensor(buf1294, (128, 32), (32, 1), 0), reinterpret_tensor(buf1292, (32, 896), (896, 1), 0), reinterpret_tensor(buf1289, (128, 32), (32, 1), 0), reinterpret_tensor(buf1287, (32, 896), (896, 1), 0), reinterpret_tensor(buf1284, (896, 32), (32, 1), 0), reinterpret_tensor(buf1282, (32, 896), (896, 1), 0), reinterpret_tensor(buf1275, (896, 32), (32, 1), 0), reinterpret_tensor(buf1273, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf1268, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1266, (32, 896), (896, 1), 0), reinterpret_tensor(buf1262, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1260, (32, 896), (896, 1), 0), reinterpret_tensor(buf1253, (896, 32), (32, 1), 0), reinterpret_tensor(buf1250, (32, 896), (896, 1), 0), reinterpret_tensor(buf1236, (128, 32), (32, 1), 0), reinterpret_tensor(buf1234, (32, 896), (896, 1), 0), reinterpret_tensor(buf1231, (128, 32), (32, 1), 0), reinterpret_tensor(buf1229, (32, 896), (896, 1), 0), reinterpret_tensor(buf1226, (896, 32), (32, 1), 0), reinterpret_tensor(buf1224, (32, 896), (896, 1), 0), reinterpret_tensor(buf1217, (896, 32), (32, 1), 0), reinterpret_tensor(buf1215, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf1210, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1208, (32, 896), (896, 1), 0), reinterpret_tensor(buf1204, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1202, (32, 896), (896, 1), 0), reinterpret_tensor(buf1195, (896, 32), (32, 1), 0), reinterpret_tensor(buf1192, (32, 896), (896, 1), 0), reinterpret_tensor(buf1178, (128, 32), (32, 1), 0), reinterpret_tensor(buf1176, (32, 896), (896, 1), 0), reinterpret_tensor(buf1173, (128, 32), (32, 1), 0), reinterpret_tensor(buf1171, (32, 896), (896, 1), 0), reinterpret_tensor(buf1168, (896, 32), (32, 1), 0), reinterpret_tensor(buf1166, (32, 896), (896, 1), 0), reinterpret_tensor(buf1159, (896, 32), (32, 1), 0), reinterpret_tensor(buf1157, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf1152, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1150, (32, 896), (896, 1), 0), reinterpret_tensor(buf1146, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1144, (32, 896), (896, 1), 0), reinterpret_tensor(buf1137, (896, 32), (32, 1), 0), reinterpret_tensor(buf1134, (32, 896), (896, 1), 0), reinterpret_tensor(buf1120, (128, 32), (32, 1), 0), reinterpret_tensor(buf1118, (32, 896), (896, 1), 0), reinterpret_tensor(buf1115, (128, 32), (32, 1), 0), reinterpret_tensor(buf1113, (32, 896), (896, 1), 0), reinterpret_tensor(buf1110, (896, 32), (32, 1), 0), reinterpret_tensor(buf1108, (32, 896), (896, 1), 0), reinterpret_tensor(buf1101, (896, 32), (32, 1), 0), reinterpret_tensor(buf1099, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf1094, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1092, (32, 896), (896, 1), 0), reinterpret_tensor(buf1088, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1086, (32, 896), (896, 1), 0), reinterpret_tensor(buf1079, (896, 32), (32, 1), 0), reinterpret_tensor(buf1076, (32, 896), (896, 1), 0), reinterpret_tensor(buf1062, (128, 32), (32, 1), 0), reinterpret_tensor(buf1060, (32, 896), (896, 1), 0), reinterpret_tensor(buf1057, (128, 32), (32, 1), 0), reinterpret_tensor(buf1055, (32, 896), (896, 1), 0), reinterpret_tensor(buf1052, (896, 32), (32, 1), 0), reinterpret_tensor(buf1050, (32, 896), (896, 1), 0), reinterpret_tensor(buf1043, (896, 32), (32, 1), 0), reinterpret_tensor(buf1041, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf1036, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1034, (32, 896), (896, 1), 0), reinterpret_tensor(buf1030, (4864, 32), (32, 1), 0), reinterpret_tensor(buf1028, (32, 896), (896, 1), 0), reinterpret_tensor(buf1021, (896, 32), (32, 1), 0), reinterpret_tensor(buf1018, (32, 896), (896, 1), 0), reinterpret_tensor(buf1004, (128, 32), (32, 1), 0), reinterpret_tensor(buf1002, (32, 896), (896, 1), 0), reinterpret_tensor(buf999, (128, 32), (32, 1), 0), reinterpret_tensor(buf997, (32, 896), (896, 1), 0), reinterpret_tensor(buf994, (896, 32), (32, 1), 0), reinterpret_tensor(buf992, (32, 896), (896, 1), 0), reinterpret_tensor(buf985, (896, 32), (32, 1), 0), reinterpret_tensor(buf983, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf978, (4864, 32), (32, 1), 0), reinterpret_tensor(buf976, (32, 896), (896, 1), 0), reinterpret_tensor(buf972, (4864, 32), (32, 1), 0), reinterpret_tensor(buf970, (32, 896), (896, 1), 0), reinterpret_tensor(buf963, (896, 32), (32, 1), 0), reinterpret_tensor(buf960, (32, 896), (896, 1), 0), reinterpret_tensor(buf946, (128, 32), (32, 1), 0), reinterpret_tensor(buf944, (32, 896), (896, 1), 0), reinterpret_tensor(buf941, (128, 32), (32, 1), 0), reinterpret_tensor(buf939, (32, 896), (896, 1), 0), reinterpret_tensor(buf936, (896, 32), (32, 1), 0), reinterpret_tensor(buf934, (32, 896), (896, 1), 0), reinterpret_tensor(buf927, (896, 32), (32, 1), 0), reinterpret_tensor(buf925, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf920, (4864, 32), (32, 1), 0), reinterpret_tensor(buf918, (32, 896), (896, 1), 0), reinterpret_tensor(buf914, (4864, 32), (32, 1), 0), reinterpret_tensor(buf912, (32, 896), (896, 1), 0), reinterpret_tensor(buf905, (896, 32), (32, 1), 0), reinterpret_tensor(buf902, (32, 896), (896, 1), 0), reinterpret_tensor(buf888, (128, 32), (32, 1), 0), reinterpret_tensor(buf886, (32, 896), (896, 1), 0), reinterpret_tensor(buf883, (128, 32), (32, 1), 0), reinterpret_tensor(buf881, (32, 896), (896, 1), 0), reinterpret_tensor(buf878, (896, 32), (32, 1), 0), reinterpret_tensor(buf876, (32, 896), (896, 1), 0), reinterpret_tensor(buf869, (896, 32), (32, 1), 0), reinterpret_tensor(buf867, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf862, (4864, 32), (32, 1), 0), reinterpret_tensor(buf860, (32, 896), (896, 1), 0), reinterpret_tensor(buf856, (4864, 32), (32, 1), 0), reinterpret_tensor(buf854, (32, 896), (896, 1), 0), reinterpret_tensor(buf847, (896, 32), (32, 1), 0), reinterpret_tensor(buf844, (32, 896), (896, 1), 0), reinterpret_tensor(buf830, (128, 32), (32, 1), 0), reinterpret_tensor(buf828, (32, 896), (896, 1), 0), reinterpret_tensor(buf825, (128, 32), (32, 1), 0), reinterpret_tensor(buf823, (32, 896), (896, 1), 0), reinterpret_tensor(buf820, (896, 32), (32, 1), 0), reinterpret_tensor(buf818, (32, 896), (896, 1), 0), reinterpret_tensor(buf811, (896, 32), (32, 1), 0), reinterpret_tensor(buf809, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf804, (4864, 32), (32, 1), 0), reinterpret_tensor(buf802, (32, 896), (896, 1), 0), reinterpret_tensor(buf798, (4864, 32), (32, 1), 0), reinterpret_tensor(buf796, (32, 896), (896, 1), 0), reinterpret_tensor(buf789, (896, 32), (32, 1), 0), reinterpret_tensor(buf786, (32, 896), (896, 1), 0), reinterpret_tensor(buf772, (128, 32), (32, 1), 0), reinterpret_tensor(buf770, (32, 896), (896, 1), 0), reinterpret_tensor(buf767, (128, 32), (32, 1), 0), reinterpret_tensor(buf765, (32, 896), (896, 1), 0), reinterpret_tensor(buf762, (896, 32), (32, 1), 0), reinterpret_tensor(buf760, (32, 896), (896, 1), 0), reinterpret_tensor(buf753, (896, 32), (32, 1), 0), reinterpret_tensor(buf751, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf746, (4864, 32), (32, 1), 0), reinterpret_tensor(buf744, (32, 896), (896, 1), 0), reinterpret_tensor(buf740, (4864, 32), (32, 1), 0), reinterpret_tensor(buf738, (32, 896), (896, 1), 0), reinterpret_tensor(buf731, (896, 32), (32, 1), 0), reinterpret_tensor(buf728, (32, 896), (896, 1), 0), reinterpret_tensor(buf714, (128, 32), (32, 1), 0), reinterpret_tensor(buf712, (32, 896), (896, 1), 0), reinterpret_tensor(buf709, (128, 32), (32, 1), 0), reinterpret_tensor(buf707, (32, 896), (896, 1), 0), reinterpret_tensor(buf704, (896, 32), (32, 1), 0), reinterpret_tensor(buf702, (32, 896), (896, 1), 0), reinterpret_tensor(buf695, (896, 32), (32, 1), 0), reinterpret_tensor(buf693, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf688, (4864, 32), (32, 1), 0), reinterpret_tensor(buf686, (32, 896), (896, 1), 0), reinterpret_tensor(buf682, (4864, 32), (32, 1), 0), reinterpret_tensor(buf680, (32, 896), (896, 1), 0), reinterpret_tensor(buf673, (896, 32), (32, 1), 0), reinterpret_tensor(buf670, (32, 896), (896, 1), 0), reinterpret_tensor(buf656, (128, 32), (32, 1), 0), reinterpret_tensor(buf654, (32, 896), (896, 1), 0), reinterpret_tensor(buf651, (128, 32), (32, 1), 0), reinterpret_tensor(buf649, (32, 896), (896, 1), 0), reinterpret_tensor(buf646, (896, 32), (32, 1), 0), reinterpret_tensor(buf644, (32, 896), (896, 1), 0), reinterpret_tensor(buf637, (896, 32), (32, 1), 0), reinterpret_tensor(buf635, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf630, (4864, 32), (32, 1), 0), reinterpret_tensor(buf628, (32, 896), (896, 1), 0), reinterpret_tensor(buf624, (4864, 32), (32, 1), 0), reinterpret_tensor(buf622, (32, 896), (896, 1), 0), reinterpret_tensor(buf615, (896, 32), (32, 1), 0), reinterpret_tensor(buf612, (32, 896), (896, 1), 0), reinterpret_tensor(buf598, (128, 32), (32, 1), 0), reinterpret_tensor(buf596, (32, 896), (896, 1), 0), reinterpret_tensor(buf593, (128, 32), (32, 1), 0), reinterpret_tensor(buf591, (32, 896), (896, 1), 0), reinterpret_tensor(buf588, (896, 32), (32, 1), 0), reinterpret_tensor(buf586, (32, 896), (896, 1), 0), reinterpret_tensor(buf579, (896, 32), (32, 1), 0), reinterpret_tensor(buf577, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf572, (4864, 32), (32, 1), 0), reinterpret_tensor(buf570, (32, 896), (896, 1), 0), reinterpret_tensor(buf566, (4864, 32), (32, 1), 0), reinterpret_tensor(buf564, (32, 896), (896, 1), 0), reinterpret_tensor(buf557, (896, 32), (32, 1), 0), reinterpret_tensor(buf554, (32, 896), (896, 1), 0), reinterpret_tensor(buf540, (128, 32), (32, 1), 0), reinterpret_tensor(buf538, (32, 896), (896, 1), 0), reinterpret_tensor(buf535, (128, 32), (32, 1), 0), reinterpret_tensor(buf533, (32, 896), (896, 1), 0), reinterpret_tensor(buf530, (896, 32), (32, 1), 0), reinterpret_tensor(buf528, (32, 896), (896, 1), 0), reinterpret_tensor(buf521, (896, 32), (32, 1), 0), reinterpret_tensor(buf519, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf514, (4864, 32), (32, 1), 0), reinterpret_tensor(buf512, (32, 896), (896, 1), 0), reinterpret_tensor(buf508, (4864, 32), (32, 1), 0), reinterpret_tensor(buf506, (32, 896), (896, 1), 0), reinterpret_tensor(buf499, (896, 32), (32, 1), 0), reinterpret_tensor(buf496, (32, 896), (896, 1), 0), reinterpret_tensor(buf482, (128, 32), (32, 1), 0), reinterpret_tensor(buf480, (32, 896), (896, 1), 0), reinterpret_tensor(buf477, (128, 32), (32, 1), 0), reinterpret_tensor(buf475, (32, 896), (896, 1), 0), reinterpret_tensor(buf472, (896, 32), (32, 1), 0), reinterpret_tensor(buf470, (32, 896), (896, 1), 0), reinterpret_tensor(buf463, (896, 32), (32, 1), 0), reinterpret_tensor(buf461, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf456, (4864, 32), (32, 1), 0), reinterpret_tensor(buf454, (32, 896), (896, 1), 0), reinterpret_tensor(buf450, (4864, 32), (32, 1), 0), reinterpret_tensor(buf448, (32, 896), (896, 1), 0), reinterpret_tensor(buf441, (896, 32), (32, 1), 0), reinterpret_tensor(buf438, (32, 896), (896, 1), 0), reinterpret_tensor(buf424, (128, 32), (32, 1), 0), reinterpret_tensor(buf422, (32, 896), (896, 1), 0), reinterpret_tensor(buf419, (128, 32), (32, 1), 0), reinterpret_tensor(buf417, (32, 896), (896, 1), 0), reinterpret_tensor(buf414, (896, 32), (32, 1), 0), reinterpret_tensor(buf412, (32, 896), (896, 1), 0), reinterpret_tensor(buf405, (896, 32), (32, 1), 0), reinterpret_tensor(buf403, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf398, (4864, 32), (32, 1), 0), reinterpret_tensor(buf396, (32, 896), (896, 1), 0), reinterpret_tensor(buf392, (4864, 32), (32, 1), 0), reinterpret_tensor(buf390, (32, 896), (896, 1), 0), reinterpret_tensor(buf383, (896, 32), (32, 1), 0), reinterpret_tensor(buf380, (32, 896), (896, 1), 0), reinterpret_tensor(buf366, (128, 32), (32, 1), 0), reinterpret_tensor(buf364, (32, 896), (896, 1), 0), reinterpret_tensor(buf361, (128, 32), (32, 1), 0), reinterpret_tensor(buf359, (32, 896), (896, 1), 0), reinterpret_tensor(buf356, (896, 32), (32, 1), 0), reinterpret_tensor(buf354, (32, 896), (896, 1), 0), reinterpret_tensor(buf347, (896, 32), (32, 1), 0), reinterpret_tensor(buf345, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf340, (4864, 32), (32, 1), 0), reinterpret_tensor(buf338, (32, 896), (896, 1), 0), reinterpret_tensor(buf334, (4864, 32), (32, 1), 0), reinterpret_tensor(buf332, (32, 896), (896, 1), 0), reinterpret_tensor(buf325, (896, 32), (32, 1), 0), reinterpret_tensor(buf322, (32, 896), (896, 1), 0), reinterpret_tensor(buf308, (128, 32), (32, 1), 0), reinterpret_tensor(buf306, (32, 896), (896, 1), 0), reinterpret_tensor(buf303, (128, 32), (32, 1), 0), reinterpret_tensor(buf301, (32, 896), (896, 1), 0), reinterpret_tensor(buf298, (896, 32), (32, 1), 0), reinterpret_tensor(buf296, (32, 896), (896, 1), 0), reinterpret_tensor(buf289, (896, 32), (32, 1), 0), reinterpret_tensor(buf287, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf282, (4864, 32), (32, 1), 0), reinterpret_tensor(buf280, (32, 896), (896, 1), 0), reinterpret_tensor(buf276, (4864, 32), (32, 1), 0), reinterpret_tensor(buf274, (32, 896), (896, 1), 0), reinterpret_tensor(buf267, (896, 32), (32, 1), 0), reinterpret_tensor(buf264, (32, 896), (896, 1), 0), reinterpret_tensor(buf250, (128, 32), (32, 1), 0), reinterpret_tensor(buf248, (32, 896), (896, 1), 0), reinterpret_tensor(buf245, (128, 32), (32, 1), 0), reinterpret_tensor(buf243, (32, 896), (896, 1), 0), reinterpret_tensor(buf240, (896, 32), (32, 1), 0), reinterpret_tensor(buf238, (32, 896), (896, 1), 0), reinterpret_tensor(buf231, (896, 32), (32, 1), 0), reinterpret_tensor(buf229, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf224, (4864, 32), (32, 1), 0), reinterpret_tensor(buf222, (32, 896), (896, 1), 0), reinterpret_tensor(buf218, (4864, 32), (32, 1), 0), reinterpret_tensor(buf216, (32, 896), (896, 1), 0), reinterpret_tensor(buf209, (896, 32), (32, 1), 0), reinterpret_tensor(buf206, (32, 896), (896, 1), 0), reinterpret_tensor(buf192, (128, 32), (32, 1), 0), reinterpret_tensor(buf190, (32, 896), (896, 1), 0), reinterpret_tensor(buf187, (128, 32), (32, 1), 0), reinterpret_tensor(buf185, (32, 896), (896, 1), 0), reinterpret_tensor(buf182, (896, 32), (32, 1), 0), reinterpret_tensor(buf180, (32, 896), (896, 1), 0), reinterpret_tensor(buf173, (896, 32), (32, 1), 0), reinterpret_tensor(buf171, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf166, (4864, 32), (32, 1), 0), reinterpret_tensor(buf164, (32, 896), (896, 1), 0), reinterpret_tensor(buf160, (4864, 32), (32, 1), 0), reinterpret_tensor(buf158, (32, 896), (896, 1), 0), reinterpret_tensor(buf151, (896, 32), (32, 1), 0), reinterpret_tensor(buf148, (32, 896), (896, 1), 0), reinterpret_tensor(buf134, (128, 32), (32, 1), 0), reinterpret_tensor(buf132, (32, 896), (896, 1), 0), reinterpret_tensor(buf129, (128, 32), (32, 1), 0), reinterpret_tensor(buf127, (32, 896), (896, 1), 0), reinterpret_tensor(buf124, (896, 32), (32, 1), 0), reinterpret_tensor(buf122, (32, 896), (896, 1), 0), reinterpret_tensor(buf115, (896, 32), (32, 1), 0), reinterpret_tensor(buf113, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf108, (4864, 32), (32, 1), 0), reinterpret_tensor(buf106, (32, 896), (896, 1), 0), reinterpret_tensor(buf102, (4864, 32), (32, 1), 0), reinterpret_tensor(buf100, (32, 896), (896, 1), 0), reinterpret_tensor(buf93, (896, 32), (32, 1), 0), reinterpret_tensor(buf90, (32, 896), (896, 1), 0), reinterpret_tensor(buf76, (128, 32), (32, 1), 0), reinterpret_tensor(buf74, (32, 896), (896, 1), 0), reinterpret_tensor(buf71, (128, 32), (32, 1), 0), reinterpret_tensor(buf69, (32, 896), (896, 1), 0), reinterpret_tensor(buf66, (896, 32), (32, 1), 0), reinterpret_tensor(buf64, (32, 896), (896, 1), 0), reinterpret_tensor(buf57, (896, 32), (32, 1), 0), reinterpret_tensor(buf55, (32, 4864), (4864, 1), 0), reinterpret_tensor(buf50, (4864, 32), (32, 1), 0), reinterpret_tensor(buf48, (32, 896), (896, 1), 0), reinterpret_tensor(buf44, (4864, 32), (32, 1), 0), reinterpret_tensor(buf42, (32, 896), (896, 1), 0), reinterpret_tensor(buf35, (896, 32), (32, 1), 0), reinterpret_tensor(buf32, (32, 896), (896, 1), 0), reinterpret_tensor(buf17, (128, 32), (32, 1), 0), reinterpret_tensor(buf12, (128, 32), (32, 1), 0), reinterpret_tensor(buf7, (896, 32), (32, 1), 0), s27, s35, (-1)*s35, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = 805
    primals_2 = rand_strided((1, 805), (805, 1), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((151936, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_4 = 805
    primals_5 = rand_strided((1, 805), (805, 1), device='cuda:0', dtype=torch.int64)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_8 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_9 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_10 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_14 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_18 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_21 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_24 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_25 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_28 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_31 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_34 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_35 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_36 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_40 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_44 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_47 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_50 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_51 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_54 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_57 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_60 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_61 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_62 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_66 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_69 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_70 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_73 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_76 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_77 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_80 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_83 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_86 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_87 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_88 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_92 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_96 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_99 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_102 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_103 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_106 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_109 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_112 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_113 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_114 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_117 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_118 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_122 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_125 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_128 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_129 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_132 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_135 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_138 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_139 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_140 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_144 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_147 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_148 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_151 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_154 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_155 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_158 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_161 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_164 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_165 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_166 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_170 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_174 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_177 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_180 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_181 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_184 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_187 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_190 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_191 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_192 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_195 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_196 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_199 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_200 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_203 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_206 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_207 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_210 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_213 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_216 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_217 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_218 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_221 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_222 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_226 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_229 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_232 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_233 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_236 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_239 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_242 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_243 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_244 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_247 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_248 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_251 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_252 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_255 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_258 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_259 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_262 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_265 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_268 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_269 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_270 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_273 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_274 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_277 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_278 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_281 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_284 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_285 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_288 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_291 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_294 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_295 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_296 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_299 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_300 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_303 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_304 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_307 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_310 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_311 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_314 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_317 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_320 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_321 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_322 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_325 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_326 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_329 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_330 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_333 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_336 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_337 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_340 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_343 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_346 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_347 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_348 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_351 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_352 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_355 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_356 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_359 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_362 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_363 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_366 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_369 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_372 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_373 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_374 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_377 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_378 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_381 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_382 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_385 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_388 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_389 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_392 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_395 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_398 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_399 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_400 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_403 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_404 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_407 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_408 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_411 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_414 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_415 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_418 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_421 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_424 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_425 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_426 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_429 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_430 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_433 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_434 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_437 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_440 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_441 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_444 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_447 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_450 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_451 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_452 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_455 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_456 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_459 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_460 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_463 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_466 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_467 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_470 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_473 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_476 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_477 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_478 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_481 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_482 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_485 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_486 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_489 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_492 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_493 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_496 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_499 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_502 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_503 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_504 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_507 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_508 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_511 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_512 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_515 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_518 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_519 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_522 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_525 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_528 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_529 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_530 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_533 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_534 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_537 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_538 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_541 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_544 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_545 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_548 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_551 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_554 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_555 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_556 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_559 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_560 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_563 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_564 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_567 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_570 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_571 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_574 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_577 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_580 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_581 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_582 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_585 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_586 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_589 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_590 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_593 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_596 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_597 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_600 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_603 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_606 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_607 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_608 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_611 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_612 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_615 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_616 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_619 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_622 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_623 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_626 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_629 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_632 = 723
    primals_633 = rand_strided((151936, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
