# AOT ID: ['18_backward']
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


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/ci/ccio5apw6irwft5y374bq7mgfhcsv7x6sar2h7xoxcyzje6tcmc7.py
# Topologically Sorted Source Nodes: [convert_element_type_1905, view_1163], Original ATen: [aten._to_copy, aten.view]
# Source node to ATen node mapping:
#   convert_element_type_1905 => convert_element_type_1905
#   view_1163 => view_1163
# Graph fragment:
#   %tangents_1 : Tensor "f32[1, 1025, 151936][155734400, 151936, 1]cuda:0" = PlaceHolder[target=tangents_1]
#   %convert_element_type_1905 : Tensor "bf16[1, 1025, 151936][155734400, 151936, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%tangents_1, torch.bfloat16), kwargs = {})
#   %view_1163 : Tensor "bf16[1025, 151936][151936, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_1905, [1025, 151936]), kwargs = {})
#   return %view_1163
triton_poi_fused__to_copy_view_0 = async_compile.triton('triton_poi_fused__to_copy_view_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 268435456}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1245875200}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 155734400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/p4/cp4igapf25uevkpfrhwlpqjuncpva2dguyjn6ueewd2p5n6tadb7.py
# Topologically Sorted Source Nodes: [view_1164, full_default_49, mul_412, convert_element_type_1910, hidden_states_240, mul_413, mul_414, sum_1, pow_50, mul_415, mul_416, expand_77, div, pow_51, mul_417, mul_418, add_314, convert_element_type_1911, mul_419, view_1165], Original ATen: [aten.view, aten.slice_backward, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div, aten.add]
# Source node to ATen node mapping:
#   add_314 => add_314
#   convert_element_type_1910 => convert_element_type_1910
#   convert_element_type_1911 => convert_element_type_1911
#   div => div
#   expand_77 => expand_77
#   full_default_49 => full_default_49
#   hidden_states_240 => convert_element_type_1900
#   mul_412 => mul_412
#   mul_413 => mul_413
#   mul_414 => mul_414
#   mul_415 => mul_415
#   mul_416 => mul_416
#   mul_417 => mul_417
#   mul_418 => mul_418
#   mul_419 => mul_419
#   pow_50 => pow_50
#   pow_51 => pow_51
#   sum_1 => sum_1
#   view_1164 => view_1164
#   view_1165 => view_1165
# Graph fragment:
#   %mm_434 : Tensor "bf16[1025, 896][896, 1]cuda:0" = PlaceHolder[target=mm_434]
#   %primals_629 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=primals_629]
#   %add_312 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0" = PlaceHolder[target=add_312]
#   %rsqrt_48 : Tensor "f32[1, 1218, 1][1248, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_48]
#   %sum_1 : Tensor "f32[1, 1218, 1][1248, 1, 1248]cuda:0" = PlaceHolder[target=sum_1]
#   %convert_element_type_1911 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0" = PlaceHolder[target=convert_element_type_1911]
#   %view_1164 : Tensor "bf16[1, 1025, 896][918400, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_434, [1, 1025, 896]), kwargs = {})
#   %full_default_49 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([1, 1218, 896], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_49, %view_1164, 1, -1025, 9223372036854775807), kwargs = {})
#   %mul_412 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_scatter_default, %primals_629), kwargs = {})
#   %convert_element_type_1910 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_412, torch.float32), kwargs = {})
#   %convert_element_type_1900 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_312, torch.float32), kwargs = {})
#   %mul_413 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1910, %convert_element_type_1900), kwargs = {})
#   %mul_414 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1910, %rsqrt_48), kwargs = {})
#   %sum_1 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_413, [2], True), kwargs = {})
#   %pow_50 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt_48, 3), kwargs = {})
#   %mul_415 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%sum_1, -0.5), kwargs = {})
#   %mul_416 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_415, %pow_50), kwargs = {})
#   %expand_77 : Tensor "f32[1, 1218, 896][1218, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_416, [1, 1218, 896]), kwargs = {})
#   %div : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_77, 896), kwargs = {})
#   %pow_51 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1900, 1.0), kwargs = {})
#   %mul_417 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_51, 2.0), kwargs = {})
#   %mul_418 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mul_417), kwargs = {})
#   %add_314 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_414, %mul_418), kwargs = {})
#   %convert_element_type_1911 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_314, torch.bfloat16), kwargs = {})
#   %mul_419 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1911, 1.0), kwargs = {})
#   %view_1165 : Tensor "bf16[1218, 896][896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_419, [1218, 896]), kwargs = {})
#   return %sum_1,%convert_element_type_1911,%view_1165
triton_per_fused__to_copy_add_div_expand_mul_pow_slice_backward_sum_view_1 = async_compile.triton('triton_per_fused__to_copy_add_div_expand_mul_pow_slice_backward_sum_view_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_expand_mul_pow_slice_backward_sum_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4872, 'r0_': 12751872}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_expand_mul_pow_slice_backward_sum_view_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1218
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
    tmp6 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tl.load(in_out_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 193, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-172928) + r0_1 + 896*x0), r0_mask & tmp2 & xmask, other=0.0).to(tl.float32)
    tmp4 = 0.0
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp17 = tmp8 * tmp16
    tmp18 = -0.5
    tmp19 = tmp15 * tmp18
    tmp20 = tmp16 * tmp16
    tmp21 = tmp20 * tmp16
    tmp22 = tmp19 * tmp21
    tmp23 = 0.0011160714285714285
    tmp24 = tmp22 * tmp23
    tmp25 = 2.0
    tmp26 = tmp10 * tmp25
    tmp27 = tmp24 * tmp26
    tmp28 = tmp17 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tl.store(in_out_ptr0 + (r0_1 + 896*x0), tmp29, r0_mask & xmask)
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp31, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/ow/cow5jp5rckwquqpblxsz7od4yhnv6rl2wci64rk27wwqocb54krp.py
# Topologically Sorted Source Nodes: [convert_element_type_1916], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_1916 => convert_element_type_1916
# Graph fragment:
#   %mm_435 : Tensor "bf16[896, 32][32, 1]cuda:0" = PlaceHolder[target=mm_435]
#   %convert_element_type_1916 : Tensor "f32[896, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_435, torch.float32), kwargs = {})
#   return %convert_element_type_1916
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 286720}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/n4/cn4fy3ocmmtsxwbjf4vryuzr2ua6jgia7uvrpzfo5iovpe77syq3.py
# Topologically Sorted Source Nodes: [convert_element_type_1922], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_1922 => convert_element_type_1922
# Graph fragment:
#   %mm_437 : Tensor "bf16[32, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_437]
#   %convert_element_type_1922 : Tensor "f32[32, 4864][4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_437, torch.float32), kwargs = {})
#   return %convert_element_type_1922
triton_poi_fused__to_copy_3 = async_compile.triton('triton_poi_fused__to_copy_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 1556480}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 155648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/u5/cu5fw67qtmrqremv63srxfbcrbs2wsx7ahutzn4aeschwfdsgluv.py
# Topologically Sorted Source Nodes: [view_1168, view_1170, add_315, silu_23, mul_420, mul_421, mul_422, convert_element_type_1940, neg_48, exp, add_317, reciprocal, mul_423, mul_424, sub, mul_425, add_318, mul_426, convert_element_type_1942, mul_427], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
# Source node to ATen node mapping:
#   add_315 => add_315
#   add_317 => add_317
#   add_318 => add_318
#   convert_element_type_1940 => convert_element_type_1940
#   convert_element_type_1942 => convert_element_type_1942
#   exp => exp
#   mul_420 => mul_420
#   mul_421 => mul_421
#   mul_422 => mul_422
#   mul_423 => mul_423
#   mul_424 => mul_424
#   mul_425 => mul_425
#   mul_426 => mul_426
#   mul_427 => mul_427
#   neg_48 => neg_48
#   reciprocal => reciprocal
#   silu_23 => convert_element_type_1878, convert_element_type_1879, mul_406, sigmoid_23
#   sub => sub
#   view_1168 => view_1168
#   view_1170 => view_1170
# Graph fragment:
#   %mm_438 : Tensor "bf16[1218, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_438]
#   %mm_439 : Tensor "bf16[1218, 4864][4864, 1]cuda:0" = PlaceHolder[target=mm_439]
#   %add_309 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0" = PlaceHolder[target=add_309]
#   %add_310 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0" = PlaceHolder[target=add_310]
#   %view_1168 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_438, [1, 1218, 4864]), kwargs = {})
#   %view_1170 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_439, [1, 1218, 4864]), kwargs = {})
#   %add_315 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1168, %view_1170), kwargs = {})
#   %convert_element_type_1878 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_309, torch.float32), kwargs = {})
#   %sigmoid_23 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1878,), kwargs = {})
#   %mul_406 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1878, %sigmoid_23), kwargs = {})
#   %convert_element_type_1879 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_406, torch.bfloat16), kwargs = {})
#   %mul_420 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_315, %convert_element_type_1879), kwargs = {})
#   %mul_421 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_315, %add_310), kwargs = {})
#   %mul_422 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_420, 1.0), kwargs = {})
#   %convert_element_type_1940 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_421, torch.float32), kwargs = {})
#   %neg_48 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%convert_element_type_1878,), kwargs = {})
#   %exp : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg_48,), kwargs = {})
#   %add_317 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %reciprocal : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%add_317,), kwargs = {})
#   %mul_423 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %mul_424 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1940, %mul_423), kwargs = {})
#   %sub : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %mul_423), kwargs = {})
#   %mul_425 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1878, %sub), kwargs = {})
#   %add_318 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_425, 1), kwargs = {})
#   %mul_426 : Tensor "f32[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_424, %add_318), kwargs = {})
#   %convert_element_type_1942 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_426, torch.bfloat16), kwargs = {})
#   %mul_427 : Tensor "bf16[1, 1218, 4864][5924352, 4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1942, 1.0), kwargs = {})
#   return %mul_422,%mul_420,%mul_427,%convert_element_type_1942
triton_poi_fused_add_mul_silu_silu_backward_view_4 = async_compile.triton('triton_poi_fused_add_mul_silu_silu_backward_view_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_silu_silu_backward_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 4, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 142184448}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_silu_silu_backward_view_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5924352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp2 * tmp7
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp12 = tmp2 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = -tmp4
    tmp15 = libdevice.exp(tmp14)
    tmp16 = tmp15 + tmp9
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = (tmp17 / tmp16)
    tmp19 = tmp18 * tmp9
    tmp20 = tmp13 * tmp19
    tmp21 = tmp9 - tmp19
    tmp22 = tmp4 * tmp21
    tmp23 = tmp22 + tmp9
    tmp24 = tmp20 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp9
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp8, xmask)
    tl.store(out_ptr2 + (x0), tmp26, xmask)
    tl.store(out_ptr3 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/j5/cj5kupozbwtw2co7wb65pt3y7nhbax63c5z4f3b2jzdsibcfvz2f.py
# Topologically Sorted Source Nodes: [view_1174, view_1176, add_316, view_1180, add_319, view_1182, add_320, mul_428, convert_element_type_1957, hidden_states_236, mul_429, mul_430, sum_2, pow_52, mul_431, mul_432, expand_78, div_1, pow_53, mul_433, mul_434, add_321, convert_element_type_1958, add_322, mul_435, view_1183], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
# Source node to ATen node mapping:
#   add_316 => add_316
#   add_319 => add_319
#   add_320 => add_320
#   add_321 => add_321
#   add_322 => add_322
#   convert_element_type_1957 => convert_element_type_1957
#   convert_element_type_1958 => convert_element_type_1958
#   div_1 => div_1
#   expand_78 => expand_78
#   hidden_states_236 => convert_element_type_1866
#   mul_428 => mul_428
#   mul_429 => mul_429
#   mul_430 => mul_430
#   mul_431 => mul_431
#   mul_432 => mul_432
#   mul_433 => mul_433
#   mul_434 => mul_434
#   mul_435 => mul_435
#   pow_52 => pow_52
#   pow_53 => pow_53
#   sum_2 => sum_2
#   view_1174 => view_1174
#   view_1176 => view_1176
#   view_1180 => view_1180
#   view_1182 => view_1182
#   view_1183 => view_1183
# Graph fragment:
#   %mm_443 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_443]
#   %mm_444 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_444]
#   %mm_448 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_448]
#   %mm_449 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_449]
#   %primals_619 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=primals_619]
#   %convert_element_type_1957 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0" = PlaceHolder[target=convert_element_type_1957]
#   %add_307 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0" = PlaceHolder[target=add_307]
#   %convert_element_type_1911 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0" = PlaceHolder[target=convert_element_type_1911]
#   %rsqrt_47 : Tensor "f32[1, 1218, 1][1248, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_47]
#   %sum_2 : Tensor "f32[1, 1218, 1][1248, 1, 1248]cuda:0" = PlaceHolder[target=sum_2]
#   %add_322 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0" = PlaceHolder[target=add_322]
#   %view_1174 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_443, [1, 1218, 896]), kwargs = {})
#   %view_1176 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_444, [1, 1218, 896]), kwargs = {})
#   %add_316 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1174, %view_1176), kwargs = {})
#   %view_1180 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_448, [1, 1218, 896]), kwargs = {})
#   %add_319 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_316, %view_1180), kwargs = {})
#   %view_1182 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_449, [1, 1218, 896]), kwargs = {})
#   %add_320 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_319, %view_1182), kwargs = {})
#   %mul_428 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_320, %primals_619), kwargs = {})
#   %convert_element_type_1957 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_428, torch.float32), kwargs = {})
#   %convert_element_type_1866 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_307, torch.float32), kwargs = {})
#   %mul_429 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1957, %convert_element_type_1866), kwargs = {})
#   %mul_430 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1957, %rsqrt_47), kwargs = {})
#   %sum_2 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_429, [2], True), kwargs = {})
#   %pow_52 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt_47, 3), kwargs = {})
#   %mul_431 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%sum_2, -0.5), kwargs = {})
#   %mul_432 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_431, %pow_52), kwargs = {})
#   %expand_78 : Tensor "f32[1, 1218, 896][1218, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_432, [1, 1218, 896]), kwargs = {})
#   %div_1 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_78, 896), kwargs = {})
#   %pow_53 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1866, 1.0), kwargs = {})
#   %mul_433 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_53, 2.0), kwargs = {})
#   %mul_434 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_1, %mul_433), kwargs = {})
#   %add_321 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_430, %mul_434), kwargs = {})
#   %convert_element_type_1958 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_321, torch.bfloat16), kwargs = {})
#   %add_322 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_1911, %convert_element_type_1958), kwargs = {})
#   %mul_435 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_322, 1.0), kwargs = {})
#   %view_1183 : Tensor "bf16[1218, 896][896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_435, [1218, 896]), kwargs = {})
#   return %convert_element_type_1957,%sum_2,%add_322,%view_1183
triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5 = async_compile.triton('triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4872, 'r0_': 21828352}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1218
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr5 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_out_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp19 = tmp9 * tmp18
    tmp20 = -0.5
    tmp21 = tmp16 * tmp20
    tmp22 = tmp18 * tmp18
    tmp23 = tmp22 * tmp18
    tmp24 = tmp21 * tmp23
    tmp25 = 0.0011160714285714285
    tmp26 = tmp24 * tmp25
    tmp27 = 2.0
    tmp28 = tmp11 * tmp27
    tmp29 = tmp26 * tmp28
    tmp30 = tmp19 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp17 + tmp31
    tmp33 = 1.0
    tmp34 = tmp32 * tmp33
    tl.store(in_out_ptr0 + (r0_1 + 896*x0), tmp32, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 896*x0), tmp34, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/lu/cluwjzlnq6pxl33aqdzxp2br5qo35gvu2fzlqev53qrnxsssfeny.py
# Topologically Sorted Source Nodes: [view_1186, view_1188, add_323, view_1189, permute_642, attn_output, _scaled_dot_product_efficient_attention_backward], Original ATen: [aten.view, aten.add, aten.transpose, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention_backward]
# Source node to ATen node mapping:
#   _scaled_dot_product_efficient_attention_backward => _scaled_dot_product_efficient_attention_backward
#   add_323 => add_323
#   attn_output => expand_7, slice_5
#   permute_642 => permute_642
#   view_1186 => view_1186
#   view_1188 => view_1188
#   view_1189 => view_1189
# Graph fragment:
#   %mm_453 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_453]
#   %mm_454 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_454]
#   %view_1186 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_453, [1, 1218, 896]), kwargs = {})
#   %view_1188 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_454, [1, 1218, 896]), kwargs = {})
#   %add_323 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1186, %view_1188), kwargs = {})
#   %view_1189 : Tensor "bf16[1, 1218, 14, 64][1091328, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add_323, [1, 1218, 14, 64]), kwargs = {})
#   %permute_642 : Tensor "bf16[1, 14, 1218, 64][1091328, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_1189, [0, 2, 1, 3]), kwargs = {})
#   %slice_5 : Tensor "bf16[1, 1, 1218, 1218][1559040, 1559040, 1280, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%constant_pad_nd, -1, 0, 1218), kwargs = {})
#   %expand_7 : Tensor "bf16[1, 14, 1218, 1218][1559040, 0, 1280, 1]cuda:0"[num_users=24] = call_function[target=torch.ops.aten.expand.default](args = (%slice_5, [1, 14, 1218, 1218]), kwargs = {})
#   %_scaled_dot_product_efficient_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention_backward.default](args = (%permute_642, %add_304, %view_1134, %view_1135, %expand_7, %getitem_92, %getitem_93, %getitem_94, %getitem_95, 0.0, [True, True, True, False]), kwargs = {scale: 0.125})
#   return %buf42
triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6 = async_compile.triton('triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 8730624}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1091328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/4l/c4lf6fxtzc5paezogv26hbyozhr7sdcgi4wco7sn5zoie6aioeuo.py
# Topologically Sorted Source Nodes: [view_1191, sum_4], Original ATen: [aten.view, aten.sum]
# Source node to ATen node mapping:
#   sum_4 => sum_4
#   view_1191 => view_1191
# Graph fragment:
#   %getitem_97 : Tensor "bf16[1, 14, 1218, 64][1091328, 64, 896, 1]cuda:0" = PlaceHolder[target=getitem_97]
#   %view_1191 : Tensor "bf16[1, 2, 7, 1218, 64][1091328, 448, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_97, [1, 2, 7, 1218, 64]), kwargs = {})
#   %sum_4 : Tensor "bf16[1, 2, 1, 1218, 64][155904, 77952, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_1191, [2], True), kwargs = {})
#   return %sum_4
triton_poi_fused_sum_view_7 = async_compile.triton('triton_poi_fused_sum_view_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2806272}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sum_view_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 155904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 448*x1), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 448*x1), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 448*x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (192 + x0 + 448*x1), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (256 + x0 + 448*x1), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (320 + x0 + 448*x1), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (384 + x0 + 448*x1), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tl.store(out_ptr0 + (x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/ux/cuxfq5rpterr4gmlursbzv6z7jeylpsf2taej36f3lr3jb4anbcm.py
# Topologically Sorted Source Nodes: [view_1190, sum_3, squeeze, permute_643, clone_50, view_1192, mul_440, view_1193], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
# Source node to ATen node mapping:
#   clone_50 => clone_50
#   mul_440 => mul_440
#   permute_643 => permute_643
#   squeeze => squeeze
#   sum_3 => sum_3
#   view_1190 => view_1190
#   view_1192 => view_1192
#   view_1193 => view_1193
# Graph fragment:
#   %getitem_98 : Tensor "bf16[1, 14, 1218, 64][1091328, 64, 896, 1]cuda:0" = PlaceHolder[target=getitem_98]
#   %clone_50 : Tensor "bf16[1, 1218, 2, 64][155904, 128, 64, 1]cuda:0" = PlaceHolder[target=clone_50]
#   %view_1190 : Tensor "bf16[1, 2, 7, 1218, 64][1091328, 448, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_98, [1, 2, 7, 1218, 64]), kwargs = {})
#   %sum_3 : Tensor "bf16[1, 2, 1, 1218, 64][155904, 77952, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_1190, [2], True), kwargs = {})
#   %squeeze : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_3, 2), kwargs = {})
#   %permute_643 : Tensor "bf16[1, 1218, 2, 64][155904, 64, 77952, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%squeeze, [0, 2, 1, 3]), kwargs = {})
#   %clone_50 : Tensor "bf16[1, 1218, 2, 64][155904, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_643,), kwargs = {memory_format: torch.contiguous_format})
#   %view_1192 : Tensor "bf16[1, 1218, 128][155904, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_50, [1, 1218, 128]), kwargs = {})
#   %mul_440 : Tensor "bf16[1, 1218, 128][155904, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1192, 1.0), kwargs = {})
#   %view_1193 : Tensor "bf16[1218, 128][128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_440, [1218, 128]), kwargs = {})
#   return %clone_50,%view_1193
triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8 = async_compile.triton('triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 3741696}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 155904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = xindex // 64
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 448*x1), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + 448*x1), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + 448*x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (192 + x0 + 448*x1), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (256 + x0 + 448*x1), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (320 + x0 + 448*x1), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (384 + x0 + 448*x1), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr0 + (x2), tmp12, xmask)
    tl.store(out_ptr1 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/p4/cp4ffpfmfciyf4kcfkhddm3xgyeaybbhfltsfvc5l4lzhv2mqeai.py
# Topologically Sorted Source Nodes: [convert_element_type_1977], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   convert_element_type_1977 => convert_element_type_1977
# Graph fragment:
#   %mm_455 : Tensor "bf16[128, 32][32, 1]cuda:0" = PlaceHolder[target=mm_455]
#   %convert_element_type_1977 : Tensor "f32[128, 32][32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_455, torch.float32), kwargs = {})
#   return %convert_element_type_1977
triton_poi_fused__to_copy_9 = async_compile.triton('triton_poi_fused__to_copy_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 40960}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/gx/cgxc6jktoiy2cj6om6zzwidmp4buzgbyvwmj7oaenozq2f2fxlnx.py
# Topologically Sorted Source Nodes: [view_1191, sum_4, squeeze_1, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, mul_436, slice_122, slice_123, neg_49, full_default_50, add_324, cos, cos_1, cos_2, cos_3, mul_437, add_325, permute_653, clone_51, view_1199, mul_441], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.cos, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   add_324 => add_324
#   add_325 => add_325
#   clone_51 => clone_51
#   cos => cos
#   cos_1 => mul
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_8
#   freqs => permute
#   full_default_50 => full_default_50
#   matmul => unsqueeze_default
#   mul_436 => mul_436
#   mul_437 => mul_437
#   mul_441 => mul_441
#   neg_49 => neg_49
#   permute_653 => permute_653
#   sin => sin
#   sin_1 => mul_1
#   sin_2 => convert_element_type_3
#   sin_3 => unsqueeze_6
#   slice_122 => slice_122
#   slice_123 => slice_123
#   squeeze_1 => squeeze_1
#   sum_4 => sum_4
#   view_1191 => view_1191
#   view_1199 => view_1199
# Graph fragment:
#   %sum_4 : Tensor "bf16[1, 2, 1, 1218, 64][155904, 64, 155904, 128, 1]cuda:0" = PlaceHolder[target=sum_4]
#   %mm_default : Tensor "f32[32, 1218][1248, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %view_1191 : Tensor "bf16[1, 2, 7, 1218, 64][1091328, 448, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_97, [1, 2, 7, 1218, 64]), kwargs = {})
#   %sum_4 : Tensor "bf16[1, 2, 1, 1218, 64][155904, 77952, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_1191, [2], True), kwargs = {})
#   %squeeze_1 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_4, 2), kwargs = {})
#   %unsqueeze_default : Tensor "f32[1, 32, 1218][39936, 1248, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, 1218, 32][39936, 1, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 1218, 1, 32][39936, 1, 39936, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1218, 2, 32][39936, 1, 0, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, 1218, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1218, 2, 32][77952, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1218, 64]), kwargs = {})
#   %sin : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_8,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, 1218, 64][77952, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 1), kwargs = {})
#   %mul_436 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, %unsqueeze_6), kwargs = {})
#   %slice_122 : Tensor "bf16[1, 2, 1218, 32][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_436, 3, 0, 32), kwargs = {})
#   %slice_123 : Tensor "bf16[1, 2, 1218, 32][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_436, 3, 32, 64), kwargs = {})
#   %neg_49 : Tensor "bf16[1, 2, 1218, 32][77952, 38976, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_122,), kwargs = {})
#   %full_default_50 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.full.default](args = ([1, 2, 1218, 64], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_1 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_50, %neg_49, 3, 32, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_2 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_50, %slice_123, 3, 0, 32), kwargs = {})
#   %add_324 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_1, %slice_scatter_default_2), kwargs = {})
#   %cos : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_8,), kwargs = {})
#   %mul : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 1218, 64][77952, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_437 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_1, %unsqueeze_5), kwargs = {})
#   %add_325 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_324, %mul_437), kwargs = {})
#   %permute_653 : Tensor "bf16[1, 1218, 2, 64][155904, 64, 77952, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_325, [0, 2, 1, 3]), kwargs = {})
#   %clone_51 : Tensor "bf16[1, 1218, 2, 64][155904, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_653,), kwargs = {memory_format: torch.contiguous_format})
#   %view_1199 : Tensor "bf16[1, 1218, 128][155904, 128, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_51, [1, 1218, 128]), kwargs = {})
#   %mul_441 : Tensor "bf16[1, 1218, 128][155904, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1199, 1.0), kwargs = {})
#   return %mul_441,%clone_51
triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 11, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 155904, 'x': 2182656}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1218
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    x2 = (xindex % 64)
    tmp27 = tl.load(in_ptr0 + (x1 + 128*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (y0 + 1248*((((x1 % 64)) % 32))), xmask & ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr1 + (y0 + 1248*((x2 % 32))), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (x1 % 64)
    tmp1 = tl.full([1, 1], 32, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-32) + x1 + 128*y0), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (y0 + 1248*((((x1 % 64)) % 32))), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl_math.sin(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp3 * tmp8
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp12, tmp13)
    tmp15 = tmp0 < tmp1
    tmp16 = tl.load(in_ptr0 + (32 + x1 + 128*y0), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr1 + (y0 + 1248*((((x1 % 64)) % 32))), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl_math.sin(tmp17)
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp16 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tl.where(tmp15, tmp24, tmp13)
    tmp26 = tmp14 + tmp25
    tmp29 = tl_math.cos(tmp28)
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tmp34 = tmp26 + tmp33
    tmp35 = tmp34 * tmp30
    tmp36 = x2
    tmp37 = tmp36 >= tmp1
    tmp38 = tl.load(in_ptr0 + ((-32) + x1 + 128*y0), tmp37 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (y0 + 1248*((x2 % 32))), tmp37 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl_math.sin(tmp39)
    tmp41 = 1.0
    tmp42 = tmp40 * tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp38 * tmp43
    tmp45 = -tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp37, tmp45, tmp46)
    tmp48 = tl.where(tmp37, tmp47, tmp13)
    tmp49 = tmp36 < tmp1
    tmp50 = tl.load(in_ptr0 + (32 + x1 + 128*y0), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp51 = tl.load(in_ptr1 + (y0 + 1248*((x2 % 32))), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl_math.sin(tmp51)
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp50 * tmp55
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp49, tmp56, tmp57)
    tmp59 = tl.where(tmp49, tmp58, tmp13)
    tmp60 = tmp48 + tmp59
    tmp62 = tl_math.cos(tmp61)
    tmp63 = tmp62 * tmp30
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp27 * tmp64
    tmp66 = tmp60 + tmp65
    tl.store(out_ptr0 + (x1 + 128*y0), tmp35, xmask & ymask)
    tl.store(out_ptr1 + (x1 + 128*y0), tmp66, xmask & ymask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/ua/cuaczlgsxxg7k65kafgjegtemduhnpnl3uy6xfrgzkjj3eqclnvo.py
# Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, mul_438, slice_124, slice_125, neg_50, full_default_52, add_326, mul_439, add_327, permute_663, clone_52, view_1206, mul_442], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   add_326 => add_326
#   add_327 => add_327
#   clone_52 => clone_52
#   cos => cos
#   cos_1 => mul
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_8
#   freqs => permute
#   full_default_52 => full_default_52
#   matmul => unsqueeze_default
#   mul_438 => mul_438
#   mul_439 => mul_439
#   mul_442 => mul_442
#   neg_50 => neg_50
#   permute_663 => permute_663
#   sin => sin
#   sin_1 => mul_1
#   sin_2 => convert_element_type_3
#   sin_3 => unsqueeze_6
#   slice_124 => slice_124
#   slice_125 => slice_125
#   view_1206 => view_1206
# Graph fragment:
#   %getitem_96 : Tensor "bf16[1, 14, 1218, 64][1091328, 64, 896, 1]cuda:0" = PlaceHolder[target=getitem_96]
#   %mm_default : Tensor "f32[32, 1218][1248, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %unsqueeze_default : Tensor "f32[1, 32, 1218][39936, 1248, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, 1218, 32][39936, 1, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 1218, 1, 32][39936, 1, 39936, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1218, 2, 32][39936, 1, 0, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, 1218, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1218, 2, 32][77952, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1218, 64]), kwargs = {})
#   %sin : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_8,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, 1218, 64][77952, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 1), kwargs = {})
#   %cos : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_8,), kwargs = {})
#   %mul : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 1218, 64][77952, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %mul_438 : Tensor "bf16[1, 14, 1218, 64][1091328, 64, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_96, %unsqueeze_6), kwargs = {})
#   %slice_124 : Tensor "bf16[1, 14, 1218, 32][1091328, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_438, 3, 0, 32), kwargs = {})
#   %slice_125 : Tensor "bf16[1, 14, 1218, 32][1091328, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_438, 3, 32, 64), kwargs = {})
#   %neg_50 : Tensor "bf16[1, 14, 1218, 32][545664, 32, 448, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_124,), kwargs = {})
#   %full_default_52 : Tensor "bf16[1, 14, 1218, 64][1091328, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.full.default](args = ([1, 14, 1218, 64], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %slice_scatter_default_3 : Tensor "bf16[1, 14, 1218, 64][1091328, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_52, %neg_50, 3, 32, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_4 : Tensor "bf16[1, 14, 1218, 64][1091328, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_52, %slice_125, 3, 0, 32), kwargs = {})
#   %add_326 : Tensor "bf16[1, 14, 1218, 64][1091328, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_3, %slice_scatter_default_4), kwargs = {})
#   %mul_439 : Tensor "bf16[1, 14, 1218, 64][1091328, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_96, %unsqueeze_5), kwargs = {})
#   %add_327 : Tensor "bf16[1, 14, 1218, 64][1091328, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_326, %mul_439), kwargs = {})
#   %permute_663 : Tensor "bf16[1, 1218, 14, 64][1091328, 64, 77952, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_327, [0, 2, 1, 3]), kwargs = {})
#   %clone_52 : Tensor "bf16[1, 1218, 14, 64][1091328, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_663,), kwargs = {memory_format: torch.contiguous_format})
#   %view_1206 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_52, [1, 1218, 896]), kwargs = {})
#   %mul_442 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1206, 1.0), kwargs = {})
#   return %mul_442,%clone_52
triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 11, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 15278592}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1091328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = (xindex % 896)
    x1 = xindex // 896
    x2 = (xindex % 64)
    tmp27 = tl.load(in_ptr0 + (x4), xmask).to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (x1 + 1248*((((x0 % 64)) % 32))), xmask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr1 + (x1 + 1248*((x2 % 32))), xmask, eviction_policy='evict_last')
    tmp0 = (x4 % 64)
    tmp1 = tl.full([1], 32, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-32) + x4), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x1 + 1248*((((x0 % 64)) % 32))), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl_math.sin(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp3 * tmp8
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp12, tmp13)
    tmp15 = tmp0 < tmp1
    tmp16 = tl.load(in_ptr0 + (32 + x4), tmp15 & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr1 + (x1 + 1248*((((x0 % 64)) % 32))), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl_math.sin(tmp17)
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp16 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tl.where(tmp15, tmp24, tmp13)
    tmp26 = tmp14 + tmp25
    tmp29 = tl_math.cos(tmp28)
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tmp34 = tmp26 + tmp33
    tmp35 = tmp34 * tmp30
    tmp36 = x2
    tmp37 = tmp36 >= tmp1
    tmp38 = tl.load(in_ptr0 + ((-32) + x4), tmp37 & xmask, other=0.0).to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (x1 + 1248*((x2 % 32))), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl_math.sin(tmp39)
    tmp41 = 1.0
    tmp42 = tmp40 * tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp38 * tmp43
    tmp45 = -tmp44
    tmp46 = tl.full(tmp45.shape, 0.0, tmp45.dtype)
    tmp47 = tl.where(tmp37, tmp45, tmp46)
    tmp48 = tl.where(tmp37, tmp47, tmp13)
    tmp49 = tmp36 < tmp1
    tmp50 = tl.load(in_ptr0 + (32 + x4), tmp49 & xmask, other=0.0).to(tl.float32)
    tmp51 = tl.load(in_ptr1 + (x1 + 1248*((x2 % 32))), tmp49 & xmask, eviction_policy='evict_last', other=0.0)
    tmp52 = tl_math.sin(tmp51)
    tmp53 = 1.0
    tmp54 = tmp52 * tmp53
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp50 * tmp55
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp49, tmp56, tmp57)
    tmp59 = tl.where(tmp49, tmp58, tmp13)
    tmp60 = tmp48 + tmp59
    tmp62 = tl_math.cos(tmp61)
    tmp63 = tmp62 * tmp30
    tmp64 = tmp63.to(tl.float32)
    tmp65 = tmp27 * tmp64
    tmp66 = tmp60 + tmp65
    tl.store(out_ptr0 + (x4), tmp35, xmask)
    tl.store(out_ptr1 + (x4), tmp66, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/f4/cf4yqstlkea6ovw46ajc3dhu5pxckq23vczkxynvilggngg4lupz.py
# Topologically Sorted Source Nodes: [view_1196, view_1198, add_328, view_1203, add_329, view_1205, add_330, view_1210, add_331, view_1212, add_332, mul_443, convert_element_type_2015, hidden_states_230, mul_444, mul_445, sum_5, pow_54, mul_446, mul_447, expand_79, div_2, pow_55, mul_448, mul_449, add_333, convert_element_type_2016, add_334, mul_450, view_1213], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
# Source node to ATen node mapping:
#   add_328 => add_328
#   add_329 => add_329
#   add_330 => add_330
#   add_331 => add_331
#   add_332 => add_332
#   add_333 => add_333
#   add_334 => add_334
#   convert_element_type_2015 => convert_element_type_2015
#   convert_element_type_2016 => convert_element_type_2016
#   div_2 => div_2
#   expand_79 => expand_79
#   hidden_states_230 => convert_element_type_1821
#   mul_443 => mul_443
#   mul_444 => mul_444
#   mul_445 => mul_445
#   mul_446 => mul_446
#   mul_447 => mul_447
#   mul_448 => mul_448
#   mul_449 => mul_449
#   mul_450 => mul_450
#   pow_54 => pow_54
#   pow_55 => pow_55
#   sum_5 => sum_5
#   view_1196 => view_1196
#   view_1198 => view_1198
#   view_1203 => view_1203
#   view_1205 => view_1205
#   view_1210 => view_1210
#   view_1212 => view_1212
#   view_1213 => view_1213
# Graph fragment:
#   %mm_458 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_458]
#   %mm_459 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_459]
#   %mm_463 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_463]
#   %mm_464 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_464]
#   %mm_468 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_468]
#   %mm_469 : Tensor "bf16[1218, 896][896, 1]cuda:0" = PlaceHolder[target=mm_469]
#   %primals_603 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=primals_603]
#   %convert_element_type_2015 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0" = PlaceHolder[target=convert_element_type_2015]
#   %add_299 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0" = PlaceHolder[target=add_299]
#   %add_322 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0" = PlaceHolder[target=add_322]
#   %rsqrt_46 : Tensor "f32[1, 1218, 1][1248, 1, 1]cuda:0" = PlaceHolder[target=rsqrt_46]
#   %sum_5 : Tensor "f32[1, 1218, 1][1248, 1, 1248]cuda:0" = PlaceHolder[target=sum_5]
#   %add_334 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0" = PlaceHolder[target=add_334]
#   %view_1196 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_458, [1, 1218, 896]), kwargs = {})
#   %view_1198 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_459, [1, 1218, 896]), kwargs = {})
#   %add_328 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1196, %view_1198), kwargs = {})
#   %view_1203 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_463, [1, 1218, 896]), kwargs = {})
#   %add_329 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_328, %view_1203), kwargs = {})
#   %view_1205 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_464, [1, 1218, 896]), kwargs = {})
#   %add_330 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_329, %view_1205), kwargs = {})
#   %view_1210 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_468, [1, 1218, 896]), kwargs = {})
#   %add_331 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_330, %view_1210), kwargs = {})
#   %view_1212 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mm_469, [1, 1218, 896]), kwargs = {})
#   %add_332 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_331, %view_1212), kwargs = {})
#   %mul_443 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_332, %primals_603), kwargs = {})
#   %convert_element_type_2015 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_443, torch.float32), kwargs = {})
#   %convert_element_type_1821 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_299, torch.float32), kwargs = {})
#   %mul_444 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2015, %convert_element_type_1821), kwargs = {})
#   %mul_445 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2015, %rsqrt_46), kwargs = {})
#   %sum_5 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_444, [2], True), kwargs = {})
#   %pow_54 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%rsqrt_46, 3), kwargs = {})
#   %mul_446 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%sum_5, -0.5), kwargs = {})
#   %mul_447 : Tensor "f32[1, 1218, 1][1218, 1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_446, %pow_54), kwargs = {})
#   %expand_79 : Tensor "f32[1, 1218, 896][1218, 1, 0]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%mul_447, [1, 1218, 896]), kwargs = {})
#   %div_2 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.div.Scalar](args = (%expand_79, 896), kwargs = {})
#   %pow_55 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_1821, 1.0), kwargs = {})
#   %mul_448 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%pow_55, 2.0), kwargs = {})
#   %mul_449 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %mul_448), kwargs = {})
#   %add_333 : Tensor "f32[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_445, %mul_449), kwargs = {})
#   %convert_element_type_2016 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_333, torch.bfloat16), kwargs = {})
#   %add_334 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_322, %convert_element_type_2016), kwargs = {})
#   %mul_450 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_334, 1.0), kwargs = {})
#   %view_1213 : Tensor "bf16[1218, 896][896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%mul_450, [1218, 896]), kwargs = {})
#   return %convert_element_type_2015,%sum_5,%add_334,%view_1213
triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12 = async_compile.triton('triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*bf16', 'in_ptr8': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 4872, 'r0_': 26193664}}
)
@triton.jit
def triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1218
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
    tmp1 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp9 = tl.load(in_ptr5 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr6 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr7 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.load(in_out_ptr0 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp23 = tmp13 * tmp22
    tmp24 = -0.5
    tmp25 = tmp20 * tmp24
    tmp26 = tmp22 * tmp22
    tmp27 = tmp26 * tmp22
    tmp28 = tmp25 * tmp27
    tmp29 = 0.0011160714285714285
    tmp30 = tmp28 * tmp29
    tmp31 = 2.0
    tmp32 = tmp15 * tmp31
    tmp33 = tmp30 * tmp32
    tmp34 = tmp23 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp21 + tmp35
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tl.store(in_out_ptr0 + (r0_1 + 896*x0), tmp36, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 896*x0), tmp38, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/zl/czlxdd6dczyxkeofthrety7clf6mc5hc7vsvsgndbxnyucxb53px.py
# Topologically Sorted Source Nodes: [view_2294, sum_95, squeeze_46, permute_2184, clone_119, view_2296, mul_1153], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
# Source node to ATen node mapping:
#   clone_119 => clone_119
#   mul_1153 => mul_1153
#   permute_2184 => permute_2184
#   squeeze_46 => squeeze_46
#   sum_95 => sum_95
#   view_2294 => view_2294
#   view_2296 => view_2296
# Graph fragment:
#   %getitem_190 : Tensor "bf16[1, 14, 1218, 64][1091328, 64, 896, 1]cuda:0" = PlaceHolder[target=getitem_190]
#   %view_2294 : Tensor "bf16[1, 2, 7, 1218, 64][1091328, 448, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_190, [1, 2, 7, 1218, 64]), kwargs = {})
#   %sum_95 : Tensor "bf16[1, 2, 1, 1218, 64][155904, 77952, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_2294, [2], True), kwargs = {})
#   %squeeze_46 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_95, 2), kwargs = {})
#   %permute_2184 : Tensor "bf16[1, 1218, 2, 64][155904, 64, 77952, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%squeeze_46, [0, 2, 1, 3]), kwargs = {})
#   %clone_119 : Tensor "bf16[1, 1218, 2, 64][155904, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2184,), kwargs = {memory_format: torch.contiguous_format})
#   %view_2296 : Tensor "bf16[1, 1218, 128][155904, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_119, [1, 1218, 128]), kwargs = {})
#   %mul_1153 : Tensor "bf16[1, 1218, 128][155904, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2296, 1.0), kwargs = {})
#   return %mul_1153
triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_13 = async_compile.triton('triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 2806272}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 155904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = xindex // 128
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (448*(x0 // 64) + 896*x1 + ((x0 % 64))), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (64 + 448*(x0 // 64) + 896*x1 + ((x0 % 64))), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (128 + 448*(x0 // 64) + 896*x1 + ((x0 % 64))), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (192 + 448*(x0 // 64) + 896*x1 + ((x0 % 64))), xmask).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (256 + 448*(x0 // 64) + 896*x1 + ((x0 % 64))), xmask).to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (320 + 448*(x0 // 64) + 896*x1 + ((x0 % 64))), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (384 + 448*(x0 // 64) + 896*x1 + ((x0 % 64))), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/gl/cglwpg6uq7m7tuodvfc74gkfncx5ummdblk7thimu3sbk2v4rvll.py
# Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2295, sum_96, squeeze_47, mul_1149, slice_214, slice_215, neg_118, add_784, mul_1150, add_785, permute_2192, clone_120, view_2300, mul_1154], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   add_784 => add_784
#   add_785 => add_785
#   clone_120 => clone_120
#   cos => cos
#   cos_1 => mul
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_8
#   freqs => permute
#   full_default_50 => full_default_50
#   matmul => unsqueeze_default
#   mul_1149 => mul_1149
#   mul_1150 => mul_1150
#   mul_1154 => mul_1154
#   neg_118 => neg_118
#   permute_2192 => permute_2192
#   sin => sin
#   sin_1 => mul_1
#   sin_2 => convert_element_type_3
#   sin_3 => unsqueeze_6
#   slice_214 => slice_214
#   slice_215 => slice_215
#   squeeze_47 => squeeze_47
#   sum_96 => sum_96
#   view_2295 => view_2295
#   view_2300 => view_2300
# Graph fragment:
#   %sum_96 : Tensor "bf16[1, 2, 1, 1218, 64][155904, 64, 155904, 128, 1]cuda:0" = PlaceHolder[target=sum_96]
#   %mm_default : Tensor "f32[32, 1218][1248, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %unsqueeze_default : Tensor "f32[1, 32, 1218][39936, 1248, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, 1218, 32][39936, 1, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 1218, 1, 32][39936, 1, 39936, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1218, 2, 32][39936, 1, 0, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, 1218, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1218, 2, 32][77952, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1218, 64]), kwargs = {})
#   %sin : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_8,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, 1218, 64][77952, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 1), kwargs = {})
#   %full_default_50 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.full.default](args = ([1, 2, 1218, 64], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %cos : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_8,), kwargs = {})
#   %mul : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 1218, 64][77952, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %view_2295 : Tensor "bf16[1, 2, 7, 1218, 64][1091328, 448, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%getitem_189, [1, 2, 7, 1218, 64]), kwargs = {})
#   %sum_96 : Tensor "bf16[1, 2, 1, 1218, 64][155904, 77952, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%view_2295, [2], True), kwargs = {})
#   %squeeze_47 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.squeeze.dim](args = (%sum_96, 2), kwargs = {})
#   %mul_1149 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_47, %unsqueeze_6), kwargs = {})
#   %slice_214 : Tensor "bf16[1, 2, 1218, 32][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_1149, 3, 0, 32), kwargs = {})
#   %slice_215 : Tensor "bf16[1, 2, 1218, 32][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_1149, 3, 32, 64), kwargs = {})
#   %neg_118 : Tensor "bf16[1, 2, 1218, 32][77952, 38976, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_214,), kwargs = {})
#   %slice_scatter_default_93 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_50, %neg_118, 3, 32, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_94 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_50, %slice_215, 3, 0, 32), kwargs = {})
#   %add_784 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_93, %slice_scatter_default_94), kwargs = {})
#   %mul_1150 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%squeeze_47, %unsqueeze_5), kwargs = {})
#   %add_785 : Tensor "bf16[1, 2, 1218, 64][155904, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_784, %mul_1150), kwargs = {})
#   %permute_2192 : Tensor "bf16[1, 1218, 2, 64][155904, 64, 77952, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_785, [0, 2, 1, 3]), kwargs = {})
#   %clone_120 : Tensor "bf16[1, 1218, 2, 64][155904, 128, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2192,), kwargs = {memory_format: torch.contiguous_format})
#   %view_2300 : Tensor "bf16[1, 1218, 128][155904, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_120, [1, 1218, 128]), kwargs = {})
#   %mul_1154 : Tensor "bf16[1, 1218, 128][155904, 128, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2300, 1.0), kwargs = {})
#   return %mul_1154
triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_14 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_14', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 155904, 'x': 1559040}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_14(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1218
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp27 = tl.load(in_ptr0 + (x1 + 128*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (y0 + 1248*((((x1 % 64)) % 32))), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (x1 % 64)
    tmp1 = tl.full([1, 1], 32, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-32) + x1 + 128*y0), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (y0 + 1248*((((x1 % 64)) % 32))), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl_math.sin(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp3 * tmp8
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp12, tmp13)
    tmp15 = tmp0 < tmp1
    tmp16 = tl.load(in_ptr0 + (32 + x1 + 128*y0), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr1 + (y0 + 1248*((((x1 % 64)) % 32))), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl_math.sin(tmp17)
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp16 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tl.where(tmp15, tmp24, tmp13)
    tmp26 = tmp14 + tmp25
    tmp29 = tl_math.cos(tmp28)
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tmp34 = tmp26 + tmp33
    tmp35 = tmp34 * tmp30
    tl.store(out_ptr0 + (x1 + 128*y0), tmp35, xmask & ymask)
''', device_str='cuda')


# kernel path: /root/autodl-tmp/HuangJieCode/Big-Yellow-J.github.io/code/Python/RL-TRL/~/autodl-tmp/tmp-sore/torchinductor_root/wg/cwghc6sdf6gdhgtjts4edbipyox6dud5khhwshdl4mdcnpajzvz7.py
# Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1151, slice_216, slice_217, neg_119, add_786, mul_1152, add_787, permute_2200, clone_121, view_2304, mul_1155], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   add_786 => add_786
#   add_787 => add_787
#   clone_121 => clone_121
#   cos => cos
#   cos_1 => mul
#   cos_2 => convert_element_type_2
#   cos_3 => unsqueeze_5
#   emb => clone, expand_4, unsqueeze_4, view_8
#   freqs => permute
#   full_default_52 => full_default_52
#   matmul => unsqueeze_default
#   mul_1151 => mul_1151
#   mul_1152 => mul_1152
#   mul_1155 => mul_1155
#   neg_119 => neg_119
#   permute_2200 => permute_2200
#   sin => sin
#   sin_1 => mul_1
#   sin_2 => convert_element_type_3
#   sin_3 => unsqueeze_6
#   slice_216 => slice_216
#   slice_217 => slice_217
#   view_2304 => view_2304
# Graph fragment:
#   %getitem_188 : Tensor "bf16[1, 14, 1218, 64][1091328, 64, 896, 1]cuda:0" = PlaceHolder[target=getitem_188]
#   %mm_default : Tensor "f32[32, 1218][1248, 1]cuda:0" = PlaceHolder[target=mm_default]
#   %unsqueeze_default : Tensor "f32[1, 32, 1218][39936, 1248, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%mm_default, 0), kwargs = {})
#   %permute : Tensor "f32[1, 1218, 32][39936, 1, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%unsqueeze_default, [0, 2, 1]), kwargs = {})
#   %unsqueeze_4 : Tensor "f32[1, 1218, 1, 32][39936, 1, 39936, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%permute, 2), kwargs = {})
#   %expand_4 : Tensor "f32[1, 1218, 2, 32][39936, 1, 0, 1248]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.expand.default](args = (%unsqueeze_4, [1, 1218, 2, 32]), kwargs = {})
#   %clone : Tensor "f32[1, 1218, 2, 32][77952, 64, 32, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
#   %view_8 : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [1, 1218, 64]), kwargs = {})
#   %sin : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%view_8,), kwargs = {})
#   %mul_1 : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 1.0), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1, torch.bfloat16), kwargs = {})
#   %unsqueeze_6 : Tensor "bf16[1, 1, 1218, 64][77952, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_3, 1), kwargs = {})
#   %cos : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%view_8,), kwargs = {})
#   %mul : Tensor "f32[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%cos, 1.0), kwargs = {})
#   %convert_element_type_2 : Tensor "bf16[1, 1218, 64][77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %unsqueeze_5 : Tensor "bf16[1, 1, 1218, 64][77952, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.unsqueeze.default](args = (%convert_element_type_2, 1), kwargs = {})
#   %full_default_52 : Tensor "bf16[1, 14, 1218, 64][1091328, 77952, 64, 1]cuda:0"[num_users=48] = call_function[target=torch.ops.aten.full.default](args = ([1, 14, 1218, 64], 0), kwargs = {dtype: torch.bfloat16, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul_1151 : Tensor "bf16[1, 14, 1218, 64][1091328, 64, 896, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_188, %unsqueeze_6), kwargs = {})
#   %slice_216 : Tensor "bf16[1, 14, 1218, 32][1091328, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_1151, 3, 0, 32), kwargs = {})
#   %slice_217 : Tensor "bf16[1, 14, 1218, 32][1091328, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mul_1151, 3, 32, 64), kwargs = {})
#   %neg_119 : Tensor "bf16[1, 14, 1218, 32][545664, 32, 448, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%slice_216,), kwargs = {})
#   %slice_scatter_default_95 : Tensor "bf16[1, 14, 1218, 64][1091328, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_52, %neg_119, 3, 32, 9223372036854775807), kwargs = {})
#   %slice_scatter_default_96 : Tensor "bf16[1, 14, 1218, 64][1091328, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice_scatter.default](args = (%full_default_52, %slice_217, 3, 0, 32), kwargs = {})
#   %add_786 : Tensor "bf16[1, 14, 1218, 64][1091328, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%slice_scatter_default_95, %slice_scatter_default_96), kwargs = {})
#   %mul_1152 : Tensor "bf16[1, 14, 1218, 64][1091328, 64, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_188, %unsqueeze_5), kwargs = {})
#   %add_787 : Tensor "bf16[1, 14, 1218, 64][1091328, 77952, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_786, %mul_1152), kwargs = {})
#   %permute_2200 : Tensor "bf16[1, 1218, 14, 64][1091328, 64, 77952, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_787, [0, 2, 1, 3]), kwargs = {})
#   %clone_121 : Tensor "bf16[1, 1218, 14, 64][1091328, 896, 64, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2200,), kwargs = {memory_format: torch.contiguous_format})
#   %view_2304 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_121, [1, 1218, 896]), kwargs = {})
#   %mul_1155 : Tensor "bf16[1, 1218, 896][1091328, 896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2304, 1.0), kwargs = {})
#   return %mul_1155
triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_15 = async_compile.triton('triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'x': 10913280}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_15(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1091328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 896)
    x1 = xindex // 896
    tmp27 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (x1 + 1248*((((x0 % 64)) % 32))), xmask, eviction_policy='evict_last')
    tmp0 = (x2 % 64)
    tmp1 = tl.full([1], 32, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-32) + x2), tmp2 & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (x1 + 1248*((((x0 % 64)) % 32))), tmp2 & xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl_math.sin(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp3 * tmp8
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp2, tmp10, tmp11)
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp12, tmp13)
    tmp15 = tmp0 < tmp1
    tmp16 = tl.load(in_ptr0 + (32 + x2), tmp15 & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr1 + (x1 + 1248*((((x0 % 64)) % 32))), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl_math.sin(tmp17)
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp16 * tmp21
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp15, tmp22, tmp23)
    tmp25 = tl.where(tmp15, tmp24, tmp13)
    tmp26 = tmp14 + tmp25
    tmp29 = tl_math.cos(tmp28)
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp27 * tmp32
    tmp34 = tmp26 + tmp33
    tmp35 = tmp34 * tmp30
    tl.store(out_ptr0 + (x2), tmp35, xmask)
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
        primals_18, primals_21, primals_22, primals_25, primals_28, primals_31, primals_32, primals_36, primals_40, primals_44, primals_47, primals_48, primals_51, primals_54, primals_57, primals_58, primals_62, primals_66, primals_70, primals_73, primals_74, primals_77, primals_80, primals_83, primals_84, primals_88, primals_92, primals_96, primals_99, primals_100, primals_103, primals_106, primals_109, primals_110, primals_114, primals_118, primals_122, primals_125, primals_126, primals_129, primals_132, primals_135, primals_136, primals_140, primals_144, primals_148, primals_151, primals_152, primals_155, primals_158, primals_161, primals_162, primals_166, primals_170, primals_174, primals_177, primals_178, primals_181, primals_184, primals_187, primals_188, primals_192, primals_196, primals_200, primals_203, primals_204, primals_207, primals_210, primals_213, primals_214, primals_218, primals_222, primals_226, primals_229, primals_230, primals_233, primals_236, primals_239, primals_240, primals_244, primals_248, primals_252, primals_255, primals_256, primals_259, primals_262, primals_265, primals_266, primals_270, primals_274, primals_278, primals_281, primals_282, primals_285, primals_288, primals_291, primals_292, primals_296, primals_300, primals_304, primals_307, primals_308, primals_311, primals_314, primals_317, primals_318, primals_322, primals_326, primals_330, primals_333, primals_334, primals_337, primals_340, primals_343, primals_344, primals_348, primals_352, primals_356, primals_359, primals_360, primals_363, primals_366, primals_369, primals_370, primals_374, primals_378, primals_382, primals_385, primals_386, primals_389, primals_392, primals_395, primals_396, primals_400, primals_404, primals_408, primals_411, primals_412, primals_415, primals_418, primals_421, primals_422, primals_426, primals_430, primals_434, primals_437, primals_438, primals_441, primals_444, primals_447, primals_448, primals_452, primals_456, primals_460, primals_463, primals_464, primals_467, primals_470, primals_473, primals_474, primals_478, primals_482, primals_486, primals_489, primals_490, primals_493, primals_496, primals_499, primals_500, primals_504, primals_508, primals_512, primals_515, primals_516, primals_519, primals_522, primals_525, primals_526, primals_530, primals_534, primals_538, primals_541, primals_542, primals_545, primals_548, primals_551, primals_552, primals_556, primals_560, primals_564, primals_567, primals_568, primals_571, primals_574, primals_577, primals_578, primals_582, primals_586, primals_590, primals_593, primals_594, primals_597, primals_600, primals_603, primals_604, primals_608, primals_612, primals_616, primals_619, primals_620, primals_623, primals_626, primals_629, primals_630, mm_default, view_11, mm, mm_2, mm_4, add_5, view_30, view_31, constant_pad_nd, getitem, getitem_1, getitem_2, getitem_3, view_35, mm_7, add_8, rsqrt_1, view_41, mm_10, add_10, mm_13, add_11, view_53, mm_16, add_13, rsqrt_2, view_59, mm_18, mm_20, mm_22, add_18, view_78, view_79, getitem_4, getitem_5, getitem_6, getitem_7, view_83, mm_25, add_21, rsqrt_3, view_89, mm_28, add_23, mm_31, add_24, view_101, mm_34, add_26, rsqrt_4, view_107, mm_36, mm_38, mm_40, add_31, view_126, view_127, getitem_8, getitem_9, getitem_10, getitem_11, view_131, mm_43, add_34, rsqrt_5, view_137, mm_46, add_36, mm_49, add_37, view_149, mm_52, add_39, rsqrt_6, view_155, mm_54, mm_56, mm_58, add_44, view_174, view_175, getitem_12, getitem_13, getitem_14, getitem_15, view_179, mm_61, add_47, rsqrt_7, view_185, mm_64, add_49, mm_67, add_50, view_197, mm_70, add_52, rsqrt_8, view_203, mm_72, mm_74, mm_76, add_57, view_222, view_223, getitem_16, getitem_17, getitem_18, getitem_19, view_227, mm_79, add_60, rsqrt_9, view_233, mm_82, add_62, mm_85, add_63, view_245, mm_88, add_65, rsqrt_10, view_251, mm_90, mm_92, mm_94, add_70, view_270, view_271, getitem_20, getitem_21, getitem_22, getitem_23, view_275, mm_97, add_73, rsqrt_11, view_281, mm_100, add_75, mm_103, add_76, view_293, mm_106, add_78, rsqrt_12, view_299, mm_108, mm_110, mm_112, add_83, view_318, view_319, getitem_24, getitem_25, getitem_26, getitem_27, view_323, mm_115, add_86, rsqrt_13, view_329, mm_118, add_88, mm_121, add_89, view_341, mm_124, add_91, rsqrt_14, view_347, mm_126, mm_128, mm_130, add_96, view_366, view_367, getitem_28, getitem_29, getitem_30, getitem_31, view_371, mm_133, add_99, rsqrt_15, view_377, mm_136, add_101, mm_139, add_102, view_389, mm_142, add_104, rsqrt_16, view_395, mm_144, mm_146, mm_148, add_109, view_414, view_415, getitem_32, getitem_33, getitem_34, getitem_35, view_419, mm_151, add_112, rsqrt_17, view_425, mm_154, add_114, mm_157, add_115, view_437, mm_160, add_117, rsqrt_18, view_443, mm_162, mm_164, mm_166, add_122, view_462, view_463, getitem_36, getitem_37, getitem_38, getitem_39, view_467, mm_169, add_125, rsqrt_19, view_473, mm_172, add_127, mm_175, add_128, view_485, mm_178, add_130, rsqrt_20, view_491, mm_180, mm_182, mm_184, add_135, view_510, view_511, getitem_40, getitem_41, getitem_42, getitem_43, view_515, mm_187, add_138, rsqrt_21, view_521, mm_190, add_140, mm_193, add_141, view_533, mm_196, add_143, rsqrt_22, view_539, mm_198, mm_200, mm_202, add_148, view_558, view_559, getitem_44, getitem_45, getitem_46, getitem_47, view_563, mm_205, add_151, rsqrt_23, view_569, mm_208, add_153, mm_211, add_154, view_581, mm_214, add_156, rsqrt_24, view_587, mm_216, mm_218, mm_220, add_161, view_606, view_607, getitem_48, getitem_49, getitem_50, getitem_51, view_611, mm_223, add_164, rsqrt_25, view_617, mm_226, add_166, mm_229, add_167, view_629, mm_232, add_169, rsqrt_26, view_635, mm_234, mm_236, mm_238, add_174, view_654, view_655, getitem_52, getitem_53, getitem_54, getitem_55, view_659, mm_241, add_177, rsqrt_27, view_665, mm_244, add_179, mm_247, add_180, view_677, mm_250, add_182, rsqrt_28, view_683, mm_252, mm_254, mm_256, add_187, view_702, view_703, getitem_56, getitem_57, getitem_58, getitem_59, view_707, mm_259, add_190, rsqrt_29, view_713, mm_262, add_192, mm_265, add_193, view_725, mm_268, add_195, rsqrt_30, view_731, mm_270, mm_272, mm_274, add_200, view_750, view_751, getitem_60, getitem_61, getitem_62, getitem_63, view_755, mm_277, add_203, rsqrt_31, view_761, mm_280, add_205, mm_283, add_206, view_773, mm_286, add_208, rsqrt_32, view_779, mm_288, mm_290, mm_292, add_213, view_798, view_799, getitem_64, getitem_65, getitem_66, getitem_67, view_803, mm_295, add_216, rsqrt_33, view_809, mm_298, add_218, mm_301, add_219, view_821, mm_304, add_221, rsqrt_34, view_827, mm_306, mm_308, mm_310, add_226, view_846, view_847, getitem_68, getitem_69, getitem_70, getitem_71, view_851, mm_313, add_229, rsqrt_35, view_857, mm_316, add_231, mm_319, add_232, view_869, mm_322, add_234, rsqrt_36, view_875, mm_324, mm_326, mm_328, add_239, view_894, view_895, getitem_72, getitem_73, getitem_74, getitem_75, view_899, mm_331, add_242, rsqrt_37, view_905, mm_334, add_244, mm_337, add_245, view_917, mm_340, add_247, rsqrt_38, view_923, mm_342, mm_344, mm_346, add_252, view_942, view_943, getitem_76, getitem_77, getitem_78, getitem_79, view_947, mm_349, add_255, rsqrt_39, view_953, mm_352, add_257, mm_355, add_258, view_965, mm_358, add_260, rsqrt_40, view_971, mm_360, mm_362, mm_364, add_265, view_990, view_991, getitem_80, getitem_81, getitem_82, getitem_83, view_995, mm_367, add_268, rsqrt_41, view_1001, mm_370, add_270, mm_373, add_271, view_1013, mm_376, add_273, rsqrt_42, view_1019, mm_378, mm_380, mm_382, add_278, view_1038, view_1039, getitem_84, getitem_85, getitem_86, getitem_87, view_1043, mm_385, add_281, rsqrt_43, view_1049, mm_388, add_283, mm_391, add_284, view_1061, mm_394, add_286, rsqrt_44, view_1067, mm_396, mm_398, mm_400, add_291, view_1086, view_1087, getitem_88, getitem_89, getitem_90, getitem_91, view_1091, mm_403, add_294, rsqrt_45, view_1097, mm_406, add_296, mm_409, add_297, view_1109, mm_412, add_299, rsqrt_46, view_1115, mm_414, mm_416, mm_418, add_304, view_1134, view_1135, getitem_92, getitem_93, getitem_94, getitem_95, view_1139, mm_421, add_307, rsqrt_47, view_1145, mm_424, add_309, mm_427, add_310, view_1157, mm_430, add_312, rsqrt_48, view_1161, permute_608, permute_612, permute_617, permute_621, permute_626, permute_630, permute_635, permute_639, permute_646, permute_650, permute_656, permute_660, permute_666, permute_670, permute_675, permute_679, permute_684, permute_688, permute_693, permute_697, permute_702, permute_706, permute_713, permute_717, permute_723, permute_727, permute_733, permute_737, permute_742, permute_746, permute_751, permute_755, permute_760, permute_764, permute_769, permute_773, permute_780, permute_784, permute_790, permute_794, permute_800, permute_804, permute_809, permute_813, permute_818, permute_822, permute_827, permute_831, permute_836, permute_840, permute_847, permute_851, permute_857, permute_861, permute_867, permute_871, permute_876, permute_880, permute_885, permute_889, permute_894, permute_898, permute_903, permute_907, permute_914, permute_918, permute_924, permute_928, permute_934, permute_938, permute_943, permute_947, permute_952, permute_956, permute_961, permute_965, permute_970, permute_974, permute_981, permute_985, permute_991, permute_995, permute_1001, permute_1005, permute_1010, permute_1014, permute_1019, permute_1023, permute_1028, permute_1032, permute_1037, permute_1041, permute_1048, permute_1052, permute_1058, permute_1062, permute_1068, permute_1072, permute_1077, permute_1081, permute_1086, permute_1090, permute_1095, permute_1099, permute_1104, permute_1108, permute_1115, permute_1119, permute_1125, permute_1129, permute_1135, permute_1139, permute_1144, permute_1148, permute_1153, permute_1157, permute_1162, permute_1166, permute_1171, permute_1175, permute_1182, permute_1186, permute_1192, permute_1196, permute_1202, permute_1206, permute_1211, permute_1215, permute_1220, permute_1224, permute_1229, permute_1233, permute_1238, permute_1242, permute_1249, permute_1253, permute_1259, permute_1263, permute_1269, permute_1273, permute_1278, permute_1282, permute_1287, permute_1291, permute_1296, permute_1300, permute_1305, permute_1309, permute_1316, permute_1320, permute_1326, permute_1330, permute_1336, permute_1340, permute_1345, permute_1349, permute_1354, permute_1358, permute_1363, permute_1367, permute_1372, permute_1376, permute_1383, permute_1387, permute_1393, permute_1397, permute_1403, permute_1407, permute_1412, permute_1416, permute_1421, permute_1425, permute_1430, permute_1434, permute_1439, permute_1443, permute_1450, permute_1454, permute_1460, permute_1464, permute_1470, permute_1474, permute_1479, permute_1483, permute_1488, permute_1492, permute_1497, permute_1501, permute_1506, permute_1510, permute_1517, permute_1521, permute_1527, permute_1531, permute_1537, permute_1541, permute_1546, permute_1550, permute_1555, permute_1559, permute_1564, permute_1568, permute_1573, permute_1577, permute_1584, permute_1588, permute_1594, permute_1598, permute_1604, permute_1608, permute_1613, permute_1617, permute_1622, permute_1626, permute_1631, permute_1635, permute_1640, permute_1644, permute_1651, permute_1655, permute_1661, permute_1665, permute_1671, permute_1675, permute_1680, permute_1684, permute_1689, permute_1693, permute_1698, permute_1702, permute_1707, permute_1711, permute_1718, permute_1722, permute_1728, permute_1732, permute_1738, permute_1742, permute_1747, permute_1751, permute_1756, permute_1760, permute_1765, permute_1769, permute_1774, permute_1778, permute_1785, permute_1789, permute_1795, permute_1799, permute_1805, permute_1809, permute_1814, permute_1818, permute_1823, permute_1827, permute_1832, permute_1836, permute_1841, permute_1845, permute_1852, permute_1856, permute_1862, permute_1866, permute_1872, permute_1876, permute_1881, permute_1885, permute_1890, permute_1894, permute_1899, permute_1903, permute_1908, permute_1912, permute_1919, permute_1923, permute_1929, permute_1933, permute_1939, permute_1943, permute_1948, permute_1952, permute_1957, permute_1961, permute_1966, permute_1970, permute_1975, permute_1979, permute_1986, permute_1990, permute_1996, permute_2000, permute_2006, permute_2010, permute_2015, permute_2019, permute_2024, permute_2028, permute_2033, permute_2037, permute_2042, permute_2046, permute_2053, permute_2057, permute_2063, permute_2067, permute_2073, permute_2077, permute_2082, permute_2086, permute_2091, permute_2095, permute_2100, permute_2104, permute_2109, permute_2113, permute_2120, permute_2124, permute_2130, permute_2134, permute_2140, permute_2144, permute_2149, permute_2153, permute_2158, permute_2162, permute_2167, permute_2171, permute_2176, permute_2180, permute_2187, permute_2195, permute_2203, tangents_1 = args
        args.clear()
        assert_size_stride(primals_18, (896, 896), (896, 1))
        assert_size_stride(primals_21, (896, ), (1, ))
        assert_size_stride(primals_22, (4864, 896), (896, 1))
        assert_size_stride(primals_25, (4864, 896), (896, 1))
        assert_size_stride(primals_28, (896, 4864), (4864, 1))
        assert_size_stride(primals_31, (896, ), (1, ))
        assert_size_stride(primals_32, (896, 896), (896, 1))
        assert_size_stride(primals_36, (128, 896), (896, 1))
        assert_size_stride(primals_40, (128, 896), (896, 1))
        assert_size_stride(primals_44, (896, 896), (896, 1))
        assert_size_stride(primals_47, (896, ), (1, ))
        assert_size_stride(primals_48, (4864, 896), (896, 1))
        assert_size_stride(primals_51, (4864, 896), (896, 1))
        assert_size_stride(primals_54, (896, 4864), (4864, 1))
        assert_size_stride(primals_57, (896, ), (1, ))
        assert_size_stride(primals_58, (896, 896), (896, 1))
        assert_size_stride(primals_62, (128, 896), (896, 1))
        assert_size_stride(primals_66, (128, 896), (896, 1))
        assert_size_stride(primals_70, (896, 896), (896, 1))
        assert_size_stride(primals_73, (896, ), (1, ))
        assert_size_stride(primals_74, (4864, 896), (896, 1))
        assert_size_stride(primals_77, (4864, 896), (896, 1))
        assert_size_stride(primals_80, (896, 4864), (4864, 1))
        assert_size_stride(primals_83, (896, ), (1, ))
        assert_size_stride(primals_84, (896, 896), (896, 1))
        assert_size_stride(primals_88, (128, 896), (896, 1))
        assert_size_stride(primals_92, (128, 896), (896, 1))
        assert_size_stride(primals_96, (896, 896), (896, 1))
        assert_size_stride(primals_99, (896, ), (1, ))
        assert_size_stride(primals_100, (4864, 896), (896, 1))
        assert_size_stride(primals_103, (4864, 896), (896, 1))
        assert_size_stride(primals_106, (896, 4864), (4864, 1))
        assert_size_stride(primals_109, (896, ), (1, ))
        assert_size_stride(primals_110, (896, 896), (896, 1))
        assert_size_stride(primals_114, (128, 896), (896, 1))
        assert_size_stride(primals_118, (128, 896), (896, 1))
        assert_size_stride(primals_122, (896, 896), (896, 1))
        assert_size_stride(primals_125, (896, ), (1, ))
        assert_size_stride(primals_126, (4864, 896), (896, 1))
        assert_size_stride(primals_129, (4864, 896), (896, 1))
        assert_size_stride(primals_132, (896, 4864), (4864, 1))
        assert_size_stride(primals_135, (896, ), (1, ))
        assert_size_stride(primals_136, (896, 896), (896, 1))
        assert_size_stride(primals_140, (128, 896), (896, 1))
        assert_size_stride(primals_144, (128, 896), (896, 1))
        assert_size_stride(primals_148, (896, 896), (896, 1))
        assert_size_stride(primals_151, (896, ), (1, ))
        assert_size_stride(primals_152, (4864, 896), (896, 1))
        assert_size_stride(primals_155, (4864, 896), (896, 1))
        assert_size_stride(primals_158, (896, 4864), (4864, 1))
        assert_size_stride(primals_161, (896, ), (1, ))
        assert_size_stride(primals_162, (896, 896), (896, 1))
        assert_size_stride(primals_166, (128, 896), (896, 1))
        assert_size_stride(primals_170, (128, 896), (896, 1))
        assert_size_stride(primals_174, (896, 896), (896, 1))
        assert_size_stride(primals_177, (896, ), (1, ))
        assert_size_stride(primals_178, (4864, 896), (896, 1))
        assert_size_stride(primals_181, (4864, 896), (896, 1))
        assert_size_stride(primals_184, (896, 4864), (4864, 1))
        assert_size_stride(primals_187, (896, ), (1, ))
        assert_size_stride(primals_188, (896, 896), (896, 1))
        assert_size_stride(primals_192, (128, 896), (896, 1))
        assert_size_stride(primals_196, (128, 896), (896, 1))
        assert_size_stride(primals_200, (896, 896), (896, 1))
        assert_size_stride(primals_203, (896, ), (1, ))
        assert_size_stride(primals_204, (4864, 896), (896, 1))
        assert_size_stride(primals_207, (4864, 896), (896, 1))
        assert_size_stride(primals_210, (896, 4864), (4864, 1))
        assert_size_stride(primals_213, (896, ), (1, ))
        assert_size_stride(primals_214, (896, 896), (896, 1))
        assert_size_stride(primals_218, (128, 896), (896, 1))
        assert_size_stride(primals_222, (128, 896), (896, 1))
        assert_size_stride(primals_226, (896, 896), (896, 1))
        assert_size_stride(primals_229, (896, ), (1, ))
        assert_size_stride(primals_230, (4864, 896), (896, 1))
        assert_size_stride(primals_233, (4864, 896), (896, 1))
        assert_size_stride(primals_236, (896, 4864), (4864, 1))
        assert_size_stride(primals_239, (896, ), (1, ))
        assert_size_stride(primals_240, (896, 896), (896, 1))
        assert_size_stride(primals_244, (128, 896), (896, 1))
        assert_size_stride(primals_248, (128, 896), (896, 1))
        assert_size_stride(primals_252, (896, 896), (896, 1))
        assert_size_stride(primals_255, (896, ), (1, ))
        assert_size_stride(primals_256, (4864, 896), (896, 1))
        assert_size_stride(primals_259, (4864, 896), (896, 1))
        assert_size_stride(primals_262, (896, 4864), (4864, 1))
        assert_size_stride(primals_265, (896, ), (1, ))
        assert_size_stride(primals_266, (896, 896), (896, 1))
        assert_size_stride(primals_270, (128, 896), (896, 1))
        assert_size_stride(primals_274, (128, 896), (896, 1))
        assert_size_stride(primals_278, (896, 896), (896, 1))
        assert_size_stride(primals_281, (896, ), (1, ))
        assert_size_stride(primals_282, (4864, 896), (896, 1))
        assert_size_stride(primals_285, (4864, 896), (896, 1))
        assert_size_stride(primals_288, (896, 4864), (4864, 1))
        assert_size_stride(primals_291, (896, ), (1, ))
        assert_size_stride(primals_292, (896, 896), (896, 1))
        assert_size_stride(primals_296, (128, 896), (896, 1))
        assert_size_stride(primals_300, (128, 896), (896, 1))
        assert_size_stride(primals_304, (896, 896), (896, 1))
        assert_size_stride(primals_307, (896, ), (1, ))
        assert_size_stride(primals_308, (4864, 896), (896, 1))
        assert_size_stride(primals_311, (4864, 896), (896, 1))
        assert_size_stride(primals_314, (896, 4864), (4864, 1))
        assert_size_stride(primals_317, (896, ), (1, ))
        assert_size_stride(primals_318, (896, 896), (896, 1))
        assert_size_stride(primals_322, (128, 896), (896, 1))
        assert_size_stride(primals_326, (128, 896), (896, 1))
        assert_size_stride(primals_330, (896, 896), (896, 1))
        assert_size_stride(primals_333, (896, ), (1, ))
        assert_size_stride(primals_334, (4864, 896), (896, 1))
        assert_size_stride(primals_337, (4864, 896), (896, 1))
        assert_size_stride(primals_340, (896, 4864), (4864, 1))
        assert_size_stride(primals_343, (896, ), (1, ))
        assert_size_stride(primals_344, (896, 896), (896, 1))
        assert_size_stride(primals_348, (128, 896), (896, 1))
        assert_size_stride(primals_352, (128, 896), (896, 1))
        assert_size_stride(primals_356, (896, 896), (896, 1))
        assert_size_stride(primals_359, (896, ), (1, ))
        assert_size_stride(primals_360, (4864, 896), (896, 1))
        assert_size_stride(primals_363, (4864, 896), (896, 1))
        assert_size_stride(primals_366, (896, 4864), (4864, 1))
        assert_size_stride(primals_369, (896, ), (1, ))
        assert_size_stride(primals_370, (896, 896), (896, 1))
        assert_size_stride(primals_374, (128, 896), (896, 1))
        assert_size_stride(primals_378, (128, 896), (896, 1))
        assert_size_stride(primals_382, (896, 896), (896, 1))
        assert_size_stride(primals_385, (896, ), (1, ))
        assert_size_stride(primals_386, (4864, 896), (896, 1))
        assert_size_stride(primals_389, (4864, 896), (896, 1))
        assert_size_stride(primals_392, (896, 4864), (4864, 1))
        assert_size_stride(primals_395, (896, ), (1, ))
        assert_size_stride(primals_396, (896, 896), (896, 1))
        assert_size_stride(primals_400, (128, 896), (896, 1))
        assert_size_stride(primals_404, (128, 896), (896, 1))
        assert_size_stride(primals_408, (896, 896), (896, 1))
        assert_size_stride(primals_411, (896, ), (1, ))
        assert_size_stride(primals_412, (4864, 896), (896, 1))
        assert_size_stride(primals_415, (4864, 896), (896, 1))
        assert_size_stride(primals_418, (896, 4864), (4864, 1))
        assert_size_stride(primals_421, (896, ), (1, ))
        assert_size_stride(primals_422, (896, 896), (896, 1))
        assert_size_stride(primals_426, (128, 896), (896, 1))
        assert_size_stride(primals_430, (128, 896), (896, 1))
        assert_size_stride(primals_434, (896, 896), (896, 1))
        assert_size_stride(primals_437, (896, ), (1, ))
        assert_size_stride(primals_438, (4864, 896), (896, 1))
        assert_size_stride(primals_441, (4864, 896), (896, 1))
        assert_size_stride(primals_444, (896, 4864), (4864, 1))
        assert_size_stride(primals_447, (896, ), (1, ))
        assert_size_stride(primals_448, (896, 896), (896, 1))
        assert_size_stride(primals_452, (128, 896), (896, 1))
        assert_size_stride(primals_456, (128, 896), (896, 1))
        assert_size_stride(primals_460, (896, 896), (896, 1))
        assert_size_stride(primals_463, (896, ), (1, ))
        assert_size_stride(primals_464, (4864, 896), (896, 1))
        assert_size_stride(primals_467, (4864, 896), (896, 1))
        assert_size_stride(primals_470, (896, 4864), (4864, 1))
        assert_size_stride(primals_473, (896, ), (1, ))
        assert_size_stride(primals_474, (896, 896), (896, 1))
        assert_size_stride(primals_478, (128, 896), (896, 1))
        assert_size_stride(primals_482, (128, 896), (896, 1))
        assert_size_stride(primals_486, (896, 896), (896, 1))
        assert_size_stride(primals_489, (896, ), (1, ))
        assert_size_stride(primals_490, (4864, 896), (896, 1))
        assert_size_stride(primals_493, (4864, 896), (896, 1))
        assert_size_stride(primals_496, (896, 4864), (4864, 1))
        assert_size_stride(primals_499, (896, ), (1, ))
        assert_size_stride(primals_500, (896, 896), (896, 1))
        assert_size_stride(primals_504, (128, 896), (896, 1))
        assert_size_stride(primals_508, (128, 896), (896, 1))
        assert_size_stride(primals_512, (896, 896), (896, 1))
        assert_size_stride(primals_515, (896, ), (1, ))
        assert_size_stride(primals_516, (4864, 896), (896, 1))
        assert_size_stride(primals_519, (4864, 896), (896, 1))
        assert_size_stride(primals_522, (896, 4864), (4864, 1))
        assert_size_stride(primals_525, (896, ), (1, ))
        assert_size_stride(primals_526, (896, 896), (896, 1))
        assert_size_stride(primals_530, (128, 896), (896, 1))
        assert_size_stride(primals_534, (128, 896), (896, 1))
        assert_size_stride(primals_538, (896, 896), (896, 1))
        assert_size_stride(primals_541, (896, ), (1, ))
        assert_size_stride(primals_542, (4864, 896), (896, 1))
        assert_size_stride(primals_545, (4864, 896), (896, 1))
        assert_size_stride(primals_548, (896, 4864), (4864, 1))
        assert_size_stride(primals_551, (896, ), (1, ))
        assert_size_stride(primals_552, (896, 896), (896, 1))
        assert_size_stride(primals_556, (128, 896), (896, 1))
        assert_size_stride(primals_560, (128, 896), (896, 1))
        assert_size_stride(primals_564, (896, 896), (896, 1))
        assert_size_stride(primals_567, (896, ), (1, ))
        assert_size_stride(primals_568, (4864, 896), (896, 1))
        assert_size_stride(primals_571, (4864, 896), (896, 1))
        assert_size_stride(primals_574, (896, 4864), (4864, 1))
        assert_size_stride(primals_577, (896, ), (1, ))
        assert_size_stride(primals_578, (896, 896), (896, 1))
        assert_size_stride(primals_582, (128, 896), (896, 1))
        assert_size_stride(primals_586, (128, 896), (896, 1))
        assert_size_stride(primals_590, (896, 896), (896, 1))
        assert_size_stride(primals_593, (896, ), (1, ))
        assert_size_stride(primals_594, (4864, 896), (896, 1))
        assert_size_stride(primals_597, (4864, 896), (896, 1))
        assert_size_stride(primals_600, (896, 4864), (4864, 1))
        assert_size_stride(primals_603, (896, ), (1, ))
        assert_size_stride(primals_604, (896, 896), (896, 1))
        assert_size_stride(primals_608, (128, 896), (896, 1))
        assert_size_stride(primals_612, (128, 896), (896, 1))
        assert_size_stride(primals_616, (896, 896), (896, 1))
        assert_size_stride(primals_619, (896, ), (1, ))
        assert_size_stride(primals_620, (4864, 896), (896, 1))
        assert_size_stride(primals_623, (4864, 896), (896, 1))
        assert_size_stride(primals_626, (896, 4864), (4864, 1))
        assert_size_stride(primals_629, (896, ), (1, ))
        assert_size_stride(primals_630, (151936, 896), (896, 1))
        assert_size_stride(mm_default, (32, 1218), (1248, 1))
        assert_size_stride(view_11, (1218, 896), (896, 1))
        assert_size_stride(mm, (1218, 32), (32, 1))
        assert_size_stride(mm_2, (1218, 32), (32, 1))
        assert_size_stride(mm_4, (1218, 32), (32, 1))
        assert_size_stride(add_5, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_30, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_31, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(constant_pad_nd, (1, 1, 1218, 1224), (1559040, 1559040, 1280, 1))
        assert_size_stride(getitem, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_1, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_2, (), ())
        assert_size_stride(getitem_3, (), ())
        assert_size_stride(view_35, (1218, 896), (896, 1))
        assert_size_stride(mm_7, (1218, 32), (32, 1))
        assert_size_stride(add_8, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_1, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_41, (1218, 896), (896, 1))
        assert_size_stride(mm_10, (1218, 32), (32, 1))
        assert_size_stride(add_10, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_13, (1218, 32), (32, 1))
        assert_size_stride(add_11, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_53, (1218, 4864), (4864, 1))
        assert_size_stride(mm_16, (1218, 32), (32, 1))
        assert_size_stride(add_13, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_2, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_59, (1218, 896), (896, 1))
        assert_size_stride(mm_18, (1218, 32), (32, 1))
        assert_size_stride(mm_20, (1218, 32), (32, 1))
        assert_size_stride(mm_22, (1218, 32), (32, 1))
        assert_size_stride(add_18, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_78, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_79, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_4, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_5, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_6, (), ())
        assert_size_stride(getitem_7, (), ())
        assert_size_stride(view_83, (1218, 896), (896, 1))
        assert_size_stride(mm_25, (1218, 32), (32, 1))
        assert_size_stride(add_21, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_3, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_89, (1218, 896), (896, 1))
        assert_size_stride(mm_28, (1218, 32), (32, 1))
        assert_size_stride(add_23, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_31, (1218, 32), (32, 1))
        assert_size_stride(add_24, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_101, (1218, 4864), (4864, 1))
        assert_size_stride(mm_34, (1218, 32), (32, 1))
        assert_size_stride(add_26, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_4, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_107, (1218, 896), (896, 1))
        assert_size_stride(mm_36, (1218, 32), (32, 1))
        assert_size_stride(mm_38, (1218, 32), (32, 1))
        assert_size_stride(mm_40, (1218, 32), (32, 1))
        assert_size_stride(add_31, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_126, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_127, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_8, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_9, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_10, (), ())
        assert_size_stride(getitem_11, (), ())
        assert_size_stride(view_131, (1218, 896), (896, 1))
        assert_size_stride(mm_43, (1218, 32), (32, 1))
        assert_size_stride(add_34, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_5, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_137, (1218, 896), (896, 1))
        assert_size_stride(mm_46, (1218, 32), (32, 1))
        assert_size_stride(add_36, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_49, (1218, 32), (32, 1))
        assert_size_stride(add_37, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_149, (1218, 4864), (4864, 1))
        assert_size_stride(mm_52, (1218, 32), (32, 1))
        assert_size_stride(add_39, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_6, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_155, (1218, 896), (896, 1))
        assert_size_stride(mm_54, (1218, 32), (32, 1))
        assert_size_stride(mm_56, (1218, 32), (32, 1))
        assert_size_stride(mm_58, (1218, 32), (32, 1))
        assert_size_stride(add_44, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_174, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_175, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_12, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_13, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_14, (), ())
        assert_size_stride(getitem_15, (), ())
        assert_size_stride(view_179, (1218, 896), (896, 1))
        assert_size_stride(mm_61, (1218, 32), (32, 1))
        assert_size_stride(add_47, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_7, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_185, (1218, 896), (896, 1))
        assert_size_stride(mm_64, (1218, 32), (32, 1))
        assert_size_stride(add_49, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_67, (1218, 32), (32, 1))
        assert_size_stride(add_50, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_197, (1218, 4864), (4864, 1))
        assert_size_stride(mm_70, (1218, 32), (32, 1))
        assert_size_stride(add_52, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_8, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_203, (1218, 896), (896, 1))
        assert_size_stride(mm_72, (1218, 32), (32, 1))
        assert_size_stride(mm_74, (1218, 32), (32, 1))
        assert_size_stride(mm_76, (1218, 32), (32, 1))
        assert_size_stride(add_57, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_222, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_223, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_16, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_17, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_18, (), ())
        assert_size_stride(getitem_19, (), ())
        assert_size_stride(view_227, (1218, 896), (896, 1))
        assert_size_stride(mm_79, (1218, 32), (32, 1))
        assert_size_stride(add_60, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_9, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_233, (1218, 896), (896, 1))
        assert_size_stride(mm_82, (1218, 32), (32, 1))
        assert_size_stride(add_62, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_85, (1218, 32), (32, 1))
        assert_size_stride(add_63, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_245, (1218, 4864), (4864, 1))
        assert_size_stride(mm_88, (1218, 32), (32, 1))
        assert_size_stride(add_65, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_10, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_251, (1218, 896), (896, 1))
        assert_size_stride(mm_90, (1218, 32), (32, 1))
        assert_size_stride(mm_92, (1218, 32), (32, 1))
        assert_size_stride(mm_94, (1218, 32), (32, 1))
        assert_size_stride(add_70, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_270, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_271, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_20, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_21, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_22, (), ())
        assert_size_stride(getitem_23, (), ())
        assert_size_stride(view_275, (1218, 896), (896, 1))
        assert_size_stride(mm_97, (1218, 32), (32, 1))
        assert_size_stride(add_73, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_11, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_281, (1218, 896), (896, 1))
        assert_size_stride(mm_100, (1218, 32), (32, 1))
        assert_size_stride(add_75, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_103, (1218, 32), (32, 1))
        assert_size_stride(add_76, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_293, (1218, 4864), (4864, 1))
        assert_size_stride(mm_106, (1218, 32), (32, 1))
        assert_size_stride(add_78, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_12, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_299, (1218, 896), (896, 1))
        assert_size_stride(mm_108, (1218, 32), (32, 1))
        assert_size_stride(mm_110, (1218, 32), (32, 1))
        assert_size_stride(mm_112, (1218, 32), (32, 1))
        assert_size_stride(add_83, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_318, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_319, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_24, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_25, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_26, (), ())
        assert_size_stride(getitem_27, (), ())
        assert_size_stride(view_323, (1218, 896), (896, 1))
        assert_size_stride(mm_115, (1218, 32), (32, 1))
        assert_size_stride(add_86, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_13, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_329, (1218, 896), (896, 1))
        assert_size_stride(mm_118, (1218, 32), (32, 1))
        assert_size_stride(add_88, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_121, (1218, 32), (32, 1))
        assert_size_stride(add_89, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_341, (1218, 4864), (4864, 1))
        assert_size_stride(mm_124, (1218, 32), (32, 1))
        assert_size_stride(add_91, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_14, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_347, (1218, 896), (896, 1))
        assert_size_stride(mm_126, (1218, 32), (32, 1))
        assert_size_stride(mm_128, (1218, 32), (32, 1))
        assert_size_stride(mm_130, (1218, 32), (32, 1))
        assert_size_stride(add_96, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_366, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_367, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_28, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_29, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_30, (), ())
        assert_size_stride(getitem_31, (), ())
        assert_size_stride(view_371, (1218, 896), (896, 1))
        assert_size_stride(mm_133, (1218, 32), (32, 1))
        assert_size_stride(add_99, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_15, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_377, (1218, 896), (896, 1))
        assert_size_stride(mm_136, (1218, 32), (32, 1))
        assert_size_stride(add_101, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_139, (1218, 32), (32, 1))
        assert_size_stride(add_102, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_389, (1218, 4864), (4864, 1))
        assert_size_stride(mm_142, (1218, 32), (32, 1))
        assert_size_stride(add_104, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_16, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_395, (1218, 896), (896, 1))
        assert_size_stride(mm_144, (1218, 32), (32, 1))
        assert_size_stride(mm_146, (1218, 32), (32, 1))
        assert_size_stride(mm_148, (1218, 32), (32, 1))
        assert_size_stride(add_109, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_414, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_415, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_32, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_33, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_34, (), ())
        assert_size_stride(getitem_35, (), ())
        assert_size_stride(view_419, (1218, 896), (896, 1))
        assert_size_stride(mm_151, (1218, 32), (32, 1))
        assert_size_stride(add_112, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_17, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_425, (1218, 896), (896, 1))
        assert_size_stride(mm_154, (1218, 32), (32, 1))
        assert_size_stride(add_114, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_157, (1218, 32), (32, 1))
        assert_size_stride(add_115, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_437, (1218, 4864), (4864, 1))
        assert_size_stride(mm_160, (1218, 32), (32, 1))
        assert_size_stride(add_117, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_18, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_443, (1218, 896), (896, 1))
        assert_size_stride(mm_162, (1218, 32), (32, 1))
        assert_size_stride(mm_164, (1218, 32), (32, 1))
        assert_size_stride(mm_166, (1218, 32), (32, 1))
        assert_size_stride(add_122, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_462, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_463, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_36, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_37, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_38, (), ())
        assert_size_stride(getitem_39, (), ())
        assert_size_stride(view_467, (1218, 896), (896, 1))
        assert_size_stride(mm_169, (1218, 32), (32, 1))
        assert_size_stride(add_125, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_19, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_473, (1218, 896), (896, 1))
        assert_size_stride(mm_172, (1218, 32), (32, 1))
        assert_size_stride(add_127, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_175, (1218, 32), (32, 1))
        assert_size_stride(add_128, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_485, (1218, 4864), (4864, 1))
        assert_size_stride(mm_178, (1218, 32), (32, 1))
        assert_size_stride(add_130, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_20, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_491, (1218, 896), (896, 1))
        assert_size_stride(mm_180, (1218, 32), (32, 1))
        assert_size_stride(mm_182, (1218, 32), (32, 1))
        assert_size_stride(mm_184, (1218, 32), (32, 1))
        assert_size_stride(add_135, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_510, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_511, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_40, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_41, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_42, (), ())
        assert_size_stride(getitem_43, (), ())
        assert_size_stride(view_515, (1218, 896), (896, 1))
        assert_size_stride(mm_187, (1218, 32), (32, 1))
        assert_size_stride(add_138, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_21, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_521, (1218, 896), (896, 1))
        assert_size_stride(mm_190, (1218, 32), (32, 1))
        assert_size_stride(add_140, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_193, (1218, 32), (32, 1))
        assert_size_stride(add_141, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_533, (1218, 4864), (4864, 1))
        assert_size_stride(mm_196, (1218, 32), (32, 1))
        assert_size_stride(add_143, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_22, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_539, (1218, 896), (896, 1))
        assert_size_stride(mm_198, (1218, 32), (32, 1))
        assert_size_stride(mm_200, (1218, 32), (32, 1))
        assert_size_stride(mm_202, (1218, 32), (32, 1))
        assert_size_stride(add_148, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_558, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_559, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_44, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_45, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_46, (), ())
        assert_size_stride(getitem_47, (), ())
        assert_size_stride(view_563, (1218, 896), (896, 1))
        assert_size_stride(mm_205, (1218, 32), (32, 1))
        assert_size_stride(add_151, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_23, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_569, (1218, 896), (896, 1))
        assert_size_stride(mm_208, (1218, 32), (32, 1))
        assert_size_stride(add_153, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_211, (1218, 32), (32, 1))
        assert_size_stride(add_154, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_581, (1218, 4864), (4864, 1))
        assert_size_stride(mm_214, (1218, 32), (32, 1))
        assert_size_stride(add_156, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_24, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_587, (1218, 896), (896, 1))
        assert_size_stride(mm_216, (1218, 32), (32, 1))
        assert_size_stride(mm_218, (1218, 32), (32, 1))
        assert_size_stride(mm_220, (1218, 32), (32, 1))
        assert_size_stride(add_161, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_606, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_607, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_48, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_49, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_50, (), ())
        assert_size_stride(getitem_51, (), ())
        assert_size_stride(view_611, (1218, 896), (896, 1))
        assert_size_stride(mm_223, (1218, 32), (32, 1))
        assert_size_stride(add_164, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_25, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_617, (1218, 896), (896, 1))
        assert_size_stride(mm_226, (1218, 32), (32, 1))
        assert_size_stride(add_166, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_229, (1218, 32), (32, 1))
        assert_size_stride(add_167, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_629, (1218, 4864), (4864, 1))
        assert_size_stride(mm_232, (1218, 32), (32, 1))
        assert_size_stride(add_169, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_26, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_635, (1218, 896), (896, 1))
        assert_size_stride(mm_234, (1218, 32), (32, 1))
        assert_size_stride(mm_236, (1218, 32), (32, 1))
        assert_size_stride(mm_238, (1218, 32), (32, 1))
        assert_size_stride(add_174, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_654, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_655, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_52, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_53, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_54, (), ())
        assert_size_stride(getitem_55, (), ())
        assert_size_stride(view_659, (1218, 896), (896, 1))
        assert_size_stride(mm_241, (1218, 32), (32, 1))
        assert_size_stride(add_177, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_27, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_665, (1218, 896), (896, 1))
        assert_size_stride(mm_244, (1218, 32), (32, 1))
        assert_size_stride(add_179, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_247, (1218, 32), (32, 1))
        assert_size_stride(add_180, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_677, (1218, 4864), (4864, 1))
        assert_size_stride(mm_250, (1218, 32), (32, 1))
        assert_size_stride(add_182, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_28, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_683, (1218, 896), (896, 1))
        assert_size_stride(mm_252, (1218, 32), (32, 1))
        assert_size_stride(mm_254, (1218, 32), (32, 1))
        assert_size_stride(mm_256, (1218, 32), (32, 1))
        assert_size_stride(add_187, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_702, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_703, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_56, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_57, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_58, (), ())
        assert_size_stride(getitem_59, (), ())
        assert_size_stride(view_707, (1218, 896), (896, 1))
        assert_size_stride(mm_259, (1218, 32), (32, 1))
        assert_size_stride(add_190, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_29, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_713, (1218, 896), (896, 1))
        assert_size_stride(mm_262, (1218, 32), (32, 1))
        assert_size_stride(add_192, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_265, (1218, 32), (32, 1))
        assert_size_stride(add_193, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_725, (1218, 4864), (4864, 1))
        assert_size_stride(mm_268, (1218, 32), (32, 1))
        assert_size_stride(add_195, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_30, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_731, (1218, 896), (896, 1))
        assert_size_stride(mm_270, (1218, 32), (32, 1))
        assert_size_stride(mm_272, (1218, 32), (32, 1))
        assert_size_stride(mm_274, (1218, 32), (32, 1))
        assert_size_stride(add_200, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_750, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_751, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_60, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_61, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_62, (), ())
        assert_size_stride(getitem_63, (), ())
        assert_size_stride(view_755, (1218, 896), (896, 1))
        assert_size_stride(mm_277, (1218, 32), (32, 1))
        assert_size_stride(add_203, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_31, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_761, (1218, 896), (896, 1))
        assert_size_stride(mm_280, (1218, 32), (32, 1))
        assert_size_stride(add_205, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_283, (1218, 32), (32, 1))
        assert_size_stride(add_206, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_773, (1218, 4864), (4864, 1))
        assert_size_stride(mm_286, (1218, 32), (32, 1))
        assert_size_stride(add_208, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_32, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_779, (1218, 896), (896, 1))
        assert_size_stride(mm_288, (1218, 32), (32, 1))
        assert_size_stride(mm_290, (1218, 32), (32, 1))
        assert_size_stride(mm_292, (1218, 32), (32, 1))
        assert_size_stride(add_213, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_798, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_799, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_64, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_65, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_66, (), ())
        assert_size_stride(getitem_67, (), ())
        assert_size_stride(view_803, (1218, 896), (896, 1))
        assert_size_stride(mm_295, (1218, 32), (32, 1))
        assert_size_stride(add_216, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_33, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_809, (1218, 896), (896, 1))
        assert_size_stride(mm_298, (1218, 32), (32, 1))
        assert_size_stride(add_218, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_301, (1218, 32), (32, 1))
        assert_size_stride(add_219, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_821, (1218, 4864), (4864, 1))
        assert_size_stride(mm_304, (1218, 32), (32, 1))
        assert_size_stride(add_221, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_34, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_827, (1218, 896), (896, 1))
        assert_size_stride(mm_306, (1218, 32), (32, 1))
        assert_size_stride(mm_308, (1218, 32), (32, 1))
        assert_size_stride(mm_310, (1218, 32), (32, 1))
        assert_size_stride(add_226, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_846, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_847, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_68, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_69, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_70, (), ())
        assert_size_stride(getitem_71, (), ())
        assert_size_stride(view_851, (1218, 896), (896, 1))
        assert_size_stride(mm_313, (1218, 32), (32, 1))
        assert_size_stride(add_229, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_35, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_857, (1218, 896), (896, 1))
        assert_size_stride(mm_316, (1218, 32), (32, 1))
        assert_size_stride(add_231, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_319, (1218, 32), (32, 1))
        assert_size_stride(add_232, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_869, (1218, 4864), (4864, 1))
        assert_size_stride(mm_322, (1218, 32), (32, 1))
        assert_size_stride(add_234, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_36, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_875, (1218, 896), (896, 1))
        assert_size_stride(mm_324, (1218, 32), (32, 1))
        assert_size_stride(mm_326, (1218, 32), (32, 1))
        assert_size_stride(mm_328, (1218, 32), (32, 1))
        assert_size_stride(add_239, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_894, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_895, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_72, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_73, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_74, (), ())
        assert_size_stride(getitem_75, (), ())
        assert_size_stride(view_899, (1218, 896), (896, 1))
        assert_size_stride(mm_331, (1218, 32), (32, 1))
        assert_size_stride(add_242, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_37, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_905, (1218, 896), (896, 1))
        assert_size_stride(mm_334, (1218, 32), (32, 1))
        assert_size_stride(add_244, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_337, (1218, 32), (32, 1))
        assert_size_stride(add_245, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_917, (1218, 4864), (4864, 1))
        assert_size_stride(mm_340, (1218, 32), (32, 1))
        assert_size_stride(add_247, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_38, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_923, (1218, 896), (896, 1))
        assert_size_stride(mm_342, (1218, 32), (32, 1))
        assert_size_stride(mm_344, (1218, 32), (32, 1))
        assert_size_stride(mm_346, (1218, 32), (32, 1))
        assert_size_stride(add_252, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_942, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_943, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_76, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_77, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_78, (), ())
        assert_size_stride(getitem_79, (), ())
        assert_size_stride(view_947, (1218, 896), (896, 1))
        assert_size_stride(mm_349, (1218, 32), (32, 1))
        assert_size_stride(add_255, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_39, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_953, (1218, 896), (896, 1))
        assert_size_stride(mm_352, (1218, 32), (32, 1))
        assert_size_stride(add_257, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_355, (1218, 32), (32, 1))
        assert_size_stride(add_258, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_965, (1218, 4864), (4864, 1))
        assert_size_stride(mm_358, (1218, 32), (32, 1))
        assert_size_stride(add_260, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_40, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_971, (1218, 896), (896, 1))
        assert_size_stride(mm_360, (1218, 32), (32, 1))
        assert_size_stride(mm_362, (1218, 32), (32, 1))
        assert_size_stride(mm_364, (1218, 32), (32, 1))
        assert_size_stride(add_265, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_990, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_991, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_80, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_81, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_82, (), ())
        assert_size_stride(getitem_83, (), ())
        assert_size_stride(view_995, (1218, 896), (896, 1))
        assert_size_stride(mm_367, (1218, 32), (32, 1))
        assert_size_stride(add_268, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_41, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_1001, (1218, 896), (896, 1))
        assert_size_stride(mm_370, (1218, 32), (32, 1))
        assert_size_stride(add_270, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_373, (1218, 32), (32, 1))
        assert_size_stride(add_271, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_1013, (1218, 4864), (4864, 1))
        assert_size_stride(mm_376, (1218, 32), (32, 1))
        assert_size_stride(add_273, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_42, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_1019, (1218, 896), (896, 1))
        assert_size_stride(mm_378, (1218, 32), (32, 1))
        assert_size_stride(mm_380, (1218, 32), (32, 1))
        assert_size_stride(mm_382, (1218, 32), (32, 1))
        assert_size_stride(add_278, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_1038, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_1039, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_84, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_85, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_86, (), ())
        assert_size_stride(getitem_87, (), ())
        assert_size_stride(view_1043, (1218, 896), (896, 1))
        assert_size_stride(mm_385, (1218, 32), (32, 1))
        assert_size_stride(add_281, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_43, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_1049, (1218, 896), (896, 1))
        assert_size_stride(mm_388, (1218, 32), (32, 1))
        assert_size_stride(add_283, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_391, (1218, 32), (32, 1))
        assert_size_stride(add_284, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_1061, (1218, 4864), (4864, 1))
        assert_size_stride(mm_394, (1218, 32), (32, 1))
        assert_size_stride(add_286, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_44, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_1067, (1218, 896), (896, 1))
        assert_size_stride(mm_396, (1218, 32), (32, 1))
        assert_size_stride(mm_398, (1218, 32), (32, 1))
        assert_size_stride(mm_400, (1218, 32), (32, 1))
        assert_size_stride(add_291, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_1086, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_1087, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_88, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_89, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_90, (), ())
        assert_size_stride(getitem_91, (), ())
        assert_size_stride(view_1091, (1218, 896), (896, 1))
        assert_size_stride(mm_403, (1218, 32), (32, 1))
        assert_size_stride(add_294, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_45, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_1097, (1218, 896), (896, 1))
        assert_size_stride(mm_406, (1218, 32), (32, 1))
        assert_size_stride(add_296, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_409, (1218, 32), (32, 1))
        assert_size_stride(add_297, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_1109, (1218, 4864), (4864, 1))
        assert_size_stride(mm_412, (1218, 32), (32, 1))
        assert_size_stride(add_299, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_46, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_1115, (1218, 896), (896, 1))
        assert_size_stride(mm_414, (1218, 32), (32, 1))
        assert_size_stride(mm_416, (1218, 32), (32, 1))
        assert_size_stride(mm_418, (1218, 32), (32, 1))
        assert_size_stride(add_304, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(view_1134, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(view_1135, (1, 14, 1218, 64), (1091328, 77952, 64, 1))
        assert_size_stride(getitem_92, (1, 14, 1218, 64), (1091328, 64, 896, 1))
        assert_size_stride(getitem_93, (1, 14, 1248), (17472, 1248, 1))
        assert_size_stride(getitem_94, (), ())
        assert_size_stride(getitem_95, (), ())
        assert_size_stride(view_1139, (1218, 896), (896, 1))
        assert_size_stride(mm_421, (1218, 32), (32, 1))
        assert_size_stride(add_307, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_47, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_1145, (1218, 896), (896, 1))
        assert_size_stride(mm_424, (1218, 32), (32, 1))
        assert_size_stride(add_309, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(mm_427, (1218, 32), (32, 1))
        assert_size_stride(add_310, (1, 1218, 4864), (5924352, 4864, 1))
        assert_size_stride(view_1157, (1218, 4864), (4864, 1))
        assert_size_stride(mm_430, (1218, 32), (32, 1))
        assert_size_stride(add_312, (1, 1218, 896), (1091328, 896, 1))
        assert_size_stride(rsqrt_48, (1, 1218, 1), (1248, 1, 1))
        assert_size_stride(view_1161, (1025, 896), (896, 1))
        assert_size_stride(permute_608, (896, 32), (32, 1))
        assert_size_stride(permute_612, (32, 4864), (4864, 1))
        assert_size_stride(permute_617, (4864, 32), (32, 1))
        assert_size_stride(permute_621, (32, 896), (896, 1))
        assert_size_stride(permute_626, (4864, 32), (32, 1))
        assert_size_stride(permute_630, (32, 896), (896, 1))
        assert_size_stride(permute_635, (896, 32), (32, 1))
        assert_size_stride(permute_639, (32, 896), (896, 1))
        assert_size_stride(permute_646, (128, 32), (32, 1))
        assert_size_stride(permute_650, (32, 896), (896, 1))
        assert_size_stride(permute_656, (128, 32), (32, 1))
        assert_size_stride(permute_660, (32, 896), (896, 1))
        assert_size_stride(permute_666, (896, 32), (32, 1))
        assert_size_stride(permute_670, (32, 896), (896, 1))
        assert_size_stride(permute_675, (896, 32), (32, 1))
        assert_size_stride(permute_679, (32, 4864), (4864, 1))
        assert_size_stride(permute_684, (4864, 32), (32, 1))
        assert_size_stride(permute_688, (32, 896), (896, 1))
        assert_size_stride(permute_693, (4864, 32), (32, 1))
        assert_size_stride(permute_697, (32, 896), (896, 1))
        assert_size_stride(permute_702, (896, 32), (32, 1))
        assert_size_stride(permute_706, (32, 896), (896, 1))
        assert_size_stride(permute_713, (128, 32), (32, 1))
        assert_size_stride(permute_717, (32, 896), (896, 1))
        assert_size_stride(permute_723, (128, 32), (32, 1))
        assert_size_stride(permute_727, (32, 896), (896, 1))
        assert_size_stride(permute_733, (896, 32), (32, 1))
        assert_size_stride(permute_737, (32, 896), (896, 1))
        assert_size_stride(permute_742, (896, 32), (32, 1))
        assert_size_stride(permute_746, (32, 4864), (4864, 1))
        assert_size_stride(permute_751, (4864, 32), (32, 1))
        assert_size_stride(permute_755, (32, 896), (896, 1))
        assert_size_stride(permute_760, (4864, 32), (32, 1))
        assert_size_stride(permute_764, (32, 896), (896, 1))
        assert_size_stride(permute_769, (896, 32), (32, 1))
        assert_size_stride(permute_773, (32, 896), (896, 1))
        assert_size_stride(permute_780, (128, 32), (32, 1))
        assert_size_stride(permute_784, (32, 896), (896, 1))
        assert_size_stride(permute_790, (128, 32), (32, 1))
        assert_size_stride(permute_794, (32, 896), (896, 1))
        assert_size_stride(permute_800, (896, 32), (32, 1))
        assert_size_stride(permute_804, (32, 896), (896, 1))
        assert_size_stride(permute_809, (896, 32), (32, 1))
        assert_size_stride(permute_813, (32, 4864), (4864, 1))
        assert_size_stride(permute_818, (4864, 32), (32, 1))
        assert_size_stride(permute_822, (32, 896), (896, 1))
        assert_size_stride(permute_827, (4864, 32), (32, 1))
        assert_size_stride(permute_831, (32, 896), (896, 1))
        assert_size_stride(permute_836, (896, 32), (32, 1))
        assert_size_stride(permute_840, (32, 896), (896, 1))
        assert_size_stride(permute_847, (128, 32), (32, 1))
        assert_size_stride(permute_851, (32, 896), (896, 1))
        assert_size_stride(permute_857, (128, 32), (32, 1))
        assert_size_stride(permute_861, (32, 896), (896, 1))
        assert_size_stride(permute_867, (896, 32), (32, 1))
        assert_size_stride(permute_871, (32, 896), (896, 1))
        assert_size_stride(permute_876, (896, 32), (32, 1))
        assert_size_stride(permute_880, (32, 4864), (4864, 1))
        assert_size_stride(permute_885, (4864, 32), (32, 1))
        assert_size_stride(permute_889, (32, 896), (896, 1))
        assert_size_stride(permute_894, (4864, 32), (32, 1))
        assert_size_stride(permute_898, (32, 896), (896, 1))
        assert_size_stride(permute_903, (896, 32), (32, 1))
        assert_size_stride(permute_907, (32, 896), (896, 1))
        assert_size_stride(permute_914, (128, 32), (32, 1))
        assert_size_stride(permute_918, (32, 896), (896, 1))
        assert_size_stride(permute_924, (128, 32), (32, 1))
        assert_size_stride(permute_928, (32, 896), (896, 1))
        assert_size_stride(permute_934, (896, 32), (32, 1))
        assert_size_stride(permute_938, (32, 896), (896, 1))
        assert_size_stride(permute_943, (896, 32), (32, 1))
        assert_size_stride(permute_947, (32, 4864), (4864, 1))
        assert_size_stride(permute_952, (4864, 32), (32, 1))
        assert_size_stride(permute_956, (32, 896), (896, 1))
        assert_size_stride(permute_961, (4864, 32), (32, 1))
        assert_size_stride(permute_965, (32, 896), (896, 1))
        assert_size_stride(permute_970, (896, 32), (32, 1))
        assert_size_stride(permute_974, (32, 896), (896, 1))
        assert_size_stride(permute_981, (128, 32), (32, 1))
        assert_size_stride(permute_985, (32, 896), (896, 1))
        assert_size_stride(permute_991, (128, 32), (32, 1))
        assert_size_stride(permute_995, (32, 896), (896, 1))
        assert_size_stride(permute_1001, (896, 32), (32, 1))
        assert_size_stride(permute_1005, (32, 896), (896, 1))
        assert_size_stride(permute_1010, (896, 32), (32, 1))
        assert_size_stride(permute_1014, (32, 4864), (4864, 1))
        assert_size_stride(permute_1019, (4864, 32), (32, 1))
        assert_size_stride(permute_1023, (32, 896), (896, 1))
        assert_size_stride(permute_1028, (4864, 32), (32, 1))
        assert_size_stride(permute_1032, (32, 896), (896, 1))
        assert_size_stride(permute_1037, (896, 32), (32, 1))
        assert_size_stride(permute_1041, (32, 896), (896, 1))
        assert_size_stride(permute_1048, (128, 32), (32, 1))
        assert_size_stride(permute_1052, (32, 896), (896, 1))
        assert_size_stride(permute_1058, (128, 32), (32, 1))
        assert_size_stride(permute_1062, (32, 896), (896, 1))
        assert_size_stride(permute_1068, (896, 32), (32, 1))
        assert_size_stride(permute_1072, (32, 896), (896, 1))
        assert_size_stride(permute_1077, (896, 32), (32, 1))
        assert_size_stride(permute_1081, (32, 4864), (4864, 1))
        assert_size_stride(permute_1086, (4864, 32), (32, 1))
        assert_size_stride(permute_1090, (32, 896), (896, 1))
        assert_size_stride(permute_1095, (4864, 32), (32, 1))
        assert_size_stride(permute_1099, (32, 896), (896, 1))
        assert_size_stride(permute_1104, (896, 32), (32, 1))
        assert_size_stride(permute_1108, (32, 896), (896, 1))
        assert_size_stride(permute_1115, (128, 32), (32, 1))
        assert_size_stride(permute_1119, (32, 896), (896, 1))
        assert_size_stride(permute_1125, (128, 32), (32, 1))
        assert_size_stride(permute_1129, (32, 896), (896, 1))
        assert_size_stride(permute_1135, (896, 32), (32, 1))
        assert_size_stride(permute_1139, (32, 896), (896, 1))
        assert_size_stride(permute_1144, (896, 32), (32, 1))
        assert_size_stride(permute_1148, (32, 4864), (4864, 1))
        assert_size_stride(permute_1153, (4864, 32), (32, 1))
        assert_size_stride(permute_1157, (32, 896), (896, 1))
        assert_size_stride(permute_1162, (4864, 32), (32, 1))
        assert_size_stride(permute_1166, (32, 896), (896, 1))
        assert_size_stride(permute_1171, (896, 32), (32, 1))
        assert_size_stride(permute_1175, (32, 896), (896, 1))
        assert_size_stride(permute_1182, (128, 32), (32, 1))
        assert_size_stride(permute_1186, (32, 896), (896, 1))
        assert_size_stride(permute_1192, (128, 32), (32, 1))
        assert_size_stride(permute_1196, (32, 896), (896, 1))
        assert_size_stride(permute_1202, (896, 32), (32, 1))
        assert_size_stride(permute_1206, (32, 896), (896, 1))
        assert_size_stride(permute_1211, (896, 32), (32, 1))
        assert_size_stride(permute_1215, (32, 4864), (4864, 1))
        assert_size_stride(permute_1220, (4864, 32), (32, 1))
        assert_size_stride(permute_1224, (32, 896), (896, 1))
        assert_size_stride(permute_1229, (4864, 32), (32, 1))
        assert_size_stride(permute_1233, (32, 896), (896, 1))
        assert_size_stride(permute_1238, (896, 32), (32, 1))
        assert_size_stride(permute_1242, (32, 896), (896, 1))
        assert_size_stride(permute_1249, (128, 32), (32, 1))
        assert_size_stride(permute_1253, (32, 896), (896, 1))
        assert_size_stride(permute_1259, (128, 32), (32, 1))
        assert_size_stride(permute_1263, (32, 896), (896, 1))
        assert_size_stride(permute_1269, (896, 32), (32, 1))
        assert_size_stride(permute_1273, (32, 896), (896, 1))
        assert_size_stride(permute_1278, (896, 32), (32, 1))
        assert_size_stride(permute_1282, (32, 4864), (4864, 1))
        assert_size_stride(permute_1287, (4864, 32), (32, 1))
        assert_size_stride(permute_1291, (32, 896), (896, 1))
        assert_size_stride(permute_1296, (4864, 32), (32, 1))
        assert_size_stride(permute_1300, (32, 896), (896, 1))
        assert_size_stride(permute_1305, (896, 32), (32, 1))
        assert_size_stride(permute_1309, (32, 896), (896, 1))
        assert_size_stride(permute_1316, (128, 32), (32, 1))
        assert_size_stride(permute_1320, (32, 896), (896, 1))
        assert_size_stride(permute_1326, (128, 32), (32, 1))
        assert_size_stride(permute_1330, (32, 896), (896, 1))
        assert_size_stride(permute_1336, (896, 32), (32, 1))
        assert_size_stride(permute_1340, (32, 896), (896, 1))
        assert_size_stride(permute_1345, (896, 32), (32, 1))
        assert_size_stride(permute_1349, (32, 4864), (4864, 1))
        assert_size_stride(permute_1354, (4864, 32), (32, 1))
        assert_size_stride(permute_1358, (32, 896), (896, 1))
        assert_size_stride(permute_1363, (4864, 32), (32, 1))
        assert_size_stride(permute_1367, (32, 896), (896, 1))
        assert_size_stride(permute_1372, (896, 32), (32, 1))
        assert_size_stride(permute_1376, (32, 896), (896, 1))
        assert_size_stride(permute_1383, (128, 32), (32, 1))
        assert_size_stride(permute_1387, (32, 896), (896, 1))
        assert_size_stride(permute_1393, (128, 32), (32, 1))
        assert_size_stride(permute_1397, (32, 896), (896, 1))
        assert_size_stride(permute_1403, (896, 32), (32, 1))
        assert_size_stride(permute_1407, (32, 896), (896, 1))
        assert_size_stride(permute_1412, (896, 32), (32, 1))
        assert_size_stride(permute_1416, (32, 4864), (4864, 1))
        assert_size_stride(permute_1421, (4864, 32), (32, 1))
        assert_size_stride(permute_1425, (32, 896), (896, 1))
        assert_size_stride(permute_1430, (4864, 32), (32, 1))
        assert_size_stride(permute_1434, (32, 896), (896, 1))
        assert_size_stride(permute_1439, (896, 32), (32, 1))
        assert_size_stride(permute_1443, (32, 896), (896, 1))
        assert_size_stride(permute_1450, (128, 32), (32, 1))
        assert_size_stride(permute_1454, (32, 896), (896, 1))
        assert_size_stride(permute_1460, (128, 32), (32, 1))
        assert_size_stride(permute_1464, (32, 896), (896, 1))
        assert_size_stride(permute_1470, (896, 32), (32, 1))
        assert_size_stride(permute_1474, (32, 896), (896, 1))
        assert_size_stride(permute_1479, (896, 32), (32, 1))
        assert_size_stride(permute_1483, (32, 4864), (4864, 1))
        assert_size_stride(permute_1488, (4864, 32), (32, 1))
        assert_size_stride(permute_1492, (32, 896), (896, 1))
        assert_size_stride(permute_1497, (4864, 32), (32, 1))
        assert_size_stride(permute_1501, (32, 896), (896, 1))
        assert_size_stride(permute_1506, (896, 32), (32, 1))
        assert_size_stride(permute_1510, (32, 896), (896, 1))
        assert_size_stride(permute_1517, (128, 32), (32, 1))
        assert_size_stride(permute_1521, (32, 896), (896, 1))
        assert_size_stride(permute_1527, (128, 32), (32, 1))
        assert_size_stride(permute_1531, (32, 896), (896, 1))
        assert_size_stride(permute_1537, (896, 32), (32, 1))
        assert_size_stride(permute_1541, (32, 896), (896, 1))
        assert_size_stride(permute_1546, (896, 32), (32, 1))
        assert_size_stride(permute_1550, (32, 4864), (4864, 1))
        assert_size_stride(permute_1555, (4864, 32), (32, 1))
        assert_size_stride(permute_1559, (32, 896), (896, 1))
        assert_size_stride(permute_1564, (4864, 32), (32, 1))
        assert_size_stride(permute_1568, (32, 896), (896, 1))
        assert_size_stride(permute_1573, (896, 32), (32, 1))
        assert_size_stride(permute_1577, (32, 896), (896, 1))
        assert_size_stride(permute_1584, (128, 32), (32, 1))
        assert_size_stride(permute_1588, (32, 896), (896, 1))
        assert_size_stride(permute_1594, (128, 32), (32, 1))
        assert_size_stride(permute_1598, (32, 896), (896, 1))
        assert_size_stride(permute_1604, (896, 32), (32, 1))
        assert_size_stride(permute_1608, (32, 896), (896, 1))
        assert_size_stride(permute_1613, (896, 32), (32, 1))
        assert_size_stride(permute_1617, (32, 4864), (4864, 1))
        assert_size_stride(permute_1622, (4864, 32), (32, 1))
        assert_size_stride(permute_1626, (32, 896), (896, 1))
        assert_size_stride(permute_1631, (4864, 32), (32, 1))
        assert_size_stride(permute_1635, (32, 896), (896, 1))
        assert_size_stride(permute_1640, (896, 32), (32, 1))
        assert_size_stride(permute_1644, (32, 896), (896, 1))
        assert_size_stride(permute_1651, (128, 32), (32, 1))
        assert_size_stride(permute_1655, (32, 896), (896, 1))
        assert_size_stride(permute_1661, (128, 32), (32, 1))
        assert_size_stride(permute_1665, (32, 896), (896, 1))
        assert_size_stride(permute_1671, (896, 32), (32, 1))
        assert_size_stride(permute_1675, (32, 896), (896, 1))
        assert_size_stride(permute_1680, (896, 32), (32, 1))
        assert_size_stride(permute_1684, (32, 4864), (4864, 1))
        assert_size_stride(permute_1689, (4864, 32), (32, 1))
        assert_size_stride(permute_1693, (32, 896), (896, 1))
        assert_size_stride(permute_1698, (4864, 32), (32, 1))
        assert_size_stride(permute_1702, (32, 896), (896, 1))
        assert_size_stride(permute_1707, (896, 32), (32, 1))
        assert_size_stride(permute_1711, (32, 896), (896, 1))
        assert_size_stride(permute_1718, (128, 32), (32, 1))
        assert_size_stride(permute_1722, (32, 896), (896, 1))
        assert_size_stride(permute_1728, (128, 32), (32, 1))
        assert_size_stride(permute_1732, (32, 896), (896, 1))
        assert_size_stride(permute_1738, (896, 32), (32, 1))
        assert_size_stride(permute_1742, (32, 896), (896, 1))
        assert_size_stride(permute_1747, (896, 32), (32, 1))
        assert_size_stride(permute_1751, (32, 4864), (4864, 1))
        assert_size_stride(permute_1756, (4864, 32), (32, 1))
        assert_size_stride(permute_1760, (32, 896), (896, 1))
        assert_size_stride(permute_1765, (4864, 32), (32, 1))
        assert_size_stride(permute_1769, (32, 896), (896, 1))
        assert_size_stride(permute_1774, (896, 32), (32, 1))
        assert_size_stride(permute_1778, (32, 896), (896, 1))
        assert_size_stride(permute_1785, (128, 32), (32, 1))
        assert_size_stride(permute_1789, (32, 896), (896, 1))
        assert_size_stride(permute_1795, (128, 32), (32, 1))
        assert_size_stride(permute_1799, (32, 896), (896, 1))
        assert_size_stride(permute_1805, (896, 32), (32, 1))
        assert_size_stride(permute_1809, (32, 896), (896, 1))
        assert_size_stride(permute_1814, (896, 32), (32, 1))
        assert_size_stride(permute_1818, (32, 4864), (4864, 1))
        assert_size_stride(permute_1823, (4864, 32), (32, 1))
        assert_size_stride(permute_1827, (32, 896), (896, 1))
        assert_size_stride(permute_1832, (4864, 32), (32, 1))
        assert_size_stride(permute_1836, (32, 896), (896, 1))
        assert_size_stride(permute_1841, (896, 32), (32, 1))
        assert_size_stride(permute_1845, (32, 896), (896, 1))
        assert_size_stride(permute_1852, (128, 32), (32, 1))
        assert_size_stride(permute_1856, (32, 896), (896, 1))
        assert_size_stride(permute_1862, (128, 32), (32, 1))
        assert_size_stride(permute_1866, (32, 896), (896, 1))
        assert_size_stride(permute_1872, (896, 32), (32, 1))
        assert_size_stride(permute_1876, (32, 896), (896, 1))
        assert_size_stride(permute_1881, (896, 32), (32, 1))
        assert_size_stride(permute_1885, (32, 4864), (4864, 1))
        assert_size_stride(permute_1890, (4864, 32), (32, 1))
        assert_size_stride(permute_1894, (32, 896), (896, 1))
        assert_size_stride(permute_1899, (4864, 32), (32, 1))
        assert_size_stride(permute_1903, (32, 896), (896, 1))
        assert_size_stride(permute_1908, (896, 32), (32, 1))
        assert_size_stride(permute_1912, (32, 896), (896, 1))
        assert_size_stride(permute_1919, (128, 32), (32, 1))
        assert_size_stride(permute_1923, (32, 896), (896, 1))
        assert_size_stride(permute_1929, (128, 32), (32, 1))
        assert_size_stride(permute_1933, (32, 896), (896, 1))
        assert_size_stride(permute_1939, (896, 32), (32, 1))
        assert_size_stride(permute_1943, (32, 896), (896, 1))
        assert_size_stride(permute_1948, (896, 32), (32, 1))
        assert_size_stride(permute_1952, (32, 4864), (4864, 1))
        assert_size_stride(permute_1957, (4864, 32), (32, 1))
        assert_size_stride(permute_1961, (32, 896), (896, 1))
        assert_size_stride(permute_1966, (4864, 32), (32, 1))
        assert_size_stride(permute_1970, (32, 896), (896, 1))
        assert_size_stride(permute_1975, (896, 32), (32, 1))
        assert_size_stride(permute_1979, (32, 896), (896, 1))
        assert_size_stride(permute_1986, (128, 32), (32, 1))
        assert_size_stride(permute_1990, (32, 896), (896, 1))
        assert_size_stride(permute_1996, (128, 32), (32, 1))
        assert_size_stride(permute_2000, (32, 896), (896, 1))
        assert_size_stride(permute_2006, (896, 32), (32, 1))
        assert_size_stride(permute_2010, (32, 896), (896, 1))
        assert_size_stride(permute_2015, (896, 32), (32, 1))
        assert_size_stride(permute_2019, (32, 4864), (4864, 1))
        assert_size_stride(permute_2024, (4864, 32), (32, 1))
        assert_size_stride(permute_2028, (32, 896), (896, 1))
        assert_size_stride(permute_2033, (4864, 32), (32, 1))
        assert_size_stride(permute_2037, (32, 896), (896, 1))
        assert_size_stride(permute_2042, (896, 32), (32, 1))
        assert_size_stride(permute_2046, (32, 896), (896, 1))
        assert_size_stride(permute_2053, (128, 32), (32, 1))
        assert_size_stride(permute_2057, (32, 896), (896, 1))
        assert_size_stride(permute_2063, (128, 32), (32, 1))
        assert_size_stride(permute_2067, (32, 896), (896, 1))
        assert_size_stride(permute_2073, (896, 32), (32, 1))
        assert_size_stride(permute_2077, (32, 896), (896, 1))
        assert_size_stride(permute_2082, (896, 32), (32, 1))
        assert_size_stride(permute_2086, (32, 4864), (4864, 1))
        assert_size_stride(permute_2091, (4864, 32), (32, 1))
        assert_size_stride(permute_2095, (32, 896), (896, 1))
        assert_size_stride(permute_2100, (4864, 32), (32, 1))
        assert_size_stride(permute_2104, (32, 896), (896, 1))
        assert_size_stride(permute_2109, (896, 32), (32, 1))
        assert_size_stride(permute_2113, (32, 896), (896, 1))
        assert_size_stride(permute_2120, (128, 32), (32, 1))
        assert_size_stride(permute_2124, (32, 896), (896, 1))
        assert_size_stride(permute_2130, (128, 32), (32, 1))
        assert_size_stride(permute_2134, (32, 896), (896, 1))
        assert_size_stride(permute_2140, (896, 32), (32, 1))
        assert_size_stride(permute_2144, (32, 896), (896, 1))
        assert_size_stride(permute_2149, (896, 32), (32, 1))
        assert_size_stride(permute_2153, (32, 4864), (4864, 1))
        assert_size_stride(permute_2158, (4864, 32), (32, 1))
        assert_size_stride(permute_2162, (32, 896), (896, 1))
        assert_size_stride(permute_2167, (4864, 32), (32, 1))
        assert_size_stride(permute_2171, (32, 896), (896, 1))
        assert_size_stride(permute_2176, (896, 32), (32, 1))
        assert_size_stride(permute_2180, (32, 896), (896, 1))
        assert_size_stride(permute_2187, (128, 32), (32, 1))
        assert_size_stride(permute_2195, (128, 32), (32, 1))
        assert_size_stride(permute_2203, (896, 32), (32, 1))
        assert_size_stride(tangents_1, (1, 1025, 151936), (155734400, 151936, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((1025, 151936), (151936, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [convert_element_type_1905, view_1163], Original ATen: [aten._to_copy, aten.view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_view_0.run(tangents_1, buf0, 155734400, stream=stream0)
            del tangents_1
            buf2 = empty_strided_cuda((1025, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [logits, permute_604, mm_434], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(buf0, primals_630, out=buf2)
            del primals_630
            buf4 = add_312; del add_312  # reuse
            buf5 = empty_strided_cuda((1218, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1164, full_default_49, mul_412, convert_element_type_1910, hidden_states_240, mul_413, mul_414, sum_1, pow_50, mul_415, mul_416, expand_77, div, pow_51, mul_417, mul_418, add_314, convert_element_type_1911, mul_419, view_1165], Original ATen: [aten.view, aten.slice_backward, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div, aten.add]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_slice_backward_sum_view_1.run(buf4, buf2, primals_629, rsqrt_48, buf5, 1218, 896, stream=stream0)
            del buf2
            del primals_629
            del rsqrt_48
            buf6 = empty_strided_cuda((896, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_606, mm_435], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf5, (896, 1218), (1, 896), 0), mm_430, out=buf6)
            del mm_430
            buf7 = empty_strided_cuda((1218, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mm_436], Original ATen: [aten.mm]
            extern_kernels.mm(buf5, permute_608, out=buf7)
            del buf5
            del permute_608
            buf9 = empty_strided_cuda((32, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_610, mm_437], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf7, (32, 1218), (1, 32), 0), view_1157, out=buf9)
            del view_1157
            buf8 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1916], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf6, buf8, 28672, stream=stream0)
            del buf6
            buf11 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1922], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf9, buf11, 155648, stream=stream0)
            del buf9
            buf10 = empty_strided_cuda((1218, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mm_438], Original ATen: [aten.mm]
            extern_kernels.mm(buf7, permute_612, out=buf10)
            del buf7
            del permute_612
            buf12 = empty_strided_cuda((1218, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1169, result_504, permute_614, mm_439], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf4, (1218, 896), (896, 1), 0), primals_626, out=buf12)
            del primals_626
            buf13 = empty_strided_cuda((1, 1218, 4864), (5924352, 4864, 1), torch.bfloat16)
            buf20 = empty_strided_cuda((1, 1218, 4864), (5924352, 4864, 1), torch.bfloat16)
            buf22 = empty_strided_cuda((1, 1218, 4864), (5924352, 4864, 1), torch.bfloat16)
            buf29 = empty_strided_cuda((1, 1218, 4864), (5924352, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1168, view_1170, add_315, silu_23, mul_420, mul_421, mul_422, convert_element_type_1940, neg_48, exp, add_317, reciprocal, mul_423, mul_424, sub, mul_425, add_318, mul_426, convert_element_type_1942, mul_427], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf10, buf12, add_309, add_310, buf13, buf20, buf22, buf29, 5924352, stream=stream0)
            del add_309
            del add_310
            buf21 = empty_strided_cuda((1218, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1168, view_1170, add_315, silu_23, mul_420, view_1175, result_501, permute_623, mm_444], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf20, (1218, 4864), (4864, 1), 0), primals_623, out=buf21)
            del primals_623
            buf30 = empty_strided_cuda((1218, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1168, view_1170, add_315, silu_23, mul_421, convert_element_type_1940, neg_48, exp, add_317, reciprocal, mul_423, mul_424, sub, mul_425, add_318, mul_426, convert_element_type_1942, view_1181, result_498, permute_632, mm_449], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf29, (1218, 4864), (4864, 1), 0), primals_620, out=buf30)
            del primals_620
            buf15 = empty_strided_cuda((1218, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1168, view_1170, add_315, silu_23, mul_420, mul_422, view_1171, mm_441], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf13, (1218, 4864), (4864, 1), 0), permute_617, out=buf15)
            del permute_617
            buf14 = empty_strided_cuda((4864, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1168, view_1170, add_315, silu_23, mul_420, mul_422, view_1171, permute_615, mm_440], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf13, (4864, 1218), (1, 4864), 0), mm_427, out=buf14)
            del mm_427
            buf24 = empty_strided_cuda((1218, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1168, view_1170, add_315, silu_23, mul_421, convert_element_type_1940, neg_48, exp, add_317, reciprocal, mul_423, mul_424, sub, mul_425, add_318, mul_426, convert_element_type_1942, mul_427, view_1177, mm_446], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf22, (1218, 4864), (4864, 1), 0), permute_626, out=buf24)
            del permute_626
            buf23 = empty_strided_cuda((4864, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1168, view_1170, add_315, silu_23, mul_421, convert_element_type_1940, neg_48, exp, add_317, reciprocal, mul_423, mul_424, sub, mul_425, add_318, mul_426, convert_element_type_1942, mul_427, view_1177, permute_624, mm_445], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf22, (4864, 1218), (1, 4864), 0), mm_424, out=buf23)
            del mm_424
            buf17 = empty_strided_cuda((32, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_619, mm_442], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf15, (32, 1218), (1, 32), 0), view_1145, out=buf17)
            buf26 = empty_strided_cuda((32, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_628, mm_447], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf24, (32, 1218), (1, 32), 0), view_1145, out=buf26)
            del view_1145
            buf19 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1936], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf17, buf19, 28672, stream=stream0)
            buf28 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1953], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf26, buf28, 28672, stream=stream0)
            buf16 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1930], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf14, buf16, 155648, stream=stream0)
            buf25 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1947], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf23, buf25, 155648, stream=stream0)
            buf18 = empty_strided_cuda((1218, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mm_443], Original ATen: [aten.mm]
            extern_kernels.mm(buf15, permute_621, out=buf18)
            del permute_621
            buf27 = empty_strided_cuda((1218, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mm_448], Original ATen: [aten.mm]
            extern_kernels.mm(buf24, permute_630, out=buf27)
            del permute_630
            buf33 = buf4; del buf4  # reuse
            buf34 = empty_strided_cuda((1218, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1174, view_1176, add_316, view_1180, add_319, view_1182, add_320, mul_428, convert_element_type_1957, hidden_states_236, mul_429, mul_430, sum_2, pow_52, mul_431, mul_432, expand_78, div_1, pow_53, mul_433, mul_434, add_321, convert_element_type_1958, add_322, mul_435, view_1183], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf33, buf18, buf21, buf27, buf30, primals_619, add_307, rsqrt_47, buf34, 1218, 896, stream=stream0)
            del add_307
            del buf18
            del primals_619
            del rsqrt_47
            buf35 = reinterpret_tensor(buf26, (896, 32), (32, 1), 0); del buf26  # reuse
            # Topologically Sorted Source Nodes: [permute_633, mm_450], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf34, (896, 1218), (1, 896), 0), mm_421, out=buf35)
            del mm_421
            buf36 = buf24; del buf24  # reuse
            # Topologically Sorted Source Nodes: [mm_451], Original ATen: [aten.mm]
            extern_kernels.mm(buf34, permute_635, out=buf36)
            del permute_635
            buf38 = buf17; del buf17  # reuse
            # Topologically Sorted Source Nodes: [permute_637, mm_452], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf36, (32, 1218), (1, 32), 0), view_1139, out=buf38)
            del view_1139
            buf37 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1963], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf35, buf37, 28672, stream=stream0)
            buf40 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1969], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf38, buf40, 28672, stream=stream0)
            buf39 = buf34; del buf34  # reuse
            # Topologically Sorted Source Nodes: [mm_453], Original ATen: [aten.mm]
            extern_kernels.mm(buf36, permute_639, out=buf39)
            del permute_639
            buf41 = buf30; del buf30  # reuse
            # Topologically Sorted Source Nodes: [view_1187, result_495, permute_641, mm_454], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf33, (1218, 896), (896, 1), 0), primals_616, out=buf41)
            del primals_616
            buf42 = reinterpret_tensor(buf39, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf39  # reuse
            # Topologically Sorted Source Nodes: [view_1186, view_1188, add_323, view_1189, permute_642, attn_output, _scaled_dot_product_efficient_attention_backward], Original ATen: [aten.view, aten.add, aten.transpose, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf42, buf41, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [view_1186, view_1188, add_323, view_1189, permute_642, attn_output, _scaled_dot_product_efficient_attention_backward], Original ATen: [aten.view, aten.add, aten.transpose, aten.slice, aten.expand, aten._scaled_dot_product_efficient_attention_backward]
            buf43 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf42, add_304, view_1134, view_1135, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_92, getitem_93, getitem_94, getitem_95, 0.0, [True, True, True, False], scale=0.125)
            del add_304
            del getitem_92
            del getitem_93
            del getitem_94
            del getitem_95
            del view_1134
            del view_1135
            buf44 = buf43[0]
            assert_size_stride(buf44, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf44, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf45 = buf43[1]
            assert_size_stride(buf45, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf45, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf47 = empty_strided_cuda((1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1191, sum_4], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf45, buf47, 155904, stream=stream0)
            buf46 = buf43[2]
            assert_size_stride(buf46, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf46, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf43
            buf48 = empty_strided_cuda((1, 1218, 2, 64), (155904, 128, 64, 1), torch.bfloat16)
            buf49 = empty_strided_cuda((1218, 128), (128, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1190, sum_3, squeeze, permute_643, clone_50, view_1192, mul_440, view_1193], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf46, buf48, buf49, 155904, stream=stream0)
            buf50 = empty_strided_cuda((128, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_644, mm_455], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf49, (128, 1218), (1, 128), 0), mm_418, out=buf50)
            del mm_418
            buf51 = buf36; del buf36  # reuse
            # Topologically Sorted Source Nodes: [mm_456], Original ATen: [aten.mm]
            extern_kernels.mm(buf49, permute_646, out=buf51)
            del permute_646
            buf52 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1977], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf50, buf52, 4096, stream=stream0)
            buf53 = buf38; del buf38  # reuse
            # Topologically Sorted Source Nodes: [permute_648, mm_457], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf51, (32, 1218), (1, 32), 0), view_1115, out=buf53)
            buf55 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1983], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf53, buf55, 28672, stream=stream0)
            buf57 = reinterpret_tensor(buf49, (1, 1218, 128), (155904, 128, 1), 0); del buf49  # reuse
            buf64 = empty_strided_cuda((1, 1218, 2, 64), (155904, 128, 64, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1191, sum_4, squeeze_1, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, mul_436, slice_122, slice_123, neg_49, full_default_50, add_324, cos, cos_1, cos_2, cos_3, mul_437, add_325, permute_653, clone_51, view_1199, mul_441], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.cos, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf47, mm_default, buf57, buf64, 1218, 128, stream=stream0)
            buf58 = buf50; del buf50  # reuse
            # Topologically Sorted Source Nodes: [view_1191, sum_4, squeeze_1, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, mul_436, slice_122, slice_123, neg_49, full_default_50, add_324, cos, cos_1, cos_2, cos_3, mul_437, add_325, permute_653, clone_51, view_1199, mul_441, view_1200, permute_654, mm_460], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.cos, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf57, (128, 1218), (1, 128), 0), mm_416, out=buf58)
            del mm_416
            buf59 = buf15; del buf15  # reuse
            # Topologically Sorted Source Nodes: [view_1191, sum_4, squeeze_1, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, mul_436, slice_122, slice_123, neg_49, full_default_50, add_324, cos, cos_1, cos_2, cos_3, mul_437, add_325, permute_653, clone_51, view_1199, mul_441, view_1200, mm_461], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.cos, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf57, (1218, 128), (128, 1), 0), permute_656, out=buf59)
            del permute_656
            buf60 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1991], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf58, buf60, 4096, stream=stream0)
            buf61 = buf53; del buf53  # reuse
            # Topologically Sorted Source Nodes: [permute_658, mm_462], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf59, (32, 1218), (1, 32), 0), view_1115, out=buf61)
            buf63 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_1997], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf61, buf63, 28672, stream=stream0)
            buf56 = reinterpret_tensor(buf46, (1218, 896), (896, 1), 0); del buf46  # reuse
            # Topologically Sorted Source Nodes: [view_1190, sum_3, squeeze, permute_643, clone_50, view_1192, view_1197, result_492, permute_652, mm_459], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf48, (1218, 128), (128, 1), 0), primals_612, out=buf56)
            del primals_612
            buf65 = reinterpret_tensor(buf45, (1218, 896), (896, 1), 0); del buf45  # reuse
            # Topologically Sorted Source Nodes: [view_1191, sum_4, squeeze_1, matmul, freqs, emb, sin, sin_1, sin_2, sin_3, mul_436, slice_122, slice_123, neg_49, full_default_50, add_324, cos, cos_1, cos_2, cos_3, mul_437, add_325, permute_653, clone_51, view_1199, view_1204, result_489, permute_662, mm_464], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.cos, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf64, (1218, 128), (128, 1), 0), primals_608, out=buf65)
            del primals_608
            buf54 = reinterpret_tensor(buf42, (1218, 896), (896, 1), 0); del buf42  # reuse
            # Topologically Sorted Source Nodes: [mm_458], Original ATen: [aten.mm]
            extern_kernels.mm(buf51, permute_650, out=buf54)
            del permute_650
            buf62 = buf41; del buf41  # reuse
            # Topologically Sorted Source Nodes: [mm_463], Original ATen: [aten.mm]
            extern_kernels.mm(buf59, permute_660, out=buf62)
            del permute_660
            buf66 = reinterpret_tensor(buf27, (1, 1218, 896), (1091328, 896, 1), 0); del buf27  # reuse
            buf73 = reinterpret_tensor(buf21, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf21  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, mul_438, slice_124, slice_125, neg_50, full_default_52, add_326, mul_439, add_327, permute_663, clone_52, view_1206, mul_442], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf44, mm_default, buf66, buf73, 1091328, stream=stream0)
            buf67 = reinterpret_tensor(buf61, (896, 32), (32, 1), 0); del buf61  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, mul_438, slice_124, slice_125, neg_50, full_default_52, add_326, mul_439, add_327, permute_663, clone_52, view_1206, mul_442, view_1207, permute_664, mm_465], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf66, (896, 1218), (1, 896), 0), mm_414, out=buf67)
            del mm_414
            buf68 = buf59; del buf59  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, mul_438, slice_124, slice_125, neg_50, full_default_52, add_326, mul_439, add_327, permute_663, clone_52, view_1206, mul_442, view_1207, mm_466], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf66, (1218, 896), (896, 1), 0), permute_666, out=buf68)
            del permute_666
            buf70 = reinterpret_tensor(buf35, (32, 896), (896, 1), 0); del buf35  # reuse
            # Topologically Sorted Source Nodes: [permute_668, mm_467], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf68, (32, 1218), (1, 32), 0), view_1115, out=buf70)
            del view_1115
            buf74 = reinterpret_tensor(buf66, (1218, 896), (896, 1), 0); del buf66  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, mul_438, slice_124, slice_125, neg_50, full_default_52, add_326, mul_439, add_327, permute_663, clone_52, view_1206, view_1211, result_486, permute_672, mm_469], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice, aten.neg, aten.slice_backward, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf73, (1218, 896), (896, 1), 0), primals_604, out=buf74)
            del primals_604
            buf69 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2005], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf67, buf69, 28672, stream=stream0)
            buf72 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2011], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf70, buf72, 28672, stream=stream0)
            buf71 = reinterpret_tensor(buf73, (1218, 896), (896, 1), 0); del buf73  # reuse
            # Topologically Sorted Source Nodes: [mm_468], Original ATen: [aten.mm]
            extern_kernels.mm(buf68, permute_670, out=buf71)
            del permute_670
            buf77 = buf33; del buf33  # reuse
            buf78 = reinterpret_tensor(buf44, (1218, 896), (896, 1), 0); del buf44  # reuse
            # Topologically Sorted Source Nodes: [view_1196, view_1198, add_328, view_1203, add_329, view_1205, add_330, view_1210, add_331, view_1212, add_332, mul_443, convert_element_type_2015, hidden_states_230, mul_444, mul_445, sum_5, pow_54, mul_446, mul_447, expand_79, div_2, pow_55, mul_448, mul_449, add_333, convert_element_type_2016, add_334, mul_450, view_1213], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf77, buf54, buf56, buf62, buf65, buf71, buf74, primals_603, add_299, rsqrt_46, buf78, 1218, 896, stream=stream0)
            del add_299
            del buf54
            del buf56
            del primals_603
            del rsqrt_46
            buf79 = reinterpret_tensor(buf70, (896, 32), (32, 1), 0); del buf70  # reuse
            # Topologically Sorted Source Nodes: [permute_673, mm_470], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf78, (896, 1218), (1, 896), 0), mm_412, out=buf79)
            del mm_412
            buf80 = buf68; del buf68  # reuse
            # Topologically Sorted Source Nodes: [mm_471], Original ATen: [aten.mm]
            extern_kernels.mm(buf78, permute_675, out=buf80)
            del permute_675
            buf82 = reinterpret_tensor(buf23, (32, 4864), (4864, 1), 0); del buf23  # reuse
            # Topologically Sorted Source Nodes: [permute_677, mm_472], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf80, (32, 1218), (1, 32), 0), view_1109, out=buf82)
            del view_1109
            buf81 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2021], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf79, buf81, 28672, stream=stream0)
            buf84 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2027], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf82, buf84, 155648, stream=stream0)
            buf83 = reinterpret_tensor(buf22, (1218, 4864), (4864, 1), 0); del buf22  # reuse
            # Topologically Sorted Source Nodes: [mm_473], Original ATen: [aten.mm]
            extern_kernels.mm(buf80, permute_679, out=buf83)
            del permute_679
            buf85 = reinterpret_tensor(buf13, (1218, 4864), (4864, 1), 0); del buf13  # reuse
            # Topologically Sorted Source Nodes: [view_1217, result_483, permute_681, mm_474], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf77, (1218, 896), (896, 1), 0), primals_600, out=buf85)
            del primals_600
            buf86 = buf29; del buf29  # reuse
            buf93 = buf20; del buf20  # reuse
            buf95 = reinterpret_tensor(buf12, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf12  # reuse
            buf102 = reinterpret_tensor(buf10, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf10  # reuse
            # Topologically Sorted Source Nodes: [view_1216, view_1218, add_335, silu_22, mul_451, mul_452, mul_453, convert_element_type_2045, neg_51, exp_1, add_337, reciprocal_1, mul_454, mul_455, sub_1, mul_456, add_338, mul_457, convert_element_type_2047, mul_458], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf83, buf85, add_296, add_297, buf86, buf93, buf95, buf102, 5924352, stream=stream0)
            del add_296
            del add_297
            buf94 = buf78; del buf78  # reuse
            # Topologically Sorted Source Nodes: [view_1216, view_1218, add_335, silu_22, mul_451, view_1223, result_480, permute_690, mm_479], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf93, (1218, 4864), (4864, 1), 0), primals_597, out=buf94)
            del primals_597
            buf103 = buf74; del buf74  # reuse
            # Topologically Sorted Source Nodes: [view_1216, view_1218, add_335, silu_22, mul_452, convert_element_type_2045, neg_51, exp_1, add_337, reciprocal_1, mul_454, mul_455, sub_1, mul_456, add_338, mul_457, convert_element_type_2047, view_1229, result_477, permute_699, mm_484], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf102, (1218, 4864), (4864, 1), 0), primals_594, out=buf103)
            del primals_594
            buf88 = buf80; del buf80  # reuse
            # Topologically Sorted Source Nodes: [view_1216, view_1218, add_335, silu_22, mul_451, mul_453, view_1219, mm_476], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf86, (1218, 4864), (4864, 1), 0), permute_684, out=buf88)
            del permute_684
            buf87 = reinterpret_tensor(buf82, (4864, 32), (32, 1), 0); del buf82  # reuse
            # Topologically Sorted Source Nodes: [view_1216, view_1218, add_335, silu_22, mul_451, mul_453, view_1219, permute_682, mm_475], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf86, (4864, 1218), (1, 4864), 0), mm_409, out=buf87)
            del mm_409
            buf97 = buf51; del buf51  # reuse
            # Topologically Sorted Source Nodes: [view_1216, view_1218, add_335, silu_22, mul_452, convert_element_type_2045, neg_51, exp_1, add_337, reciprocal_1, mul_454, mul_455, sub_1, mul_456, add_338, mul_457, convert_element_type_2047, mul_458, view_1225, mm_481], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf95, (1218, 4864), (4864, 1), 0), permute_693, out=buf97)
            del permute_693
            buf96 = buf14; del buf14  # reuse
            # Topologically Sorted Source Nodes: [view_1216, view_1218, add_335, silu_22, mul_452, convert_element_type_2045, neg_51, exp_1, add_337, reciprocal_1, mul_454, mul_455, sub_1, mul_456, add_338, mul_457, convert_element_type_2047, mul_458, view_1225, permute_691, mm_480], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf95, (4864, 1218), (1, 4864), 0), mm_406, out=buf96)
            del mm_406
            buf90 = reinterpret_tensor(buf79, (32, 896), (896, 1), 0); del buf79  # reuse
            # Topologically Sorted Source Nodes: [permute_686, mm_477], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf88, (32, 1218), (1, 32), 0), view_1097, out=buf90)
            buf99 = reinterpret_tensor(buf67, (32, 896), (896, 1), 0); del buf67  # reuse
            # Topologically Sorted Source Nodes: [permute_695, mm_482], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf97, (32, 1218), (1, 32), 0), view_1097, out=buf99)
            del view_1097
            buf92 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2041], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf90, buf92, 28672, stream=stream0)
            buf101 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2058], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf99, buf101, 28672, stream=stream0)
            buf89 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2035], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf87, buf89, 155648, stream=stream0)
            buf98 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2052], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf96, buf98, 155648, stream=stream0)
            buf91 = buf71; del buf71  # reuse
            # Topologically Sorted Source Nodes: [mm_478], Original ATen: [aten.mm]
            extern_kernels.mm(buf88, permute_688, out=buf91)
            del permute_688
            buf100 = buf65; del buf65  # reuse
            # Topologically Sorted Source Nodes: [mm_483], Original ATen: [aten.mm]
            extern_kernels.mm(buf97, permute_697, out=buf100)
            del permute_697
            buf106 = buf77; del buf77  # reuse
            buf107 = buf62; del buf62  # reuse
            # Topologically Sorted Source Nodes: [view_1222, view_1224, add_336, view_1228, add_339, view_1230, add_340, mul_459, convert_element_type_2062, hidden_states_226, mul_460, mul_461, sum_6, pow_56, mul_462, mul_463, expand_80, div_3, pow_57, mul_464, mul_465, add_341, convert_element_type_2063, add_342, mul_466, view_1231], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf106, buf91, buf94, buf100, buf103, primals_593, add_294, rsqrt_45, buf107, 1218, 896, stream=stream0)
            del add_294
            del buf100
            del primals_593
            del rsqrt_45
            buf108 = reinterpret_tensor(buf99, (896, 32), (32, 1), 0); del buf99  # reuse
            # Topologically Sorted Source Nodes: [permute_700, mm_485], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf107, (896, 1218), (1, 896), 0), mm_403, out=buf108)
            del mm_403
            buf109 = buf97; del buf97  # reuse
            # Topologically Sorted Source Nodes: [mm_486], Original ATen: [aten.mm]
            extern_kernels.mm(buf107, permute_702, out=buf109)
            del permute_702
            buf111 = buf90; del buf90  # reuse
            # Topologically Sorted Source Nodes: [permute_704, mm_487], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf109, (32, 1218), (1, 32), 0), view_1091, out=buf111)
            del view_1091
            buf110 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2068], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf108, buf110, 28672, stream=stream0)
            buf113 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2074], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf111, buf113, 28672, stream=stream0)
            buf112 = buf107; del buf107  # reuse
            # Topologically Sorted Source Nodes: [mm_488], Original ATen: [aten.mm]
            extern_kernels.mm(buf109, permute_706, out=buf112)
            del permute_706
            buf114 = buf94; del buf94  # reuse
            # Topologically Sorted Source Nodes: [view_1235, result_474, permute_708, mm_489], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf106, (1218, 896), (896, 1), 0), primals_590, out=buf114)
            del primals_590
            buf115 = reinterpret_tensor(buf112, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf112  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1234, view_1236, add_343, view_1237, permute_709, _scaled_dot_product_efficient_attention_backward_1], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf115, buf114, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1234, view_1236, add_343, view_1237, permute_709, _scaled_dot_product_efficient_attention_backward_1], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf116 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf115, add_291, view_1086, view_1087, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_88, getitem_89, getitem_90, getitem_91, 0.0, [True, True, True, False], scale=0.125)
            del add_291
            del getitem_88
            del getitem_89
            del getitem_90
            del getitem_91
            del view_1086
            del view_1087
            buf117 = buf116[0]
            assert_size_stride(buf117, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf117, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf118 = buf116[1]
            assert_size_stride(buf118, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf118, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf120 = reinterpret_tensor(buf64, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf64  # reuse
            # Topologically Sorted Source Nodes: [view_1239, sum_8], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf118, buf120, 155904, stream=stream0)
            buf119 = buf116[2]
            assert_size_stride(buf119, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf119, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf116
            buf121 = buf48; del buf48  # reuse
            buf122 = reinterpret_tensor(buf57, (1218, 128), (128, 1), 0); del buf57  # reuse
            # Topologically Sorted Source Nodes: [view_1238, sum_7, squeeze_2, permute_710, clone_53, view_1240, mul_471, view_1241], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf119, buf121, buf122, 155904, stream=stream0)
            buf123 = buf58; del buf58  # reuse
            # Topologically Sorted Source Nodes: [permute_711, mm_490], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf122, (128, 1218), (1, 128), 0), mm_400, out=buf123)
            del mm_400
            buf124 = buf109; del buf109  # reuse
            # Topologically Sorted Source Nodes: [mm_491], Original ATen: [aten.mm]
            extern_kernels.mm(buf122, permute_713, out=buf124)
            del permute_713
            buf125 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2082], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf123, buf125, 4096, stream=stream0)
            buf126 = buf111; del buf111  # reuse
            # Topologically Sorted Source Nodes: [permute_715, mm_492], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf124, (32, 1218), (1, 32), 0), view_1067, out=buf126)
            buf128 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2088], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf126, buf128, 28672, stream=stream0)
            buf130 = reinterpret_tensor(buf122, (1, 1218, 128), (155904, 128, 1), 0); del buf122  # reuse
            buf137 = reinterpret_tensor(buf47, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf47  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1239, sum_8, squeeze_3, mul_467, slice_126, slice_127, neg_52, add_344, mul_468, add_345, permute_720, clone_54, view_1247, mul_472], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf120, mm_default, buf130, buf137, 1218, 128, stream=stream0)
            buf131 = buf123; del buf123  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1239, sum_8, squeeze_3, mul_467, slice_126, slice_127, neg_52, add_344, mul_468, add_345, permute_720, clone_54, view_1247, mul_472, view_1248, permute_721, mm_495], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf130, (128, 1218), (1, 128), 0), mm_398, out=buf131)
            del mm_398
            buf132 = buf88; del buf88  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1239, sum_8, squeeze_3, mul_467, slice_126, slice_127, neg_52, add_344, mul_468, add_345, permute_720, clone_54, view_1247, mul_472, view_1248, mm_496], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf130, (1218, 128), (128, 1), 0), permute_723, out=buf132)
            del permute_723
            buf133 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2096], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf131, buf133, 4096, stream=stream0)
            buf134 = buf126; del buf126  # reuse
            # Topologically Sorted Source Nodes: [permute_725, mm_497], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf132, (32, 1218), (1, 32), 0), view_1067, out=buf134)
            buf136 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2102], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf134, buf136, 28672, stream=stream0)
            buf129 = reinterpret_tensor(buf119, (1218, 896), (896, 1), 0); del buf119  # reuse
            # Topologically Sorted Source Nodes: [view_1238, sum_7, squeeze_2, permute_710, clone_53, view_1240, view_1245, result_471, permute_719, mm_494], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf121, (1218, 128), (128, 1), 0), primals_586, out=buf129)
            del primals_586
            buf138 = reinterpret_tensor(buf118, (1218, 896), (896, 1), 0); del buf118  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1239, sum_8, squeeze_3, mul_467, slice_126, slice_127, neg_52, add_344, mul_468, add_345, permute_720, clone_54, view_1247, view_1252, result_468, permute_729, mm_499], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf137, (1218, 128), (128, 1), 0), primals_582, out=buf138)
            del primals_582
            buf127 = reinterpret_tensor(buf115, (1218, 896), (896, 1), 0); del buf115  # reuse
            # Topologically Sorted Source Nodes: [mm_493], Original ATen: [aten.mm]
            extern_kernels.mm(buf124, permute_717, out=buf127)
            del permute_717
            buf135 = buf114; del buf114  # reuse
            # Topologically Sorted Source Nodes: [mm_498], Original ATen: [aten.mm]
            extern_kernels.mm(buf132, permute_727, out=buf135)
            del permute_727
            buf139 = reinterpret_tensor(buf91, (1, 1218, 896), (1091328, 896, 1), 0); del buf91  # reuse
            buf146 = reinterpret_tensor(buf103, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf103  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_469, slice_128, slice_129, neg_53, add_346, mul_470, add_347, permute_730, clone_55, view_1254, mul_473], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf117, mm_default, buf139, buf146, 1091328, stream=stream0)
            buf140 = reinterpret_tensor(buf134, (896, 32), (32, 1), 0); del buf134  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_469, slice_128, slice_129, neg_53, add_346, mul_470, add_347, permute_730, clone_55, view_1254, mul_473, view_1255, permute_731, mm_500], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf139, (896, 1218), (1, 896), 0), mm_396, out=buf140)
            del mm_396
            buf141 = buf132; del buf132  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_469, slice_128, slice_129, neg_53, add_346, mul_470, add_347, permute_730, clone_55, view_1254, mul_473, view_1255, mm_501], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf139, (1218, 896), (896, 1), 0), permute_733, out=buf141)
            del permute_733
            buf143 = reinterpret_tensor(buf108, (32, 896), (896, 1), 0); del buf108  # reuse
            # Topologically Sorted Source Nodes: [permute_735, mm_502], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf141, (32, 1218), (1, 32), 0), view_1067, out=buf143)
            del view_1067
            buf147 = reinterpret_tensor(buf139, (1218, 896), (896, 1), 0); del buf139  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_469, slice_128, slice_129, neg_53, add_346, mul_470, add_347, permute_730, clone_55, view_1254, view_1259, result_465, permute_739, mm_504], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf146, (1218, 896), (896, 1), 0), primals_578, out=buf147)
            del primals_578
            buf142 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2110], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf140, buf142, 28672, stream=stream0)
            buf145 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2116], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf143, buf145, 28672, stream=stream0)
            buf144 = reinterpret_tensor(buf146, (1218, 896), (896, 1), 0); del buf146  # reuse
            # Topologically Sorted Source Nodes: [mm_503], Original ATen: [aten.mm]
            extern_kernels.mm(buf141, permute_737, out=buf144)
            del permute_737
            buf150 = buf106; del buf106  # reuse
            buf151 = reinterpret_tensor(buf117, (1218, 896), (896, 1), 0); del buf117  # reuse
            # Topologically Sorted Source Nodes: [view_1244, view_1246, add_348, view_1251, add_349, view_1253, add_350, view_1258, add_351, view_1260, add_352, mul_474, convert_element_type_2120, hidden_states_220, mul_475, mul_476, sum_9, pow_58, mul_477, mul_478, expand_81, div_4, pow_59, mul_479, mul_480, add_353, convert_element_type_2121, add_354, mul_481, view_1261], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf150, buf127, buf129, buf135, buf138, buf144, buf147, primals_577, add_286, rsqrt_44, buf151, 1218, 896, stream=stream0)
            del add_286
            del buf127
            del buf129
            del primals_577
            del rsqrt_44
            buf152 = reinterpret_tensor(buf143, (896, 32), (32, 1), 0); del buf143  # reuse
            # Topologically Sorted Source Nodes: [permute_740, mm_505], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf151, (896, 1218), (1, 896), 0), mm_394, out=buf152)
            del mm_394
            buf153 = buf141; del buf141  # reuse
            # Topologically Sorted Source Nodes: [mm_506], Original ATen: [aten.mm]
            extern_kernels.mm(buf151, permute_742, out=buf153)
            del permute_742
            buf155 = reinterpret_tensor(buf96, (32, 4864), (4864, 1), 0); del buf96  # reuse
            # Topologically Sorted Source Nodes: [permute_744, mm_507], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf153, (32, 1218), (1, 32), 0), view_1061, out=buf155)
            del view_1061
            buf154 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2126], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf152, buf154, 28672, stream=stream0)
            buf157 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2132], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf155, buf157, 155648, stream=stream0)
            buf156 = reinterpret_tensor(buf95, (1218, 4864), (4864, 1), 0); del buf95  # reuse
            # Topologically Sorted Source Nodes: [mm_508], Original ATen: [aten.mm]
            extern_kernels.mm(buf153, permute_746, out=buf156)
            del permute_746
            buf158 = reinterpret_tensor(buf86, (1218, 4864), (4864, 1), 0); del buf86  # reuse
            # Topologically Sorted Source Nodes: [view_1265, result_462, permute_748, mm_509], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf150, (1218, 896), (896, 1), 0), primals_574, out=buf158)
            del primals_574
            buf159 = buf102; del buf102  # reuse
            buf166 = buf93; del buf93  # reuse
            buf168 = reinterpret_tensor(buf85, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf85  # reuse
            buf175 = reinterpret_tensor(buf83, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf83  # reuse
            # Topologically Sorted Source Nodes: [view_1264, view_1266, add_355, silu_21, mul_482, mul_483, mul_484, convert_element_type_2150, neg_54, exp_2, add_357, reciprocal_2, mul_485, mul_486, sub_2, mul_487, add_358, mul_488, convert_element_type_2152, mul_489], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf156, buf158, add_283, add_284, buf159, buf166, buf168, buf175, 5924352, stream=stream0)
            del add_283
            del add_284
            buf167 = buf151; del buf151  # reuse
            # Topologically Sorted Source Nodes: [view_1264, view_1266, add_355, silu_21, mul_482, view_1271, result_459, permute_757, mm_514], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf166, (1218, 4864), (4864, 1), 0), primals_571, out=buf167)
            del primals_571
            buf176 = buf147; del buf147  # reuse
            # Topologically Sorted Source Nodes: [view_1264, view_1266, add_355, silu_21, mul_483, convert_element_type_2150, neg_54, exp_2, add_357, reciprocal_2, mul_485, mul_486, sub_2, mul_487, add_358, mul_488, convert_element_type_2152, view_1277, result_456, permute_766, mm_519], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf175, (1218, 4864), (4864, 1), 0), primals_568, out=buf176)
            del primals_568
            buf161 = buf153; del buf153  # reuse
            # Topologically Sorted Source Nodes: [view_1264, view_1266, add_355, silu_21, mul_482, mul_484, view_1267, mm_511], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf159, (1218, 4864), (4864, 1), 0), permute_751, out=buf161)
            del permute_751
            buf160 = reinterpret_tensor(buf155, (4864, 32), (32, 1), 0); del buf155  # reuse
            # Topologically Sorted Source Nodes: [view_1264, view_1266, add_355, silu_21, mul_482, mul_484, view_1267, permute_749, mm_510], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf159, (4864, 1218), (1, 4864), 0), mm_391, out=buf160)
            del mm_391
            buf170 = buf124; del buf124  # reuse
            # Topologically Sorted Source Nodes: [view_1264, view_1266, add_355, silu_21, mul_483, convert_element_type_2150, neg_54, exp_2, add_357, reciprocal_2, mul_485, mul_486, sub_2, mul_487, add_358, mul_488, convert_element_type_2152, mul_489, view_1273, mm_516], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf168, (1218, 4864), (4864, 1), 0), permute_760, out=buf170)
            del permute_760
            buf169 = buf87; del buf87  # reuse
            # Topologically Sorted Source Nodes: [view_1264, view_1266, add_355, silu_21, mul_483, convert_element_type_2150, neg_54, exp_2, add_357, reciprocal_2, mul_485, mul_486, sub_2, mul_487, add_358, mul_488, convert_element_type_2152, mul_489, view_1273, permute_758, mm_515], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf168, (4864, 1218), (1, 4864), 0), mm_388, out=buf169)
            del mm_388
            buf163 = reinterpret_tensor(buf152, (32, 896), (896, 1), 0); del buf152  # reuse
            # Topologically Sorted Source Nodes: [permute_753, mm_512], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf161, (32, 1218), (1, 32), 0), view_1049, out=buf163)
            buf172 = reinterpret_tensor(buf140, (32, 896), (896, 1), 0); del buf140  # reuse
            # Topologically Sorted Source Nodes: [permute_762, mm_517], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf170, (32, 1218), (1, 32), 0), view_1049, out=buf172)
            del view_1049
            buf165 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2146], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf163, buf165, 28672, stream=stream0)
            buf174 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2163], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf172, buf174, 28672, stream=stream0)
            buf162 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2140], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf160, buf162, 155648, stream=stream0)
            buf171 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2157], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf169, buf171, 155648, stream=stream0)
            buf164 = buf144; del buf144  # reuse
            # Topologically Sorted Source Nodes: [mm_513], Original ATen: [aten.mm]
            extern_kernels.mm(buf161, permute_755, out=buf164)
            del permute_755
            buf173 = buf138; del buf138  # reuse
            # Topologically Sorted Source Nodes: [mm_518], Original ATen: [aten.mm]
            extern_kernels.mm(buf170, permute_764, out=buf173)
            del permute_764
            buf179 = buf150; del buf150  # reuse
            buf180 = buf135; del buf135  # reuse
            # Topologically Sorted Source Nodes: [view_1270, view_1272, add_356, view_1276, add_359, view_1278, add_360, mul_490, convert_element_type_2167, hidden_states_216, mul_491, mul_492, sum_10, pow_60, mul_493, mul_494, expand_82, div_5, pow_61, mul_495, mul_496, add_361, convert_element_type_2168, add_362, mul_497, view_1279], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf179, buf164, buf167, buf173, buf176, primals_567, add_281, rsqrt_43, buf180, 1218, 896, stream=stream0)
            del add_281
            del buf164
            del primals_567
            del rsqrt_43
            buf181 = reinterpret_tensor(buf172, (896, 32), (32, 1), 0); del buf172  # reuse
            # Topologically Sorted Source Nodes: [permute_767, mm_520], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf180, (896, 1218), (1, 896), 0), mm_385, out=buf181)
            del mm_385
            buf182 = buf170; del buf170  # reuse
            # Topologically Sorted Source Nodes: [mm_521], Original ATen: [aten.mm]
            extern_kernels.mm(buf180, permute_769, out=buf182)
            del permute_769
            buf184 = buf163; del buf163  # reuse
            # Topologically Sorted Source Nodes: [permute_771, mm_522], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf182, (32, 1218), (1, 32), 0), view_1043, out=buf184)
            del view_1043
            buf183 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2173], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf181, buf183, 28672, stream=stream0)
            buf186 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2179], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf184, buf186, 28672, stream=stream0)
            buf185 = buf180; del buf180  # reuse
            # Topologically Sorted Source Nodes: [mm_523], Original ATen: [aten.mm]
            extern_kernels.mm(buf182, permute_773, out=buf185)
            del permute_773
            buf187 = buf176; del buf176  # reuse
            # Topologically Sorted Source Nodes: [view_1283, result_453, permute_775, mm_524], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf179, (1218, 896), (896, 1), 0), primals_564, out=buf187)
            del primals_564
            buf188 = reinterpret_tensor(buf185, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf185  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1282, view_1284, add_363, view_1285, permute_776, _scaled_dot_product_efficient_attention_backward_2], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf188, buf187, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1282, view_1284, add_363, view_1285, permute_776, _scaled_dot_product_efficient_attention_backward_2], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf189 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf188, add_278, view_1038, view_1039, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_84, getitem_85, getitem_86, getitem_87, 0.0, [True, True, True, False], scale=0.125)
            del add_278
            del getitem_84
            del getitem_85
            del getitem_86
            del getitem_87
            del view_1038
            del view_1039
            buf190 = buf189[0]
            assert_size_stride(buf190, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf190, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf191 = buf189[1]
            assert_size_stride(buf191, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf191, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf193 = reinterpret_tensor(buf137, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf137  # reuse
            # Topologically Sorted Source Nodes: [view_1287, sum_12], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf191, buf193, 155904, stream=stream0)
            buf192 = buf189[2]
            assert_size_stride(buf192, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf192, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf189
            buf194 = buf121; del buf121  # reuse
            buf195 = reinterpret_tensor(buf130, (1218, 128), (128, 1), 0); del buf130  # reuse
            # Topologically Sorted Source Nodes: [view_1286, sum_11, squeeze_4, permute_777, clone_56, view_1288, mul_502, view_1289], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf192, buf194, buf195, 155904, stream=stream0)
            buf196 = buf131; del buf131  # reuse
            # Topologically Sorted Source Nodes: [permute_778, mm_525], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf195, (128, 1218), (1, 128), 0), mm_382, out=buf196)
            del mm_382
            buf197 = buf182; del buf182  # reuse
            # Topologically Sorted Source Nodes: [mm_526], Original ATen: [aten.mm]
            extern_kernels.mm(buf195, permute_780, out=buf197)
            del permute_780
            buf198 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2187], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf196, buf198, 4096, stream=stream0)
            buf199 = buf184; del buf184  # reuse
            # Topologically Sorted Source Nodes: [permute_782, mm_527], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf197, (32, 1218), (1, 32), 0), view_1019, out=buf199)
            buf201 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2193], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf199, buf201, 28672, stream=stream0)
            buf203 = reinterpret_tensor(buf195, (1, 1218, 128), (155904, 128, 1), 0); del buf195  # reuse
            buf210 = reinterpret_tensor(buf120, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf120  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1287, sum_12, squeeze_5, mul_498, slice_130, slice_131, neg_55, add_364, mul_499, add_365, permute_787, clone_57, view_1295, mul_503], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf193, mm_default, buf203, buf210, 1218, 128, stream=stream0)
            buf204 = buf196; del buf196  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1287, sum_12, squeeze_5, mul_498, slice_130, slice_131, neg_55, add_364, mul_499, add_365, permute_787, clone_57, view_1295, mul_503, view_1296, permute_788, mm_530], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf203, (128, 1218), (1, 128), 0), mm_380, out=buf204)
            del mm_380
            buf205 = buf161; del buf161  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1287, sum_12, squeeze_5, mul_498, slice_130, slice_131, neg_55, add_364, mul_499, add_365, permute_787, clone_57, view_1295, mul_503, view_1296, mm_531], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf203, (1218, 128), (128, 1), 0), permute_790, out=buf205)
            del permute_790
            buf206 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2201], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf204, buf206, 4096, stream=stream0)
            buf207 = buf199; del buf199  # reuse
            # Topologically Sorted Source Nodes: [permute_792, mm_532], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf205, (32, 1218), (1, 32), 0), view_1019, out=buf207)
            buf209 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2207], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf207, buf209, 28672, stream=stream0)
            buf202 = reinterpret_tensor(buf192, (1218, 896), (896, 1), 0); del buf192  # reuse
            # Topologically Sorted Source Nodes: [view_1286, sum_11, squeeze_4, permute_777, clone_56, view_1288, view_1293, result_450, permute_786, mm_529], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf194, (1218, 128), (128, 1), 0), primals_560, out=buf202)
            del primals_560
            buf211 = reinterpret_tensor(buf191, (1218, 896), (896, 1), 0); del buf191  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1287, sum_12, squeeze_5, mul_498, slice_130, slice_131, neg_55, add_364, mul_499, add_365, permute_787, clone_57, view_1295, view_1300, result_447, permute_796, mm_534], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf210, (1218, 128), (128, 1), 0), primals_556, out=buf211)
            del primals_556
            buf200 = reinterpret_tensor(buf188, (1218, 896), (896, 1), 0); del buf188  # reuse
            # Topologically Sorted Source Nodes: [mm_528], Original ATen: [aten.mm]
            extern_kernels.mm(buf197, permute_784, out=buf200)
            del permute_784
            buf208 = buf187; del buf187  # reuse
            # Topologically Sorted Source Nodes: [mm_533], Original ATen: [aten.mm]
            extern_kernels.mm(buf205, permute_794, out=buf208)
            del permute_794
            buf212 = reinterpret_tensor(buf173, (1, 1218, 896), (1091328, 896, 1), 0); del buf173  # reuse
            buf219 = reinterpret_tensor(buf167, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf167  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_500, slice_132, slice_133, neg_56, add_366, mul_501, add_367, permute_797, clone_58, view_1302, mul_504], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf190, mm_default, buf212, buf219, 1091328, stream=stream0)
            buf213 = reinterpret_tensor(buf207, (896, 32), (32, 1), 0); del buf207  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_500, slice_132, slice_133, neg_56, add_366, mul_501, add_367, permute_797, clone_58, view_1302, mul_504, view_1303, permute_798, mm_535], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf212, (896, 1218), (1, 896), 0), mm_378, out=buf213)
            del mm_378
            buf214 = buf205; del buf205  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_500, slice_132, slice_133, neg_56, add_366, mul_501, add_367, permute_797, clone_58, view_1302, mul_504, view_1303, mm_536], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf212, (1218, 896), (896, 1), 0), permute_800, out=buf214)
            del permute_800
            buf216 = reinterpret_tensor(buf181, (32, 896), (896, 1), 0); del buf181  # reuse
            # Topologically Sorted Source Nodes: [permute_802, mm_537], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf214, (32, 1218), (1, 32), 0), view_1019, out=buf216)
            del view_1019
            buf220 = reinterpret_tensor(buf212, (1218, 896), (896, 1), 0); del buf212  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_500, slice_132, slice_133, neg_56, add_366, mul_501, add_367, permute_797, clone_58, view_1302, view_1307, result_444, permute_806, mm_539], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf219, (1218, 896), (896, 1), 0), primals_552, out=buf220)
            del primals_552
            buf215 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2215], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf213, buf215, 28672, stream=stream0)
            buf218 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2221], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf216, buf218, 28672, stream=stream0)
            buf217 = reinterpret_tensor(buf219, (1218, 896), (896, 1), 0); del buf219  # reuse
            # Topologically Sorted Source Nodes: [mm_538], Original ATen: [aten.mm]
            extern_kernels.mm(buf214, permute_804, out=buf217)
            del permute_804
            buf223 = buf179; del buf179  # reuse
            buf224 = reinterpret_tensor(buf190, (1218, 896), (896, 1), 0); del buf190  # reuse
            # Topologically Sorted Source Nodes: [view_1292, view_1294, add_368, view_1299, add_369, view_1301, add_370, view_1306, add_371, view_1308, add_372, mul_505, convert_element_type_2225, hidden_states_210, mul_506, mul_507, sum_13, pow_62, mul_508, mul_509, expand_83, div_6, pow_63, mul_510, mul_511, add_373, convert_element_type_2226, add_374, mul_512, view_1309], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf223, buf200, buf202, buf208, buf211, buf217, buf220, primals_551, add_273, rsqrt_42, buf224, 1218, 896, stream=stream0)
            del add_273
            del buf200
            del buf202
            del primals_551
            del rsqrt_42
            buf225 = reinterpret_tensor(buf216, (896, 32), (32, 1), 0); del buf216  # reuse
            # Topologically Sorted Source Nodes: [permute_807, mm_540], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf224, (896, 1218), (1, 896), 0), mm_376, out=buf225)
            del mm_376
            buf226 = buf214; del buf214  # reuse
            # Topologically Sorted Source Nodes: [mm_541], Original ATen: [aten.mm]
            extern_kernels.mm(buf224, permute_809, out=buf226)
            del permute_809
            buf228 = reinterpret_tensor(buf169, (32, 4864), (4864, 1), 0); del buf169  # reuse
            # Topologically Sorted Source Nodes: [permute_811, mm_542], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf226, (32, 1218), (1, 32), 0), view_1013, out=buf228)
            del view_1013
            buf227 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2231], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf225, buf227, 28672, stream=stream0)
            buf230 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2237], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf228, buf230, 155648, stream=stream0)
            buf229 = reinterpret_tensor(buf168, (1218, 4864), (4864, 1), 0); del buf168  # reuse
            # Topologically Sorted Source Nodes: [mm_543], Original ATen: [aten.mm]
            extern_kernels.mm(buf226, permute_813, out=buf229)
            del permute_813
            buf231 = reinterpret_tensor(buf159, (1218, 4864), (4864, 1), 0); del buf159  # reuse
            # Topologically Sorted Source Nodes: [view_1313, result_441, permute_815, mm_544], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf223, (1218, 896), (896, 1), 0), primals_548, out=buf231)
            del primals_548
            buf232 = buf175; del buf175  # reuse
            buf239 = buf166; del buf166  # reuse
            buf241 = reinterpret_tensor(buf158, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf158  # reuse
            buf248 = reinterpret_tensor(buf156, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf156  # reuse
            # Topologically Sorted Source Nodes: [view_1312, view_1314, add_375, silu_20, mul_513, mul_514, mul_515, convert_element_type_2255, neg_57, exp_3, add_377, reciprocal_3, mul_516, mul_517, sub_3, mul_518, add_378, mul_519, convert_element_type_2257, mul_520], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf229, buf231, add_270, add_271, buf232, buf239, buf241, buf248, 5924352, stream=stream0)
            del add_270
            del add_271
            del buf229
            del buf231
            buf240 = buf224; del buf224  # reuse
            # Topologically Sorted Source Nodes: [view_1312, view_1314, add_375, silu_20, mul_513, view_1319, result_438, permute_824, mm_549], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf239, (1218, 4864), (4864, 1), 0), primals_545, out=buf240)
            del buf239
            del primals_545
            buf249 = buf220; del buf220  # reuse
            # Topologically Sorted Source Nodes: [view_1312, view_1314, add_375, silu_20, mul_514, convert_element_type_2255, neg_57, exp_3, add_377, reciprocal_3, mul_516, mul_517, sub_3, mul_518, add_378, mul_519, convert_element_type_2257, view_1325, result_435, permute_833, mm_554], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf248, (1218, 4864), (4864, 1), 0), primals_542, out=buf249)
            del buf248
            del primals_542
            buf234 = buf226; del buf226  # reuse
            # Topologically Sorted Source Nodes: [view_1312, view_1314, add_375, silu_20, mul_513, mul_515, view_1315, mm_546], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf232, (1218, 4864), (4864, 1), 0), permute_818, out=buf234)
            del permute_818
            buf233 = reinterpret_tensor(buf228, (4864, 32), (32, 1), 0); del buf228  # reuse
            # Topologically Sorted Source Nodes: [view_1312, view_1314, add_375, silu_20, mul_513, mul_515, view_1315, permute_816, mm_545], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf232, (4864, 1218), (1, 4864), 0), mm_373, out=buf233)
            del buf232
            del mm_373
            buf243 = buf197; del buf197  # reuse
            # Topologically Sorted Source Nodes: [view_1312, view_1314, add_375, silu_20, mul_514, convert_element_type_2255, neg_57, exp_3, add_377, reciprocal_3, mul_516, mul_517, sub_3, mul_518, add_378, mul_519, convert_element_type_2257, mul_520, view_1321, mm_551], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf241, (1218, 4864), (4864, 1), 0), permute_827, out=buf243)
            del permute_827
            buf242 = buf160; del buf160  # reuse
            # Topologically Sorted Source Nodes: [view_1312, view_1314, add_375, silu_20, mul_514, convert_element_type_2255, neg_57, exp_3, add_377, reciprocal_3, mul_516, mul_517, sub_3, mul_518, add_378, mul_519, convert_element_type_2257, mul_520, view_1321, permute_825, mm_550], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf241, (4864, 1218), (1, 4864), 0), mm_370, out=buf242)
            del buf241
            del mm_370
            buf236 = reinterpret_tensor(buf225, (32, 896), (896, 1), 0); del buf225  # reuse
            # Topologically Sorted Source Nodes: [permute_820, mm_547], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf234, (32, 1218), (1, 32), 0), view_1001, out=buf236)
            buf245 = reinterpret_tensor(buf213, (32, 896), (896, 1), 0); del buf213  # reuse
            # Topologically Sorted Source Nodes: [permute_829, mm_552], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf243, (32, 1218), (1, 32), 0), view_1001, out=buf245)
            del view_1001
            buf238 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2251], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf236, buf238, 28672, stream=stream0)
            buf247 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2268], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf245, buf247, 28672, stream=stream0)
            buf235 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2245], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf233, buf235, 155648, stream=stream0)
            buf244 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2262], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf242, buf244, 155648, stream=stream0)
            buf237 = buf217; del buf217  # reuse
            # Topologically Sorted Source Nodes: [mm_548], Original ATen: [aten.mm]
            extern_kernels.mm(buf234, permute_822, out=buf237)
            del permute_822
            buf246 = buf211; del buf211  # reuse
            # Topologically Sorted Source Nodes: [mm_553], Original ATen: [aten.mm]
            extern_kernels.mm(buf243, permute_831, out=buf246)
            del permute_831
            buf252 = buf223; del buf223  # reuse
            buf253 = buf208; del buf208  # reuse
            # Topologically Sorted Source Nodes: [view_1318, view_1320, add_376, view_1324, add_379, view_1326, add_380, mul_521, convert_element_type_2272, hidden_states_206, mul_522, mul_523, sum_14, pow_64, mul_524, mul_525, expand_84, div_7, pow_65, mul_526, mul_527, add_381, convert_element_type_2273, add_382, mul_528, view_1327], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf252, buf237, buf240, buf246, buf249, primals_541, add_268, rsqrt_41, buf253, 1218, 896, stream=stream0)
            del add_268
            del buf237
            del primals_541
            del rsqrt_41
            buf254 = reinterpret_tensor(buf245, (896, 32), (32, 1), 0); del buf245  # reuse
            # Topologically Sorted Source Nodes: [permute_834, mm_555], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf253, (896, 1218), (1, 896), 0), mm_367, out=buf254)
            del mm_367
            buf255 = buf243; del buf243  # reuse
            # Topologically Sorted Source Nodes: [mm_556], Original ATen: [aten.mm]
            extern_kernels.mm(buf253, permute_836, out=buf255)
            del permute_836
            buf257 = buf236; del buf236  # reuse
            # Topologically Sorted Source Nodes: [permute_838, mm_557], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf255, (32, 1218), (1, 32), 0), view_995, out=buf257)
            del view_995
            buf256 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2278], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf254, buf256, 28672, stream=stream0)
            buf259 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2284], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf257, buf259, 28672, stream=stream0)
            buf258 = buf253; del buf253  # reuse
            # Topologically Sorted Source Nodes: [mm_558], Original ATen: [aten.mm]
            extern_kernels.mm(buf255, permute_840, out=buf258)
            del permute_840
            buf260 = buf249; del buf249  # reuse
            # Topologically Sorted Source Nodes: [view_1331, result_432, permute_842, mm_559], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf252, (1218, 896), (896, 1), 0), primals_538, out=buf260)
            del primals_538
            buf261 = reinterpret_tensor(buf258, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf258  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1330, view_1332, add_383, view_1333, permute_843, _scaled_dot_product_efficient_attention_backward_3], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf261, buf260, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1330, view_1332, add_383, view_1333, permute_843, _scaled_dot_product_efficient_attention_backward_3], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf262 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf261, add_265, view_990, view_991, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_80, getitem_81, getitem_82, getitem_83, 0.0, [True, True, True, False], scale=0.125)
            del add_265
            del getitem_80
            del getitem_81
            del getitem_82
            del getitem_83
            del view_990
            del view_991
            buf263 = buf262[0]
            assert_size_stride(buf263, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf263, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf264 = buf262[1]
            assert_size_stride(buf264, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf264, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf266 = reinterpret_tensor(buf210, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf210  # reuse
            # Topologically Sorted Source Nodes: [view_1335, sum_16], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf264, buf266, 155904, stream=stream0)
            buf265 = buf262[2]
            assert_size_stride(buf265, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf265, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf262
            buf267 = buf194; del buf194  # reuse
            buf268 = reinterpret_tensor(buf203, (1218, 128), (128, 1), 0); del buf203  # reuse
            # Topologically Sorted Source Nodes: [view_1334, sum_15, squeeze_6, permute_844, clone_59, view_1336, mul_533, view_1337], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf265, buf267, buf268, 155904, stream=stream0)
            buf269 = buf204; del buf204  # reuse
            # Topologically Sorted Source Nodes: [permute_845, mm_560], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf268, (128, 1218), (1, 128), 0), mm_364, out=buf269)
            del mm_364
            buf270 = buf255; del buf255  # reuse
            # Topologically Sorted Source Nodes: [mm_561], Original ATen: [aten.mm]
            extern_kernels.mm(buf268, permute_847, out=buf270)
            del permute_847
            buf271 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2292], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf269, buf271, 4096, stream=stream0)
            buf272 = buf257; del buf257  # reuse
            # Topologically Sorted Source Nodes: [permute_849, mm_562], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf270, (32, 1218), (1, 32), 0), view_971, out=buf272)
            buf274 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2298], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf272, buf274, 28672, stream=stream0)
            buf276 = reinterpret_tensor(buf268, (1, 1218, 128), (155904, 128, 1), 0); del buf268  # reuse
            buf283 = reinterpret_tensor(buf193, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf193  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1335, sum_16, squeeze_7, mul_529, slice_134, slice_135, neg_58, add_384, mul_530, add_385, permute_854, clone_60, view_1343, mul_534], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf266, mm_default, buf276, buf283, 1218, 128, stream=stream0)
            buf277 = buf269; del buf269  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1335, sum_16, squeeze_7, mul_529, slice_134, slice_135, neg_58, add_384, mul_530, add_385, permute_854, clone_60, view_1343, mul_534, view_1344, permute_855, mm_565], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf276, (128, 1218), (1, 128), 0), mm_362, out=buf277)
            del mm_362
            buf278 = buf234; del buf234  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1335, sum_16, squeeze_7, mul_529, slice_134, slice_135, neg_58, add_384, mul_530, add_385, permute_854, clone_60, view_1343, mul_534, view_1344, mm_566], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf276, (1218, 128), (128, 1), 0), permute_857, out=buf278)
            del permute_857
            buf279 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2306], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf277, buf279, 4096, stream=stream0)
            buf280 = buf272; del buf272  # reuse
            # Topologically Sorted Source Nodes: [permute_859, mm_567], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf278, (32, 1218), (1, 32), 0), view_971, out=buf280)
            buf282 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2312], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf280, buf282, 28672, stream=stream0)
            buf275 = reinterpret_tensor(buf265, (1218, 896), (896, 1), 0); del buf265  # reuse
            # Topologically Sorted Source Nodes: [view_1334, sum_15, squeeze_6, permute_844, clone_59, view_1336, view_1341, result_429, permute_853, mm_564], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf267, (1218, 128), (128, 1), 0), primals_534, out=buf275)
            del primals_534
            buf284 = reinterpret_tensor(buf264, (1218, 896), (896, 1), 0); del buf264  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1335, sum_16, squeeze_7, mul_529, slice_134, slice_135, neg_58, add_384, mul_530, add_385, permute_854, clone_60, view_1343, view_1348, result_426, permute_863, mm_569], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf283, (1218, 128), (128, 1), 0), primals_530, out=buf284)
            del primals_530
            buf273 = reinterpret_tensor(buf261, (1218, 896), (896, 1), 0); del buf261  # reuse
            # Topologically Sorted Source Nodes: [mm_563], Original ATen: [aten.mm]
            extern_kernels.mm(buf270, permute_851, out=buf273)
            del permute_851
            buf281 = buf260; del buf260  # reuse
            # Topologically Sorted Source Nodes: [mm_568], Original ATen: [aten.mm]
            extern_kernels.mm(buf278, permute_861, out=buf281)
            del permute_861
            buf285 = reinterpret_tensor(buf246, (1, 1218, 896), (1091328, 896, 1), 0); del buf246  # reuse
            buf292 = reinterpret_tensor(buf240, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf240  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_531, slice_136, slice_137, neg_59, add_386, mul_532, add_387, permute_864, clone_61, view_1350, mul_535], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf263, mm_default, buf285, buf292, 1091328, stream=stream0)
            buf286 = reinterpret_tensor(buf280, (896, 32), (32, 1), 0); del buf280  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_531, slice_136, slice_137, neg_59, add_386, mul_532, add_387, permute_864, clone_61, view_1350, mul_535, view_1351, permute_865, mm_570], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf285, (896, 1218), (1, 896), 0), mm_360, out=buf286)
            del mm_360
            buf287 = buf278; del buf278  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_531, slice_136, slice_137, neg_59, add_386, mul_532, add_387, permute_864, clone_61, view_1350, mul_535, view_1351, mm_571], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf285, (1218, 896), (896, 1), 0), permute_867, out=buf287)
            del permute_867
            buf289 = reinterpret_tensor(buf254, (32, 896), (896, 1), 0); del buf254  # reuse
            # Topologically Sorted Source Nodes: [permute_869, mm_572], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf287, (32, 1218), (1, 32), 0), view_971, out=buf289)
            del view_971
            buf293 = reinterpret_tensor(buf285, (1218, 896), (896, 1), 0); del buf285  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_531, slice_136, slice_137, neg_59, add_386, mul_532, add_387, permute_864, clone_61, view_1350, view_1355, result_423, permute_873, mm_574], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf292, (1218, 896), (896, 1), 0), primals_526, out=buf293)
            del primals_526
            buf288 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2320], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf286, buf288, 28672, stream=stream0)
            buf291 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2326], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf289, buf291, 28672, stream=stream0)
            buf290 = reinterpret_tensor(buf292, (1218, 896), (896, 1), 0); del buf292  # reuse
            # Topologically Sorted Source Nodes: [mm_573], Original ATen: [aten.mm]
            extern_kernels.mm(buf287, permute_871, out=buf290)
            del permute_871
            buf296 = buf252; del buf252  # reuse
            buf297 = reinterpret_tensor(buf263, (1218, 896), (896, 1), 0); del buf263  # reuse
            # Topologically Sorted Source Nodes: [view_1340, view_1342, add_388, view_1347, add_389, view_1349, add_390, view_1354, add_391, view_1356, add_392, mul_536, convert_element_type_2330, hidden_states_200, mul_537, mul_538, sum_17, pow_66, mul_539, mul_540, expand_85, div_8, pow_67, mul_541, mul_542, add_393, convert_element_type_2331, add_394, mul_543, view_1357], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf296, buf273, buf275, buf281, buf284, buf290, buf293, primals_525, add_260, rsqrt_40, buf297, 1218, 896, stream=stream0)
            del add_260
            del buf273
            del buf275
            del buf281
            del buf284
            del buf290
            del buf293
            del primals_525
            del rsqrt_40
            buf1 = empty_strided_cuda((151936, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_602, mm_433], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf0, (151936, 1025), (1, 151936), 0), view_1161, out=buf1)
            del buf0
            del view_1161
            buf298 = reinterpret_tensor(buf289, (896, 32), (32, 1), 0); del buf289  # reuse
            # Topologically Sorted Source Nodes: [permute_874, mm_575], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf297, (896, 1218), (1, 896), 0), mm_358, out=buf298)
            del mm_358
            buf299 = buf287; del buf287  # reuse
            # Topologically Sorted Source Nodes: [mm_576], Original ATen: [aten.mm]
            extern_kernels.mm(buf297, permute_876, out=buf299)
            del permute_876
            buf301 = reinterpret_tensor(buf242, (32, 4864), (4864, 1), 0); del buf242  # reuse
            # Topologically Sorted Source Nodes: [permute_878, mm_577], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf299, (32, 1218), (1, 32), 0), view_965, out=buf301)
            del view_965
            buf300 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2336], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf298, buf300, 28672, stream=stream0)
            buf303 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2342], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf301, buf303, 155648, stream=stream0)
            buf302 = empty_strided_cuda((1218, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mm_578], Original ATen: [aten.mm]
            extern_kernels.mm(buf299, permute_880, out=buf302)
            del permute_880
            buf304 = empty_strided_cuda((1218, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1361, result_420, permute_882, mm_579], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf296, (1218, 896), (896, 1), 0), primals_522, out=buf304)
            del primals_522
            buf305 = empty_strided_cuda((1, 1218, 4864), (5924352, 4864, 1), torch.bfloat16)
            buf312 = empty_strided_cuda((1, 1218, 4864), (5924352, 4864, 1), torch.bfloat16)
            buf314 = empty_strided_cuda((1, 1218, 4864), (5924352, 4864, 1), torch.bfloat16)
            buf321 = empty_strided_cuda((1, 1218, 4864), (5924352, 4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1360, view_1362, add_395, silu_19, mul_544, mul_545, mul_546, convert_element_type_2360, neg_60, exp_4, add_397, reciprocal_4, mul_547, mul_548, sub_4, mul_549, add_398, mul_550, convert_element_type_2362, mul_551], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf302, buf304, add_257, add_258, buf305, buf312, buf314, buf321, 5924352, stream=stream0)
            del add_257
            del add_258
            buf313 = buf297; del buf297  # reuse
            # Topologically Sorted Source Nodes: [view_1360, view_1362, add_395, silu_19, mul_544, view_1367, result_417, permute_891, mm_584], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf312, (1218, 4864), (4864, 1), 0), primals_519, out=buf313)
            del primals_519
            buf322 = empty_strided_cuda((1218, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1360, view_1362, add_395, silu_19, mul_545, convert_element_type_2360, neg_60, exp_4, add_397, reciprocal_4, mul_547, mul_548, sub_4, mul_549, add_398, mul_550, convert_element_type_2362, view_1373, result_414, permute_900, mm_589], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf321, (1218, 4864), (4864, 1), 0), primals_516, out=buf322)
            del primals_516
            buf307 = buf299; del buf299  # reuse
            # Topologically Sorted Source Nodes: [view_1360, view_1362, add_395, silu_19, mul_544, mul_546, view_1363, mm_581], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf305, (1218, 4864), (4864, 1), 0), permute_885, out=buf307)
            del permute_885
            buf306 = reinterpret_tensor(buf301, (4864, 32), (32, 1), 0); del buf301  # reuse
            # Topologically Sorted Source Nodes: [view_1360, view_1362, add_395, silu_19, mul_544, mul_546, view_1363, permute_883, mm_580], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf305, (4864, 1218), (1, 4864), 0), mm_355, out=buf306)
            del mm_355
            buf316 = buf270; del buf270  # reuse
            # Topologically Sorted Source Nodes: [view_1360, view_1362, add_395, silu_19, mul_545, convert_element_type_2360, neg_60, exp_4, add_397, reciprocal_4, mul_547, mul_548, sub_4, mul_549, add_398, mul_550, convert_element_type_2362, mul_551, view_1369, mm_586], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf314, (1218, 4864), (4864, 1), 0), permute_894, out=buf316)
            del permute_894
            buf315 = buf233; del buf233  # reuse
            # Topologically Sorted Source Nodes: [view_1360, view_1362, add_395, silu_19, mul_545, convert_element_type_2360, neg_60, exp_4, add_397, reciprocal_4, mul_547, mul_548, sub_4, mul_549, add_398, mul_550, convert_element_type_2362, mul_551, view_1369, permute_892, mm_585], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf314, (4864, 1218), (1, 4864), 0), mm_352, out=buf315)
            del mm_352
            buf309 = reinterpret_tensor(buf298, (32, 896), (896, 1), 0); del buf298  # reuse
            # Topologically Sorted Source Nodes: [permute_887, mm_582], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf307, (32, 1218), (1, 32), 0), view_953, out=buf309)
            buf318 = reinterpret_tensor(buf286, (32, 896), (896, 1), 0); del buf286  # reuse
            # Topologically Sorted Source Nodes: [permute_896, mm_587], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf316, (32, 1218), (1, 32), 0), view_953, out=buf318)
            del view_953
            buf311 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2356], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf309, buf311, 28672, stream=stream0)
            buf320 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2373], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf318, buf320, 28672, stream=stream0)
            buf308 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2350], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf306, buf308, 155648, stream=stream0)
            buf317 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2367], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf315, buf317, 155648, stream=stream0)
            buf310 = empty_strided_cuda((1218, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mm_583], Original ATen: [aten.mm]
            extern_kernels.mm(buf307, permute_889, out=buf310)
            del permute_889
            buf319 = empty_strided_cuda((1218, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [mm_588], Original ATen: [aten.mm]
            extern_kernels.mm(buf316, permute_898, out=buf319)
            del permute_898
            buf325 = buf296; del buf296  # reuse
            buf326 = empty_strided_cuda((1218, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view_1366, view_1368, add_396, view_1372, add_399, view_1374, add_400, mul_552, convert_element_type_2377, hidden_states_196, mul_553, mul_554, sum_18, pow_68, mul_555, mul_556, expand_86, div_9, pow_69, mul_557, mul_558, add_401, convert_element_type_2378, add_402, mul_559, view_1375], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf325, buf310, buf313, buf319, buf322, primals_515, add_255, rsqrt_39, buf326, 1218, 896, stream=stream0)
            del add_255
            del buf310
            del primals_515
            del rsqrt_39
            buf327 = reinterpret_tensor(buf318, (896, 32), (32, 1), 0); del buf318  # reuse
            # Topologically Sorted Source Nodes: [permute_901, mm_590], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf326, (896, 1218), (1, 896), 0), mm_349, out=buf327)
            del mm_349
            buf328 = buf316; del buf316  # reuse
            # Topologically Sorted Source Nodes: [mm_591], Original ATen: [aten.mm]
            extern_kernels.mm(buf326, permute_903, out=buf328)
            del permute_903
            buf330 = buf309; del buf309  # reuse
            # Topologically Sorted Source Nodes: [permute_905, mm_592], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf328, (32, 1218), (1, 32), 0), view_947, out=buf330)
            del view_947
            buf329 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2383], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf327, buf329, 28672, stream=stream0)
            buf332 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2389], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf330, buf332, 28672, stream=stream0)
            buf331 = buf326; del buf326  # reuse
            # Topologically Sorted Source Nodes: [mm_593], Original ATen: [aten.mm]
            extern_kernels.mm(buf328, permute_907, out=buf331)
            del permute_907
            buf333 = buf322; del buf322  # reuse
            # Topologically Sorted Source Nodes: [view_1379, result_411, permute_909, mm_594], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf325, (1218, 896), (896, 1), 0), primals_512, out=buf333)
            del primals_512
            buf334 = reinterpret_tensor(buf331, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf331  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1378, view_1380, add_403, view_1381, permute_910, _scaled_dot_product_efficient_attention_backward_4], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf334, buf333, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1378, view_1380, add_403, view_1381, permute_910, _scaled_dot_product_efficient_attention_backward_4], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf335 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf334, add_252, view_942, view_943, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_76, getitem_77, getitem_78, getitem_79, 0.0, [True, True, True, False], scale=0.125)
            del add_252
            del getitem_76
            del getitem_77
            del getitem_78
            del getitem_79
            del view_942
            del view_943
            buf336 = buf335[0]
            assert_size_stride(buf336, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf336, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf337 = buf335[1]
            assert_size_stride(buf337, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf337, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf339 = reinterpret_tensor(buf283, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf283  # reuse
            # Topologically Sorted Source Nodes: [view_1383, sum_20], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf337, buf339, 155904, stream=stream0)
            buf338 = buf335[2]
            assert_size_stride(buf338, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf338, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf335
            buf340 = buf267; del buf267  # reuse
            buf341 = reinterpret_tensor(buf276, (1218, 128), (128, 1), 0); del buf276  # reuse
            # Topologically Sorted Source Nodes: [view_1382, sum_19, squeeze_8, permute_911, clone_62, view_1384, mul_564, view_1385], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf338, buf340, buf341, 155904, stream=stream0)
            buf342 = buf277; del buf277  # reuse
            # Topologically Sorted Source Nodes: [permute_912, mm_595], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf341, (128, 1218), (1, 128), 0), mm_346, out=buf342)
            del mm_346
            buf343 = buf328; del buf328  # reuse
            # Topologically Sorted Source Nodes: [mm_596], Original ATen: [aten.mm]
            extern_kernels.mm(buf341, permute_914, out=buf343)
            del permute_914
            buf344 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2397], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf342, buf344, 4096, stream=stream0)
            buf345 = buf330; del buf330  # reuse
            # Topologically Sorted Source Nodes: [permute_916, mm_597], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf343, (32, 1218), (1, 32), 0), view_923, out=buf345)
            buf347 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2403], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf345, buf347, 28672, stream=stream0)
            buf349 = reinterpret_tensor(buf341, (1, 1218, 128), (155904, 128, 1), 0); del buf341  # reuse
            buf356 = reinterpret_tensor(buf266, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf266  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1383, sum_20, squeeze_9, mul_560, slice_138, slice_139, neg_61, add_404, mul_561, add_405, permute_921, clone_63, view_1391, mul_565], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf339, mm_default, buf349, buf356, 1218, 128, stream=stream0)
            buf350 = buf342; del buf342  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1383, sum_20, squeeze_9, mul_560, slice_138, slice_139, neg_61, add_404, mul_561, add_405, permute_921, clone_63, view_1391, mul_565, view_1392, permute_922, mm_600], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf349, (128, 1218), (1, 128), 0), mm_344, out=buf350)
            del mm_344
            buf351 = buf307; del buf307  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1383, sum_20, squeeze_9, mul_560, slice_138, slice_139, neg_61, add_404, mul_561, add_405, permute_921, clone_63, view_1391, mul_565, view_1392, mm_601], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf349, (1218, 128), (128, 1), 0), permute_924, out=buf351)
            del permute_924
            buf352 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2411], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf350, buf352, 4096, stream=stream0)
            buf353 = buf345; del buf345  # reuse
            # Topologically Sorted Source Nodes: [permute_926, mm_602], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf351, (32, 1218), (1, 32), 0), view_923, out=buf353)
            buf355 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2417], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf353, buf355, 28672, stream=stream0)
            buf348 = reinterpret_tensor(buf338, (1218, 896), (896, 1), 0); del buf338  # reuse
            # Topologically Sorted Source Nodes: [view_1382, sum_19, squeeze_8, permute_911, clone_62, view_1384, view_1389, result_408, permute_920, mm_599], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf340, (1218, 128), (128, 1), 0), primals_508, out=buf348)
            del primals_508
            buf357 = reinterpret_tensor(buf337, (1218, 896), (896, 1), 0); del buf337  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1383, sum_20, squeeze_9, mul_560, slice_138, slice_139, neg_61, add_404, mul_561, add_405, permute_921, clone_63, view_1391, view_1396, result_405, permute_930, mm_604], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf356, (1218, 128), (128, 1), 0), primals_504, out=buf357)
            del primals_504
            buf346 = reinterpret_tensor(buf334, (1218, 896), (896, 1), 0); del buf334  # reuse
            # Topologically Sorted Source Nodes: [mm_598], Original ATen: [aten.mm]
            extern_kernels.mm(buf343, permute_918, out=buf346)
            del permute_918
            buf354 = buf333; del buf333  # reuse
            # Topologically Sorted Source Nodes: [mm_603], Original ATen: [aten.mm]
            extern_kernels.mm(buf351, permute_928, out=buf354)
            del permute_928
            buf358 = reinterpret_tensor(buf319, (1, 1218, 896), (1091328, 896, 1), 0); del buf319  # reuse
            buf365 = reinterpret_tensor(buf313, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf313  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_562, slice_140, slice_141, neg_62, add_406, mul_563, add_407, permute_931, clone_64, view_1398, mul_566], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf336, mm_default, buf358, buf365, 1091328, stream=stream0)
            buf359 = reinterpret_tensor(buf353, (896, 32), (32, 1), 0); del buf353  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_562, slice_140, slice_141, neg_62, add_406, mul_563, add_407, permute_931, clone_64, view_1398, mul_566, view_1399, permute_932, mm_605], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf358, (896, 1218), (1, 896), 0), mm_342, out=buf359)
            del mm_342
            buf360 = buf351; del buf351  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_562, slice_140, slice_141, neg_62, add_406, mul_563, add_407, permute_931, clone_64, view_1398, mul_566, view_1399, mm_606], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf358, (1218, 896), (896, 1), 0), permute_934, out=buf360)
            del permute_934
            buf362 = reinterpret_tensor(buf327, (32, 896), (896, 1), 0); del buf327  # reuse
            # Topologically Sorted Source Nodes: [permute_936, mm_607], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf360, (32, 1218), (1, 32), 0), view_923, out=buf362)
            del view_923
            buf366 = reinterpret_tensor(buf358, (1218, 896), (896, 1), 0); del buf358  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_562, slice_140, slice_141, neg_62, add_406, mul_563, add_407, permute_931, clone_64, view_1398, view_1403, result_402, permute_940, mm_609], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf365, (1218, 896), (896, 1), 0), primals_500, out=buf366)
            del primals_500
            buf361 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2425], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf359, buf361, 28672, stream=stream0)
            buf364 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2431], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf362, buf364, 28672, stream=stream0)
            buf363 = reinterpret_tensor(buf365, (1218, 896), (896, 1), 0); del buf365  # reuse
            # Topologically Sorted Source Nodes: [mm_608], Original ATen: [aten.mm]
            extern_kernels.mm(buf360, permute_938, out=buf363)
            del permute_938
            buf369 = buf325; del buf325  # reuse
            buf370 = reinterpret_tensor(buf336, (1218, 896), (896, 1), 0); del buf336  # reuse
            # Topologically Sorted Source Nodes: [view_1388, view_1390, add_408, view_1395, add_409, view_1397, add_410, view_1402, add_411, view_1404, add_412, mul_567, convert_element_type_2435, hidden_states_190, mul_568, mul_569, sum_21, pow_70, mul_570, mul_571, expand_87, div_10, pow_71, mul_572, mul_573, add_413, convert_element_type_2436, add_414, mul_574, view_1405], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf369, buf346, buf348, buf354, buf357, buf363, buf366, primals_499, add_247, rsqrt_38, buf370, 1218, 896, stream=stream0)
            del add_247
            del buf346
            del buf348
            del primals_499
            del rsqrt_38
            buf371 = reinterpret_tensor(buf362, (896, 32), (32, 1), 0); del buf362  # reuse
            # Topologically Sorted Source Nodes: [permute_941, mm_610], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf370, (896, 1218), (1, 896), 0), mm_340, out=buf371)
            del mm_340
            buf372 = buf360; del buf360  # reuse
            # Topologically Sorted Source Nodes: [mm_611], Original ATen: [aten.mm]
            extern_kernels.mm(buf370, permute_943, out=buf372)
            del permute_943
            buf374 = reinterpret_tensor(buf315, (32, 4864), (4864, 1), 0); del buf315  # reuse
            # Topologically Sorted Source Nodes: [permute_945, mm_612], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf372, (32, 1218), (1, 32), 0), view_917, out=buf374)
            del view_917
            buf373 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2441], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf371, buf373, 28672, stream=stream0)
            buf376 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2447], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf374, buf376, 155648, stream=stream0)
            buf375 = reinterpret_tensor(buf314, (1218, 4864), (4864, 1), 0); del buf314  # reuse
            # Topologically Sorted Source Nodes: [mm_613], Original ATen: [aten.mm]
            extern_kernels.mm(buf372, permute_947, out=buf375)
            del permute_947
            buf377 = reinterpret_tensor(buf305, (1218, 4864), (4864, 1), 0); del buf305  # reuse
            # Topologically Sorted Source Nodes: [view_1409, result_399, permute_949, mm_614], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf369, (1218, 896), (896, 1), 0), primals_496, out=buf377)
            del primals_496
            buf378 = buf321; del buf321  # reuse
            buf385 = buf312; del buf312  # reuse
            buf387 = reinterpret_tensor(buf304, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf304  # reuse
            buf394 = reinterpret_tensor(buf302, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf302  # reuse
            # Topologically Sorted Source Nodes: [view_1408, view_1410, add_415, silu_18, mul_575, mul_576, mul_577, convert_element_type_2465, neg_63, exp_5, add_417, reciprocal_5, mul_578, mul_579, sub_5, mul_580, add_418, mul_581, convert_element_type_2467, mul_582], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf375, buf377, add_244, add_245, buf378, buf385, buf387, buf394, 5924352, stream=stream0)
            del add_244
            del add_245
            buf386 = buf370; del buf370  # reuse
            # Topologically Sorted Source Nodes: [view_1408, view_1410, add_415, silu_18, mul_575, view_1415, result_396, permute_958, mm_619], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf385, (1218, 4864), (4864, 1), 0), primals_493, out=buf386)
            del primals_493
            buf395 = buf366; del buf366  # reuse
            # Topologically Sorted Source Nodes: [view_1408, view_1410, add_415, silu_18, mul_576, convert_element_type_2465, neg_63, exp_5, add_417, reciprocal_5, mul_578, mul_579, sub_5, mul_580, add_418, mul_581, convert_element_type_2467, view_1421, result_393, permute_967, mm_624], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf394, (1218, 4864), (4864, 1), 0), primals_490, out=buf395)
            del primals_490
            buf380 = buf372; del buf372  # reuse
            # Topologically Sorted Source Nodes: [view_1408, view_1410, add_415, silu_18, mul_575, mul_577, view_1411, mm_616], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf378, (1218, 4864), (4864, 1), 0), permute_952, out=buf380)
            del permute_952
            buf379 = reinterpret_tensor(buf374, (4864, 32), (32, 1), 0); del buf374  # reuse
            # Topologically Sorted Source Nodes: [view_1408, view_1410, add_415, silu_18, mul_575, mul_577, view_1411, permute_950, mm_615], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf378, (4864, 1218), (1, 4864), 0), mm_337, out=buf379)
            del mm_337
            buf389 = buf343; del buf343  # reuse
            # Topologically Sorted Source Nodes: [view_1408, view_1410, add_415, silu_18, mul_576, convert_element_type_2465, neg_63, exp_5, add_417, reciprocal_5, mul_578, mul_579, sub_5, mul_580, add_418, mul_581, convert_element_type_2467, mul_582, view_1417, mm_621], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf387, (1218, 4864), (4864, 1), 0), permute_961, out=buf389)
            del permute_961
            buf388 = buf306; del buf306  # reuse
            # Topologically Sorted Source Nodes: [view_1408, view_1410, add_415, silu_18, mul_576, convert_element_type_2465, neg_63, exp_5, add_417, reciprocal_5, mul_578, mul_579, sub_5, mul_580, add_418, mul_581, convert_element_type_2467, mul_582, view_1417, permute_959, mm_620], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf387, (4864, 1218), (1, 4864), 0), mm_334, out=buf388)
            del mm_334
            buf382 = reinterpret_tensor(buf371, (32, 896), (896, 1), 0); del buf371  # reuse
            # Topologically Sorted Source Nodes: [permute_954, mm_617], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf380, (32, 1218), (1, 32), 0), view_905, out=buf382)
            buf391 = reinterpret_tensor(buf359, (32, 896), (896, 1), 0); del buf359  # reuse
            # Topologically Sorted Source Nodes: [permute_963, mm_622], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf389, (32, 1218), (1, 32), 0), view_905, out=buf391)
            del view_905
            buf384 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2461], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf382, buf384, 28672, stream=stream0)
            buf393 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2478], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf391, buf393, 28672, stream=stream0)
            buf381 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2455], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf379, buf381, 155648, stream=stream0)
            buf390 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2472], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf388, buf390, 155648, stream=stream0)
            buf383 = buf363; del buf363  # reuse
            # Topologically Sorted Source Nodes: [mm_618], Original ATen: [aten.mm]
            extern_kernels.mm(buf380, permute_956, out=buf383)
            del permute_956
            buf392 = buf357; del buf357  # reuse
            # Topologically Sorted Source Nodes: [mm_623], Original ATen: [aten.mm]
            extern_kernels.mm(buf389, permute_965, out=buf392)
            del permute_965
            buf398 = buf369; del buf369  # reuse
            buf399 = buf354; del buf354  # reuse
            # Topologically Sorted Source Nodes: [view_1414, view_1416, add_416, view_1420, add_419, view_1422, add_420, mul_583, convert_element_type_2482, hidden_states_186, mul_584, mul_585, sum_22, pow_72, mul_586, mul_587, expand_88, div_11, pow_73, mul_588, mul_589, add_421, convert_element_type_2483, add_422, mul_590, view_1423], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf398, buf383, buf386, buf392, buf395, primals_489, add_242, rsqrt_37, buf399, 1218, 896, stream=stream0)
            del add_242
            del buf383
            del primals_489
            del rsqrt_37
            buf400 = reinterpret_tensor(buf391, (896, 32), (32, 1), 0); del buf391  # reuse
            # Topologically Sorted Source Nodes: [permute_968, mm_625], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf399, (896, 1218), (1, 896), 0), mm_331, out=buf400)
            del mm_331
            buf401 = buf389; del buf389  # reuse
            # Topologically Sorted Source Nodes: [mm_626], Original ATen: [aten.mm]
            extern_kernels.mm(buf399, permute_970, out=buf401)
            del permute_970
            buf403 = buf382; del buf382  # reuse
            # Topologically Sorted Source Nodes: [permute_972, mm_627], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf401, (32, 1218), (1, 32), 0), view_899, out=buf403)
            del view_899
            buf402 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2488], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf400, buf402, 28672, stream=stream0)
            buf405 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2494], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf403, buf405, 28672, stream=stream0)
            buf404 = buf399; del buf399  # reuse
            # Topologically Sorted Source Nodes: [mm_628], Original ATen: [aten.mm]
            extern_kernels.mm(buf401, permute_974, out=buf404)
            del permute_974
            buf406 = buf395; del buf395  # reuse
            # Topologically Sorted Source Nodes: [view_1427, result_390, permute_976, mm_629], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf398, (1218, 896), (896, 1), 0), primals_486, out=buf406)
            del primals_486
            buf407 = reinterpret_tensor(buf404, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf404  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1426, view_1428, add_423, view_1429, permute_977, _scaled_dot_product_efficient_attention_backward_5], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf407, buf406, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1426, view_1428, add_423, view_1429, permute_977, _scaled_dot_product_efficient_attention_backward_5], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf408 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf407, add_239, view_894, view_895, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_72, getitem_73, getitem_74, getitem_75, 0.0, [True, True, True, False], scale=0.125)
            del add_239
            del getitem_72
            del getitem_73
            del getitem_74
            del getitem_75
            del view_894
            del view_895
            buf409 = buf408[0]
            assert_size_stride(buf409, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf409, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf410 = buf408[1]
            assert_size_stride(buf410, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf410, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf412 = reinterpret_tensor(buf356, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf356  # reuse
            # Topologically Sorted Source Nodes: [view_1431, sum_24], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf410, buf412, 155904, stream=stream0)
            buf411 = buf408[2]
            assert_size_stride(buf411, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf411, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf408
            buf413 = buf340; del buf340  # reuse
            buf414 = reinterpret_tensor(buf349, (1218, 128), (128, 1), 0); del buf349  # reuse
            # Topologically Sorted Source Nodes: [view_1430, sum_23, squeeze_10, permute_978, clone_65, view_1432, mul_595, view_1433], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf411, buf413, buf414, 155904, stream=stream0)
            buf415 = buf350; del buf350  # reuse
            # Topologically Sorted Source Nodes: [permute_979, mm_630], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf414, (128, 1218), (1, 128), 0), mm_328, out=buf415)
            del mm_328
            buf416 = buf401; del buf401  # reuse
            # Topologically Sorted Source Nodes: [mm_631], Original ATen: [aten.mm]
            extern_kernels.mm(buf414, permute_981, out=buf416)
            del permute_981
            buf417 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2502], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf415, buf417, 4096, stream=stream0)
            buf418 = buf403; del buf403  # reuse
            # Topologically Sorted Source Nodes: [permute_983, mm_632], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf416, (32, 1218), (1, 32), 0), view_875, out=buf418)
            buf420 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2508], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf418, buf420, 28672, stream=stream0)
            buf422 = reinterpret_tensor(buf414, (1, 1218, 128), (155904, 128, 1), 0); del buf414  # reuse
            buf429 = reinterpret_tensor(buf339, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf339  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1431, sum_24, squeeze_11, mul_591, slice_142, slice_143, neg_64, add_424, mul_592, add_425, permute_988, clone_66, view_1439, mul_596], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf412, mm_default, buf422, buf429, 1218, 128, stream=stream0)
            buf423 = buf415; del buf415  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1431, sum_24, squeeze_11, mul_591, slice_142, slice_143, neg_64, add_424, mul_592, add_425, permute_988, clone_66, view_1439, mul_596, view_1440, permute_989, mm_635], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf422, (128, 1218), (1, 128), 0), mm_326, out=buf423)
            del mm_326
            buf424 = buf380; del buf380  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1431, sum_24, squeeze_11, mul_591, slice_142, slice_143, neg_64, add_424, mul_592, add_425, permute_988, clone_66, view_1439, mul_596, view_1440, mm_636], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf422, (1218, 128), (128, 1), 0), permute_991, out=buf424)
            del permute_991
            buf425 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2516], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf423, buf425, 4096, stream=stream0)
            buf426 = buf418; del buf418  # reuse
            # Topologically Sorted Source Nodes: [permute_993, mm_637], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf424, (32, 1218), (1, 32), 0), view_875, out=buf426)
            buf428 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2522], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf426, buf428, 28672, stream=stream0)
            buf421 = reinterpret_tensor(buf411, (1218, 896), (896, 1), 0); del buf411  # reuse
            # Topologically Sorted Source Nodes: [view_1430, sum_23, squeeze_10, permute_978, clone_65, view_1432, view_1437, result_387, permute_987, mm_634], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf413, (1218, 128), (128, 1), 0), primals_482, out=buf421)
            del primals_482
            buf430 = reinterpret_tensor(buf410, (1218, 896), (896, 1), 0); del buf410  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1431, sum_24, squeeze_11, mul_591, slice_142, slice_143, neg_64, add_424, mul_592, add_425, permute_988, clone_66, view_1439, view_1444, result_384, permute_997, mm_639], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf429, (1218, 128), (128, 1), 0), primals_478, out=buf430)
            del primals_478
            buf419 = reinterpret_tensor(buf407, (1218, 896), (896, 1), 0); del buf407  # reuse
            # Topologically Sorted Source Nodes: [mm_633], Original ATen: [aten.mm]
            extern_kernels.mm(buf416, permute_985, out=buf419)
            del permute_985
            buf427 = buf406; del buf406  # reuse
            # Topologically Sorted Source Nodes: [mm_638], Original ATen: [aten.mm]
            extern_kernels.mm(buf424, permute_995, out=buf427)
            del permute_995
            buf431 = reinterpret_tensor(buf392, (1, 1218, 896), (1091328, 896, 1), 0); del buf392  # reuse
            buf438 = reinterpret_tensor(buf386, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf386  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_593, slice_144, slice_145, neg_65, add_426, mul_594, add_427, permute_998, clone_67, view_1446, mul_597], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf409, mm_default, buf431, buf438, 1091328, stream=stream0)
            buf432 = reinterpret_tensor(buf426, (896, 32), (32, 1), 0); del buf426  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_593, slice_144, slice_145, neg_65, add_426, mul_594, add_427, permute_998, clone_67, view_1446, mul_597, view_1447, permute_999, mm_640], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf431, (896, 1218), (1, 896), 0), mm_324, out=buf432)
            del mm_324
            buf433 = buf424; del buf424  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_593, slice_144, slice_145, neg_65, add_426, mul_594, add_427, permute_998, clone_67, view_1446, mul_597, view_1447, mm_641], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf431, (1218, 896), (896, 1), 0), permute_1001, out=buf433)
            del permute_1001
            buf435 = reinterpret_tensor(buf400, (32, 896), (896, 1), 0); del buf400  # reuse
            # Topologically Sorted Source Nodes: [permute_1003, mm_642], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf433, (32, 1218), (1, 32), 0), view_875, out=buf435)
            del view_875
            buf439 = reinterpret_tensor(buf431, (1218, 896), (896, 1), 0); del buf431  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_593, slice_144, slice_145, neg_65, add_426, mul_594, add_427, permute_998, clone_67, view_1446, view_1451, result_381, permute_1007, mm_644], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf438, (1218, 896), (896, 1), 0), primals_474, out=buf439)
            del primals_474
            buf434 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2530], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf432, buf434, 28672, stream=stream0)
            buf437 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2536], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf435, buf437, 28672, stream=stream0)
            buf436 = reinterpret_tensor(buf438, (1218, 896), (896, 1), 0); del buf438  # reuse
            # Topologically Sorted Source Nodes: [mm_643], Original ATen: [aten.mm]
            extern_kernels.mm(buf433, permute_1005, out=buf436)
            del permute_1005
            buf442 = buf398; del buf398  # reuse
            buf443 = reinterpret_tensor(buf409, (1218, 896), (896, 1), 0); del buf409  # reuse
            # Topologically Sorted Source Nodes: [view_1436, view_1438, add_428, view_1443, add_429, view_1445, add_430, view_1450, add_431, view_1452, add_432, mul_598, convert_element_type_2540, hidden_states_180, mul_599, mul_600, sum_25, pow_74, mul_601, mul_602, expand_89, div_12, pow_75, mul_603, mul_604, add_433, convert_element_type_2541, add_434, mul_605, view_1453], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf442, buf419, buf421, buf427, buf430, buf436, buf439, primals_473, add_234, rsqrt_36, buf443, 1218, 896, stream=stream0)
            del add_234
            del buf419
            del buf421
            del primals_473
            del rsqrt_36
            buf444 = reinterpret_tensor(buf435, (896, 32), (32, 1), 0); del buf435  # reuse
            # Topologically Sorted Source Nodes: [permute_1008, mm_645], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf443, (896, 1218), (1, 896), 0), mm_322, out=buf444)
            del mm_322
            buf445 = buf433; del buf433  # reuse
            # Topologically Sorted Source Nodes: [mm_646], Original ATen: [aten.mm]
            extern_kernels.mm(buf443, permute_1010, out=buf445)
            del permute_1010
            buf447 = reinterpret_tensor(buf388, (32, 4864), (4864, 1), 0); del buf388  # reuse
            # Topologically Sorted Source Nodes: [permute_1012, mm_647], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf445, (32, 1218), (1, 32), 0), view_869, out=buf447)
            del view_869
            buf446 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2546], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf444, buf446, 28672, stream=stream0)
            buf449 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2552], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf447, buf449, 155648, stream=stream0)
            buf448 = reinterpret_tensor(buf387, (1218, 4864), (4864, 1), 0); del buf387  # reuse
            # Topologically Sorted Source Nodes: [mm_648], Original ATen: [aten.mm]
            extern_kernels.mm(buf445, permute_1014, out=buf448)
            del permute_1014
            buf450 = reinterpret_tensor(buf378, (1218, 4864), (4864, 1), 0); del buf378  # reuse
            # Topologically Sorted Source Nodes: [view_1457, result_378, permute_1016, mm_649], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf442, (1218, 896), (896, 1), 0), primals_470, out=buf450)
            del primals_470
            buf451 = buf394; del buf394  # reuse
            buf458 = buf385; del buf385  # reuse
            buf460 = reinterpret_tensor(buf377, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf377  # reuse
            buf467 = reinterpret_tensor(buf375, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf375  # reuse
            # Topologically Sorted Source Nodes: [view_1456, view_1458, add_435, silu_17, mul_606, mul_607, mul_608, convert_element_type_2570, neg_66, exp_6, add_437, reciprocal_6, mul_609, mul_610, sub_6, mul_611, add_438, mul_612, convert_element_type_2572, mul_613], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf448, buf450, add_231, add_232, buf451, buf458, buf460, buf467, 5924352, stream=stream0)
            del add_231
            del add_232
            buf459 = buf443; del buf443  # reuse
            # Topologically Sorted Source Nodes: [view_1456, view_1458, add_435, silu_17, mul_606, view_1463, result_375, permute_1025, mm_654], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf458, (1218, 4864), (4864, 1), 0), primals_467, out=buf459)
            del primals_467
            buf468 = buf439; del buf439  # reuse
            # Topologically Sorted Source Nodes: [view_1456, view_1458, add_435, silu_17, mul_607, convert_element_type_2570, neg_66, exp_6, add_437, reciprocal_6, mul_609, mul_610, sub_6, mul_611, add_438, mul_612, convert_element_type_2572, view_1469, result_372, permute_1034, mm_659], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf467, (1218, 4864), (4864, 1), 0), primals_464, out=buf468)
            del primals_464
            buf453 = buf445; del buf445  # reuse
            # Topologically Sorted Source Nodes: [view_1456, view_1458, add_435, silu_17, mul_606, mul_608, view_1459, mm_651], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf451, (1218, 4864), (4864, 1), 0), permute_1019, out=buf453)
            del permute_1019
            buf452 = reinterpret_tensor(buf447, (4864, 32), (32, 1), 0); del buf447  # reuse
            # Topologically Sorted Source Nodes: [view_1456, view_1458, add_435, silu_17, mul_606, mul_608, view_1459, permute_1017, mm_650], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf451, (4864, 1218), (1, 4864), 0), mm_319, out=buf452)
            del mm_319
            buf462 = buf416; del buf416  # reuse
            # Topologically Sorted Source Nodes: [view_1456, view_1458, add_435, silu_17, mul_607, convert_element_type_2570, neg_66, exp_6, add_437, reciprocal_6, mul_609, mul_610, sub_6, mul_611, add_438, mul_612, convert_element_type_2572, mul_613, view_1465, mm_656], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf460, (1218, 4864), (4864, 1), 0), permute_1028, out=buf462)
            del permute_1028
            buf461 = buf379; del buf379  # reuse
            # Topologically Sorted Source Nodes: [view_1456, view_1458, add_435, silu_17, mul_607, convert_element_type_2570, neg_66, exp_6, add_437, reciprocal_6, mul_609, mul_610, sub_6, mul_611, add_438, mul_612, convert_element_type_2572, mul_613, view_1465, permute_1026, mm_655], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf460, (4864, 1218), (1, 4864), 0), mm_316, out=buf461)
            del mm_316
            buf455 = reinterpret_tensor(buf444, (32, 896), (896, 1), 0); del buf444  # reuse
            # Topologically Sorted Source Nodes: [permute_1021, mm_652], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf453, (32, 1218), (1, 32), 0), view_857, out=buf455)
            buf464 = reinterpret_tensor(buf432, (32, 896), (896, 1), 0); del buf432  # reuse
            # Topologically Sorted Source Nodes: [permute_1030, mm_657], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf462, (32, 1218), (1, 32), 0), view_857, out=buf464)
            del view_857
            buf457 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2566], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf455, buf457, 28672, stream=stream0)
            buf466 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2583], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf464, buf466, 28672, stream=stream0)
            buf454 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2560], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf452, buf454, 155648, stream=stream0)
            buf463 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2577], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf461, buf463, 155648, stream=stream0)
            buf456 = buf436; del buf436  # reuse
            # Topologically Sorted Source Nodes: [mm_653], Original ATen: [aten.mm]
            extern_kernels.mm(buf453, permute_1023, out=buf456)
            del permute_1023
            buf465 = buf430; del buf430  # reuse
            # Topologically Sorted Source Nodes: [mm_658], Original ATen: [aten.mm]
            extern_kernels.mm(buf462, permute_1032, out=buf465)
            del permute_1032
            buf471 = buf442; del buf442  # reuse
            buf472 = buf427; del buf427  # reuse
            # Topologically Sorted Source Nodes: [view_1462, view_1464, add_436, view_1468, add_439, view_1470, add_440, mul_614, convert_element_type_2587, hidden_states_176, mul_615, mul_616, sum_26, pow_76, mul_617, mul_618, expand_90, div_13, pow_77, mul_619, mul_620, add_441, convert_element_type_2588, add_442, mul_621, view_1471], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf471, buf456, buf459, buf465, buf468, primals_463, add_229, rsqrt_35, buf472, 1218, 896, stream=stream0)
            del add_229
            del buf456
            del primals_463
            del rsqrt_35
            buf473 = reinterpret_tensor(buf464, (896, 32), (32, 1), 0); del buf464  # reuse
            # Topologically Sorted Source Nodes: [permute_1035, mm_660], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf472, (896, 1218), (1, 896), 0), mm_313, out=buf473)
            del mm_313
            buf474 = buf462; del buf462  # reuse
            # Topologically Sorted Source Nodes: [mm_661], Original ATen: [aten.mm]
            extern_kernels.mm(buf472, permute_1037, out=buf474)
            del permute_1037
            buf476 = buf455; del buf455  # reuse
            # Topologically Sorted Source Nodes: [permute_1039, mm_662], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf474, (32, 1218), (1, 32), 0), view_851, out=buf476)
            del view_851
            buf475 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2593], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf473, buf475, 28672, stream=stream0)
            buf478 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2599], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf476, buf478, 28672, stream=stream0)
            buf477 = buf472; del buf472  # reuse
            # Topologically Sorted Source Nodes: [mm_663], Original ATen: [aten.mm]
            extern_kernels.mm(buf474, permute_1041, out=buf477)
            del permute_1041
            buf479 = buf468; del buf468  # reuse
            # Topologically Sorted Source Nodes: [view_1475, result_369, permute_1043, mm_664], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf471, (1218, 896), (896, 1), 0), primals_460, out=buf479)
            del primals_460
            buf480 = reinterpret_tensor(buf477, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf477  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1474, view_1476, add_443, view_1477, permute_1044, _scaled_dot_product_efficient_attention_backward_6], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf480, buf479, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1474, view_1476, add_443, view_1477, permute_1044, _scaled_dot_product_efficient_attention_backward_6], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf481 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf480, add_226, view_846, view_847, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_68, getitem_69, getitem_70, getitem_71, 0.0, [True, True, True, False], scale=0.125)
            del add_226
            del getitem_68
            del getitem_69
            del getitem_70
            del getitem_71
            del view_846
            del view_847
            buf482 = buf481[0]
            assert_size_stride(buf482, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf482, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf483 = buf481[1]
            assert_size_stride(buf483, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf483, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf485 = reinterpret_tensor(buf429, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf429  # reuse
            # Topologically Sorted Source Nodes: [view_1479, sum_28], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf483, buf485, 155904, stream=stream0)
            buf484 = buf481[2]
            assert_size_stride(buf484, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf484, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf481
            buf486 = buf413; del buf413  # reuse
            buf487 = reinterpret_tensor(buf422, (1218, 128), (128, 1), 0); del buf422  # reuse
            # Topologically Sorted Source Nodes: [view_1478, sum_27, squeeze_12, permute_1045, clone_68, view_1480, mul_626, view_1481], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf484, buf486, buf487, 155904, stream=stream0)
            buf488 = buf423; del buf423  # reuse
            # Topologically Sorted Source Nodes: [permute_1046, mm_665], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf487, (128, 1218), (1, 128), 0), mm_310, out=buf488)
            del mm_310
            buf489 = buf474; del buf474  # reuse
            # Topologically Sorted Source Nodes: [mm_666], Original ATen: [aten.mm]
            extern_kernels.mm(buf487, permute_1048, out=buf489)
            del permute_1048
            buf490 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2607], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf488, buf490, 4096, stream=stream0)
            buf491 = buf476; del buf476  # reuse
            # Topologically Sorted Source Nodes: [permute_1050, mm_667], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf489, (32, 1218), (1, 32), 0), view_827, out=buf491)
            buf493 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2613], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf491, buf493, 28672, stream=stream0)
            buf495 = reinterpret_tensor(buf487, (1, 1218, 128), (155904, 128, 1), 0); del buf487  # reuse
            buf502 = reinterpret_tensor(buf412, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf412  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1479, sum_28, squeeze_13, mul_622, slice_146, slice_147, neg_67, add_444, mul_623, add_445, permute_1055, clone_69, view_1487, mul_627], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf485, mm_default, buf495, buf502, 1218, 128, stream=stream0)
            buf496 = buf488; del buf488  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1479, sum_28, squeeze_13, mul_622, slice_146, slice_147, neg_67, add_444, mul_623, add_445, permute_1055, clone_69, view_1487, mul_627, view_1488, permute_1056, mm_670], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf495, (128, 1218), (1, 128), 0), mm_308, out=buf496)
            del mm_308
            buf497 = buf453; del buf453  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1479, sum_28, squeeze_13, mul_622, slice_146, slice_147, neg_67, add_444, mul_623, add_445, permute_1055, clone_69, view_1487, mul_627, view_1488, mm_671], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf495, (1218, 128), (128, 1), 0), permute_1058, out=buf497)
            del permute_1058
            buf498 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2621], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf496, buf498, 4096, stream=stream0)
            buf499 = buf491; del buf491  # reuse
            # Topologically Sorted Source Nodes: [permute_1060, mm_672], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf497, (32, 1218), (1, 32), 0), view_827, out=buf499)
            buf501 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2627], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf499, buf501, 28672, stream=stream0)
            buf494 = reinterpret_tensor(buf484, (1218, 896), (896, 1), 0); del buf484  # reuse
            # Topologically Sorted Source Nodes: [view_1478, sum_27, squeeze_12, permute_1045, clone_68, view_1480, view_1485, result_366, permute_1054, mm_669], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf486, (1218, 128), (128, 1), 0), primals_456, out=buf494)
            del primals_456
            buf503 = reinterpret_tensor(buf483, (1218, 896), (896, 1), 0); del buf483  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1479, sum_28, squeeze_13, mul_622, slice_146, slice_147, neg_67, add_444, mul_623, add_445, permute_1055, clone_69, view_1487, view_1492, result_363, permute_1064, mm_674], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf502, (1218, 128), (128, 1), 0), primals_452, out=buf503)
            del primals_452
            buf492 = reinterpret_tensor(buf480, (1218, 896), (896, 1), 0); del buf480  # reuse
            # Topologically Sorted Source Nodes: [mm_668], Original ATen: [aten.mm]
            extern_kernels.mm(buf489, permute_1052, out=buf492)
            del permute_1052
            buf500 = buf479; del buf479  # reuse
            # Topologically Sorted Source Nodes: [mm_673], Original ATen: [aten.mm]
            extern_kernels.mm(buf497, permute_1062, out=buf500)
            del permute_1062
            buf504 = reinterpret_tensor(buf465, (1, 1218, 896), (1091328, 896, 1), 0); del buf465  # reuse
            buf511 = reinterpret_tensor(buf459, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf459  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_624, slice_148, slice_149, neg_68, add_446, mul_625, add_447, permute_1065, clone_70, view_1494, mul_628], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf482, mm_default, buf504, buf511, 1091328, stream=stream0)
            buf505 = reinterpret_tensor(buf499, (896, 32), (32, 1), 0); del buf499  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_624, slice_148, slice_149, neg_68, add_446, mul_625, add_447, permute_1065, clone_70, view_1494, mul_628, view_1495, permute_1066, mm_675], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf504, (896, 1218), (1, 896), 0), mm_306, out=buf505)
            del mm_306
            buf506 = buf497; del buf497  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_624, slice_148, slice_149, neg_68, add_446, mul_625, add_447, permute_1065, clone_70, view_1494, mul_628, view_1495, mm_676], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf504, (1218, 896), (896, 1), 0), permute_1068, out=buf506)
            del permute_1068
            buf508 = reinterpret_tensor(buf473, (32, 896), (896, 1), 0); del buf473  # reuse
            # Topologically Sorted Source Nodes: [permute_1070, mm_677], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf506, (32, 1218), (1, 32), 0), view_827, out=buf508)
            del view_827
            buf512 = reinterpret_tensor(buf504, (1218, 896), (896, 1), 0); del buf504  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_624, slice_148, slice_149, neg_68, add_446, mul_625, add_447, permute_1065, clone_70, view_1494, view_1499, result_360, permute_1074, mm_679], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf511, (1218, 896), (896, 1), 0), primals_448, out=buf512)
            del primals_448
            buf507 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2635], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf505, buf507, 28672, stream=stream0)
            buf510 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2641], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf508, buf510, 28672, stream=stream0)
            buf509 = reinterpret_tensor(buf511, (1218, 896), (896, 1), 0); del buf511  # reuse
            # Topologically Sorted Source Nodes: [mm_678], Original ATen: [aten.mm]
            extern_kernels.mm(buf506, permute_1072, out=buf509)
            del permute_1072
            buf515 = buf471; del buf471  # reuse
            buf516 = reinterpret_tensor(buf482, (1218, 896), (896, 1), 0); del buf482  # reuse
            # Topologically Sorted Source Nodes: [view_1484, view_1486, add_448, view_1491, add_449, view_1493, add_450, view_1498, add_451, view_1500, add_452, mul_629, convert_element_type_2645, hidden_states_170, mul_630, mul_631, sum_29, pow_78, mul_632, mul_633, expand_91, div_14, pow_79, mul_634, mul_635, add_453, convert_element_type_2646, add_454, mul_636, view_1501], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf515, buf492, buf494, buf500, buf503, buf509, buf512, primals_447, add_221, rsqrt_34, buf516, 1218, 896, stream=stream0)
            del add_221
            del buf492
            del buf494
            del primals_447
            del rsqrt_34
            buf517 = reinterpret_tensor(buf508, (896, 32), (32, 1), 0); del buf508  # reuse
            # Topologically Sorted Source Nodes: [permute_1075, mm_680], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf516, (896, 1218), (1, 896), 0), mm_304, out=buf517)
            del mm_304
            buf518 = buf506; del buf506  # reuse
            # Topologically Sorted Source Nodes: [mm_681], Original ATen: [aten.mm]
            extern_kernels.mm(buf516, permute_1077, out=buf518)
            del permute_1077
            buf520 = reinterpret_tensor(buf461, (32, 4864), (4864, 1), 0); del buf461  # reuse
            # Topologically Sorted Source Nodes: [permute_1079, mm_682], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf518, (32, 1218), (1, 32), 0), view_821, out=buf520)
            del view_821
            buf519 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2651], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf517, buf519, 28672, stream=stream0)
            buf522 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2657], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf520, buf522, 155648, stream=stream0)
            buf521 = reinterpret_tensor(buf460, (1218, 4864), (4864, 1), 0); del buf460  # reuse
            # Topologically Sorted Source Nodes: [mm_683], Original ATen: [aten.mm]
            extern_kernels.mm(buf518, permute_1081, out=buf521)
            del permute_1081
            buf523 = reinterpret_tensor(buf451, (1218, 4864), (4864, 1), 0); del buf451  # reuse
            # Topologically Sorted Source Nodes: [view_1505, result_357, permute_1083, mm_684], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf515, (1218, 896), (896, 1), 0), primals_444, out=buf523)
            del primals_444
            buf524 = buf467; del buf467  # reuse
            buf531 = buf458; del buf458  # reuse
            buf533 = reinterpret_tensor(buf450, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf450  # reuse
            buf540 = reinterpret_tensor(buf448, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf448  # reuse
            # Topologically Sorted Source Nodes: [view_1504, view_1506, add_455, silu_16, mul_637, mul_638, mul_639, convert_element_type_2675, neg_69, exp_7, add_457, reciprocal_7, mul_640, mul_641, sub_7, mul_642, add_458, mul_643, convert_element_type_2677, mul_644], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf521, buf523, add_218, add_219, buf524, buf531, buf533, buf540, 5924352, stream=stream0)
            del add_218
            del add_219
            buf532 = buf516; del buf516  # reuse
            # Topologically Sorted Source Nodes: [view_1504, view_1506, add_455, silu_16, mul_637, view_1511, result_354, permute_1092, mm_689], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf531, (1218, 4864), (4864, 1), 0), primals_441, out=buf532)
            del primals_441
            buf541 = buf512; del buf512  # reuse
            # Topologically Sorted Source Nodes: [view_1504, view_1506, add_455, silu_16, mul_638, convert_element_type_2675, neg_69, exp_7, add_457, reciprocal_7, mul_640, mul_641, sub_7, mul_642, add_458, mul_643, convert_element_type_2677, view_1517, result_351, permute_1101, mm_694], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf540, (1218, 4864), (4864, 1), 0), primals_438, out=buf541)
            del primals_438
            buf526 = buf518; del buf518  # reuse
            # Topologically Sorted Source Nodes: [view_1504, view_1506, add_455, silu_16, mul_637, mul_639, view_1507, mm_686], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf524, (1218, 4864), (4864, 1), 0), permute_1086, out=buf526)
            del permute_1086
            buf525 = reinterpret_tensor(buf520, (4864, 32), (32, 1), 0); del buf520  # reuse
            # Topologically Sorted Source Nodes: [view_1504, view_1506, add_455, silu_16, mul_637, mul_639, view_1507, permute_1084, mm_685], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf524, (4864, 1218), (1, 4864), 0), mm_301, out=buf525)
            del mm_301
            buf535 = buf489; del buf489  # reuse
            # Topologically Sorted Source Nodes: [view_1504, view_1506, add_455, silu_16, mul_638, convert_element_type_2675, neg_69, exp_7, add_457, reciprocal_7, mul_640, mul_641, sub_7, mul_642, add_458, mul_643, convert_element_type_2677, mul_644, view_1513, mm_691], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf533, (1218, 4864), (4864, 1), 0), permute_1095, out=buf535)
            del permute_1095
            buf534 = buf452; del buf452  # reuse
            # Topologically Sorted Source Nodes: [view_1504, view_1506, add_455, silu_16, mul_638, convert_element_type_2675, neg_69, exp_7, add_457, reciprocal_7, mul_640, mul_641, sub_7, mul_642, add_458, mul_643, convert_element_type_2677, mul_644, view_1513, permute_1093, mm_690], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf533, (4864, 1218), (1, 4864), 0), mm_298, out=buf534)
            del mm_298
            buf528 = reinterpret_tensor(buf517, (32, 896), (896, 1), 0); del buf517  # reuse
            # Topologically Sorted Source Nodes: [permute_1088, mm_687], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf526, (32, 1218), (1, 32), 0), view_809, out=buf528)
            buf537 = reinterpret_tensor(buf505, (32, 896), (896, 1), 0); del buf505  # reuse
            # Topologically Sorted Source Nodes: [permute_1097, mm_692], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf535, (32, 1218), (1, 32), 0), view_809, out=buf537)
            del view_809
            buf530 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2671], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf528, buf530, 28672, stream=stream0)
            buf539 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2688], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf537, buf539, 28672, stream=stream0)
            buf527 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2665], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf525, buf527, 155648, stream=stream0)
            buf536 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2682], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf534, buf536, 155648, stream=stream0)
            buf529 = buf509; del buf509  # reuse
            # Topologically Sorted Source Nodes: [mm_688], Original ATen: [aten.mm]
            extern_kernels.mm(buf526, permute_1090, out=buf529)
            del permute_1090
            buf538 = buf503; del buf503  # reuse
            # Topologically Sorted Source Nodes: [mm_693], Original ATen: [aten.mm]
            extern_kernels.mm(buf535, permute_1099, out=buf538)
            del permute_1099
            buf544 = buf515; del buf515  # reuse
            buf545 = buf500; del buf500  # reuse
            # Topologically Sorted Source Nodes: [view_1510, view_1512, add_456, view_1516, add_459, view_1518, add_460, mul_645, convert_element_type_2692, hidden_states_166, mul_646, mul_647, sum_30, pow_80, mul_648, mul_649, expand_92, div_15, pow_81, mul_650, mul_651, add_461, convert_element_type_2693, add_462, mul_652, view_1519], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf544, buf529, buf532, buf538, buf541, primals_437, add_216, rsqrt_33, buf545, 1218, 896, stream=stream0)
            del add_216
            del buf529
            del primals_437
            del rsqrt_33
            buf546 = reinterpret_tensor(buf537, (896, 32), (32, 1), 0); del buf537  # reuse
            # Topologically Sorted Source Nodes: [permute_1102, mm_695], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf545, (896, 1218), (1, 896), 0), mm_295, out=buf546)
            del mm_295
            buf547 = buf535; del buf535  # reuse
            # Topologically Sorted Source Nodes: [mm_696], Original ATen: [aten.mm]
            extern_kernels.mm(buf545, permute_1104, out=buf547)
            del permute_1104
            buf549 = buf528; del buf528  # reuse
            # Topologically Sorted Source Nodes: [permute_1106, mm_697], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf547, (32, 1218), (1, 32), 0), view_803, out=buf549)
            del view_803
            buf548 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2698], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf546, buf548, 28672, stream=stream0)
            buf551 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2704], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf549, buf551, 28672, stream=stream0)
            buf550 = buf545; del buf545  # reuse
            # Topologically Sorted Source Nodes: [mm_698], Original ATen: [aten.mm]
            extern_kernels.mm(buf547, permute_1108, out=buf550)
            del permute_1108
            buf552 = buf541; del buf541  # reuse
            # Topologically Sorted Source Nodes: [view_1523, result_348, permute_1110, mm_699], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf544, (1218, 896), (896, 1), 0), primals_434, out=buf552)
            del primals_434
            buf553 = reinterpret_tensor(buf550, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf550  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1522, view_1524, add_463, view_1525, permute_1111, _scaled_dot_product_efficient_attention_backward_7], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf553, buf552, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1522, view_1524, add_463, view_1525, permute_1111, _scaled_dot_product_efficient_attention_backward_7], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf554 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf553, add_213, view_798, view_799, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_64, getitem_65, getitem_66, getitem_67, 0.0, [True, True, True, False], scale=0.125)
            del add_213
            del getitem_64
            del getitem_65
            del getitem_66
            del getitem_67
            del view_798
            del view_799
            buf555 = buf554[0]
            assert_size_stride(buf555, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf555, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf556 = buf554[1]
            assert_size_stride(buf556, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf556, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf558 = reinterpret_tensor(buf502, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf502  # reuse
            # Topologically Sorted Source Nodes: [view_1527, sum_32], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf556, buf558, 155904, stream=stream0)
            buf557 = buf554[2]
            assert_size_stride(buf557, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf557, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf554
            buf559 = buf486; del buf486  # reuse
            buf560 = reinterpret_tensor(buf495, (1218, 128), (128, 1), 0); del buf495  # reuse
            # Topologically Sorted Source Nodes: [view_1526, sum_31, squeeze_14, permute_1112, clone_71, view_1528, mul_657, view_1529], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf557, buf559, buf560, 155904, stream=stream0)
            buf561 = buf496; del buf496  # reuse
            # Topologically Sorted Source Nodes: [permute_1113, mm_700], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf560, (128, 1218), (1, 128), 0), mm_292, out=buf561)
            del mm_292
            buf562 = buf547; del buf547  # reuse
            # Topologically Sorted Source Nodes: [mm_701], Original ATen: [aten.mm]
            extern_kernels.mm(buf560, permute_1115, out=buf562)
            del permute_1115
            buf563 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2712], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf561, buf563, 4096, stream=stream0)
            buf564 = buf549; del buf549  # reuse
            # Topologically Sorted Source Nodes: [permute_1117, mm_702], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf562, (32, 1218), (1, 32), 0), view_779, out=buf564)
            buf566 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2718], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf564, buf566, 28672, stream=stream0)
            buf568 = reinterpret_tensor(buf560, (1, 1218, 128), (155904, 128, 1), 0); del buf560  # reuse
            buf575 = reinterpret_tensor(buf485, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf485  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1527, sum_32, squeeze_15, mul_653, slice_150, slice_151, neg_70, add_464, mul_654, add_465, permute_1122, clone_72, view_1535, mul_658], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf558, mm_default, buf568, buf575, 1218, 128, stream=stream0)
            buf569 = buf561; del buf561  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1527, sum_32, squeeze_15, mul_653, slice_150, slice_151, neg_70, add_464, mul_654, add_465, permute_1122, clone_72, view_1535, mul_658, view_1536, permute_1123, mm_705], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf568, (128, 1218), (1, 128), 0), mm_290, out=buf569)
            del mm_290
            buf570 = buf526; del buf526  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1527, sum_32, squeeze_15, mul_653, slice_150, slice_151, neg_70, add_464, mul_654, add_465, permute_1122, clone_72, view_1535, mul_658, view_1536, mm_706], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf568, (1218, 128), (128, 1), 0), permute_1125, out=buf570)
            del permute_1125
            buf571 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2726], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf569, buf571, 4096, stream=stream0)
            buf572 = buf564; del buf564  # reuse
            # Topologically Sorted Source Nodes: [permute_1127, mm_707], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf570, (32, 1218), (1, 32), 0), view_779, out=buf572)
            buf574 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2732], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf572, buf574, 28672, stream=stream0)
            buf567 = reinterpret_tensor(buf557, (1218, 896), (896, 1), 0); del buf557  # reuse
            # Topologically Sorted Source Nodes: [view_1526, sum_31, squeeze_14, permute_1112, clone_71, view_1528, view_1533, result_345, permute_1121, mm_704], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf559, (1218, 128), (128, 1), 0), primals_430, out=buf567)
            del primals_430
            buf576 = reinterpret_tensor(buf556, (1218, 896), (896, 1), 0); del buf556  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1527, sum_32, squeeze_15, mul_653, slice_150, slice_151, neg_70, add_464, mul_654, add_465, permute_1122, clone_72, view_1535, view_1540, result_342, permute_1131, mm_709], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf575, (1218, 128), (128, 1), 0), primals_426, out=buf576)
            del primals_426
            buf565 = reinterpret_tensor(buf553, (1218, 896), (896, 1), 0); del buf553  # reuse
            # Topologically Sorted Source Nodes: [mm_703], Original ATen: [aten.mm]
            extern_kernels.mm(buf562, permute_1119, out=buf565)
            del permute_1119
            buf573 = buf552; del buf552  # reuse
            # Topologically Sorted Source Nodes: [mm_708], Original ATen: [aten.mm]
            extern_kernels.mm(buf570, permute_1129, out=buf573)
            del permute_1129
            buf577 = reinterpret_tensor(buf538, (1, 1218, 896), (1091328, 896, 1), 0); del buf538  # reuse
            buf584 = reinterpret_tensor(buf532, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf532  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_655, slice_152, slice_153, neg_71, add_466, mul_656, add_467, permute_1132, clone_73, view_1542, mul_659], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf555, mm_default, buf577, buf584, 1091328, stream=stream0)
            buf578 = reinterpret_tensor(buf572, (896, 32), (32, 1), 0); del buf572  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_655, slice_152, slice_153, neg_71, add_466, mul_656, add_467, permute_1132, clone_73, view_1542, mul_659, view_1543, permute_1133, mm_710], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf577, (896, 1218), (1, 896), 0), mm_288, out=buf578)
            del mm_288
            buf579 = buf570; del buf570  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_655, slice_152, slice_153, neg_71, add_466, mul_656, add_467, permute_1132, clone_73, view_1542, mul_659, view_1543, mm_711], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf577, (1218, 896), (896, 1), 0), permute_1135, out=buf579)
            del permute_1135
            buf581 = reinterpret_tensor(buf546, (32, 896), (896, 1), 0); del buf546  # reuse
            # Topologically Sorted Source Nodes: [permute_1137, mm_712], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf579, (32, 1218), (1, 32), 0), view_779, out=buf581)
            del view_779
            buf585 = reinterpret_tensor(buf577, (1218, 896), (896, 1), 0); del buf577  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_655, slice_152, slice_153, neg_71, add_466, mul_656, add_467, permute_1132, clone_73, view_1542, view_1547, result_339, permute_1141, mm_714], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf584, (1218, 896), (896, 1), 0), primals_422, out=buf585)
            del primals_422
            buf580 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2740], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf578, buf580, 28672, stream=stream0)
            buf583 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2746], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf581, buf583, 28672, stream=stream0)
            buf582 = reinterpret_tensor(buf584, (1218, 896), (896, 1), 0); del buf584  # reuse
            # Topologically Sorted Source Nodes: [mm_713], Original ATen: [aten.mm]
            extern_kernels.mm(buf579, permute_1139, out=buf582)
            del permute_1139
            buf588 = buf544; del buf544  # reuse
            buf589 = reinterpret_tensor(buf555, (1218, 896), (896, 1), 0); del buf555  # reuse
            # Topologically Sorted Source Nodes: [view_1532, view_1534, add_468, view_1539, add_469, view_1541, add_470, view_1546, add_471, view_1548, add_472, mul_660, convert_element_type_2750, hidden_states_160, mul_661, mul_662, sum_33, pow_82, mul_663, mul_664, expand_93, div_16, pow_83, mul_665, mul_666, add_473, convert_element_type_2751, add_474, mul_667, view_1549], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf588, buf565, buf567, buf573, buf576, buf582, buf585, primals_421, add_208, rsqrt_32, buf589, 1218, 896, stream=stream0)
            del add_208
            del buf565
            del buf567
            del primals_421
            del rsqrt_32
            buf590 = reinterpret_tensor(buf581, (896, 32), (32, 1), 0); del buf581  # reuse
            # Topologically Sorted Source Nodes: [permute_1142, mm_715], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf589, (896, 1218), (1, 896), 0), mm_286, out=buf590)
            del mm_286
            buf591 = buf579; del buf579  # reuse
            # Topologically Sorted Source Nodes: [mm_716], Original ATen: [aten.mm]
            extern_kernels.mm(buf589, permute_1144, out=buf591)
            del permute_1144
            buf593 = reinterpret_tensor(buf534, (32, 4864), (4864, 1), 0); del buf534  # reuse
            # Topologically Sorted Source Nodes: [permute_1146, mm_717], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf591, (32, 1218), (1, 32), 0), view_773, out=buf593)
            del view_773
            buf592 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2756], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf590, buf592, 28672, stream=stream0)
            buf595 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2762], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf593, buf595, 155648, stream=stream0)
            buf594 = reinterpret_tensor(buf533, (1218, 4864), (4864, 1), 0); del buf533  # reuse
            # Topologically Sorted Source Nodes: [mm_718], Original ATen: [aten.mm]
            extern_kernels.mm(buf591, permute_1148, out=buf594)
            del permute_1148
            buf596 = reinterpret_tensor(buf524, (1218, 4864), (4864, 1), 0); del buf524  # reuse
            # Topologically Sorted Source Nodes: [view_1553, result_336, permute_1150, mm_719], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf588, (1218, 896), (896, 1), 0), primals_418, out=buf596)
            del primals_418
            buf597 = buf540; del buf540  # reuse
            buf604 = buf531; del buf531  # reuse
            buf606 = reinterpret_tensor(buf523, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf523  # reuse
            buf613 = reinterpret_tensor(buf521, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf521  # reuse
            # Topologically Sorted Source Nodes: [view_1552, view_1554, add_475, silu_15, mul_668, mul_669, mul_670, convert_element_type_2780, neg_72, exp_8, add_477, reciprocal_8, mul_671, mul_672, sub_8, mul_673, add_478, mul_674, convert_element_type_2782, mul_675], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf594, buf596, add_205, add_206, buf597, buf604, buf606, buf613, 5924352, stream=stream0)
            del add_205
            del add_206
            buf605 = buf589; del buf589  # reuse
            # Topologically Sorted Source Nodes: [view_1552, view_1554, add_475, silu_15, mul_668, view_1559, result_333, permute_1159, mm_724], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf604, (1218, 4864), (4864, 1), 0), primals_415, out=buf605)
            del primals_415
            buf614 = buf585; del buf585  # reuse
            # Topologically Sorted Source Nodes: [view_1552, view_1554, add_475, silu_15, mul_669, convert_element_type_2780, neg_72, exp_8, add_477, reciprocal_8, mul_671, mul_672, sub_8, mul_673, add_478, mul_674, convert_element_type_2782, view_1565, result_330, permute_1168, mm_729], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf613, (1218, 4864), (4864, 1), 0), primals_412, out=buf614)
            del primals_412
            buf599 = buf591; del buf591  # reuse
            # Topologically Sorted Source Nodes: [view_1552, view_1554, add_475, silu_15, mul_668, mul_670, view_1555, mm_721], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf597, (1218, 4864), (4864, 1), 0), permute_1153, out=buf599)
            del permute_1153
            buf598 = reinterpret_tensor(buf593, (4864, 32), (32, 1), 0); del buf593  # reuse
            # Topologically Sorted Source Nodes: [view_1552, view_1554, add_475, silu_15, mul_668, mul_670, view_1555, permute_1151, mm_720], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf597, (4864, 1218), (1, 4864), 0), mm_283, out=buf598)
            del mm_283
            buf608 = buf562; del buf562  # reuse
            # Topologically Sorted Source Nodes: [view_1552, view_1554, add_475, silu_15, mul_669, convert_element_type_2780, neg_72, exp_8, add_477, reciprocal_8, mul_671, mul_672, sub_8, mul_673, add_478, mul_674, convert_element_type_2782, mul_675, view_1561, mm_726], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf606, (1218, 4864), (4864, 1), 0), permute_1162, out=buf608)
            del permute_1162
            buf607 = buf525; del buf525  # reuse
            # Topologically Sorted Source Nodes: [view_1552, view_1554, add_475, silu_15, mul_669, convert_element_type_2780, neg_72, exp_8, add_477, reciprocal_8, mul_671, mul_672, sub_8, mul_673, add_478, mul_674, convert_element_type_2782, mul_675, view_1561, permute_1160, mm_725], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf606, (4864, 1218), (1, 4864), 0), mm_280, out=buf607)
            del mm_280
            buf601 = reinterpret_tensor(buf590, (32, 896), (896, 1), 0); del buf590  # reuse
            # Topologically Sorted Source Nodes: [permute_1155, mm_722], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf599, (32, 1218), (1, 32), 0), view_761, out=buf601)
            buf610 = reinterpret_tensor(buf578, (32, 896), (896, 1), 0); del buf578  # reuse
            # Topologically Sorted Source Nodes: [permute_1164, mm_727], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf608, (32, 1218), (1, 32), 0), view_761, out=buf610)
            del view_761
            buf603 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2776], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf601, buf603, 28672, stream=stream0)
            buf612 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2793], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf610, buf612, 28672, stream=stream0)
            buf600 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2770], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf598, buf600, 155648, stream=stream0)
            buf609 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2787], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf607, buf609, 155648, stream=stream0)
            buf602 = buf582; del buf582  # reuse
            # Topologically Sorted Source Nodes: [mm_723], Original ATen: [aten.mm]
            extern_kernels.mm(buf599, permute_1157, out=buf602)
            del permute_1157
            buf611 = buf576; del buf576  # reuse
            # Topologically Sorted Source Nodes: [mm_728], Original ATen: [aten.mm]
            extern_kernels.mm(buf608, permute_1166, out=buf611)
            del permute_1166
            buf617 = buf588; del buf588  # reuse
            buf618 = buf573; del buf573  # reuse
            # Topologically Sorted Source Nodes: [view_1558, view_1560, add_476, view_1564, add_479, view_1566, add_480, mul_676, convert_element_type_2797, hidden_states_156, mul_677, mul_678, sum_34, pow_84, mul_679, mul_680, expand_94, div_17, pow_85, mul_681, mul_682, add_481, convert_element_type_2798, add_482, mul_683, view_1567], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf617, buf602, buf605, buf611, buf614, primals_411, add_203, rsqrt_31, buf618, 1218, 896, stream=stream0)
            del add_203
            del buf602
            del primals_411
            del rsqrt_31
            buf619 = reinterpret_tensor(buf610, (896, 32), (32, 1), 0); del buf610  # reuse
            # Topologically Sorted Source Nodes: [permute_1169, mm_730], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf618, (896, 1218), (1, 896), 0), mm_277, out=buf619)
            del mm_277
            buf620 = buf608; del buf608  # reuse
            # Topologically Sorted Source Nodes: [mm_731], Original ATen: [aten.mm]
            extern_kernels.mm(buf618, permute_1171, out=buf620)
            del permute_1171
            buf622 = buf601; del buf601  # reuse
            # Topologically Sorted Source Nodes: [permute_1173, mm_732], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf620, (32, 1218), (1, 32), 0), view_755, out=buf622)
            del view_755
            buf621 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2803], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf619, buf621, 28672, stream=stream0)
            buf624 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2809], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf622, buf624, 28672, stream=stream0)
            buf623 = buf618; del buf618  # reuse
            # Topologically Sorted Source Nodes: [mm_733], Original ATen: [aten.mm]
            extern_kernels.mm(buf620, permute_1175, out=buf623)
            del permute_1175
            buf625 = buf614; del buf614  # reuse
            # Topologically Sorted Source Nodes: [view_1571, result_327, permute_1177, mm_734], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf617, (1218, 896), (896, 1), 0), primals_408, out=buf625)
            del primals_408
            buf626 = reinterpret_tensor(buf623, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf623  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1570, view_1572, add_483, view_1573, permute_1178, _scaled_dot_product_efficient_attention_backward_8], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf626, buf625, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1570, view_1572, add_483, view_1573, permute_1178, _scaled_dot_product_efficient_attention_backward_8], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf627 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf626, add_200, view_750, view_751, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_60, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False], scale=0.125)
            del add_200
            del getitem_60
            del getitem_61
            del getitem_62
            del getitem_63
            del view_750
            del view_751
            buf628 = buf627[0]
            assert_size_stride(buf628, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf628, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf629 = buf627[1]
            assert_size_stride(buf629, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf629, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf631 = reinterpret_tensor(buf575, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf575  # reuse
            # Topologically Sorted Source Nodes: [view_1575, sum_36], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf629, buf631, 155904, stream=stream0)
            buf630 = buf627[2]
            assert_size_stride(buf630, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf630, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf627
            buf632 = buf559; del buf559  # reuse
            buf633 = reinterpret_tensor(buf568, (1218, 128), (128, 1), 0); del buf568  # reuse
            # Topologically Sorted Source Nodes: [view_1574, sum_35, squeeze_16, permute_1179, clone_74, view_1576, mul_688, view_1577], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf630, buf632, buf633, 155904, stream=stream0)
            buf634 = buf569; del buf569  # reuse
            # Topologically Sorted Source Nodes: [permute_1180, mm_735], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf633, (128, 1218), (1, 128), 0), mm_274, out=buf634)
            del mm_274
            buf635 = buf620; del buf620  # reuse
            # Topologically Sorted Source Nodes: [mm_736], Original ATen: [aten.mm]
            extern_kernels.mm(buf633, permute_1182, out=buf635)
            del permute_1182
            buf636 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2817], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf634, buf636, 4096, stream=stream0)
            buf637 = buf622; del buf622  # reuse
            # Topologically Sorted Source Nodes: [permute_1184, mm_737], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf635, (32, 1218), (1, 32), 0), view_731, out=buf637)
            buf639 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2823], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf637, buf639, 28672, stream=stream0)
            buf641 = reinterpret_tensor(buf633, (1, 1218, 128), (155904, 128, 1), 0); del buf633  # reuse
            buf648 = reinterpret_tensor(buf558, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf558  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1575, sum_36, squeeze_17, mul_684, slice_154, slice_155, neg_73, add_484, mul_685, add_485, permute_1189, clone_75, view_1583, mul_689], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf631, mm_default, buf641, buf648, 1218, 128, stream=stream0)
            buf642 = buf634; del buf634  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1575, sum_36, squeeze_17, mul_684, slice_154, slice_155, neg_73, add_484, mul_685, add_485, permute_1189, clone_75, view_1583, mul_689, view_1584, permute_1190, mm_740], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf641, (128, 1218), (1, 128), 0), mm_272, out=buf642)
            del mm_272
            buf643 = buf599; del buf599  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1575, sum_36, squeeze_17, mul_684, slice_154, slice_155, neg_73, add_484, mul_685, add_485, permute_1189, clone_75, view_1583, mul_689, view_1584, mm_741], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf641, (1218, 128), (128, 1), 0), permute_1192, out=buf643)
            del permute_1192
            buf644 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2831], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf642, buf644, 4096, stream=stream0)
            buf645 = buf637; del buf637  # reuse
            # Topologically Sorted Source Nodes: [permute_1194, mm_742], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf643, (32, 1218), (1, 32), 0), view_731, out=buf645)
            buf647 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2837], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf645, buf647, 28672, stream=stream0)
            buf640 = reinterpret_tensor(buf630, (1218, 896), (896, 1), 0); del buf630  # reuse
            # Topologically Sorted Source Nodes: [view_1574, sum_35, squeeze_16, permute_1179, clone_74, view_1576, view_1581, result_324, permute_1188, mm_739], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf632, (1218, 128), (128, 1), 0), primals_404, out=buf640)
            del primals_404
            buf649 = reinterpret_tensor(buf629, (1218, 896), (896, 1), 0); del buf629  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1575, sum_36, squeeze_17, mul_684, slice_154, slice_155, neg_73, add_484, mul_685, add_485, permute_1189, clone_75, view_1583, view_1588, result_321, permute_1198, mm_744], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf648, (1218, 128), (128, 1), 0), primals_400, out=buf649)
            del primals_400
            buf638 = reinterpret_tensor(buf626, (1218, 896), (896, 1), 0); del buf626  # reuse
            # Topologically Sorted Source Nodes: [mm_738], Original ATen: [aten.mm]
            extern_kernels.mm(buf635, permute_1186, out=buf638)
            del permute_1186
            buf646 = buf625; del buf625  # reuse
            # Topologically Sorted Source Nodes: [mm_743], Original ATen: [aten.mm]
            extern_kernels.mm(buf643, permute_1196, out=buf646)
            del permute_1196
            buf650 = reinterpret_tensor(buf611, (1, 1218, 896), (1091328, 896, 1), 0); del buf611  # reuse
            buf657 = reinterpret_tensor(buf605, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf605  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_686, slice_156, slice_157, neg_74, add_486, mul_687, add_487, permute_1199, clone_76, view_1590, mul_690], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf628, mm_default, buf650, buf657, 1091328, stream=stream0)
            buf651 = reinterpret_tensor(buf645, (896, 32), (32, 1), 0); del buf645  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_686, slice_156, slice_157, neg_74, add_486, mul_687, add_487, permute_1199, clone_76, view_1590, mul_690, view_1591, permute_1200, mm_745], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf650, (896, 1218), (1, 896), 0), mm_270, out=buf651)
            del mm_270
            buf652 = buf643; del buf643  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_686, slice_156, slice_157, neg_74, add_486, mul_687, add_487, permute_1199, clone_76, view_1590, mul_690, view_1591, mm_746], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf650, (1218, 896), (896, 1), 0), permute_1202, out=buf652)
            del permute_1202
            buf654 = reinterpret_tensor(buf619, (32, 896), (896, 1), 0); del buf619  # reuse
            # Topologically Sorted Source Nodes: [permute_1204, mm_747], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf652, (32, 1218), (1, 32), 0), view_731, out=buf654)
            del view_731
            buf658 = reinterpret_tensor(buf650, (1218, 896), (896, 1), 0); del buf650  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_686, slice_156, slice_157, neg_74, add_486, mul_687, add_487, permute_1199, clone_76, view_1590, view_1595, result_318, permute_1208, mm_749], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf657, (1218, 896), (896, 1), 0), primals_396, out=buf658)
            del primals_396
            buf653 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2845], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf651, buf653, 28672, stream=stream0)
            buf656 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2851], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf654, buf656, 28672, stream=stream0)
            buf655 = reinterpret_tensor(buf657, (1218, 896), (896, 1), 0); del buf657  # reuse
            # Topologically Sorted Source Nodes: [mm_748], Original ATen: [aten.mm]
            extern_kernels.mm(buf652, permute_1206, out=buf655)
            del permute_1206
            buf661 = buf617; del buf617  # reuse
            buf662 = reinterpret_tensor(buf628, (1218, 896), (896, 1), 0); del buf628  # reuse
            # Topologically Sorted Source Nodes: [view_1580, view_1582, add_488, view_1587, add_489, view_1589, add_490, view_1594, add_491, view_1596, add_492, mul_691, convert_element_type_2855, hidden_states_150, mul_692, mul_693, sum_37, pow_86, mul_694, mul_695, expand_95, div_18, pow_87, mul_696, mul_697, add_493, convert_element_type_2856, add_494, mul_698, view_1597], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf661, buf638, buf640, buf646, buf649, buf655, buf658, primals_395, add_195, rsqrt_30, buf662, 1218, 896, stream=stream0)
            del add_195
            del buf638
            del buf640
            del primals_395
            del rsqrt_30
            buf663 = reinterpret_tensor(buf654, (896, 32), (32, 1), 0); del buf654  # reuse
            # Topologically Sorted Source Nodes: [permute_1209, mm_750], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf662, (896, 1218), (1, 896), 0), mm_268, out=buf663)
            del mm_268
            buf664 = buf652; del buf652  # reuse
            # Topologically Sorted Source Nodes: [mm_751], Original ATen: [aten.mm]
            extern_kernels.mm(buf662, permute_1211, out=buf664)
            del permute_1211
            buf666 = reinterpret_tensor(buf607, (32, 4864), (4864, 1), 0); del buf607  # reuse
            # Topologically Sorted Source Nodes: [permute_1213, mm_752], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf664, (32, 1218), (1, 32), 0), view_725, out=buf666)
            del view_725
            buf665 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2861], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf663, buf665, 28672, stream=stream0)
            buf668 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2867], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf666, buf668, 155648, stream=stream0)
            buf667 = reinterpret_tensor(buf606, (1218, 4864), (4864, 1), 0); del buf606  # reuse
            # Topologically Sorted Source Nodes: [mm_753], Original ATen: [aten.mm]
            extern_kernels.mm(buf664, permute_1215, out=buf667)
            del permute_1215
            buf669 = reinterpret_tensor(buf597, (1218, 4864), (4864, 1), 0); del buf597  # reuse
            # Topologically Sorted Source Nodes: [view_1601, result_315, permute_1217, mm_754], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf661, (1218, 896), (896, 1), 0), primals_392, out=buf669)
            del primals_392
            buf670 = buf613; del buf613  # reuse
            buf677 = buf604; del buf604  # reuse
            buf679 = reinterpret_tensor(buf596, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf596  # reuse
            buf686 = reinterpret_tensor(buf594, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf594  # reuse
            # Topologically Sorted Source Nodes: [view_1600, view_1602, add_495, silu_14, mul_699, mul_700, mul_701, convert_element_type_2885, neg_75, exp_9, add_497, reciprocal_9, mul_702, mul_703, sub_9, mul_704, add_498, mul_705, convert_element_type_2887, mul_706], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf667, buf669, add_192, add_193, buf670, buf677, buf679, buf686, 5924352, stream=stream0)
            del add_192
            del add_193
            buf678 = buf662; del buf662  # reuse
            # Topologically Sorted Source Nodes: [view_1600, view_1602, add_495, silu_14, mul_699, view_1607, result_312, permute_1226, mm_759], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf677, (1218, 4864), (4864, 1), 0), primals_389, out=buf678)
            del primals_389
            buf687 = buf658; del buf658  # reuse
            # Topologically Sorted Source Nodes: [view_1600, view_1602, add_495, silu_14, mul_700, convert_element_type_2885, neg_75, exp_9, add_497, reciprocal_9, mul_702, mul_703, sub_9, mul_704, add_498, mul_705, convert_element_type_2887, view_1613, result_309, permute_1235, mm_764], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf686, (1218, 4864), (4864, 1), 0), primals_386, out=buf687)
            del primals_386
            buf672 = buf664; del buf664  # reuse
            # Topologically Sorted Source Nodes: [view_1600, view_1602, add_495, silu_14, mul_699, mul_701, view_1603, mm_756], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf670, (1218, 4864), (4864, 1), 0), permute_1220, out=buf672)
            del permute_1220
            buf671 = reinterpret_tensor(buf666, (4864, 32), (32, 1), 0); del buf666  # reuse
            # Topologically Sorted Source Nodes: [view_1600, view_1602, add_495, silu_14, mul_699, mul_701, view_1603, permute_1218, mm_755], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf670, (4864, 1218), (1, 4864), 0), mm_265, out=buf671)
            del mm_265
            buf681 = buf635; del buf635  # reuse
            # Topologically Sorted Source Nodes: [view_1600, view_1602, add_495, silu_14, mul_700, convert_element_type_2885, neg_75, exp_9, add_497, reciprocal_9, mul_702, mul_703, sub_9, mul_704, add_498, mul_705, convert_element_type_2887, mul_706, view_1609, mm_761], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf679, (1218, 4864), (4864, 1), 0), permute_1229, out=buf681)
            del permute_1229
            buf680 = buf598; del buf598  # reuse
            # Topologically Sorted Source Nodes: [view_1600, view_1602, add_495, silu_14, mul_700, convert_element_type_2885, neg_75, exp_9, add_497, reciprocal_9, mul_702, mul_703, sub_9, mul_704, add_498, mul_705, convert_element_type_2887, mul_706, view_1609, permute_1227, mm_760], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf679, (4864, 1218), (1, 4864), 0), mm_262, out=buf680)
            del mm_262
            buf674 = reinterpret_tensor(buf663, (32, 896), (896, 1), 0); del buf663  # reuse
            # Topologically Sorted Source Nodes: [permute_1222, mm_757], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf672, (32, 1218), (1, 32), 0), view_713, out=buf674)
            buf683 = reinterpret_tensor(buf651, (32, 896), (896, 1), 0); del buf651  # reuse
            # Topologically Sorted Source Nodes: [permute_1231, mm_762], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf681, (32, 1218), (1, 32), 0), view_713, out=buf683)
            del view_713
            buf676 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2881], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf674, buf676, 28672, stream=stream0)
            buf685 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2898], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf683, buf685, 28672, stream=stream0)
            buf673 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2875], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf671, buf673, 155648, stream=stream0)
            buf682 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2892], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf680, buf682, 155648, stream=stream0)
            buf675 = buf655; del buf655  # reuse
            # Topologically Sorted Source Nodes: [mm_758], Original ATen: [aten.mm]
            extern_kernels.mm(buf672, permute_1224, out=buf675)
            del permute_1224
            buf684 = buf649; del buf649  # reuse
            # Topologically Sorted Source Nodes: [mm_763], Original ATen: [aten.mm]
            extern_kernels.mm(buf681, permute_1233, out=buf684)
            del permute_1233
            buf690 = buf661; del buf661  # reuse
            buf691 = buf646; del buf646  # reuse
            # Topologically Sorted Source Nodes: [view_1606, view_1608, add_496, view_1612, add_499, view_1614, add_500, mul_707, convert_element_type_2902, hidden_states_146, mul_708, mul_709, sum_38, pow_88, mul_710, mul_711, expand_96, div_19, pow_89, mul_712, mul_713, add_501, convert_element_type_2903, add_502, mul_714, view_1615], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf690, buf675, buf678, buf684, buf687, primals_385, add_190, rsqrt_29, buf691, 1218, 896, stream=stream0)
            del add_190
            del buf675
            del primals_385
            del rsqrt_29
            buf692 = reinterpret_tensor(buf683, (896, 32), (32, 1), 0); del buf683  # reuse
            # Topologically Sorted Source Nodes: [permute_1236, mm_765], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf691, (896, 1218), (1, 896), 0), mm_259, out=buf692)
            del mm_259
            buf693 = buf681; del buf681  # reuse
            # Topologically Sorted Source Nodes: [mm_766], Original ATen: [aten.mm]
            extern_kernels.mm(buf691, permute_1238, out=buf693)
            del permute_1238
            buf695 = buf674; del buf674  # reuse
            # Topologically Sorted Source Nodes: [permute_1240, mm_767], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf693, (32, 1218), (1, 32), 0), view_707, out=buf695)
            del view_707
            buf694 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2908], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf692, buf694, 28672, stream=stream0)
            buf697 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2914], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf695, buf697, 28672, stream=stream0)
            buf696 = buf691; del buf691  # reuse
            # Topologically Sorted Source Nodes: [mm_768], Original ATen: [aten.mm]
            extern_kernels.mm(buf693, permute_1242, out=buf696)
            del permute_1242
            buf698 = buf687; del buf687  # reuse
            # Topologically Sorted Source Nodes: [view_1619, result_306, permute_1244, mm_769], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf690, (1218, 896), (896, 1), 0), primals_382, out=buf698)
            del primals_382
            buf699 = reinterpret_tensor(buf696, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf696  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1618, view_1620, add_503, view_1621, permute_1245, _scaled_dot_product_efficient_attention_backward_9], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf699, buf698, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1618, view_1620, add_503, view_1621, permute_1245, _scaled_dot_product_efficient_attention_backward_9], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf700 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf699, add_187, view_702, view_703, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_56, getitem_57, getitem_58, getitem_59, 0.0, [True, True, True, False], scale=0.125)
            del add_187
            del getitem_56
            del getitem_57
            del getitem_58
            del getitem_59
            del view_702
            del view_703
            buf701 = buf700[0]
            assert_size_stride(buf701, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf701, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf702 = buf700[1]
            assert_size_stride(buf702, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf702, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf704 = reinterpret_tensor(buf648, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf648  # reuse
            # Topologically Sorted Source Nodes: [view_1623, sum_40], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf702, buf704, 155904, stream=stream0)
            buf703 = buf700[2]
            assert_size_stride(buf703, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf703, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf700
            buf705 = buf632; del buf632  # reuse
            buf706 = reinterpret_tensor(buf641, (1218, 128), (128, 1), 0); del buf641  # reuse
            # Topologically Sorted Source Nodes: [view_1622, sum_39, squeeze_18, permute_1246, clone_77, view_1624, mul_719, view_1625], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf703, buf705, buf706, 155904, stream=stream0)
            buf707 = buf642; del buf642  # reuse
            # Topologically Sorted Source Nodes: [permute_1247, mm_770], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf706, (128, 1218), (1, 128), 0), mm_256, out=buf707)
            del mm_256
            buf708 = buf693; del buf693  # reuse
            # Topologically Sorted Source Nodes: [mm_771], Original ATen: [aten.mm]
            extern_kernels.mm(buf706, permute_1249, out=buf708)
            del permute_1249
            buf709 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2922], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf707, buf709, 4096, stream=stream0)
            buf710 = buf695; del buf695  # reuse
            # Topologically Sorted Source Nodes: [permute_1251, mm_772], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf708, (32, 1218), (1, 32), 0), view_683, out=buf710)
            buf712 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2928], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf710, buf712, 28672, stream=stream0)
            buf714 = reinterpret_tensor(buf706, (1, 1218, 128), (155904, 128, 1), 0); del buf706  # reuse
            buf721 = reinterpret_tensor(buf631, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf631  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1623, sum_40, squeeze_19, mul_715, slice_158, slice_159, neg_76, add_504, mul_716, add_505, permute_1256, clone_78, view_1631, mul_720], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf704, mm_default, buf714, buf721, 1218, 128, stream=stream0)
            buf715 = buf707; del buf707  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1623, sum_40, squeeze_19, mul_715, slice_158, slice_159, neg_76, add_504, mul_716, add_505, permute_1256, clone_78, view_1631, mul_720, view_1632, permute_1257, mm_775], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf714, (128, 1218), (1, 128), 0), mm_254, out=buf715)
            del mm_254
            buf716 = buf672; del buf672  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1623, sum_40, squeeze_19, mul_715, slice_158, slice_159, neg_76, add_504, mul_716, add_505, permute_1256, clone_78, view_1631, mul_720, view_1632, mm_776], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf714, (1218, 128), (128, 1), 0), permute_1259, out=buf716)
            del permute_1259
            buf717 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2936], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf715, buf717, 4096, stream=stream0)
            buf718 = buf710; del buf710  # reuse
            # Topologically Sorted Source Nodes: [permute_1261, mm_777], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf716, (32, 1218), (1, 32), 0), view_683, out=buf718)
            buf720 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2942], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf718, buf720, 28672, stream=stream0)
            buf713 = reinterpret_tensor(buf703, (1218, 896), (896, 1), 0); del buf703  # reuse
            # Topologically Sorted Source Nodes: [view_1622, sum_39, squeeze_18, permute_1246, clone_77, view_1624, view_1629, result_303, permute_1255, mm_774], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf705, (1218, 128), (128, 1), 0), primals_378, out=buf713)
            del primals_378
            buf722 = reinterpret_tensor(buf702, (1218, 896), (896, 1), 0); del buf702  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1623, sum_40, squeeze_19, mul_715, slice_158, slice_159, neg_76, add_504, mul_716, add_505, permute_1256, clone_78, view_1631, view_1636, result_300, permute_1265, mm_779], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf721, (1218, 128), (128, 1), 0), primals_374, out=buf722)
            del primals_374
            buf711 = reinterpret_tensor(buf699, (1218, 896), (896, 1), 0); del buf699  # reuse
            # Topologically Sorted Source Nodes: [mm_773], Original ATen: [aten.mm]
            extern_kernels.mm(buf708, permute_1253, out=buf711)
            del permute_1253
            buf719 = buf698; del buf698  # reuse
            # Topologically Sorted Source Nodes: [mm_778], Original ATen: [aten.mm]
            extern_kernels.mm(buf716, permute_1263, out=buf719)
            del permute_1263
            buf723 = reinterpret_tensor(buf684, (1, 1218, 896), (1091328, 896, 1), 0); del buf684  # reuse
            buf730 = reinterpret_tensor(buf678, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf678  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_717, slice_160, slice_161, neg_77, add_506, mul_718, add_507, permute_1266, clone_79, view_1638, mul_721], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf701, mm_default, buf723, buf730, 1091328, stream=stream0)
            buf724 = reinterpret_tensor(buf718, (896, 32), (32, 1), 0); del buf718  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_717, slice_160, slice_161, neg_77, add_506, mul_718, add_507, permute_1266, clone_79, view_1638, mul_721, view_1639, permute_1267, mm_780], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf723, (896, 1218), (1, 896), 0), mm_252, out=buf724)
            del mm_252
            buf725 = buf716; del buf716  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_717, slice_160, slice_161, neg_77, add_506, mul_718, add_507, permute_1266, clone_79, view_1638, mul_721, view_1639, mm_781], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf723, (1218, 896), (896, 1), 0), permute_1269, out=buf725)
            del permute_1269
            buf727 = reinterpret_tensor(buf692, (32, 896), (896, 1), 0); del buf692  # reuse
            # Topologically Sorted Source Nodes: [permute_1271, mm_782], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf725, (32, 1218), (1, 32), 0), view_683, out=buf727)
            del view_683
            buf731 = reinterpret_tensor(buf723, (1218, 896), (896, 1), 0); del buf723  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_717, slice_160, slice_161, neg_77, add_506, mul_718, add_507, permute_1266, clone_79, view_1638, view_1643, result_297, permute_1275, mm_784], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf730, (1218, 896), (896, 1), 0), primals_370, out=buf731)
            del primals_370
            buf726 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2950], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf724, buf726, 28672, stream=stream0)
            buf729 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2956], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf727, buf729, 28672, stream=stream0)
            buf728 = reinterpret_tensor(buf730, (1218, 896), (896, 1), 0); del buf730  # reuse
            # Topologically Sorted Source Nodes: [mm_783], Original ATen: [aten.mm]
            extern_kernels.mm(buf725, permute_1273, out=buf728)
            del permute_1273
            buf734 = buf690; del buf690  # reuse
            buf735 = reinterpret_tensor(buf701, (1218, 896), (896, 1), 0); del buf701  # reuse
            # Topologically Sorted Source Nodes: [view_1628, view_1630, add_508, view_1635, add_509, view_1637, add_510, view_1642, add_511, view_1644, add_512, mul_722, convert_element_type_2960, hidden_states_140, mul_723, mul_724, sum_41, pow_90, mul_725, mul_726, expand_97, div_20, pow_91, mul_727, mul_728, add_513, convert_element_type_2961, add_514, mul_729, view_1645], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf734, buf711, buf713, buf719, buf722, buf728, buf731, primals_369, add_182, rsqrt_28, buf735, 1218, 896, stream=stream0)
            del add_182
            del buf711
            del buf713
            del primals_369
            del rsqrt_28
            buf736 = reinterpret_tensor(buf727, (896, 32), (32, 1), 0); del buf727  # reuse
            # Topologically Sorted Source Nodes: [permute_1276, mm_785], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf735, (896, 1218), (1, 896), 0), mm_250, out=buf736)
            del mm_250
            buf737 = buf725; del buf725  # reuse
            # Topologically Sorted Source Nodes: [mm_786], Original ATen: [aten.mm]
            extern_kernels.mm(buf735, permute_1278, out=buf737)
            del permute_1278
            buf739 = reinterpret_tensor(buf680, (32, 4864), (4864, 1), 0); del buf680  # reuse
            # Topologically Sorted Source Nodes: [permute_1280, mm_787], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf737, (32, 1218), (1, 32), 0), view_677, out=buf739)
            del view_677
            buf738 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2966], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf736, buf738, 28672, stream=stream0)
            buf741 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2972], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf739, buf741, 155648, stream=stream0)
            buf740 = reinterpret_tensor(buf679, (1218, 4864), (4864, 1), 0); del buf679  # reuse
            # Topologically Sorted Source Nodes: [mm_788], Original ATen: [aten.mm]
            extern_kernels.mm(buf737, permute_1282, out=buf740)
            del permute_1282
            buf742 = reinterpret_tensor(buf670, (1218, 4864), (4864, 1), 0); del buf670  # reuse
            # Topologically Sorted Source Nodes: [view_1649, result_294, permute_1284, mm_789], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf734, (1218, 896), (896, 1), 0), primals_366, out=buf742)
            del primals_366
            buf743 = buf686; del buf686  # reuse
            buf750 = buf677; del buf677  # reuse
            buf752 = reinterpret_tensor(buf669, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf669  # reuse
            buf759 = reinterpret_tensor(buf667, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf667  # reuse
            # Topologically Sorted Source Nodes: [view_1648, view_1650, add_515, silu_13, mul_730, mul_731, mul_732, convert_element_type_2990, neg_78, exp_10, add_517, reciprocal_10, mul_733, mul_734, sub_10, mul_735, add_518, mul_736, convert_element_type_2992, mul_737], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf740, buf742, add_179, add_180, buf743, buf750, buf752, buf759, 5924352, stream=stream0)
            del add_179
            del add_180
            buf751 = buf735; del buf735  # reuse
            # Topologically Sorted Source Nodes: [view_1648, view_1650, add_515, silu_13, mul_730, view_1655, result_291, permute_1293, mm_794], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf750, (1218, 4864), (4864, 1), 0), primals_363, out=buf751)
            del primals_363
            buf760 = buf731; del buf731  # reuse
            # Topologically Sorted Source Nodes: [view_1648, view_1650, add_515, silu_13, mul_731, convert_element_type_2990, neg_78, exp_10, add_517, reciprocal_10, mul_733, mul_734, sub_10, mul_735, add_518, mul_736, convert_element_type_2992, view_1661, result_288, permute_1302, mm_799], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf759, (1218, 4864), (4864, 1), 0), primals_360, out=buf760)
            del primals_360
            buf745 = buf737; del buf737  # reuse
            # Topologically Sorted Source Nodes: [view_1648, view_1650, add_515, silu_13, mul_730, mul_732, view_1651, mm_791], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf743, (1218, 4864), (4864, 1), 0), permute_1287, out=buf745)
            del permute_1287
            buf744 = reinterpret_tensor(buf739, (4864, 32), (32, 1), 0); del buf739  # reuse
            # Topologically Sorted Source Nodes: [view_1648, view_1650, add_515, silu_13, mul_730, mul_732, view_1651, permute_1285, mm_790], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf743, (4864, 1218), (1, 4864), 0), mm_247, out=buf744)
            del mm_247
            buf754 = buf708; del buf708  # reuse
            # Topologically Sorted Source Nodes: [view_1648, view_1650, add_515, silu_13, mul_731, convert_element_type_2990, neg_78, exp_10, add_517, reciprocal_10, mul_733, mul_734, sub_10, mul_735, add_518, mul_736, convert_element_type_2992, mul_737, view_1657, mm_796], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf752, (1218, 4864), (4864, 1), 0), permute_1296, out=buf754)
            del permute_1296
            buf753 = buf671; del buf671  # reuse
            # Topologically Sorted Source Nodes: [view_1648, view_1650, add_515, silu_13, mul_731, convert_element_type_2990, neg_78, exp_10, add_517, reciprocal_10, mul_733, mul_734, sub_10, mul_735, add_518, mul_736, convert_element_type_2992, mul_737, view_1657, permute_1294, mm_795], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf752, (4864, 1218), (1, 4864), 0), mm_244, out=buf753)
            del mm_244
            buf747 = reinterpret_tensor(buf736, (32, 896), (896, 1), 0); del buf736  # reuse
            # Topologically Sorted Source Nodes: [permute_1289, mm_792], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf745, (32, 1218), (1, 32), 0), view_665, out=buf747)
            buf756 = reinterpret_tensor(buf724, (32, 896), (896, 1), 0); del buf724  # reuse
            # Topologically Sorted Source Nodes: [permute_1298, mm_797], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf754, (32, 1218), (1, 32), 0), view_665, out=buf756)
            del view_665
            buf749 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2986], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf747, buf749, 28672, stream=stream0)
            buf758 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3003], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf756, buf758, 28672, stream=stream0)
            buf746 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2980], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf744, buf746, 155648, stream=stream0)
            buf755 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_2997], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf753, buf755, 155648, stream=stream0)
            buf748 = buf728; del buf728  # reuse
            # Topologically Sorted Source Nodes: [mm_793], Original ATen: [aten.mm]
            extern_kernels.mm(buf745, permute_1291, out=buf748)
            del permute_1291
            buf757 = buf722; del buf722  # reuse
            # Topologically Sorted Source Nodes: [mm_798], Original ATen: [aten.mm]
            extern_kernels.mm(buf754, permute_1300, out=buf757)
            del permute_1300
            buf763 = buf734; del buf734  # reuse
            buf764 = buf719; del buf719  # reuse
            # Topologically Sorted Source Nodes: [view_1654, view_1656, add_516, view_1660, add_519, view_1662, add_520, mul_738, convert_element_type_3007, hidden_states_136, mul_739, mul_740, sum_42, pow_92, mul_741, mul_742, expand_98, div_21, pow_93, mul_743, mul_744, add_521, convert_element_type_3008, add_522, mul_745, view_1663], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf763, buf748, buf751, buf757, buf760, primals_359, add_177, rsqrt_27, buf764, 1218, 896, stream=stream0)
            del add_177
            del buf748
            del primals_359
            del rsqrt_27
            buf765 = reinterpret_tensor(buf756, (896, 32), (32, 1), 0); del buf756  # reuse
            # Topologically Sorted Source Nodes: [permute_1303, mm_800], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf764, (896, 1218), (1, 896), 0), mm_241, out=buf765)
            del mm_241
            buf766 = buf754; del buf754  # reuse
            # Topologically Sorted Source Nodes: [mm_801], Original ATen: [aten.mm]
            extern_kernels.mm(buf764, permute_1305, out=buf766)
            del permute_1305
            buf768 = buf747; del buf747  # reuse
            # Topologically Sorted Source Nodes: [permute_1307, mm_802], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf766, (32, 1218), (1, 32), 0), view_659, out=buf768)
            del view_659
            buf767 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3013], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf765, buf767, 28672, stream=stream0)
            buf770 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3019], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf768, buf770, 28672, stream=stream0)
            buf769 = buf764; del buf764  # reuse
            # Topologically Sorted Source Nodes: [mm_803], Original ATen: [aten.mm]
            extern_kernels.mm(buf766, permute_1309, out=buf769)
            del permute_1309
            buf771 = buf760; del buf760  # reuse
            # Topologically Sorted Source Nodes: [view_1667, result_285, permute_1311, mm_804], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf763, (1218, 896), (896, 1), 0), primals_356, out=buf771)
            del primals_356
            buf772 = reinterpret_tensor(buf769, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf769  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1666, view_1668, add_523, view_1669, permute_1312, _scaled_dot_product_efficient_attention_backward_10], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf772, buf771, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1666, view_1668, add_523, view_1669, permute_1312, _scaled_dot_product_efficient_attention_backward_10], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf773 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf772, add_174, view_654, view_655, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_52, getitem_53, getitem_54, getitem_55, 0.0, [True, True, True, False], scale=0.125)
            del add_174
            del getitem_52
            del getitem_53
            del getitem_54
            del getitem_55
            del view_654
            del view_655
            buf774 = buf773[0]
            assert_size_stride(buf774, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf774, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf775 = buf773[1]
            assert_size_stride(buf775, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf775, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf777 = reinterpret_tensor(buf721, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf721  # reuse
            # Topologically Sorted Source Nodes: [view_1671, sum_44], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf775, buf777, 155904, stream=stream0)
            buf776 = buf773[2]
            assert_size_stride(buf776, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf776, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf773
            buf778 = buf705; del buf705  # reuse
            buf779 = reinterpret_tensor(buf714, (1218, 128), (128, 1), 0); del buf714  # reuse
            # Topologically Sorted Source Nodes: [view_1670, sum_43, squeeze_20, permute_1313, clone_80, view_1672, mul_750, view_1673], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf776, buf778, buf779, 155904, stream=stream0)
            buf780 = buf715; del buf715  # reuse
            # Topologically Sorted Source Nodes: [permute_1314, mm_805], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf779, (128, 1218), (1, 128), 0), mm_238, out=buf780)
            del mm_238
            buf781 = buf766; del buf766  # reuse
            # Topologically Sorted Source Nodes: [mm_806], Original ATen: [aten.mm]
            extern_kernels.mm(buf779, permute_1316, out=buf781)
            del permute_1316
            buf782 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3027], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf780, buf782, 4096, stream=stream0)
            buf783 = buf768; del buf768  # reuse
            # Topologically Sorted Source Nodes: [permute_1318, mm_807], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf781, (32, 1218), (1, 32), 0), view_635, out=buf783)
            buf785 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3033], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf783, buf785, 28672, stream=stream0)
            buf787 = reinterpret_tensor(buf779, (1, 1218, 128), (155904, 128, 1), 0); del buf779  # reuse
            buf794 = reinterpret_tensor(buf704, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf704  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1671, sum_44, squeeze_21, mul_746, slice_162, slice_163, neg_79, add_524, mul_747, add_525, permute_1323, clone_81, view_1679, mul_751], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf777, mm_default, buf787, buf794, 1218, 128, stream=stream0)
            buf788 = buf780; del buf780  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1671, sum_44, squeeze_21, mul_746, slice_162, slice_163, neg_79, add_524, mul_747, add_525, permute_1323, clone_81, view_1679, mul_751, view_1680, permute_1324, mm_810], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf787, (128, 1218), (1, 128), 0), mm_236, out=buf788)
            del mm_236
            buf789 = buf745; del buf745  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1671, sum_44, squeeze_21, mul_746, slice_162, slice_163, neg_79, add_524, mul_747, add_525, permute_1323, clone_81, view_1679, mul_751, view_1680, mm_811], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf787, (1218, 128), (128, 1), 0), permute_1326, out=buf789)
            del permute_1326
            buf790 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3041], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf788, buf790, 4096, stream=stream0)
            buf791 = buf783; del buf783  # reuse
            # Topologically Sorted Source Nodes: [permute_1328, mm_812], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf789, (32, 1218), (1, 32), 0), view_635, out=buf791)
            buf793 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3047], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf791, buf793, 28672, stream=stream0)
            buf786 = reinterpret_tensor(buf776, (1218, 896), (896, 1), 0); del buf776  # reuse
            # Topologically Sorted Source Nodes: [view_1670, sum_43, squeeze_20, permute_1313, clone_80, view_1672, view_1677, result_282, permute_1322, mm_809], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf778, (1218, 128), (128, 1), 0), primals_352, out=buf786)
            del primals_352
            buf795 = reinterpret_tensor(buf775, (1218, 896), (896, 1), 0); del buf775  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1671, sum_44, squeeze_21, mul_746, slice_162, slice_163, neg_79, add_524, mul_747, add_525, permute_1323, clone_81, view_1679, view_1684, result_279, permute_1332, mm_814], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf794, (1218, 128), (128, 1), 0), primals_348, out=buf795)
            del primals_348
            buf784 = reinterpret_tensor(buf772, (1218, 896), (896, 1), 0); del buf772  # reuse
            # Topologically Sorted Source Nodes: [mm_808], Original ATen: [aten.mm]
            extern_kernels.mm(buf781, permute_1320, out=buf784)
            del permute_1320
            buf792 = buf771; del buf771  # reuse
            # Topologically Sorted Source Nodes: [mm_813], Original ATen: [aten.mm]
            extern_kernels.mm(buf789, permute_1330, out=buf792)
            del permute_1330
            buf796 = reinterpret_tensor(buf757, (1, 1218, 896), (1091328, 896, 1), 0); del buf757  # reuse
            buf803 = reinterpret_tensor(buf751, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf751  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_748, slice_164, slice_165, neg_80, add_526, mul_749, add_527, permute_1333, clone_82, view_1686, mul_752], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf774, mm_default, buf796, buf803, 1091328, stream=stream0)
            buf797 = reinterpret_tensor(buf791, (896, 32), (32, 1), 0); del buf791  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_748, slice_164, slice_165, neg_80, add_526, mul_749, add_527, permute_1333, clone_82, view_1686, mul_752, view_1687, permute_1334, mm_815], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf796, (896, 1218), (1, 896), 0), mm_234, out=buf797)
            del mm_234
            buf798 = buf789; del buf789  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_748, slice_164, slice_165, neg_80, add_526, mul_749, add_527, permute_1333, clone_82, view_1686, mul_752, view_1687, mm_816], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf796, (1218, 896), (896, 1), 0), permute_1336, out=buf798)
            del permute_1336
            buf800 = reinterpret_tensor(buf765, (32, 896), (896, 1), 0); del buf765  # reuse
            # Topologically Sorted Source Nodes: [permute_1338, mm_817], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf798, (32, 1218), (1, 32), 0), view_635, out=buf800)
            del view_635
            buf804 = reinterpret_tensor(buf796, (1218, 896), (896, 1), 0); del buf796  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_748, slice_164, slice_165, neg_80, add_526, mul_749, add_527, permute_1333, clone_82, view_1686, view_1691, result_276, permute_1342, mm_819], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf803, (1218, 896), (896, 1), 0), primals_344, out=buf804)
            del primals_344
            buf799 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3055], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf797, buf799, 28672, stream=stream0)
            buf802 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3061], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf800, buf802, 28672, stream=stream0)
            buf801 = reinterpret_tensor(buf803, (1218, 896), (896, 1), 0); del buf803  # reuse
            # Topologically Sorted Source Nodes: [mm_818], Original ATen: [aten.mm]
            extern_kernels.mm(buf798, permute_1340, out=buf801)
            del permute_1340
            buf807 = buf763; del buf763  # reuse
            buf808 = reinterpret_tensor(buf774, (1218, 896), (896, 1), 0); del buf774  # reuse
            # Topologically Sorted Source Nodes: [view_1676, view_1678, add_528, view_1683, add_529, view_1685, add_530, view_1690, add_531, view_1692, add_532, mul_753, convert_element_type_3065, hidden_states_130, mul_754, mul_755, sum_45, pow_94, mul_756, mul_757, expand_99, div_22, pow_95, mul_758, mul_759, add_533, convert_element_type_3066, add_534, mul_760, view_1693], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf807, buf784, buf786, buf792, buf795, buf801, buf804, primals_343, add_169, rsqrt_26, buf808, 1218, 896, stream=stream0)
            del add_169
            del buf784
            del buf786
            del primals_343
            del rsqrt_26
            buf809 = reinterpret_tensor(buf800, (896, 32), (32, 1), 0); del buf800  # reuse
            # Topologically Sorted Source Nodes: [permute_1343, mm_820], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf808, (896, 1218), (1, 896), 0), mm_232, out=buf809)
            del mm_232
            buf810 = buf798; del buf798  # reuse
            # Topologically Sorted Source Nodes: [mm_821], Original ATen: [aten.mm]
            extern_kernels.mm(buf808, permute_1345, out=buf810)
            del permute_1345
            buf812 = reinterpret_tensor(buf753, (32, 4864), (4864, 1), 0); del buf753  # reuse
            # Topologically Sorted Source Nodes: [permute_1347, mm_822], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf810, (32, 1218), (1, 32), 0), view_629, out=buf812)
            del view_629
            buf811 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3071], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf809, buf811, 28672, stream=stream0)
            buf814 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3077], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf812, buf814, 155648, stream=stream0)
            buf813 = reinterpret_tensor(buf752, (1218, 4864), (4864, 1), 0); del buf752  # reuse
            # Topologically Sorted Source Nodes: [mm_823], Original ATen: [aten.mm]
            extern_kernels.mm(buf810, permute_1349, out=buf813)
            del permute_1349
            buf815 = reinterpret_tensor(buf743, (1218, 4864), (4864, 1), 0); del buf743  # reuse
            # Topologically Sorted Source Nodes: [view_1697, result_273, permute_1351, mm_824], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf807, (1218, 896), (896, 1), 0), primals_340, out=buf815)
            del primals_340
            buf816 = buf759; del buf759  # reuse
            buf823 = buf750; del buf750  # reuse
            buf825 = reinterpret_tensor(buf742, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf742  # reuse
            buf832 = reinterpret_tensor(buf740, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf740  # reuse
            # Topologically Sorted Source Nodes: [view_1696, view_1698, add_535, silu_12, mul_761, mul_762, mul_763, convert_element_type_3095, neg_81, exp_11, add_537, reciprocal_11, mul_764, mul_765, sub_11, mul_766, add_538, mul_767, convert_element_type_3097, mul_768], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf813, buf815, add_166, add_167, buf816, buf823, buf825, buf832, 5924352, stream=stream0)
            del add_166
            del add_167
            buf824 = buf808; del buf808  # reuse
            # Topologically Sorted Source Nodes: [view_1696, view_1698, add_535, silu_12, mul_761, view_1703, result_270, permute_1360, mm_829], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf823, (1218, 4864), (4864, 1), 0), primals_337, out=buf824)
            del primals_337
            buf833 = buf804; del buf804  # reuse
            # Topologically Sorted Source Nodes: [view_1696, view_1698, add_535, silu_12, mul_762, convert_element_type_3095, neg_81, exp_11, add_537, reciprocal_11, mul_764, mul_765, sub_11, mul_766, add_538, mul_767, convert_element_type_3097, view_1709, result_267, permute_1369, mm_834], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf832, (1218, 4864), (4864, 1), 0), primals_334, out=buf833)
            del primals_334
            buf818 = buf810; del buf810  # reuse
            # Topologically Sorted Source Nodes: [view_1696, view_1698, add_535, silu_12, mul_761, mul_763, view_1699, mm_826], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf816, (1218, 4864), (4864, 1), 0), permute_1354, out=buf818)
            del permute_1354
            buf817 = reinterpret_tensor(buf812, (4864, 32), (32, 1), 0); del buf812  # reuse
            # Topologically Sorted Source Nodes: [view_1696, view_1698, add_535, silu_12, mul_761, mul_763, view_1699, permute_1352, mm_825], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf816, (4864, 1218), (1, 4864), 0), mm_229, out=buf817)
            del mm_229
            buf827 = buf781; del buf781  # reuse
            # Topologically Sorted Source Nodes: [view_1696, view_1698, add_535, silu_12, mul_762, convert_element_type_3095, neg_81, exp_11, add_537, reciprocal_11, mul_764, mul_765, sub_11, mul_766, add_538, mul_767, convert_element_type_3097, mul_768, view_1705, mm_831], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf825, (1218, 4864), (4864, 1), 0), permute_1363, out=buf827)
            del permute_1363
            buf826 = buf744; del buf744  # reuse
            # Topologically Sorted Source Nodes: [view_1696, view_1698, add_535, silu_12, mul_762, convert_element_type_3095, neg_81, exp_11, add_537, reciprocal_11, mul_764, mul_765, sub_11, mul_766, add_538, mul_767, convert_element_type_3097, mul_768, view_1705, permute_1361, mm_830], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf825, (4864, 1218), (1, 4864), 0), mm_226, out=buf826)
            del mm_226
            buf820 = reinterpret_tensor(buf809, (32, 896), (896, 1), 0); del buf809  # reuse
            # Topologically Sorted Source Nodes: [permute_1356, mm_827], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf818, (32, 1218), (1, 32), 0), view_617, out=buf820)
            buf829 = reinterpret_tensor(buf797, (32, 896), (896, 1), 0); del buf797  # reuse
            # Topologically Sorted Source Nodes: [permute_1365, mm_832], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf827, (32, 1218), (1, 32), 0), view_617, out=buf829)
            del view_617
            buf822 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3091], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf820, buf822, 28672, stream=stream0)
            buf831 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3108], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf829, buf831, 28672, stream=stream0)
            buf819 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3085], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf817, buf819, 155648, stream=stream0)
            buf828 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3102], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf826, buf828, 155648, stream=stream0)
            buf821 = buf801; del buf801  # reuse
            # Topologically Sorted Source Nodes: [mm_828], Original ATen: [aten.mm]
            extern_kernels.mm(buf818, permute_1358, out=buf821)
            del permute_1358
            buf830 = buf795; del buf795  # reuse
            # Topologically Sorted Source Nodes: [mm_833], Original ATen: [aten.mm]
            extern_kernels.mm(buf827, permute_1367, out=buf830)
            del permute_1367
            buf836 = buf807; del buf807  # reuse
            buf837 = buf792; del buf792  # reuse
            # Topologically Sorted Source Nodes: [view_1702, view_1704, add_536, view_1708, add_539, view_1710, add_540, mul_769, convert_element_type_3112, hidden_states_126, mul_770, mul_771, sum_46, pow_96, mul_772, mul_773, expand_100, div_23, pow_97, mul_774, mul_775, add_541, convert_element_type_3113, add_542, mul_776, view_1711], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf836, buf821, buf824, buf830, buf833, primals_333, add_164, rsqrt_25, buf837, 1218, 896, stream=stream0)
            del add_164
            del buf821
            del primals_333
            del rsqrt_25
            buf838 = reinterpret_tensor(buf829, (896, 32), (32, 1), 0); del buf829  # reuse
            # Topologically Sorted Source Nodes: [permute_1370, mm_835], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf837, (896, 1218), (1, 896), 0), mm_223, out=buf838)
            del mm_223
            buf839 = buf827; del buf827  # reuse
            # Topologically Sorted Source Nodes: [mm_836], Original ATen: [aten.mm]
            extern_kernels.mm(buf837, permute_1372, out=buf839)
            del permute_1372
            buf841 = buf820; del buf820  # reuse
            # Topologically Sorted Source Nodes: [permute_1374, mm_837], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf839, (32, 1218), (1, 32), 0), view_611, out=buf841)
            del view_611
            buf840 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3118], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf838, buf840, 28672, stream=stream0)
            buf843 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3124], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf841, buf843, 28672, stream=stream0)
            buf842 = buf837; del buf837  # reuse
            # Topologically Sorted Source Nodes: [mm_838], Original ATen: [aten.mm]
            extern_kernels.mm(buf839, permute_1376, out=buf842)
            del permute_1376
            buf844 = buf833; del buf833  # reuse
            # Topologically Sorted Source Nodes: [view_1715, result_264, permute_1378, mm_839], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf836, (1218, 896), (896, 1), 0), primals_330, out=buf844)
            del primals_330
            buf845 = reinterpret_tensor(buf842, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf842  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1714, view_1716, add_543, view_1717, permute_1379, _scaled_dot_product_efficient_attention_backward_11], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf845, buf844, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1714, view_1716, add_543, view_1717, permute_1379, _scaled_dot_product_efficient_attention_backward_11], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf846 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf845, add_161, view_606, view_607, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_48, getitem_49, getitem_50, getitem_51, 0.0, [True, True, True, False], scale=0.125)
            del add_161
            del getitem_48
            del getitem_49
            del getitem_50
            del getitem_51
            del view_606
            del view_607
            buf847 = buf846[0]
            assert_size_stride(buf847, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf847, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf848 = buf846[1]
            assert_size_stride(buf848, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf848, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf850 = reinterpret_tensor(buf794, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf794  # reuse
            # Topologically Sorted Source Nodes: [view_1719, sum_48], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf848, buf850, 155904, stream=stream0)
            buf849 = buf846[2]
            assert_size_stride(buf849, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf849, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf846
            buf851 = buf778; del buf778  # reuse
            buf852 = reinterpret_tensor(buf787, (1218, 128), (128, 1), 0); del buf787  # reuse
            # Topologically Sorted Source Nodes: [view_1718, sum_47, squeeze_22, permute_1380, clone_83, view_1720, mul_781, view_1721], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf849, buf851, buf852, 155904, stream=stream0)
            buf853 = buf788; del buf788  # reuse
            # Topologically Sorted Source Nodes: [permute_1381, mm_840], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf852, (128, 1218), (1, 128), 0), mm_220, out=buf853)
            del mm_220
            buf854 = buf839; del buf839  # reuse
            # Topologically Sorted Source Nodes: [mm_841], Original ATen: [aten.mm]
            extern_kernels.mm(buf852, permute_1383, out=buf854)
            del permute_1383
            buf855 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3132], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf853, buf855, 4096, stream=stream0)
            buf856 = buf841; del buf841  # reuse
            # Topologically Sorted Source Nodes: [permute_1385, mm_842], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf854, (32, 1218), (1, 32), 0), view_587, out=buf856)
            buf858 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3138], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf856, buf858, 28672, stream=stream0)
            buf860 = reinterpret_tensor(buf852, (1, 1218, 128), (155904, 128, 1), 0); del buf852  # reuse
            buf867 = reinterpret_tensor(buf777, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf777  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1719, sum_48, squeeze_23, mul_777, slice_166, slice_167, neg_82, add_544, mul_778, add_545, permute_1390, clone_84, view_1727, mul_782], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf850, mm_default, buf860, buf867, 1218, 128, stream=stream0)
            buf861 = buf853; del buf853  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1719, sum_48, squeeze_23, mul_777, slice_166, slice_167, neg_82, add_544, mul_778, add_545, permute_1390, clone_84, view_1727, mul_782, view_1728, permute_1391, mm_845], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf860, (128, 1218), (1, 128), 0), mm_218, out=buf861)
            del mm_218
            buf862 = buf818; del buf818  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1719, sum_48, squeeze_23, mul_777, slice_166, slice_167, neg_82, add_544, mul_778, add_545, permute_1390, clone_84, view_1727, mul_782, view_1728, mm_846], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf860, (1218, 128), (128, 1), 0), permute_1393, out=buf862)
            del permute_1393
            buf863 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3146], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf861, buf863, 4096, stream=stream0)
            buf864 = buf856; del buf856  # reuse
            # Topologically Sorted Source Nodes: [permute_1395, mm_847], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf862, (32, 1218), (1, 32), 0), view_587, out=buf864)
            buf866 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3152], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf864, buf866, 28672, stream=stream0)
            buf859 = reinterpret_tensor(buf849, (1218, 896), (896, 1), 0); del buf849  # reuse
            # Topologically Sorted Source Nodes: [view_1718, sum_47, squeeze_22, permute_1380, clone_83, view_1720, view_1725, result_261, permute_1389, mm_844], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf851, (1218, 128), (128, 1), 0), primals_326, out=buf859)
            del primals_326
            buf868 = reinterpret_tensor(buf848, (1218, 896), (896, 1), 0); del buf848  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1719, sum_48, squeeze_23, mul_777, slice_166, slice_167, neg_82, add_544, mul_778, add_545, permute_1390, clone_84, view_1727, view_1732, result_258, permute_1399, mm_849], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf867, (1218, 128), (128, 1), 0), primals_322, out=buf868)
            del primals_322
            buf857 = reinterpret_tensor(buf845, (1218, 896), (896, 1), 0); del buf845  # reuse
            # Topologically Sorted Source Nodes: [mm_843], Original ATen: [aten.mm]
            extern_kernels.mm(buf854, permute_1387, out=buf857)
            del permute_1387
            buf865 = buf844; del buf844  # reuse
            # Topologically Sorted Source Nodes: [mm_848], Original ATen: [aten.mm]
            extern_kernels.mm(buf862, permute_1397, out=buf865)
            del permute_1397
            buf869 = reinterpret_tensor(buf830, (1, 1218, 896), (1091328, 896, 1), 0); del buf830  # reuse
            buf876 = reinterpret_tensor(buf824, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf824  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_779, slice_168, slice_169, neg_83, add_546, mul_780, add_547, permute_1400, clone_85, view_1734, mul_783], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf847, mm_default, buf869, buf876, 1091328, stream=stream0)
            buf870 = reinterpret_tensor(buf864, (896, 32), (32, 1), 0); del buf864  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_779, slice_168, slice_169, neg_83, add_546, mul_780, add_547, permute_1400, clone_85, view_1734, mul_783, view_1735, permute_1401, mm_850], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf869, (896, 1218), (1, 896), 0), mm_216, out=buf870)
            del mm_216
            buf871 = buf862; del buf862  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_779, slice_168, slice_169, neg_83, add_546, mul_780, add_547, permute_1400, clone_85, view_1734, mul_783, view_1735, mm_851], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf869, (1218, 896), (896, 1), 0), permute_1403, out=buf871)
            del permute_1403
            buf873 = reinterpret_tensor(buf838, (32, 896), (896, 1), 0); del buf838  # reuse
            # Topologically Sorted Source Nodes: [permute_1405, mm_852], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf871, (32, 1218), (1, 32), 0), view_587, out=buf873)
            del view_587
            buf877 = reinterpret_tensor(buf869, (1218, 896), (896, 1), 0); del buf869  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_779, slice_168, slice_169, neg_83, add_546, mul_780, add_547, permute_1400, clone_85, view_1734, view_1739, result_255, permute_1409, mm_854], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf876, (1218, 896), (896, 1), 0), primals_318, out=buf877)
            del primals_318
            buf872 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3160], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf870, buf872, 28672, stream=stream0)
            buf875 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3166], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf873, buf875, 28672, stream=stream0)
            buf874 = reinterpret_tensor(buf876, (1218, 896), (896, 1), 0); del buf876  # reuse
            # Topologically Sorted Source Nodes: [mm_853], Original ATen: [aten.mm]
            extern_kernels.mm(buf871, permute_1407, out=buf874)
            del permute_1407
            buf880 = buf836; del buf836  # reuse
            buf881 = reinterpret_tensor(buf847, (1218, 896), (896, 1), 0); del buf847  # reuse
            # Topologically Sorted Source Nodes: [view_1724, view_1726, add_548, view_1731, add_549, view_1733, add_550, view_1738, add_551, view_1740, add_552, mul_784, convert_element_type_3170, hidden_states_120, mul_785, mul_786, sum_49, pow_98, mul_787, mul_788, expand_101, div_24, pow_99, mul_789, mul_790, add_553, convert_element_type_3171, add_554, mul_791, view_1741], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf880, buf857, buf859, buf865, buf868, buf874, buf877, primals_317, add_156, rsqrt_24, buf881, 1218, 896, stream=stream0)
            del add_156
            del buf857
            del buf859
            del primals_317
            del rsqrt_24
            buf882 = reinterpret_tensor(buf873, (896, 32), (32, 1), 0); del buf873  # reuse
            # Topologically Sorted Source Nodes: [permute_1410, mm_855], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf881, (896, 1218), (1, 896), 0), mm_214, out=buf882)
            del mm_214
            buf883 = buf871; del buf871  # reuse
            # Topologically Sorted Source Nodes: [mm_856], Original ATen: [aten.mm]
            extern_kernels.mm(buf881, permute_1412, out=buf883)
            del permute_1412
            buf885 = reinterpret_tensor(buf826, (32, 4864), (4864, 1), 0); del buf826  # reuse
            # Topologically Sorted Source Nodes: [permute_1414, mm_857], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf883, (32, 1218), (1, 32), 0), view_581, out=buf885)
            del view_581
            buf884 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3176], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf882, buf884, 28672, stream=stream0)
            buf887 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3182], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf885, buf887, 155648, stream=stream0)
            buf886 = reinterpret_tensor(buf825, (1218, 4864), (4864, 1), 0); del buf825  # reuse
            # Topologically Sorted Source Nodes: [mm_858], Original ATen: [aten.mm]
            extern_kernels.mm(buf883, permute_1416, out=buf886)
            del permute_1416
            buf888 = reinterpret_tensor(buf816, (1218, 4864), (4864, 1), 0); del buf816  # reuse
            # Topologically Sorted Source Nodes: [view_1745, result_252, permute_1418, mm_859], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf880, (1218, 896), (896, 1), 0), primals_314, out=buf888)
            del primals_314
            buf889 = buf832; del buf832  # reuse
            buf896 = buf823; del buf823  # reuse
            buf898 = reinterpret_tensor(buf815, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf815  # reuse
            buf905 = reinterpret_tensor(buf813, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf813  # reuse
            # Topologically Sorted Source Nodes: [view_1744, view_1746, add_555, silu_11, mul_792, mul_793, mul_794, convert_element_type_3200, neg_84, exp_12, add_557, reciprocal_12, mul_795, mul_796, sub_12, mul_797, add_558, mul_798, convert_element_type_3202, mul_799], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf886, buf888, add_153, add_154, buf889, buf896, buf898, buf905, 5924352, stream=stream0)
            del add_153
            del add_154
            buf897 = buf881; del buf881  # reuse
            # Topologically Sorted Source Nodes: [view_1744, view_1746, add_555, silu_11, mul_792, view_1751, result_249, permute_1427, mm_864], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf896, (1218, 4864), (4864, 1), 0), primals_311, out=buf897)
            del primals_311
            buf906 = buf877; del buf877  # reuse
            # Topologically Sorted Source Nodes: [view_1744, view_1746, add_555, silu_11, mul_793, convert_element_type_3200, neg_84, exp_12, add_557, reciprocal_12, mul_795, mul_796, sub_12, mul_797, add_558, mul_798, convert_element_type_3202, view_1757, result_246, permute_1436, mm_869], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf905, (1218, 4864), (4864, 1), 0), primals_308, out=buf906)
            del primals_308
            buf891 = buf883; del buf883  # reuse
            # Topologically Sorted Source Nodes: [view_1744, view_1746, add_555, silu_11, mul_792, mul_794, view_1747, mm_861], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf889, (1218, 4864), (4864, 1), 0), permute_1421, out=buf891)
            del permute_1421
            buf890 = reinterpret_tensor(buf885, (4864, 32), (32, 1), 0); del buf885  # reuse
            # Topologically Sorted Source Nodes: [view_1744, view_1746, add_555, silu_11, mul_792, mul_794, view_1747, permute_1419, mm_860], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf889, (4864, 1218), (1, 4864), 0), mm_211, out=buf890)
            del mm_211
            buf900 = buf854; del buf854  # reuse
            # Topologically Sorted Source Nodes: [view_1744, view_1746, add_555, silu_11, mul_793, convert_element_type_3200, neg_84, exp_12, add_557, reciprocal_12, mul_795, mul_796, sub_12, mul_797, add_558, mul_798, convert_element_type_3202, mul_799, view_1753, mm_866], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf898, (1218, 4864), (4864, 1), 0), permute_1430, out=buf900)
            del permute_1430
            buf899 = buf817; del buf817  # reuse
            # Topologically Sorted Source Nodes: [view_1744, view_1746, add_555, silu_11, mul_793, convert_element_type_3200, neg_84, exp_12, add_557, reciprocal_12, mul_795, mul_796, sub_12, mul_797, add_558, mul_798, convert_element_type_3202, mul_799, view_1753, permute_1428, mm_865], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf898, (4864, 1218), (1, 4864), 0), mm_208, out=buf899)
            del mm_208
            buf893 = reinterpret_tensor(buf882, (32, 896), (896, 1), 0); del buf882  # reuse
            # Topologically Sorted Source Nodes: [permute_1423, mm_862], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf891, (32, 1218), (1, 32), 0), view_569, out=buf893)
            buf902 = reinterpret_tensor(buf870, (32, 896), (896, 1), 0); del buf870  # reuse
            # Topologically Sorted Source Nodes: [permute_1432, mm_867], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf900, (32, 1218), (1, 32), 0), view_569, out=buf902)
            del view_569
            buf895 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3196], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf893, buf895, 28672, stream=stream0)
            buf904 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3213], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf902, buf904, 28672, stream=stream0)
            buf892 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3190], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf890, buf892, 155648, stream=stream0)
            buf901 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3207], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf899, buf901, 155648, stream=stream0)
            buf894 = buf874; del buf874  # reuse
            # Topologically Sorted Source Nodes: [mm_863], Original ATen: [aten.mm]
            extern_kernels.mm(buf891, permute_1425, out=buf894)
            del permute_1425
            buf903 = buf868; del buf868  # reuse
            # Topologically Sorted Source Nodes: [mm_868], Original ATen: [aten.mm]
            extern_kernels.mm(buf900, permute_1434, out=buf903)
            del permute_1434
            buf909 = buf880; del buf880  # reuse
            buf910 = buf865; del buf865  # reuse
            # Topologically Sorted Source Nodes: [view_1750, view_1752, add_556, view_1756, add_559, view_1758, add_560, mul_800, convert_element_type_3217, hidden_states_116, mul_801, mul_802, sum_50, pow_100, mul_803, mul_804, expand_102, div_25, pow_101, mul_805, mul_806, add_561, convert_element_type_3218, add_562, mul_807, view_1759], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf909, buf894, buf897, buf903, buf906, primals_307, add_151, rsqrt_23, buf910, 1218, 896, stream=stream0)
            del add_151
            del buf894
            del primals_307
            del rsqrt_23
            buf911 = reinterpret_tensor(buf902, (896, 32), (32, 1), 0); del buf902  # reuse
            # Topologically Sorted Source Nodes: [permute_1437, mm_870], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf910, (896, 1218), (1, 896), 0), mm_205, out=buf911)
            del mm_205
            buf912 = buf900; del buf900  # reuse
            # Topologically Sorted Source Nodes: [mm_871], Original ATen: [aten.mm]
            extern_kernels.mm(buf910, permute_1439, out=buf912)
            del permute_1439
            buf914 = buf893; del buf893  # reuse
            # Topologically Sorted Source Nodes: [permute_1441, mm_872], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf912, (32, 1218), (1, 32), 0), view_563, out=buf914)
            del view_563
            buf913 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3223], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf911, buf913, 28672, stream=stream0)
            buf916 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3229], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf914, buf916, 28672, stream=stream0)
            buf915 = buf910; del buf910  # reuse
            # Topologically Sorted Source Nodes: [mm_873], Original ATen: [aten.mm]
            extern_kernels.mm(buf912, permute_1443, out=buf915)
            del permute_1443
            buf917 = buf906; del buf906  # reuse
            # Topologically Sorted Source Nodes: [view_1763, result_243, permute_1445, mm_874], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf909, (1218, 896), (896, 1), 0), primals_304, out=buf917)
            del primals_304
            buf918 = reinterpret_tensor(buf915, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf915  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1762, view_1764, add_563, view_1765, permute_1446, _scaled_dot_product_efficient_attention_backward_12], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf918, buf917, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1762, view_1764, add_563, view_1765, permute_1446, _scaled_dot_product_efficient_attention_backward_12], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf919 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf918, add_148, view_558, view_559, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_44, getitem_45, getitem_46, getitem_47, 0.0, [True, True, True, False], scale=0.125)
            del add_148
            del getitem_44
            del getitem_45
            del getitem_46
            del getitem_47
            del view_558
            del view_559
            buf920 = buf919[0]
            assert_size_stride(buf920, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf920, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf921 = buf919[1]
            assert_size_stride(buf921, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf921, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf923 = reinterpret_tensor(buf867, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf867  # reuse
            # Topologically Sorted Source Nodes: [view_1767, sum_52], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf921, buf923, 155904, stream=stream0)
            buf922 = buf919[2]
            assert_size_stride(buf922, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf922, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf919
            buf924 = buf851; del buf851  # reuse
            buf925 = reinterpret_tensor(buf860, (1218, 128), (128, 1), 0); del buf860  # reuse
            # Topologically Sorted Source Nodes: [view_1766, sum_51, squeeze_24, permute_1447, clone_86, view_1768, mul_812, view_1769], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf922, buf924, buf925, 155904, stream=stream0)
            buf926 = buf861; del buf861  # reuse
            # Topologically Sorted Source Nodes: [permute_1448, mm_875], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf925, (128, 1218), (1, 128), 0), mm_202, out=buf926)
            del mm_202
            buf927 = buf912; del buf912  # reuse
            # Topologically Sorted Source Nodes: [mm_876], Original ATen: [aten.mm]
            extern_kernels.mm(buf925, permute_1450, out=buf927)
            del permute_1450
            buf928 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3237], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf926, buf928, 4096, stream=stream0)
            buf929 = buf914; del buf914  # reuse
            # Topologically Sorted Source Nodes: [permute_1452, mm_877], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf927, (32, 1218), (1, 32), 0), view_539, out=buf929)
            buf931 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3243], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf929, buf931, 28672, stream=stream0)
            buf933 = reinterpret_tensor(buf925, (1, 1218, 128), (155904, 128, 1), 0); del buf925  # reuse
            buf940 = reinterpret_tensor(buf850, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf850  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1767, sum_52, squeeze_25, mul_808, slice_170, slice_171, neg_85, add_564, mul_809, add_565, permute_1457, clone_87, view_1775, mul_813], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf923, mm_default, buf933, buf940, 1218, 128, stream=stream0)
            buf934 = buf926; del buf926  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1767, sum_52, squeeze_25, mul_808, slice_170, slice_171, neg_85, add_564, mul_809, add_565, permute_1457, clone_87, view_1775, mul_813, view_1776, permute_1458, mm_880], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf933, (128, 1218), (1, 128), 0), mm_200, out=buf934)
            del mm_200
            buf935 = buf891; del buf891  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1767, sum_52, squeeze_25, mul_808, slice_170, slice_171, neg_85, add_564, mul_809, add_565, permute_1457, clone_87, view_1775, mul_813, view_1776, mm_881], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf933, (1218, 128), (128, 1), 0), permute_1460, out=buf935)
            del permute_1460
            buf936 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3251], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf934, buf936, 4096, stream=stream0)
            buf937 = buf929; del buf929  # reuse
            # Topologically Sorted Source Nodes: [permute_1462, mm_882], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf935, (32, 1218), (1, 32), 0), view_539, out=buf937)
            buf939 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3257], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf937, buf939, 28672, stream=stream0)
            buf932 = reinterpret_tensor(buf922, (1218, 896), (896, 1), 0); del buf922  # reuse
            # Topologically Sorted Source Nodes: [view_1766, sum_51, squeeze_24, permute_1447, clone_86, view_1768, view_1773, result_240, permute_1456, mm_879], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf924, (1218, 128), (128, 1), 0), primals_300, out=buf932)
            del primals_300
            buf941 = reinterpret_tensor(buf921, (1218, 896), (896, 1), 0); del buf921  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1767, sum_52, squeeze_25, mul_808, slice_170, slice_171, neg_85, add_564, mul_809, add_565, permute_1457, clone_87, view_1775, view_1780, result_237, permute_1466, mm_884], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf940, (1218, 128), (128, 1), 0), primals_296, out=buf941)
            del primals_296
            buf930 = reinterpret_tensor(buf918, (1218, 896), (896, 1), 0); del buf918  # reuse
            # Topologically Sorted Source Nodes: [mm_878], Original ATen: [aten.mm]
            extern_kernels.mm(buf927, permute_1454, out=buf930)
            del permute_1454
            buf938 = buf917; del buf917  # reuse
            # Topologically Sorted Source Nodes: [mm_883], Original ATen: [aten.mm]
            extern_kernels.mm(buf935, permute_1464, out=buf938)
            del permute_1464
            buf942 = reinterpret_tensor(buf903, (1, 1218, 896), (1091328, 896, 1), 0); del buf903  # reuse
            buf949 = reinterpret_tensor(buf897, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf897  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_810, slice_172, slice_173, neg_86, add_566, mul_811, add_567, permute_1467, clone_88, view_1782, mul_814], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf920, mm_default, buf942, buf949, 1091328, stream=stream0)
            buf943 = reinterpret_tensor(buf937, (896, 32), (32, 1), 0); del buf937  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_810, slice_172, slice_173, neg_86, add_566, mul_811, add_567, permute_1467, clone_88, view_1782, mul_814, view_1783, permute_1468, mm_885], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf942, (896, 1218), (1, 896), 0), mm_198, out=buf943)
            del mm_198
            buf944 = buf935; del buf935  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_810, slice_172, slice_173, neg_86, add_566, mul_811, add_567, permute_1467, clone_88, view_1782, mul_814, view_1783, mm_886], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf942, (1218, 896), (896, 1), 0), permute_1470, out=buf944)
            del permute_1470
            buf946 = reinterpret_tensor(buf911, (32, 896), (896, 1), 0); del buf911  # reuse
            # Topologically Sorted Source Nodes: [permute_1472, mm_887], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf944, (32, 1218), (1, 32), 0), view_539, out=buf946)
            del view_539
            buf950 = reinterpret_tensor(buf942, (1218, 896), (896, 1), 0); del buf942  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_810, slice_172, slice_173, neg_86, add_566, mul_811, add_567, permute_1467, clone_88, view_1782, view_1787, result_234, permute_1476, mm_889], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf949, (1218, 896), (896, 1), 0), primals_292, out=buf950)
            del primals_292
            buf945 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3265], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf943, buf945, 28672, stream=stream0)
            buf948 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3271], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf946, buf948, 28672, stream=stream0)
            buf947 = reinterpret_tensor(buf949, (1218, 896), (896, 1), 0); del buf949  # reuse
            # Topologically Sorted Source Nodes: [mm_888], Original ATen: [aten.mm]
            extern_kernels.mm(buf944, permute_1474, out=buf947)
            del permute_1474
            buf953 = buf909; del buf909  # reuse
            buf954 = reinterpret_tensor(buf920, (1218, 896), (896, 1), 0); del buf920  # reuse
            # Topologically Sorted Source Nodes: [view_1772, view_1774, add_568, view_1779, add_569, view_1781, add_570, view_1786, add_571, view_1788, add_572, mul_815, convert_element_type_3275, hidden_states_110, mul_816, mul_817, sum_53, pow_102, mul_818, mul_819, expand_103, div_26, pow_103, mul_820, mul_821, add_573, convert_element_type_3276, add_574, mul_822, view_1789], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf953, buf930, buf932, buf938, buf941, buf947, buf950, primals_291, add_143, rsqrt_22, buf954, 1218, 896, stream=stream0)
            del add_143
            del buf930
            del buf932
            del primals_291
            del rsqrt_22
            buf955 = reinterpret_tensor(buf946, (896, 32), (32, 1), 0); del buf946  # reuse
            # Topologically Sorted Source Nodes: [permute_1477, mm_890], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf954, (896, 1218), (1, 896), 0), mm_196, out=buf955)
            del mm_196
            buf956 = buf944; del buf944  # reuse
            # Topologically Sorted Source Nodes: [mm_891], Original ATen: [aten.mm]
            extern_kernels.mm(buf954, permute_1479, out=buf956)
            del permute_1479
            buf958 = reinterpret_tensor(buf899, (32, 4864), (4864, 1), 0); del buf899  # reuse
            # Topologically Sorted Source Nodes: [permute_1481, mm_892], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf956, (32, 1218), (1, 32), 0), view_533, out=buf958)
            del view_533
            buf957 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3281], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf955, buf957, 28672, stream=stream0)
            buf960 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3287], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf958, buf960, 155648, stream=stream0)
            buf959 = reinterpret_tensor(buf898, (1218, 4864), (4864, 1), 0); del buf898  # reuse
            # Topologically Sorted Source Nodes: [mm_893], Original ATen: [aten.mm]
            extern_kernels.mm(buf956, permute_1483, out=buf959)
            del permute_1483
            buf961 = reinterpret_tensor(buf889, (1218, 4864), (4864, 1), 0); del buf889  # reuse
            # Topologically Sorted Source Nodes: [view_1793, result_231, permute_1485, mm_894], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf953, (1218, 896), (896, 1), 0), primals_288, out=buf961)
            del primals_288
            buf962 = buf905; del buf905  # reuse
            buf969 = buf896; del buf896  # reuse
            buf971 = reinterpret_tensor(buf888, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf888  # reuse
            buf978 = reinterpret_tensor(buf886, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf886  # reuse
            # Topologically Sorted Source Nodes: [view_1792, view_1794, add_575, silu_10, mul_823, mul_824, mul_825, convert_element_type_3305, neg_87, exp_13, add_577, reciprocal_13, mul_826, mul_827, sub_13, mul_828, add_578, mul_829, convert_element_type_3307, mul_830], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf959, buf961, add_140, add_141, buf962, buf969, buf971, buf978, 5924352, stream=stream0)
            del add_140
            del add_141
            buf970 = buf954; del buf954  # reuse
            # Topologically Sorted Source Nodes: [view_1792, view_1794, add_575, silu_10, mul_823, view_1799, result_228, permute_1494, mm_899], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf969, (1218, 4864), (4864, 1), 0), primals_285, out=buf970)
            del primals_285
            buf979 = buf950; del buf950  # reuse
            # Topologically Sorted Source Nodes: [view_1792, view_1794, add_575, silu_10, mul_824, convert_element_type_3305, neg_87, exp_13, add_577, reciprocal_13, mul_826, mul_827, sub_13, mul_828, add_578, mul_829, convert_element_type_3307, view_1805, result_225, permute_1503, mm_904], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf978, (1218, 4864), (4864, 1), 0), primals_282, out=buf979)
            del primals_282
            buf964 = buf956; del buf956  # reuse
            # Topologically Sorted Source Nodes: [view_1792, view_1794, add_575, silu_10, mul_823, mul_825, view_1795, mm_896], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf962, (1218, 4864), (4864, 1), 0), permute_1488, out=buf964)
            del permute_1488
            buf963 = reinterpret_tensor(buf958, (4864, 32), (32, 1), 0); del buf958  # reuse
            # Topologically Sorted Source Nodes: [view_1792, view_1794, add_575, silu_10, mul_823, mul_825, view_1795, permute_1486, mm_895], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf962, (4864, 1218), (1, 4864), 0), mm_193, out=buf963)
            del mm_193
            buf973 = buf927; del buf927  # reuse
            # Topologically Sorted Source Nodes: [view_1792, view_1794, add_575, silu_10, mul_824, convert_element_type_3305, neg_87, exp_13, add_577, reciprocal_13, mul_826, mul_827, sub_13, mul_828, add_578, mul_829, convert_element_type_3307, mul_830, view_1801, mm_901], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf971, (1218, 4864), (4864, 1), 0), permute_1497, out=buf973)
            del permute_1497
            buf972 = buf890; del buf890  # reuse
            # Topologically Sorted Source Nodes: [view_1792, view_1794, add_575, silu_10, mul_824, convert_element_type_3305, neg_87, exp_13, add_577, reciprocal_13, mul_826, mul_827, sub_13, mul_828, add_578, mul_829, convert_element_type_3307, mul_830, view_1801, permute_1495, mm_900], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf971, (4864, 1218), (1, 4864), 0), mm_190, out=buf972)
            del mm_190
            buf966 = reinterpret_tensor(buf955, (32, 896), (896, 1), 0); del buf955  # reuse
            # Topologically Sorted Source Nodes: [permute_1490, mm_897], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf964, (32, 1218), (1, 32), 0), view_521, out=buf966)
            buf975 = reinterpret_tensor(buf943, (32, 896), (896, 1), 0); del buf943  # reuse
            # Topologically Sorted Source Nodes: [permute_1499, mm_902], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf973, (32, 1218), (1, 32), 0), view_521, out=buf975)
            del view_521
            buf968 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3301], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf966, buf968, 28672, stream=stream0)
            buf977 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3318], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf975, buf977, 28672, stream=stream0)
            buf965 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3295], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf963, buf965, 155648, stream=stream0)
            buf974 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3312], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf972, buf974, 155648, stream=stream0)
            buf967 = buf947; del buf947  # reuse
            # Topologically Sorted Source Nodes: [mm_898], Original ATen: [aten.mm]
            extern_kernels.mm(buf964, permute_1492, out=buf967)
            del permute_1492
            buf976 = buf941; del buf941  # reuse
            # Topologically Sorted Source Nodes: [mm_903], Original ATen: [aten.mm]
            extern_kernels.mm(buf973, permute_1501, out=buf976)
            del permute_1501
            buf982 = buf953; del buf953  # reuse
            buf983 = buf938; del buf938  # reuse
            # Topologically Sorted Source Nodes: [view_1798, view_1800, add_576, view_1804, add_579, view_1806, add_580, mul_831, convert_element_type_3322, hidden_states_106, mul_832, mul_833, sum_54, pow_104, mul_834, mul_835, expand_104, div_27, pow_105, mul_836, mul_837, add_581, convert_element_type_3323, add_582, mul_838, view_1807], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf982, buf967, buf970, buf976, buf979, primals_281, add_138, rsqrt_21, buf983, 1218, 896, stream=stream0)
            del add_138
            del buf967
            del primals_281
            del rsqrt_21
            buf984 = reinterpret_tensor(buf975, (896, 32), (32, 1), 0); del buf975  # reuse
            # Topologically Sorted Source Nodes: [permute_1504, mm_905], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf983, (896, 1218), (1, 896), 0), mm_187, out=buf984)
            del mm_187
            buf985 = buf973; del buf973  # reuse
            # Topologically Sorted Source Nodes: [mm_906], Original ATen: [aten.mm]
            extern_kernels.mm(buf983, permute_1506, out=buf985)
            del permute_1506
            buf987 = buf966; del buf966  # reuse
            # Topologically Sorted Source Nodes: [permute_1508, mm_907], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf985, (32, 1218), (1, 32), 0), view_515, out=buf987)
            del view_515
            buf986 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3328], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf984, buf986, 28672, stream=stream0)
            buf989 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3334], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf987, buf989, 28672, stream=stream0)
            buf988 = buf983; del buf983  # reuse
            # Topologically Sorted Source Nodes: [mm_908], Original ATen: [aten.mm]
            extern_kernels.mm(buf985, permute_1510, out=buf988)
            del permute_1510
            buf990 = buf979; del buf979  # reuse
            # Topologically Sorted Source Nodes: [view_1811, result_222, permute_1512, mm_909], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf982, (1218, 896), (896, 1), 0), primals_278, out=buf990)
            del primals_278
            buf991 = reinterpret_tensor(buf988, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf988  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1810, view_1812, add_583, view_1813, permute_1513, _scaled_dot_product_efficient_attention_backward_13], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf991, buf990, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1810, view_1812, add_583, view_1813, permute_1513, _scaled_dot_product_efficient_attention_backward_13], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf992 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf991, add_135, view_510, view_511, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_40, getitem_41, getitem_42, getitem_43, 0.0, [True, True, True, False], scale=0.125)
            del add_135
            del getitem_40
            del getitem_41
            del getitem_42
            del getitem_43
            del view_510
            del view_511
            buf993 = buf992[0]
            assert_size_stride(buf993, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf993, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf994 = buf992[1]
            assert_size_stride(buf994, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf994, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf996 = reinterpret_tensor(buf940, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf940  # reuse
            # Topologically Sorted Source Nodes: [view_1815, sum_56], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf994, buf996, 155904, stream=stream0)
            buf995 = buf992[2]
            assert_size_stride(buf995, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf995, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf992
            buf997 = buf924; del buf924  # reuse
            buf998 = reinterpret_tensor(buf933, (1218, 128), (128, 1), 0); del buf933  # reuse
            # Topologically Sorted Source Nodes: [view_1814, sum_55, squeeze_26, permute_1514, clone_89, view_1816, mul_843, view_1817], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf995, buf997, buf998, 155904, stream=stream0)
            buf999 = buf934; del buf934  # reuse
            # Topologically Sorted Source Nodes: [permute_1515, mm_910], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf998, (128, 1218), (1, 128), 0), mm_184, out=buf999)
            del mm_184
            buf1000 = buf985; del buf985  # reuse
            # Topologically Sorted Source Nodes: [mm_911], Original ATen: [aten.mm]
            extern_kernels.mm(buf998, permute_1517, out=buf1000)
            del permute_1517
            buf1001 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3342], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf999, buf1001, 4096, stream=stream0)
            buf1002 = buf987; del buf987  # reuse
            # Topologically Sorted Source Nodes: [permute_1519, mm_912], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1000, (32, 1218), (1, 32), 0), view_491, out=buf1002)
            buf1004 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3348], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1002, buf1004, 28672, stream=stream0)
            buf1006 = reinterpret_tensor(buf998, (1, 1218, 128), (155904, 128, 1), 0); del buf998  # reuse
            buf1013 = reinterpret_tensor(buf923, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf923  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1815, sum_56, squeeze_27, mul_839, slice_174, slice_175, neg_88, add_584, mul_840, add_585, permute_1524, clone_90, view_1823, mul_844], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf996, mm_default, buf1006, buf1013, 1218, 128, stream=stream0)
            buf1007 = buf999; del buf999  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1815, sum_56, squeeze_27, mul_839, slice_174, slice_175, neg_88, add_584, mul_840, add_585, permute_1524, clone_90, view_1823, mul_844, view_1824, permute_1525, mm_915], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1006, (128, 1218), (1, 128), 0), mm_182, out=buf1007)
            del mm_182
            buf1008 = buf964; del buf964  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1815, sum_56, squeeze_27, mul_839, slice_174, slice_175, neg_88, add_584, mul_840, add_585, permute_1524, clone_90, view_1823, mul_844, view_1824, mm_916], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1006, (1218, 128), (128, 1), 0), permute_1527, out=buf1008)
            del permute_1527
            buf1009 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3356], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1007, buf1009, 4096, stream=stream0)
            buf1010 = buf1002; del buf1002  # reuse
            # Topologically Sorted Source Nodes: [permute_1529, mm_917], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1008, (32, 1218), (1, 32), 0), view_491, out=buf1010)
            buf1012 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3362], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1010, buf1012, 28672, stream=stream0)
            buf1005 = reinterpret_tensor(buf995, (1218, 896), (896, 1), 0); del buf995  # reuse
            # Topologically Sorted Source Nodes: [view_1814, sum_55, squeeze_26, permute_1514, clone_89, view_1816, view_1821, result_219, permute_1523, mm_914], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf997, (1218, 128), (128, 1), 0), primals_274, out=buf1005)
            del primals_274
            buf1014 = reinterpret_tensor(buf994, (1218, 896), (896, 1), 0); del buf994  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1815, sum_56, squeeze_27, mul_839, slice_174, slice_175, neg_88, add_584, mul_840, add_585, permute_1524, clone_90, view_1823, view_1828, result_216, permute_1533, mm_919], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1013, (1218, 128), (128, 1), 0), primals_270, out=buf1014)
            del primals_270
            buf1003 = reinterpret_tensor(buf991, (1218, 896), (896, 1), 0); del buf991  # reuse
            # Topologically Sorted Source Nodes: [mm_913], Original ATen: [aten.mm]
            extern_kernels.mm(buf1000, permute_1521, out=buf1003)
            del permute_1521
            buf1011 = buf990; del buf990  # reuse
            # Topologically Sorted Source Nodes: [mm_918], Original ATen: [aten.mm]
            extern_kernels.mm(buf1008, permute_1531, out=buf1011)
            del permute_1531
            buf1015 = reinterpret_tensor(buf976, (1, 1218, 896), (1091328, 896, 1), 0); del buf976  # reuse
            buf1022 = reinterpret_tensor(buf970, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf970  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_841, slice_176, slice_177, neg_89, add_586, mul_842, add_587, permute_1534, clone_91, view_1830, mul_845], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf993, mm_default, buf1015, buf1022, 1091328, stream=stream0)
            buf1016 = reinterpret_tensor(buf1010, (896, 32), (32, 1), 0); del buf1010  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_841, slice_176, slice_177, neg_89, add_586, mul_842, add_587, permute_1534, clone_91, view_1830, mul_845, view_1831, permute_1535, mm_920], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1015, (896, 1218), (1, 896), 0), mm_180, out=buf1016)
            del mm_180
            buf1017 = buf1008; del buf1008  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_841, slice_176, slice_177, neg_89, add_586, mul_842, add_587, permute_1534, clone_91, view_1830, mul_845, view_1831, mm_921], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1015, (1218, 896), (896, 1), 0), permute_1537, out=buf1017)
            del permute_1537
            buf1019 = reinterpret_tensor(buf984, (32, 896), (896, 1), 0); del buf984  # reuse
            # Topologically Sorted Source Nodes: [permute_1539, mm_922], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1017, (32, 1218), (1, 32), 0), view_491, out=buf1019)
            del view_491
            buf1023 = reinterpret_tensor(buf1015, (1218, 896), (896, 1), 0); del buf1015  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_841, slice_176, slice_177, neg_89, add_586, mul_842, add_587, permute_1534, clone_91, view_1830, view_1835, result_213, permute_1543, mm_924], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1022, (1218, 896), (896, 1), 0), primals_266, out=buf1023)
            del primals_266
            buf1018 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3370], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1016, buf1018, 28672, stream=stream0)
            buf1021 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3376], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1019, buf1021, 28672, stream=stream0)
            buf1020 = reinterpret_tensor(buf1022, (1218, 896), (896, 1), 0); del buf1022  # reuse
            # Topologically Sorted Source Nodes: [mm_923], Original ATen: [aten.mm]
            extern_kernels.mm(buf1017, permute_1541, out=buf1020)
            del permute_1541
            buf1026 = buf982; del buf982  # reuse
            buf1027 = reinterpret_tensor(buf993, (1218, 896), (896, 1), 0); del buf993  # reuse
            # Topologically Sorted Source Nodes: [view_1820, view_1822, add_588, view_1827, add_589, view_1829, add_590, view_1834, add_591, view_1836, add_592, mul_846, convert_element_type_3380, hidden_states_100, mul_847, mul_848, sum_57, pow_106, mul_849, mul_850, expand_105, div_28, pow_107, mul_851, mul_852, add_593, convert_element_type_3381, add_594, mul_853, view_1837], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf1026, buf1003, buf1005, buf1011, buf1014, buf1020, buf1023, primals_265, add_130, rsqrt_20, buf1027, 1218, 896, stream=stream0)
            del add_130
            del buf1003
            del buf1005
            del primals_265
            del rsqrt_20
            buf1028 = reinterpret_tensor(buf1019, (896, 32), (32, 1), 0); del buf1019  # reuse
            # Topologically Sorted Source Nodes: [permute_1544, mm_925], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1027, (896, 1218), (1, 896), 0), mm_178, out=buf1028)
            del mm_178
            buf1029 = buf1017; del buf1017  # reuse
            # Topologically Sorted Source Nodes: [mm_926], Original ATen: [aten.mm]
            extern_kernels.mm(buf1027, permute_1546, out=buf1029)
            del permute_1546
            buf1031 = reinterpret_tensor(buf972, (32, 4864), (4864, 1), 0); del buf972  # reuse
            # Topologically Sorted Source Nodes: [permute_1548, mm_927], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1029, (32, 1218), (1, 32), 0), view_485, out=buf1031)
            del view_485
            buf1030 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3386], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1028, buf1030, 28672, stream=stream0)
            buf1033 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3392], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1031, buf1033, 155648, stream=stream0)
            buf1032 = reinterpret_tensor(buf971, (1218, 4864), (4864, 1), 0); del buf971  # reuse
            # Topologically Sorted Source Nodes: [mm_928], Original ATen: [aten.mm]
            extern_kernels.mm(buf1029, permute_1550, out=buf1032)
            del permute_1550
            buf1034 = reinterpret_tensor(buf962, (1218, 4864), (4864, 1), 0); del buf962  # reuse
            # Topologically Sorted Source Nodes: [view_1841, result_210, permute_1552, mm_929], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1026, (1218, 896), (896, 1), 0), primals_262, out=buf1034)
            del primals_262
            buf1035 = buf978; del buf978  # reuse
            buf1042 = buf969; del buf969  # reuse
            buf1044 = reinterpret_tensor(buf961, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf961  # reuse
            buf1051 = reinterpret_tensor(buf959, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf959  # reuse
            # Topologically Sorted Source Nodes: [view_1840, view_1842, add_595, silu_9, mul_854, mul_855, mul_856, convert_element_type_3410, neg_90, exp_14, add_597, reciprocal_14, mul_857, mul_858, sub_14, mul_859, add_598, mul_860, convert_element_type_3412, mul_861], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf1032, buf1034, add_127, add_128, buf1035, buf1042, buf1044, buf1051, 5924352, stream=stream0)
            del add_127
            del add_128
            buf1043 = buf1027; del buf1027  # reuse
            # Topologically Sorted Source Nodes: [view_1840, view_1842, add_595, silu_9, mul_854, view_1847, result_207, permute_1561, mm_934], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1042, (1218, 4864), (4864, 1), 0), primals_259, out=buf1043)
            del primals_259
            buf1052 = buf1023; del buf1023  # reuse
            # Topologically Sorted Source Nodes: [view_1840, view_1842, add_595, silu_9, mul_855, convert_element_type_3410, neg_90, exp_14, add_597, reciprocal_14, mul_857, mul_858, sub_14, mul_859, add_598, mul_860, convert_element_type_3412, view_1853, result_204, permute_1570, mm_939], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1051, (1218, 4864), (4864, 1), 0), primals_256, out=buf1052)
            del primals_256
            buf1037 = buf1029; del buf1029  # reuse
            # Topologically Sorted Source Nodes: [view_1840, view_1842, add_595, silu_9, mul_854, mul_856, view_1843, mm_931], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1035, (1218, 4864), (4864, 1), 0), permute_1555, out=buf1037)
            del permute_1555
            buf1036 = reinterpret_tensor(buf1031, (4864, 32), (32, 1), 0); del buf1031  # reuse
            # Topologically Sorted Source Nodes: [view_1840, view_1842, add_595, silu_9, mul_854, mul_856, view_1843, permute_1553, mm_930], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1035, (4864, 1218), (1, 4864), 0), mm_175, out=buf1036)
            del mm_175
            buf1046 = buf1000; del buf1000  # reuse
            # Topologically Sorted Source Nodes: [view_1840, view_1842, add_595, silu_9, mul_855, convert_element_type_3410, neg_90, exp_14, add_597, reciprocal_14, mul_857, mul_858, sub_14, mul_859, add_598, mul_860, convert_element_type_3412, mul_861, view_1849, mm_936], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1044, (1218, 4864), (4864, 1), 0), permute_1564, out=buf1046)
            del permute_1564
            buf1045 = buf963; del buf963  # reuse
            # Topologically Sorted Source Nodes: [view_1840, view_1842, add_595, silu_9, mul_855, convert_element_type_3410, neg_90, exp_14, add_597, reciprocal_14, mul_857, mul_858, sub_14, mul_859, add_598, mul_860, convert_element_type_3412, mul_861, view_1849, permute_1562, mm_935], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1044, (4864, 1218), (1, 4864), 0), mm_172, out=buf1045)
            del mm_172
            buf1039 = reinterpret_tensor(buf1028, (32, 896), (896, 1), 0); del buf1028  # reuse
            # Topologically Sorted Source Nodes: [permute_1557, mm_932], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1037, (32, 1218), (1, 32), 0), view_473, out=buf1039)
            buf1048 = reinterpret_tensor(buf1016, (32, 896), (896, 1), 0); del buf1016  # reuse
            # Topologically Sorted Source Nodes: [permute_1566, mm_937], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1046, (32, 1218), (1, 32), 0), view_473, out=buf1048)
            del view_473
            buf1041 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3406], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1039, buf1041, 28672, stream=stream0)
            buf1050 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3423], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1048, buf1050, 28672, stream=stream0)
            buf1038 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3400], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1036, buf1038, 155648, stream=stream0)
            buf1047 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3417], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1045, buf1047, 155648, stream=stream0)
            buf1040 = buf1020; del buf1020  # reuse
            # Topologically Sorted Source Nodes: [mm_933], Original ATen: [aten.mm]
            extern_kernels.mm(buf1037, permute_1559, out=buf1040)
            del permute_1559
            buf1049 = buf1014; del buf1014  # reuse
            # Topologically Sorted Source Nodes: [mm_938], Original ATen: [aten.mm]
            extern_kernels.mm(buf1046, permute_1568, out=buf1049)
            del permute_1568
            buf1055 = buf1026; del buf1026  # reuse
            buf1056 = buf1011; del buf1011  # reuse
            # Topologically Sorted Source Nodes: [view_1846, view_1848, add_596, view_1852, add_599, view_1854, add_600, mul_862, convert_element_type_3427, hidden_states_96, mul_863, mul_864, sum_58, pow_108, mul_865, mul_866, expand_106, div_29, pow_109, mul_867, mul_868, add_601, convert_element_type_3428, add_602, mul_869, view_1855], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf1055, buf1040, buf1043, buf1049, buf1052, primals_255, add_125, rsqrt_19, buf1056, 1218, 896, stream=stream0)
            del add_125
            del buf1040
            del primals_255
            del rsqrt_19
            buf1057 = reinterpret_tensor(buf1048, (896, 32), (32, 1), 0); del buf1048  # reuse
            # Topologically Sorted Source Nodes: [permute_1571, mm_940], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1056, (896, 1218), (1, 896), 0), mm_169, out=buf1057)
            del mm_169
            buf1058 = buf1046; del buf1046  # reuse
            # Topologically Sorted Source Nodes: [mm_941], Original ATen: [aten.mm]
            extern_kernels.mm(buf1056, permute_1573, out=buf1058)
            del permute_1573
            buf1060 = buf1039; del buf1039  # reuse
            # Topologically Sorted Source Nodes: [permute_1575, mm_942], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1058, (32, 1218), (1, 32), 0), view_467, out=buf1060)
            del view_467
            buf1059 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3433], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1057, buf1059, 28672, stream=stream0)
            buf1062 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3439], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1060, buf1062, 28672, stream=stream0)
            buf1061 = buf1056; del buf1056  # reuse
            # Topologically Sorted Source Nodes: [mm_943], Original ATen: [aten.mm]
            extern_kernels.mm(buf1058, permute_1577, out=buf1061)
            del permute_1577
            buf1063 = buf1052; del buf1052  # reuse
            # Topologically Sorted Source Nodes: [view_1859, result_201, permute_1579, mm_944], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1055, (1218, 896), (896, 1), 0), primals_252, out=buf1063)
            del primals_252
            buf1064 = reinterpret_tensor(buf1061, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf1061  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1858, view_1860, add_603, view_1861, permute_1580, _scaled_dot_product_efficient_attention_backward_14], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf1064, buf1063, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1858, view_1860, add_603, view_1861, permute_1580, _scaled_dot_product_efficient_attention_backward_14], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf1065 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf1064, add_122, view_462, view_463, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_36, getitem_37, getitem_38, getitem_39, 0.0, [True, True, True, False], scale=0.125)
            del add_122
            del getitem_36
            del getitem_37
            del getitem_38
            del getitem_39
            del view_462
            del view_463
            buf1066 = buf1065[0]
            assert_size_stride(buf1066, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1066, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1067 = buf1065[1]
            assert_size_stride(buf1067, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1067, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1069 = reinterpret_tensor(buf1013, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf1013  # reuse
            # Topologically Sorted Source Nodes: [view_1863, sum_60], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf1067, buf1069, 155904, stream=stream0)
            buf1068 = buf1065[2]
            assert_size_stride(buf1068, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1068, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf1065
            buf1070 = buf997; del buf997  # reuse
            buf1071 = reinterpret_tensor(buf1006, (1218, 128), (128, 1), 0); del buf1006  # reuse
            # Topologically Sorted Source Nodes: [view_1862, sum_59, squeeze_28, permute_1581, clone_92, view_1864, mul_874, view_1865], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf1068, buf1070, buf1071, 155904, stream=stream0)
            buf1072 = buf1007; del buf1007  # reuse
            # Topologically Sorted Source Nodes: [permute_1582, mm_945], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1071, (128, 1218), (1, 128), 0), mm_166, out=buf1072)
            del mm_166
            buf1073 = buf1058; del buf1058  # reuse
            # Topologically Sorted Source Nodes: [mm_946], Original ATen: [aten.mm]
            extern_kernels.mm(buf1071, permute_1584, out=buf1073)
            del permute_1584
            buf1074 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3447], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1072, buf1074, 4096, stream=stream0)
            buf1075 = buf1060; del buf1060  # reuse
            # Topologically Sorted Source Nodes: [permute_1586, mm_947], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1073, (32, 1218), (1, 32), 0), view_443, out=buf1075)
            buf1077 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3453], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1075, buf1077, 28672, stream=stream0)
            buf1079 = reinterpret_tensor(buf1071, (1, 1218, 128), (155904, 128, 1), 0); del buf1071  # reuse
            buf1086 = reinterpret_tensor(buf996, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf996  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1863, sum_60, squeeze_29, mul_870, slice_178, slice_179, neg_91, add_604, mul_871, add_605, permute_1591, clone_93, view_1871, mul_875], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf1069, mm_default, buf1079, buf1086, 1218, 128, stream=stream0)
            buf1080 = buf1072; del buf1072  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1863, sum_60, squeeze_29, mul_870, slice_178, slice_179, neg_91, add_604, mul_871, add_605, permute_1591, clone_93, view_1871, mul_875, view_1872, permute_1592, mm_950], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1079, (128, 1218), (1, 128), 0), mm_164, out=buf1080)
            del mm_164
            buf1081 = buf1037; del buf1037  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1863, sum_60, squeeze_29, mul_870, slice_178, slice_179, neg_91, add_604, mul_871, add_605, permute_1591, clone_93, view_1871, mul_875, view_1872, mm_951], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1079, (1218, 128), (128, 1), 0), permute_1594, out=buf1081)
            del permute_1594
            buf1082 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3461], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1080, buf1082, 4096, stream=stream0)
            buf1083 = buf1075; del buf1075  # reuse
            # Topologically Sorted Source Nodes: [permute_1596, mm_952], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1081, (32, 1218), (1, 32), 0), view_443, out=buf1083)
            buf1085 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3467], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1083, buf1085, 28672, stream=stream0)
            buf1078 = reinterpret_tensor(buf1068, (1218, 896), (896, 1), 0); del buf1068  # reuse
            # Topologically Sorted Source Nodes: [view_1862, sum_59, squeeze_28, permute_1581, clone_92, view_1864, view_1869, result_198, permute_1590, mm_949], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1070, (1218, 128), (128, 1), 0), primals_248, out=buf1078)
            del primals_248
            buf1087 = reinterpret_tensor(buf1067, (1218, 896), (896, 1), 0); del buf1067  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1863, sum_60, squeeze_29, mul_870, slice_178, slice_179, neg_91, add_604, mul_871, add_605, permute_1591, clone_93, view_1871, view_1876, result_195, permute_1600, mm_954], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1086, (1218, 128), (128, 1), 0), primals_244, out=buf1087)
            del primals_244
            buf1076 = reinterpret_tensor(buf1064, (1218, 896), (896, 1), 0); del buf1064  # reuse
            # Topologically Sorted Source Nodes: [mm_948], Original ATen: [aten.mm]
            extern_kernels.mm(buf1073, permute_1588, out=buf1076)
            del permute_1588
            buf1084 = buf1063; del buf1063  # reuse
            # Topologically Sorted Source Nodes: [mm_953], Original ATen: [aten.mm]
            extern_kernels.mm(buf1081, permute_1598, out=buf1084)
            del permute_1598
            buf1088 = reinterpret_tensor(buf1049, (1, 1218, 896), (1091328, 896, 1), 0); del buf1049  # reuse
            buf1095 = reinterpret_tensor(buf1043, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf1043  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_872, slice_180, slice_181, neg_92, add_606, mul_873, add_607, permute_1601, clone_94, view_1878, mul_876], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf1066, mm_default, buf1088, buf1095, 1091328, stream=stream0)
            buf1089 = reinterpret_tensor(buf1083, (896, 32), (32, 1), 0); del buf1083  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_872, slice_180, slice_181, neg_92, add_606, mul_873, add_607, permute_1601, clone_94, view_1878, mul_876, view_1879, permute_1602, mm_955], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1088, (896, 1218), (1, 896), 0), mm_162, out=buf1089)
            del mm_162
            buf1090 = buf1081; del buf1081  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_872, slice_180, slice_181, neg_92, add_606, mul_873, add_607, permute_1601, clone_94, view_1878, mul_876, view_1879, mm_956], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1088, (1218, 896), (896, 1), 0), permute_1604, out=buf1090)
            del permute_1604
            buf1092 = reinterpret_tensor(buf1057, (32, 896), (896, 1), 0); del buf1057  # reuse
            # Topologically Sorted Source Nodes: [permute_1606, mm_957], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1090, (32, 1218), (1, 32), 0), view_443, out=buf1092)
            del view_443
            buf1096 = reinterpret_tensor(buf1088, (1218, 896), (896, 1), 0); del buf1088  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_872, slice_180, slice_181, neg_92, add_606, mul_873, add_607, permute_1601, clone_94, view_1878, view_1883, result_192, permute_1610, mm_959], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1095, (1218, 896), (896, 1), 0), primals_240, out=buf1096)
            del primals_240
            buf1091 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3475], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1089, buf1091, 28672, stream=stream0)
            buf1094 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3481], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1092, buf1094, 28672, stream=stream0)
            buf1093 = reinterpret_tensor(buf1095, (1218, 896), (896, 1), 0); del buf1095  # reuse
            # Topologically Sorted Source Nodes: [mm_958], Original ATen: [aten.mm]
            extern_kernels.mm(buf1090, permute_1608, out=buf1093)
            del permute_1608
            buf1099 = buf1055; del buf1055  # reuse
            buf1100 = reinterpret_tensor(buf1066, (1218, 896), (896, 1), 0); del buf1066  # reuse
            # Topologically Sorted Source Nodes: [view_1868, view_1870, add_608, view_1875, add_609, view_1877, add_610, view_1882, add_611, view_1884, add_612, mul_877, convert_element_type_3485, hidden_states_90, mul_878, mul_879, sum_61, pow_110, mul_880, mul_881, expand_107, div_30, pow_111, mul_882, mul_883, add_613, convert_element_type_3486, add_614, mul_884, view_1885], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf1099, buf1076, buf1078, buf1084, buf1087, buf1093, buf1096, primals_239, add_117, rsqrt_18, buf1100, 1218, 896, stream=stream0)
            del add_117
            del buf1076
            del buf1078
            del primals_239
            del rsqrt_18
            buf1101 = reinterpret_tensor(buf1092, (896, 32), (32, 1), 0); del buf1092  # reuse
            # Topologically Sorted Source Nodes: [permute_1611, mm_960], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1100, (896, 1218), (1, 896), 0), mm_160, out=buf1101)
            del mm_160
            buf1102 = buf1090; del buf1090  # reuse
            # Topologically Sorted Source Nodes: [mm_961], Original ATen: [aten.mm]
            extern_kernels.mm(buf1100, permute_1613, out=buf1102)
            del permute_1613
            buf1104 = reinterpret_tensor(buf1045, (32, 4864), (4864, 1), 0); del buf1045  # reuse
            # Topologically Sorted Source Nodes: [permute_1615, mm_962], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1102, (32, 1218), (1, 32), 0), view_437, out=buf1104)
            del view_437
            buf1103 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3491], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1101, buf1103, 28672, stream=stream0)
            buf1106 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3497], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1104, buf1106, 155648, stream=stream0)
            buf1105 = reinterpret_tensor(buf1044, (1218, 4864), (4864, 1), 0); del buf1044  # reuse
            # Topologically Sorted Source Nodes: [mm_963], Original ATen: [aten.mm]
            extern_kernels.mm(buf1102, permute_1617, out=buf1105)
            del permute_1617
            buf1107 = reinterpret_tensor(buf1035, (1218, 4864), (4864, 1), 0); del buf1035  # reuse
            # Topologically Sorted Source Nodes: [view_1889, result_189, permute_1619, mm_964], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1099, (1218, 896), (896, 1), 0), primals_236, out=buf1107)
            del primals_236
            buf1108 = buf1051; del buf1051  # reuse
            buf1115 = buf1042; del buf1042  # reuse
            buf1117 = reinterpret_tensor(buf1034, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1034  # reuse
            buf1124 = reinterpret_tensor(buf1032, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1032  # reuse
            # Topologically Sorted Source Nodes: [view_1888, view_1890, add_615, silu_8, mul_885, mul_886, mul_887, convert_element_type_3515, neg_93, exp_15, add_617, reciprocal_15, mul_888, mul_889, sub_15, mul_890, add_618, mul_891, convert_element_type_3517, mul_892], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf1105, buf1107, add_114, add_115, buf1108, buf1115, buf1117, buf1124, 5924352, stream=stream0)
            del add_114
            del add_115
            buf1116 = buf1100; del buf1100  # reuse
            # Topologically Sorted Source Nodes: [view_1888, view_1890, add_615, silu_8, mul_885, view_1895, result_186, permute_1628, mm_969], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1115, (1218, 4864), (4864, 1), 0), primals_233, out=buf1116)
            del primals_233
            buf1125 = buf1096; del buf1096  # reuse
            # Topologically Sorted Source Nodes: [view_1888, view_1890, add_615, silu_8, mul_886, convert_element_type_3515, neg_93, exp_15, add_617, reciprocal_15, mul_888, mul_889, sub_15, mul_890, add_618, mul_891, convert_element_type_3517, view_1901, result_183, permute_1637, mm_974], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1124, (1218, 4864), (4864, 1), 0), primals_230, out=buf1125)
            del primals_230
            buf1110 = buf1102; del buf1102  # reuse
            # Topologically Sorted Source Nodes: [view_1888, view_1890, add_615, silu_8, mul_885, mul_887, view_1891, mm_966], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1108, (1218, 4864), (4864, 1), 0), permute_1622, out=buf1110)
            del permute_1622
            buf1109 = reinterpret_tensor(buf1104, (4864, 32), (32, 1), 0); del buf1104  # reuse
            # Topologically Sorted Source Nodes: [view_1888, view_1890, add_615, silu_8, mul_885, mul_887, view_1891, permute_1620, mm_965], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1108, (4864, 1218), (1, 4864), 0), mm_157, out=buf1109)
            del mm_157
            buf1119 = buf1073; del buf1073  # reuse
            # Topologically Sorted Source Nodes: [view_1888, view_1890, add_615, silu_8, mul_886, convert_element_type_3515, neg_93, exp_15, add_617, reciprocal_15, mul_888, mul_889, sub_15, mul_890, add_618, mul_891, convert_element_type_3517, mul_892, view_1897, mm_971], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1117, (1218, 4864), (4864, 1), 0), permute_1631, out=buf1119)
            del permute_1631
            buf1118 = buf1036; del buf1036  # reuse
            # Topologically Sorted Source Nodes: [view_1888, view_1890, add_615, silu_8, mul_886, convert_element_type_3515, neg_93, exp_15, add_617, reciprocal_15, mul_888, mul_889, sub_15, mul_890, add_618, mul_891, convert_element_type_3517, mul_892, view_1897, permute_1629, mm_970], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1117, (4864, 1218), (1, 4864), 0), mm_154, out=buf1118)
            del mm_154
            buf1112 = reinterpret_tensor(buf1101, (32, 896), (896, 1), 0); del buf1101  # reuse
            # Topologically Sorted Source Nodes: [permute_1624, mm_967], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1110, (32, 1218), (1, 32), 0), view_425, out=buf1112)
            buf1121 = reinterpret_tensor(buf1089, (32, 896), (896, 1), 0); del buf1089  # reuse
            # Topologically Sorted Source Nodes: [permute_1633, mm_972], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1119, (32, 1218), (1, 32), 0), view_425, out=buf1121)
            del view_425
            buf1114 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3511], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1112, buf1114, 28672, stream=stream0)
            buf1123 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3528], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1121, buf1123, 28672, stream=stream0)
            buf1111 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3505], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1109, buf1111, 155648, stream=stream0)
            buf1120 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3522], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1118, buf1120, 155648, stream=stream0)
            buf1113 = buf1093; del buf1093  # reuse
            # Topologically Sorted Source Nodes: [mm_968], Original ATen: [aten.mm]
            extern_kernels.mm(buf1110, permute_1626, out=buf1113)
            del permute_1626
            buf1122 = buf1087; del buf1087  # reuse
            # Topologically Sorted Source Nodes: [mm_973], Original ATen: [aten.mm]
            extern_kernels.mm(buf1119, permute_1635, out=buf1122)
            del permute_1635
            buf1128 = buf1099; del buf1099  # reuse
            buf1129 = buf1084; del buf1084  # reuse
            # Topologically Sorted Source Nodes: [view_1894, view_1896, add_616, view_1900, add_619, view_1902, add_620, mul_893, convert_element_type_3532, hidden_states_86, mul_894, mul_895, sum_62, pow_112, mul_896, mul_897, expand_108, div_31, pow_113, mul_898, mul_899, add_621, convert_element_type_3533, add_622, mul_900, view_1903], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf1128, buf1113, buf1116, buf1122, buf1125, primals_229, add_112, rsqrt_17, buf1129, 1218, 896, stream=stream0)
            del add_112
            del buf1113
            del primals_229
            del rsqrt_17
            buf1130 = reinterpret_tensor(buf1121, (896, 32), (32, 1), 0); del buf1121  # reuse
            # Topologically Sorted Source Nodes: [permute_1638, mm_975], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1129, (896, 1218), (1, 896), 0), mm_151, out=buf1130)
            del mm_151
            buf1131 = buf1119; del buf1119  # reuse
            # Topologically Sorted Source Nodes: [mm_976], Original ATen: [aten.mm]
            extern_kernels.mm(buf1129, permute_1640, out=buf1131)
            del permute_1640
            buf1133 = buf1112; del buf1112  # reuse
            # Topologically Sorted Source Nodes: [permute_1642, mm_977], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1131, (32, 1218), (1, 32), 0), view_419, out=buf1133)
            del view_419
            buf1132 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3538], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1130, buf1132, 28672, stream=stream0)
            buf1135 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3544], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1133, buf1135, 28672, stream=stream0)
            buf1134 = buf1129; del buf1129  # reuse
            # Topologically Sorted Source Nodes: [mm_978], Original ATen: [aten.mm]
            extern_kernels.mm(buf1131, permute_1644, out=buf1134)
            del permute_1644
            buf1136 = buf1125; del buf1125  # reuse
            # Topologically Sorted Source Nodes: [view_1907, result_180, permute_1646, mm_979], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1128, (1218, 896), (896, 1), 0), primals_226, out=buf1136)
            del primals_226
            buf1137 = reinterpret_tensor(buf1134, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf1134  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1906, view_1908, add_623, view_1909, permute_1647, _scaled_dot_product_efficient_attention_backward_15], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf1137, buf1136, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1906, view_1908, add_623, view_1909, permute_1647, _scaled_dot_product_efficient_attention_backward_15], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf1138 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf1137, add_109, view_414, view_415, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_32, getitem_33, getitem_34, getitem_35, 0.0, [True, True, True, False], scale=0.125)
            del add_109
            del getitem_32
            del getitem_33
            del getitem_34
            del getitem_35
            del view_414
            del view_415
            buf1139 = buf1138[0]
            assert_size_stride(buf1139, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1139, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1140 = buf1138[1]
            assert_size_stride(buf1140, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1140, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1142 = reinterpret_tensor(buf1086, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf1086  # reuse
            # Topologically Sorted Source Nodes: [view_1911, sum_64], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf1140, buf1142, 155904, stream=stream0)
            buf1141 = buf1138[2]
            assert_size_stride(buf1141, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1141, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf1138
            buf1143 = buf1070; del buf1070  # reuse
            buf1144 = reinterpret_tensor(buf1079, (1218, 128), (128, 1), 0); del buf1079  # reuse
            # Topologically Sorted Source Nodes: [view_1910, sum_63, squeeze_30, permute_1648, clone_95, view_1912, mul_905, view_1913], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf1141, buf1143, buf1144, 155904, stream=stream0)
            buf1145 = buf1080; del buf1080  # reuse
            # Topologically Sorted Source Nodes: [permute_1649, mm_980], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1144, (128, 1218), (1, 128), 0), mm_148, out=buf1145)
            del mm_148
            buf1146 = buf1131; del buf1131  # reuse
            # Topologically Sorted Source Nodes: [mm_981], Original ATen: [aten.mm]
            extern_kernels.mm(buf1144, permute_1651, out=buf1146)
            del permute_1651
            buf1147 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3552], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1145, buf1147, 4096, stream=stream0)
            buf1148 = buf1133; del buf1133  # reuse
            # Topologically Sorted Source Nodes: [permute_1653, mm_982], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1146, (32, 1218), (1, 32), 0), view_395, out=buf1148)
            buf1150 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3558], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1148, buf1150, 28672, stream=stream0)
            buf1152 = reinterpret_tensor(buf1144, (1, 1218, 128), (155904, 128, 1), 0); del buf1144  # reuse
            buf1159 = reinterpret_tensor(buf1069, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf1069  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1911, sum_64, squeeze_31, mul_901, slice_182, slice_183, neg_94, add_624, mul_902, add_625, permute_1658, clone_96, view_1919, mul_906], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf1142, mm_default, buf1152, buf1159, 1218, 128, stream=stream0)
            buf1153 = buf1145; del buf1145  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1911, sum_64, squeeze_31, mul_901, slice_182, slice_183, neg_94, add_624, mul_902, add_625, permute_1658, clone_96, view_1919, mul_906, view_1920, permute_1659, mm_985], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1152, (128, 1218), (1, 128), 0), mm_146, out=buf1153)
            del mm_146
            buf1154 = buf1110; del buf1110  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1911, sum_64, squeeze_31, mul_901, slice_182, slice_183, neg_94, add_624, mul_902, add_625, permute_1658, clone_96, view_1919, mul_906, view_1920, mm_986], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1152, (1218, 128), (128, 1), 0), permute_1661, out=buf1154)
            del permute_1661
            buf1155 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3566], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1153, buf1155, 4096, stream=stream0)
            buf1156 = buf1148; del buf1148  # reuse
            # Topologically Sorted Source Nodes: [permute_1663, mm_987], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1154, (32, 1218), (1, 32), 0), view_395, out=buf1156)
            buf1158 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3572], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1156, buf1158, 28672, stream=stream0)
            buf1151 = reinterpret_tensor(buf1141, (1218, 896), (896, 1), 0); del buf1141  # reuse
            # Topologically Sorted Source Nodes: [view_1910, sum_63, squeeze_30, permute_1648, clone_95, view_1912, view_1917, result_177, permute_1657, mm_984], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1143, (1218, 128), (128, 1), 0), primals_222, out=buf1151)
            del primals_222
            buf1160 = reinterpret_tensor(buf1140, (1218, 896), (896, 1), 0); del buf1140  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1911, sum_64, squeeze_31, mul_901, slice_182, slice_183, neg_94, add_624, mul_902, add_625, permute_1658, clone_96, view_1919, view_1924, result_174, permute_1667, mm_989], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1159, (1218, 128), (128, 1), 0), primals_218, out=buf1160)
            del primals_218
            buf1149 = reinterpret_tensor(buf1137, (1218, 896), (896, 1), 0); del buf1137  # reuse
            # Topologically Sorted Source Nodes: [mm_983], Original ATen: [aten.mm]
            extern_kernels.mm(buf1146, permute_1655, out=buf1149)
            del permute_1655
            buf1157 = buf1136; del buf1136  # reuse
            # Topologically Sorted Source Nodes: [mm_988], Original ATen: [aten.mm]
            extern_kernels.mm(buf1154, permute_1665, out=buf1157)
            del permute_1665
            buf1161 = reinterpret_tensor(buf1122, (1, 1218, 896), (1091328, 896, 1), 0); del buf1122  # reuse
            buf1168 = reinterpret_tensor(buf1116, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf1116  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_903, slice_184, slice_185, neg_95, add_626, mul_904, add_627, permute_1668, clone_97, view_1926, mul_907], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf1139, mm_default, buf1161, buf1168, 1091328, stream=stream0)
            buf1162 = reinterpret_tensor(buf1156, (896, 32), (32, 1), 0); del buf1156  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_903, slice_184, slice_185, neg_95, add_626, mul_904, add_627, permute_1668, clone_97, view_1926, mul_907, view_1927, permute_1669, mm_990], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1161, (896, 1218), (1, 896), 0), mm_144, out=buf1162)
            del mm_144
            buf1163 = buf1154; del buf1154  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_903, slice_184, slice_185, neg_95, add_626, mul_904, add_627, permute_1668, clone_97, view_1926, mul_907, view_1927, mm_991], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1161, (1218, 896), (896, 1), 0), permute_1671, out=buf1163)
            del permute_1671
            buf1165 = reinterpret_tensor(buf1130, (32, 896), (896, 1), 0); del buf1130  # reuse
            # Topologically Sorted Source Nodes: [permute_1673, mm_992], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1163, (32, 1218), (1, 32), 0), view_395, out=buf1165)
            del view_395
            buf1169 = reinterpret_tensor(buf1161, (1218, 896), (896, 1), 0); del buf1161  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_903, slice_184, slice_185, neg_95, add_626, mul_904, add_627, permute_1668, clone_97, view_1926, view_1931, result_171, permute_1677, mm_994], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1168, (1218, 896), (896, 1), 0), primals_214, out=buf1169)
            del primals_214
            buf1164 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3580], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1162, buf1164, 28672, stream=stream0)
            buf1167 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3586], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1165, buf1167, 28672, stream=stream0)
            buf1166 = reinterpret_tensor(buf1168, (1218, 896), (896, 1), 0); del buf1168  # reuse
            # Topologically Sorted Source Nodes: [mm_993], Original ATen: [aten.mm]
            extern_kernels.mm(buf1163, permute_1675, out=buf1166)
            del permute_1675
            buf1172 = buf1128; del buf1128  # reuse
            buf1173 = reinterpret_tensor(buf1139, (1218, 896), (896, 1), 0); del buf1139  # reuse
            # Topologically Sorted Source Nodes: [view_1916, view_1918, add_628, view_1923, add_629, view_1925, add_630, view_1930, add_631, view_1932, add_632, mul_908, convert_element_type_3590, hidden_states_80, mul_909, mul_910, sum_65, pow_114, mul_911, mul_912, expand_109, div_32, pow_115, mul_913, mul_914, add_633, convert_element_type_3591, add_634, mul_915, view_1933], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf1172, buf1149, buf1151, buf1157, buf1160, buf1166, buf1169, primals_213, add_104, rsqrt_16, buf1173, 1218, 896, stream=stream0)
            del add_104
            del buf1149
            del buf1151
            del primals_213
            del rsqrt_16
            buf1174 = reinterpret_tensor(buf1165, (896, 32), (32, 1), 0); del buf1165  # reuse
            # Topologically Sorted Source Nodes: [permute_1678, mm_995], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1173, (896, 1218), (1, 896), 0), mm_142, out=buf1174)
            del mm_142
            buf1175 = buf1163; del buf1163  # reuse
            # Topologically Sorted Source Nodes: [mm_996], Original ATen: [aten.mm]
            extern_kernels.mm(buf1173, permute_1680, out=buf1175)
            del permute_1680
            buf1177 = reinterpret_tensor(buf1118, (32, 4864), (4864, 1), 0); del buf1118  # reuse
            # Topologically Sorted Source Nodes: [permute_1682, mm_997], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1175, (32, 1218), (1, 32), 0), view_389, out=buf1177)
            del view_389
            buf1176 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3596], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1174, buf1176, 28672, stream=stream0)
            buf1179 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3602], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1177, buf1179, 155648, stream=stream0)
            buf1178 = reinterpret_tensor(buf1117, (1218, 4864), (4864, 1), 0); del buf1117  # reuse
            # Topologically Sorted Source Nodes: [mm_998], Original ATen: [aten.mm]
            extern_kernels.mm(buf1175, permute_1684, out=buf1178)
            del permute_1684
            buf1180 = reinterpret_tensor(buf1108, (1218, 4864), (4864, 1), 0); del buf1108  # reuse
            # Topologically Sorted Source Nodes: [view_1937, result_168, permute_1686, mm_999], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1172, (1218, 896), (896, 1), 0), primals_210, out=buf1180)
            del primals_210
            buf1181 = buf1124; del buf1124  # reuse
            buf1188 = buf1115; del buf1115  # reuse
            buf1190 = reinterpret_tensor(buf1107, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1107  # reuse
            buf1197 = reinterpret_tensor(buf1105, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1105  # reuse
            # Topologically Sorted Source Nodes: [view_1936, view_1938, add_635, silu_7, mul_916, mul_917, mul_918, convert_element_type_3620, neg_96, exp_16, add_637, reciprocal_16, mul_919, mul_920, sub_16, mul_921, add_638, mul_922, convert_element_type_3622, mul_923], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf1178, buf1180, add_101, add_102, buf1181, buf1188, buf1190, buf1197, 5924352, stream=stream0)
            del add_101
            del add_102
            buf1189 = buf1173; del buf1173  # reuse
            # Topologically Sorted Source Nodes: [view_1936, view_1938, add_635, silu_7, mul_916, view_1943, result_165, permute_1695, mm_1004], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1188, (1218, 4864), (4864, 1), 0), primals_207, out=buf1189)
            del primals_207
            buf1198 = buf1169; del buf1169  # reuse
            # Topologically Sorted Source Nodes: [view_1936, view_1938, add_635, silu_7, mul_917, convert_element_type_3620, neg_96, exp_16, add_637, reciprocal_16, mul_919, mul_920, sub_16, mul_921, add_638, mul_922, convert_element_type_3622, view_1949, result_162, permute_1704, mm_1009], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1197, (1218, 4864), (4864, 1), 0), primals_204, out=buf1198)
            del primals_204
            buf1183 = buf1175; del buf1175  # reuse
            # Topologically Sorted Source Nodes: [view_1936, view_1938, add_635, silu_7, mul_916, mul_918, view_1939, mm_1001], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1181, (1218, 4864), (4864, 1), 0), permute_1689, out=buf1183)
            del permute_1689
            buf1182 = reinterpret_tensor(buf1177, (4864, 32), (32, 1), 0); del buf1177  # reuse
            # Topologically Sorted Source Nodes: [view_1936, view_1938, add_635, silu_7, mul_916, mul_918, view_1939, permute_1687, mm_1000], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1181, (4864, 1218), (1, 4864), 0), mm_139, out=buf1182)
            del mm_139
            buf1192 = buf1146; del buf1146  # reuse
            # Topologically Sorted Source Nodes: [view_1936, view_1938, add_635, silu_7, mul_917, convert_element_type_3620, neg_96, exp_16, add_637, reciprocal_16, mul_919, mul_920, sub_16, mul_921, add_638, mul_922, convert_element_type_3622, mul_923, view_1945, mm_1006], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1190, (1218, 4864), (4864, 1), 0), permute_1698, out=buf1192)
            del permute_1698
            buf1191 = buf1109; del buf1109  # reuse
            # Topologically Sorted Source Nodes: [view_1936, view_1938, add_635, silu_7, mul_917, convert_element_type_3620, neg_96, exp_16, add_637, reciprocal_16, mul_919, mul_920, sub_16, mul_921, add_638, mul_922, convert_element_type_3622, mul_923, view_1945, permute_1696, mm_1005], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1190, (4864, 1218), (1, 4864), 0), mm_136, out=buf1191)
            del mm_136
            buf1185 = reinterpret_tensor(buf1174, (32, 896), (896, 1), 0); del buf1174  # reuse
            # Topologically Sorted Source Nodes: [permute_1691, mm_1002], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1183, (32, 1218), (1, 32), 0), view_377, out=buf1185)
            buf1194 = reinterpret_tensor(buf1162, (32, 896), (896, 1), 0); del buf1162  # reuse
            # Topologically Sorted Source Nodes: [permute_1700, mm_1007], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1192, (32, 1218), (1, 32), 0), view_377, out=buf1194)
            del view_377
            buf1187 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3616], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1185, buf1187, 28672, stream=stream0)
            buf1196 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3633], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1194, buf1196, 28672, stream=stream0)
            buf1184 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3610], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1182, buf1184, 155648, stream=stream0)
            buf1193 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3627], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1191, buf1193, 155648, stream=stream0)
            buf1186 = buf1166; del buf1166  # reuse
            # Topologically Sorted Source Nodes: [mm_1003], Original ATen: [aten.mm]
            extern_kernels.mm(buf1183, permute_1693, out=buf1186)
            del permute_1693
            buf1195 = buf1160; del buf1160  # reuse
            # Topologically Sorted Source Nodes: [mm_1008], Original ATen: [aten.mm]
            extern_kernels.mm(buf1192, permute_1702, out=buf1195)
            del permute_1702
            buf1201 = buf1172; del buf1172  # reuse
            buf1202 = buf1157; del buf1157  # reuse
            # Topologically Sorted Source Nodes: [view_1942, view_1944, add_636, view_1948, add_639, view_1950, add_640, mul_924, convert_element_type_3637, hidden_states_76, mul_925, mul_926, sum_66, pow_116, mul_927, mul_928, expand_110, div_33, pow_117, mul_929, mul_930, add_641, convert_element_type_3638, add_642, mul_931, view_1951], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf1201, buf1186, buf1189, buf1195, buf1198, primals_203, add_99, rsqrt_15, buf1202, 1218, 896, stream=stream0)
            del add_99
            del buf1186
            del primals_203
            del rsqrt_15
            buf1203 = reinterpret_tensor(buf1194, (896, 32), (32, 1), 0); del buf1194  # reuse
            # Topologically Sorted Source Nodes: [permute_1705, mm_1010], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1202, (896, 1218), (1, 896), 0), mm_133, out=buf1203)
            del mm_133
            buf1204 = buf1192; del buf1192  # reuse
            # Topologically Sorted Source Nodes: [mm_1011], Original ATen: [aten.mm]
            extern_kernels.mm(buf1202, permute_1707, out=buf1204)
            del permute_1707
            buf1206 = buf1185; del buf1185  # reuse
            # Topologically Sorted Source Nodes: [permute_1709, mm_1012], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1204, (32, 1218), (1, 32), 0), view_371, out=buf1206)
            del view_371
            buf1205 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3643], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1203, buf1205, 28672, stream=stream0)
            buf1208 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3649], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1206, buf1208, 28672, stream=stream0)
            buf1207 = buf1202; del buf1202  # reuse
            # Topologically Sorted Source Nodes: [mm_1013], Original ATen: [aten.mm]
            extern_kernels.mm(buf1204, permute_1711, out=buf1207)
            del permute_1711
            buf1209 = buf1198; del buf1198  # reuse
            # Topologically Sorted Source Nodes: [view_1955, result_159, permute_1713, mm_1014], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1201, (1218, 896), (896, 1), 0), primals_200, out=buf1209)
            del primals_200
            buf1210 = reinterpret_tensor(buf1207, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf1207  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_1954, view_1956, add_643, view_1957, permute_1714, _scaled_dot_product_efficient_attention_backward_16], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf1210, buf1209, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_1954, view_1956, add_643, view_1957, permute_1714, _scaled_dot_product_efficient_attention_backward_16], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf1211 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf1210, add_96, view_366, view_367, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_28, getitem_29, getitem_30, getitem_31, 0.0, [True, True, True, False], scale=0.125)
            del add_96
            del getitem_28
            del getitem_29
            del getitem_30
            del getitem_31
            del view_366
            del view_367
            buf1212 = buf1211[0]
            assert_size_stride(buf1212, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1212, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1213 = buf1211[1]
            assert_size_stride(buf1213, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1213, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1215 = reinterpret_tensor(buf1159, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf1159  # reuse
            # Topologically Sorted Source Nodes: [view_1959, sum_68], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf1213, buf1215, 155904, stream=stream0)
            buf1214 = buf1211[2]
            assert_size_stride(buf1214, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1214, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf1211
            buf1216 = buf1143; del buf1143  # reuse
            buf1217 = reinterpret_tensor(buf1152, (1218, 128), (128, 1), 0); del buf1152  # reuse
            # Topologically Sorted Source Nodes: [view_1958, sum_67, squeeze_32, permute_1715, clone_98, view_1960, mul_936, view_1961], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf1214, buf1216, buf1217, 155904, stream=stream0)
            buf1218 = buf1153; del buf1153  # reuse
            # Topologically Sorted Source Nodes: [permute_1716, mm_1015], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1217, (128, 1218), (1, 128), 0), mm_130, out=buf1218)
            del mm_130
            buf1219 = buf1204; del buf1204  # reuse
            # Topologically Sorted Source Nodes: [mm_1016], Original ATen: [aten.mm]
            extern_kernels.mm(buf1217, permute_1718, out=buf1219)
            del permute_1718
            buf1220 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3657], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1218, buf1220, 4096, stream=stream0)
            buf1221 = buf1206; del buf1206  # reuse
            # Topologically Sorted Source Nodes: [permute_1720, mm_1017], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1219, (32, 1218), (1, 32), 0), view_347, out=buf1221)
            buf1223 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3663], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1221, buf1223, 28672, stream=stream0)
            buf1225 = reinterpret_tensor(buf1217, (1, 1218, 128), (155904, 128, 1), 0); del buf1217  # reuse
            buf1232 = reinterpret_tensor(buf1142, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf1142  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1959, sum_68, squeeze_33, mul_932, slice_186, slice_187, neg_97, add_644, mul_933, add_645, permute_1725, clone_99, view_1967, mul_937], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf1215, mm_default, buf1225, buf1232, 1218, 128, stream=stream0)
            buf1226 = buf1218; del buf1218  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1959, sum_68, squeeze_33, mul_932, slice_186, slice_187, neg_97, add_644, mul_933, add_645, permute_1725, clone_99, view_1967, mul_937, view_1968, permute_1726, mm_1020], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1225, (128, 1218), (1, 128), 0), mm_128, out=buf1226)
            del mm_128
            buf1227 = buf1183; del buf1183  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1959, sum_68, squeeze_33, mul_932, slice_186, slice_187, neg_97, add_644, mul_933, add_645, permute_1725, clone_99, view_1967, mul_937, view_1968, mm_1021], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1225, (1218, 128), (128, 1), 0), permute_1728, out=buf1227)
            del permute_1728
            buf1228 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3671], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1226, buf1228, 4096, stream=stream0)
            buf1229 = buf1221; del buf1221  # reuse
            # Topologically Sorted Source Nodes: [permute_1730, mm_1022], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1227, (32, 1218), (1, 32), 0), view_347, out=buf1229)
            buf1231 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3677], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1229, buf1231, 28672, stream=stream0)
            buf1224 = reinterpret_tensor(buf1214, (1218, 896), (896, 1), 0); del buf1214  # reuse
            # Topologically Sorted Source Nodes: [view_1958, sum_67, squeeze_32, permute_1715, clone_98, view_1960, view_1965, result_156, permute_1724, mm_1019], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1216, (1218, 128), (128, 1), 0), primals_196, out=buf1224)
            del primals_196
            buf1233 = reinterpret_tensor(buf1213, (1218, 896), (896, 1), 0); del buf1213  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_1959, sum_68, squeeze_33, mul_932, slice_186, slice_187, neg_97, add_644, mul_933, add_645, permute_1725, clone_99, view_1967, view_1972, result_153, permute_1734, mm_1024], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1232, (1218, 128), (128, 1), 0), primals_192, out=buf1233)
            del primals_192
            buf1222 = reinterpret_tensor(buf1210, (1218, 896), (896, 1), 0); del buf1210  # reuse
            # Topologically Sorted Source Nodes: [mm_1018], Original ATen: [aten.mm]
            extern_kernels.mm(buf1219, permute_1722, out=buf1222)
            del permute_1722
            buf1230 = buf1209; del buf1209  # reuse
            # Topologically Sorted Source Nodes: [mm_1023], Original ATen: [aten.mm]
            extern_kernels.mm(buf1227, permute_1732, out=buf1230)
            del permute_1732
            buf1234 = reinterpret_tensor(buf1195, (1, 1218, 896), (1091328, 896, 1), 0); del buf1195  # reuse
            buf1241 = reinterpret_tensor(buf1189, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf1189  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_934, slice_188, slice_189, neg_98, add_646, mul_935, add_647, permute_1735, clone_100, view_1974, mul_938], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf1212, mm_default, buf1234, buf1241, 1091328, stream=stream0)
            buf1235 = reinterpret_tensor(buf1229, (896, 32), (32, 1), 0); del buf1229  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_934, slice_188, slice_189, neg_98, add_646, mul_935, add_647, permute_1735, clone_100, view_1974, mul_938, view_1975, permute_1736, mm_1025], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1234, (896, 1218), (1, 896), 0), mm_126, out=buf1235)
            del mm_126
            buf1236 = buf1227; del buf1227  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_934, slice_188, slice_189, neg_98, add_646, mul_935, add_647, permute_1735, clone_100, view_1974, mul_938, view_1975, mm_1026], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1234, (1218, 896), (896, 1), 0), permute_1738, out=buf1236)
            del permute_1738
            buf1238 = reinterpret_tensor(buf1203, (32, 896), (896, 1), 0); del buf1203  # reuse
            # Topologically Sorted Source Nodes: [permute_1740, mm_1027], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1236, (32, 1218), (1, 32), 0), view_347, out=buf1238)
            del view_347
            buf1242 = reinterpret_tensor(buf1234, (1218, 896), (896, 1), 0); del buf1234  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_934, slice_188, slice_189, neg_98, add_646, mul_935, add_647, permute_1735, clone_100, view_1974, view_1979, result_150, permute_1744, mm_1029], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1241, (1218, 896), (896, 1), 0), primals_188, out=buf1242)
            del primals_188
            buf1237 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3685], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1235, buf1237, 28672, stream=stream0)
            buf1240 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3691], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1238, buf1240, 28672, stream=stream0)
            buf1239 = reinterpret_tensor(buf1241, (1218, 896), (896, 1), 0); del buf1241  # reuse
            # Topologically Sorted Source Nodes: [mm_1028], Original ATen: [aten.mm]
            extern_kernels.mm(buf1236, permute_1742, out=buf1239)
            del permute_1742
            buf1245 = buf1201; del buf1201  # reuse
            buf1246 = reinterpret_tensor(buf1212, (1218, 896), (896, 1), 0); del buf1212  # reuse
            # Topologically Sorted Source Nodes: [view_1964, view_1966, add_648, view_1971, add_649, view_1973, add_650, view_1978, add_651, view_1980, add_652, mul_939, convert_element_type_3695, hidden_states_70, mul_940, mul_941, sum_69, pow_118, mul_942, mul_943, expand_111, div_34, pow_119, mul_944, mul_945, add_653, convert_element_type_3696, add_654, mul_946, view_1981], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf1245, buf1222, buf1224, buf1230, buf1233, buf1239, buf1242, primals_187, add_91, rsqrt_14, buf1246, 1218, 896, stream=stream0)
            del add_91
            del buf1222
            del buf1224
            del primals_187
            del rsqrt_14
            buf1247 = reinterpret_tensor(buf1238, (896, 32), (32, 1), 0); del buf1238  # reuse
            # Topologically Sorted Source Nodes: [permute_1745, mm_1030], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1246, (896, 1218), (1, 896), 0), mm_124, out=buf1247)
            del mm_124
            buf1248 = buf1236; del buf1236  # reuse
            # Topologically Sorted Source Nodes: [mm_1031], Original ATen: [aten.mm]
            extern_kernels.mm(buf1246, permute_1747, out=buf1248)
            del permute_1747
            buf1250 = reinterpret_tensor(buf1191, (32, 4864), (4864, 1), 0); del buf1191  # reuse
            # Topologically Sorted Source Nodes: [permute_1749, mm_1032], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1248, (32, 1218), (1, 32), 0), view_341, out=buf1250)
            del view_341
            buf1249 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3701], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1247, buf1249, 28672, stream=stream0)
            buf1252 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3707], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1250, buf1252, 155648, stream=stream0)
            buf1251 = reinterpret_tensor(buf1190, (1218, 4864), (4864, 1), 0); del buf1190  # reuse
            # Topologically Sorted Source Nodes: [mm_1033], Original ATen: [aten.mm]
            extern_kernels.mm(buf1248, permute_1751, out=buf1251)
            del permute_1751
            buf1253 = reinterpret_tensor(buf1181, (1218, 4864), (4864, 1), 0); del buf1181  # reuse
            # Topologically Sorted Source Nodes: [view_1985, result_147, permute_1753, mm_1034], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1245, (1218, 896), (896, 1), 0), primals_184, out=buf1253)
            del primals_184
            buf1254 = buf1197; del buf1197  # reuse
            buf1261 = buf1188; del buf1188  # reuse
            buf1263 = reinterpret_tensor(buf1180, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1180  # reuse
            buf1270 = reinterpret_tensor(buf1178, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1178  # reuse
            # Topologically Sorted Source Nodes: [view_1984, view_1986, add_655, silu_6, mul_947, mul_948, mul_949, convert_element_type_3725, neg_99, exp_17, add_657, reciprocal_17, mul_950, mul_951, sub_17, mul_952, add_658, mul_953, convert_element_type_3727, mul_954], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf1251, buf1253, add_88, add_89, buf1254, buf1261, buf1263, buf1270, 5924352, stream=stream0)
            del add_88
            del add_89
            buf1262 = buf1246; del buf1246  # reuse
            # Topologically Sorted Source Nodes: [view_1984, view_1986, add_655, silu_6, mul_947, view_1991, result_144, permute_1762, mm_1039], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1261, (1218, 4864), (4864, 1), 0), primals_181, out=buf1262)
            del primals_181
            buf1271 = buf1242; del buf1242  # reuse
            # Topologically Sorted Source Nodes: [view_1984, view_1986, add_655, silu_6, mul_948, convert_element_type_3725, neg_99, exp_17, add_657, reciprocal_17, mul_950, mul_951, sub_17, mul_952, add_658, mul_953, convert_element_type_3727, view_1997, result_141, permute_1771, mm_1044], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1270, (1218, 4864), (4864, 1), 0), primals_178, out=buf1271)
            del primals_178
            buf1256 = buf1248; del buf1248  # reuse
            # Topologically Sorted Source Nodes: [view_1984, view_1986, add_655, silu_6, mul_947, mul_949, view_1987, mm_1036], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1254, (1218, 4864), (4864, 1), 0), permute_1756, out=buf1256)
            del permute_1756
            buf1255 = reinterpret_tensor(buf1250, (4864, 32), (32, 1), 0); del buf1250  # reuse
            # Topologically Sorted Source Nodes: [view_1984, view_1986, add_655, silu_6, mul_947, mul_949, view_1987, permute_1754, mm_1035], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1254, (4864, 1218), (1, 4864), 0), mm_121, out=buf1255)
            del mm_121
            buf1265 = buf1219; del buf1219  # reuse
            # Topologically Sorted Source Nodes: [view_1984, view_1986, add_655, silu_6, mul_948, convert_element_type_3725, neg_99, exp_17, add_657, reciprocal_17, mul_950, mul_951, sub_17, mul_952, add_658, mul_953, convert_element_type_3727, mul_954, view_1993, mm_1041], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1263, (1218, 4864), (4864, 1), 0), permute_1765, out=buf1265)
            del permute_1765
            buf1264 = buf1182; del buf1182  # reuse
            # Topologically Sorted Source Nodes: [view_1984, view_1986, add_655, silu_6, mul_948, convert_element_type_3725, neg_99, exp_17, add_657, reciprocal_17, mul_950, mul_951, sub_17, mul_952, add_658, mul_953, convert_element_type_3727, mul_954, view_1993, permute_1763, mm_1040], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1263, (4864, 1218), (1, 4864), 0), mm_118, out=buf1264)
            del mm_118
            buf1258 = reinterpret_tensor(buf1247, (32, 896), (896, 1), 0); del buf1247  # reuse
            # Topologically Sorted Source Nodes: [permute_1758, mm_1037], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1256, (32, 1218), (1, 32), 0), view_329, out=buf1258)
            buf1267 = reinterpret_tensor(buf1235, (32, 896), (896, 1), 0); del buf1235  # reuse
            # Topologically Sorted Source Nodes: [permute_1767, mm_1042], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1265, (32, 1218), (1, 32), 0), view_329, out=buf1267)
            del view_329
            buf1260 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3721], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1258, buf1260, 28672, stream=stream0)
            buf1269 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3738], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1267, buf1269, 28672, stream=stream0)
            buf1257 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3715], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1255, buf1257, 155648, stream=stream0)
            buf1266 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3732], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1264, buf1266, 155648, stream=stream0)
            buf1259 = buf1239; del buf1239  # reuse
            # Topologically Sorted Source Nodes: [mm_1038], Original ATen: [aten.mm]
            extern_kernels.mm(buf1256, permute_1760, out=buf1259)
            del permute_1760
            buf1268 = buf1233; del buf1233  # reuse
            # Topologically Sorted Source Nodes: [mm_1043], Original ATen: [aten.mm]
            extern_kernels.mm(buf1265, permute_1769, out=buf1268)
            del permute_1769
            buf1274 = buf1245; del buf1245  # reuse
            buf1275 = buf1230; del buf1230  # reuse
            # Topologically Sorted Source Nodes: [view_1990, view_1992, add_656, view_1996, add_659, view_1998, add_660, mul_955, convert_element_type_3742, hidden_states_66, mul_956, mul_957, sum_70, pow_120, mul_958, mul_959, expand_112, div_35, pow_121, mul_960, mul_961, add_661, convert_element_type_3743, add_662, mul_962, view_1999], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf1274, buf1259, buf1262, buf1268, buf1271, primals_177, add_86, rsqrt_13, buf1275, 1218, 896, stream=stream0)
            del add_86
            del buf1259
            del primals_177
            del rsqrt_13
            buf1276 = reinterpret_tensor(buf1267, (896, 32), (32, 1), 0); del buf1267  # reuse
            # Topologically Sorted Source Nodes: [permute_1772, mm_1045], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1275, (896, 1218), (1, 896), 0), mm_115, out=buf1276)
            del mm_115
            buf1277 = buf1265; del buf1265  # reuse
            # Topologically Sorted Source Nodes: [mm_1046], Original ATen: [aten.mm]
            extern_kernels.mm(buf1275, permute_1774, out=buf1277)
            del permute_1774
            buf1279 = buf1258; del buf1258  # reuse
            # Topologically Sorted Source Nodes: [permute_1776, mm_1047], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1277, (32, 1218), (1, 32), 0), view_323, out=buf1279)
            del view_323
            buf1278 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3748], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1276, buf1278, 28672, stream=stream0)
            buf1281 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3754], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1279, buf1281, 28672, stream=stream0)
            buf1280 = buf1275; del buf1275  # reuse
            # Topologically Sorted Source Nodes: [mm_1048], Original ATen: [aten.mm]
            extern_kernels.mm(buf1277, permute_1778, out=buf1280)
            del permute_1778
            buf1282 = buf1271; del buf1271  # reuse
            # Topologically Sorted Source Nodes: [view_2003, result_138, permute_1780, mm_1049], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1274, (1218, 896), (896, 1), 0), primals_174, out=buf1282)
            del primals_174
            buf1283 = reinterpret_tensor(buf1280, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf1280  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_2002, view_2004, add_663, view_2005, permute_1781, _scaled_dot_product_efficient_attention_backward_17], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf1283, buf1282, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_2002, view_2004, add_663, view_2005, permute_1781, _scaled_dot_product_efficient_attention_backward_17], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf1284 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf1283, add_83, view_318, view_319, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_24, getitem_25, getitem_26, getitem_27, 0.0, [True, True, True, False], scale=0.125)
            del add_83
            del getitem_24
            del getitem_25
            del getitem_26
            del getitem_27
            del view_318
            del view_319
            buf1285 = buf1284[0]
            assert_size_stride(buf1285, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1285, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1286 = buf1284[1]
            assert_size_stride(buf1286, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1286, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1288 = reinterpret_tensor(buf1232, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf1232  # reuse
            # Topologically Sorted Source Nodes: [view_2007, sum_72], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf1286, buf1288, 155904, stream=stream0)
            buf1287 = buf1284[2]
            assert_size_stride(buf1287, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1287, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf1284
            buf1289 = buf1216; del buf1216  # reuse
            buf1290 = reinterpret_tensor(buf1225, (1218, 128), (128, 1), 0); del buf1225  # reuse
            # Topologically Sorted Source Nodes: [view_2006, sum_71, squeeze_34, permute_1782, clone_101, view_2008, mul_967, view_2009], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf1287, buf1289, buf1290, 155904, stream=stream0)
            buf1291 = buf1226; del buf1226  # reuse
            # Topologically Sorted Source Nodes: [permute_1783, mm_1050], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1290, (128, 1218), (1, 128), 0), mm_112, out=buf1291)
            del mm_112
            buf1292 = buf1277; del buf1277  # reuse
            # Topologically Sorted Source Nodes: [mm_1051], Original ATen: [aten.mm]
            extern_kernels.mm(buf1290, permute_1785, out=buf1292)
            del permute_1785
            buf1293 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3762], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1291, buf1293, 4096, stream=stream0)
            buf1294 = buf1279; del buf1279  # reuse
            # Topologically Sorted Source Nodes: [permute_1787, mm_1052], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1292, (32, 1218), (1, 32), 0), view_299, out=buf1294)
            buf1296 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3768], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1294, buf1296, 28672, stream=stream0)
            buf1298 = reinterpret_tensor(buf1290, (1, 1218, 128), (155904, 128, 1), 0); del buf1290  # reuse
            buf1305 = reinterpret_tensor(buf1215, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf1215  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2007, sum_72, squeeze_35, mul_963, slice_190, slice_191, neg_100, add_664, mul_964, add_665, permute_1792, clone_102, view_2015, mul_968], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf1288, mm_default, buf1298, buf1305, 1218, 128, stream=stream0)
            buf1299 = buf1291; del buf1291  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2007, sum_72, squeeze_35, mul_963, slice_190, slice_191, neg_100, add_664, mul_964, add_665, permute_1792, clone_102, view_2015, mul_968, view_2016, permute_1793, mm_1055], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1298, (128, 1218), (1, 128), 0), mm_110, out=buf1299)
            del mm_110
            buf1300 = buf1256; del buf1256  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2007, sum_72, squeeze_35, mul_963, slice_190, slice_191, neg_100, add_664, mul_964, add_665, permute_1792, clone_102, view_2015, mul_968, view_2016, mm_1056], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1298, (1218, 128), (128, 1), 0), permute_1795, out=buf1300)
            del permute_1795
            buf1301 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3776], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1299, buf1301, 4096, stream=stream0)
            buf1302 = buf1294; del buf1294  # reuse
            # Topologically Sorted Source Nodes: [permute_1797, mm_1057], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1300, (32, 1218), (1, 32), 0), view_299, out=buf1302)
            buf1304 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3782], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1302, buf1304, 28672, stream=stream0)
            buf1297 = reinterpret_tensor(buf1287, (1218, 896), (896, 1), 0); del buf1287  # reuse
            # Topologically Sorted Source Nodes: [view_2006, sum_71, squeeze_34, permute_1782, clone_101, view_2008, view_2013, result_135, permute_1791, mm_1054], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1289, (1218, 128), (128, 1), 0), primals_170, out=buf1297)
            del primals_170
            buf1306 = reinterpret_tensor(buf1286, (1218, 896), (896, 1), 0); del buf1286  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2007, sum_72, squeeze_35, mul_963, slice_190, slice_191, neg_100, add_664, mul_964, add_665, permute_1792, clone_102, view_2015, view_2020, result_132, permute_1801, mm_1059], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1305, (1218, 128), (128, 1), 0), primals_166, out=buf1306)
            del primals_166
            buf1295 = reinterpret_tensor(buf1283, (1218, 896), (896, 1), 0); del buf1283  # reuse
            # Topologically Sorted Source Nodes: [mm_1053], Original ATen: [aten.mm]
            extern_kernels.mm(buf1292, permute_1789, out=buf1295)
            del permute_1789
            buf1303 = buf1282; del buf1282  # reuse
            # Topologically Sorted Source Nodes: [mm_1058], Original ATen: [aten.mm]
            extern_kernels.mm(buf1300, permute_1799, out=buf1303)
            del permute_1799
            buf1307 = reinterpret_tensor(buf1268, (1, 1218, 896), (1091328, 896, 1), 0); del buf1268  # reuse
            buf1314 = reinterpret_tensor(buf1262, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf1262  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_965, slice_192, slice_193, neg_101, add_666, mul_966, add_667, permute_1802, clone_103, view_2022, mul_969], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf1285, mm_default, buf1307, buf1314, 1091328, stream=stream0)
            buf1308 = reinterpret_tensor(buf1302, (896, 32), (32, 1), 0); del buf1302  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_965, slice_192, slice_193, neg_101, add_666, mul_966, add_667, permute_1802, clone_103, view_2022, mul_969, view_2023, permute_1803, mm_1060], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1307, (896, 1218), (1, 896), 0), mm_108, out=buf1308)
            del mm_108
            buf1309 = buf1300; del buf1300  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_965, slice_192, slice_193, neg_101, add_666, mul_966, add_667, permute_1802, clone_103, view_2022, mul_969, view_2023, mm_1061], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1307, (1218, 896), (896, 1), 0), permute_1805, out=buf1309)
            del permute_1805
            buf1311 = reinterpret_tensor(buf1276, (32, 896), (896, 1), 0); del buf1276  # reuse
            # Topologically Sorted Source Nodes: [permute_1807, mm_1062], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1309, (32, 1218), (1, 32), 0), view_299, out=buf1311)
            del view_299
            buf1315 = reinterpret_tensor(buf1307, (1218, 896), (896, 1), 0); del buf1307  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_965, slice_192, slice_193, neg_101, add_666, mul_966, add_667, permute_1802, clone_103, view_2022, view_2027, result_129, permute_1811, mm_1064], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1314, (1218, 896), (896, 1), 0), primals_162, out=buf1315)
            del primals_162
            buf1310 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3790], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1308, buf1310, 28672, stream=stream0)
            buf1313 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3796], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1311, buf1313, 28672, stream=stream0)
            buf1312 = reinterpret_tensor(buf1314, (1218, 896), (896, 1), 0); del buf1314  # reuse
            # Topologically Sorted Source Nodes: [mm_1063], Original ATen: [aten.mm]
            extern_kernels.mm(buf1309, permute_1809, out=buf1312)
            del permute_1809
            buf1318 = buf1274; del buf1274  # reuse
            buf1319 = reinterpret_tensor(buf1285, (1218, 896), (896, 1), 0); del buf1285  # reuse
            # Topologically Sorted Source Nodes: [view_2012, view_2014, add_668, view_2019, add_669, view_2021, add_670, view_2026, add_671, view_2028, add_672, mul_970, convert_element_type_3800, hidden_states_60, mul_971, mul_972, sum_73, pow_122, mul_973, mul_974, expand_113, div_36, pow_123, mul_975, mul_976, add_673, convert_element_type_3801, add_674, mul_977, view_2029], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf1318, buf1295, buf1297, buf1303, buf1306, buf1312, buf1315, primals_161, add_78, rsqrt_12, buf1319, 1218, 896, stream=stream0)
            del add_78
            del buf1295
            del buf1297
            del primals_161
            del rsqrt_12
            buf1320 = reinterpret_tensor(buf1311, (896, 32), (32, 1), 0); del buf1311  # reuse
            # Topologically Sorted Source Nodes: [permute_1812, mm_1065], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1319, (896, 1218), (1, 896), 0), mm_106, out=buf1320)
            del mm_106
            buf1321 = buf1309; del buf1309  # reuse
            # Topologically Sorted Source Nodes: [mm_1066], Original ATen: [aten.mm]
            extern_kernels.mm(buf1319, permute_1814, out=buf1321)
            del permute_1814
            buf1323 = reinterpret_tensor(buf1264, (32, 4864), (4864, 1), 0); del buf1264  # reuse
            # Topologically Sorted Source Nodes: [permute_1816, mm_1067], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1321, (32, 1218), (1, 32), 0), view_293, out=buf1323)
            del view_293
            buf1322 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3806], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1320, buf1322, 28672, stream=stream0)
            buf1325 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3812], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1323, buf1325, 155648, stream=stream0)
            buf1324 = reinterpret_tensor(buf1263, (1218, 4864), (4864, 1), 0); del buf1263  # reuse
            # Topologically Sorted Source Nodes: [mm_1068], Original ATen: [aten.mm]
            extern_kernels.mm(buf1321, permute_1818, out=buf1324)
            del permute_1818
            buf1326 = reinterpret_tensor(buf1254, (1218, 4864), (4864, 1), 0); del buf1254  # reuse
            # Topologically Sorted Source Nodes: [view_2033, result_126, permute_1820, mm_1069], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1318, (1218, 896), (896, 1), 0), primals_158, out=buf1326)
            del primals_158
            buf1327 = buf1270; del buf1270  # reuse
            buf1334 = buf1261; del buf1261  # reuse
            buf1336 = reinterpret_tensor(buf1253, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1253  # reuse
            buf1343 = reinterpret_tensor(buf1251, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1251  # reuse
            # Topologically Sorted Source Nodes: [view_2032, view_2034, add_675, silu_5, mul_978, mul_979, mul_980, convert_element_type_3830, neg_102, exp_18, add_677, reciprocal_18, mul_981, mul_982, sub_18, mul_983, add_678, mul_984, convert_element_type_3832, mul_985], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf1324, buf1326, add_75, add_76, buf1327, buf1334, buf1336, buf1343, 5924352, stream=stream0)
            del add_75
            del add_76
            buf1335 = buf1319; del buf1319  # reuse
            # Topologically Sorted Source Nodes: [view_2032, view_2034, add_675, silu_5, mul_978, view_2039, result_123, permute_1829, mm_1074], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1334, (1218, 4864), (4864, 1), 0), primals_155, out=buf1335)
            del primals_155
            buf1344 = buf1315; del buf1315  # reuse
            # Topologically Sorted Source Nodes: [view_2032, view_2034, add_675, silu_5, mul_979, convert_element_type_3830, neg_102, exp_18, add_677, reciprocal_18, mul_981, mul_982, sub_18, mul_983, add_678, mul_984, convert_element_type_3832, view_2045, result_120, permute_1838, mm_1079], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1343, (1218, 4864), (4864, 1), 0), primals_152, out=buf1344)
            del primals_152
            buf1329 = buf1321; del buf1321  # reuse
            # Topologically Sorted Source Nodes: [view_2032, view_2034, add_675, silu_5, mul_978, mul_980, view_2035, mm_1071], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1327, (1218, 4864), (4864, 1), 0), permute_1823, out=buf1329)
            del permute_1823
            buf1328 = reinterpret_tensor(buf1323, (4864, 32), (32, 1), 0); del buf1323  # reuse
            # Topologically Sorted Source Nodes: [view_2032, view_2034, add_675, silu_5, mul_978, mul_980, view_2035, permute_1821, mm_1070], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1327, (4864, 1218), (1, 4864), 0), mm_103, out=buf1328)
            del mm_103
            buf1338 = buf1292; del buf1292  # reuse
            # Topologically Sorted Source Nodes: [view_2032, view_2034, add_675, silu_5, mul_979, convert_element_type_3830, neg_102, exp_18, add_677, reciprocal_18, mul_981, mul_982, sub_18, mul_983, add_678, mul_984, convert_element_type_3832, mul_985, view_2041, mm_1076], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1336, (1218, 4864), (4864, 1), 0), permute_1832, out=buf1338)
            del permute_1832
            buf1337 = buf1255; del buf1255  # reuse
            # Topologically Sorted Source Nodes: [view_2032, view_2034, add_675, silu_5, mul_979, convert_element_type_3830, neg_102, exp_18, add_677, reciprocal_18, mul_981, mul_982, sub_18, mul_983, add_678, mul_984, convert_element_type_3832, mul_985, view_2041, permute_1830, mm_1075], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1336, (4864, 1218), (1, 4864), 0), mm_100, out=buf1337)
            del mm_100
            buf1331 = reinterpret_tensor(buf1320, (32, 896), (896, 1), 0); del buf1320  # reuse
            # Topologically Sorted Source Nodes: [permute_1825, mm_1072], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1329, (32, 1218), (1, 32), 0), view_281, out=buf1331)
            buf1340 = reinterpret_tensor(buf1308, (32, 896), (896, 1), 0); del buf1308  # reuse
            # Topologically Sorted Source Nodes: [permute_1834, mm_1077], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1338, (32, 1218), (1, 32), 0), view_281, out=buf1340)
            del view_281
            buf1333 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3826], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1331, buf1333, 28672, stream=stream0)
            buf1342 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3843], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1340, buf1342, 28672, stream=stream0)
            buf1330 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3820], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1328, buf1330, 155648, stream=stream0)
            buf1339 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3837], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1337, buf1339, 155648, stream=stream0)
            buf1332 = buf1312; del buf1312  # reuse
            # Topologically Sorted Source Nodes: [mm_1073], Original ATen: [aten.mm]
            extern_kernels.mm(buf1329, permute_1827, out=buf1332)
            del permute_1827
            buf1341 = buf1306; del buf1306  # reuse
            # Topologically Sorted Source Nodes: [mm_1078], Original ATen: [aten.mm]
            extern_kernels.mm(buf1338, permute_1836, out=buf1341)
            del permute_1836
            buf1347 = buf1318; del buf1318  # reuse
            buf1348 = buf1303; del buf1303  # reuse
            # Topologically Sorted Source Nodes: [view_2038, view_2040, add_676, view_2044, add_679, view_2046, add_680, mul_986, convert_element_type_3847, hidden_states_56, mul_987, mul_988, sum_74, pow_124, mul_989, mul_990, expand_114, div_37, pow_125, mul_991, mul_992, add_681, convert_element_type_3848, add_682, mul_993, view_2047], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf1347, buf1332, buf1335, buf1341, buf1344, primals_151, add_73, rsqrt_11, buf1348, 1218, 896, stream=stream0)
            del add_73
            del buf1332
            del primals_151
            del rsqrt_11
            buf1349 = reinterpret_tensor(buf1340, (896, 32), (32, 1), 0); del buf1340  # reuse
            # Topologically Sorted Source Nodes: [permute_1839, mm_1080], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1348, (896, 1218), (1, 896), 0), mm_97, out=buf1349)
            del mm_97
            buf1350 = buf1338; del buf1338  # reuse
            # Topologically Sorted Source Nodes: [mm_1081], Original ATen: [aten.mm]
            extern_kernels.mm(buf1348, permute_1841, out=buf1350)
            del permute_1841
            buf1352 = buf1331; del buf1331  # reuse
            # Topologically Sorted Source Nodes: [permute_1843, mm_1082], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1350, (32, 1218), (1, 32), 0), view_275, out=buf1352)
            del view_275
            buf1351 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3853], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1349, buf1351, 28672, stream=stream0)
            buf1354 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3859], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1352, buf1354, 28672, stream=stream0)
            buf1353 = buf1348; del buf1348  # reuse
            # Topologically Sorted Source Nodes: [mm_1083], Original ATen: [aten.mm]
            extern_kernels.mm(buf1350, permute_1845, out=buf1353)
            del permute_1845
            buf1355 = buf1344; del buf1344  # reuse
            # Topologically Sorted Source Nodes: [view_2051, result_117, permute_1847, mm_1084], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1347, (1218, 896), (896, 1), 0), primals_148, out=buf1355)
            del primals_148
            buf1356 = reinterpret_tensor(buf1353, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf1353  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_2050, view_2052, add_683, view_2053, permute_1848, _scaled_dot_product_efficient_attention_backward_18], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf1356, buf1355, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_2050, view_2052, add_683, view_2053, permute_1848, _scaled_dot_product_efficient_attention_backward_18], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf1357 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf1356, add_70, view_270, view_271, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_20, getitem_21, getitem_22, getitem_23, 0.0, [True, True, True, False], scale=0.125)
            del add_70
            del getitem_20
            del getitem_21
            del getitem_22
            del getitem_23
            del view_270
            del view_271
            buf1358 = buf1357[0]
            assert_size_stride(buf1358, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1358, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1359 = buf1357[1]
            assert_size_stride(buf1359, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1359, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1361 = reinterpret_tensor(buf1305, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf1305  # reuse
            # Topologically Sorted Source Nodes: [view_2055, sum_76], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf1359, buf1361, 155904, stream=stream0)
            buf1360 = buf1357[2]
            assert_size_stride(buf1360, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1360, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf1357
            buf1362 = buf1289; del buf1289  # reuse
            buf1363 = reinterpret_tensor(buf1298, (1218, 128), (128, 1), 0); del buf1298  # reuse
            # Topologically Sorted Source Nodes: [view_2054, sum_75, squeeze_36, permute_1849, clone_104, view_2056, mul_998, view_2057], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf1360, buf1362, buf1363, 155904, stream=stream0)
            buf1364 = buf1299; del buf1299  # reuse
            # Topologically Sorted Source Nodes: [permute_1850, mm_1085], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1363, (128, 1218), (1, 128), 0), mm_94, out=buf1364)
            del mm_94
            buf1365 = buf1350; del buf1350  # reuse
            # Topologically Sorted Source Nodes: [mm_1086], Original ATen: [aten.mm]
            extern_kernels.mm(buf1363, permute_1852, out=buf1365)
            del permute_1852
            buf1366 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3867], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1364, buf1366, 4096, stream=stream0)
            buf1367 = buf1352; del buf1352  # reuse
            # Topologically Sorted Source Nodes: [permute_1854, mm_1087], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1365, (32, 1218), (1, 32), 0), view_251, out=buf1367)
            buf1369 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3873], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1367, buf1369, 28672, stream=stream0)
            buf1371 = reinterpret_tensor(buf1363, (1, 1218, 128), (155904, 128, 1), 0); del buf1363  # reuse
            buf1378 = reinterpret_tensor(buf1288, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf1288  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2055, sum_76, squeeze_37, mul_994, slice_194, slice_195, neg_103, add_684, mul_995, add_685, permute_1859, clone_105, view_2063, mul_999], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf1361, mm_default, buf1371, buf1378, 1218, 128, stream=stream0)
            buf1372 = buf1364; del buf1364  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2055, sum_76, squeeze_37, mul_994, slice_194, slice_195, neg_103, add_684, mul_995, add_685, permute_1859, clone_105, view_2063, mul_999, view_2064, permute_1860, mm_1090], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1371, (128, 1218), (1, 128), 0), mm_92, out=buf1372)
            del mm_92
            buf1373 = buf1329; del buf1329  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2055, sum_76, squeeze_37, mul_994, slice_194, slice_195, neg_103, add_684, mul_995, add_685, permute_1859, clone_105, view_2063, mul_999, view_2064, mm_1091], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1371, (1218, 128), (128, 1), 0), permute_1862, out=buf1373)
            del permute_1862
            buf1374 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3881], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1372, buf1374, 4096, stream=stream0)
            buf1375 = buf1367; del buf1367  # reuse
            # Topologically Sorted Source Nodes: [permute_1864, mm_1092], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1373, (32, 1218), (1, 32), 0), view_251, out=buf1375)
            buf1377 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3887], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1375, buf1377, 28672, stream=stream0)
            buf1370 = reinterpret_tensor(buf1360, (1218, 896), (896, 1), 0); del buf1360  # reuse
            # Topologically Sorted Source Nodes: [view_2054, sum_75, squeeze_36, permute_1849, clone_104, view_2056, view_2061, result_114, permute_1858, mm_1089], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1362, (1218, 128), (128, 1), 0), primals_144, out=buf1370)
            del primals_144
            buf1379 = reinterpret_tensor(buf1359, (1218, 896), (896, 1), 0); del buf1359  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2055, sum_76, squeeze_37, mul_994, slice_194, slice_195, neg_103, add_684, mul_995, add_685, permute_1859, clone_105, view_2063, view_2068, result_111, permute_1868, mm_1094], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1378, (1218, 128), (128, 1), 0), primals_140, out=buf1379)
            del primals_140
            buf1368 = reinterpret_tensor(buf1356, (1218, 896), (896, 1), 0); del buf1356  # reuse
            # Topologically Sorted Source Nodes: [mm_1088], Original ATen: [aten.mm]
            extern_kernels.mm(buf1365, permute_1856, out=buf1368)
            del permute_1856
            buf1376 = buf1355; del buf1355  # reuse
            # Topologically Sorted Source Nodes: [mm_1093], Original ATen: [aten.mm]
            extern_kernels.mm(buf1373, permute_1866, out=buf1376)
            del permute_1866
            buf1380 = reinterpret_tensor(buf1341, (1, 1218, 896), (1091328, 896, 1), 0); del buf1341  # reuse
            buf1387 = reinterpret_tensor(buf1335, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf1335  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_996, slice_196, slice_197, neg_104, add_686, mul_997, add_687, permute_1869, clone_106, view_2070, mul_1000], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf1358, mm_default, buf1380, buf1387, 1091328, stream=stream0)
            buf1381 = reinterpret_tensor(buf1375, (896, 32), (32, 1), 0); del buf1375  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_996, slice_196, slice_197, neg_104, add_686, mul_997, add_687, permute_1869, clone_106, view_2070, mul_1000, view_2071, permute_1870, mm_1095], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1380, (896, 1218), (1, 896), 0), mm_90, out=buf1381)
            del mm_90
            buf1382 = buf1373; del buf1373  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_996, slice_196, slice_197, neg_104, add_686, mul_997, add_687, permute_1869, clone_106, view_2070, mul_1000, view_2071, mm_1096], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1380, (1218, 896), (896, 1), 0), permute_1872, out=buf1382)
            del permute_1872
            buf1384 = reinterpret_tensor(buf1349, (32, 896), (896, 1), 0); del buf1349  # reuse
            # Topologically Sorted Source Nodes: [permute_1874, mm_1097], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1382, (32, 1218), (1, 32), 0), view_251, out=buf1384)
            del view_251
            buf1388 = reinterpret_tensor(buf1380, (1218, 896), (896, 1), 0); del buf1380  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_996, slice_196, slice_197, neg_104, add_686, mul_997, add_687, permute_1869, clone_106, view_2070, view_2075, result_108, permute_1878, mm_1099], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1387, (1218, 896), (896, 1), 0), primals_136, out=buf1388)
            del primals_136
            buf1383 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3895], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1381, buf1383, 28672, stream=stream0)
            buf1386 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3901], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1384, buf1386, 28672, stream=stream0)
            buf1385 = reinterpret_tensor(buf1387, (1218, 896), (896, 1), 0); del buf1387  # reuse
            # Topologically Sorted Source Nodes: [mm_1098], Original ATen: [aten.mm]
            extern_kernels.mm(buf1382, permute_1876, out=buf1385)
            del permute_1876
            buf1391 = buf1347; del buf1347  # reuse
            buf1392 = reinterpret_tensor(buf1358, (1218, 896), (896, 1), 0); del buf1358  # reuse
            # Topologically Sorted Source Nodes: [view_2060, view_2062, add_688, view_2067, add_689, view_2069, add_690, view_2074, add_691, view_2076, add_692, mul_1001, convert_element_type_3905, hidden_states_50, mul_1002, mul_1003, sum_77, pow_126, mul_1004, mul_1005, expand_115, div_38, pow_127, mul_1006, mul_1007, add_693, convert_element_type_3906, add_694, mul_1008, view_2077], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf1391, buf1368, buf1370, buf1376, buf1379, buf1385, buf1388, primals_135, add_65, rsqrt_10, buf1392, 1218, 896, stream=stream0)
            del add_65
            del buf1368
            del buf1370
            del primals_135
            del rsqrt_10
            buf1393 = reinterpret_tensor(buf1384, (896, 32), (32, 1), 0); del buf1384  # reuse
            # Topologically Sorted Source Nodes: [permute_1879, mm_1100], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1392, (896, 1218), (1, 896), 0), mm_88, out=buf1393)
            del mm_88
            buf1394 = buf1382; del buf1382  # reuse
            # Topologically Sorted Source Nodes: [mm_1101], Original ATen: [aten.mm]
            extern_kernels.mm(buf1392, permute_1881, out=buf1394)
            del permute_1881
            buf1396 = reinterpret_tensor(buf1337, (32, 4864), (4864, 1), 0); del buf1337  # reuse
            # Topologically Sorted Source Nodes: [permute_1883, mm_1102], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1394, (32, 1218), (1, 32), 0), view_245, out=buf1396)
            del view_245
            buf1395 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3911], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1393, buf1395, 28672, stream=stream0)
            buf1398 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3917], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1396, buf1398, 155648, stream=stream0)
            buf1397 = reinterpret_tensor(buf1336, (1218, 4864), (4864, 1), 0); del buf1336  # reuse
            # Topologically Sorted Source Nodes: [mm_1103], Original ATen: [aten.mm]
            extern_kernels.mm(buf1394, permute_1885, out=buf1397)
            del permute_1885
            buf1399 = reinterpret_tensor(buf1327, (1218, 4864), (4864, 1), 0); del buf1327  # reuse
            # Topologically Sorted Source Nodes: [view_2081, result_105, permute_1887, mm_1104], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1391, (1218, 896), (896, 1), 0), primals_132, out=buf1399)
            del primals_132
            buf1400 = buf1343; del buf1343  # reuse
            buf1407 = buf1334; del buf1334  # reuse
            buf1409 = reinterpret_tensor(buf1326, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1326  # reuse
            buf1416 = reinterpret_tensor(buf1324, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1324  # reuse
            # Topologically Sorted Source Nodes: [view_2080, view_2082, add_695, silu_4, mul_1009, mul_1010, mul_1011, convert_element_type_3935, neg_105, exp_19, add_697, reciprocal_19, mul_1012, mul_1013, sub_19, mul_1014, add_698, mul_1015, convert_element_type_3937, mul_1016], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf1397, buf1399, add_62, add_63, buf1400, buf1407, buf1409, buf1416, 5924352, stream=stream0)
            del add_62
            del add_63
            buf1408 = buf1392; del buf1392  # reuse
            # Topologically Sorted Source Nodes: [view_2080, view_2082, add_695, silu_4, mul_1009, view_2087, result_102, permute_1896, mm_1109], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1407, (1218, 4864), (4864, 1), 0), primals_129, out=buf1408)
            del primals_129
            buf1417 = buf1388; del buf1388  # reuse
            # Topologically Sorted Source Nodes: [view_2080, view_2082, add_695, silu_4, mul_1010, convert_element_type_3935, neg_105, exp_19, add_697, reciprocal_19, mul_1012, mul_1013, sub_19, mul_1014, add_698, mul_1015, convert_element_type_3937, view_2093, result_99, permute_1905, mm_1114], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1416, (1218, 4864), (4864, 1), 0), primals_126, out=buf1417)
            del primals_126
            buf1402 = buf1394; del buf1394  # reuse
            # Topologically Sorted Source Nodes: [view_2080, view_2082, add_695, silu_4, mul_1009, mul_1011, view_2083, mm_1106], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1400, (1218, 4864), (4864, 1), 0), permute_1890, out=buf1402)
            del permute_1890
            buf1401 = reinterpret_tensor(buf1396, (4864, 32), (32, 1), 0); del buf1396  # reuse
            # Topologically Sorted Source Nodes: [view_2080, view_2082, add_695, silu_4, mul_1009, mul_1011, view_2083, permute_1888, mm_1105], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1400, (4864, 1218), (1, 4864), 0), mm_85, out=buf1401)
            del mm_85
            buf1411 = buf1365; del buf1365  # reuse
            # Topologically Sorted Source Nodes: [view_2080, view_2082, add_695, silu_4, mul_1010, convert_element_type_3935, neg_105, exp_19, add_697, reciprocal_19, mul_1012, mul_1013, sub_19, mul_1014, add_698, mul_1015, convert_element_type_3937, mul_1016, view_2089, mm_1111], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1409, (1218, 4864), (4864, 1), 0), permute_1899, out=buf1411)
            del permute_1899
            buf1410 = buf1328; del buf1328  # reuse
            # Topologically Sorted Source Nodes: [view_2080, view_2082, add_695, silu_4, mul_1010, convert_element_type_3935, neg_105, exp_19, add_697, reciprocal_19, mul_1012, mul_1013, sub_19, mul_1014, add_698, mul_1015, convert_element_type_3937, mul_1016, view_2089, permute_1897, mm_1110], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1409, (4864, 1218), (1, 4864), 0), mm_82, out=buf1410)
            del mm_82
            buf1404 = reinterpret_tensor(buf1393, (32, 896), (896, 1), 0); del buf1393  # reuse
            # Topologically Sorted Source Nodes: [permute_1892, mm_1107], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1402, (32, 1218), (1, 32), 0), view_233, out=buf1404)
            buf1413 = reinterpret_tensor(buf1381, (32, 896), (896, 1), 0); del buf1381  # reuse
            # Topologically Sorted Source Nodes: [permute_1901, mm_1112], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1411, (32, 1218), (1, 32), 0), view_233, out=buf1413)
            del view_233
            buf1406 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3931], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1404, buf1406, 28672, stream=stream0)
            buf1415 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3948], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1413, buf1415, 28672, stream=stream0)
            buf1403 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3925], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1401, buf1403, 155648, stream=stream0)
            buf1412 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3942], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1410, buf1412, 155648, stream=stream0)
            buf1405 = buf1385; del buf1385  # reuse
            # Topologically Sorted Source Nodes: [mm_1108], Original ATen: [aten.mm]
            extern_kernels.mm(buf1402, permute_1894, out=buf1405)
            del permute_1894
            buf1414 = buf1379; del buf1379  # reuse
            # Topologically Sorted Source Nodes: [mm_1113], Original ATen: [aten.mm]
            extern_kernels.mm(buf1411, permute_1903, out=buf1414)
            del permute_1903
            buf1420 = buf1391; del buf1391  # reuse
            buf1421 = buf1376; del buf1376  # reuse
            # Topologically Sorted Source Nodes: [view_2086, view_2088, add_696, view_2092, add_699, view_2094, add_700, mul_1017, convert_element_type_3952, hidden_states_46, mul_1018, mul_1019, sum_78, pow_128, mul_1020, mul_1021, expand_116, div_39, pow_129, mul_1022, mul_1023, add_701, convert_element_type_3953, add_702, mul_1024, view_2095], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf1420, buf1405, buf1408, buf1414, buf1417, primals_125, add_60, rsqrt_9, buf1421, 1218, 896, stream=stream0)
            del add_60
            del buf1405
            del primals_125
            del rsqrt_9
            buf1422 = reinterpret_tensor(buf1413, (896, 32), (32, 1), 0); del buf1413  # reuse
            # Topologically Sorted Source Nodes: [permute_1906, mm_1115], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1421, (896, 1218), (1, 896), 0), mm_79, out=buf1422)
            del mm_79
            buf1423 = buf1411; del buf1411  # reuse
            # Topologically Sorted Source Nodes: [mm_1116], Original ATen: [aten.mm]
            extern_kernels.mm(buf1421, permute_1908, out=buf1423)
            del permute_1908
            buf1425 = buf1404; del buf1404  # reuse
            # Topologically Sorted Source Nodes: [permute_1910, mm_1117], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1423, (32, 1218), (1, 32), 0), view_227, out=buf1425)
            del view_227
            buf1424 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3958], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1422, buf1424, 28672, stream=stream0)
            buf1427 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3964], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1425, buf1427, 28672, stream=stream0)
            buf1426 = buf1421; del buf1421  # reuse
            # Topologically Sorted Source Nodes: [mm_1118], Original ATen: [aten.mm]
            extern_kernels.mm(buf1423, permute_1912, out=buf1426)
            del permute_1912
            buf1428 = buf1417; del buf1417  # reuse
            # Topologically Sorted Source Nodes: [view_2099, result_96, permute_1914, mm_1119], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1420, (1218, 896), (896, 1), 0), primals_122, out=buf1428)
            del primals_122
            buf1429 = reinterpret_tensor(buf1426, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf1426  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_2098, view_2100, add_703, view_2101, permute_1915, _scaled_dot_product_efficient_attention_backward_19], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf1429, buf1428, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_2098, view_2100, add_703, view_2101, permute_1915, _scaled_dot_product_efficient_attention_backward_19], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf1430 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf1429, add_57, view_222, view_223, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_16, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False], scale=0.125)
            del add_57
            del getitem_16
            del getitem_17
            del getitem_18
            del getitem_19
            del view_222
            del view_223
            buf1431 = buf1430[0]
            assert_size_stride(buf1431, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1431, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1432 = buf1430[1]
            assert_size_stride(buf1432, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1432, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1434 = reinterpret_tensor(buf1378, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf1378  # reuse
            # Topologically Sorted Source Nodes: [view_2103, sum_80], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf1432, buf1434, 155904, stream=stream0)
            buf1433 = buf1430[2]
            assert_size_stride(buf1433, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1433, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf1430
            buf1435 = buf1362; del buf1362  # reuse
            buf1436 = reinterpret_tensor(buf1371, (1218, 128), (128, 1), 0); del buf1371  # reuse
            # Topologically Sorted Source Nodes: [view_2102, sum_79, squeeze_38, permute_1916, clone_107, view_2104, mul_1029, view_2105], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf1433, buf1435, buf1436, 155904, stream=stream0)
            buf1437 = buf1372; del buf1372  # reuse
            # Topologically Sorted Source Nodes: [permute_1917, mm_1120], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1436, (128, 1218), (1, 128), 0), mm_76, out=buf1437)
            del mm_76
            buf1438 = buf1423; del buf1423  # reuse
            # Topologically Sorted Source Nodes: [mm_1121], Original ATen: [aten.mm]
            extern_kernels.mm(buf1436, permute_1919, out=buf1438)
            del permute_1919
            buf1439 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3972], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1437, buf1439, 4096, stream=stream0)
            buf1440 = buf1425; del buf1425  # reuse
            # Topologically Sorted Source Nodes: [permute_1921, mm_1122], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1438, (32, 1218), (1, 32), 0), view_203, out=buf1440)
            buf1442 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3978], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1440, buf1442, 28672, stream=stream0)
            buf1444 = reinterpret_tensor(buf1436, (1, 1218, 128), (155904, 128, 1), 0); del buf1436  # reuse
            buf1451 = reinterpret_tensor(buf1361, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf1361  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2103, sum_80, squeeze_39, mul_1025, slice_198, slice_199, neg_106, add_704, mul_1026, add_705, permute_1926, clone_108, view_2111, mul_1030], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf1434, mm_default, buf1444, buf1451, 1218, 128, stream=stream0)
            buf1445 = buf1437; del buf1437  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2103, sum_80, squeeze_39, mul_1025, slice_198, slice_199, neg_106, add_704, mul_1026, add_705, permute_1926, clone_108, view_2111, mul_1030, view_2112, permute_1927, mm_1125], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1444, (128, 1218), (1, 128), 0), mm_74, out=buf1445)
            del mm_74
            buf1446 = buf1402; del buf1402  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2103, sum_80, squeeze_39, mul_1025, slice_198, slice_199, neg_106, add_704, mul_1026, add_705, permute_1926, clone_108, view_2111, mul_1030, view_2112, mm_1126], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1444, (1218, 128), (128, 1), 0), permute_1929, out=buf1446)
            del permute_1929
            buf1447 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3986], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1445, buf1447, 4096, stream=stream0)
            buf1448 = buf1440; del buf1440  # reuse
            # Topologically Sorted Source Nodes: [permute_1931, mm_1127], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1446, (32, 1218), (1, 32), 0), view_203, out=buf1448)
            buf1450 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_3992], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1448, buf1450, 28672, stream=stream0)
            buf1443 = reinterpret_tensor(buf1433, (1218, 896), (896, 1), 0); del buf1433  # reuse
            # Topologically Sorted Source Nodes: [view_2102, sum_79, squeeze_38, permute_1916, clone_107, view_2104, view_2109, result_93, permute_1925, mm_1124], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1435, (1218, 128), (128, 1), 0), primals_118, out=buf1443)
            del primals_118
            buf1452 = reinterpret_tensor(buf1432, (1218, 896), (896, 1), 0); del buf1432  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2103, sum_80, squeeze_39, mul_1025, slice_198, slice_199, neg_106, add_704, mul_1026, add_705, permute_1926, clone_108, view_2111, view_2116, result_90, permute_1935, mm_1129], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1451, (1218, 128), (128, 1), 0), primals_114, out=buf1452)
            del primals_114
            buf1441 = reinterpret_tensor(buf1429, (1218, 896), (896, 1), 0); del buf1429  # reuse
            # Topologically Sorted Source Nodes: [mm_1123], Original ATen: [aten.mm]
            extern_kernels.mm(buf1438, permute_1923, out=buf1441)
            del permute_1923
            buf1449 = buf1428; del buf1428  # reuse
            # Topologically Sorted Source Nodes: [mm_1128], Original ATen: [aten.mm]
            extern_kernels.mm(buf1446, permute_1933, out=buf1449)
            del permute_1933
            buf1453 = reinterpret_tensor(buf1414, (1, 1218, 896), (1091328, 896, 1), 0); del buf1414  # reuse
            buf1460 = reinterpret_tensor(buf1408, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf1408  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1027, slice_200, slice_201, neg_107, add_706, mul_1028, add_707, permute_1936, clone_109, view_2118, mul_1031], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf1431, mm_default, buf1453, buf1460, 1091328, stream=stream0)
            buf1454 = reinterpret_tensor(buf1448, (896, 32), (32, 1), 0); del buf1448  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1027, slice_200, slice_201, neg_107, add_706, mul_1028, add_707, permute_1936, clone_109, view_2118, mul_1031, view_2119, permute_1937, mm_1130], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1453, (896, 1218), (1, 896), 0), mm_72, out=buf1454)
            del mm_72
            buf1455 = buf1446; del buf1446  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1027, slice_200, slice_201, neg_107, add_706, mul_1028, add_707, permute_1936, clone_109, view_2118, mul_1031, view_2119, mm_1131], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1453, (1218, 896), (896, 1), 0), permute_1939, out=buf1455)
            del permute_1939
            buf1457 = reinterpret_tensor(buf1422, (32, 896), (896, 1), 0); del buf1422  # reuse
            # Topologically Sorted Source Nodes: [permute_1941, mm_1132], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1455, (32, 1218), (1, 32), 0), view_203, out=buf1457)
            del view_203
            buf1461 = reinterpret_tensor(buf1453, (1218, 896), (896, 1), 0); del buf1453  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1027, slice_200, slice_201, neg_107, add_706, mul_1028, add_707, permute_1936, clone_109, view_2118, view_2123, result_87, permute_1945, mm_1134], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1460, (1218, 896), (896, 1), 0), primals_110, out=buf1461)
            del primals_110
            buf1456 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4000], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1454, buf1456, 28672, stream=stream0)
            buf1459 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4006], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1457, buf1459, 28672, stream=stream0)
            buf1458 = reinterpret_tensor(buf1460, (1218, 896), (896, 1), 0); del buf1460  # reuse
            # Topologically Sorted Source Nodes: [mm_1133], Original ATen: [aten.mm]
            extern_kernels.mm(buf1455, permute_1943, out=buf1458)
            del permute_1943
            buf1464 = buf1420; del buf1420  # reuse
            buf1465 = reinterpret_tensor(buf1431, (1218, 896), (896, 1), 0); del buf1431  # reuse
            # Topologically Sorted Source Nodes: [view_2108, view_2110, add_708, view_2115, add_709, view_2117, add_710, view_2122, add_711, view_2124, add_712, mul_1032, convert_element_type_4010, hidden_states_40, mul_1033, mul_1034, sum_81, pow_130, mul_1035, mul_1036, expand_117, div_40, pow_131, mul_1037, mul_1038, add_713, convert_element_type_4011, add_714, mul_1039, view_2125], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf1464, buf1441, buf1443, buf1449, buf1452, buf1458, buf1461, primals_109, add_52, rsqrt_8, buf1465, 1218, 896, stream=stream0)
            del add_52
            del buf1441
            del buf1443
            del primals_109
            del rsqrt_8
            buf1466 = reinterpret_tensor(buf1457, (896, 32), (32, 1), 0); del buf1457  # reuse
            # Topologically Sorted Source Nodes: [permute_1946, mm_1135], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1465, (896, 1218), (1, 896), 0), mm_70, out=buf1466)
            del mm_70
            buf1467 = buf1455; del buf1455  # reuse
            # Topologically Sorted Source Nodes: [mm_1136], Original ATen: [aten.mm]
            extern_kernels.mm(buf1465, permute_1948, out=buf1467)
            del permute_1948
            buf1469 = reinterpret_tensor(buf1410, (32, 4864), (4864, 1), 0); del buf1410  # reuse
            # Topologically Sorted Source Nodes: [permute_1950, mm_1137], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1467, (32, 1218), (1, 32), 0), view_197, out=buf1469)
            del view_197
            buf1468 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4016], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1466, buf1468, 28672, stream=stream0)
            buf1471 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4022], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1469, buf1471, 155648, stream=stream0)
            buf1470 = reinterpret_tensor(buf1409, (1218, 4864), (4864, 1), 0); del buf1409  # reuse
            # Topologically Sorted Source Nodes: [mm_1138], Original ATen: [aten.mm]
            extern_kernels.mm(buf1467, permute_1952, out=buf1470)
            del permute_1952
            buf1472 = reinterpret_tensor(buf1400, (1218, 4864), (4864, 1), 0); del buf1400  # reuse
            # Topologically Sorted Source Nodes: [view_2129, result_84, permute_1954, mm_1139], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1464, (1218, 896), (896, 1), 0), primals_106, out=buf1472)
            del primals_106
            buf1473 = buf1416; del buf1416  # reuse
            buf1480 = buf1407; del buf1407  # reuse
            buf1482 = reinterpret_tensor(buf1399, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1399  # reuse
            buf1489 = reinterpret_tensor(buf1397, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1397  # reuse
            # Topologically Sorted Source Nodes: [view_2128, view_2130, add_715, silu_3, mul_1040, mul_1041, mul_1042, convert_element_type_4040, neg_108, exp_20, add_717, reciprocal_20, mul_1043, mul_1044, sub_20, mul_1045, add_718, mul_1046, convert_element_type_4042, mul_1047], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf1470, buf1472, add_49, add_50, buf1473, buf1480, buf1482, buf1489, 5924352, stream=stream0)
            del add_49
            del add_50
            buf1481 = buf1465; del buf1465  # reuse
            # Topologically Sorted Source Nodes: [view_2128, view_2130, add_715, silu_3, mul_1040, view_2135, result_81, permute_1963, mm_1144], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1480, (1218, 4864), (4864, 1), 0), primals_103, out=buf1481)
            del primals_103
            buf1490 = buf1461; del buf1461  # reuse
            # Topologically Sorted Source Nodes: [view_2128, view_2130, add_715, silu_3, mul_1041, convert_element_type_4040, neg_108, exp_20, add_717, reciprocal_20, mul_1043, mul_1044, sub_20, mul_1045, add_718, mul_1046, convert_element_type_4042, view_2141, result_78, permute_1972, mm_1149], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1489, (1218, 4864), (4864, 1), 0), primals_100, out=buf1490)
            del primals_100
            buf1475 = buf1467; del buf1467  # reuse
            # Topologically Sorted Source Nodes: [view_2128, view_2130, add_715, silu_3, mul_1040, mul_1042, view_2131, mm_1141], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1473, (1218, 4864), (4864, 1), 0), permute_1957, out=buf1475)
            del permute_1957
            buf1474 = reinterpret_tensor(buf1469, (4864, 32), (32, 1), 0); del buf1469  # reuse
            # Topologically Sorted Source Nodes: [view_2128, view_2130, add_715, silu_3, mul_1040, mul_1042, view_2131, permute_1955, mm_1140], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1473, (4864, 1218), (1, 4864), 0), mm_67, out=buf1474)
            del mm_67
            buf1484 = buf1438; del buf1438  # reuse
            # Topologically Sorted Source Nodes: [view_2128, view_2130, add_715, silu_3, mul_1041, convert_element_type_4040, neg_108, exp_20, add_717, reciprocal_20, mul_1043, mul_1044, sub_20, mul_1045, add_718, mul_1046, convert_element_type_4042, mul_1047, view_2137, mm_1146], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1482, (1218, 4864), (4864, 1), 0), permute_1966, out=buf1484)
            del permute_1966
            buf1483 = buf1401; del buf1401  # reuse
            # Topologically Sorted Source Nodes: [view_2128, view_2130, add_715, silu_3, mul_1041, convert_element_type_4040, neg_108, exp_20, add_717, reciprocal_20, mul_1043, mul_1044, sub_20, mul_1045, add_718, mul_1046, convert_element_type_4042, mul_1047, view_2137, permute_1964, mm_1145], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1482, (4864, 1218), (1, 4864), 0), mm_64, out=buf1483)
            del mm_64
            buf1477 = reinterpret_tensor(buf1466, (32, 896), (896, 1), 0); del buf1466  # reuse
            # Topologically Sorted Source Nodes: [permute_1959, mm_1142], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1475, (32, 1218), (1, 32), 0), view_185, out=buf1477)
            buf1486 = reinterpret_tensor(buf1454, (32, 896), (896, 1), 0); del buf1454  # reuse
            # Topologically Sorted Source Nodes: [permute_1968, mm_1147], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1484, (32, 1218), (1, 32), 0), view_185, out=buf1486)
            del view_185
            buf1479 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4036], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1477, buf1479, 28672, stream=stream0)
            buf1488 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4053], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1486, buf1488, 28672, stream=stream0)
            buf1476 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4030], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1474, buf1476, 155648, stream=stream0)
            buf1485 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4047], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1483, buf1485, 155648, stream=stream0)
            buf1478 = buf1458; del buf1458  # reuse
            # Topologically Sorted Source Nodes: [mm_1143], Original ATen: [aten.mm]
            extern_kernels.mm(buf1475, permute_1961, out=buf1478)
            del permute_1961
            buf1487 = buf1452; del buf1452  # reuse
            # Topologically Sorted Source Nodes: [mm_1148], Original ATen: [aten.mm]
            extern_kernels.mm(buf1484, permute_1970, out=buf1487)
            del permute_1970
            buf1493 = buf1464; del buf1464  # reuse
            buf1494 = buf1449; del buf1449  # reuse
            # Topologically Sorted Source Nodes: [view_2134, view_2136, add_716, view_2140, add_719, view_2142, add_720, mul_1048, convert_element_type_4057, hidden_states_36, mul_1049, mul_1050, sum_82, pow_132, mul_1051, mul_1052, expand_118, div_41, pow_133, mul_1053, mul_1054, add_721, convert_element_type_4058, add_722, mul_1055, view_2143], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf1493, buf1478, buf1481, buf1487, buf1490, primals_99, add_47, rsqrt_7, buf1494, 1218, 896, stream=stream0)
            del add_47
            del buf1478
            del primals_99
            del rsqrt_7
            buf1495 = reinterpret_tensor(buf1486, (896, 32), (32, 1), 0); del buf1486  # reuse
            # Topologically Sorted Source Nodes: [permute_1973, mm_1150], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1494, (896, 1218), (1, 896), 0), mm_61, out=buf1495)
            del mm_61
            buf1496 = buf1484; del buf1484  # reuse
            # Topologically Sorted Source Nodes: [mm_1151], Original ATen: [aten.mm]
            extern_kernels.mm(buf1494, permute_1975, out=buf1496)
            del permute_1975
            buf1498 = buf1477; del buf1477  # reuse
            # Topologically Sorted Source Nodes: [permute_1977, mm_1152], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1496, (32, 1218), (1, 32), 0), view_179, out=buf1498)
            del view_179
            buf1497 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4063], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1495, buf1497, 28672, stream=stream0)
            buf1500 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4069], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1498, buf1500, 28672, stream=stream0)
            buf1499 = buf1494; del buf1494  # reuse
            # Topologically Sorted Source Nodes: [mm_1153], Original ATen: [aten.mm]
            extern_kernels.mm(buf1496, permute_1979, out=buf1499)
            del permute_1979
            buf1501 = buf1490; del buf1490  # reuse
            # Topologically Sorted Source Nodes: [view_2147, result_75, permute_1981, mm_1154], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1493, (1218, 896), (896, 1), 0), primals_96, out=buf1501)
            del primals_96
            buf1502 = reinterpret_tensor(buf1499, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf1499  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_2146, view_2148, add_723, view_2149, permute_1982, _scaled_dot_product_efficient_attention_backward_20], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf1502, buf1501, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_2146, view_2148, add_723, view_2149, permute_1982, _scaled_dot_product_efficient_attention_backward_20], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf1503 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf1502, add_44, view_174, view_175, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_12, getitem_13, getitem_14, getitem_15, 0.0, [True, True, True, False], scale=0.125)
            del add_44
            del getitem_12
            del getitem_13
            del getitem_14
            del getitem_15
            del view_174
            del view_175
            buf1504 = buf1503[0]
            assert_size_stride(buf1504, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1504, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1505 = buf1503[1]
            assert_size_stride(buf1505, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1505, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1507 = reinterpret_tensor(buf1451, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf1451  # reuse
            # Topologically Sorted Source Nodes: [view_2151, sum_84], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf1505, buf1507, 155904, stream=stream0)
            buf1506 = buf1503[2]
            assert_size_stride(buf1506, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1506, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf1503
            buf1508 = buf1435; del buf1435  # reuse
            buf1509 = reinterpret_tensor(buf1444, (1218, 128), (128, 1), 0); del buf1444  # reuse
            # Topologically Sorted Source Nodes: [view_2150, sum_83, squeeze_40, permute_1983, clone_110, view_2152, mul_1060, view_2153], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf1506, buf1508, buf1509, 155904, stream=stream0)
            buf1510 = buf1445; del buf1445  # reuse
            # Topologically Sorted Source Nodes: [permute_1984, mm_1155], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1509, (128, 1218), (1, 128), 0), mm_58, out=buf1510)
            del mm_58
            buf1511 = buf1496; del buf1496  # reuse
            # Topologically Sorted Source Nodes: [mm_1156], Original ATen: [aten.mm]
            extern_kernels.mm(buf1509, permute_1986, out=buf1511)
            del permute_1986
            buf1512 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4077], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1510, buf1512, 4096, stream=stream0)
            buf1513 = buf1498; del buf1498  # reuse
            # Topologically Sorted Source Nodes: [permute_1988, mm_1157], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1511, (32, 1218), (1, 32), 0), view_155, out=buf1513)
            buf1515 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4083], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1513, buf1515, 28672, stream=stream0)
            buf1517 = reinterpret_tensor(buf1509, (1, 1218, 128), (155904, 128, 1), 0); del buf1509  # reuse
            buf1524 = reinterpret_tensor(buf1434, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf1434  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2151, sum_84, squeeze_41, mul_1056, slice_202, slice_203, neg_109, add_724, mul_1057, add_725, permute_1993, clone_111, view_2159, mul_1061], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf1507, mm_default, buf1517, buf1524, 1218, 128, stream=stream0)
            buf1518 = buf1510; del buf1510  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2151, sum_84, squeeze_41, mul_1056, slice_202, slice_203, neg_109, add_724, mul_1057, add_725, permute_1993, clone_111, view_2159, mul_1061, view_2160, permute_1994, mm_1160], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1517, (128, 1218), (1, 128), 0), mm_56, out=buf1518)
            del mm_56
            buf1519 = buf1475; del buf1475  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2151, sum_84, squeeze_41, mul_1056, slice_202, slice_203, neg_109, add_724, mul_1057, add_725, permute_1993, clone_111, view_2159, mul_1061, view_2160, mm_1161], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1517, (1218, 128), (128, 1), 0), permute_1996, out=buf1519)
            del permute_1996
            buf1520 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4091], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1518, buf1520, 4096, stream=stream0)
            buf1521 = buf1513; del buf1513  # reuse
            # Topologically Sorted Source Nodes: [permute_1998, mm_1162], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1519, (32, 1218), (1, 32), 0), view_155, out=buf1521)
            buf1523 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4097], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1521, buf1523, 28672, stream=stream0)
            buf1516 = reinterpret_tensor(buf1506, (1218, 896), (896, 1), 0); del buf1506  # reuse
            # Topologically Sorted Source Nodes: [view_2150, sum_83, squeeze_40, permute_1983, clone_110, view_2152, view_2157, result_72, permute_1992, mm_1159], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1508, (1218, 128), (128, 1), 0), primals_92, out=buf1516)
            del primals_92
            buf1525 = reinterpret_tensor(buf1505, (1218, 896), (896, 1), 0); del buf1505  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2151, sum_84, squeeze_41, mul_1056, slice_202, slice_203, neg_109, add_724, mul_1057, add_725, permute_1993, clone_111, view_2159, view_2164, result_69, permute_2002, mm_1164], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1524, (1218, 128), (128, 1), 0), primals_88, out=buf1525)
            del primals_88
            buf1514 = reinterpret_tensor(buf1502, (1218, 896), (896, 1), 0); del buf1502  # reuse
            # Topologically Sorted Source Nodes: [mm_1158], Original ATen: [aten.mm]
            extern_kernels.mm(buf1511, permute_1990, out=buf1514)
            del permute_1990
            buf1522 = buf1501; del buf1501  # reuse
            # Topologically Sorted Source Nodes: [mm_1163], Original ATen: [aten.mm]
            extern_kernels.mm(buf1519, permute_2000, out=buf1522)
            del permute_2000
            buf1526 = reinterpret_tensor(buf1487, (1, 1218, 896), (1091328, 896, 1), 0); del buf1487  # reuse
            buf1533 = reinterpret_tensor(buf1481, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf1481  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1058, slice_204, slice_205, neg_110, add_726, mul_1059, add_727, permute_2003, clone_112, view_2166, mul_1062], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf1504, mm_default, buf1526, buf1533, 1091328, stream=stream0)
            buf1527 = reinterpret_tensor(buf1521, (896, 32), (32, 1), 0); del buf1521  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1058, slice_204, slice_205, neg_110, add_726, mul_1059, add_727, permute_2003, clone_112, view_2166, mul_1062, view_2167, permute_2004, mm_1165], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1526, (896, 1218), (1, 896), 0), mm_54, out=buf1527)
            del mm_54
            buf1528 = buf1519; del buf1519  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1058, slice_204, slice_205, neg_110, add_726, mul_1059, add_727, permute_2003, clone_112, view_2166, mul_1062, view_2167, mm_1166], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1526, (1218, 896), (896, 1), 0), permute_2006, out=buf1528)
            del permute_2006
            buf1530 = reinterpret_tensor(buf1495, (32, 896), (896, 1), 0); del buf1495  # reuse
            # Topologically Sorted Source Nodes: [permute_2008, mm_1167], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1528, (32, 1218), (1, 32), 0), view_155, out=buf1530)
            del view_155
            buf1534 = reinterpret_tensor(buf1526, (1218, 896), (896, 1), 0); del buf1526  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1058, slice_204, slice_205, neg_110, add_726, mul_1059, add_727, permute_2003, clone_112, view_2166, view_2171, result_66, permute_2012, mm_1169], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1533, (1218, 896), (896, 1), 0), primals_84, out=buf1534)
            del primals_84
            buf1529 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4105], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1527, buf1529, 28672, stream=stream0)
            buf1532 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4111], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1530, buf1532, 28672, stream=stream0)
            buf1531 = reinterpret_tensor(buf1533, (1218, 896), (896, 1), 0); del buf1533  # reuse
            # Topologically Sorted Source Nodes: [mm_1168], Original ATen: [aten.mm]
            extern_kernels.mm(buf1528, permute_2010, out=buf1531)
            del permute_2010
            buf1537 = buf1493; del buf1493  # reuse
            buf1538 = reinterpret_tensor(buf1504, (1218, 896), (896, 1), 0); del buf1504  # reuse
            # Topologically Sorted Source Nodes: [view_2156, view_2158, add_728, view_2163, add_729, view_2165, add_730, view_2170, add_731, view_2172, add_732, mul_1063, convert_element_type_4115, hidden_states_30, mul_1064, mul_1065, sum_85, pow_134, mul_1066, mul_1067, expand_119, div_42, pow_135, mul_1068, mul_1069, add_733, convert_element_type_4116, add_734, mul_1070, view_2173], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf1537, buf1514, buf1516, buf1522, buf1525, buf1531, buf1534, primals_83, add_39, rsqrt_6, buf1538, 1218, 896, stream=stream0)
            del add_39
            del buf1514
            del buf1516
            del primals_83
            del rsqrt_6
            buf1539 = reinterpret_tensor(buf1530, (896, 32), (32, 1), 0); del buf1530  # reuse
            # Topologically Sorted Source Nodes: [permute_2013, mm_1170], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1538, (896, 1218), (1, 896), 0), mm_52, out=buf1539)
            del mm_52
            buf1540 = buf1528; del buf1528  # reuse
            # Topologically Sorted Source Nodes: [mm_1171], Original ATen: [aten.mm]
            extern_kernels.mm(buf1538, permute_2015, out=buf1540)
            del permute_2015
            buf1542 = reinterpret_tensor(buf1483, (32, 4864), (4864, 1), 0); del buf1483  # reuse
            # Topologically Sorted Source Nodes: [permute_2017, mm_1172], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1540, (32, 1218), (1, 32), 0), view_149, out=buf1542)
            del view_149
            buf1541 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4121], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1539, buf1541, 28672, stream=stream0)
            buf1544 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4127], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1542, buf1544, 155648, stream=stream0)
            buf1543 = reinterpret_tensor(buf1482, (1218, 4864), (4864, 1), 0); del buf1482  # reuse
            # Topologically Sorted Source Nodes: [mm_1173], Original ATen: [aten.mm]
            extern_kernels.mm(buf1540, permute_2019, out=buf1543)
            del permute_2019
            buf1545 = reinterpret_tensor(buf1473, (1218, 4864), (4864, 1), 0); del buf1473  # reuse
            # Topologically Sorted Source Nodes: [view_2177, result_63, permute_2021, mm_1174], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1537, (1218, 896), (896, 1), 0), primals_80, out=buf1545)
            del primals_80
            buf1546 = buf1489; del buf1489  # reuse
            buf1553 = buf1480; del buf1480  # reuse
            buf1555 = reinterpret_tensor(buf1472, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1472  # reuse
            buf1562 = reinterpret_tensor(buf1470, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1470  # reuse
            # Topologically Sorted Source Nodes: [view_2176, view_2178, add_735, silu_2, mul_1071, mul_1072, mul_1073, convert_element_type_4145, neg_111, exp_21, add_737, reciprocal_21, mul_1074, mul_1075, sub_21, mul_1076, add_738, mul_1077, convert_element_type_4147, mul_1078], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf1543, buf1545, add_36, add_37, buf1546, buf1553, buf1555, buf1562, 5924352, stream=stream0)
            del add_36
            del add_37
            buf1554 = buf1538; del buf1538  # reuse
            # Topologically Sorted Source Nodes: [view_2176, view_2178, add_735, silu_2, mul_1071, view_2183, result_60, permute_2030, mm_1179], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1553, (1218, 4864), (4864, 1), 0), primals_77, out=buf1554)
            del primals_77
            buf1563 = buf1534; del buf1534  # reuse
            # Topologically Sorted Source Nodes: [view_2176, view_2178, add_735, silu_2, mul_1072, convert_element_type_4145, neg_111, exp_21, add_737, reciprocal_21, mul_1074, mul_1075, sub_21, mul_1076, add_738, mul_1077, convert_element_type_4147, view_2189, result_57, permute_2039, mm_1184], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1562, (1218, 4864), (4864, 1), 0), primals_74, out=buf1563)
            del primals_74
            buf1548 = buf1540; del buf1540  # reuse
            # Topologically Sorted Source Nodes: [view_2176, view_2178, add_735, silu_2, mul_1071, mul_1073, view_2179, mm_1176], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1546, (1218, 4864), (4864, 1), 0), permute_2024, out=buf1548)
            del permute_2024
            buf1547 = reinterpret_tensor(buf1542, (4864, 32), (32, 1), 0); del buf1542  # reuse
            # Topologically Sorted Source Nodes: [view_2176, view_2178, add_735, silu_2, mul_1071, mul_1073, view_2179, permute_2022, mm_1175], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1546, (4864, 1218), (1, 4864), 0), mm_49, out=buf1547)
            del mm_49
            buf1557 = buf1511; del buf1511  # reuse
            # Topologically Sorted Source Nodes: [view_2176, view_2178, add_735, silu_2, mul_1072, convert_element_type_4145, neg_111, exp_21, add_737, reciprocal_21, mul_1074, mul_1075, sub_21, mul_1076, add_738, mul_1077, convert_element_type_4147, mul_1078, view_2185, mm_1181], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1555, (1218, 4864), (4864, 1), 0), permute_2033, out=buf1557)
            del permute_2033
            buf1556 = buf1474; del buf1474  # reuse
            # Topologically Sorted Source Nodes: [view_2176, view_2178, add_735, silu_2, mul_1072, convert_element_type_4145, neg_111, exp_21, add_737, reciprocal_21, mul_1074, mul_1075, sub_21, mul_1076, add_738, mul_1077, convert_element_type_4147, mul_1078, view_2185, permute_2031, mm_1180], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1555, (4864, 1218), (1, 4864), 0), mm_46, out=buf1556)
            del mm_46
            buf1550 = reinterpret_tensor(buf1539, (32, 896), (896, 1), 0); del buf1539  # reuse
            # Topologically Sorted Source Nodes: [permute_2026, mm_1177], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1548, (32, 1218), (1, 32), 0), view_137, out=buf1550)
            buf1559 = reinterpret_tensor(buf1527, (32, 896), (896, 1), 0); del buf1527  # reuse
            # Topologically Sorted Source Nodes: [permute_2035, mm_1182], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1557, (32, 1218), (1, 32), 0), view_137, out=buf1559)
            del view_137
            buf1552 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4141], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1550, buf1552, 28672, stream=stream0)
            buf1561 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4158], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1559, buf1561, 28672, stream=stream0)
            buf1549 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4135], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1547, buf1549, 155648, stream=stream0)
            buf1558 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4152], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1556, buf1558, 155648, stream=stream0)
            buf1551 = buf1531; del buf1531  # reuse
            # Topologically Sorted Source Nodes: [mm_1178], Original ATen: [aten.mm]
            extern_kernels.mm(buf1548, permute_2028, out=buf1551)
            del permute_2028
            buf1560 = buf1525; del buf1525  # reuse
            # Topologically Sorted Source Nodes: [mm_1183], Original ATen: [aten.mm]
            extern_kernels.mm(buf1557, permute_2037, out=buf1560)
            del permute_2037
            buf1566 = buf1537; del buf1537  # reuse
            buf1567 = buf1522; del buf1522  # reuse
            # Topologically Sorted Source Nodes: [view_2182, view_2184, add_736, view_2188, add_739, view_2190, add_740, mul_1079, convert_element_type_4162, hidden_states_26, mul_1080, mul_1081, sum_86, pow_136, mul_1082, mul_1083, expand_120, div_43, pow_137, mul_1084, mul_1085, add_741, convert_element_type_4163, add_742, mul_1086, view_2191], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf1566, buf1551, buf1554, buf1560, buf1563, primals_73, add_34, rsqrt_5, buf1567, 1218, 896, stream=stream0)
            del add_34
            del buf1551
            del primals_73
            del rsqrt_5
            buf1568 = reinterpret_tensor(buf1559, (896, 32), (32, 1), 0); del buf1559  # reuse
            # Topologically Sorted Source Nodes: [permute_2040, mm_1185], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1567, (896, 1218), (1, 896), 0), mm_43, out=buf1568)
            del mm_43
            buf1569 = buf1557; del buf1557  # reuse
            # Topologically Sorted Source Nodes: [mm_1186], Original ATen: [aten.mm]
            extern_kernels.mm(buf1567, permute_2042, out=buf1569)
            del permute_2042
            buf1571 = buf1550; del buf1550  # reuse
            # Topologically Sorted Source Nodes: [permute_2044, mm_1187], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1569, (32, 1218), (1, 32), 0), view_131, out=buf1571)
            del view_131
            buf1570 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4168], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1568, buf1570, 28672, stream=stream0)
            buf1573 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4174], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1571, buf1573, 28672, stream=stream0)
            buf1572 = buf1567; del buf1567  # reuse
            # Topologically Sorted Source Nodes: [mm_1188], Original ATen: [aten.mm]
            extern_kernels.mm(buf1569, permute_2046, out=buf1572)
            del permute_2046
            buf1574 = buf1563; del buf1563  # reuse
            # Topologically Sorted Source Nodes: [view_2195, result_54, permute_2048, mm_1189], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1566, (1218, 896), (896, 1), 0), primals_70, out=buf1574)
            del primals_70
            buf1575 = reinterpret_tensor(buf1572, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf1572  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_2194, view_2196, add_743, view_2197, permute_2049, _scaled_dot_product_efficient_attention_backward_21], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf1575, buf1574, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_2194, view_2196, add_743, view_2197, permute_2049, _scaled_dot_product_efficient_attention_backward_21], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf1576 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf1575, add_31, view_126, view_127, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_8, getitem_9, getitem_10, getitem_11, 0.0, [True, True, True, False], scale=0.125)
            del add_31
            del getitem_10
            del getitem_11
            del getitem_8
            del getitem_9
            del view_126
            del view_127
            buf1577 = buf1576[0]
            assert_size_stride(buf1577, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1577, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1578 = buf1576[1]
            assert_size_stride(buf1578, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1578, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1580 = reinterpret_tensor(buf1524, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf1524  # reuse
            # Topologically Sorted Source Nodes: [view_2199, sum_88], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf1578, buf1580, 155904, stream=stream0)
            buf1579 = buf1576[2]
            assert_size_stride(buf1579, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1579, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf1576
            buf1581 = buf1508; del buf1508  # reuse
            buf1582 = reinterpret_tensor(buf1517, (1218, 128), (128, 1), 0); del buf1517  # reuse
            # Topologically Sorted Source Nodes: [view_2198, sum_87, squeeze_42, permute_2050, clone_113, view_2200, mul_1091, view_2201], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf1579, buf1581, buf1582, 155904, stream=stream0)
            buf1583 = buf1518; del buf1518  # reuse
            # Topologically Sorted Source Nodes: [permute_2051, mm_1190], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1582, (128, 1218), (1, 128), 0), mm_40, out=buf1583)
            del mm_40
            buf1584 = buf1569; del buf1569  # reuse
            # Topologically Sorted Source Nodes: [mm_1191], Original ATen: [aten.mm]
            extern_kernels.mm(buf1582, permute_2053, out=buf1584)
            del permute_2053
            buf1585 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4182], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1583, buf1585, 4096, stream=stream0)
            buf1586 = buf1571; del buf1571  # reuse
            # Topologically Sorted Source Nodes: [permute_2055, mm_1192], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1584, (32, 1218), (1, 32), 0), view_107, out=buf1586)
            buf1588 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4188], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1586, buf1588, 28672, stream=stream0)
            buf1590 = reinterpret_tensor(buf1582, (1, 1218, 128), (155904, 128, 1), 0); del buf1582  # reuse
            buf1597 = reinterpret_tensor(buf1507, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf1507  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2199, sum_88, squeeze_43, mul_1087, slice_206, slice_207, neg_112, add_744, mul_1088, add_745, permute_2060, clone_114, view_2207, mul_1092], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf1580, mm_default, buf1590, buf1597, 1218, 128, stream=stream0)
            buf1591 = buf1583; del buf1583  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2199, sum_88, squeeze_43, mul_1087, slice_206, slice_207, neg_112, add_744, mul_1088, add_745, permute_2060, clone_114, view_2207, mul_1092, view_2208, permute_2061, mm_1195], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1590, (128, 1218), (1, 128), 0), mm_38, out=buf1591)
            del mm_38
            buf1592 = buf1548; del buf1548  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2199, sum_88, squeeze_43, mul_1087, slice_206, slice_207, neg_112, add_744, mul_1088, add_745, permute_2060, clone_114, view_2207, mul_1092, view_2208, mm_1196], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1590, (1218, 128), (128, 1), 0), permute_2063, out=buf1592)
            del permute_2063
            buf1593 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4196], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1591, buf1593, 4096, stream=stream0)
            buf1594 = buf1586; del buf1586  # reuse
            # Topologically Sorted Source Nodes: [permute_2065, mm_1197], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1592, (32, 1218), (1, 32), 0), view_107, out=buf1594)
            buf1596 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4202], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1594, buf1596, 28672, stream=stream0)
            buf1589 = reinterpret_tensor(buf1579, (1218, 896), (896, 1), 0); del buf1579  # reuse
            # Topologically Sorted Source Nodes: [view_2198, sum_87, squeeze_42, permute_2050, clone_113, view_2200, view_2205, result_51, permute_2059, mm_1194], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1581, (1218, 128), (128, 1), 0), primals_66, out=buf1589)
            del primals_66
            buf1598 = reinterpret_tensor(buf1578, (1218, 896), (896, 1), 0); del buf1578  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2199, sum_88, squeeze_43, mul_1087, slice_206, slice_207, neg_112, add_744, mul_1088, add_745, permute_2060, clone_114, view_2207, view_2212, result_48, permute_2069, mm_1199], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1597, (1218, 128), (128, 1), 0), primals_62, out=buf1598)
            del primals_62
            buf1587 = reinterpret_tensor(buf1575, (1218, 896), (896, 1), 0); del buf1575  # reuse
            # Topologically Sorted Source Nodes: [mm_1193], Original ATen: [aten.mm]
            extern_kernels.mm(buf1584, permute_2057, out=buf1587)
            del permute_2057
            buf1595 = buf1574; del buf1574  # reuse
            # Topologically Sorted Source Nodes: [mm_1198], Original ATen: [aten.mm]
            extern_kernels.mm(buf1592, permute_2067, out=buf1595)
            del permute_2067
            buf1599 = reinterpret_tensor(buf1560, (1, 1218, 896), (1091328, 896, 1), 0); del buf1560  # reuse
            buf1606 = reinterpret_tensor(buf1554, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf1554  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1089, slice_208, slice_209, neg_113, add_746, mul_1090, add_747, permute_2070, clone_115, view_2214, mul_1093], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf1577, mm_default, buf1599, buf1606, 1091328, stream=stream0)
            buf1600 = reinterpret_tensor(buf1594, (896, 32), (32, 1), 0); del buf1594  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1089, slice_208, slice_209, neg_113, add_746, mul_1090, add_747, permute_2070, clone_115, view_2214, mul_1093, view_2215, permute_2071, mm_1200], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1599, (896, 1218), (1, 896), 0), mm_36, out=buf1600)
            del mm_36
            buf1601 = buf1592; del buf1592  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1089, slice_208, slice_209, neg_113, add_746, mul_1090, add_747, permute_2070, clone_115, view_2214, mul_1093, view_2215, mm_1201], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1599, (1218, 896), (896, 1), 0), permute_2073, out=buf1601)
            del permute_2073
            buf1603 = reinterpret_tensor(buf1568, (32, 896), (896, 1), 0); del buf1568  # reuse
            # Topologically Sorted Source Nodes: [permute_2075, mm_1202], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1601, (32, 1218), (1, 32), 0), view_107, out=buf1603)
            del view_107
            buf1607 = reinterpret_tensor(buf1599, (1218, 896), (896, 1), 0); del buf1599  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1089, slice_208, slice_209, neg_113, add_746, mul_1090, add_747, permute_2070, clone_115, view_2214, view_2219, result_45, permute_2079, mm_1204], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1606, (1218, 896), (896, 1), 0), primals_58, out=buf1607)
            del primals_58
            buf1602 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4210], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1600, buf1602, 28672, stream=stream0)
            buf1605 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4216], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1603, buf1605, 28672, stream=stream0)
            buf1604 = reinterpret_tensor(buf1606, (1218, 896), (896, 1), 0); del buf1606  # reuse
            # Topologically Sorted Source Nodes: [mm_1203], Original ATen: [aten.mm]
            extern_kernels.mm(buf1601, permute_2077, out=buf1604)
            del permute_2077
            buf1610 = buf1566; del buf1566  # reuse
            buf1611 = reinterpret_tensor(buf1577, (1218, 896), (896, 1), 0); del buf1577  # reuse
            # Topologically Sorted Source Nodes: [view_2204, view_2206, add_748, view_2211, add_749, view_2213, add_750, view_2218, add_751, view_2220, add_752, mul_1094, convert_element_type_4220, hidden_states_20, mul_1095, mul_1096, sum_89, pow_138, mul_1097, mul_1098, expand_121, div_44, pow_139, mul_1099, mul_1100, add_753, convert_element_type_4221, add_754, mul_1101, view_2221], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf1610, buf1587, buf1589, buf1595, buf1598, buf1604, buf1607, primals_57, add_26, rsqrt_4, buf1611, 1218, 896, stream=stream0)
            del add_26
            del buf1587
            del buf1589
            del primals_57
            del rsqrt_4
            buf1612 = reinterpret_tensor(buf1603, (896, 32), (32, 1), 0); del buf1603  # reuse
            # Topologically Sorted Source Nodes: [permute_2080, mm_1205], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1611, (896, 1218), (1, 896), 0), mm_34, out=buf1612)
            del mm_34
            buf1613 = buf1601; del buf1601  # reuse
            # Topologically Sorted Source Nodes: [mm_1206], Original ATen: [aten.mm]
            extern_kernels.mm(buf1611, permute_2082, out=buf1613)
            del permute_2082
            buf1615 = reinterpret_tensor(buf1556, (32, 4864), (4864, 1), 0); del buf1556  # reuse
            # Topologically Sorted Source Nodes: [permute_2084, mm_1207], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1613, (32, 1218), (1, 32), 0), view_101, out=buf1615)
            del view_101
            buf1614 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4226], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1612, buf1614, 28672, stream=stream0)
            buf1617 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4232], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1615, buf1617, 155648, stream=stream0)
            buf1616 = reinterpret_tensor(buf1555, (1218, 4864), (4864, 1), 0); del buf1555  # reuse
            # Topologically Sorted Source Nodes: [mm_1208], Original ATen: [aten.mm]
            extern_kernels.mm(buf1613, permute_2086, out=buf1616)
            del permute_2086
            buf1618 = reinterpret_tensor(buf1546, (1218, 4864), (4864, 1), 0); del buf1546  # reuse
            # Topologically Sorted Source Nodes: [view_2225, result_42, permute_2088, mm_1209], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1610, (1218, 896), (896, 1), 0), primals_54, out=buf1618)
            del primals_54
            buf1619 = buf1562; del buf1562  # reuse
            buf1626 = buf1553; del buf1553  # reuse
            buf1628 = reinterpret_tensor(buf1545, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1545  # reuse
            buf1635 = reinterpret_tensor(buf1543, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1543  # reuse
            # Topologically Sorted Source Nodes: [view_2224, view_2226, add_755, silu_1, mul_1102, mul_1103, mul_1104, convert_element_type_4250, neg_114, exp_22, add_757, reciprocal_22, mul_1105, mul_1106, sub_22, mul_1107, add_758, mul_1108, convert_element_type_4252, mul_1109], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf1616, buf1618, add_23, add_24, buf1619, buf1626, buf1628, buf1635, 5924352, stream=stream0)
            del add_23
            del add_24
            buf1627 = buf1611; del buf1611  # reuse
            # Topologically Sorted Source Nodes: [view_2224, view_2226, add_755, silu_1, mul_1102, view_2231, result_39, permute_2097, mm_1214], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1626, (1218, 4864), (4864, 1), 0), primals_51, out=buf1627)
            del primals_51
            buf1636 = buf1607; del buf1607  # reuse
            # Topologically Sorted Source Nodes: [view_2224, view_2226, add_755, silu_1, mul_1103, convert_element_type_4250, neg_114, exp_22, add_757, reciprocal_22, mul_1105, mul_1106, sub_22, mul_1107, add_758, mul_1108, convert_element_type_4252, view_2237, result_36, permute_2106, mm_1219], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1635, (1218, 4864), (4864, 1), 0), primals_48, out=buf1636)
            del primals_48
            buf1621 = buf1613; del buf1613  # reuse
            # Topologically Sorted Source Nodes: [view_2224, view_2226, add_755, silu_1, mul_1102, mul_1104, view_2227, mm_1211], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1619, (1218, 4864), (4864, 1), 0), permute_2091, out=buf1621)
            del permute_2091
            buf1620 = reinterpret_tensor(buf1615, (4864, 32), (32, 1), 0); del buf1615  # reuse
            # Topologically Sorted Source Nodes: [view_2224, view_2226, add_755, silu_1, mul_1102, mul_1104, view_2227, permute_2089, mm_1210], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1619, (4864, 1218), (1, 4864), 0), mm_31, out=buf1620)
            del mm_31
            buf1630 = buf1584; del buf1584  # reuse
            # Topologically Sorted Source Nodes: [view_2224, view_2226, add_755, silu_1, mul_1103, convert_element_type_4250, neg_114, exp_22, add_757, reciprocal_22, mul_1105, mul_1106, sub_22, mul_1107, add_758, mul_1108, convert_element_type_4252, mul_1109, view_2233, mm_1216], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1628, (1218, 4864), (4864, 1), 0), permute_2100, out=buf1630)
            del permute_2100
            buf1629 = buf1547; del buf1547  # reuse
            # Topologically Sorted Source Nodes: [view_2224, view_2226, add_755, silu_1, mul_1103, convert_element_type_4250, neg_114, exp_22, add_757, reciprocal_22, mul_1105, mul_1106, sub_22, mul_1107, add_758, mul_1108, convert_element_type_4252, mul_1109, view_2233, permute_2098, mm_1215], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1628, (4864, 1218), (1, 4864), 0), mm_28, out=buf1629)
            del mm_28
            buf1623 = reinterpret_tensor(buf1612, (32, 896), (896, 1), 0); del buf1612  # reuse
            # Topologically Sorted Source Nodes: [permute_2093, mm_1212], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1621, (32, 1218), (1, 32), 0), view_89, out=buf1623)
            buf1632 = reinterpret_tensor(buf1600, (32, 896), (896, 1), 0); del buf1600  # reuse
            # Topologically Sorted Source Nodes: [permute_2102, mm_1217], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1630, (32, 1218), (1, 32), 0), view_89, out=buf1632)
            del view_89
            buf1625 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4246], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1623, buf1625, 28672, stream=stream0)
            buf1634 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4263], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1632, buf1634, 28672, stream=stream0)
            buf1622 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4240], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1620, buf1622, 155648, stream=stream0)
            buf1631 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4257], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1629, buf1631, 155648, stream=stream0)
            buf1624 = buf1604; del buf1604  # reuse
            # Topologically Sorted Source Nodes: [mm_1213], Original ATen: [aten.mm]
            extern_kernels.mm(buf1621, permute_2095, out=buf1624)
            del permute_2095
            buf1633 = buf1598; del buf1598  # reuse
            # Topologically Sorted Source Nodes: [mm_1218], Original ATen: [aten.mm]
            extern_kernels.mm(buf1630, permute_2104, out=buf1633)
            del permute_2104
            buf1639 = buf1610; del buf1610  # reuse
            buf1640 = buf1595; del buf1595  # reuse
            # Topologically Sorted Source Nodes: [view_2230, view_2232, add_756, view_2236, add_759, view_2238, add_760, mul_1110, convert_element_type_4267, hidden_states_16, mul_1111, mul_1112, sum_90, pow_140, mul_1113, mul_1114, expand_122, div_45, pow_141, mul_1115, mul_1116, add_761, convert_element_type_4268, add_762, mul_1117, view_2239], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf1639, buf1624, buf1627, buf1633, buf1636, primals_47, add_21, rsqrt_3, buf1640, 1218, 896, stream=stream0)
            del add_21
            del buf1624
            del primals_47
            del rsqrt_3
            buf1641 = reinterpret_tensor(buf1632, (896, 32), (32, 1), 0); del buf1632  # reuse
            # Topologically Sorted Source Nodes: [permute_2107, mm_1220], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1640, (896, 1218), (1, 896), 0), mm_25, out=buf1641)
            del mm_25
            buf1642 = buf1630; del buf1630  # reuse
            # Topologically Sorted Source Nodes: [mm_1221], Original ATen: [aten.mm]
            extern_kernels.mm(buf1640, permute_2109, out=buf1642)
            del permute_2109
            buf1644 = buf1623; del buf1623  # reuse
            # Topologically Sorted Source Nodes: [permute_2111, mm_1222], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1642, (32, 1218), (1, 32), 0), view_83, out=buf1644)
            del view_83
            buf1643 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4273], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1641, buf1643, 28672, stream=stream0)
            buf1646 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4279], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1644, buf1646, 28672, stream=stream0)
            buf1645 = buf1640; del buf1640  # reuse
            # Topologically Sorted Source Nodes: [mm_1223], Original ATen: [aten.mm]
            extern_kernels.mm(buf1642, permute_2113, out=buf1645)
            del permute_2113
            buf1647 = buf1636; del buf1636  # reuse
            # Topologically Sorted Source Nodes: [view_2243, result_33, permute_2115, mm_1224], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1639, (1218, 896), (896, 1), 0), primals_44, out=buf1647)
            del primals_44
            buf1648 = reinterpret_tensor(buf1645, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf1645  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_2242, view_2244, add_763, view_2245, permute_2116, _scaled_dot_product_efficient_attention_backward_22], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf1648, buf1647, 1091328, stream=stream0)
            # Topologically Sorted Source Nodes: [attn_output, view_2242, view_2244, add_763, view_2245, permute_2116, _scaled_dot_product_efficient_attention_backward_22], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf1649 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf1648, add_18, view_78, view_79, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem_4, getitem_5, getitem_6, getitem_7, 0.0, [True, True, True, False], scale=0.125)
            del add_18
            del getitem_4
            del getitem_5
            del getitem_6
            del getitem_7
            del view_78
            del view_79
            buf1650 = buf1649[0]
            assert_size_stride(buf1650, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1650, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1651 = buf1649[1]
            assert_size_stride(buf1651, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1651, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1653 = reinterpret_tensor(buf1597, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf1597  # reuse
            # Topologically Sorted Source Nodes: [view_2247, sum_92], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf1651, buf1653, 155904, stream=stream0)
            buf1652 = buf1649[2]
            assert_size_stride(buf1652, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1652, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf1649
            buf1654 = buf1581; del buf1581  # reuse
            buf1655 = reinterpret_tensor(buf1590, (1218, 128), (128, 1), 0); del buf1590  # reuse
            # Topologically Sorted Source Nodes: [view_2246, sum_91, squeeze_44, permute_2117, clone_116, view_2248, mul_1122, view_2249], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_8.run(buf1652, buf1654, buf1655, 155904, stream=stream0)
            buf1656 = buf1591; del buf1591  # reuse
            # Topologically Sorted Source Nodes: [permute_2118, mm_1225], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1655, (128, 1218), (1, 128), 0), mm_22, out=buf1656)
            del mm_22
            buf1657 = buf1642; del buf1642  # reuse
            # Topologically Sorted Source Nodes: [mm_1226], Original ATen: [aten.mm]
            extern_kernels.mm(buf1655, permute_2120, out=buf1657)
            del permute_2120
            buf1658 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4287], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1656, buf1658, 4096, stream=stream0)
            buf1659 = buf1644; del buf1644  # reuse
            # Topologically Sorted Source Nodes: [permute_2122, mm_1227], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1657, (32, 1218), (1, 32), 0), view_59, out=buf1659)
            buf1661 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4293], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1659, buf1661, 28672, stream=stream0)
            buf1663 = reinterpret_tensor(buf1655, (1, 1218, 128), (155904, 128, 1), 0); del buf1655  # reuse
            buf1670 = reinterpret_tensor(buf1580, (1, 1218, 2, 64), (155904, 128, 64, 1), 0); del buf1580  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2247, sum_92, squeeze_45, mul_1118, slice_210, slice_211, neg_115, add_764, mul_1119, add_765, permute_2127, clone_117, view_2255, mul_1123], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10.run(buf1653, mm_default, buf1663, buf1670, 1218, 128, stream=stream0)
            del buf1653
            buf1664 = buf1656; del buf1656  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2247, sum_92, squeeze_45, mul_1118, slice_210, slice_211, neg_115, add_764, mul_1119, add_765, permute_2127, clone_117, view_2255, mul_1123, view_2256, permute_2128, mm_1230], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1663, (128, 1218), (1, 128), 0), mm_20, out=buf1664)
            del mm_20
            buf1665 = buf1621; del buf1621  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2247, sum_92, squeeze_45, mul_1118, slice_210, slice_211, neg_115, add_764, mul_1119, add_765, permute_2127, clone_117, view_2255, mul_1123, view_2256, mm_1231], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1663, (1218, 128), (128, 1), 0), permute_2130, out=buf1665)
            del buf1663
            del permute_2130
            buf1666 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4301], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1664, buf1666, 4096, stream=stream0)
            buf1667 = buf1659; del buf1659  # reuse
            # Topologically Sorted Source Nodes: [permute_2132, mm_1232], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1665, (32, 1218), (1, 32), 0), view_59, out=buf1667)
            buf1669 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4307], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1667, buf1669, 28672, stream=stream0)
            buf1662 = reinterpret_tensor(buf1652, (1218, 896), (896, 1), 0); del buf1652  # reuse
            # Topologically Sorted Source Nodes: [view_2246, sum_91, squeeze_44, permute_2117, clone_116, view_2248, view_2253, result_30, permute_2126, mm_1229], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1654, (1218, 128), (128, 1), 0), primals_40, out=buf1662)
            del primals_40
            buf1671 = reinterpret_tensor(buf1651, (1218, 896), (896, 1), 0); del buf1651  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2247, sum_92, squeeze_45, mul_1118, slice_210, slice_211, neg_115, add_764, mul_1119, add_765, permute_2127, clone_117, view_2255, view_2260, result_27, permute_2136, mm_1234], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1670, (1218, 128), (128, 1), 0), primals_36, out=buf1671)
            del primals_36
            buf1660 = reinterpret_tensor(buf1648, (1218, 896), (896, 1), 0); del buf1648  # reuse
            # Topologically Sorted Source Nodes: [mm_1228], Original ATen: [aten.mm]
            extern_kernels.mm(buf1657, permute_2124, out=buf1660)
            del permute_2124
            buf1668 = buf1647; del buf1647  # reuse
            # Topologically Sorted Source Nodes: [mm_1233], Original ATen: [aten.mm]
            extern_kernels.mm(buf1665, permute_2134, out=buf1668)
            del permute_2134
            buf1672 = reinterpret_tensor(buf1633, (1, 1218, 896), (1091328, 896, 1), 0); del buf1633  # reuse
            buf1679 = reinterpret_tensor(buf1627, (1, 1218, 14, 64), (1091328, 896, 64, 1), 0); del buf1627  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1120, slice_212, slice_213, neg_116, add_766, mul_1121, add_767, permute_2137, clone_118, view_2262, mul_1124], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_11.run(buf1650, mm_default, buf1672, buf1679, 1091328, stream=stream0)
            buf1673 = reinterpret_tensor(buf1667, (896, 32), (32, 1), 0); del buf1667  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1120, slice_212, slice_213, neg_116, add_766, mul_1121, add_767, permute_2137, clone_118, view_2262, mul_1124, view_2263, permute_2138, mm_1235], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1672, (896, 1218), (1, 896), 0), mm_18, out=buf1673)
            del mm_18
            buf1674 = buf1665; del buf1665  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1120, slice_212, slice_213, neg_116, add_766, mul_1121, add_767, permute_2137, clone_118, view_2262, mul_1124, view_2263, mm_1236], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1672, (1218, 896), (896, 1), 0), permute_2140, out=buf1674)
            del permute_2140
            buf1676 = reinterpret_tensor(buf1641, (32, 896), (896, 1), 0); del buf1641  # reuse
            # Topologically Sorted Source Nodes: [permute_2142, mm_1237], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1674, (32, 1218), (1, 32), 0), view_59, out=buf1676)
            del view_59
            buf1680 = reinterpret_tensor(buf1672, (1218, 896), (896, 1), 0); del buf1672  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1120, slice_212, slice_213, neg_116, add_766, mul_1121, add_767, permute_2137, clone_118, view_2262, view_2267, result_24, permute_2146, mm_1239], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1679, (1218, 896), (896, 1), 0), primals_32, out=buf1680)
            del primals_32
            buf1675 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4315], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1673, buf1675, 28672, stream=stream0)
            buf1678 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4321], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1676, buf1678, 28672, stream=stream0)
            buf1677 = reinterpret_tensor(buf1679, (1218, 896), (896, 1), 0); del buf1679  # reuse
            # Topologically Sorted Source Nodes: [mm_1238], Original ATen: [aten.mm]
            extern_kernels.mm(buf1674, permute_2144, out=buf1677)
            del permute_2144
            buf1683 = buf1639; del buf1639  # reuse
            buf1684 = reinterpret_tensor(buf1650, (1218, 896), (896, 1), 0); del buf1650  # reuse
            # Topologically Sorted Source Nodes: [view_2252, view_2254, add_768, view_2259, add_769, view_2261, add_770, view_2266, add_771, view_2268, add_772, mul_1125, convert_element_type_4325, hidden_states_10, mul_1126, mul_1127, sum_93, pow_142, mul_1128, mul_1129, expand_123, div_46, pow_143, mul_1130, mul_1131, add_773, convert_element_type_4326, add_774, mul_1132, view_2269], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_12.run(buf1683, buf1660, buf1662, buf1668, buf1671, buf1677, buf1680, primals_31, add_13, rsqrt_2, buf1684, 1218, 896, stream=stream0)
            del add_13
            del buf1660
            del buf1662
            del primals_31
            del rsqrt_2
            buf1685 = reinterpret_tensor(buf1676, (896, 32), (32, 1), 0); del buf1676  # reuse
            # Topologically Sorted Source Nodes: [permute_2147, mm_1240], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1684, (896, 1218), (1, 896), 0), mm_16, out=buf1685)
            del mm_16
            buf1686 = buf1674; del buf1674  # reuse
            # Topologically Sorted Source Nodes: [mm_1241], Original ATen: [aten.mm]
            extern_kernels.mm(buf1684, permute_2149, out=buf1686)
            del permute_2149
            buf1688 = reinterpret_tensor(buf1629, (32, 4864), (4864, 1), 0); del buf1629  # reuse
            # Topologically Sorted Source Nodes: [permute_2151, mm_1242], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1686, (32, 1218), (1, 32), 0), view_53, out=buf1688)
            del view_53
            buf1687 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4331], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1685, buf1687, 28672, stream=stream0)
            buf1690 = empty_strided_cuda((32, 4864), (4864, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4337], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1688, buf1690, 155648, stream=stream0)
            buf1689 = reinterpret_tensor(buf1628, (1218, 4864), (4864, 1), 0); del buf1628  # reuse
            # Topologically Sorted Source Nodes: [mm_1243], Original ATen: [aten.mm]
            extern_kernels.mm(buf1686, permute_2153, out=buf1689)
            del permute_2153
            buf1691 = reinterpret_tensor(buf1619, (1218, 4864), (4864, 1), 0); del buf1619  # reuse
            # Topologically Sorted Source Nodes: [view_2273, result_21, permute_2155, mm_1244], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1683, (1218, 896), (896, 1), 0), primals_28, out=buf1691)
            del primals_28
            buf1692 = buf1635; del buf1635  # reuse
            buf1699 = buf1626; del buf1626  # reuse
            buf1701 = reinterpret_tensor(buf1618, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1618  # reuse
            buf1708 = reinterpret_tensor(buf1616, (1, 1218, 4864), (5924352, 4864, 1), 0); del buf1616  # reuse
            # Topologically Sorted Source Nodes: [view_2272, view_2274, add_775, silu, mul_1133, mul_1134, mul_1135, convert_element_type_4355, neg_117, exp_23, add_777, reciprocal_23, mul_1136, mul_1137, sub_23, mul_1138, add_778, mul_1139, convert_element_type_4357, mul_1140], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused_add_mul_silu_silu_backward_view_4.run(buf1689, buf1691, add_10, add_11, buf1692, buf1699, buf1701, buf1708, 5924352, stream=stream0)
            del add_10
            del add_11
            del buf1689
            del buf1691
            buf1700 = buf1684; del buf1684  # reuse
            # Topologically Sorted Source Nodes: [view_2272, view_2274, add_775, silu, mul_1133, view_2279, result_18, permute_2164, mm_1249], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1699, (1218, 4864), (4864, 1), 0), primals_25, out=buf1700)
            del buf1699
            del primals_25
            buf1709 = buf1680; del buf1680  # reuse
            # Topologically Sorted Source Nodes: [view_2272, view_2274, add_775, silu, mul_1134, convert_element_type_4355, neg_117, exp_23, add_777, reciprocal_23, mul_1136, mul_1137, sub_23, mul_1138, add_778, mul_1139, convert_element_type_4357, view_2285, result_15, permute_2173, mm_1254], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1708, (1218, 4864), (4864, 1), 0), primals_22, out=buf1709)
            del buf1708
            del primals_22
            buf1694 = buf1686; del buf1686  # reuse
            # Topologically Sorted Source Nodes: [view_2272, view_2274, add_775, silu, mul_1133, mul_1135, view_2275, mm_1246], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1692, (1218, 4864), (4864, 1), 0), permute_2158, out=buf1694)
            del permute_2158
            buf1693 = reinterpret_tensor(buf1688, (4864, 32), (32, 1), 0); del buf1688  # reuse
            # Topologically Sorted Source Nodes: [view_2272, view_2274, add_775, silu, mul_1133, mul_1135, view_2275, permute_2156, mm_1245], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1692, (4864, 1218), (1, 4864), 0), mm_13, out=buf1693)
            del buf1692
            del mm_13
            buf1703 = buf1657; del buf1657  # reuse
            # Topologically Sorted Source Nodes: [view_2272, view_2274, add_775, silu, mul_1134, convert_element_type_4355, neg_117, exp_23, add_777, reciprocal_23, mul_1136, mul_1137, sub_23, mul_1138, add_778, mul_1139, convert_element_type_4357, mul_1140, view_2281, mm_1251], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1701, (1218, 4864), (4864, 1), 0), permute_2167, out=buf1703)
            del permute_2167
            buf1702 = buf1620; del buf1620  # reuse
            # Topologically Sorted Source Nodes: [view_2272, view_2274, add_775, silu, mul_1134, convert_element_type_4355, neg_117, exp_23, add_777, reciprocal_23, mul_1136, mul_1137, sub_23, mul_1138, add_778, mul_1139, convert_element_type_4357, mul_1140, view_2281, permute_2165, mm_1250], Original ATen: [aten.view, aten.add, aten.silu, aten.mul, aten.silu_backward, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1701, (4864, 1218), (1, 4864), 0), mm_10, out=buf1702)
            del buf1701
            del mm_10
            buf1696 = reinterpret_tensor(buf1685, (32, 896), (896, 1), 0); del buf1685  # reuse
            # Topologically Sorted Source Nodes: [permute_2160, mm_1247], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1694, (32, 1218), (1, 32), 0), view_41, out=buf1696)
            buf1705 = reinterpret_tensor(buf1673, (32, 896), (896, 1), 0); del buf1673  # reuse
            # Topologically Sorted Source Nodes: [permute_2169, mm_1252], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1703, (32, 1218), (1, 32), 0), view_41, out=buf1705)
            del view_41
            buf1698 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4351], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1696, buf1698, 28672, stream=stream0)
            buf1707 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4368], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1705, buf1707, 28672, stream=stream0)
            buf1695 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4345], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1693, buf1695, 155648, stream=stream0)
            del buf1693
            buf1704 = empty_strided_cuda((4864, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4362], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_3.run(buf1702, buf1704, 155648, stream=stream0)
            del buf1702
            buf1697 = buf1677; del buf1677  # reuse
            # Topologically Sorted Source Nodes: [mm_1248], Original ATen: [aten.mm]
            extern_kernels.mm(buf1694, permute_2162, out=buf1697)
            del buf1694
            del permute_2162
            buf1706 = buf1671; del buf1671  # reuse
            # Topologically Sorted Source Nodes: [mm_1253], Original ATen: [aten.mm]
            extern_kernels.mm(buf1703, permute_2171, out=buf1706)
            del permute_2171
            buf1712 = buf1683; del buf1683  # reuse
            buf1713 = buf1668; del buf1668  # reuse
            # Topologically Sorted Source Nodes: [view_2278, view_2280, add_776, view_2284, add_779, view_2286, add_780, mul_1141, convert_element_type_4372, hidden_states_6, mul_1142, mul_1143, sum_94, pow_144, mul_1144, mul_1145, expand_124, div_47, pow_145, mul_1146, mul_1147, add_781, convert_element_type_4373, add_782, mul_1148, view_2287], Original ATen: [aten.view, aten.add, aten.mul, aten._to_copy, aten.sum, aten.pow, aten.expand, aten.div]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_div_expand_mul_pow_sum_view_5.run(buf1712, buf1697, buf1700, buf1706, buf1709, primals_21, add_8, rsqrt_1, buf1713, 1218, 896, stream=stream0)
            del add_8
            del buf1697
            del buf1700
            del buf1706
            del buf1709
            del primals_21
            del rsqrt_1
            buf1714 = reinterpret_tensor(buf1705, (896, 32), (32, 1), 0); del buf1705  # reuse
            # Topologically Sorted Source Nodes: [permute_2174, mm_1255], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1713, (896, 1218), (1, 896), 0), mm_7, out=buf1714)
            del mm_7
            buf1715 = buf1703; del buf1703  # reuse
            # Topologically Sorted Source Nodes: [mm_1256], Original ATen: [aten.mm]
            extern_kernels.mm(buf1713, permute_2176, out=buf1715)
            del permute_2176
            buf1717 = buf1696; del buf1696  # reuse
            # Topologically Sorted Source Nodes: [permute_2178, mm_1257], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1715, (32, 1218), (1, 32), 0), view_35, out=buf1717)
            del view_35
            buf1720 = buf1713; del buf1713  # reuse
            # Topologically Sorted Source Nodes: [view_2291, result_12, permute_2182, mm_1259], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1712, (1218, 896), (896, 1), 0), primals_18, out=buf1720)
            del primals_18
            buf1716 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4378], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1714, buf1716, 28672, stream=stream0)
            buf1719 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4384], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1717, buf1719, 28672, stream=stream0)
            buf1718 = reinterpret_tensor(buf1712, (1218, 896), (896, 1), 0); del buf1712  # reuse
            # Topologically Sorted Source Nodes: [mm_1258], Original ATen: [aten.mm]
            extern_kernels.mm(buf1715, permute_2180, out=buf1718)
            del permute_2180
            buf1721 = reinterpret_tensor(buf1718, (1, 14, 1218, 64), (1091328, 64, 896, 1), 0); del buf1718  # reuse
            # Topologically Sorted Source Nodes: [attn_output, view_2290, view_2292, add_783, view_2293, permute_2183, _scaled_dot_product_efficient_attention_backward_23], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            stream0 = get_raw_stream(0)
            triton_poi_fused__scaled_dot_product_efficient_attention_backward_add_expand_slice_transpose_view_6.run(buf1721, buf1720, 1091328, stream=stream0)
            del buf1720
            # Topologically Sorted Source Nodes: [attn_output, view_2290, view_2292, add_783, view_2293, permute_2183, _scaled_dot_product_efficient_attention_backward_23], Original ATen: [aten.slice, aten.expand, aten.view, aten.add, aten.transpose, aten._scaled_dot_product_efficient_attention_backward]
            buf1722 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(buf1721, add_5, view_30, view_31, reinterpret_tensor(constant_pad_nd, (1, 14, 1218, 1218), (1559040, 0, 1280, 1), 0), getitem, getitem_1, getitem_2, getitem_3, 0.0, [True, True, True, False], scale=0.125)
            del add_5
            del buf1721
            del constant_pad_nd
            del getitem
            del getitem_1
            del getitem_2
            del getitem_3
            del view_30
            del view_31
            buf1723 = buf1722[0]
            assert_size_stride(buf1723, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1723, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1724 = buf1722[1]
            assert_size_stride(buf1724, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1724, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            buf1726 = reinterpret_tensor(buf1670, (1, 2, 1, 1218, 64), (155904, 64, 155904, 128, 1), 0); del buf1670  # reuse
            # Topologically Sorted Source Nodes: [view_2295, sum_96], Original ATen: [aten.view, aten.sum]
            stream0 = get_raw_stream(0)
            triton_poi_fused_sum_view_7.run(buf1724, buf1726, 155904, stream=stream0)
            del buf1724
            buf1725 = buf1722[2]
            assert_size_stride(buf1725, (1, 14, 1218, 64), (1091328, 64, 896, 1), 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            assert_alignment(buf1725, 16, 'torch.ops.aten._scaled_dot_product_efficient_attention_backward.default')
            del buf1722
            buf1727 = reinterpret_tensor(buf1654, (1, 1218, 128), (155904, 128, 1), 0); del buf1654  # reuse
            # Topologically Sorted Source Nodes: [view_2294, sum_95, squeeze_46, permute_2184, clone_119, view_2296, mul_1153], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul]
            stream0 = get_raw_stream(0)
            triton_poi_fused__unsafe_view_clone_mul_squeeze_sum_transpose_view_13.run(buf1725, buf1727, 155904, stream=stream0)
            buf1728 = buf1664; del buf1664  # reuse
            # Topologically Sorted Source Nodes: [view_2294, sum_95, squeeze_46, permute_2184, clone_119, view_2296, mul_1153, view_2297, permute_2185, mm_1260], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1727, (128, 1218), (1, 128), 0), mm_4, out=buf1728)
            del mm_4
            buf1729 = buf1715; del buf1715  # reuse
            # Topologically Sorted Source Nodes: [view_2294, sum_95, squeeze_46, permute_2184, clone_119, view_2296, mul_1153, view_2297, mm_1261], Original ATen: [aten.view, aten.sum, aten.squeeze, aten.transpose, aten.clone, aten._unsafe_view, aten.mul, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1727, (1218, 128), (128, 1), 0), permute_2187, out=buf1729)
            del permute_2187
            buf1731 = buf1717; del buf1717  # reuse
            # Topologically Sorted Source Nodes: [permute_2189, mm_1262], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1729, (32, 1218), (1, 32), 0), view_11, out=buf1731)
            buf1733 = buf1727; del buf1727  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2295, sum_96, squeeze_47, mul_1149, slice_214, slice_215, neg_118, add_784, mul_1150, add_785, permute_2192, clone_120, view_2300, mul_1154], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_14.run(buf1726, mm_default, buf1733, 1218, 128, stream=stream0)
            del buf1726
            buf1739 = reinterpret_tensor(buf1725, (1, 1218, 896), (1091328, 896, 1), 0); del buf1725  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1151, slice_216, slice_217, neg_119, add_786, mul_1152, add_787, permute_2200, clone_121, view_2304, mul_1155], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_transpose_unsqueeze_15.run(buf1723, mm_default, buf1739, 1091328, stream=stream0)
            del buf1723
            del mm_default
            buf1734 = empty_strided_cuda((128, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2295, sum_96, squeeze_47, mul_1149, slice_214, slice_215, neg_118, add_784, mul_1150, add_785, permute_2192, clone_120, view_2300, mul_1154, view_2301, permute_2193, mm_1263], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1733, (128, 1218), (1, 128), 0), mm_2, out=buf1734)
            del mm_2
            buf1735 = buf1729; del buf1729  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, full_default_50, cos, cos_1, cos_2, cos_3, view_2295, sum_96, squeeze_47, mul_1149, slice_214, slice_215, neg_118, add_784, mul_1150, add_785, permute_2192, clone_120, view_2300, mul_1154, view_2301, mm_1264], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.slice_backward, aten.cos, aten.view, aten.sum, aten.squeeze, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1733, (1218, 128), (128, 1), 0), permute_2195, out=buf1735)
            del buf1733
            del permute_2195
            buf1737 = reinterpret_tensor(buf1714, (32, 896), (896, 1), 0); del buf1714  # reuse
            # Topologically Sorted Source Nodes: [permute_2197, mm_1265], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1735, (32, 1218), (1, 32), 0), view_11, out=buf1737)
            buf1740 = empty_strided_cuda((896, 32), (32, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1151, slice_216, slice_217, neg_119, add_786, mul_1152, add_787, permute_2200, clone_121, view_2304, mul_1155, view_2305, permute_2201, mm_1266], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1739, (896, 1218), (1, 896), 0), mm, out=buf1740)
            del mm
            buf1741 = buf1735; del buf1735  # reuse
            # Topologically Sorted Source Nodes: [matmul, freqs, emb, sin, sin_1, sin_2, sin_3, cos, cos_1, cos_2, cos_3, full_default_52, mul_1151, slice_216, slice_217, neg_119, add_786, mul_1152, add_787, permute_2200, clone_121, view_2304, mul_1155, view_2305, mm_1267], Original ATen: [aten.bmm, aten.transpose, aten.cat, aten.sin, aten.mul, aten._to_copy, aten.unsqueeze, aten.cos, aten.slice_backward, aten.slice, aten.neg, aten.add, aten.clone, aten._unsafe_view, aten.view, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1739, (1218, 896), (896, 1), 0), permute_2203, out=buf1741)
            del buf1739
            del permute_2203
            buf1743 = empty_strided_cuda((32, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [permute_2205, mm_1268], Original ATen: [aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(buf1741, (32, 1218), (1, 32), 0), view_11, out=buf1743)
            del buf1741
            del view_11
            buf1730 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4392], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1728, buf1730, 4096, stream=stream0)
            del buf1728
            buf1736 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4400], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_9.run(buf1734, buf1736, 4096, stream=stream0)
            del buf1734
            buf1732 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4395], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1731, buf1732, 28672, stream=stream0)
            del buf1731
            buf1738 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4403], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1737, buf1738, 28672, stream=stream0)
            del buf1737
            buf1742 = empty_strided_cuda((896, 32), (32, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4408], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1740, buf1742, 28672, stream=stream0)
            del buf1740
            buf1744 = empty_strided_cuda((32, 896), (896, 1), torch.float32)
            # Topologically Sorted Source Nodes: [convert_element_type_4411], Original ATen: [aten._to_copy]
            stream0 = get_raw_stream(0)
            triton_poi_fused__to_copy_2.run(buf1743, buf1744, 28672, stream=stream0)
            del buf1743
        return (None, None, None, None, None, None, None, buf1744, buf1742, None, None, buf1738, buf1736, None, None, buf1732, buf1730, None, buf1719, buf1716, None, None, buf1707, buf1704, None, buf1698, buf1695, None, buf1690, buf1687, None, None, None, buf1678, buf1675, None, None, buf1669, buf1666, None, None, buf1661, buf1658, None, buf1646, buf1643, None, None, buf1634, buf1631, None, buf1625, buf1622, None, buf1617, buf1614, None, None, None, buf1605, buf1602, None, None, buf1596, buf1593, None, None, buf1588, buf1585, None, buf1573, buf1570, None, None, buf1561, buf1558, None, buf1552, buf1549, None, buf1544, buf1541, None, None, None, buf1532, buf1529, None, None, buf1523, buf1520, None, None, buf1515, buf1512, None, buf1500, buf1497, None, None, buf1488, buf1485, None, buf1479, buf1476, None, buf1471, buf1468, None, None, None, buf1459, buf1456, None, None, buf1450, buf1447, None, None, buf1442, buf1439, None, buf1427, buf1424, None, None, buf1415, buf1412, None, buf1406, buf1403, None, buf1398, buf1395, None, None, None, buf1386, buf1383, None, None, buf1377, buf1374, None, None, buf1369, buf1366, None, buf1354, buf1351, None, None, buf1342, buf1339, None, buf1333, buf1330, None, buf1325, buf1322, None, None, None, buf1313, buf1310, None, None, buf1304, buf1301, None, None, buf1296, buf1293, None, buf1281, buf1278, None, None, buf1269, buf1266, None, buf1260, buf1257, None, buf1252, buf1249, None, None, None, buf1240, buf1237, None, None, buf1231, buf1228, None, None, buf1223, buf1220, None, buf1208, buf1205, None, None, buf1196, buf1193, None, buf1187, buf1184, None, buf1179, buf1176, None, None, None, buf1167, buf1164, None, None, buf1158, buf1155, None, None, buf1150, buf1147, None, buf1135, buf1132, None, None, buf1123, buf1120, None, buf1114, buf1111, None, buf1106, buf1103, None, None, None, buf1094, buf1091, None, None, buf1085, buf1082, None, None, buf1077, buf1074, None, buf1062, buf1059, None, None, buf1050, buf1047, None, buf1041, buf1038, None, buf1033, buf1030, None, None, None, buf1021, buf1018, None, None, buf1012, buf1009, None, None, buf1004, buf1001, None, buf989, buf986, None, None, buf977, buf974, None, buf968, buf965, None, buf960, buf957, None, None, None, buf948, buf945, None, None, buf939, buf936, None, None, buf931, buf928, None, buf916, buf913, None, None, buf904, buf901, None, buf895, buf892, None, buf887, buf884, None, None, None, buf875, buf872, None, None, buf866, buf863, None, None, buf858, buf855, None, buf843, buf840, None, None, buf831, buf828, None, buf822, buf819, None, buf814, buf811, None, None, None, buf802, buf799, None, None, buf793, buf790, None, None, buf785, buf782, None, buf770, buf767, None, None, buf758, buf755, None, buf749, buf746, None, buf741, buf738, None, None, None, buf729, buf726, None, None, buf720, buf717, None, None, buf712, buf709, None, buf697, buf694, None, None, buf685, buf682, None, buf676, buf673, None, buf668, buf665, None, None, None, buf656, buf653, None, None, buf647, buf644, None, None, buf639, buf636, None, buf624, buf621, None, None, buf612, buf609, None, buf603, buf600, None, buf595, buf592, None, None, None, buf583, buf580, None, None, buf574, buf571, None, None, buf566, buf563, None, buf551, buf548, None, None, buf539, buf536, None, buf530, buf527, None, buf522, buf519, None, None, None, buf510, buf507, None, None, buf501, buf498, None, None, buf493, buf490, None, buf478, buf475, None, None, buf466, buf463, None, buf457, buf454, None, buf449, buf446, None, None, None, buf437, buf434, None, None, buf428, buf425, None, None, buf420, buf417, None, buf405, buf402, None, None, buf393, buf390, None, buf384, buf381, None, buf376, buf373, None, None, None, buf364, buf361, None, None, buf355, buf352, None, None, buf347, buf344, None, buf332, buf329, None, None, buf320, buf317, None, buf311, buf308, None, buf303, buf300, None, None, None, buf291, buf288, None, None, buf282, buf279, None, None, buf274, buf271, None, buf259, buf256, None, None, buf247, buf244, None, buf238, buf235, None, buf230, buf227, None, None, None, buf218, buf215, None, None, buf209, buf206, None, None, buf201, buf198, None, buf186, buf183, None, None, buf174, buf171, None, buf165, buf162, None, buf157, buf154, None, None, None, buf145, buf142, None, None, buf136, buf133, None, None, buf128, buf125, None, buf113, buf110, None, None, buf101, buf98, None, buf92, buf89, None, buf84, buf81, None, None, None, buf72, buf69, None, None, buf63, buf60, None, None, buf55, buf52, None, buf40, buf37, None, None, buf28, buf25, None, buf19, buf16, None, buf11, buf8, None, buf1, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_18 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_21 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_22 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_25 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_28 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_31 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_32 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_36 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_40 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_44 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_47 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_48 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_51 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_54 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_57 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_58 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_62 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_66 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_70 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_73 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_74 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_77 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_80 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_83 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_84 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_88 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_92 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_96 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_99 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_100 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_103 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_106 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_109 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_110 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_114 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_118 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_122 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_125 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_126 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_129 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_132 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_135 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_136 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_140 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_144 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_148 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_151 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_152 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_155 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_158 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_161 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_162 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_166 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_170 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_174 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_177 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_178 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_181 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_184 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_187 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_188 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_192 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_196 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_200 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_203 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_204 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_207 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_210 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_213 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_214 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_218 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_222 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_226 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_229 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_230 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_233 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_236 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_239 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_240 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_244 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_248 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_252 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_255 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_256 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_259 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_262 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_265 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_266 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_270 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_274 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_278 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_281 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_282 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_285 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_288 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_291 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_292 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_296 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_300 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_304 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_307 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_308 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_311 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_314 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_317 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_318 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_322 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_326 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_330 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_333 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_334 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_337 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_340 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_343 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_344 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_348 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_352 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_356 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_359 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_360 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_363 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_366 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_369 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_370 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_374 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_378 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_382 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_385 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_386 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_389 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_392 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_395 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_396 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_400 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_404 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_408 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_411 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_412 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_415 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_418 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_421 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_422 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_426 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_430 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_434 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_437 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_438 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_441 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_444 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_447 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_448 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_452 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_456 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_460 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_463 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_464 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_467 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_470 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_473 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_474 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_478 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_482 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_486 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_489 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_490 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_493 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_496 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_499 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_500 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_504 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_508 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_512 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_515 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_516 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_519 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_522 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_525 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_526 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_530 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_534 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_538 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_541 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_542 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_545 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_548 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_551 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_552 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_556 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_560 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_564 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_567 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_568 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_571 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_574 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_577 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_578 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_582 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_586 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_590 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_593 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_594 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_597 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_600 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_603 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_604 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_608 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_612 = rand_strided((128, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_616 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_619 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_620 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_623 = rand_strided((4864, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_626 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_629 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    primals_630 = rand_strided((151936, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_default = rand_strided((32, 1218), (1248, 1), device='cuda:0', dtype=torch.float32)
    view_11 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_2 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_4 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_5 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_30 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_31 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    constant_pad_nd = rand_strided((1, 1, 1218, 1224), (1559040, 1559040, 1280, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_1 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_2 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_3 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_35 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_7 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_8 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_1 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_41 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_10 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_10 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_13 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_11 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_53 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_16 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_13 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_2 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_59 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_18 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_20 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_22 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_18 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_78 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_79 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_4 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_5 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_6 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_7 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_83 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_25 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_21 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_3 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_89 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_28 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_23 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_31 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_24 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_101 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_34 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_26 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_4 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_107 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_36 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_38 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_40 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_31 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_126 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_127 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_8 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_9 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_10 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_11 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_131 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_43 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_34 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_5 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_137 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_46 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_36 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_49 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_37 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_149 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_52 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_39 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_6 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_155 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_54 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_56 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_58 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_44 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_174 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_175 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_12 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_13 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_14 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_15 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_179 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_61 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_47 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_7 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_185 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_64 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_49 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_67 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_50 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_197 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_70 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_52 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_8 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_203 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_72 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_74 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_76 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_57 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_222 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_223 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_16 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_17 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_18 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_19 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_227 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_79 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_60 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_9 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_233 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_82 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_62 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_85 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_63 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_245 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_88 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_65 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_10 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_251 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_90 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_92 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_94 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_70 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_270 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_271 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_20 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_21 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_22 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_23 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_275 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_97 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_73 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_11 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_281 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_100 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_75 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_103 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_76 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_293 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_106 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_78 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_12 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_299 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_108 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_110 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_112 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_83 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_318 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_319 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_24 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_25 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_26 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_27 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_323 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_115 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_86 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_13 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_329 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_118 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_88 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_121 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_89 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_341 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_124 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_91 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_14 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_347 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_126 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_128 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_130 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_96 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_366 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_367 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_28 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_29 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_30 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_31 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_371 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_133 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_99 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_15 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_377 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_136 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_101 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_139 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_102 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_389 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_142 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_104 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_16 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_395 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_144 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_146 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_148 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_109 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_414 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_415 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_32 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_33 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_34 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_35 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_419 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_151 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_112 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_17 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_425 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_154 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_114 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_157 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_115 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_437 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_160 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_117 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_18 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_443 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_162 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_164 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_166 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_122 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_462 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_463 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_36 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_37 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_38 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_39 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_467 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_169 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_125 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_19 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_473 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_172 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_127 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_175 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_128 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_485 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_178 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_130 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_20 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_491 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_180 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_182 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_184 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_135 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_510 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_511 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_40 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_41 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_42 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_43 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_515 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_187 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_138 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_21 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_521 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_190 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_140 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_193 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_141 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_533 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_196 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_143 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_22 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_539 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_198 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_200 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_202 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_148 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_558 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_559 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_44 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_45 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_46 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_47 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_563 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_205 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_151 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_23 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_569 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_208 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_153 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_211 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_154 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_581 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_214 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_156 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_24 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_587 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_216 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_218 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_220 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_161 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_606 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_607 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_48 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_49 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_50 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_51 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_611 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_223 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_164 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_25 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_617 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_226 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_166 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_229 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_167 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_629 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_232 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_169 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_26 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_635 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_234 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_236 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_238 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_174 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_654 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_655 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_52 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_53 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_54 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_55 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_659 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_241 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_177 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_27 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_665 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_244 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_179 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_247 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_180 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_677 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_250 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_182 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_28 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_683 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_252 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_254 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_256 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_187 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_702 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_703 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_56 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_57 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_58 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_59 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_707 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_259 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_190 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_29 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_713 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_262 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_192 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_265 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_193 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_725 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_268 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_195 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_30 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_731 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_270 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_272 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_274 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_200 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_750 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_751 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_60 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_61 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_62 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_63 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_755 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_277 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_203 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_31 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_761 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_280 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_205 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_283 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_206 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_773 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_286 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_208 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_32 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_779 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_288 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_290 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_292 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_213 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_798 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_799 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_64 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_65 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_66 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_67 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_803 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_295 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_216 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_33 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_809 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_298 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_218 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_301 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_219 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_821 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_304 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_221 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_34 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_827 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_306 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_308 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_310 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_226 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_846 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_847 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_68 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_69 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_70 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_71 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_851 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_313 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_229 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_35 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_857 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_316 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_231 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_319 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_232 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_869 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_322 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_234 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_36 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_875 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_324 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_326 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_328 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_239 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_894 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_895 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_72 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_73 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_74 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_75 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_899 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_331 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_242 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_37 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_905 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_334 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_244 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_337 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_245 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_917 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_340 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_247 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_38 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_923 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_342 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_344 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_346 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_252 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_942 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_943 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_76 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_77 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_78 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_79 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_947 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_349 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_255 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_39 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_953 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_352 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_257 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_355 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_258 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_965 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_358 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_260 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_40 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_971 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_360 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_362 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_364 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_265 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_990 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_991 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_80 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_81 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_82 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_83 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_995 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_367 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_268 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_41 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1001 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_370 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_270 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_373 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_271 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1013 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_376 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_273 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_42 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1019 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_378 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_380 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_382 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_278 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1038 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1039 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_84 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_85 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_86 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_87 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_1043 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_385 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_281 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_43 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1049 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_388 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_283 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_391 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_284 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1061 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_394 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_286 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_44 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1067 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_396 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_398 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_400 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_291 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1086 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1087 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_88 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_89 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_90 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_91 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_1091 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_403 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_294 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_45 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1097 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_406 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_296 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_409 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_297 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1109 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_412 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_299 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_46 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1115 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_414 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_416 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_418 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_304 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1134 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1135 = rand_strided((1, 14, 1218, 64), (1091328, 77952, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_92 = rand_strided((1, 14, 1218, 64), (1091328, 64, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    getitem_93 = rand_strided((1, 14, 1248), (17472, 1248, 1), device='cuda:0', dtype=torch.float32)
    getitem_94 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    getitem_95 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    view_1139 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_421 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_307 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_47 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1145 = rand_strided((1218, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_424 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_309 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_427 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_310 = rand_strided((1, 1218, 4864), (5924352, 4864, 1), device='cuda:0', dtype=torch.bfloat16)
    view_1157 = rand_strided((1218, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    mm_430 = rand_strided((1218, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    add_312 = rand_strided((1, 1218, 896), (1091328, 896, 1), device='cuda:0', dtype=torch.bfloat16)
    rsqrt_48 = rand_strided((1, 1218, 1), (1248, 1, 1), device='cuda:0', dtype=torch.float32)
    view_1161 = rand_strided((1025, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_608 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_612 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_617 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_621 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_626 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_630 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_635 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_639 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_646 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_650 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_656 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_660 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_666 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_670 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_675 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_679 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_684 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_688 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_693 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_697 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_702 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_706 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_713 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_717 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_723 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_727 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_733 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_737 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_742 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_746 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_751 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_755 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_760 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_764 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_769 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_773 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_780 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_784 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_790 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_794 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_800 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_804 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_809 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_813 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_818 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_822 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_827 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_831 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_836 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_840 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_847 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_851 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_857 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_861 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_867 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_871 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_876 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_880 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_885 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_889 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_894 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_898 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_903 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_907 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_914 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_918 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_924 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_928 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_934 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_938 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_943 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_947 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_952 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_956 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_961 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_965 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_970 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_974 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_981 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_985 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_991 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_995 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1001 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1005 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1010 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1014 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1019 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1023 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1028 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1032 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1037 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1041 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1048 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1052 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1058 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1062 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1068 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1072 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1077 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1081 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1086 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1090 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1095 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1099 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1104 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1108 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1115 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1119 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1125 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1129 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1135 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1139 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1144 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1148 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1153 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1157 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1162 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1166 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1171 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1175 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1182 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1186 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1192 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1196 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1202 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1206 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1211 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1215 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1220 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1224 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1229 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1233 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1238 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1242 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1249 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1253 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1259 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1263 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1269 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1273 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1278 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1282 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1287 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1291 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1296 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1300 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1305 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1309 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1316 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1320 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1326 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1330 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1336 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1340 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1345 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1349 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1354 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1358 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1363 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1367 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1372 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1376 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1383 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1387 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1393 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1397 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1403 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1407 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1412 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1416 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1421 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1425 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1430 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1434 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1439 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1443 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1450 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1454 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1460 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1464 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1470 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1474 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1479 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1483 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1488 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1492 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1497 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1501 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1506 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1510 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1517 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1521 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1527 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1531 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1537 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1541 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1546 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1550 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1555 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1559 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1564 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1568 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1573 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1577 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1584 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1588 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1594 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1598 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1604 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1608 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1613 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1617 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1622 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1626 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1631 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1635 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1640 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1644 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1651 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1655 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1661 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1665 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1671 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1675 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1680 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1684 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1689 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1693 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1698 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1702 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1707 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1711 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1718 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1722 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1728 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1732 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1738 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1742 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1747 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1751 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1756 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1760 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1765 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1769 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1774 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1778 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1785 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1789 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1795 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1799 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1805 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1809 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1814 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1818 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1823 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1827 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1832 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1836 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1841 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1845 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1852 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1856 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1862 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1866 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1872 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1876 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1881 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1885 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1890 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1894 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1899 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1903 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1908 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1912 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1919 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1923 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1929 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1933 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1939 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1943 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1948 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1952 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1957 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1961 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1966 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1970 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1975 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1979 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1986 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1990 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_1996 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2000 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2006 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2010 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2015 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2019 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2024 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2028 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2033 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2037 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2042 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2046 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2053 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2057 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2063 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2067 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2073 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2077 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2082 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2086 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2091 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2095 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2100 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2104 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2109 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2113 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2120 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2124 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2130 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2134 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2140 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2144 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2149 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2153 = rand_strided((32, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2158 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2162 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2167 = rand_strided((4864, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2171 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2176 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2180 = rand_strided((32, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2187 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2195 = rand_strided((128, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    permute_2203 = rand_strided((896, 32), (32, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((1, 1025, 151936), (155734400, 151936, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_18, primals_21, primals_22, primals_25, primals_28, primals_31, primals_32, primals_36, primals_40, primals_44, primals_47, primals_48, primals_51, primals_54, primals_57, primals_58, primals_62, primals_66, primals_70, primals_73, primals_74, primals_77, primals_80, primals_83, primals_84, primals_88, primals_92, primals_96, primals_99, primals_100, primals_103, primals_106, primals_109, primals_110, primals_114, primals_118, primals_122, primals_125, primals_126, primals_129, primals_132, primals_135, primals_136, primals_140, primals_144, primals_148, primals_151, primals_152, primals_155, primals_158, primals_161, primals_162, primals_166, primals_170, primals_174, primals_177, primals_178, primals_181, primals_184, primals_187, primals_188, primals_192, primals_196, primals_200, primals_203, primals_204, primals_207, primals_210, primals_213, primals_214, primals_218, primals_222, primals_226, primals_229, primals_230, primals_233, primals_236, primals_239, primals_240, primals_244, primals_248, primals_252, primals_255, primals_256, primals_259, primals_262, primals_265, primals_266, primals_270, primals_274, primals_278, primals_281, primals_282, primals_285, primals_288, primals_291, primals_292, primals_296, primals_300, primals_304, primals_307, primals_308, primals_311, primals_314, primals_317, primals_318, primals_322, primals_326, primals_330, primals_333, primals_334, primals_337, primals_340, primals_343, primals_344, primals_348, primals_352, primals_356, primals_359, primals_360, primals_363, primals_366, primals_369, primals_370, primals_374, primals_378, primals_382, primals_385, primals_386, primals_389, primals_392, primals_395, primals_396, primals_400, primals_404, primals_408, primals_411, primals_412, primals_415, primals_418, primals_421, primals_422, primals_426, primals_430, primals_434, primals_437, primals_438, primals_441, primals_444, primals_447, primals_448, primals_452, primals_456, primals_460, primals_463, primals_464, primals_467, primals_470, primals_473, primals_474, primals_478, primals_482, primals_486, primals_489, primals_490, primals_493, primals_496, primals_499, primals_500, primals_504, primals_508, primals_512, primals_515, primals_516, primals_519, primals_522, primals_525, primals_526, primals_530, primals_534, primals_538, primals_541, primals_542, primals_545, primals_548, primals_551, primals_552, primals_556, primals_560, primals_564, primals_567, primals_568, primals_571, primals_574, primals_577, primals_578, primals_582, primals_586, primals_590, primals_593, primals_594, primals_597, primals_600, primals_603, primals_604, primals_608, primals_612, primals_616, primals_619, primals_620, primals_623, primals_626, primals_629, primals_630, mm_default, view_11, mm, mm_2, mm_4, add_5, view_30, view_31, constant_pad_nd, getitem, getitem_1, getitem_2, getitem_3, view_35, mm_7, add_8, rsqrt_1, view_41, mm_10, add_10, mm_13, add_11, view_53, mm_16, add_13, rsqrt_2, view_59, mm_18, mm_20, mm_22, add_18, view_78, view_79, getitem_4, getitem_5, getitem_6, getitem_7, view_83, mm_25, add_21, rsqrt_3, view_89, mm_28, add_23, mm_31, add_24, view_101, mm_34, add_26, rsqrt_4, view_107, mm_36, mm_38, mm_40, add_31, view_126, view_127, getitem_8, getitem_9, getitem_10, getitem_11, view_131, mm_43, add_34, rsqrt_5, view_137, mm_46, add_36, mm_49, add_37, view_149, mm_52, add_39, rsqrt_6, view_155, mm_54, mm_56, mm_58, add_44, view_174, view_175, getitem_12, getitem_13, getitem_14, getitem_15, view_179, mm_61, add_47, rsqrt_7, view_185, mm_64, add_49, mm_67, add_50, view_197, mm_70, add_52, rsqrt_8, view_203, mm_72, mm_74, mm_76, add_57, view_222, view_223, getitem_16, getitem_17, getitem_18, getitem_19, view_227, mm_79, add_60, rsqrt_9, view_233, mm_82, add_62, mm_85, add_63, view_245, mm_88, add_65, rsqrt_10, view_251, mm_90, mm_92, mm_94, add_70, view_270, view_271, getitem_20, getitem_21, getitem_22, getitem_23, view_275, mm_97, add_73, rsqrt_11, view_281, mm_100, add_75, mm_103, add_76, view_293, mm_106, add_78, rsqrt_12, view_299, mm_108, mm_110, mm_112, add_83, view_318, view_319, getitem_24, getitem_25, getitem_26, getitem_27, view_323, mm_115, add_86, rsqrt_13, view_329, mm_118, add_88, mm_121, add_89, view_341, mm_124, add_91, rsqrt_14, view_347, mm_126, mm_128, mm_130, add_96, view_366, view_367, getitem_28, getitem_29, getitem_30, getitem_31, view_371, mm_133, add_99, rsqrt_15, view_377, mm_136, add_101, mm_139, add_102, view_389, mm_142, add_104, rsqrt_16, view_395, mm_144, mm_146, mm_148, add_109, view_414, view_415, getitem_32, getitem_33, getitem_34, getitem_35, view_419, mm_151, add_112, rsqrt_17, view_425, mm_154, add_114, mm_157, add_115, view_437, mm_160, add_117, rsqrt_18, view_443, mm_162, mm_164, mm_166, add_122, view_462, view_463, getitem_36, getitem_37, getitem_38, getitem_39, view_467, mm_169, add_125, rsqrt_19, view_473, mm_172, add_127, mm_175, add_128, view_485, mm_178, add_130, rsqrt_20, view_491, mm_180, mm_182, mm_184, add_135, view_510, view_511, getitem_40, getitem_41, getitem_42, getitem_43, view_515, mm_187, add_138, rsqrt_21, view_521, mm_190, add_140, mm_193, add_141, view_533, mm_196, add_143, rsqrt_22, view_539, mm_198, mm_200, mm_202, add_148, view_558, view_559, getitem_44, getitem_45, getitem_46, getitem_47, view_563, mm_205, add_151, rsqrt_23, view_569, mm_208, add_153, mm_211, add_154, view_581, mm_214, add_156, rsqrt_24, view_587, mm_216, mm_218, mm_220, add_161, view_606, view_607, getitem_48, getitem_49, getitem_50, getitem_51, view_611, mm_223, add_164, rsqrt_25, view_617, mm_226, add_166, mm_229, add_167, view_629, mm_232, add_169, rsqrt_26, view_635, mm_234, mm_236, mm_238, add_174, view_654, view_655, getitem_52, getitem_53, getitem_54, getitem_55, view_659, mm_241, add_177, rsqrt_27, view_665, mm_244, add_179, mm_247, add_180, view_677, mm_250, add_182, rsqrt_28, view_683, mm_252, mm_254, mm_256, add_187, view_702, view_703, getitem_56, getitem_57, getitem_58, getitem_59, view_707, mm_259, add_190, rsqrt_29, view_713, mm_262, add_192, mm_265, add_193, view_725, mm_268, add_195, rsqrt_30, view_731, mm_270, mm_272, mm_274, add_200, view_750, view_751, getitem_60, getitem_61, getitem_62, getitem_63, view_755, mm_277, add_203, rsqrt_31, view_761, mm_280, add_205, mm_283, add_206, view_773, mm_286, add_208, rsqrt_32, view_779, mm_288, mm_290, mm_292, add_213, view_798, view_799, getitem_64, getitem_65, getitem_66, getitem_67, view_803, mm_295, add_216, rsqrt_33, view_809, mm_298, add_218, mm_301, add_219, view_821, mm_304, add_221, rsqrt_34, view_827, mm_306, mm_308, mm_310, add_226, view_846, view_847, getitem_68, getitem_69, getitem_70, getitem_71, view_851, mm_313, add_229, rsqrt_35, view_857, mm_316, add_231, mm_319, add_232, view_869, mm_322, add_234, rsqrt_36, view_875, mm_324, mm_326, mm_328, add_239, view_894, view_895, getitem_72, getitem_73, getitem_74, getitem_75, view_899, mm_331, add_242, rsqrt_37, view_905, mm_334, add_244, mm_337, add_245, view_917, mm_340, add_247, rsqrt_38, view_923, mm_342, mm_344, mm_346, add_252, view_942, view_943, getitem_76, getitem_77, getitem_78, getitem_79, view_947, mm_349, add_255, rsqrt_39, view_953, mm_352, add_257, mm_355, add_258, view_965, mm_358, add_260, rsqrt_40, view_971, mm_360, mm_362, mm_364, add_265, view_990, view_991, getitem_80, getitem_81, getitem_82, getitem_83, view_995, mm_367, add_268, rsqrt_41, view_1001, mm_370, add_270, mm_373, add_271, view_1013, mm_376, add_273, rsqrt_42, view_1019, mm_378, mm_380, mm_382, add_278, view_1038, view_1039, getitem_84, getitem_85, getitem_86, getitem_87, view_1043, mm_385, add_281, rsqrt_43, view_1049, mm_388, add_283, mm_391, add_284, view_1061, mm_394, add_286, rsqrt_44, view_1067, mm_396, mm_398, mm_400, add_291, view_1086, view_1087, getitem_88, getitem_89, getitem_90, getitem_91, view_1091, mm_403, add_294, rsqrt_45, view_1097, mm_406, add_296, mm_409, add_297, view_1109, mm_412, add_299, rsqrt_46, view_1115, mm_414, mm_416, mm_418, add_304, view_1134, view_1135, getitem_92, getitem_93, getitem_94, getitem_95, view_1139, mm_421, add_307, rsqrt_47, view_1145, mm_424, add_309, mm_427, add_310, view_1157, mm_430, add_312, rsqrt_48, view_1161, permute_608, permute_612, permute_617, permute_621, permute_626, permute_630, permute_635, permute_639, permute_646, permute_650, permute_656, permute_660, permute_666, permute_670, permute_675, permute_679, permute_684, permute_688, permute_693, permute_697, permute_702, permute_706, permute_713, permute_717, permute_723, permute_727, permute_733, permute_737, permute_742, permute_746, permute_751, permute_755, permute_760, permute_764, permute_769, permute_773, permute_780, permute_784, permute_790, permute_794, permute_800, permute_804, permute_809, permute_813, permute_818, permute_822, permute_827, permute_831, permute_836, permute_840, permute_847, permute_851, permute_857, permute_861, permute_867, permute_871, permute_876, permute_880, permute_885, permute_889, permute_894, permute_898, permute_903, permute_907, permute_914, permute_918, permute_924, permute_928, permute_934, permute_938, permute_943, permute_947, permute_952, permute_956, permute_961, permute_965, permute_970, permute_974, permute_981, permute_985, permute_991, permute_995, permute_1001, permute_1005, permute_1010, permute_1014, permute_1019, permute_1023, permute_1028, permute_1032, permute_1037, permute_1041, permute_1048, permute_1052, permute_1058, permute_1062, permute_1068, permute_1072, permute_1077, permute_1081, permute_1086, permute_1090, permute_1095, permute_1099, permute_1104, permute_1108, permute_1115, permute_1119, permute_1125, permute_1129, permute_1135, permute_1139, permute_1144, permute_1148, permute_1153, permute_1157, permute_1162, permute_1166, permute_1171, permute_1175, permute_1182, permute_1186, permute_1192, permute_1196, permute_1202, permute_1206, permute_1211, permute_1215, permute_1220, permute_1224, permute_1229, permute_1233, permute_1238, permute_1242, permute_1249, permute_1253, permute_1259, permute_1263, permute_1269, permute_1273, permute_1278, permute_1282, permute_1287, permute_1291, permute_1296, permute_1300, permute_1305, permute_1309, permute_1316, permute_1320, permute_1326, permute_1330, permute_1336, permute_1340, permute_1345, permute_1349, permute_1354, permute_1358, permute_1363, permute_1367, permute_1372, permute_1376, permute_1383, permute_1387, permute_1393, permute_1397, permute_1403, permute_1407, permute_1412, permute_1416, permute_1421, permute_1425, permute_1430, permute_1434, permute_1439, permute_1443, permute_1450, permute_1454, permute_1460, permute_1464, permute_1470, permute_1474, permute_1479, permute_1483, permute_1488, permute_1492, permute_1497, permute_1501, permute_1506, permute_1510, permute_1517, permute_1521, permute_1527, permute_1531, permute_1537, permute_1541, permute_1546, permute_1550, permute_1555, permute_1559, permute_1564, permute_1568, permute_1573, permute_1577, permute_1584, permute_1588, permute_1594, permute_1598, permute_1604, permute_1608, permute_1613, permute_1617, permute_1622, permute_1626, permute_1631, permute_1635, permute_1640, permute_1644, permute_1651, permute_1655, permute_1661, permute_1665, permute_1671, permute_1675, permute_1680, permute_1684, permute_1689, permute_1693, permute_1698, permute_1702, permute_1707, permute_1711, permute_1718, permute_1722, permute_1728, permute_1732, permute_1738, permute_1742, permute_1747, permute_1751, permute_1756, permute_1760, permute_1765, permute_1769, permute_1774, permute_1778, permute_1785, permute_1789, permute_1795, permute_1799, permute_1805, permute_1809, permute_1814, permute_1818, permute_1823, permute_1827, permute_1832, permute_1836, permute_1841, permute_1845, permute_1852, permute_1856, permute_1862, permute_1866, permute_1872, permute_1876, permute_1881, permute_1885, permute_1890, permute_1894, permute_1899, permute_1903, permute_1908, permute_1912, permute_1919, permute_1923, permute_1929, permute_1933, permute_1939, permute_1943, permute_1948, permute_1952, permute_1957, permute_1961, permute_1966, permute_1970, permute_1975, permute_1979, permute_1986, permute_1990, permute_1996, permute_2000, permute_2006, permute_2010, permute_2015, permute_2019, permute_2024, permute_2028, permute_2033, permute_2037, permute_2042, permute_2046, permute_2053, permute_2057, permute_2063, permute_2067, permute_2073, permute_2077, permute_2082, permute_2086, permute_2091, permute_2095, permute_2100, permute_2104, permute_2109, permute_2113, permute_2120, permute_2124, permute_2130, permute_2134, permute_2140, permute_2144, permute_2149, permute_2153, permute_2158, permute_2162, permute_2167, permute_2171, permute_2176, permute_2180, permute_2187, permute_2195, permute_2203, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
