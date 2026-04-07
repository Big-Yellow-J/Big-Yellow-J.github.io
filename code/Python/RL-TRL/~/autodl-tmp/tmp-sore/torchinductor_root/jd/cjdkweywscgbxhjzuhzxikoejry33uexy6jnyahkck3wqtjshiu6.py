r"""
Compile-time auto-tuning block: 

import torch
from torch._dynamo.testing import rand_strided
from torch._dynamo.utils import preserve_rng_state
from torch._inductor.select_algorithm import AlgorithmSelectorCache
from torch._inductor.async_compile import AsyncCompile

async_compile = AsyncCompile()
generate_example_value = AlgorithmSelectorCache.generate_example_value
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
get_raw_stream = torch._C._cuda_getCurrentRawStream


# kernel path: /tmp/torchinductor_root/2t/c2tfs3cqg3egceszsphjoxdodx4gpodaqua37232mv3eevq57a6j.py
# Topologically Sorted Source Nodes: [to, add, pow_1, mean, add_1, rsqrt, mul, to_2, mul_1], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_9
#   add_1 => add_22
#   mean => mean
#   mul => mul_15
#   mul_1 => mul_20
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to => convert_element_type_2
#   to_2 => convert_element_type_4
# Graph fragment:
#   %mm : Tensor "bf16[s72, 896][896, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg4_1 : Tensor "bf16[s72, 896][896, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type_2 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %add_9 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, %arg4_1), kwargs = {})
#   %pow_1 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_9, 2), kwargs = {})
#   %mean : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_22 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_15 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, %rsqrt), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %mul_20 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %arg3_1), kwargs = {})
#   return %buf1,%mul_20
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = 896.0
    tmp11 = (tmp9 / tmp10)
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp18, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/x7/cx7p3qurmep7evi3stnxadytci2halvdqwmudx7uudmkzjev4p6q.py
# Topologically Sorted Source Nodes: [getitem, silu, getitem_1, mul_2], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem => slice_1
#   getitem_1 => slice_2
#   mul_2 => mul_32
#   silu => convert_element_type_7, convert_element_type_8, mul_27, sigmoid
# Graph fragment:
#   %mm_1 : Tensor "bf16[s72, 9728][9728, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %slice_1 : Tensor "bf16[s72, 4864][9728, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 0, 4864), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[s72, 4864][4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[s72, 4864][4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_7,), kwargs = {})
#   %mul_27 : Tensor "f32[s72, 4864][4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_7, %sigmoid), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[s72, 4864][4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_27, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s72, 4864][9728, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 4864, 9223372036854775807), kwargs = {})
#   %mul_32 : Tensor "bf16[s72, 4864][4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %slice_2), kwargs = {})
#   return %mul_32
triton_poi_fused_mul_silu_slice_1 = async_compile.triton('triton_poi_fused_mul_silu_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4864)
    x1 = xindex // 4864
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 9728*x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (4864 + x0 + 9728*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rq/crqtzp7snaqx3vbykeur3kz3ivffqn4xqjnhxxmylz7thcuqlpvh.py
# Topologically Sorted Source Nodes: [to, add, to_3, to_1, add_2, pow_2, mean_1, add_3, rsqrt_1, mul_3, to_5, mul_4, to_4], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_9
#   add_2 => add_59
#   add_3 => add_72
#   mean_1 => mean_1
#   mul_3 => mul_48
#   mul_4 => mul_53
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   to => convert_element_type_2
#   to_1 => convert_element_type_3
#   to_3 => convert_element_type_11
#   to_4 => convert_element_type_12
#   to_5 => convert_element_type_13
# Graph fragment:
#   %mm_2 : Tensor "bf16[s72, 896][896, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %mm : Tensor "bf16[s72, 896][896, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg4_1 : Tensor "bf16[s72, 896][896, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf6 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf6]
#   %arg7_1 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_2 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %add_9 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, %arg4_1), kwargs = {})
#   %convert_element_type_11 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.bfloat16), kwargs = {})
#   %add_59 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_11, %convert_element_type_3), kwargs = {})
#   %pow_2 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_59, 2), kwargs = {})
#   %mean_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_72 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_72,), kwargs = {})
#   %mul_48 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_59, %rsqrt_1), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_48, torch.bfloat16), kwargs = {})
#   %mul_53 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_13, %arg7_1), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_59, torch.bfloat16), kwargs = {})
#   return %buf6,%mul_53,%convert_element_type_12
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp1 + tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp15 = 896.0
    tmp16 = (tmp14 / tmp15)
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp9 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp9.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp23, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 896*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ys/cys2mr3duouv3qvmorbv6dyenv7rb6nso6daqnyexkobe3ubfr6p.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 2)
        x2 = xindex // 128
        x4 = xindex
        tmp0 = x0
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 32, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (896 + 64*x1 + 1152*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full([XBLOCK], 32768, tl.int32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp6 < 0
        tmp10 = tl.where(tmp9, tmp8, tmp6)
        tl.device_assert(((0 <= tl.broadcast_to(tmp10, [XBLOCK])) & (tl.broadcast_to(tmp10, [XBLOCK]) < 32768)) | ~(tmp4 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp10, [XBLOCK]) < 32768")
        tmp12 = tl.load(in_ptr2 + (64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp5 * tmp12
        tmp14 = tl.load(in_ptr0 + (928 + 64*x1 + 1152*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (32 + 64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp4, tmp17, tmp18)
        tmp20 = tmp0 >= tmp3
        tmp21 = tl.full([1], 64, tl.int64)
        tmp22 = tmp0 < tmp21
        tmp23 = tl.load(in_ptr0 + (928 + 64*x1 + 1152*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr1 + (x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.full([XBLOCK], 32768, tl.int32)
        tmp26 = tmp24 + tmp25
        tmp27 = tmp24 < 0
        tmp28 = tl.where(tmp27, tmp26, tmp24)
        tl.device_assert(((0 <= tl.broadcast_to(tmp28, [XBLOCK])) & (tl.broadcast_to(tmp28, [XBLOCK]) < 32768)) | ~(tmp20 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp28, [XBLOCK]) < 32768")
        tmp30 = tl.load(in_ptr2 + (64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tmp23 * tmp30
        tmp32 = tl.load(in_ptr0 + (896 + 64*x1 + 1152*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (32 + 64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tmp32 * tmp33
        tmp35 = tmp31 + tmp34
        tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
        tmp37 = tl.where(tmp20, tmp35, tmp36)
        tmp38 = tl.where(tmp4, tmp19, tmp37)
        tl.store(out_ptr0 + (x4), tmp38, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x5 = (xindex % 64)
        x6 = ((xindex // 64) % 14)
        x7 = xindex // 896
        x9 = xindex
        tmp39 = x5
        tmp40 = tl.full([1], 0, tl.int64)
        tmp41 = tmp39 >= tmp40
        tmp42 = tl.full([1], 32, tl.int64)
        tmp43 = tmp39 < tmp42
        tmp44 = tl.load(in_ptr0 + (64*x6 + 1152*x7 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp45 = tl.load(in_ptr1 + (x7), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.full([XBLOCK], 32768, tl.int32)
        tmp47 = tmp45 + tmp46
        tmp48 = tmp45 < 0
        tmp49 = tl.where(tmp48, tmp47, tmp45)
        tl.device_assert(((0 <= tl.broadcast_to(tmp49, [XBLOCK])) & (tl.broadcast_to(tmp49, [XBLOCK]) < 32768)) | ~(tmp43 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp49, [XBLOCK]) < 32768")
        tmp51 = tl.load(in_ptr2 + (64*tmp49 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp52 = tmp44 * tmp51
        tmp53 = tl.load(in_ptr0 + (32 + 64*x6 + 1152*x7 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp54 = tl.load(in_ptr2 + (32 + 64*tmp49 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp55 = tmp53 * tmp54
        tmp56 = tmp52 - tmp55
        tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
        tmp58 = tl.where(tmp43, tmp56, tmp57)
        tmp59 = tmp39 >= tmp42
        tmp60 = tl.full([1], 64, tl.int64)
        tmp61 = tmp39 < tmp60
        tmp62 = tl.load(in_ptr0 + (32 + 64*x6 + 1152*x7 + ((-32) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp63 = tl.load(in_ptr1 + (x7), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
        tmp64 = tl.full([XBLOCK], 32768, tl.int32)
        tmp65 = tmp63 + tmp64
        tmp66 = tmp63 < 0
        tmp67 = tl.where(tmp66, tmp65, tmp63)
        tl.device_assert(((0 <= tl.broadcast_to(tmp67, [XBLOCK])) & (tl.broadcast_to(tmp67, [XBLOCK]) < 32768)) | ~(tmp59 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp67, [XBLOCK]) < 32768")
        tmp69 = tl.load(in_ptr2 + (64*tmp67 + ((-32) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp70 = tmp62 * tmp69
        tmp71 = tl.load(in_ptr0 + (64*x6 + 1152*x7 + ((-32) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp72 = tl.load(in_ptr2 + (32 + 64*tmp67 + ((-32) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp73 = tmp71 * tmp72
        tmp74 = tmp70 + tmp73
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp59, tmp74, tmp75)
        tmp77 = tl.where(tmp43, tmp58, tmp76)
        tl.store(out_ptr1 + (x9), tmp77, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((4096, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((4096,), (1,), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((32768, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((4096, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((4096, 14, 64), (896, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 524288, 3670016,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
''', device_str='cuda')

async_compile.wait(globals())
del async_compile

import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
with torch.cuda._DeviceGuard(0):
    stream0 = get_raw_stream(0)
stream0 = get_raw_stream(0)
buf0 = generate_example_value((4096, 896), (896, 1), 'cuda:0', torch.bfloat16, 0, (4096, 896))
arg4_1 = generate_example_value((4096, 896), (896, 1), 'cuda:0', torch.bfloat16, 0, (4096, 896))
arg3_1 = generate_example_value((896,), (1,), 'cuda:0', torch.bfloat16, 0, (896,))
buf2 = generate_example_value((4096, 896), (896, 1), 'cuda:0', torch.bfloat16, 0, (4096, 896))
with torch.cuda._DeviceGuard(0):
    triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0.run(buf0, arg4_1, arg3_1, buf2, 4096, 896, stream=stream0)
del arg3_1, buf2

stream0 = get_raw_stream(0)
buf3 = generate_example_value((4096, 9728), (9728, 1), 'cuda:0', torch.bfloat16, 0, (4096, 9728))
buf4 = generate_example_value((4096, 4864), (4864, 1), 'cuda:0', torch.bfloat16, 0, (4096, 4864))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_mul_silu_slice_1.run(buf3, buf4, 19922944, stream=stream0)
del buf3, buf4

stream0 = get_raw_stream(0)
buf5 = generate_example_value((4096, 896), (896, 1), 'cuda:0', torch.bfloat16, 0, (4096, 896))
arg7_1 = generate_example_value((896,), (1,), 'cuda:0', torch.bfloat16, 0, (896,))
buf7 = generate_example_value((4096, 896), (896, 1), 'cuda:0', torch.bfloat16, 0, (4096, 896))
buf12 = generate_example_value((4096, 896), (896, 1), 'cuda:0', torch.bfloat16, 0, (4096, 896))
with torch.cuda._DeviceGuard(0):
    triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2.run(buf5, buf0, arg4_1, arg7_1, buf7, buf12, 4096, 896, stream=stream0)
del buf0, arg4_1, buf5, arg7_1, buf7, buf12

stream0 = get_raw_stream(0)
buf8 = generate_example_value((4096, 1152), (1152, 1), 'cuda:0', torch.bfloat16, 0, (4096, 1152))
arg10_1 = generate_example_value((4096,), (1,), 'cuda:0', torch.int64, 0, (4096,))
arg11_1 = generate_example_value((32768, 64), (64, 1), 'cuda:0', torch.bfloat16, 0, (32768, 64))
buf9 = generate_example_value((4096, 2, 64), (128, 64, 1), 'cuda:0', torch.bfloat16, 0, (4096, 2, 64))
buf10 = generate_example_value((4096, 14, 64), (896, 64, 1), 'cuda:0', torch.bfloat16, 0, (4096, 14, 64))
with torch.cuda._DeviceGuard(0):
    triton_poi_fused_3.run(buf8, arg10_1, arg11_1, buf9, buf10, 524288, 3670016, stream=stream0)
del buf8, arg10_1, arg11_1, buf9, buf10

"""
# AOT ID: ['1_inference']
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


# kernel path: /tmp/torchinductor_root/2t/c2tfs3cqg3egceszsphjoxdodx4gpodaqua37232mv3eevq57a6j.py
# Topologically Sorted Source Nodes: [to, add, pow_1, mean, add_1, rsqrt, mul, to_2, mul_1], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_9
#   add_1 => add_22
#   mean => mean
#   mul => mul_15
#   mul_1 => mul_20
#   pow_1 => pow_1
#   rsqrt => rsqrt
#   to => convert_element_type_2
#   to_2 => convert_element_type_4
# Graph fragment:
#   %mm : Tensor "bf16[s72, 896][896, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg4_1 : Tensor "bf16[s72, 896][896, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf1 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf1]
#   %arg3_1 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=arg3_1]
#   %convert_element_type_2 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %add_9 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, %arg4_1), kwargs = {})
#   %pow_1 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_9, 2), kwargs = {})
#   %mean : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_22 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_22,), kwargs = {})
#   %mul_15 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_9, %rsqrt), kwargs = {})
#   %convert_element_type_4 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
#   %mul_20 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_4, %arg3_1), kwargs = {})
#   return %buf1,%mul_20
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 1, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask & xmask, tmp6, 0)
    tmp9 = tl.sum(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = 896.0
    tmp11 = (tmp9 / tmp10)
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp4 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tmp18 = tmp16 * tmp17
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp18, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/x7/cx7p3qurmep7evi3stnxadytci2halvdqwmudx7uudmkzjev4p6q.py
# Topologically Sorted Source Nodes: [getitem, silu, getitem_1, mul_2], Original ATen: [aten.slice, aten.silu, aten.mul]
# Source node to ATen node mapping:
#   getitem => slice_1
#   getitem_1 => slice_2
#   mul_2 => mul_32
#   silu => convert_element_type_7, convert_element_type_8, mul_27, sigmoid
# Graph fragment:
#   %mm_1 : Tensor "bf16[s72, 9728][9728, 1]cuda:0" = PlaceHolder[target=mm_1]
#   %slice_1 : Tensor "bf16[s72, 4864][9728, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 0, 4864), kwargs = {})
#   %convert_element_type_7 : Tensor "f32[s72, 4864][4864, 1]cuda:0"[num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_1, torch.float32), kwargs = {})
#   %sigmoid : Tensor "f32[s72, 4864][4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_7,), kwargs = {})
#   %mul_27 : Tensor "f32[s72, 4864][4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_7, %sigmoid), kwargs = {})
#   %convert_element_type_8 : Tensor "bf16[s72, 4864][4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_27, torch.bfloat16), kwargs = {})
#   %slice_2 : Tensor "bf16[s72, 4864][9728, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.slice.Tensor](args = (%mm_1, 1, 4864, 9223372036854775807), kwargs = {})
#   %mul_32 : Tensor "bf16[s72, 4864][4864, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %slice_2), kwargs = {})
#   return %mul_32
triton_poi_fused_mul_silu_slice_1 = async_compile.triton('triton_poi_fused_mul_silu_slice_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_slice_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_slice_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4864)
    x1 = xindex // 4864
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 9728*x1), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (4864 + x0 + 9728*x1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/rq/crqtzp7snaqx3vbykeur3kz3ivffqn4xqjnhxxmylz7thcuqlpvh.py
# Topologically Sorted Source Nodes: [to, add, to_3, to_1, add_2, pow_2, mean_1, add_3, rsqrt_1, mul_3, to_5, mul_4, to_4], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_9
#   add_2 => add_59
#   add_3 => add_72
#   mean_1 => mean_1
#   mul_3 => mul_48
#   mul_4 => mul_53
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
#   to => convert_element_type_2
#   to_1 => convert_element_type_3
#   to_3 => convert_element_type_11
#   to_4 => convert_element_type_12
#   to_5 => convert_element_type_13
# Graph fragment:
#   %mm_2 : Tensor "bf16[s72, 896][896, 1]cuda:0" = PlaceHolder[target=mm_2]
#   %mm : Tensor "bf16[s72, 896][896, 1]cuda:0" = PlaceHolder[target=mm]
#   %arg4_1 : Tensor "bf16[s72, 896][896, 1]cuda:0" = PlaceHolder[target=arg4_1]
#   %buf6 : Tensor "f32[s72, 1][1, s72]cuda:0" = PlaceHolder[target=buf6]
#   %arg7_1 : Tensor "bf16[896][1]cuda:0" = PlaceHolder[target=arg7_1]
#   %convert_element_type_2 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm, torch.float32), kwargs = {})
#   %add_9 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, %arg4_1), kwargs = {})
#   %convert_element_type_11 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_2, torch.float32), kwargs = {})
#   %convert_element_type_3 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_9, torch.bfloat16), kwargs = {})
#   %add_59 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_11, %convert_element_type_3), kwargs = {})
#   %pow_2 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_59, 2), kwargs = {})
#   %mean_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [-1], True), kwargs = {})
#   %add_72 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-06), kwargs = {})
#   %rsqrt_1 : Tensor "f32[s72, 1][1, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_72,), kwargs = {})
#   %mul_48 : Tensor "f32[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_59, %rsqrt_1), kwargs = {})
#   %convert_element_type_13 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_48, torch.bfloat16), kwargs = {})
#   %mul_53 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_13, %arg7_1), kwargs = {})
#   %convert_element_type_12 : Tensor "bf16[s72, 896][896, 1]cuda:0"[num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_59, torch.bfloat16), kwargs = {})
#   return %buf6,%mul_53,%convert_element_type_12
triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r0_': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 2, 'num_reduction': 1, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp2 = tl.load(in_ptr1 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r0_1 + 896*x0), r0_mask & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp1 + tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(r0_mask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp15 = 896.0
    tmp16 = (tmp14 / tmp15)
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp9 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp9.to(tl.float32)
    tl.store(out_ptr1 + (r0_1 + 896*x0), tmp23, r0_mask & xmask)
    tl.store(out_ptr2 + (r0_1 + 896*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/ys/cys2mr3duouv3qvmorbv6dyenv7rb6nso6daqnyexkobe3ubfr6p.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

from torch._dynamo.testing import rand_strided
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel_0': 'i32', 'xnumel_1': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'combo_grid_meta': {'num_kernels': 2, 'min_blocks': None, 'default_config': None, 'no_x_dim_0': False, 'xnumel_0': None, 'no_x_dim_1': False, 'xnumel_1': None}, 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def triton_poi_fused_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % 64)
        x1 = ((xindex // 64) % 2)
        x2 = xindex // 128
        x4 = xindex
        tmp0 = x0
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1], 32, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (896 + 64*x1 + 1152*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.full([XBLOCK], 32768, tl.int32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp6 < 0
        tmp10 = tl.where(tmp9, tmp8, tmp6)
        tl.device_assert(((0 <= tl.broadcast_to(tmp10, [XBLOCK])) & (tl.broadcast_to(tmp10, [XBLOCK]) < 32768)) | ~(tmp4 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp10, [XBLOCK]) < 32768")
        tmp12 = tl.load(in_ptr2 + (64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp5 * tmp12
        tmp14 = tl.load(in_ptr0 + (928 + 64*x1 + 1152*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (32 + 64*tmp10 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = tmp13 - tmp16
        tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
        tmp19 = tl.where(tmp4, tmp17, tmp18)
        tmp20 = tmp0 >= tmp3
        tmp21 = tl.full([1], 64, tl.int64)
        tmp22 = tmp0 < tmp21
        tmp23 = tl.load(in_ptr0 + (928 + 64*x1 + 1152*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr1 + (x2), tmp20 & xmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.full([XBLOCK], 32768, tl.int32)
        tmp26 = tmp24 + tmp25
        tmp27 = tmp24 < 0
        tmp28 = tl.where(tmp27, tmp26, tmp24)
        tl.device_assert(((0 <= tl.broadcast_to(tmp28, [XBLOCK])) & (tl.broadcast_to(tmp28, [XBLOCK]) < 32768)) | ~(tmp20 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp28, [XBLOCK]) < 32768")
        tmp30 = tl.load(in_ptr2 + (64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tmp23 * tmp30
        tmp32 = tl.load(in_ptr0 + (896 + 64*x1 + 1152*x2 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (32 + 64*tmp28 + ((-32) + x0)), tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tmp32 * tmp33
        tmp35 = tmp31 + tmp34
        tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
        tmp37 = tl.where(tmp20, tmp35, tmp36)
        tmp38 = tl.where(tmp4, tmp19, tmp37)
        tl.store(out_ptr0 + (x4), tmp38, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x5 = (xindex % 64)
        x6 = ((xindex // 64) % 14)
        x7 = xindex // 896
        x9 = xindex
        tmp39 = x5
        tmp40 = tl.full([1], 0, tl.int64)
        tmp41 = tmp39 >= tmp40
        tmp42 = tl.full([1], 32, tl.int64)
        tmp43 = tmp39 < tmp42
        tmp44 = tl.load(in_ptr0 + (64*x6 + 1152*x7 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp45 = tl.load(in_ptr1 + (x7), tmp43 & xmask, eviction_policy='evict_last', other=0.0)
        tmp46 = tl.full([XBLOCK], 32768, tl.int32)
        tmp47 = tmp45 + tmp46
        tmp48 = tmp45 < 0
        tmp49 = tl.where(tmp48, tmp47, tmp45)
        tl.device_assert(((0 <= tl.broadcast_to(tmp49, [XBLOCK])) & (tl.broadcast_to(tmp49, [XBLOCK]) < 32768)) | ~(tmp43 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp49, [XBLOCK]) < 32768")
        tmp51 = tl.load(in_ptr2 + (64*tmp49 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp52 = tmp44 * tmp51
        tmp53 = tl.load(in_ptr0 + (32 + 64*x6 + 1152*x7 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp54 = tl.load(in_ptr2 + (32 + 64*tmp49 + (x5)), tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp55 = tmp53 * tmp54
        tmp56 = tmp52 - tmp55
        tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
        tmp58 = tl.where(tmp43, tmp56, tmp57)
        tmp59 = tmp39 >= tmp42
        tmp60 = tl.full([1], 64, tl.int64)
        tmp61 = tmp39 < tmp60
        tmp62 = tl.load(in_ptr0 + (32 + 64*x6 + 1152*x7 + ((-32) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp63 = tl.load(in_ptr1 + (x7), tmp59 & xmask, eviction_policy='evict_last', other=0.0)
        tmp64 = tl.full([XBLOCK], 32768, tl.int32)
        tmp65 = tmp63 + tmp64
        tmp66 = tmp63 < 0
        tmp67 = tl.where(tmp66, tmp65, tmp63)
        tl.device_assert(((0 <= tl.broadcast_to(tmp67, [XBLOCK])) & (tl.broadcast_to(tmp67, [XBLOCK]) < 32768)) | ~(tmp59 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp67, [XBLOCK]) < 32768")
        tmp69 = tl.load(in_ptr2 + (64*tmp67 + ((-32) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp70 = tmp62 * tmp69
        tmp71 = tl.load(in_ptr0 + (64*x6 + 1152*x7 + ((-32) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp72 = tl.load(in_ptr2 + (32 + 64*tmp67 + ((-32) + x5)), tmp59 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp73 = tmp71 * tmp72
        tmp74 = tmp70 + tmp73
        tmp75 = tl.full(tmp74.shape, 0.0, tmp74.dtype)
        tmp76 = tl.where(tmp59, tmp74, tmp75)
        tmp77 = tl.where(tmp43, tmp58, tmp76)
        tl.store(out_ptr1 + (x9), tmp77, xmask)
    else:
        pass


def get_args():
    arg_0 = rand_strided((4096, 1152), (1152, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_1 = rand_strided((4096,), (1,), device='cuda:0', dtype=torch.int64)
    arg_2 = rand_strided((32768, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_3 = rand_strided((4096, 2, 64), (128, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg_4 = rand_strided((4096, 14, 64), (896, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    return arg_0, arg_1, arg_2, arg_3, arg_4, 524288, 3670016,


def call(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(*args, stream=stream0)


def benchmark_all_configs(args):
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        return triton_poi_fused_3.benchmark_all_configs(*args)


if __name__ == '__main__':
    from torch._inductor.runtime.benchmarking import benchmarker

    args = get_args()
    ms = benchmarker.benchmark(call, fn_args=(args,), device=cuda,rep=40)
    num_gb = 0
    gb_per_s = num_gb / (ms / 1e3)
    print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")
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
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1 = args
        args.clear()
        s72 = arg1_1
        assert_size_stride(arg0_1, (s72, 14, 64), (896, 64, 1))
        assert_size_stride(arg2_1, (896, 896), (896, 1))
        assert_size_stride(arg3_1, (896, ), (1, ))
        assert_size_stride(arg4_1, (s72, 896), (896, 1))
        assert_size_stride(arg5_1, (9728, 896), (896, 1))
        assert_size_stride(arg6_1, (896, 4864), (4864, 1))
        assert_size_stride(arg7_1, (896, ), (1, ))
        assert_size_stride(arg8_1, (1152, 896), (896, 1))
        assert_size_stride(arg9_1, (1152, ), (1, ))
        assert_size_stride(arg10_1, (s72, ), (1, ))
        assert_size_stride(arg11_1, (32768, 64), (64, 1))
        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            buf0 = empty_strided_cuda((s72, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [view, linear], Original ATen: [aten.view, aten.t, aten.mm]
            extern_kernels.mm(reinterpret_tensor(arg0_1, (s72, 896), (896, 1), 0), reinterpret_tensor(arg2_1, (896, 896), (1, 896), 0), out=buf0)
            del arg0_1
            del arg2_1
            buf2 = empty_strided_cuda((s72, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to, add, pow_1, mean, add_1, rsqrt, mul, to_2, mul_1], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_0.run(buf0, arg4_1, arg3_1, buf2, s72, 896, stream=stream0)
            del arg3_1
            buf3 = empty_strided_cuda((s72, 9728), (9728, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to, add, pow_1, mean, add_1, rsqrt, mul, to_2, mul_1, linear_1], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(buf2, reinterpret_tensor(arg5_1, (896, 9728), (1, 896), 0), out=buf3)
            del arg5_1
            del buf2
            buf4 = empty_strided_cuda((s72, 4864), (4864, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, silu, getitem_1, mul_2], Original ATen: [aten.slice, aten.silu, aten.mul]
            triton_poi_fused_mul_silu_slice_1_xnumel = 4864*s72
            stream0 = get_raw_stream(0)
            triton_poi_fused_mul_silu_slice_1.run(buf3, buf4, triton_poi_fused_mul_silu_slice_1_xnumel, stream=stream0)
            del buf3
            buf5 = empty_strided_cuda((s72, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [getitem, silu, getitem_1, mul_2, linear_2], Original ATen: [aten.slice, aten.silu, aten.mul, aten.t, aten.mm]
            extern_kernels.mm(buf4, reinterpret_tensor(arg6_1, (4864, 896), (1, 4864), 0), out=buf5)
            del arg6_1
            del buf4
            buf7 = empty_strided_cuda((s72, 896), (896, 1), torch.bfloat16)
            buf12 = empty_strided_cuda((s72, 896), (896, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to, add, to_3, to_1, add_2, pow_2, mean_1, add_3, rsqrt_1, mul_3, to_5, mul_4, to_4], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul]
            stream0 = get_raw_stream(0)
            triton_per_fused__to_copy_add_mean_mul_pow_rsqrt_2.run(buf5, buf0, arg4_1, arg7_1, buf7, buf12, s72, 896, stream=stream0)
            del arg4_1
            del arg7_1
            del buf0
            buf8 = empty_strided_cuda((s72, 1152), (1152, 1), torch.bfloat16)
            # Topologically Sorted Source Nodes: [to, add, to_3, to_1, add_2, pow_2, mean_1, add_3, rsqrt_1, mul_3, to_5, mul_4, linear_3], Original ATen: [aten._to_copy, aten.add, aten.pow, aten.mean, aten.rsqrt, aten.mul, aten.t, aten.addmm]
            extern_kernels.addmm(arg9_1, buf7, reinterpret_tensor(arg8_1, (896, 1152), (1, 896), 0), alpha=1, beta=1, out=buf8)
            del arg8_1
            del arg9_1
            buf9 = empty_strided_cuda((s72, 2, 64), (128, 64, 1), torch.bfloat16)
            buf10 = reinterpret_tensor(buf7, (s72, 14, 64), (896, 64, 1), 0); del buf7  # reuse
            # Topologically Sorted Source Nodes: [split, index_select, chunk, view_1, chunk_1, view_2, chunk_2, unsqueeze_2, mul_9, unsqueeze_3, mul_10, sub_1, mul_11, mul_12, add_5, cat_2, unsqueeze, mul_5, unsqueeze_1, mul_6, sub, mul_7, mul_8, add_4, cat], Original ATen: [aten.split_with_sizes, aten.index_select, aten.split, aten.view, aten.unsqueeze, aten.mul, aten.sub, aten.add, aten.cat]
            triton_poi_fused_3_xnumel_0 = 128*s72
            triton_poi_fused_3_xnumel_1 = 896*s72
            stream0 = get_raw_stream(0)
            triton_poi_fused_3.run(buf8, arg10_1, arg11_1, buf9, buf10, triton_poi_fused_3_xnumel_0, triton_poi_fused_3_xnumel_1, stream=stream0)
            del arg10_1
            del arg11_1
            buf11 = buf5; del buf5  # reuse
        return (buf9, reinterpret_tensor(buf8, (s72, 2, 64), (1152, 64, 1), 1024), buf10, reinterpret_tensor(buf11, (s72, 14, 64), (896, 64, 1), 0), buf12, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4096, 14, 64), (896, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = 4096
    arg2_1 = rand_strided((896, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((4096, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((9728, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((896, 4864), (4864, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((1152, 896), (896, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.int64)
    arg11_1 = rand_strided((32768, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
