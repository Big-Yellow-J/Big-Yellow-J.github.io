
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i64', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 11, 'num_store': 2, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_mul_neg_sin_slice_slice_backward_squeeze_sum_transpose_unsqueeze_view_10(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 128
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    x2 = (xindex % 64)
    tmp27 = tl.load(in_ptr0 + (x1 + 128*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (y0 + ks0*((((x1 % 64)) % 32))), xmask & ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr1 + (y0 + ks0*((x2 % 32))), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (x1 % 64)
    tmp1 = tl.full([1, 1], 32, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.load(in_ptr0 + ((-32) + x1 + 128*y0), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (y0 + ks0*((((x1 % 64)) % 32))), tmp2 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tmp17 = tl.load(in_ptr1 + (y0 + ks0*((((x1 % 64)) % 32))), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tmp39 = tl.load(in_ptr1 + (y0 + ks0*((x2 % 32))), tmp37 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
    tmp51 = tl.load(in_ptr1 + (y0 + ks0*((x2 % 32))), tmp49 & xmask & ymask, eviction_policy='evict_last', other=0.0)
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
