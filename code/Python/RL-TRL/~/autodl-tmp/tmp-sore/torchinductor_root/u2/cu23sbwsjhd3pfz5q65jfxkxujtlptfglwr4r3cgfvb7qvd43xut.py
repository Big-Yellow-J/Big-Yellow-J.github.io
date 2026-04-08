
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 128}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=80, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}], 'enable_fp_fusion': True},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_neg_sin_slice_transpose_unsqueeze_view_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 1, 'num_reduction': 0, 'backend_hash': '530A4EEDF49C5716AE98C01E4E74B49F3D6F7913EC4A9A06FFBD6D251F721D80', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'are_deterministic_algorithms_enabled': False, 'tiling_scores': {'y': 155904, 'x': 2495232}},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_view_add_addmm_bmm_cat_cos_mul_neg_sin_slice_transpose_unsqueeze_view_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1218
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = (xindex % 64)
    x2 = xindex // 64
    y0 = yindex
    x3 = xindex
    tmp28 = tl.load(in_ptr3 + (y0 + 1248*((x1 % 32))), xmask & ymask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp35 = tl.load(in_ptr1 + (x3 + 128*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp37 = tl.load(in_ptr2 + (x3 + 128*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(32 + 64*x2 + (x1), [YBLOCK, XBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (32 + 64*x2 + 128*y0 + (x1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (32 + 64*x2 + 128*y0 + (x1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = -tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1, 1], 64, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr0 + (tl.broadcast_to(64*x2 + ((-32) + x1), [YBLOCK, XBLOCK])), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr1 + (64*x2 + 128*y0 + ((-32) + x1)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.load(in_ptr2 + (64*x2 + 128*y0 + ((-32) + x1)), tmp15 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tmp36 = tmp34 + tmp35
    tmp38 = tmp37 * tmp30
    tmp39 = tmp36 + tmp38
    tmp40 = tl_math.cos(tmp28)
    tmp41 = tmp40 * tmp30
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp39 * tmp42
    tmp44 = tmp43 + tmp33
    tl.store(out_ptr1 + (x3 + 128*y0), tmp44, xmask & ymask)
