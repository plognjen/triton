from dataclasses import dataclass, field
import itertools
import math
from typing import List

from triton.language.core import _unwrap_if_constexpr, _unwrap_shape, constexpr_type
from triton.runtime.jit import constexpr_function
from triton._C.libtriton import gluon_ir


class DistributedLayout:
    """
    Base class for distributed memory layouts in Gluon IR.
    """

    @property
    def type(self):
        return constexpr_type(self)

    @property
    def rank(self):
        raise NotImplementedError("DistributedLayout subclasses must define rank")

    def format_tensor_view(self, shape: list[int]) -> str:
        return gluon_ir.get_layout_view(self, [_unwrap_if_constexpr(s) for s in shape], False)

    def format_hardware_view(self, shape: list[int]) -> str:
        return gluon_ir.get_layout_view(self, [_unwrap_if_constexpr(s) for s in shape], True)


@dataclass(frozen=True)
class AutoLayout(DistributedLayout):

    def _to_ir(self, builder):
        return builder.get_auto_layout()

    def mangle(self):
        return "AL"

    @property
    def rank(self):
        raise ValueError("AutoLayout has no rank")


@dataclass(frozen=True)
class CoalescedLayout(DistributedLayout):

    def _to_ir(self, builder):
        return builder.get_coalesced_layout()

    def mangle(self):
        return "CL"

    @property
    def rank(self):
        raise ValueError("CoalescedLayout has no rank")


@dataclass(frozen=True)
class BlockedLayout(DistributedLayout):
    """
    Represents a blocked layout, partitioning a tensor across threads, warps, and CTAs.

    Args:
        size_per_thread (List[int]): Number of elements per thread per dimension.
        threads_per_warp (List[int]): Number of threads per warp per dimension.
        warps_per_cta (List[int]): Number of warps per CTA per dimension.
        order (List[int]): The ordering of dimensions for partitioning.
        cga_layout (Optional[List[List[int]]]): Bases describing how CTAs tile each dimension.
    """
    size_per_thread: List[int]
    threads_per_warp: List[int]
    warps_per_cta: List[int]
    order: List[int]
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("size_per_thread", _unwrap_if_constexpr(self.size_per_thread))
        super().__setattr__("threads_per_warp", _unwrap_if_constexpr(self.threads_per_warp))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("order", _unwrap_if_constexpr(self.order))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))

        rank = len(self.size_per_thread)
        assert len(self.threads_per_warp) == rank
        assert len(self.warps_per_cta) == rank
        assert len(self.order) == rank

    def _to_ir(self, builder):
        return builder.get_blocked_layout(
            self.size_per_thread,
            self.threads_per_warp,
            self.warps_per_cta,
            self.order,
            self.cga_layout,
        )

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        size_per_thread = stringify(self.size_per_thread)
        threads_per_warp = stringify(self.threads_per_warp)
        warps_per_cta = stringify(self.warps_per_cta)
        order = stringify(self.order)
        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return f"B{size_per_thread}_{threads_per_warp}_{warps_per_cta}_{order}_{cga_layout}B"

    def __hash__(self):
        return hash((tuple(self.size_per_thread), tuple(self.threads_per_warp), tuple(self.warps_per_cta),
                     tuple(self.order), tuple(tuple(vec) for vec in self.cga_layout)))

    @property
    def rank(self):
        return len(self.order)


@dataclass(frozen=True)
class SliceLayout(DistributedLayout):
    """
    Represents a layout corresponding to slicing a distributed tensor along one dimension.

    Args:
        dim (int): The dimension index to slice.
        parent (DistributedLayout): The parent layout before slicing.
    """
    dim: int
    parent: DistributedLayout

    def __post_init__(self):
        super().__setattr__("dim", _unwrap_if_constexpr(self.dim))
        super().__setattr__("parent", _unwrap_if_constexpr(self.parent))

    def _to_ir(self, builder):
        return builder.get_slice_layout(
            self.dim,
            self.parent._to_ir(builder),
        )

    def mangle(self) -> str:
        return f"SL{self.dim}_{self.parent.mangle()}SL"

    def __hash__(self):
        return hash((self.dim, self.parent))

    @property
    def rank(self):
        return self.parent.rank - 1

    @property
    def cga_layout(self):
        parent_cga_layout = self.parent.cga_layout
        if not parent_cga_layout:
            return []

        rank = self.parent.rank
        assert 0 <= self.dim < rank
        return [basis[:self.dim] + basis[self.dim + 1:] for basis in parent_cga_layout]


@dataclass(frozen=True)
class DistributedLinearLayout(DistributedLayout):
    """
    Represents a linear distributed layout with explicit bases at register, lane, warp, and block levels.
    See: https://arxiv.org/abs/2505.23819 for reference.

    Args:
        reg_bases (List[List[int]]): Bases for register-level distribution.
        lane_bases (List[List[int]]): Bases for lane-level distribution.
        warp_bases (List[List[int]]): Bases for warp-level distribution.
        block_bases (List[List[int]]): Bases for block-level distribution.
        shape (List[int]): The tensor global shape.
    """
    reg_bases: List[List[int]]
    lane_bases: List[List[int]]
    warp_bases: List[List[int]]
    block_bases: List[List[int]]
    shape: List[int]

    def __post_init__(self):
        super().__setattr__("reg_bases", _unwrap_shape(self.reg_bases))
        super().__setattr__("lane_bases", _unwrap_shape(self.lane_bases))
        super().__setattr__("warp_bases", _unwrap_shape(self.warp_bases))
        super().__setattr__("block_bases", _unwrap_shape(self.block_bases))
        super().__setattr__("shape", _unwrap_shape(self.shape))

        rank = len(self.shape)

        for basis in self.reg_bases:
            assert len(basis) == rank
        for basis in self.lane_bases:
            assert len(basis) == rank
        for basis in self.warp_bases:
            assert len(basis) == rank
        for basis in self.block_bases:
            assert len(basis) == rank

    def _to_ir(self, builder):
        return builder.get_distributed_linear_layout(self.reg_bases, self.lane_bases, self.warp_bases, self.block_bases,
                                                     self.shape)

    def mangle(self):
        return f"DLL{self.reg_bases}_{self.lane_bases}_{self.warp_bases}_{self.block_bases}_{self.shape}DLL"

    def __hash__(self):
        return hash((
            tuple(map(tuple, self.reg_bases)),
            tuple(map(tuple, self.lane_bases)),
            tuple(map(tuple, self.warp_bases)),
            tuple(map(tuple, self.block_bases)),
            tuple(self.shape),
        ))

    @property
    def rank(self):
        return len(self.shape)


@dataclass(frozen=True)
class DotOperandLayout(DistributedLayout):
    """
    Represents a layout for a dot operand.

    Args:
        operand_index (int): 0 for LHS and 1 for RHS of the dot operation.
        parent (DistributedLayout): The parent layout, representing the MMA.
        k_width (int): Number of elements per 32-bits.
    """
    operand_index: int
    parent: DistributedLayout
    k_width: int

    def __post_init__(self):
        super().__setattr__("operand_index", _unwrap_if_constexpr(self.operand_index))
        super().__setattr__("parent", _unwrap_if_constexpr(self.parent))
        super().__setattr__("k_width", _unwrap_if_constexpr(self.k_width))

    def _to_ir(self, builder):
        return builder.get_dot_operand_layout(self.operand_index, self.parent._to_ir(builder), self.k_width)

    def mangle(self) -> str:
        return f"DO{self.operand_index}_{self.parent.mangle()}_{self.k_width}DO"

    def __hash__(self):
        return hash((self.operand_index, self.parent, self.k_width))

    @property
    def rank(self):
        return self.parent.rank

    @property
    def cga_layout(self):
        parent_cga_layout = _unwrap_if_constexpr(getattr(self.parent, "cga_layout", [])) or []
        if not parent_cga_layout:
            return []

        rank = self.parent.rank
        assert all(len(basis) == rank for basis in parent_cga_layout)

        k_dim = rank - 1 if self.operand_index == 0 else rank - 2
        assert 0 <= k_dim < rank

        derived = []
        for basis in parent_cga_layout:
            new_basis = list(basis)
            new_basis[k_dim] = 0
            derived.append(new_basis)
        return derived


@dataclass(frozen=True, eq=True)
class NVMMADistributedLayout(DistributedLayout):
    """
    Represents a layout for NVIDIA MMA (tensor core) operations.

    Args:
        version (List[int]): Version identifier for the MMA instruction.
        warps_per_cta (List[int]): Number of warps per CTA.
        instr_shape (List[int]): Instruction shape for MMA.
        cga_layout (Optional[List[List[int]]]): Bases describing CTA tiling.
    """
    version: List[int]
    warps_per_cta: List[int]
    instr_shape: List[int]
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("version", _unwrap_if_constexpr(self.version))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("instr_shape", _unwrap_if_constexpr(self.instr_shape))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))

    def _to_ir(self, builder):
        return builder.get_mma_layout(
            self.version,
            self.warps_per_cta,
            self.cga_layout,
            self.instr_shape,
        )

    def mangle(self) -> str:
        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return f"MMA_{self.version}_{self.warps_per_cta}_{self.instr_shape}_{cga_layout}_MMA"

    def __hash__(self):
        return hash((tuple(self.version), tuple(self.warps_per_cta), tuple(self.instr_shape),
                     tuple(tuple(vec) for vec in self.cga_layout)))

    @property
    def rank(self):
        return len(self.warps_per_cta)


class SharedLayout:
    """
    Base class for shared memory layouts in Gluon IR.
    """

    @property
    def type(self):
        return constexpr_type(self)

    def format_tensor_view(self, shape: list[int]) -> str:
        return gluon_ir.get_layout_view(self, [_unwrap_if_constexpr(s) for s in shape], False)

    def format_hardware_view(self, shape: list[int]) -> str:
        return gluon_ir.get_layout_view(self, [_unwrap_if_constexpr(s) for s in shape], True)


@constexpr_function
def _get_shape_per_cta(shape, cga_layout):
    if not cga_layout:
        return shape
    shape_per_cta = list(shape)
    rank = len(cga_layout[0])
    cga_shape = [0] * rank
    for basis in cga_layout:
        assert len(basis) == rank
        for i in range(rank):
            cga_shape[i] = max(cga_shape[i], basis[i])
    # The shape is the largest stride * 2, or 1 if the stride was always zero
    for i in range(rank):
        if cga_shape[i] == 0:
            cga_shape[i] = 1
        else:
            cga_shape[i] *= 2
    for dim in range(rank):
        assert shape_per_cta[dim] % cga_shape[dim] == 0, f"Shape {shape} is not divisible by CGA layout {cga_layout}"
        shape_per_cta[dim] //= cga_shape[dim]
    return shape_per_cta


@dataclass(frozen=True)
class NVMMASharedLayout(SharedLayout):
    """
    Represents a layout for shared memory suitable for NVIDIA MMA operations.

    Args:
        swizzle_byte_width (int): Width in bytes for swizzling.
        element_bitwidth (int): Bitwidth of element type.
        rank (int): Rank of the tensor.
        transposed (bool): Whether the layout is transposed.
        fp4_padded (bool): Whether FP4 padding is used.
        cga_layout (Optional[List[List[int]]]): Bases describing CTA tiling.
    """
    swizzle_byte_width: int
    element_bitwidth: int
    rank: int = 2
    transposed: bool = False
    fp4_padded: bool = False
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("swizzle_byte_width", _unwrap_if_constexpr(self.swizzle_byte_width))
        super().__setattr__("element_bitwidth", _unwrap_if_constexpr(self.element_bitwidth))
        super().__setattr__("transposed", _unwrap_if_constexpr(self.transposed))
        super().__setattr__("fp4_padded", _unwrap_if_constexpr(self.fp4_padded))

        # TODO: Make rank optional and check that (rank or cga_layout)
        cga_layout = self.cga_layout or []
        if cga_layout:
            assert len(cga_layout[0]) == self.rank

        super().__setattr__("rank", _unwrap_if_constexpr(self.rank))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(cga_layout))

        assert self.element_bitwidth in [8, 16, 32, 64]
        assert self.swizzle_byte_width in [0, 32, 64, 128]

        if self.fp4_padded:
            assert self.swizzle_byte_width == 128, "fp4_padded only supports 128 byte swizzling"
            assert self.element_bitwidth == 8, "fp4_padded is only supported for element_bitwidth=8"

    def _to_ir(self, builder):
        return builder.get_nvmma_shared_layout(
            self.swizzle_byte_width,
            self.element_bitwidth,
            self.transposed,
            self.fp4_padded,
            self.cga_layout,
            self.rank,
        )

    @staticmethod
    @constexpr_function
    def get_default_for(block_shape, dtype, transposed=False, fp4_padded=False, cga_layout=None):
        """Returns an NVMMASharedLayout with default swizzling for a given shape.

        This picks the largest swizzle pattern compatible with the shape, which
        allows emitting the fewest TMA or MMA messages.
        """
        packing_factor = 2 if fp4_padded else 1
        shape_per_cta = block_shape if cga_layout is None else _get_shape_per_cta(block_shape, cga_layout)
        rank = len(block_shape)
        if transposed:
            shape_per_cta = shape_per_cta[1:] + shape_per_cta[:1]
        contig_dim_size = shape_per_cta[-1] * packing_factor
        contig_dim_bytes = contig_dim_size * dtype.primitive_bitwidth // 8
        if contig_dim_bytes >= 128 and contig_dim_bytes % 128 == 0:
            swizzle_byte_width = 128
        elif contig_dim_bytes >= 64 and contig_dim_bytes % 64 == 0:
            swizzle_byte_width = 64
        elif contig_dim_bytes >= 32 and contig_dim_bytes % 32 == 0:
            swizzle_byte_width = 32
        else:
            swizzle_byte_width = 0

        flatten_outer_dim = 1
        for size in shape_per_cta[:-1]:
            flatten_outer_dim *= size
        if len(block_shape) < 2 or flatten_outer_dim < 8:
            swizzle_byte_width = 0

        return NVMMASharedLayout(
            swizzle_byte_width=swizzle_byte_width,
            element_bitwidth=dtype.primitive_bitwidth,
            rank=rank,
            transposed=transposed,
            fp4_padded=fp4_padded,
            cga_layout=cga_layout,
        )

    def mangle(self) -> str:
        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return f"NVMMA_{self.swizzle_byte_width}_{self.element_bitwidth}_{self.transposed}_{self.fp4_padded}_{cga_layout}_NVMMA"

    def __hash__(self):
        return hash((self.swizzle_byte_width, self.element_bitwidth, self.rank, self.transposed, self.fp4_padded,
                     tuple(tuple(vec) for vec in self.cga_layout) if self.cga_layout else None))


@dataclass(frozen=True, eq=True)
class SwizzledSharedLayout(SharedLayout):
    """
    Represents a generic swizzled shared memory layout.

    Args:
        vec (int): Vector width for swizzling.
        per_phase (int): Elements per swizzle phase.
        max_phase (int): Maximum number of swizzle phases.
        order (List[int]): Dimension ordering for swizzling.
        cga_layout (Optional[List[List[int]]]): Bases describing CTA tiling.
    """
    vec: int
    per_phase: int
    max_phase: int
    order: List[int]
    cga_layout: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        super().__setattr__("vec", _unwrap_if_constexpr(self.vec))
        super().__setattr__("per_phase", _unwrap_if_constexpr(self.per_phase))
        super().__setattr__("max_phase", _unwrap_if_constexpr(self.max_phase))
        super().__setattr__("order", _unwrap_if_constexpr(self.order))
        super().__setattr__("cga_layout", _unwrap_if_constexpr(self.cga_layout))

    def _to_ir(self, builder):
        return builder.get_swizzled_shared_layout(
            self.vec,
            self.per_phase,
            self.max_phase,
            self.order,
            self.cga_layout,
        )

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        cga_layout = "_".join("~".join(map(str, vec)) for vec in self.cga_layout) if self.cga_layout else ""
        return f"SSS_{self.vec}_{self.per_phase}_{self.max_phase}_{stringify(self.order)}_{cga_layout}_SSS"

    def __hash__(self):
        return hash(
            (self.vec, self.per_phase, self.max_phase, tuple(self.order), tuple(tuple(vec) for vec in self.cga_layout)))


@dataclass(frozen=True, eq=True)
class PaddedSharedLayout(SharedLayout):
    """
    Represents a layout for the access to shared memory. Compared to SwizzledSharedLayout,
    it combined padding and element reordering via linear transformation (e.g. row permutation)
    to avoid shared memory bank conflicts. After every interval tensor elements, the
    corresponding number of padding elements are inserted. If a position corresponds to
    multiple intervals, the padding amounts are summed.

    In the following example of a tensor,
    `eM` represents original elements in the and `pN` represents padded element.

    Before padding, the shared memory looks like:
    [e0, e1,
     e2, e3,
     e4, e5,
     e6, e7,
     ...]

    After padding with interval-padding list [[2, 1], [4, 2]] with an identity remapping,
    the shared memory will be
    [e0, e1, p0,
     e2, e3, p1, p2, p3,
     e4, e5, p4,
     e6, e7, p5, p6, p7,
     ...]

    Furthermore this encoding allows for a linear remapping from the 1-D shared
    memory offset to logical n-D tensor elements. The remapping is given in the form
    of linear bases mapping from offset to [dim0, dim1...dimN-1].
    See LinearLayout.h for more details how linear layouts are applied to remap
    elements.
    Some concrete examples using `xN` and `yN` to mean the logical n-D tensor elements
    and `pN` to mean padding:

    After padding for shape = [8] with interval-padding list [[2, 2]], offset_bases = [[2], [1]] and cga_layout = []:
    [x0, x2, p0 p1, x1, x3]

    After padding for shape = [8, 4] with interval_padding_pairs = [[8, 1]], offset_bases = [[0, 1], [0, 2], /*gap, stride by 2 rows*/[2, 0], [4, 0], [1, 0]]] and cga_layout = []:
    [
        x0y0, x0y1, x0y2, x0y3,
        x2y0, x2y1, x2y2, x2y3,
        p0,
        x4y0, x4y1, x4y2, x4y3,
        x6y0, x6y1, x6y2, x6y3,
        p1,
        x1y0, x1y1, x1y2, x1y3,
        x3y0, x3y1, x3y2, x3y3,
        p2,
        x5y0, x5y1, x5y2, x5y3,
        x7y0, x7y1, x7y2, x7y3,
    ]

    Args:
        interval_padding_pairs (List[int]): List of [interval, padding] pair and both interval and padding must be powers of 2.
        offset_bases (List[int]): Bases for shared memory offsets
        cga_layout (List[List[int]]): Bases for block-level shared memory offsets.
        shape (List[int]): n-D logical shared memory shape
    """
    interval_padding_pairs: List[List[int]]
    offset_bases: List[List[int]]
    cga_layout: List[List[int]]
    shape: List[int]

    def __post_init__(self):
        super().__setattr__("interval_padding_pairs", _unwrap_shape(self.interval_padding_pairs))
        super().__setattr__("offset_bases", _unwrap_shape(self.offset_bases))
        super().__setattr__("cga_layout", _unwrap_shape(self.cga_layout))
        super().__setattr__("shape", _unwrap_shape(self.shape))

        rank = len(self.shape)

        for basis in self.offset_bases:
            assert len(basis) == rank
        for basis in self.cga_layout:
            assert len(basis) == rank

        self.verify()

    def _to_ir(self, builder):
        intervals, paddings = zip(*self.interval_padding_pairs)
        return builder.get_padded_shared_layout(intervals, paddings, self.offset_bases, self.cga_layout, self.shape)

    def mangle(self) -> str:
        return f"PaddedShared_{self.interval_padding_pairs}_{self.offset_bases}_{self.cga_layout}_{self.shape}_PaddedShared"

    def verify(self):
        pairs = self.interval_padding_pairs
        assert len(pairs) > 0, "PaddedSharedLayout interval_padding_pairs must have at least one interval-padding pair"
        assert all(len(pair) == 2 for pair in pairs)
        intervals, paddings = zip(*pairs)

        unique_intervals = list(set(intervals))
        assert len(unique_intervals) == len(intervals)

        is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
        assert all(is_power_of_2(n) for n in intervals), "PaddedSharedLayout interval values must all be power of two"
        assert all(is_power_of_2(n) for n in paddings), "PaddedSharedLayout padding values must all be power of two"

        rank = len(self.shape)
        assert rank > 0, "PaddedSharedLayout order must not be empty"

    @staticmethod
    @constexpr_function
    def with_identity_for(interval_padding_pairs, shape, order, cga_layout=[]):
        """Returns a PaddedSharedLayout with the given interval and padding pairs and an identity mapping as the linear component for the given shape and order.
        """
        assert len(shape) == len(order)
        is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
        assert all(is_power_of_2(n) for n in shape)

        rank = len(shape)
        shape_per_cta = _get_shape_per_cta(shape, cga_layout) if cga_layout else shape
        # Create a idendity mapping based on shape_per_cta + order
        offset_bases = []
        for dim in order:
            for basis in range(int(math.log2(shape_per_cta[dim]))):
                offset_bases.append([1 << basis if i == dim else 0 for i in range(rank)])

        return PaddedSharedLayout(interval_padding_pairs, offset_bases, cga_layout, shape)

    def __hash__(self):
        return hash((tuple(map(tuple, self.interval_padding_pairs)), tuple(map(tuple, self.offset_bases)),
                     tuple(map(tuple, self.cga_layout)), tuple(self.shape)))


@dataclass(frozen=True)
class SharedLinearLayout(SharedLayout):
    """Represents a shared memory layout defined via an explicit LinearLayout."""

    offset_bases: List[List[int]]
    block_bases: List[List[int]] = field(default_factory=list)
    alignment: int = 16

    def __post_init__(self):
        super().__setattr__("offset_bases", _unwrap_shape(self.offset_bases))
        super().__setattr__("block_bases", _unwrap_shape(self.block_bases))
        super().__setattr__("alignment", _unwrap_if_constexpr(self.alignment))

        assert len(self.offset_bases) != 0, "SharedLinearLayout offset_bases must not be empty"
        rank = len(self.offset_bases[0])
        assert rank > 0, "SharedLinearLayout offset_bases must not be empty"
        for basis in self.offset_bases:
            assert len(basis) == rank
        for basis in self.block_bases:
            assert len(basis) == rank
        assert self.alignment > 0 and (self.alignment & (self.alignment - 1)) == 0, \
            "SharedLinearLayout alignment must be a positive power of two"

    def _to_ir(self, builder):
        return builder.get_shared_linear_layout(self.offset_bases, self.block_bases, self.alignment)

    def mangle(self) -> str:
        return f"SharedLinear_{self.offset_bases}_{self.block_bases}_{self.alignment}_SharedLinear"

    @property
    def shape(self):
        rank = len(self.offset_bases[0])
        max_stride = [1] * rank
        for b in itertools.chain(self.offset_bases, self.block_bases):
            for i, bi in enumerate(b):
                max_stride[i] = max(max_stride[i], bi)
        return [2 * s for s in max_stride]

    def __hash__(self):
        return hash((
            tuple(map(tuple, self.offset_bases)),
            tuple(map(tuple, self.block_bases)),
            self.alignment,
        ))


# Python impl of LinearEncodingAttr::basesPerDim
def bases_per_dim(bases, rank, skip_broadcast=True):
    result = [1] * rank

    if not bases:
        return result

    non_zero_idx = None

    for basis in bases:
        # Find the first non-zero index in the current basis
        idx = next((i for i, v in enumerate(basis) if v != 0), None)
        if idx is not None:
            non_zero_idx = idx
            result[idx] *= 2
        elif not skip_broadcast:
            # If no non-zero found and we're not skipping broadcasts, use the last found non-zero index
            assert non_zero_idx is not None
            result[non_zero_idx] *= 2

    return result


def warps_per_cta(layout, shape):
    if hasattr(layout, 'warp_bases'):
        return bases_per_dim(layout.warp_bases, len(shape))
    elif isinstance(layout, (SliceLayout, DotOperandLayout)):
        return warps_per_cta(layout.parent, shape)
    else:
        return layout.warps_per_cta


@constexpr_function
def _make_partitioned_operand_layout(sublayout, partition_dim, block_dim_size, warp_coverage, num_partitions, order):
    """Build a ``PartitionedSharedLayout`` for one GEMM operand.

    The operand tile (``sublayout``) is split along ``partition_dim`` into
    ``num_partitions * num_groups`` logical pieces, where
    ``num_groups = block_dim_size // warp_coverage``.  Each piece carries an
    inner ``PaddedSharedLayout`` rebuilt with an identity mapping using the
    original sublayout's ``interval_padding_pairs`` and ``cga_layout``.
    """
    from triton.experimental.gluon.language.amd.gfx1250._layouts import PartitionedSharedLayout

    is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
    num_groups = block_dim_size // warp_coverage
    assert num_groups >= 1, f"block dim ({block_dim_size}) must be >= {warp_coverage}"
    assert is_power_of_2(num_groups), \
        f"block_dim / warp_coverage = {num_groups} must be a power of 2"

    num_logical_pieces = num_partitions * num_groups
    inner_shape = list(sublayout.shape)
    assert inner_shape[partition_dim] % num_logical_pieces == 0
    inner_shape[partition_dim] //= num_logical_pieces

    # TODO: when the partitioned dimension is the contiguous (fastest-varying)
    # dimension, the interval_padding_pairs may need adjustment to preserve
    # bank-conflict avoidance properties for the smaller piece shape.
    inner = PaddedSharedLayout.with_identity_for(sublayout.interval_padding_pairs, inner_shape, order,
                                                 sublayout.cga_layout)
    return PartitionedSharedLayout(num_partitions, num_groups, partition_dim, inner)


@dataclass(frozen=True)
class GemmLayouts:
    shared_layout_a: object
    shared_layout_b: object
    wmma_layout: object


@constexpr_function
def make_partitioned_gemm_layouts(block_m, block_n, sublayout_a, sublayout_b, num_warps, instr_shape,
                                  a_transposed=False, b_transposed=False):
    """Create partitioned shared memory layouts and WMMA layout for a GFX1250 GEMM
       in order to avoid LDS partition conflicts.

    Args:
        block_m: M dimension tile size.  Must be at least
            ``WARP_TILES_M * instr_shape[0]`` (the per-CTA M extent covered by
            one warp+register cycle), and a power-of-2 multiple thereof.
        block_n: N dimension tile size.  Must be at least
            ``WARP_TILES_N * instr_shape[1]``, and a power-of-2 multiple
            thereof.
        sublayout_a: ``PaddedSharedLayout`` for operand A.  Shape is
            ``[block_m, block_k]`` when not transposed (K contiguous) and
            ``[block_k, block_m]`` when transposed (M contiguous).
        sublayout_b: ``PaddedSharedLayout`` for operand B.  Shape is
            ``[block_k, block_n]`` when not transposed (N contiguous) and
            ``[block_n, block_k]`` when transposed (K contiguous).
        num_warps: Number of warps per CTA.  Currently must be 4 or 8.
        instr_shape: WMMA instruction shape as ``[M, N, K]``.
        a_transposed: Whether A is transposed in shared memory, i.e. M is
            the contiguous axis instead of K.
        b_transposed: Whether B is transposed in shared memory, i.e. K is
            the contiguous axis instead of N.

    Returns:
        A ``GemmLayouts`` with fields ``shared_layout_a``, ``shared_layout_b``,
        and ``wmma_layout``.
    """
    from triton.experimental.gluon.language.amd._layouts import AMDWMMALayout

    INSTR_SHAPE = list(instr_shape)

    assert num_warps in (4, 8), f"Only 4 or 8 warps are currently supported, got {num_warps}"

    NUM_PARTITIONS = 2

    # The caller passes each sublayout with its contiguous axis at dim 1 of the
    # tile (i.e. the tile's shape is already expressed in memory order):
    #   A non-transposed: [block_m, block_k] — K contiguous at dim 1
    #   A transposed:     [block_k, block_m] — M contiguous at dim 1
    #   B non-transposed: [block_k, block_n] — N contiguous at dim 1
    #   B transposed:     [block_n, block_k] — K contiguous at dim 1
    # So the linear component is always built with order [1, 0] regardless of
    # transposition. TDM additionally requires order [rank-1, ..., 0].
    order = [1, 0]

    # WMMA CTA layout: Below, M runs vertically (rows) and N
    # horizontally (cols); each cell is one INSTR_SHAPE-sized instruction tile,
    # labelled with the warp that computes it.  ``warp_bases`` map warp-id bits
    # to (M, N) tile offsets, ``reg_bases`` map register-id
    # (more specifically, instruction repetition registers) bits the same way.
    #
    # 4-warp case (warp_bases = [[2, 1], [1, 0]], reg_bases = [[2, 0]]):
    #
    #   M=0:  w0 w1   <- second tile computed by w1 (reg=1)
    #   M=1:  w2 w3
    #   M=2:  w0 w1   <- first tile computed by w1 (reg=0)
    #   M=3:  w2 w3
    #
    # Such layout allows w0 and w1 as well as w2 and w3 to read different A/B
    # operand data blocks in a single instruction, which is necessary precondition
    # for avoiding LDS partition conflicts.
    if num_warps == 4:
        warp_bases = [[2, 1], [1, 0]]
        reg_bases = [[2, 0]]
    else:  # num_warps == 8
        # Same idea as the 4-warp case, but the third warp bit replaces the
        # register bit. This means there's no need to define repetition registers,
        # single instruction CTA layout is enough to describe the layout we need.
        # Each warp now owns a single instruction tile and the extra warp dimension
        # is folded into the M-axis of the warp grid.
        warp_bases = [[2, 1], [1, 0], [2, 0]]
        reg_bases = []

    # The tile extent along each dimension is 2^m, where m is the largest
    # basis component in that dimension across both warp and register bases.
    def _tile_extent(dim):
        m = max((b[dim] for b in warp_bases + reg_bases), default=0)
        return 1 << m

    WARP_TILES_M = _tile_extent(0)
    WARP_TILES_N = _tile_extent(1)

    wmma_layout = AMDWMMALayout(3, True, warp_bases, reg_bases, INSTR_SHAPE)

    # Per-CTA extent that one warp+register cycle covers in each dimension.
    warp_coverage_m = WARP_TILES_M * INSTR_SHAPE[0]
    warp_coverage_n = WARP_TILES_N * INSTR_SHAPE[1]

    # Partition A along its M axis and B along its N axis.  These dims live
    # at different positions in the tile depending on transposition (the
    # contiguous axis is always at dim 1, so M / N moves to dim 0 or dim 1).
    a_partition_dim = 1 if a_transposed else 0
    b_partition_dim = 0 if b_transposed else 1

    shared_layout_a = _make_partitioned_operand_layout(sublayout_a, partition_dim=a_partition_dim,
                                                       block_dim_size=block_m, warp_coverage=warp_coverage_m,
                                                       num_partitions=NUM_PARTITIONS, order=order)
    shared_layout_b = _make_partitioned_operand_layout(sublayout_b, partition_dim=b_partition_dim,
                                                       block_dim_size=block_n, warp_coverage=warp_coverage_n,
                                                       num_partitions=NUM_PARTITIONS, order=order)

    return GemmLayouts(shared_layout_a, shared_layout_b, wmma_layout)
