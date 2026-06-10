from triton.experimental.gluon import aggregate
from triton.experimental.gluon.language import _core as ttgl, _standard as stdlib
from triton.experimental.gluon._runtime import constexpr_function, jit

__all__ = [
    "pack2",
    "unpack2",
    "pack",
    "unpack",
    "pack_e4m3x2",
    "fma",
    "Float2Tensor",
]


@ttgl.builtin
def _is_fpsan(_semantic=None):
    mode = getattr(_semantic.builder.options, "instrumentation_mode", "")
    return ttgl.constexpr("fpsan" in mode)


@jit
def _pack_f32x2(x0, x1):
    return ttgl.inline_asm_elementwise(
        """
        mov.b64 $0, { $1, $2 };
        """,
        "=l,r,r",
        [x0, x1],
        dtype=ttgl.int64,
        is_pure=True,
        pack=1,
    )


@jit
def _unpack_f32x2(x):
    return ttgl.inline_asm_elementwise(
        """
        mov.b64 { $0, $1 }, $2;
        """,
        "=r,=r,l",
        [x],
        dtype=[ttgl.float32, ttgl.float32],
        is_pure=True,
        pack=1,
    )


@jit
def _add_f32x2(a, b):
    if _is_fpsan():
        a0, a1 = _unpack_f32x2(a)
        b0, b1 = _unpack_f32x2(b)
        return _pack_f32x2(a0 + b0, a1 + b1)
    return ttgl.inline_asm_elementwise(
        """
        add.f32x2 $0, $1, $2;
        """,
        "=l,l,l",
        [a, b],
        dtype=ttgl.int64,
        is_pure=True,
        pack=1,
    )


@jit
def _sub_f32x2(a, b):
    if _is_fpsan():
        a0, a1 = _unpack_f32x2(a)
        b0, b1 = _unpack_f32x2(b)
        return _pack_f32x2(a0 - b0, a1 - b1)
    return ttgl.inline_asm_elementwise(
        """
        sub.f32x2 $0, $1, $2;
        """,
        "=l,l,l",
        [a, b],
        dtype=ttgl.int64,
        is_pure=True,
        pack=1,
    )


@jit
def _mul_f32x2(a, b):
    if _is_fpsan():
        a0, a1 = _unpack_f32x2(a)
        b0, b1 = _unpack_f32x2(b)
        return _pack_f32x2(a0 * b0, a1 * b1)
    return ttgl.inline_asm_elementwise(
        """
        mul.f32x2 $0, $1, $2;
        """,
        "=l,l,l",
        [a, b],
        dtype=ttgl.int64,
        is_pure=True,
        pack=1,
    )


@jit
def _fma_f32x2(a, b, c):
    if _is_fpsan():
        a0, a1 = _unpack_f32x2(a)
        b0, b1 = _unpack_f32x2(b)
        c0, c1 = _unpack_f32x2(c)
        return _pack_f32x2(a0 * b0 + c0, a1 * b1 + c1)
    return ttgl.inline_asm_elementwise(
        """
        fma.rn.f32x2 $0, $1, $2, $3;
        """,
        "=l,l,l,l",
        [a, b, c],
        dtype=ttgl.int64,
        is_pure=True,
        pack=1,
    )


@aggregate
class Float2Tensor:
    value: ttgl.tensor

    @constexpr_function
    def __init__(self, value: ttgl.tensor):
        self.value = value

    @jit
    def __add__(self, rhs):
        ttgl.static_assert(isinstance(rhs, Float2Tensor), "rhs must be a Float2Tensor")
        return Float2Tensor(_add_f32x2(self.value, rhs.value))

    @jit
    def __sub__(self, rhs):
        ttgl.static_assert(isinstance(rhs, Float2Tensor), "rhs must be a Float2Tensor")
        return Float2Tensor(_sub_f32x2(self.value, rhs.value))

    @jit
    def __mul__(self, rhs):
        ttgl.static_assert(isinstance(rhs, Float2Tensor), "rhs must be a Float2Tensor")
        return Float2Tensor(_mul_f32x2(self.value, rhs.value))

    @jit
    def sum(self, axis: ttgl.constexpr):
        return Float2Tensor(ttgl.reduce(self.value, axis=axis, combine_fn=_add_f32x2))


@jit
def pack2(x0, x1):
    return Float2Tensor(_pack_f32x2(x0, x1))


@jit
def unpack2(x):
    return _unpack_f32x2(x.value)


@jit
def pack_e4m3x2(x, saturate_inf: ttgl.constexpr = False):
    if saturate_inf:
        x0, x1 = unpack2(x)
        x0_clipped = ttgl.clamp(x0, -448.0, 448.0)
        x1_clipped = ttgl.clamp(x1, -448.0, 448.0)
        if _is_fpsan():
            x0, x1 = x0_clipped, x1_clipped
        else:
            x0 = ttgl.where(x0 == x0, x0_clipped, 448.0)
            x1 = ttgl.where(x1 == x1, x1_clipped, 448.0)
        x = pack2(x0, x1)
    if _is_fpsan():
        x0, x1 = unpack2(x)
        x0 = x0.to(ttgl.float8e4nv).to(ttgl.uint8, bitcast=True).to(ttgl.uint16)
        x1 = x1.to(ttgl.float8e4nv).to(ttgl.uint8, bitcast=True).to(ttgl.uint16)
        return (x0 | (x1 << 8)).to(ttgl.int16, bitcast=True)
    return ttgl.inline_asm_elementwise(
        """
        {
            .reg .f32 lane<2>;
            mov.b64 {lane0, lane1}, $1;
            cvt.rn.satfinite.e4m3x2.f32 $0, lane1, lane0;
        }
        """,
        "=h,l",
        [x.value],
        dtype=ttgl.int16,
        is_pure=True,
        pack=1,
    )


@constexpr_function
def _get_split_shape(shape, axis):
    shape = [d for d in shape]
    assert shape[axis] >= 2, f"not enough elements to pack along axis {axis}"
    shape[axis] //= 2
    shape.insert(axis + 1, 2)
    permute = list(range(len(shape)))
    permute[axis + 1], permute[len(permute) - 1] = permute[len(permute) - 1], permute[axis + 1]
    return ttgl.tuple(shape), ttgl.tuple(permute)


@constexpr_function
def _get_join_shape(shape, axis):
    shape = [d for d in shape]
    shape[axis] *= 2
    permute = list(range(len(shape)))
    permute.insert(axis + 1, len(permute))
    return ttgl.tuple(shape), ttgl.tuple(permute)


@jit
def pack(x, axis):
    sp: ttgl.constexpr = _get_split_shape(x.shape, axis)
    x0, x1 = x.reshape(*sp[0]).permute(*sp[1]).split()
    return pack2(x0, x1)


@jit
def unpack(x, axis):
    shape: ttgl.constexpr = x.value.shape
    sp: ttgl.constexpr = _get_join_shape(shape, axis)
    x0, x1 = unpack2(x)
    return ttgl.join(x0, x1).permute(*sp[1]).reshape(*sp[0])


@jit
def full_like(x, fill_value):
    ttgl.static_assert(fill_value.dtype == ttgl.float32, "fill_value must be a float32")
    fill = stdlib.full_like(x.value, fill_value, dtype=ttgl.float32)
    return pack2(fill, fill)


@jit
def fma(a, b, c):
    return Float2Tensor(_fma_f32x2(a.value, b.value, c.value))
