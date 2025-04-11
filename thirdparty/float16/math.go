package float16

import (
	"math"
	"math/bits"
)

// Mul returns the IEEE 754 binary64 product of a and b.
func (a Float16) Mul(b Float16) Float16 {
	if a.IsNaN() || b.IsNaN() {
		// anything * NaN = NaN
		// NaN * anything = NaN
		return uvnan
	}

	signA := a & signMask16
	expA := int((a>>shift16)&mask16) - bias16
	signB := b & signMask16
	expB := int((b>>shift16)&mask16) - bias16

	if expA == mask16-bias16 {
		// NaN check is done above; b is ±inf
		if expB == -bias16 && b&fracMask16 == 0 {
			// b is zero, the result is NaN
			return Float16(uvnan)
		} else {
			// otherwise the result is infinity
			return a ^ signB
		}
	}

	if expB == mask16-bias16 {
		// NaN check is done above; b is ±inf
		if expA == -bias16 && a&fracMask16 == 0 {
			// a is zero, the result is NaN
			return Float16(uvnan)
		} else {
			// NaN check is done above
			// so a is not zero nor NaN. the result is infinity
			return b ^ signA
		}
	}

	sign := signA ^ signB

	var fracA uint32
	if expA == -bias16 {
		// a is subnormal
		fracA = uint32(a & fracMask16)
		l := bits.Len32(fracA)
		fracA <<= shift16 - l + 1
		expA = -(bias16 + shift16) + l
	} else {
		// a is normal
		fracA = uint32(a&fracMask16) | (1 << shift16)
	}

	var fracB uint32
	if expB == -bias16 {
		// b is subnormal
		fracB = uint32(b & fracMask16)
		l := bits.Len32(fracB)
		fracB <<= shift16 - l + 1
		expB = -(bias16 + shift16) + l
	} else {
		// b is normal
		fracB = uint32(b&fracMask16) | (1 << shift16)
	}

	exp := expA + expB
	frac := fracA * fracB
	shift := bits.Len32(frac) - (shift16 + 1)
	exp += shift - shift16

	if exp < -(bias16 + shift16) {
		// underflow
		return sign
	} else if exp <= -bias16 {
		// the result is subnormal
		shift := shift16 - (expA + expB + bias16) + 1
		frac += (1<<(shift-1) - 1) + ((frac >> shift) & 1) // round to nearest even
		frac >>= shift
		return sign | Float16(frac)
	}

	exp = expA + expB + bias16
	frac += (1<<(shift-1) - 1) + ((frac >> shift) & 1) // round to nearest even
	shift = bits.Len32(frac) - (shift16 + 1)
	exp += shift - shift16
	if exp >= mask16 {
		// overflow
		return sign | (mask16 << shift16)
	}
	frac >>= shift
	frac &= fracMask16
	return sign | Float16(exp<<shift16) | Float16(frac)
}

// Quo returns the IEEE 754 binary64 quotient of a and b.
func (a Float16) Quo(b Float16) Float16 {
	if a.IsNaN() || b.IsNaN() {
		// anything / NaN = NaN
		// NaN / anything = NaN
		return uvnan
	}

	signA := a & signMask16
	expA := int((a>>shift16)&mask16) - bias16
	signB := b & signMask16
	expB := int((b>>shift16)&mask16) - bias16
	sign := signA ^ signB

	if b&^signMask16 == 0x0000 {
		// division by zero
		if a&^signMask16 == 0 {
			// ±0 / ±0 = NaN
			// ±0 / ∓0 = NaN
			return Float16(uvnan)
		}
		// +x / ±0 = ±Inf
		// -x / ±0 = ∓Inf
		return sign | (mask16 << shift16)
	}
	if expA == mask16-bias16 {
		// NaN check is done above; a is ±Inf
		if expB == mask16-bias16 {
			// +Inf / ±Inf = NaN
			// -Inf / ±Inf = NaN
			// ±Inf / NaN = NaN
			return Float16(uvnan)
		} else {
			// otherwise the result is infinity
			return a ^ signB
		}
	}

	if expB == mask16-bias16 {
		// NaN check is done above; b is ±Inf
		// +x / ±Inf = ±0
		// -x / ±Inf = ∓0
		return sign
	}

	var fracA uint32
	if expA == -bias16 {
		// a is subnormal
		fracA = uint32(a & fracMask16)
		l := bits.Len32(fracA)
		fracA <<= shift16 - l + 1
		expA = -(bias16 + shift16) + l
	} else {
		// a is normal
		fracA = uint32(a&fracMask16) | (1 << shift16)
	}
	if fracA == 0 {
		// a is zero
		return sign
	}

	var fracB uint32
	if expB == -bias16 {
		// b is subnormal
		fracB = uint32(b & fracMask16)
		l := bits.Len32(fracB)
		fracB <<= shift16 - l + 1
		expB = -(bias16 + shift16) + l
	} else {
		// b is normal
		fracB = uint32(b&fracMask16) | (1 << shift16)
	}

	exp := expA - expB + bias16
	if fracA < fracB {
		exp--
		fracA <<= 1
	}
	if exp >= mask16 {
		// overflow
		return sign | (mask16 << shift16)
	}
	shift := shift16 + 3 // 1 for the implicit bit, 1 for the rounding bit, 1 for the guard bit
	fracA = (fracA << shift)
	frac := uint16(fracA / fracB)
	mod := uint16(fracA % fracB)
	frac |= squash(mod)
	if exp <= 0 {
		// the result is subnormal
		shift := -exp + 3 + 1
		frac += (1<<(shift-1) - 1) + ((frac >> shift) & 1) // round to nearest even
		frac >>= shift
		return sign | Float16(frac)
	}

	frac += 0b11 + ((frac >> 3) & 1) // round to nearest even
	frac >>= 3
	return sign | Float16(exp<<shift16) | Float16(frac&fracMask16)
}

func squash(x uint16) uint16 {
	x |= x >> 8
	x |= x >> 4
	x |= x >> 2
	x |= x >> 1
	return x & 1
}

// Add returns the IEEE 754 binary64 sum of a and b.
func (a Float16) Add(b Float16) Float16 {
	if a.IsNaN() || b.IsNaN() {
		// anything + NaN = NaN
		// NaN + anything = NaN
		return uvnan
	}
	if a^signMask16 == 0 { // a is ±0
		return b
	}

	if (a>>shift16)&mask16 == mask16 {
		// NaN is already handled; a is ±inf
		if a&fracMask16 == 0 {
			// a is infinity
			if b == a^signMask16 {
				// ±inf + ∓inf = NaN
				return NaN()
			}
			return a // ±inf + anything = ±inf
		}
	}

	fixA := a.fix24()
	fixB := b.fix24()
	return (fixA + fixB).Float16()
}

// Sub returns the IEEE 754 binary64 difference of a and b.
func (a Float16) Sub(b Float16) Float16 {
	return a.Add(b ^ signMask16)
}

// fix24 is a fixed-point number with 24 bits of precision.
type fix24 int64

const fix24inf = fix24(1 << 61)

func (f Float16) fix24() fix24 {
	var ret fix24
	exp := uint32(f>>shift16) & mask16
	frac := uint32(f & fracMask16)
	if exp == 0 {
		// subnormal number
		ret = fix24(frac)
	} else if exp == mask16 {
		// infinity or NaN
		ret = fix24inf
	} else {
		// normal number
		ret = fix24(frac|(1<<shift16)) << (exp - 1)
	}
	sign := uint32(f & signMask16)
	if sign != 0 {
		ret = -ret
	}
	return ret
}

func (f fix24) Float16() Float16 {
	if f == 0 {
		return 0
	}

	var sign uint16
	if f < 0 {
		sign = signMask16
		f = -f
	}
	l := bits.Len64(uint64(f))
	if l <= shift16 {
		// subnormal number
		return Float16(sign | uint16(f))
	}
	shift := l - shift16 - 1
	if shift > 0 {
		f += (1<<(shift-1) - 1) + ((f >> shift) & 1) // round to nearest even
		l = bits.Len64(uint64(f))
	}

	exp := uint16(l) - shift16
	if exp >= mask16 {
		// overflow
		return Float16(sign | (mask16 << shift16))
	}
	frac := uint16(f>>(exp-1)) & fracMask16
	return Float16(sign | (exp << shift16) | frac)
}

// Compare compares x and y and returns:
//
//	-1 if x <  y
//	 0 if x == y (incl. -0 == 0, -Inf == -Inf, and +Inf == +Inf)
//	+1 if x >  y
//
// a NaN is considered less than any non-NaN, and two NaNs are equal.
func (a Float16) Compare(b Float16) int {
	aNaN := a.IsNaN()
	bNaN := b.IsNaN()
	if aNaN && bNaN {
		return 0
	}
	if aNaN {
		return -1
	}
	if bNaN {
		return 1
	}

	ia := a.comparable()
	ib := b.comparable()
	if ia < ib {
		return -1
	}
	if ia > ib {
		return 1
	}
	return 0
}

// comparable converts a to a comparable form.
func (a Float16) comparable() int16 {
	i := int16(a) ^ ((int16(a) >> 15) & 0x7fff)
	return i + int16(a>>15) // normalize -0 to 0
}

// Eq returns a == b.
// NaNs are not equal to anything, including NaN.
func (a Float16) Eq(b Float16) bool {
	if a.IsNaN() || b.IsNaN() {
		return false
	}

	// a == b if a and b have the same bit pattern,
	// or if they are both ±0.
	return a == b || (a|b)&^signMask16 == 0
}

// Ne returns a != b.
// NaNs are not equal to anything, including NaN.
func (a Float16) Ne(b Float16) bool {
	return !a.Eq(b)
}

// Lt returns a < b.
//
// Special cases are:
//
//	Lt(NaN, x) == false
//	Lt(x, NaN) == false
func (a Float16) Lt(b Float16) bool {
	if a.IsNaN() || b.IsNaN() {
		return false
	}

	return a.comparable() < b.comparable()
}

// Gt returns a > b.
//
// Special cases are:
//
//	Gt(x, NaN) == false
//	Gt(NaN, x) == false
func (a Float16) Gt(b Float16) bool {
	return b.Lt(a)
}

// Le returns a <= b.
//
// Special cases are:
//
//	Le(x, NaN) == false
//	Le(NaN, x) == false
func (a Float16) Le(b Float16) bool {
	if a.IsNaN() || b.IsNaN() {
		return false
	}

	return a.comparable() <= b.comparable()
}

// Ge returns a >= b.
//
// Special cases are:
//
//	Ge(x, NaN) == false
//	Ge(NaN, x) == false
func (a Float16) Ge(b Float16) bool {
	return b.Le(a)
}

// FMA returns x * y + z, computed with only one rounding.
// (That is, FMA returns the fused multiply-add of x, y, and z.)
func FMA(x, y, z Float16) Float16 {
	if x.IsNaN() || y.IsNaN() || z.IsNaN() {
		return uvnan
	}

	fx := x.Float64()
	fy := y.Float64()
	fz := z.Float64()
	f := math.FMA(fx, fy, fz)
	return FromFloat64(f)
}
