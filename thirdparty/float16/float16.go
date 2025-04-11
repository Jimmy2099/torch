package float16

import (
	"math"
	"math/bits"
)

const (
	uvnan      = 0x7e00     // "not-a-number"
	uvinf      = 0x7c00     // infinity
	uvneginf   = 0xfc00     // negative infinity
	uvone      = 0x3c00     // one
	mask16     = 0x1f       // mask for exponent
	shift16    = 16 - 5 - 1 // shift for exponent
	bias16     = 15         // bias for exponent
	signMask16 = 1 << 15    // mask for sign bit
	fracMask16 = 1<<shift16 - 1
)

const (
	mask32     = 0xff       // mask for exponent
	shift32    = 32 - 8 - 1 // shift for exponent
	bias32     = 127        // bias for exponent
	signMask32 = 1 << 31    // mask for sign bit
	fracMask32 = 1<<shift32 - 1
)

const (
	mask64     = 0x7ff       // mask for exponent
	shift64    = 64 - 11 - 1 // shift for exponent
	bias64     = 1023        // bias for exponent
	signMask64 = 1 << 63     // mask for sign bit
	fracMask64 = 1<<shift64 - 1
)

// Float16 represents a 16-bit floating point number.
type Float16 uint16

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf(sign int) Float16 {
	if sign >= 0 {
		return Float16(uvinf)
	} else {
		return Float16(uvneginf)
	}
}

// IsInf reports whether f is an infinity, according to sign.
// If sign > 0, IsInf reports whether f is positive infinity.
// If sign < 0, IsInf reports whether f is negative infinity.
// If sign == 0, IsInf reports whether f is either infinity.
func (f Float16) IsInf(sign int) bool {
	return sign >= 0 && f == uvinf || sign <= 0 && f == uvneginf
}

// FromBits returns the floating point number corresponding
// the IEEE 754 binary representation b.
func FromBits(b uint16) Float16 {
	return Float16(b)
}

// Bits returns the IEEE 754 binary representation of f.
func (f Float16) Bits() uint16 {
	return uint16(f)
}

// FromFloat32 returns the floating point number corresponding
// to the IEEE 754 binary representation of f.
func FromFloat32(f float32) Float16 {
	b := math.Float32bits(f)
	sign := uint16((b & signMask32) >> (32 - 16))
	exp := int((b >> shift32) & mask32)

	if exp == mask32 {
		frac := b & fracMask32
		if frac == 0 {
			// infinity or negative infinity
			return Float16(sign | (mask16 << shift16))
		} else {
			// NaN
			return Float16(sign | uvnan | uint16(frac>>(shift32-shift16)&fracMask16))
		}
	}

	exp -= bias32

	if exp <= -bias16 {
		// handle subnormal number
		roundBit := -exp + shift32 - (bias16 + shift16 - 1)
		frac := (b & fracMask32) | (1 << shift32)
		halfMinusULP := uint32(1<<(roundBit-1) - 1)
		frac += halfMinusULP + ((frac >> roundBit) & 1)
		return Float16(sign | uint16(frac>>roundBit))
	}

	// handle normal number

	// round to nearest even
	const halfMinusULP = (1 << (shift32 - shift16 - 1)) - 1
	b += halfMinusULP + (b >> (shift32 - shift16) & 1)

	exp16 := uint16((b>>shift32)&mask32) - bias32 + bias16
	if exp16 >= mask16 {
		// overflow
		return Float16(sign | (mask16 << shift16))
	}
	frac16 := uint16(b>>(shift32-shift16)) & fracMask16
	return Float16(sign | (exp16 << shift16) | frac16)
}

// FromFloat64 returns the floating point number corresponding
// to the IEEE 754 binary representation of f.
func FromFloat64(f float64) Float16 {
	b := math.Float64bits(f)
	sign := uint16((b & signMask64) >> (64 - 16))
	exp := int((b >> shift64) & mask64)
	frac := b & fracMask64

	if exp == mask64 {
		if frac == 0 {
			// infinity or negative infinity
			return Float16(sign | (mask16 << shift16))
		} else {
			// NaN
			return Float16(uvnan)
		}
	}

	exp -= bias64

	if exp <= -bias16 {
		// handle subnormal number
		roundBit := -exp + shift64 - (bias16 + shift16 - 1)
		frac := (b & fracMask64) | (1 << shift64)
		halfMinusULP := uint64(1<<(roundBit-1) - 1)
		frac += halfMinusULP + ((frac >> roundBit) & 1)
		return Float16(sign | uint16(frac>>roundBit))
	}

	// handle normal number

	// round to nearest even
	const halfMinusULP = (1 << (shift64 - shift16 - 1)) - 1
	b += halfMinusULP + (b >> (shift64 - shift16) & 1)

	exp16 := uint16((b>>shift64)&mask64) - bias64 + bias16
	if exp16 >= mask16 {
		// overflow
		return Float16(sign | (mask16 << shift16))
	}
	frac16 := uint16(b>>(shift64-shift16)) & fracMask16
	return Float16(sign | (exp16 << shift16) | frac16)
}

// Float32 returns the float32 representation of f.
func (f Float16) Float32() float32 {
	sign := uint32(f&signMask16) << (32 - 16)
	exp := uint32(f>>shift16) & mask16
	frac := uint32(f & fracMask16)

	if exp == 0 {
		// subnormal number
		if frac == 0 {
			exp = 0
		} else {
			l := bits.Len32(frac)
			frac = (frac << (shift16 - l + 1)) & fracMask16
			exp = bias32 - (bias16 + shift16) + uint32(l)
		}
	} else if exp == mask16 {
		// infinity or NaN
		exp = mask32
	} else {
		// normal number
		exp += bias32 - bias16
	}
	return math.Float32frombits(sign | (exp << shift32) | (frac << (shift32 - shift16)))
}

// Float64 returns the float64 representation of f.
func (f Float16) Float64() float64 {
	sign := uint64(f&signMask16) << (64 - 16)
	exp := uint64(f>>shift16) & mask16
	frac := uint64(f & fracMask16)

	if exp == 0 {
		// subnormal number
		l := bits.Len64(frac)
		if l == 0 {
			exp = 0
		} else {
			frac = (frac << (shift64 - l + 1)) & fracMask64
			exp = bias64 - (bias16 + shift16) + uint64(l)
		}
	} else if exp == mask16 {
		// infinity or NaN
		exp = mask64
		if frac != 0 {
			frac = 1 << (shift64 - 1)
		}
	} else {
		// normal number
		exp += bias64 - bias16
		frac <<= shift64 - shift16
	}
	return math.Float64frombits(sign | (exp << shift64) | frac)
}

// NaN returns an IEEE 754 “not-a-number” value.
func NaN() Float16 {
	return Float16(uvnan)
}

// IsNaN reports whether f is an IEEE 754 “not-a-number” value.
func (f Float16) IsNaN() bool {
	return f&(mask16<<shift16) == (mask16<<shift16) && f&fracMask16 != 0
}

func (f Float16) split() (sign uint16, exp int32, frac uint16) {
	sign = uint16(f & signMask16)
	exp = int32((f>>shift16)&mask16) - bias16
	frac = uint16(f & fracMask16)

	// normalize f
	if exp == -bias16 {
		// subnormal number
		l := bits.Len16(frac)
		frac <<= shift16 - l + 1
		exp = -(bias16 + shift16) + int32(l)
	} else {
		// normal number
		frac |= 1 << shift16
	}
	return
}
