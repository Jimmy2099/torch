package float16

import "math/bits"

// Sqrt returns the square root of x.
//
// Special cases are:
//
//	Sqrt(+Inf) = +Inf
//	Sqrt(±0) = ±0
//	Sqrt(x < 0) = NaN
//	Sqrt(NaN) = NaN
func (x Float16) Sqrt() Float16 {
	// special cases
	switch {
	case x&^signMask16 == 0 || x.IsNaN() || x.IsInf(1):
		return x
	case x&signMask16 != 0:
		return uvnan
	}

	// normalize x
	exp := int((x >> shift16) & mask16)
	frac := uint16(x & fracMask16)
	if exp == 0 {
		// subnormal number
		l := bits.Len32(uint32(frac))
		frac <<= shift16 - l + 1
		exp = -(bias16 + shift16) + l
	} else {
		// normal number
		frac |= 1 << shift16
		exp -= bias16
	}

	if exp%2 != 0 { // odd exp, double x to make it even
		frac <<= 1
	}
	// exponent of square root
	exp >>= 1

	// generate sqrt(frac) bit by bit
	frac <<= 1
	var q, s uint16 // q = sqrt(frac)
	r := uint16(1 << (shift16 + 1))
	for r != 0 {
		t := s + r
		if t <= frac {
			s = t + r
			frac -= t
			q += r
		}
		frac <<= 1
		r >>= 1
	}

	// final rounding
	if frac != 0 {
		q += q & 1
	}
	return Float16((exp-1+bias16)<<shift16) + Float16(q>>1)
}
