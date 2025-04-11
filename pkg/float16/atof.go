// convert string to float32

package float16

import (
	"strconv"
)

// lower(c) is a lower-case letter if and only if
// c is either that lower-case letter or the equivalent upper-case letter.
// Instead of writing c == 'x' || c == 'X' one can write lower(c) == 'x'.
// Note that lower of non-letters can produce other non-letters.
func lower(c byte) byte {
	return c | ('x' - 'X')
}

// commonPrefixLenIgnoreCase returns the length of the common
// prefix of s and prefix, with the character case of s ignored.
// The prefix argument must be all lower-case.
func commonPrefixLenIgnoreCase(s, prefix string) int {
	n := len(prefix)
	if n > len(s) {
		n = len(s)
	}
	for i := 0; i < n; i++ {
		c := s[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != prefix[i] {
			return i
		}
	}
	return n
}

func special(s string) (f Float16, n int, ok bool) {
	if len(s) == 0 {
		return 0, 0, false
	}

	sign := 1
	nsign := 0
	switch s[0] {
	case '+', '-':
		if s[0] == '-' {
			sign = -1
		}
		nsign = 1
		s = s[1:]
		fallthrough
	case 'i', 'I':
		n := commonPrefixLenIgnoreCase(s, "infinity")
		// Anything longer than "inf" is ok, but if we
		// don't have "infinity", only consume "inf".
		if 3 < n && n < 8 {
			n = 3
		}
		if n == 3 || n == 8 {
			return Inf(sign), nsign + n, true
		}
	case 'n', 'N':
		n := commonPrefixLenIgnoreCase(s, "nan")
		if n == 3 {
			return NaN(), n, true
		}
	}
	return 0, 0, false
}

func (b *decimal) set(s string) (ok bool) {
	i := 0
	b.neg = false
	b.trunc = false

	// optional sign
	if i >= len(s) {
		return
	}
	switch {
	case s[i] == '+':
		i++
	case s[i] == '-':
		b.neg = true
		i++
	}

	// digits
	sawdot := false
	sawdigits := false
	for ; i < len(s); i++ {
		switch {
		case s[i] == '_':
			// readFloat already checked underscores
			continue
		case s[i] == '.':
			if sawdot {
				return
			}
			sawdot = true
			b.dp = b.nd
			continue

		case '0' <= s[i] && s[i] <= '9':
			sawdigits = true
			if s[i] == '0' && b.nd == 0 { // ignore leading zeros
				b.dp--
				continue
			}
			if b.nd < len(b.d) {
				b.d[b.nd] = s[i]
				b.nd++
			} else if s[i] != '0' {
				b.trunc = true
			}
			continue
		}
		break
	}
	if !sawdigits {
		return
	}
	if !sawdot {
		b.dp = b.nd
	}

	// optional exponent moves decimal point.
	// if we read a very large, very long number,
	// just be sure to move the decimal point by
	// a lot (say, 100000).  it doesn't matter if it's
	// not the exact number.
	if i < len(s) && lower(s[i]) == 'e' {
		i++
		if i >= len(s) {
			return
		}
		esign := 1
		if s[i] == '+' {
			i++
		} else if s[i] == '-' {
			i++
			esign = -1
		}
		if i >= len(s) || s[i] < '0' || s[i] > '9' {
			return
		}
		e := 0
		for ; i < len(s) && ('0' <= s[i] && s[i] <= '9' || s[i] == '_'); i++ {
			if s[i] == '_' {
				// readFloat already checked underscores
				continue
			}
			if e < 10000 {
				e = e*10 + int(s[i]) - '0'
			}
		}
		b.dp += e * esign
	}

	if i != len(s) {
		return
	}

	ok = true
	return
}

// decimal power of ten to binary power of two.
var powtab = []int{1, 3, 6, 9, 13, 16, 19, 23, 26}

func (d *decimal) floatBits() (b uint16, overflow bool) {
	var exp int
	var mant uint64

	// Zero is always special.
	if d.nd == 0 {
		mant = 0
		exp = -bias16
		goto out
	}

	// Obvious overflow/underflow.
	if d.dp > 5 {
		goto overflow
	}
	if d.dp < -8 {
		// zero
		mant = 0
		exp = -bias16
		goto out
	}

	// Scale by powers of two until in range [0.5, 1.0)
	exp = 0
	for d.dp > 0 {
		var n int
		if d.dp >= len(powtab) {
			n = 27
		} else {
			n = powtab[d.dp]
		}
		d.Shift(-n)
		exp += n
	}
	for d.dp < 0 || d.dp == 0 && d.d[0] < '5' {
		var n int
		if -d.dp >= len(powtab) {
			n = 27
		} else {
			n = powtab[-d.dp]
		}
		d.Shift(n)
		exp -= n
	}

	// Our rage is [0.5,1) but floating point range is [1,2).
	exp--

	// Minimum exponent is -bias16.
	// If the exponent is smaller, denormalize.
	if exp <= -bias16 {
		n := -bias16 - exp + 1
		d.Shift(-n)
		exp += n
	}
	if exp+bias16 >= mask16 {
		goto overflow
	}

	// Extract 1+shift16 bits
	d.Shift(1 + shift16)
	mant = d.RoundedInteger()

	// Rounding might have added a bit; shift down.
	if mant == 2<<shift16 {
		mant >>= 1
		exp++
		if exp+bias16 >= mask16 {
			goto overflow
		}
	}

	// Denormalized?
	if mant&(1<<shift16) == 0 {
		exp = -bias16
	}
	goto out

overflow:
	// ±Inf
	mant = 0
	exp = mask16 - bias16
	overflow = true

out:
	// Assemble bits.
	bits := mant & fracMask16
	bits |= (uint64(exp+bias16) & mask16) << shift16
	if d.neg {
		bits |= signMask16
	}
	return uint16(bits), overflow
}

// readFloat reads a decimal or hexadecimal mantissa and exponent from a float
// string representation in s; the number may be followed by other characters.
// readFloat reports the number of bytes consumed (i), and whether the number
// is valid (ok).
//
// comes from https://github.com/golang/go/blob/8c92897e15d15fbc664cd5a05132ce800cf4017f/src/strconv/atof.go#L171-L309
func readFloat(s string) (mantissa uint64, exp int, neg, trunc, hex bool, i int, ok bool) {
	underscores := false

	// optional sign
	if i >= len(s) {
		return
	}
	switch {
	case s[i] == '+':
		i++
	case s[i] == '-':
		neg = true
		i++
	}

	// digits
	base := uint64(10)
	maxMantDigits := 19 // 10^19 fits in uint64
	expChar := byte('e')
	if i+2 < len(s) && s[i] == '0' && lower(s[i+1]) == 'x' {
		base = 16
		maxMantDigits = 16 // 16^16 fits in uint64
		i += 2
		expChar = 'p'
		hex = true
	}
	sawdot := false
	sawdigits := false
	nd := 0
	ndMant := 0
	dp := 0
loop:
	for ; i < len(s); i++ {
		switch c := s[i]; true {
		case c == '_':
			underscores = true
			continue

		case c == '.':
			if sawdot {
				break loop
			}
			sawdot = true
			dp = nd
			continue

		case '0' <= c && c <= '9':
			sawdigits = true
			if c == '0' && nd == 0 { // ignore leading zeros
				dp--
				continue
			}
			nd++
			if ndMant < maxMantDigits {
				mantissa *= base
				mantissa += uint64(c - '0')
				ndMant++
			} else if c != '0' {
				trunc = true
			}
			continue

		case base == 16 && 'a' <= lower(c) && lower(c) <= 'f':
			sawdigits = true
			nd++
			if ndMant < maxMantDigits {
				mantissa *= 16
				mantissa += uint64(lower(c) - 'a' + 10)
				ndMant++
			} else {
				trunc = true
			}
			continue
		}
		break
	}
	if !sawdigits {
		return
	}
	if !sawdot {
		dp = nd
	}

	if base == 16 {
		dp *= 4
		ndMant *= 4
	}

	// optional exponent moves decimal point.
	// if we read a very large, very long number,
	// just be sure to move the decimal point by
	// a lot (say, 100000).  it doesn't matter if it's
	// not the exact number.
	if i < len(s) && lower(s[i]) == expChar {
		i++
		if i >= len(s) {
			return
		}
		esign := 1
		if s[i] == '+' {
			i++
		} else if s[i] == '-' {
			i++
			esign = -1
		}
		if i >= len(s) || s[i] < '0' || s[i] > '9' {
			return
		}
		e := 0
		for ; i < len(s) && ('0' <= s[i] && s[i] <= '9' || s[i] == '_'); i++ {
			if s[i] == '_' {
				underscores = true
				continue
			}
			if e < 10000 {
				e = e*10 + int(s[i]) - '0'
			}
		}
		dp += e * esign
	} else if base == 16 {
		// Must have exponent.
		return
	}

	if mantissa != 0 {
		exp = dp - ndMant
	}

	if underscores && !underscoreOK(s[:i]) {
		return
	}

	ok = true
	return
}

// atofHex converts the hex floating-point string s
// The string s has already been parsed into a mantissa, exponent, and sign (neg==true for negative).
// If trunc is true, trailing non-zero bits have been omitted from the mantissa.
//
// based on https://github.com/golang/go/blob/8c92897e15d15fbc664cd5a05132ce800cf4017f/src/strconv/atof.go#L494-L562
func atofHex(s string, mantissa uint64, exp int, neg, trunc bool) (Float16, error) {
	maxExp := mask16 - bias16 - 1
	minExp := -bias16 + 1
	exp += int(shift16) // mantissa now implicitly divided by 2^shift16.

	// Shift mantissa and exponent to bring representation into float range.
	// Eventually we want a mantissa with a leading 1-bit followed by mantbits other bits.
	// For rounding, we need two more, where the bottom bit represents
	// whether that bit or any later bit was non-zero.
	// (If the mantissa has already lost non-zero bits, trunc is true,
	// and we OR in a 1 below after shifting left appropriately.)
	for mantissa != 0 && mantissa>>(shift16+2) == 0 {
		mantissa <<= 1
		exp--
	}
	if trunc {
		mantissa |= 1
	}
	for mantissa>>(1+shift16+2) != 0 {
		mantissa = mantissa>>1 | mantissa&1
		exp++
	}

	// If exponent is too negative,
	// denormalize in hopes of making it representable.
	// (The -2 is for the rounding bits.)
	for mantissa > 1 && exp < minExp-2 {
		mantissa = mantissa>>1 | mantissa&1
		exp++
	}

	// Round using two bottom bits.
	round := mantissa & 3
	mantissa >>= 2
	round |= mantissa & 1 // round to even (round up if mantissa is odd)
	exp += 2
	if round == 3 {
		mantissa++
		if mantissa == 1<<(1+shift16) {
			mantissa >>= 1
			exp++
		}
	}

	if mantissa>>shift16 == 0 { // subnormal or zero
		exp = -bias16
	}

	var err error
	if exp > maxExp {
		mantissa = 0
		exp = mask16 - bias16
		err = &strconv.NumError{Func: "float16.Parse", Num: s, Err: strconv.ErrRange}
	}

	bits := mantissa & fracMask16
	bits |= (uint64(exp+bias16) & mask16) << shift16
	if neg {
		bits |= signMask16
	}
	return Float16(bits), err
}

func atof16(s string) (f Float16, n int, err error) {
	if val, n, ok := special(s); ok {
		return val, n, nil
	}

	mantissa, exp, neg, trunc, hex, n, ok := readFloat(s)
	if !ok {
		return 0, n, &strconv.NumError{Func: "float16.Parse", Num: s, Err: strconv.ErrSyntax}
	}

	if hex {
		f, err = atofHex(s, mantissa, exp, neg, trunc)
		return f, n, err
	}

	var d decimal
	if !d.set(s[:n]) {
		return 0, n, &strconv.NumError{Func: "float16.Parse", Num: s, Err: strconv.ErrSyntax}
	}
	b, ovf := d.floatBits()
	f = Float16(b)
	if ovf {
		err = &strconv.NumError{Func: "float16.Parse", Num: s, Err: strconv.ErrRange}
	}
	return f, n, err
}

func Parse(s string) (Float16, error) {
	f, n, err := atof16(s)
	if n != len(s) && (err == nil || err.(*strconv.NumError).Err != strconv.ErrSyntax) {
		return 0, &strconv.NumError{Func: "float16.Parse", Num: s, Err: strconv.ErrSyntax}
	}
	return f, err
}
