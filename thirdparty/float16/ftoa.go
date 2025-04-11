// convert float32 to string

package float16

import (
	"math/bits"
	"strconv"

	"github.com/shogo82148/int128"
)

func (x Float16) String() string {
	return x.Text('g', -1)
}

func (x Float16) Text(fmt byte, prec int) string {
	return string(x.Append(make([]byte, 0, 8), fmt, prec))
}

func (x Float16) Append(buf []byte, fmt byte, prec int) []byte {
	switch {
	case x.IsNaN():
		return append(buf, "NaN"...)
	case x == uvinf:
		return append(buf, "+Inf"...)
	case x == uvneginf:
		return append(buf, "-Inf"...)
	}

	switch fmt {
	case 'b':
		return x.appendBin(buf)
	case 'x', 'X':
		return x.appendHex(buf, fmt, prec)
	case 'f':
		return x.appendDec(buf, fmt, prec)
	case 'e', 'E':
		return x.appendSci(buf, fmt, prec)
	case 'g', 'G':
		if prec >= 0 {
			// In this case, bitSize is ignored anyway so it's ok to pass.
			return strconv.AppendFloat(buf, x.Float64(), fmt, prec, 64)
		}

		if x&signMask16 != 0 {
			buf = append(buf, '-')
		}
		x = x &^ signMask16
		if x == 0 {
			return append(buf, '0')
		}
		if x <= 0x068d { // 9.996e-05
			return x.appendSci(buf, fmt+'e'-'g', prec-1)
		}
		return x.appendDec(buf, fmt, prec)
	}

	return x.appendDec(buf, fmt, prec)
}

func (x Float16) appendBin(buf []byte) []byte {
	if x&signMask16 != 0 {
		buf = append(buf, '-')
	}
	exp := int(x>>shift16&mask16) - bias16
	frac := x & fracMask16

	if exp == -bias16 {
		exp++
	} else {
		frac |= 1 << shift16
	}
	exp -= shift16

	switch {
	case frac >= 1000:
		buf = append(buf, byte((frac/1000)%10)+'0')
		fallthrough
	case frac >= 100:
		buf = append(buf, byte((frac/100)%10)+'0')
		fallthrough
	case frac >= 10:
		buf = append(buf, byte((frac/10)%10)+'0')
		fallthrough
	default:
		buf = append(buf, byte(frac%10)+'0')
	}

	buf = append(buf, 'p')
	if exp >= 0 {
		buf = append(buf, '+')
	} else {
		buf = append(buf, '-')
		exp = -exp
	}

	switch {
	case exp >= 10:
		buf = append(buf, byte(exp/10)+'0')
		fallthrough
	default:
		buf = append(buf, byte(exp%10)+'0')
	}
	return buf
}

func (x Float16) appendDec(buf []byte, fmt byte, prec int) []byte {
	ten := int128.Uint128{L: 10}

	// sign
	if x&signMask16 != 0 {
		buf = append(buf, '-')
	}

	if prec >= 0 {
		const five24 = 59604644775390625 // = 5^24

		var dec24 int128.Uint128
		fix := x.fix24()

		// round to nearest even
		dec24.H, dec24.L = bits.Mul64(uint64(fix), five24)
		if prec < 24 {
			n := int128.Uint128{L: 1}
			for i := 0; i < 24-prec; i++ {
				n = n.Mul(ten)
			}
			n2 := n.Rsh(1)
			div, mod := dec24.DivMod(n)
			dec24 = dec24.Sub(mod)
			if mod.Cmp(n2) > 0 {
				// round up
				dec24 = dec24.Add(n)
			} else if mod.Cmp(n2) == 0 {
				// round to even
				if div.L&1 != 0 {
					dec24 = dec24.Add(n)
				}
			}
		}

		// convert to decimal
		var data [24]byte
		for i := 0; i < 24; i++ {
			var mod int128.Uint128
			dec24, mod = dec24.DivMod(ten)
			data[i] = byte(mod.L)
		}

		// convert integer part
		switch {
		case dec24.L >= 10000:
			buf = append(buf, byte((dec24.L/10000)%10)+'0')
			fallthrough
		case dec24.L >= 1000:
			buf = append(buf, byte((dec24.L/1000)%10)+'0')
			fallthrough
		case dec24.L >= 100:
			buf = append(buf, byte((dec24.L/100)%10)+'0')
			fallthrough
		case dec24.L >= 10:
			buf = append(buf, byte((dec24.L/10)%10)+'0')
			fallthrough
		default:
			buf = append(buf, byte(dec24.L%10)+'0')
		}
		if prec == 0 {
			return buf
		}

		// convert fractional part
		buf = append(buf, '.')
		var i int
		for i = 0; i < prec && i < len(data); i++ {
			buf = append(buf, data[23-i]+'0')
		}
		for ; i < prec; i++ {
			buf = append(buf, '0')
		}

		return buf
	}

	// find the intermediate value between two adjacent floating-point numbers.
	var exact, lower, upper int128.Uint128
	exp := int(x >> shift16 & mask16)
	frac := uint64(x & fracMask16)
	if exp == 0 {
		// subnormal number
		if frac == 0 {
			return append(buf, '0')
		}
		exact.L = frac * 2
		lower.L = exact.L - 1
		upper.L = exact.L + 1
	} else {
		// normal number
		exact.L = (frac | (1 << shift16)) << exp
		if frac&(frac-1) == 0 && exp > 1 {
			// frac is power of 2
			lower.L = exact.L - (1 << (exp - 2))
		} else {
			lower.L = exact.L - (1 << (exp - 1))
		}
		upper.L = exact.L + (1 << (exp - 1))
	}

	const five25 = 298023223876953125 // = 5^25
	exact.H, exact.L = bits.Mul64(exact.L, five25)
	lower.H, lower.L = bits.Mul64(lower.L, five25)
	upper.H, upper.L = bits.Mul64(upper.L, five25)

	var n int = 25
	var dec25 int128.Uint128
	for ; n > 0; n-- {
		dec25 = roundUint128(exact, n)
		if dec25.Cmp(lower) >= 0 && dec25.Cmp(upper) <= 0 {
			break
		}
	}

	// convert to decimal
	var data [25]byte
	for i := 0; i < 25; i++ {
		var mod int128.Uint128
		dec25, mod = dec25.DivMod(ten)
		data[i] = byte(mod.L)
	}

	// convert integer part
	switch {
	case dec25.L >= 10000:
		buf = append(buf, byte((dec25.L/10000)%10)+'0')
		fallthrough
	case dec25.L >= 1000:
		buf = append(buf, byte((dec25.L/1000)%10)+'0')
		fallthrough
	case dec25.L >= 100:
		buf = append(buf, byte((dec25.L/100)%10)+'0')
		fallthrough
	case dec25.L >= 10:
		buf = append(buf, byte((dec25.L/10)%10)+'0')
		fallthrough
	default:
		buf = append(buf, byte(dec25.L%10)+'0')
	}
	if n == 25 {
		return buf
	}

	// convert fractional part
	buf = append(buf, '.')
	var i int
	for i = 24; i >= n; i-- {
		buf = append(buf, data[i]+'0')
	}

	return buf
}

func (x Float16) appendSci(buf []byte, fmt byte, prec int) []byte {
	ten := int128.Uint128{L: 10}

	// sign
	if x&signMask16 != 0 {
		buf = append(buf, '-')
		x &^= signMask16
	}

	if prec >= 0 {
		const five24 = 59604644775390625 // = 5^24

		var dec24 int128.Uint128
		fix := x.fix24()
		dec24.H, dec24.L = bits.Mul64(uint64(fix), five24)

		if fix == 0 {
			buf = append(buf, '0')
			if prec > 0 {
				buf = append(buf, '.')
				for i := 0; i < prec; i++ {
					buf = append(buf, '0')
				}
			}
			buf = append(buf, fmt, '+', '0', '0')
			return buf
		}

		// find the first non-zero digit
		tmp := dec24
		var n int
		for ; tmp.H != 0 || tmp.L != 0; n++ {
			tmp = tmp.Div(ten)
		}
		n--

		// round to nearest even
		if prec < n {
			m := int128.Uint128{L: 1}
			for i := 0; i < n-prec; i++ {
				m = m.Mul(ten)
			}
			m2 := m.Rsh(1)
			div, mod := dec24.DivMod(m)
			dec24 = dec24.Sub(mod)
			if mod.Cmp(m2) > 0 {
				// round up
				dec24 = dec24.Add(m)
			} else if mod.Cmp(m2) == 0 {
				// round to even
				if div.L&1 != 0 {
					dec24 = dec24.Add(m)
				}
			}
		}

		// convert to decimal
		var data [30]byte
		for i := 0; i < 30; i++ {
			var mod int128.Uint128
			dec24, mod = dec24.DivMod(ten)
			data[i] = byte(mod.L)
		}

		// find the first non-zero digit
		i := len(data) - 1
		for ; i >= 0; i-- {
			if data[i] != 0 {
				break
			}
		}
		buf = append(buf, data[i]+'0')
		i--

		if prec != 0 {
			buf = append(buf, '.')
			var j int
			for ; i >= 0 && j < prec; i, j = i-1, j+1 {
				buf = append(buf, data[i]+'0')
			}
			for ; j < prec; j++ {
				buf = append(buf, '0')
			}
		}

		buf = append(buf, fmt)
		n -= 24
		if n >= 0 {
			buf = append(buf, '+')
		} else {
			buf = append(buf, '-')
			n = -n
		}
		buf = append(buf, byte((n/10)%10)+'0', byte(n%10)+'0')
		return buf
	}

	// find the intermediate value between two adjacent floating-point numbers.
	var exact, lower, upper int128.Uint128
	exp := int(x >> shift16 & mask16)
	frac := uint64(x & fracMask16)
	if exp == 0 {
		// subnormal number
		if frac == 0 {
			return append(buf, '0', fmt, '+', '0', '0')
		}
		exact.L = frac * 2
		lower.L = exact.L - 1
		upper.L = exact.L + 1
	} else {
		// normal number
		exact.L = (frac | (1 << shift16)) << exp
		if frac&(frac-1) == 0 && exp > 1 {
			// frac is power of 2
			lower.L = exact.L - (1 << (exp - 2))
		} else {
			lower.L = exact.L - (1 << (exp - 1))
		}
		upper.L = exact.L + (1 << (exp - 1))
	}

	const five25 = 298023223876953125 // = 5^25
	exact.H, exact.L = bits.Mul64(exact.L, five25)
	lower.H, lower.L = bits.Mul64(lower.L, five25)
	upper.H, upper.L = bits.Mul64(upper.L, five25)

	var n int = 30
	var dec25 int128.Uint128
	for ; n > 0; n-- {
		dec25 = roundUint128(exact, n)
		if dec25.Cmp(lower) > 0 && dec25.Cmp(upper) < 0 {
			break
		}
	}

	// convert to decimal
	var data [30]byte
	for i := 0; i < 30; i++ {
		var mod int128.Uint128
		dec25, mod = dec25.DivMod(ten)
		data[i] = byte(mod.L)
	}

	// find the first non-zero digit
	i := len(data) - 1
	for ; i >= 0; i-- {
		if data[i] != 0 {
			break
		}
	}
	buf = append(buf, data[i]+'0')
	i--
	m := i

	// convert fractional part
	if i+1 != n {
		buf = append(buf, '.')
		for ; i >= n; i-- {
			buf = append(buf, data[i]+'0')
		}
	}

	// exponent
	buf = append(buf, fmt)
	m -= 24
	if m >= 0 {
		buf = append(buf, '+')
	} else {
		buf = append(buf, '-')
		m = -m
	}
	buf = append(buf, byte((m/10)%10)+'0', byte(m%10)+'0')
	return buf
}

func roundUint128(x int128.Uint128, n int) int128.Uint128 {
	ten := int128.Uint128{L: 10}
	y := int128.Uint128{L: 1}
	for i := 0; i < n; i++ {
		y = y.Mul(ten)
	}
	y2 := y.Rsh(1)

	// round to nearest even
	div, mod := x.DivMod(y)
	x = x.Sub(mod)
	cmp := mod.Cmp(y2)
	if cmp > 0 {
		// round up
		return x.Add(y)
	}
	if cmp < 0 {
		// round down
		return x
	}

	// round to even
	if div.L&1 != 0 {
		return x.Add(y)
	}
	return x
}

func (x Float16) appendHex(buf []byte, fmt byte, prec int) []byte {
	// normalize x
	sign, exp, frac := x.split()
	if sign != 0 {
		buf = append(buf, '-')
	}
	buf = append(buf, '0', fmt) // 0x or 0X
	if x&^signMask16 == 0 {
		// zero
		buf = append(buf, '0')
		if prec >= 1 {
			buf = append(buf, '.')
			for i := 0; i < prec; i++ {
				buf = append(buf, '0')
			}
		}
		buf = append(buf, fmt-('x'-'p'))
		buf = append(buf, '+', '0', '0')
		return buf
	}

	switch prec {
	case -1:
		if frac&0x3ff == 0 {
			buf = append(buf, '1')
		} else if frac&0x3f == 0 {
			buf = append(buf, '1', '.')
			buf = append(buf, nibble(fmt, frac>>6))
		} else if frac&0x3 == 0 {
			buf = append(buf, '1', '.')
			buf = append(buf, nibble(fmt, frac>>6))
			buf = append(buf, nibble(fmt, frac>>2))
		} else {
			buf = append(buf, '1', '.')
			buf = append(buf, nibble(fmt, frac>>6))
			buf = append(buf, nibble(fmt, frac>>2))
			buf = append(buf, nibble(fmt, frac<<2))
		}

	case 0:
		// round to nearest even
		frac += 1 << (shift16 - 1)
		if frac >= 1<<(shift16+1) {
			exp++
			frac >>= 1
		}

		buf = append(buf, '1')

	case 1:
		// round to nearest even
		frac += 0x1f + (frac>>6)&1
		if frac >= 1<<(shift16+1) {
			exp++
			frac >>= 1
		}

		buf = append(buf, '1', '.')
		buf = append(buf, nibble(fmt, frac>>6))

	case 2:
		// round to nearest even
		frac += 1 + (frac>>2)&1
		if frac >= 1<<(shift16+1) {
			exp++
			frac >>= 1

		}

		buf = append(buf, '1', '.')
		buf = append(buf, nibble(fmt, frac>>6))
		buf = append(buf, nibble(fmt, frac>>2))

	default:
		if prec < 0 {
			panic("invalid precision")
		}
		buf = append(buf, '1', '.')
		buf = append(buf, nibble(fmt, frac>>6))
		buf = append(buf, nibble(fmt, frac>>2))
		buf = append(buf, nibble(fmt, frac<<2))
		for i := 3; i < prec; i++ {
			buf = append(buf, '0')
		}
	}

	buf = append(buf, fmt-('x'-'p'))
	if exp >= 0 {
		buf = append(buf, '+')
	} else {
		buf = append(buf, '-')
		exp = -exp
	}
	buf = append(buf, byte(exp/10)+'0', byte(exp%10)+'0')
	return buf
}

func nibble(fmt byte, x uint16) byte {
	x &= 0xf
	if x < 10 {
		return '0' + byte(x)
	}
	return ('A' + byte(x-10)) | (fmt & ('a' - 'A'))
}

// underscoreOK reports whether the underscores in s are allowed.
// Checking them in this one function lets all the parsers skip over them simply.
// Underscore must appear only between digits or between a base prefix and a digit.
func underscoreOK(s string) bool {
	// saw tracks the last character (class) we saw:
	// ^ for beginning of number,
	// 0 for a digit or base prefix,
	// _ for an underscore,
	// ! for none of the above.
	saw := '^'
	i := 0

	// Optional sign.
	if len(s) >= 1 && (s[0] == '-' || s[0] == '+') {
		s = s[1:]
	}

	// Optional base prefix.
	hex := false
	if len(s) >= 2 && s[0] == '0' && (lower(s[1]) == 'b' || lower(s[1]) == 'o' || lower(s[1]) == 'x') {
		i = 2
		saw = '0' // base prefix counts as a digit for "underscore as digit separator"
		hex = lower(s[1]) == 'x'
	}

	// Number proper.
	for ; i < len(s); i++ {
		// Digits are always okay.
		if '0' <= s[i] && s[i] <= '9' || hex && 'a' <= lower(s[i]) && lower(s[i]) <= 'f' {
			saw = '0'
			continue
		}
		// Underscore must follow digit.
		if s[i] == '_' {
			if saw != '0' {
				return false
			}
			saw = '_'
			continue
		}
		// Underscore must also be followed by digit.
		if saw == '_' {
			return false
		}
		// Saw non-digit, non-underscore.
		saw = '!'
	}
	return saw != '_'
}
