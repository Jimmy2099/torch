package float16

import (
	"bufio"
	"cmp"
	"compress/gzip"
	"errors"
	"math"
	"math/big"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/quick"
)

// xorshift32 is a pseudo random number generator.
// https://en.wikipedia.org/wiki/Xorshift
type xorshift32 uint32 // xorshift32

func newXorshift32() *xorshift32 {
	x := xorshift32(42)
	return &x
}

func (x *xorshift32) Uint32() uint32 {
	a := *x
	a ^= a << 13
	a ^= a >> 17
	a ^= a << 5
	*x = a
	return uint32(a)
}

// Float16Pair returns a pair of random Float16 values.
// It is used to obtain benchmarks in case of CPU branch mis-prediction.
func (x *xorshift32) Float16Pair() (Float16, Float16) {
	u32 := x.Uint32()
	a := Float16(u32 & 0xffff)
	b := Float16((u32 >> 16) & 0xffff)
	return a, b
}

func (x *xorshift32) Float32() float32 {
	bits := x.Uint32()
	return math.Float32frombits(bits)
}

type xorshift64 uint64 // xorshift64

func newXorshift64() *xorshift64 {
	x := xorshift64(42)
	return &x
}

func (x *xorshift64) Uint64() uint64 {
	a := *x
	a ^= a << 13
	a ^= a >> 7
	a ^= a << 17
	*x = a
	return uint64(a)
}

func (x *xorshift64) Float64() float64 {
	bits := x.Uint64()
	return math.Float64frombits(bits)
}

func BenchmarkFloat16Pair(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		x.Float16Pair()
	}
}

func checkEqual(t *testing.T, f, g func(a, b uint16) uint16, op string) {
	t.Helper()
	if testing.Short() {
		if err := quick.CheckEqual(f, g, &quick.Config{
			MaxCountScale: 100,
		}); err != nil {
			var checkErr *quick.CheckEqualError
			if errors.As(err, &checkErr) {
				a := checkErr.In[0].(uint16)
				b := checkErr.In[1].(uint16)
				c1 := checkErr.Out1[0].(uint16)
				c2 := checkErr.Out2[0].(uint16)

				fa := FromBits(a).Float64()
				fb := FromBits(b).Float64()
				fc1 := FromBits(c1).Float64()
				fc2 := FromBits(c2).Float64()

				t.Errorf("%x(%x) %s %x(%x): got %x(%x), expected %x(%x)", fa, a, op, fb, b, fc1, c1, fc2, c2)
			}
			t.Error(err)
		}
	} else {
		var wg sync.WaitGroup
		for a := 0; a < 0x10000; a++ {
			a := a
			wg.Add(1)
			go func() {
				defer wg.Done()
				for b := 0; b < 0x10000; b++ {
					got := f(uint16(a), uint16(b))
					want := g(uint16(a), uint16(b))
					if got != want {
						fa := Float16(a).Float64()
						fb := Float16(b).Float64()
						fc1 := Float16(got).Float64()
						fc2 := Float16(want).Float64()
						t.Errorf("%x(%x) %s %x(%x): got %x(%x), expected %x(%x)", fa, a, op, fb, b, fc1, got, fc2, want)
					}
				}
			}()
		}
		wg.Wait()
	}
}

func checkEqualInt(t *testing.T, f, g func(a, b uint16) int, op string) {
	if testing.Short() {
		if err := quick.CheckEqual(f, g, &quick.Config{
			MaxCountScale: 100,
		}); err != nil {
			var checkErr *quick.CheckEqualError
			if errors.As(err, &checkErr) {
				a := checkErr.In[0].(uint16)
				b := checkErr.In[1].(uint16)
				c1 := checkErr.Out1[0].(int)
				c2 := checkErr.Out2[0].(int)

				fa := FromBits(a).Float64()
				fb := FromBits(b).Float64()

				t.Errorf("%x(%x) %s %x(%x): got %d, expected %d", fa, a, op, fb, b, c1, c2)
			}
			t.Error(err)
		}
	} else {
		var wg sync.WaitGroup
		for a := 0; a < 0x10000; a++ {
			a := a
			wg.Add(1)
			go func() {
				defer wg.Done()
				for b := 0; b < 0x10000; b++ {
					got := f(uint16(a), uint16(b))
					want := g(uint16(a), uint16(b))
					if got != want {
						fa := Float16(a).Float64()
						fb := Float16(b).Float64()
						t.Errorf("%x(%x) %s %x(%x): got %d, expected %d", fa, a, op, fb, b, got, want)
					}
				}
			}()
		}
		wg.Wait()
	}
}

func TestMul(t *testing.T) {
	tests := []struct {
		a, b float64
	}{
		// normal * normal => normal
		{1, 1}, // 1 * 1 = 1
		{1, 2}, // 1 * 2 = 2
		{0x1.f44p-01, 0x1.fa8p-01},
		{0x1.efp-01, 0x1.08cp+00},
		{0x1p-14, 0x1p-10},
		{0x1p-10, 0x1p-14},

		// normal * normal => subnormal
		{-0x1.c14p-12, -0x1.32cp-12},

		// subnormal * normal => normal
		{0x1p-15, 2}, // 0x1p-15 * 2  = 0x1p-14

		// normal * subnormal => normal
		{2, 0x1p-15}, // 0x1p-15 * 2 = 0x1p-14

		// subnormal * normal => subnormal
		{0x1p-24, 2}, // 0x1p-24 * 2 = 0x1p-23
		{-0x1.1d8p-09, 0x1.07p-13},

		// subnormal * subnormal => subnormal
		{0, 0}, // 0 * 0 = 0
		{negZero, 0},

		// underflow
		{0x1p-14, 0x1p-11},
		{0x1p-11, 0x1p-14},

		// overflow
		{-0x1.b9p+06, -0x1.e24p+09},

		// Infinity * 0 => NaN
		{math.Inf(1), 0},
		{0, math.Inf(1)},

		// Infinity * anything => Infinity
		{math.Inf(1), 1},
		{math.Inf(1), -1},
		{math.Inf(-1), 1},
		{math.Inf(-1), -1},

		// anything * Infinity => Infinity
		{1, math.Inf(1)},
		{-1, math.Inf(1)},
		{1, math.Inf(-1)},
		{-1, math.Inf(-1)},

		// NaN * anything = NaN
		// anything * NaN = NaN
		{math.NaN(), 1},
		{math.NaN(), math.Inf(1)},
		{1, math.NaN()},
		{math.Inf(1), math.NaN()},
		{math.NaN(), math.NaN()},
	}
	for _, tt := range tests {
		fa := FromFloat64(tt.a)
		if !fa.IsNaN() && fa.Float64() != tt.a {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.a)
		}
		fb := FromFloat64(tt.b)
		if !fb.IsNaN() && fb.Float64() != tt.b {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.b)
		}
		fr := FromFloat64(tt.a * tt.b)
		fc := fa.Mul(fb)
		if fc.Compare(fr) != 0 {
			t.Errorf("%x * %x: expected %x (0x%04x), got %x (0x%04x)", tt.a, tt.b, fr.Float64(), fr, fc.Float64(), fc)
		}
	}
}

//go:generate sh -c "perl scripts/f16_mul.pl | gofmt > f16_mul_test.go"

func TestMul_TestFloat(t *testing.T) {
	for _, tt := range f16Mul {
		fa := tt.a
		fb := tt.b
		got := fa.Mul(fb)
		if got.Compare(tt.want) != 0 {
			t.Errorf("%x * %x: expected %x, got %x", tt.a, tt.b, tt.want, got)
		}
	}
}

func BenchmarkMul(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		fa, fb := x.Float16Pair()
		runtime.KeepAlive(fa.Mul(fb))
	}
}

func BenchmarkMul2(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		fa, fb := x.Float16Pair()
		fc := fa.Float64() * fb.Float64()
		runtime.KeepAlive(FromFloat64(fc))
	}
}

func TestMul_All(t *testing.T) {
	f := func(a, b uint16) uint16 {
		fa := Float16(a)
		fb := Float16(b)
		fc := fa.Mul(fb)
		if fc.IsNaN() {
			return uvnan
		}
		return fc.Bits()
	}

	g := func(a, b uint16) uint16 {
		fa := Float16(a).Float64()
		fb := Float16(b).Float64()
		fc := fa * fb // This calculation does not cause any rounding.
		if math.IsNaN(fc) {
			return uvnan
		}
		return FromFloat64(fc).Bits()
	}

	checkEqual(t, f, g, "*")
}

func TestQuo(t *testing.T) {
	tests := []struct {
		a, b, r float64
	}{
		{1, 2, 0.5},
		{1, 3, 0x1.554p-02},
		{1, 5, 0x1.998p-03},
		{0x1.46p-11, 0x1.13cp+02, 0x1.2ecp-13},
		{0x1p+15, 0.5, math.Inf(1)},
		{0x1p-14, 2, 0x1p-15},
		{0x1p-24, 0x1p-24, 1},
		{0x1.144p+11, 0x1.d18p-05, 0x1.2fcp+15},
		{0x1.e7p+10, 0x1.0acp-04, 0x1.d34p+14},
		{0x1.cp-21, 0x1.1p-20, 0x1.a5cp-01},
		{0x1.bf4p-05, 0x1.dp+09, 0x1.ed8p-15},

		// NaN / anything = NaN
		{math.NaN(), 1, math.NaN()},
		{math.NaN(), 0, math.NaN()},
		{math.NaN(), negZero, math.NaN()},
		{math.NaN(), math.Inf(1), math.NaN()},
		{math.NaN(), math.Inf(-1), math.NaN()},

		// anything / NaN = NaN
		{1, math.NaN(), math.NaN()},
		{0, math.NaN(), math.NaN()},
		{negZero, math.NaN(), math.NaN()},

		{math.Inf(1), 1, math.Inf(1)},
		{math.Inf(1), 0, math.Inf(1)},
		{math.Inf(1), math.Inf(1), math.NaN()},
		{1, math.Inf(1), 0},
		{1, math.Inf(-1), negZero},
		{1, 0, math.Inf(1)},

		// 0 / 0
		{negZero, 0, math.NaN()},
		{0, negZero, math.NaN()},
		{0, 0, math.NaN()},
		{negZero, negZero, math.NaN()},

		// 0 / anything = 0
		{0, 1, 0},
		{0, -1, negZero},
	}

	for _, tt := range tests {
		fa := FromFloat64(tt.a)
		if !fa.IsNaN() && fa.Float64() != tt.a {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.a)
		}
		fb := FromFloat64(tt.b)
		if !fb.IsNaN() && fb.Float64() != tt.b {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.b)
		}
		fr := FromFloat64(tt.r)
		if !fr.IsNaN() && fr.Float64() != tt.r {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.b)
		}
		fc := fa.Quo(fb)
		if fc.Compare(fr) != 0 {
			t.Errorf("%x / %x: expected %x (0x%04x), got %x (0x%04x)", tt.a, tt.b, fr.Float64(), fr, fc.Float64(), fc)
		}
	}
}

//go:generate sh -c "perl scripts/f16_div.pl | gofmt > f16_div_test.go"
func TestQuo_TestFloat(t *testing.T) {
	for _, tt := range f16Div {
		fa := tt.a
		fb := tt.b
		got := fa.Quo(fb)
		if got.Compare(tt.want) != 0 {
			t.Errorf("%x / %x: expected %x, got %x", tt.a, tt.b, tt.want, got)
		}
	}
}

func TestQuo_All(t *testing.T) {
	f := func(a, b uint16) uint16 {
		fa := Float16(a)
		fb := Float16(b)
		fc := fa.Quo(fb)
		if fc.IsNaN() {
			return uvnan
		}
		return fc.Bits()
	}

	g := func(a, b uint16) uint16 {
		fa := Float16(a).Float64()
		fb := Float16(b).Float64()

		if math.IsNaN(fa) || math.IsNaN(fb) || math.IsInf(fa, 0) || math.IsInf(fb, 0) || fb == 0 {
			// big.Float can't handle these special cases.
			fc := FromFloat64(fa / fb)
			if fc.IsNaN() {
				return uvnan
			}
			return fc.Bits()
		}

		bigA := new(big.Float).SetFloat64(fa)
		bigB := new(big.Float).SetFloat64(fb)
		bigC := new(big.Float).SetPrec(53).SetMode(big.AwayFromZero)
		bigC = bigC.Quo(bigA, bigB)
		f64, _ := bigC.Float64()
		return FromFloat64(f64).Bits()
	}

	checkEqual(t, f, g, "/")
}

func TestAdd(t *testing.T) {
	tests := []struct {
		a, b float64
	}{
		// normal + normal => normal
		{1, 1},
		{1, 2},
		{2, -1},
		{1, 0x1p-11},
		{1 + 0x1p-10, 0x1p-11},
		{0x1p+15, 0x1p+15}, // overflow

		// subnormal + subnormal => subnormal
		{0, 0},
		{negZero, 0},
		{negZero, negZero},
		{0, negZero},
		{0x1p-24, 0x1p-24},

		// infinity + anything => infinity
		{math.Inf(-1), 0x1p+14},
		{math.Inf(1), math.Inf(1)},

		// anything + infinity => infinity
		{0x1p+14, math.Inf(-1)},

		// infinity - infinity => infinity
		{math.Inf(1), math.Inf(-1)},
		{math.Inf(-1), math.Inf(1)},

		// NaN + anything = NaN
		{math.NaN(), -0x1p+14},
		{math.NaN(), math.Inf(1)},

		// anything + NaN = NaN
		{-0x1p+14, math.NaN()},
		{math.Inf(1), math.NaN()},
	}
	for _, tt := range tests {
		fa := FromFloat64(tt.a)
		if !fa.IsNaN() && fa.Float64() != tt.a {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.a)
		}
		fb := FromFloat64(tt.b)
		if !fb.IsNaN() && fb.Float64() != tt.b {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.b)
		}
		fr := FromFloat64(tt.a + tt.b)
		fc := fa.Add(fb)
		if fc.Compare(fr) != 0 {
			t.Errorf("%x + %x: expected %x (0x%04x), got %x (0x%04x)", tt.a, tt.b, fr.Float64(), fr, fc.Float64(), fc)
		}

		fr = FromFloat64(tt.b - tt.a)
		fc = fb.Sub(fa)
		if fc.Compare(fr) != 0 {
			t.Errorf("%x - %x: expected %x (0x%04x), got %x (0x%04x)", tt.b, tt.a, fr.Float64(), fr, fc.Float64(), fc)
		}
	}
}

//go:generate sh -c "perl scripts/f16_add.pl | gofmt > f16_add_test.go"
func TestAdd_TestFloat(t *testing.T) {
	for _, tt := range f16Add {
		fa := tt.a
		fb := tt.b
		got := fa.Add(fb)
		if got.Compare(tt.want) != 0 {
			t.Errorf("%x + %x: expected %x, got %x", tt.a, tt.b, tt.want, got)
		}
	}
}

func TestAdd_All(t *testing.T) {
	f := func(a, b uint16) uint16 {
		fa := Float16(a)
		fb := Float16(b)
		fc := fa.Add(fb)
		if fc.IsNaN() {
			return uvnan
		}
		return fc.Bits()
	}

	g := func(a, b uint16) uint16 {
		fa := Float16(a).Float64()
		fb := Float16(b).Float64()
		fc := fa + fb // This calculation does not cause any rounding.
		if math.IsNaN(fc) {
			return uvnan
		}
		return FromFloat64(fc).Bits()
	}

	checkEqual(t, f, g, "+")
}

func BenchmarkAdd(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		fa, fb := x.Float16Pair()
		runtime.KeepAlive(fa.Add(fb))
	}
}

func BenchmarkAdd2(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		fa, fb := x.Float16Pair()
		fc := fa.Float64() + fb.Float64()
		runtime.KeepAlive(FromFloat64(fc))
	}
}

func TestCompare(t *testing.T) {
	tests := []struct {
		a, b float64
	}{
		// positive numbers
		{1, 1},
		{1, 2},
		{2, 1},

		// negative numbers
		{-1, -1},
		{-1, -2},
		{-2, -1},

		// positive and negative numbers
		{-1, 1},
		{1, -1},
		{-2, 1},
		{2, -1},

		// infinity
		{math.Inf(1), 1},
		{math.Inf(-1), -1},
		{math.Inf(1), math.Inf(1)},
		{math.Inf(1), math.Inf(-1)},
		{math.Inf(-1), math.Inf(1)},
		{math.Inf(-1), math.Inf(-1)},

		// a NaN is considered less than any non-NaN
		{math.NaN(), 0},
		{math.NaN(), 1},
		{math.NaN(), math.Inf(1)},
		{math.NaN(), math.Inf(-1)},
		{math.NaN(), math.NaN()},

		// negative zero
		{0, 0},
		{negZero, 0},
		{0, negZero},
		{negZero, negZero},
	}
	for _, tt := range tests {
		fa := FromFloat64(tt.a)
		if !fa.IsNaN() && fa.Float64() != tt.a {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.a)
		}
		fb := FromFloat64(tt.b)
		if !fb.IsNaN() && fb.Float64() != tt.b {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.b)
		}
		fr := cmp.Compare(tt.a, tt.b)
		fc := fa.Compare(fb)
		if fc != fr {
			t.Errorf("%x <=> %x: expected %d, got %d", tt.a, tt.b, fr, fc)
		}
	}
}

func TestCmp_All(t *testing.T) {
	f := func(a, b uint16) int {
		fa := Float16(a)
		fb := Float16(b)
		return fa.Compare(fb)
	}

	g := func(a, b uint16) int {
		fa := Float16(a).Float64()
		fb := Float16(b).Float64()
		return cmp.Compare(fa, fb)
	}

	checkEqualInt(t, f, g, "<=>")
}

func BenchmarkCompare(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		fa, fb := x.Float16Pair()
		runtime.KeepAlive(fa.Compare(fb))
	}
}

func BenchmarkCompare2(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		fa, fb := x.Float16Pair()
		c := cmp.Compare(fa.Float64(), fb.Float64())
		runtime.KeepAlive(c)
	}
}

func TestEq(t *testing.T) {
	tests := []struct {
		a, b float64
	}{
		// positive numbers
		{1, 1},
		{1, 2},
		{2, 1},

		// negative numbers
		{-1, -1},
		{-1, -2},
		{-2, -1},

		// positive and negative numbers
		{-1, 1},
		{1, -1},
		{-2, 1},
		{2, -1},

		// infinity
		{math.Inf(1), 1},
		{math.Inf(-1), -1},
		{math.Inf(1), math.Inf(1)},
		{math.Inf(1), math.Inf(-1)},
		{math.Inf(-1), math.Inf(1)},
		{math.Inf(-1), math.Inf(-1)},

		// a NaN is not equal to any number, including NaN
		{math.NaN(), 0},
		{math.NaN(), 1},
		{math.NaN(), math.Inf(1)},
		{math.NaN(), math.Inf(-1)},
		{math.NaN(), math.NaN()},

		// negative zero
		{0, 0},
		{negZero, 0},
		{0, negZero},
		{negZero, negZero},
	}
	for _, tt := range tests {
		fa := FromFloat64(tt.a)
		if !fa.IsNaN() && fa.Float64() != tt.a {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.a)
		}
		fb := FromFloat64(tt.b)
		if !fb.IsNaN() && fb.Float64() != tt.b {
			t.Errorf("%x + %x: invalid test case: converting %x to float16 loss data", tt.a, tt.b, tt.b)
		}
		var want, got bool

		want = tt.a == tt.b
		got = fa.Eq(fb)
		if want != got {
			t.Errorf("%x == %x: expected %t, got %t", tt.a, tt.b, want, got)
		}

		want = tt.a != tt.b
		got = fa.Ne(fb)
		if want != got {
			t.Errorf("%x != %x: expected %t, got %t", tt.a, tt.b, want, got)
		}

		want = tt.a < tt.b
		got = fa.Lt(fb)
		if want != got {
			t.Errorf("%x < %x: expected %t, got %t", tt.a, tt.b, want, got)
		}

		want = tt.a > tt.b
		got = fa.Gt(fb)
		if want != got {
			t.Errorf("%x > %x: expected %t, got %t", tt.a, tt.b, want, got)
		}

		want = tt.a <= tt.b
		got = fa.Le(fb)
		if want != got {
			t.Errorf("%x <= %x: expected %t, got %t", tt.a, tt.b, want, got)
		}

		want = tt.a >= tt.b
		got = fa.Ge(fb)
		if want != got {
			t.Errorf("%x >= %x: expected %t, got %t", tt.a, tt.b, want, got)
		}
	}
}

//go:generate sh -c "perl scripts/f16_eq.pl | gofmt > f16_eq_test.go"

func TestEq_TestFloat(t *testing.T) {
	for _, tt := range f16Eq {
		fa := tt.a
		fb := tt.b
		got := fa.Eq(fb)
		if got != tt.want {
			t.Errorf("%x == %x: expected %t, got %t", tt.a, tt.b, tt.want, got)
		}
	}
}

//go:generate sh -c "perl scripts/f16_lt.pl | gofmt > f16_lt_test.go"

func TestLt_TestFloat(t *testing.T) {
	for _, tt := range f16Lt {
		fa := tt.a
		fb := tt.b
		got := fa.Lt(fb)
		if got != tt.want {
			t.Errorf("%x < %x: expected %t, got %t", tt.a, tt.b, tt.want, got)
		}
	}
}

//go:generate sh -c "perl scripts/f16_le.pl | gofmt > f16_le_test.go"

func TestLe_TestFloat(t *testing.T) {
	for _, tt := range f16Le {
		fa := tt.a
		fb := tt.b
		got := fa.Le(fb)
		if got != tt.want {
			t.Errorf("%x <= %x: expected %t, got %t", tt.a, tt.b, tt.want, got)
		}
	}
}

func BenchmarkEq(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		fa, fb := x.Float16Pair()
		runtime.KeepAlive(fa.Eq(fb))
	}
}

func BenchmarkLt(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		fa, fb := x.Float16Pair()
		runtime.KeepAlive(fa.Lt(fb))
	}
}

func BenchmarkLe(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		fa, fb := x.Float16Pair()
		runtime.KeepAlive(fa.Le(fb))
	}
}

func TestFMA_TestFloat(t *testing.T) {
	f, err := os.Open("testdata/f16_mulAdd.txt.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	r, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	s := bufio.NewScanner(r)
	for s.Scan() {
		line := s.Text()
		line = strings.TrimSpace(line)
		seg := strings.Split(line, " ")
		if len(seg) < 4 {
			t.Fatal("invalid test data")
		}
		x, err := strconv.ParseUint(seg[0], 16, 16)
		if err != nil {
			t.Fatal(err)
		}
		y, err := strconv.ParseUint(seg[1], 16, 16)
		if err != nil {
			t.Fatal(err)
		}
		z, err := strconv.ParseUint(seg[2], 16, 16)
		if err != nil {
			t.Fatal(err)
		}
		a, err := strconv.ParseUint(seg[3], 16, 16)
		if err != nil {
			t.Fatal(err)
		}

		fx := Float16(x)
		fy := Float16(y)
		fz := Float16(z)
		fa := Float16(a)
		got := FMA(fx, fy, fz)
		if got.Compare(fa) != 0 {
			t.Errorf("%x * %x + %x: expected %x, got %x", x, y, z, fa, got)
		}
	}
}

func BenchmarkFMA(b *testing.B) {
	x := newXorshift32()
	for i := 0; i < b.N; i++ {
		fa, fb := x.Float16Pair()
		fc, _ := x.Float16Pair()
		runtime.KeepAlive(FMA(fa, fb, fc))
	}
}
