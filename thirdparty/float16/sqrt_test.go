package float16

import (
	"math"
	"runtime"
	"testing"
)

func TestSqrt(t *testing.T) {
	tests := []struct {
		x float64
		y float64
	}{
		// special cases
		{0, 0},
		{negZero, negZero},
		{math.Inf(1), math.Inf(1)},
		{-1, math.NaN()},

		// normal numbers
		{1, 1},
		{2, 0x1.6ap+00},
		{3, 0x1.bb8p+00},
		{4, 0x2p+00},
	}

	for _, tt := range tests {
		x := FromFloat64(tt.x)
		y := x.Sqrt()
		if y != FromFloat64(tt.y) {
			t.Errorf("expected %x, got %x", tt.y, y.Float64())
		}
	}
}

func BenchmarkSqrt(b *testing.B) {
	r := newXorshift32()
	for i := 0; i < b.N; i++ {
		f, _ := r.Float16Pair()
		runtime.KeepAlive(f.Sqrt())
	}
}

//go:generate sh -c "perl scripts/f16_sqrt.pl | gofmt > f16_sqrt_test.go"

func TestSqrt_TestFloat(t *testing.T) {
	for _, tt := range f16Sqrt {
		x := FromBits(tt.x)
		got := x.Sqrt()
		want := FromBits(tt.y)
		if got.IsNaN() && want.IsNaN() {
			continue
		}
		if got != want {
			t.Errorf("expected %04x, got %04x", got.Bits(), want.Bits())
		}
	}
}
