package float16

import (
	"math"
	"runtime"
	"testing"
)

var negZero = math.Float64frombits(1 << 63)

func TestIsNaN(t *testing.T) {
	if !NaN().IsNaN() {
		t.Errorf("expected NaN")
	}
}

func TestIsInf(t *testing.T) {
	tests := []struct {
		f    Float16
		sign int
		inf  bool
	}{
		{Inf(1), 1, true},
		{Inf(-1), 1, false},
		{Inf(1), -1, false},
		{Inf(-1), -1, true},
		{Inf(1), 0, true},
		{Inf(-1), 0, true},
	}
	for _, tt := range tests {
		if tt.f.IsInf(tt.sign) != tt.inf {
			t.Errorf("%x: expected %v", tt.f, tt.sign)
		}
	}
}

func TestFromFloat32(t *testing.T) {
	tests := []struct {
		f float32
		r Float16
	}{
		// from https://en.wikipedia.org/wiki/Half-precision_floating-point_format
		{0, 0x0000},
		{0x1p-24, 0x0001},     // smallest positive subnormal number
		{0x1.ff8p-15, 0x03ff}, // largest positive subnormal number
		{0x1p-14, 0x0400},     // smallest positive normal number
		{0x1.554p-02, 0x3555}, // nearest value to 1/3
		{0x1.ffcp-01, 0x3bff}, // largest number less than one
		{0x1p+00, 0x3c00},     // one
		{0x1.004p+00, 0x3c01}, // smallest number larger than one
		{0x1.ffcp+15, 0x7bff}, // largest normal number
		{float32(negZero), 0x8000},
		{-2, 0xc000},

		// rounds to nearest even
		{0x1.002p+00, 0x3c00},
		{math.Nextafter32(0x1.002p+00, 2), 0x3c01},
		{math.Nextafter32(0x1.006p+00, 0), 0x3c01},
		{0x1.006p+00, 0x3c02},
		{0x1.ffcp-15, 0x0400},

		// underflow
		{0x1p-25, 0x0000},
		{0x1p-126, 0x0000},
		{0x1.fffffcp-127, 0x0000},

		// overflow
		{0x1p+16, 0x7c00},
		{0x1p+17, 0x7c00},
		{-0x1p+16, 0xfc00},
		{-0x1p+17, 0xfc00},

		// infinities
		{float32(math.Inf(1)), 0x7c00},
		{float32(math.Inf(-1)), 0xfc00},

		// NaN
		{float32(math.NaN()), 0x7e00},
	}
	for _, tt := range tests {
		r := FromFloat32(tt.f)
		if r != tt.r {
			t.Errorf("%x: expected %x, got %x", tt.f, tt.r, r)
		}
	}
}

//go:generate sh -c "perl scripts/f32_to_f16.pl | gofmt > f32_to_f16_test.go"
func TestFromFloat32_TestFloat(t *testing.T) {
	for _, tt := range f32ToF16 {
		f32 := math.Float32frombits(tt.f32)
		got := FromFloat32(f32)
		if got.Bits() != tt.f16 {
			t.Errorf("%08x: expected %04x, got %04x", tt.f32, tt.f16, got.Bits())
		}
	}
}

func TestFromFloat32_All(t *testing.T) {
	for bits := 0; bits < 1<<16; bits++ {
		f := FromBits(uint16(bits))
		if !f.IsNaN() && f != FromFloat32(f.Float32()) {
			t.Errorf("%x: expected %x, got %x", bits, f, FromFloat32(f.Float32()))
		}
	}
}

func TestFromFloat64(t *testing.T) {
	tests := []struct {
		f float64
		r Float16
	}{
		// from https://en.wikipedia.org/wiki/Half-precision_floating-point_format
		{0, 0x0000},
		{0x1p-24, 0x0001},     // smallest positive subnormal number
		{0x1.ff8p-15, 0x03ff}, // largest positive subnormal number
		{0x1p-14, 0x0400},     // smallest positive normal number
		{0x1.554p-02, 0x3555}, // nearest value to 1/3
		{0x1.ffcp-01, 0x3bff}, // largest number less than one
		{0x1p+00, 0x3c00},     // one
		{0x1.004p+00, 0x3c01}, // smallest number larger than one
		{0x1.ffcp+15, 0x7bff}, // largest normal number
		{float64(negZero), 0x8000},
		{-2, 0xc000},

		// rounds to nearest even
		{0x1.002p+00, 0x3c00},
		{math.Nextafter(0x1.002p+00, 2), 0x3c01},
		{math.Nextafter(0x1.006p+00, 0), 0x3c01},
		{0x1.006p+00, 0x3c02},
		{0x1.ffcp-15, 0x0400},

		// underflow
		{0x1p-25, 0x0000},
		{0x1p-126, 0x0000},
		{0x1.fffffcp-127, 0x0000},

		// overflow
		{0x1p+16, 0x7c00},
		{0x1p+17, 0x7c00},
		{-0x1p+16, 0xfc00},
		{-0x1p+17, 0xfc00},

		// infinities
		{math.Inf(1), 0x7c00},
		{math.Inf(-1), 0xfc00},

		// NaN
		{math.NaN(), 0x7e00},
	}
	for _, tt := range tests {
		r := FromFloat64(tt.f)
		if r != tt.r {
			t.Errorf("%x: expected %x, got %x", tt.f, tt.r, r)
		}
	}
}

//go:generate sh -c "perl scripts/f64_to_f16.pl | gofmt > f64_to_f16_test.go"
func TestFromFloat64_TestFloat(t *testing.T) {
	for _, tt := range f64ToF16 {
		f64 := math.Float64frombits(tt.f64)
		got := FromFloat64(f64)
		want := FromBits(tt.f16)
		if got.IsNaN() && want.IsNaN() {
			continue
		}
		if got != want {
			t.Errorf("%016x: expected %04x, got %04x", tt.f64, tt.f16, got.Bits())
		}
	}
}

func TestFromFloat64_All(t *testing.T) {
	for bits := 0; bits < 1<<16; bits++ {
		f16 := FromBits(uint16(bits))
		f64 := f16.Float64()
		got := FromFloat64(f64)
		if f16.IsNaN() && got.IsNaN() {
			continue
		}
		if got != f16 {
			t.Errorf("%x: expected %x, got %x", bits, f16, got)
		}
	}
}

func TestFloat32(t *testing.T) {
	tests := []struct {
		f Float16
		r float32
	}{
		// from https://en.wikipedia.org/wiki/Half-precision_floating-point_format
		{0x0000, 0},
		{0x0001, 0x1p-24},     // smallest positive subnormal number
		{0x03ff, 0x1.ff8p-15}, // largest positive subnormal number
		{0x0400, 0x1p-14},     // smallest positive normal number
		{0x3555, 0x1.554p-02}, // nearest value to 1/3
		{0x3bff, 0x1.ffcp-01}, // largest number less than one
		{0x3c00, 0x1p+00},     // one
		{0x3c01, 0x1.004p+00}, // smallest number larger than one
		{0x7bff, 0x1.ffcp+15}, // largest normal number
		{0x8000, -0},
		{0xc000, -2},
	}

	for _, tt := range tests {
		r := tt.f.Float32()
		if r != tt.r {
			t.Errorf("expected %x, got %x", tt.r, r)
		}
	}
}

func TestFloat32_Specials(t *testing.T) {
	// infinity
	if r := Inf(1).Float32(); !math.IsInf(float64(r), 1) {
		t.Errorf("expected +Inf, got %x", r)
	}

	// negative infinity
	if r := Inf(-1).Float32(); !math.IsInf(float64(r), -1) {
		t.Errorf("expected -Inf, got %x", r)
	}

	// NaN
	if r := NaN().Float32(); !math.IsNaN(float64(r)) {
		t.Errorf("expected NaN, got %x", r)
	}

	// zero
	if r := Float16(0x0000).Float32(); r != 0 || !math.IsInf(float64(1/r), 1) {
		t.Errorf("expected +0, got %x", r)
	}

	// negative zero
	if r := Float16(0x8000).Float32(); r != 0 || !math.IsInf(float64(1/r), -1) {
		t.Errorf("expected -0, got %x", r)
	}
}

func TestFloat64(t *testing.T) {
	tests := []struct {
		f Float16
		r float64
	}{
		// from https://en.wikipedia.org/wiki/Half-precision_floating-point_format
		{0x0000, 0},
		{0x0001, 0x1p-24},     // smallest positive subnormal number
		{0x03ff, 0x1.ff8p-15}, // largest positive subnormal number
		{0x0400, 0x1p-14},     // smallest positive normal number
		{0x3555, 0x1.554p-02}, // nearest value to 1/3
		{0x3bff, 0x1.ffcp-01}, // largest number less than one
		{0x3c00, 0x1p+00},     // one
		{0x3c01, 0x1.004p+00}, // smallest number larger than one
		{0x7bff, 0x1.ffcp+15}, // largest normal number
		{0x8000, -0},
		{0xc000, -2},
	}

	for _, tt := range tests {
		r := tt.f.Float64()
		if r != tt.r {
			t.Errorf("expected %x, got %x", tt.r, r)
		}
	}
}

func TestFloat64_Specials(t *testing.T) {
	// infinity
	if r := Inf(1).Float64(); !math.IsInf(r, 1) {
		t.Errorf("expected +Inf, got %x", r)
	}

	// negative infinity
	if r := Inf(-1).Float64(); !math.IsInf(r, -1) {
		t.Errorf("expected -Inf, got %x", r)
	}

	// NaN
	if r := NaN().Float64(); !math.IsNaN(r) {
		t.Errorf("expected NaN, got %x", r)
	}

	// zero
	if r := Float16(0x0000).Float64(); r != 0 || !math.IsInf(1/r, 1) {
		t.Errorf("expected +0, got %x", r)
	}

	// negative zero
	if r := Float16(0x8000).Float64(); r != 0 || !math.IsInf(1/r, -1) {
		t.Errorf("expected -0, got %x", r)
	}
}

func BenchmarkFromFloat32(b *testing.B) {
	r := newXorshift32()
	for i := 0; i < b.N; i++ {
		f := r.Float32()
		runtime.KeepAlive(FromFloat32(f))
	}
}

func BenchmarkFloat32(b *testing.B) {
	r := newXorshift32()
	for i := 0; i < b.N; i++ {
		f, _ := r.Float16Pair()
		runtime.KeepAlive(f.Float32())
	}
}

func BenchmarkFromFloat64(b *testing.B) {
	r := newXorshift64()
	for i := 0; i < b.N; i++ {
		f := r.Float64()
		runtime.KeepAlive(FromFloat64(f))
	}
}

func BenchmarkFloat64(b *testing.B) {
	r := newXorshift32()
	for i := 0; i < b.N; i++ {
		f, _ := r.Float16Pair()
		runtime.KeepAlive(f.Float64())
	}
}
