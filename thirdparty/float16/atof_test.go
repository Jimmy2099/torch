package float16

import (
	"cmp"
	"fmt"
	"strconv"
	"testing"
)

// exact returns the exact float16 representation of f.
// it panics if f doesn't have an exact float16 representation.
func exact(f float64) Float16 {
	ret := FromFloat64(f)
	if cmp.Compare(ret.Float64(), f) != 0 {
		panic(fmt.Sprintf("%f doesn't have exact float16 representation", f))
	}
	return ret
}

func TestParse(t *testing.T) {
	tests := []struct {
		s string
		x Float16
	}{
		{"0", 0},
		{"-0", 0x8000},
		{"+Inf", Inf(1)},
		{"-Inf", Inf(-1)},
		{"+infinity", Inf(1)},
		{"-infinity", Inf(-1)},
		{"NaN", NaN()},

		// greater than one
		{"1", exact(1)},
		{"2", exact(2)},
		{"4", exact(4)},
		{"8", exact(8)},
		{"16", exact(16)},
		{"32", exact(32)},
		{"64", exact(64)},
		{"128", exact(128)},
		{"65504", exact(65504)}, // maximum finite value

		// 65519 is greater than the maximum finite value 65504 but it's ok.
		// because it round down to 65504.
		{"65519", exact(65504)},

		// less than one
		{"0.5", exact(0.5)},
		{"0.25", exact(0.25)},
		{"0.125", exact(0.125)},
		{"0.00024414062", exact(0x1p-12)},
		{"0.00012207031", exact(0x1p-13)},
		{"0.00006103515625", exact(0x1p-14)},

		// subnormal numbers
		{"0.000030517578125", exact(0x1p-15)},
		{"0.000000059604644775390625", exact(0x1p-24)},

		// test rounding
		{"1.0009765625", exact(1.0009765625)}, // minimum value greater than one
		{"1.00048828125", exact(1)},
		{"1.0004882812500001", exact(1.0009765625)},
		{"1.0014648437499999", exact(1.0009765625)},
		{"1.00146484375", exact(1.001953125)},

		{"7.812e-03", exact(0x1p-7)},
		{"1.5625e-02", exact(0x1p-6)},
		{"0.007812", exact(0x1p-7)},
		{"0.015625", exact(0x1p-6)},

		// hexadecimal
		{"0x1p0", exact(1)},
		{"0x1.ffc0p+15", exact(65504)},
		{"0x1.ffcfp+15", exact(65504)}, // round down
		{"0x1p-14", exact(0x1p-14)},    // minimum nominal
		{"0x1p-24", exact(0x1p-24)},    // minimum subnormal greater than zero

		// test rounding
		{"0x1.000p0", exact(0x1.000p0)},
		{"0x1.001p0", exact(0x1.000p0)},
		{"0x1.002p0", exact(0x1.000p0)},
		{"0x1.003p0", exact(0x1.004p0)},
		{"0x1.004p0", exact(0x1.004p0)},
		{"0x1.005p0", exact(0x1.004p0)},
		{"0x1.006p0", exact(0x1.008p0)},
		{"0x1.007p0", exact(0x1.008p0)},
		{"0x1.008p0", exact(0x1.008p0)},
		{"0x1.009p0", exact(0x1.008p0)},
		{"0x1.00ap0", exact(0x1.008p0)},
		{"0x1.00bp0", exact(0x1.00cp0)},
		{"0x1.00cp0", exact(0x1.00cp0)},
		{"0x1.00dp0", exact(0x1.00cp0)},
		{"0x1.00ep0", exact(0x1.010p0)},
		{"0x1.00fp0", exact(0x1.010p0)},
	}

	for _, tt := range tests {
		got, err := Parse(tt.s)
		if err != nil {
			t.Errorf("%q: expected no error, got %v", tt.s, err)
		}
		if got != tt.x {
			t.Errorf("%q: expected %x, got %x", tt.s, tt.x, got)
		}
	}
}

func TestParse_overflow(t *testing.T) {
	test := []string{
		"65520",
		"6.552e4",
		"0x1.ffep+15",
	}

	for _, tt := range test {
		got, err := Parse(tt)
		if err == nil {
			t.Errorf("%q: expected overflow error, but nil", tt)
		}
		if numErr, ok := err.(*strconv.NumError); !ok {
			t.Errorf("%q: expected strconv.NumError, got %T", tt, err)
		} else {
			if numErr.Err != strconv.ErrRange {
				t.Errorf("%q: expected strconv.ErrRange, got %v", tt, numErr.Err)
			}
			if numErr.Num != tt {
				t.Errorf("%q: expected %q, got %q", tt, tt, numErr.Num)
			}
			if numErr.Func != "float16.Parse" {
				t.Errorf("%q: expected float16.Parse, got %q", tt, numErr.Func)
			}
		}
		if got != uvinf {
			t.Errorf("%q: expected +Inf, got %x", tt, got)
		}
	}
}

func TestParse_SyntaxError(t *testing.T) {
	tests := []string{
		"",
		" ",
		"NaN.0",
		"0x",
		"0x.",
		"0x.p",
		"0x.0p",
		"0x.0p+",
		"0x.0p-",
		"0x0p",
		"0x0p+",
		"0x0p-",
	}

	for _, tt := range tests {
		_, err := Parse(tt)
		if err == nil {
			t.Errorf("%q: expected syntax error, but nil", tt)
			continue
		}
		if numErr, ok := err.(*strconv.NumError); !ok {
			t.Errorf("%q: expected strconv.NumError, got %T", tt, err)
		} else {
			if numErr.Err != strconv.ErrSyntax {
				t.Errorf("%q: expected strconv.ErrSyntax, got %v", tt, numErr.Err)
			}
			if numErr.Num != tt {
				t.Errorf("%q: expected %q, got %q", tt, tt, numErr.Num)
			}
			if numErr.Func != "float16.Parse" {
				t.Errorf("%q: expected float16.Parse, got %q", tt, numErr.Func)
			}
		}
	}
}

func FuzzParse(f *testing.F) {
	f.Add("0")
	f.Add("-0")
	f.Add("+Inf")
	f.Add("-Inf")
	f.Add("NaN")
	f.Add("1")
	f.Add("1.0009765625")
	f.Add("1.00048828125")

	f.Fuzz(func(t *testing.T, s string) {
		x0, err := Parse(s)
		if err != nil {
			return
		}
		s1 := x0.String()

		x1, err := Parse(s1)
		if err != nil {
			t.Errorf("expected no error, got %v", err)
		}
		if x0.IsNaN() && x1.IsNaN() {
			return
		}
		if x0 != x1 {
			t.Errorf("expected %v, got %v", x0, x1)
		}
	})
}
