package float16

import (
	"fmt"
	"testing"
)

func TestFormat(t *testing.T) {
	tests := []struct {
		format string
		x      Float16
		want   string
	}{
		// verb "%b"
		{"%b", FromFloat64(0), "0p-24"},

		// verb "%f"
		{"%f", FromFloat64(0.5), "0.5"},
		{"%f", FromFloat64(-0.5), "-0.5"},
		{"%+f", FromFloat64(0.5), "+0.5"},
		{"%+f", FromFloat64(-0.5), "-0.5"},
		{"% f", FromFloat64(0.5), " 0.5"},
		{"% f", FromFloat64(-0.5), "-0.5"},
		{"%8f", FromFloat64(0.5), "     0.5"},
		{"%-8f", FromFloat64(0.5), "0.5     "},
		{"%.2f", FromFloat64(0.5), "0.50"},

		// verb "%e"
		{"%.6e", FromFloat64(0.5), "5.000000e-01"},

		// verb "%g"
		{"%g", FromFloat64(0.5), "0.5"},
		{"%.1g", FromFloat64(0.25), "0.2"},

		// verb "%x"
		{"%x", FromFloat64(0.5), "0x1p-01"},
		{"%#x", FromFloat64(0.5), "0x1p-01"},
		{"%.1x", FromFloat64(0.5), "0x1.0p-01"},

		// verb "%X"
		{"%X", FromFloat64(0.5), "0X1P-01"},
		{"%#X", FromFloat64(0.5), "0X1P-01"},
		{"%.1X", FromFloat64(0.5), "0X1.0P-01"},

		// verb "%v"
		{"%v", FromFloat64(0.5), "0.5"},
		{"%v", uvnan, "NaN"},
		{"%v", uvnan | signMask16, "NaN"},
	}

	for _, tt := range tests {
		got := fmt.Sprintf(tt.format, tt.x)
		if got != tt.want {
			t.Errorf("expected %s, got %s", tt.want, got)
		}
	}
}
