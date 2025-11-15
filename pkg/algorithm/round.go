package algorithm

import (
	math "github.com/chewxy/math32"
)

func RoundFloat32(f float32, precision int) float32 {
	if precision < 0 {
		return f
	}
	multiplier := math.Pow10(precision)
	return math.Round(f*multiplier) / multiplier
}
