package compute_graph

import (
	"math"
)

func compareSlices(a, b []float32, tolerance ...float32) bool {
	if len(a) != len(b) {
		return false
	}

	tol := float32(1e-5)
	if len(tolerance) > 0 {
		tol = tolerance[0]
	}

	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > float64(tol) {
			return false
		}
	}
	return true
}
