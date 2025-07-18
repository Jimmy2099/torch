package compute_graph

import (
	"testing"
)

func TestSqrt(t *testing.T) {
	graph := NewComputationalGraph()
	input := graph.NewGraphTensor([]float32{4, 9, 0.25}, []int{3}, "input")
	output := input.Sqrt("sqrt_output")
	graph.SetOutput(output)

	graph.Forward()
	expected := []float32{2, 3, 0.5}
	if !compareSlices(output.value.Data, expected) {
		t.Errorf("Forward pass failed. Expected %v, got %v", expected, output.value.Data)
	}

	graph.Backward()
	expectedGrad := []float32{0.25, 1.0 / 6, 1}
	if !compareSlices(input.Grad().Data, expectedGrad, 1e-5) {
		t.Errorf("Backward pass failed. Expected %v, got %v", expectedGrad, input.Grad().Data)
	}
}
