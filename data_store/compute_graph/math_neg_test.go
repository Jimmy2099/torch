package compute_graph

import (
	"testing"
)

func TestNeg(t *testing.T) {
	graph := NewComputationalGraph()
	input := graph.NewGraphTensor([]float32{2, -3, 0.5}, []int{3}, "input")
	output := input.Neg("neg_output")
	graph.SetOutput(output)

	// Forward pass
	graph.Forward()
	expected := []float32{-2, 3, -0.5}
	if !compareSlices(output.Value().Data, expected) {
		t.Errorf("Forward pass failed. Expected %v, got %v", expected, output.Value().Data)
	}

	// Backward pass
	graph.Backward()
	expectedGrad := []float32{-1, -1, -1}
	if !compareSlices(input.Grad().Data, expectedGrad) {
		t.Errorf("Backward pass failed. Expected %v, got %v", expectedGrad, input.Grad().Data)
	}
}
