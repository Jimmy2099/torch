package compute_graph

import (
	"math"
	"testing"
)

func TestLog(t *testing.T) {
	graph := NewComputationalGraph()
	input := graph.NewGraphTensor([]float32{1, float32(math.E), 7.389056}, []int{3}, "input")
	output := input.Log("log_output")
	graph.SetOutput(output)

	// Forward pass
	graph.Forward()
	expected := []float32{0, 1, 2} // ln(1)=0, ln(e)=1, ln(e^2)=2
	if !compareSlices(output.value.Data, expected, 1e-5) {
		t.Errorf("Forward pass failed. Expected %v, got %v", expected, output.value.Data)
	}

	// Backward pass
	graph.Backward()
	expectedGrad := []float32{1, 1 / float32(math.E), 1 / 7.389056}
	if !compareSlices(input.Grad().Data, expectedGrad, 1e-5) {
		t.Errorf("Backward pass failed. Expected %v, got %v", expectedGrad, input.Grad().Data)
	}
}
