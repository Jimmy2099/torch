package compute_graph

import (
	"testing"
)

func TestSqrt(t *testing.T) {
	graph := NewComputationalGraph()
	input := graph.NewGraphTensor([]float32{4, 9, 0.25}, []int{3}, "input")
	output := input.Sqrt("sqrt_output")
	graph.SetOutput(output)

	// Forward pass
	graph.Forward()
	expected := []float32{2, 3, 0.5} // √4=2, √9=3, √0.25=0.5
	if !compareSlices(output.value.Data, expected) {
		t.Errorf("Forward pass failed. Expected %v, got %v", expected, output.value.Data)
	}

	// Backward pass
	graph.Backward()
	// 正确梯度: [1/(2*2), 1/(2*3), 1/(2*0.5)]
	expectedGrad := []float32{0.25, 1.0 / 6, 1}
	if !compareSlices(input.Grad().Data, expectedGrad, 1e-5) {
		t.Errorf("Backward pass failed. Expected %v, got %v", expectedGrad, input.Grad().Data)
	}
}
