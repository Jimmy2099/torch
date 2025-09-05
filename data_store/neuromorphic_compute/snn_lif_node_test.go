package neuromorphic_compute

import (
	"testing"

	"github.com/Jimmy2099/torch/data_store/compute_graph"
	"github.com/chewxy/math32"
)

func TestLIFNodeForward(t *testing.T) {
	graph := compute_graph.NewComputationalGraph()
	inputData := []float32{0.3, 0.6, 0.9, 1.2, -0.2}
	inputTensor := graph.NewGraphTensor(inputData, []int{5}, "input")

	lifOutput := AddLIFNode(graph, inputTensor, 0.5, 0.1, "lif_output")
	graph.SetOutput(lifOutput)
	graph.Forward()

	outputValue := lifOutput.Value()
	expectedOutput := []float32{0.0, 1.0, 1.0, 1.0, 0.0}

	for i, val := range outputValue.Data {
		if val != expectedOutput[i] {
			t.Errorf("Output at index %d does not match: expected %f, actual %f", i, expectedOutput[i], val)
		}
	}
	t.Logf("Forward propagation test passed: input %v, output %v", inputData, outputValue.Data)
}

func TestLIFNodeBackward(t *testing.T) {
	graph := compute_graph.NewComputationalGraph()
	inputData := []float32{0.4, 0.6, 0.5}
	inputTensor := graph.NewGraphTensor(inputData, []int{3}, "input")

	lifOutput := AddLIFNode(graph, inputTensor, 0.5, 0.1, "lif_output")
	graph.SetOutput(lifOutput)

	graph.Forward()
	graph.Backward()

	inputGrad := inputTensor.Grad()
	expectedGrad := []float32{1.0, 1.0, 1.0}

	for i, grad := range inputGrad.Data {
		if grad != expectedGrad[i] {
			t.Errorf("Gradient at index %d does not match: input %f, expected %f, actual %f",
				i, inputData[i], expectedGrad[i], grad)
		}
	}
	t.Logf("Backward propagation test passed: gradient = %v", inputGrad.Data)
}

func TestLIFNodeMultipleLayers(t *testing.T) {
	graph := compute_graph.NewComputationalGraph()
	inputData := []float32{0.3, 0.7, 0.2, 0.8}
	inputTensor := graph.NewGraphTensor(inputData, []int{4}, "input")

	lif1Output := AddLIFNode(graph, inputTensor, 0.5, 0.1, "lif1_output")
	lif2Output := AddLIFNode(graph, lif1Output, 0.8, 0.1, "lif2_output")

	graph.SetOutput(lif2Output)
	graph.Forward()

	outputValue := lif2Output.Value()
	expectedOutput := []float32{0.0, 1.0, 0.0, 1.0}

	for i, val := range outputValue.Data {
		if val != expectedOutput[i] {
			t.Errorf("Output at index %d does not match: expected %f, actual %f", i, expectedOutput[i], val)
		}
	}
	t.Logf("Multi-layer LIF node test passed: final output %v", outputValue.Data)
}

func TestLIFNodeWithOtherOps(t *testing.T) {
	graph := compute_graph.NewComputationalGraph()

	inputTensor1 := graph.NewGraphTensor([]float32{0.3, 0.6}, []int{2}, "input1")
	inputTensor2 := graph.NewGraphTensor([]float32{0.4, 0.5}, []int{2}, "input2")

	multiplied := inputTensor1.Multiply(inputTensor2, "multiplied")
	lifOutput := AddLIFNode(graph, multiplied, 0.2, 0.1, "lif_output")

	graph.SetOutput(lifOutput)
	graph.Forward()

	multipliedValue := multiplied.Value()
	expectedMultiplied := []float32{0.12, 0.3}

	for i, val := range multipliedValue.Data {
		if math32.Abs(val-expectedMultiplied[i]) > 1e-6 {
			t.Errorf("Multiplication result mismatch at index %d: expected %f, actual %f", i, expectedMultiplied[i], val)
		}
	}

	lifValue := lifOutput.Value()
	expectedLIFOutput := []float32{0.0, 1.0}

	for i, val := range lifValue.Data {
		if val != expectedLIFOutput[i] {
			t.Errorf("LIF output mismatch at index %d: expected %f, actual %f", i, expectedLIFOutput[i], val)
		}
	}
	t.Logf("LIF combined with other operations test passed")
}

func TestLIFNodeEdgeCases(t *testing.T) {
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Logf("Empty input test: Correctly handled empty input, panic triggered: %v", r)
			}
		}()
		graph := compute_graph.NewComputationalGraph()
		inputTensor := graph.NewGraphTensor([]float32{}, []int{0}, "empty_input")
		lifOutput := AddLIFNode(graph, inputTensor, 0.5, 0.1, "lif_output")
		graph.SetOutput(lifOutput)
		graph.Forward()
	}()

	graph := compute_graph.NewComputationalGraph()
	inputData := []float32{0.5, 0.5, 0.5}
	inputTensor := graph.NewGraphTensor(inputData, []int{3}, "threshold_input")
	lifOutput := AddLIFNode(graph, inputTensor, 0.5, 0.1, "lif_output")
	graph.SetOutput(lifOutput)
	graph.Forward()

	outputValue := lifOutput.Value()
	for i, val := range outputValue.Data {
		if val != 0.0 {
			t.Errorf("When threshold equals input value, expected output 0.0, actual output %f", val)
		}
		t.Logf("Threshold equals input value test: input %f, output %f", inputData[i], val)
	}
	t.Logf("Edge case test completed")
}

func BenchmarkLIFNode(b *testing.B) {
	graph := compute_graph.NewComputationalGraph()
	size := 10000
	inputData := make([]float32, size)
	for i := range inputData {
		inputData[i] = float32(i%100) / 100.0
	}
	inputTensor := graph.NewGraphTensor(inputData, []int{size}, "large_input")

	lifOutput := AddLIFNode(graph, inputTensor, 0.5, 0.1, "lif_output")
	graph.SetOutput(lifOutput)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		graph.Forward()
		graph.Backward()
	}
}
