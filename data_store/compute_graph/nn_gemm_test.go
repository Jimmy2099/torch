package compute_graph_test

import (
	test "github.com/Jimmy2099/torch/testing"
	"github.com/stretchr/testify/assert"
	"testing"

	"github.com/Jimmy2099/torch/data_store/compute_graph"
)

func TestGemm(t *testing.T) {
	graph := compute_graph.NewComputationalGraph()

	a := graph.NewGraphTensor(
		[]float32{6, 2, 3, 4},
		[]int{2, 2},
		"a",
	)

	b := graph.NewGraphTensor(
		[]float32{9, 6, 7, 8},
		[]int{2, 2},
		"b",
	)

	c := graph.NewGraphTensor(
		[]float32{3, 1, 1, 1},
		[]int{2, 2},
		"c",
	)

	output := a.Gemm(b, c, false, false, 1.0, 1.0, "gemm_output")
	graph.SetOutput(output)

	graph.Forward()

	expected := []float32{
		71, 53,
		56, 51,
	}
	assert.Equal(t, expected, output.Value().Data)

	graph.Backward()

	expectedAGrad := []float32{15, 15, 15, 15}
	expectedBGrad := []float32{9, 9, 6, 6}
	expectedCGrad := []float32{1, 1, 1, 1}

	assert.Equal(t, expectedAGrad, a.Grad().Data)
	assert.Equal(t, expectedBGrad, b.Grad().Data)
	assert.Equal(t, expectedCGrad, c.Grad().Data)
}

func TestGemmPython(t *testing.T) {
	script := `import torch

a = torch.tensor([[6, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
b = torch.tensor([[9, 6], [7, 8]], dtype=torch.float32, requires_grad=True)
c = torch.tensor([[3, 1], [1, 1]], dtype=torch.float32, requires_grad=True)

alpha = 1.0
beta = 1.0
output = alpha * torch.mm(a, b) + beta * c

print("output: ",output)

loss = output.sum()
loss.backward()

print("grad result:")
print("A grad:")
print(a.grad)
print("B grad:")
print(b.grad)
print("C grad:")
print(c.grad)
`
	test.RunPyScript(script)
}
