package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
	test "github.com/Jimmy2099/torch/testing"
	"testing"
)

func TestConv(t *testing.T) {
	graph := NewComputationalGraph()

	inputData := []float32{
		1, 2, 3, 4,
		1, 2, 3, 4,
		1, 2, 3, 4,
		1, 2, 3, 4,
	}
	input := graph.NewGraphTensor(
		inputData,
		[]int{1, 1, 4, 4},
		"input",
	)

	kernelData := []float32{
		1, 0,
		0, 1,
	}
	kernel := graph.NewGraphTensor(
		kernelData,
		[]int{1, 1, 2, 2},
		"kernel",
	)

	output := input.Conv(kernel, []int{1, 1}, []int{0, 0}, "conv_output")
	graph.SetOutput(output)

	fmt.Println("Convolution Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Output: %v\n", output.value.Data)

	gradShape := output.value.GetShape()
	gradData := make([]float32, output.value.Size())
	for i := range gradData {
		gradData[i] = 1
	}
	output.grad = tensor.NewTensor(gradData, gradShape)

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v\n", input.Grad().Data)
	fmt.Printf("Kernel Gradients: %v\n", kernel.Grad().Data)
}

func TestConvPython(t *testing.T) {
	script := `import torch
import torch.nn as nn
import torch.nn.functional as F

class ComputationalGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.tensor([[[
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]]], dtype=torch.float32, requires_grad=True)
        
        self.kernel = torch.tensor([[[
            [1, 0],
            [0, 1]
        ]]], dtype=torch.float32, requires_grad=True)
    
    def forward(self):
        conv_output = F.conv2d(
            input=self.input,
            weight=self.kernel,
            stride=1,
            padding=0
        )
        return conv_output

graph = ComputationalGraph()

print("Convolution Computation Graph Structure:")
print(f"Input shape: {graph.input.shape}\n{graph.input.detach().squeeze()}")
print(f"\nKernel shape: {graph.kernel.shape}\n{graph.kernel.detach().squeeze()}")

output = graph()
print("\nAfter Forward Pass:")
print(f"Output: \n{output.detach().squeeze().numpy()}")

output_grad = torch.ones_like(output)
output.backward(gradient=output_grad)

print("\nAfter Backward Pass:")
print(f"Input Gradients: \n{graph.input.grad.squeeze().numpy()}")
print(f"Kernel Gradients: \n{graph.kernel.grad.squeeze().numpy()}")
`
	test.RunPyScript(script)
}
