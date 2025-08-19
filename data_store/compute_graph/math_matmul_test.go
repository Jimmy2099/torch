package compute_graph

import (
	"fmt"
	"testing"

	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/stretchr/testify/assert"
)

func TestMatMul(t *testing.T) {
	graph := NewComputationalGraph()

	A := graph.NewGraphTensor([]float32{
		1, 2,
		3, 4,
		5, 6,
	}, []int{3, 2}, "A")

	B := graph.NewGraphTensor([]float32{
		7, 8, 9,
		10, 11, 12,
	}, []int{2, 3}, "B")

	C := A.MatMul(B, "matmul_result")

	graph.SetOutput(C)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Matrix C:\n%v\n", C.Value().String())

	expectedC := []float32{
		1*7 + 2*10, 1*8 + 2*11, 1*9 + 2*12,
		3*7 + 4*10, 3*8 + 4*11, 3*9 + 4*12,
		5*7 + 6*10, 5*8 + 6*11, 5*9 + 6*12,
	}
	assert.Equal(t, expectedC, C.Value().Data, "Matrix multiplication result is incorrect")
	assert.Equal(t, []int{3, 3}, C.Value().GetShape(), "Result shape is incorrect")

	outputGrad := tensor.NewTensor([]float32{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}, []int{3, 3})
	C.SetGrad(outputGrad)

	graph.Backward()

	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("A Gradients:\n%v\n", A.Grad().String())
	fmt.Printf("B Gradients:\n%v\n", B.Grad().String())

	expectedAGrad := []float32{
		7 + 8 + 9, 10 + 11 + 12,
		7 + 8 + 9, 10 + 11 + 12,
		7 + 8 + 9, 10 + 11 + 12,
	}
	assert.Equal(t, expectedAGrad, A.Grad().Data, "Gradient for A is incorrect")

	expectedBGrad := []float32{
		1 + 3 + 5, 1 + 3 + 5, 1 + 3 + 5,
		2 + 4 + 6, 2 + 4 + 6, 2 + 4 + 6,
	}
	assert.Equal(t, expectedBGrad, B.Grad().Data, "Gradient for B is incorrect")

	fmt.Println("\nUpdating parameters...")
	lr := float32(0.1)

	bData := B.Value().Data
	bGrad := B.Grad().Data
	for i := range bData {
		bData[i] -= lr * bGrad[i]
	}

	fmt.Printf("Updated B:\n%v\n", B.Value().String())

	graph.Forward()
	fmt.Println("\nAfter Update Forward Pass:")
	fmt.Printf("New Matrix C:\n%v\n", C.Value().String())

	expectedUpdatedC := []float32{
		1*(7-0.1*9) + 2*(10-0.1*12),
		1*(8-0.1*9) + 2*(11-0.1*12),
		1*(9-0.1*9) + 2*(12-0.1*12),

		3*(7-0.1*9) + 4*(10-0.1*12),
		3*(8-0.1*9) + 4*(11-0.1*12),
		3*(9-0.1*9) + 4*(12-0.1*12),

		5*(7-0.1*9) + 6*(10-0.1*12),
		5*(8-0.1*9) + 6*(11-0.1*12),
		5*(9-0.1*9) + 6*(12-0.1*12),
	}
	assert.InDeltaSlice(t, expectedUpdatedC, C.Value().Data, 0.001, "Updated matrix C is incorrect")
}
