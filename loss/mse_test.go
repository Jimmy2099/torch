package loss

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/compute_graph"
	"testing"
)

func TestComputeGraph(t *testing.T) {
	graph := compute_graph.NewComputationalGraph()

	x := graph.NewGraphTensor([]float32{2.0, 2.0, 2.0, 2.0}, []int{2, 2}, "x")
	w := graph.NewGraphTensor([]float32{3.0, 3.0, 3.0, 3.0}, []int{2, 2}, "w")
	b := graph.NewGraphTensor([]float32{1.0, 1.0, 1.0, 1.0}, []int{2, 2}, "b")

	target := graph.NewGraphTensor([]float32{25.0, 25.0, 25.0, 25.0}, []int{2, 2}, "target")

	wx := x.Multiply(w)
	xb := x.Multiply(b)
	add := wx.Multiply(xb)

	loss := MSE(graph, add, target, "mse_loss")
	graph.SetOutput(loss)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Output: %v\n", add.Value().Data)
	fmt.Printf("Loss: %v\n", loss.Value().Data[0])

	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Weights grad: %v\n", w.Grad().Data)
	fmt.Printf("Bias grad: %v\n", b.Grad().Data)

	learningRate := float32(0.1)
	wData := w.Value().Data
	wGrad := w.Grad().Data
	for i := range wData {
		wData[i] -= learningRate * wGrad[i]
	}

	bData := b.Value().Data
	bGrad := b.Grad().Data
	for i := range bData {
		bData[i] -= learningRate * bGrad[i]
	}

	fmt.Println("\nAfter Parameter Update:")
	fmt.Printf("Weight updated: %v\n", w.Value().Data)
	fmt.Printf("Bias updated: %v\n", b.Value().Data)

	graph.Forward()
	fmt.Println("\nAfter Update Forward Pass:")
	fmt.Printf("Output: %v\n", add.Value().Data)
	fmt.Printf("Loss: %v\n", loss.Value().Data[0])
}
