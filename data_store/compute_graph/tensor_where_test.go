package compute_graph

import (
	"fmt"
	"testing"
)

func TestWhere(t *testing.T) {
	graph := NewComputationalGraph()

	condition := graph.NewGraphTensor([]float32{1, 0, 1, 0}, []int{2, 2}, "condition")

	trueValues := graph.NewGraphTensor([]float32{10, 20, 30, 40}, []int{2, 2}, "true_values")

	falseValues := graph.NewGraphTensor([]float32{1, 2, 3, 4}, []int{2, 2}, "false_values")

	output := condition.Where(trueValues, falseValues, "where_output")
	graph.SetOutput(output)

	fmt.Println("Computation Graph Structure:")
	graph.PrintStructure()

	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Where Output: %v\n", output.value.Data)

	output.Grad().Fill(1.0)
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Condition Gradients: %v\n", condition.Grad().Data)
	fmt.Printf("True Values Gradients: %v\n", trueValues.Grad().Data)
	fmt.Printf("False Values Gradients: %v\n", falseValues.Grad().Data)
}
