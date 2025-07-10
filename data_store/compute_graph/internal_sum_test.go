package compute_graph

import (
	"fmt"
	"testing"
)

func TestSum(t *testing.T) {
	graph := NewComputationalGraph()

	// 创建输入张量
	data := []float32{1.0, 2.0, 3.0, 4.0}
	input := graph.NewGraphTensor(data, []int{2, 2}, "input")

	// 应用Sum操作
	sum := input.Sum("total_sum")

	// 设置输出节点
	graph.SetOutput(sum)

	fmt.Println("\n\nComputation Graph Structure for Sum Test:")
	graph.PrintStructure()

	// 前向传播
	graph.Forward()
	fmt.Println("\nAfter Forward Pass:")
	fmt.Printf("Sum: %v (expected: [10])\n", sum.value.Data)

	// 反向传播
	graph.Backward()
	fmt.Println("\nAfter Backward Pass:")
	fmt.Printf("Input Gradients: %v (expected: [1 1 1 1])\n", input.Grad().Data)

	// 验证结果
	if len(sum.value.Data) != 1 || sum.value.Data[0] != 10 {
		t.Errorf("Sum calculation failed. Got %v, expected [10]", sum.value.Data)
	}

	expectedGrad := []float32{1.0, 1.0, 1.0, 1.0}
	for i, grad := range input.Grad().Data {
		if grad != expectedGrad[i] {
			t.Errorf("Gradient mismatch at index %d. Got %f, expected %f", i, grad, expectedGrad[i])
			break
		}
	}
}
