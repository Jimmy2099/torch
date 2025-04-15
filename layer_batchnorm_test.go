package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor" // 假设这个库存在
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"reflect"
	"testing"
)

func TestMeanCalculation(t *testing.T) {
	data := make([]float32, 2*3*4*4)

	for n := 0; n < 2; n++ { // N维度
		for c := 0; c < 3; c++ { // C维度
			for h := 0; h < 4; h++ { // H维度
				for w := 0; w < 4; w++ { // W维度
					index := n*(3*4*4) + c*(4*4) + h*4 + w
					data[index] = float32(c) // 每个通道的值等于通道号
				}
			}
		}
	}

	x := tensor.NewTensor(data, []int{2, 3, 4, 4})
	bn := NewBatchNormLayer(3, 1e-5, 0.9)

	mean := bn.computeMean(x)

	if !reflect.DeepEqual(mean.Shape, []int{3}) {
		t.Fatalf("形状错误: 期望 [3]，实际 %v", mean.Shape)
	}

	tolerance := float32(1e-6)
	for i := 0; i < 3; i++ {
		expected := float32(i)
		if math.Abs(mean.Data[i]-expected) > tolerance {
			t.Errorf("通道 %d 均值错误: 期望 %.2f，实际 %.2f", i, expected, mean.Data[i])
		}
	}
}

func TestBatchNormShapeMismatch(t *testing.T) {
	numFeatures := 256
	batchSize := 64
	inputShape := []int{batchSize, numFeatures, 8, 8} // 模拟dec_convT0的输出形状

	bn := NewBatchNormLayer(numFeatures, 1e-5, 0.1)

	inputData := make([]float32, batchSize*numFeatures*8*8)
	x := tensor.NewTensor(inputData, inputShape)

	fmt.Println("=== 触发形状不匹配测试 ===")
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Expected panic captured: %v\n", r)
		}
	}()

	output := bn.Forward(x) // 这里应该触发panic

	t.Error("Expected panic did not occur")
	_ = output
}

func TestComputeMeanShape(t *testing.T) {
	numFeatures := 256
	bn := NewBatchNormLayer(numFeatures, 1e-5, 0.1)

	x := tensor.Ones([]int{64, 256, 8, 8})

	batchMean := bn.computeMean(x) // 假设有导出这个方法用于测试

	expectedShape := []int{numFeatures}

	if len(batchMean.Shape) != len(expectedShape) {
		t.Errorf("Batch mean shape length mismatch! Expected %v, Got %v",
			expectedShape, batchMean.Shape)

	}

	for i := range batchMean.Shape {
		if batchMean.Shape[i] != expectedShape[i] {
			t.Errorf("Batch mean shape length mismatch! Expected %v, Got %v",
				expectedShape, batchMean.Shape)
		}
	}
}
