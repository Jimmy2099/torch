package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"reflect"
	"testing"
)

func TestMeanCalculation(t *testing.T) {
	data := make([]float32, 2*3*4*4)

	for n := 0; n < 2; n++ {
		for c := 0; c < 3; c++ {
			for h := 0; h < 4; h++ {
				for w := 0; w < 4; w++ {
					index := n*(3*4*4) + c*(4*4) + h*4 + w
					data[index] = float32(c)
				}
			}
		}
	}

	x := tensor.NewTensor(data, []int{2, 3, 4, 4})
	bn := NewBatchNormLayer(3, 1e-5, 0.9)

	mean := bn.computeMean(x)

	if !reflect.DeepEqual(mean.Shape, []int{3}) {
		t.Fatalf("Shape error: Expected [3], Actual %v", mean.Shape)
	}

	tolerance := float32(1e-6)
	for i := 0; i < 3; i++ {
		expected := float32(i)
		if math.Abs(mean.Data[i]-expected) > tolerance {
			t.Errorf("Channel %d mean error: Expected %.2f, Actual %.2f", i, expected, mean.Data[i])
		}
	}
}

func TestBatchNormShapeMismatch(t *testing.T) {
	numFeatures := 256
	batchSize := 64
	inputShape := []int{batchSize, numFeatures, 8, 8}

	bn := NewBatchNormLayer(numFeatures, 1e-5, 0.1)

	inputData := make([]float32, batchSize*numFeatures*8*8)
	x := tensor.NewTensor(inputData, inputShape)

	fmt.Println("=== Triggering shape mismatch test ===")
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Expected panic captured: %v\n", r)
		}
	}()

	output := bn.Forward(x)

	t.Error("Expected panic did not occur")
	_ = output
}

func TestComputeMeanShape(t *testing.T) {
	numFeatures := 256
	bn := NewBatchNormLayer(numFeatures, 1e-5, 0.1)

	x := tensor.Ones([]int{64, 256, 8, 8})

	batchMean := bn.computeMean(x)

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
