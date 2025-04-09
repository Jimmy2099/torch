package torch

import (
	"github.com/Jimmy2099/torch/data_struct/tensor" // 假设这个库存在
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"reflect"
	"testing"
	// "github.com/Jimmy2099/torch/pkg/log" // 如果需要添加日志或调试信息
)

func TestMeanCalculation(t *testing.T) {
	// 创建测试张量 (N=2, C=3, H=4, W=4)
	data := make([]float32, 2*3*4*4)

	// 正确填充每个通道的值
	for n := 0; n < 2; n++ { // N维度
		for c := 0; c < 3; c++ { // C维度
			for h := 0; h < 4; h++ { // H维度
				for w := 0; w < 4; w++ { // W维度
					// 计算线性索引
					index := n*(3*4*4) + c*(4*4) + h*4 + w
					data[index] = float32(c) // 每个通道的值等于通道号
				}
			}
		}
	}

	x := tensor.NewTensor(data, []int{2, 3, 4, 4})
	bn := NewBatchNormLayer(3, 1e-5, 0.9)

	// 计算均值
	mean := bn.computeMean(x)

	// 验证形状
	if !reflect.DeepEqual(mean.Shape, []int{3}) {
		t.Fatalf("形状错误: 期望 [3]，实际 %v", mean.Shape)
	}

	// 验证数值
	tolerance := float32(1e-6)
	for i := 0; i < 3; i++ {
		expected := float32(i)
		if math.Abs(mean.Data[i]-expected) > tolerance {
			t.Errorf("通道 %d 均值错误: 期望 %.2f，实际 %.2f", i, expected, mean.Data[i])
		}
	}
}

func TestBatchNormShapeMismatch(t *testing.T) {
	// 配置参数
	numFeatures := 256
	batchSize := 64
	inputShape := []int{batchSize, numFeatures, 8, 8} // 模拟dec_convT0的输出形状

	// 创建BN层
	bn := NewBatchNormLayer(numFeatures, 1e-5, 0.1)

	// 创建模拟输入（注意：实际应该使用随机/正态分布数据）
	inputData := make([]float32, batchSize*numFeatures*8*8)
	x := tensor.NewTensor(inputData, inputShape)

	// 运行前向传播
	fmt.Println("=== 触发形状不匹配测试 ===")
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("Expected panic captured: %v\n", r)
		}
	}()

	output := bn.Forward(x) // 这里应该触发panic

	// 如果未panic则测试失败
	t.Error("Expected panic did not occur")
	_ = output
}

// 辅助测试：验证维度压缩是否正确
func TestComputeMeanShape(t *testing.T) {
	numFeatures := 256
	bn := NewBatchNormLayer(numFeatures, 1e-5, 0.1)

	// 创建模拟输入 [64, 256, 8, 8]
	x := tensor.Ones([]int{64, 256, 8, 8})

	// 计算均值
	batchMean := bn.computeMean(x) // 假设有导出这个方法用于测试

	// 验证形状
	expectedShape := []int{numFeatures}

	// Check if the number of dimensions is the same
	if len(batchMean.Shape) != len(expectedShape) {
		t.Errorf("Batch mean shape length mismatch! Expected %v, Got %v",
			expectedShape, batchMean.Shape)

	}

	// Check if each dimension size matches
	for i := range batchMean.Shape {
		if batchMean.Shape[i] != expectedShape[i] {
			t.Errorf("Batch mean shape length mismatch! Expected %v, Got %v",
				expectedShape, batchMean.Shape)
		}
	}
}
