package torch

import (
	"github.com/Jimmy2099/torch/data_struct/tensor" // 假设这个库存在
	"math"
	"reflect"
	"testing"
	// "log" // 如果需要添加日志或调试信息
)

func TestMeanCalculation(t *testing.T) {
	// 创建测试张量 (N=2, C=3, H=4, W=4)
	data := make([]float64, 2*3*4*4)

	// 正确填充每个通道的值
	for n := 0; n < 2; n++ { // N维度
		for c := 0; c < 3; c++ { // C维度
			for h := 0; h < 4; h++ { // H维度
				for w := 0; w < 4; w++ { // W维度
					// 计算线性索引
					index := n*(3*4*4) + c*(4*4) + h*4 + w
					data[index] = float64(c) // 每个通道的值等于通道号
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
	tolerance := 1e-6
	for i := 0; i < 3; i++ {
		expected := float64(i)
		if math.Abs(mean.Data[i]-expected) > tolerance {
			t.Errorf("通道 %d 均值错误: 期望 %.2f，实际 %.2f", i, expected, mean.Data[i])
		}
	}
}
