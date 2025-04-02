package testing

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"testing"
)

func TestGetLayerTestResult(t *testing.T) {

	t.Run("linear layer", func(t *testing.T) {
		script := fmt.Sprintf(`torch.nn.Linear(in_features=%d, out_features=%d)`, 64, 64)
		t1 := tensor.Random([]int{64, 64}, -100, 100)
		weights := tensor.Random([]int{64, 64}, -100, 100)
		bias := tensor.Random([]int{64}, -100, 100)

		l := torch.NewLinearLayer(64, 64)
		l.SetWeightsAndShape(weights.Data, weights.Shape)
		l.SetBiasAndShape(bias.Data, bias.Shape)

		result := GetLayerTestResult(script, l, t1)
		t2 := l.Forward(t1)

		if !result.EqualFloat32(t2) {
			t.Errorf("Element-wise multiplication failed:\nExpected:\n%v\nGot:\n%v", t2, result)
		}
	})

	t.Run("linear layer1", func(t *testing.T) {
		// 参数化线性层测试
		testCases := []struct {
			name        string
			inFeatures  int
			outFeatures int
			inputShape  []int // 输入张量形状
			weightShape []int // 权重张量形状
			biasShape   []int // 偏置张量形状
		}{
			{
				name:        "64x64 linear layer",
				inFeatures:  64,
				outFeatures: 64,
				inputShape:  []int{64, 64}, // [batch, in_features]
				weightShape: []int{64, 64}, // [out_features, in_features]
				biasShape:   []int{64},
			},
			{
				name:        "128x64 linear layer",
				inFeatures:  128,
				outFeatures: 64,
				inputShape:  []int{32, 128},
				weightShape: []int{64, 128},
				biasShape:   []int{64},
			},
			{
				name:        "256x128 linear layer",
				inFeatures:  256,
				outFeatures: 128,
				inputShape:  []int{16, 256},
				weightShape: []int{128, 256},
				biasShape:   []int{128},
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				// 生成PyTorch脚本
				script := fmt.Sprintf(
					`torch.nn.Linear(in_features=%d, out_features=%d)`,
					tc.inFeatures,
					tc.outFeatures,
				)

				// 生成测试数据
				t1 := tensor.Random(tc.inputShape, -100, 100)
				weights := tensor.Random(tc.weightShape, -100, 100)
				bias := tensor.Random(tc.biasShape, -100, 100)

				// 初始化层
				l := torch.NewLinearLayer(tc.inFeatures, tc.outFeatures)
				l.SetWeightsAndShape(weights.Data, weights.Shape)
				l.SetBiasAndShape(bias.Data, bias.Shape)

				// 获取结果并验证
				result := GetLayerTestResult(script, l, t1)
				t2 := l.Forward(t1)

				if !result.EqualFloat32(t2) {
					t.Errorf("%s failed:\nInput shape: %v\nExpected:\n%v\nGot:\n%v",
						tc.name,
						tc.inputShape,
						t2,
						result)
				}
			})
		}
	})

	// 卷积层测试
	t.Run("conv2d layer", func(t *testing.T) {
		inChannels := 3
		outChannels := 64
		kernelSize := []int{5, 5}
		stride := []int{2, 2}
		padding := []int{2, 2}

		script := fmt.Sprintf(
			`torch.nn.Conv2d(in_channels=%d, out_channels=%d, kernel_size=[%v,%v], stride=[%v,%v], padding=[%v,%v])`,
			inChannels, outChannels, kernelSize[0], kernelSize[1], stride[0], stride[1], padding[0], padding[1],
		)
		// 输入形状 [batch, channels, height, width]
		input := tensor.Random([]int{1, 3, 64, 64}, -1, 1)
		weights := tensor.Random([]int{64, 3, 5, 5}, -1, 1)
		bias := tensor.Random([]int{64}, -1, 1)

		l := torch.NewConvLayer(inChannels, outChannels, kernelSize[0], stride[0], padding[0])
		l.SetWeightsAndShape(weights.Data, weights.Shape) // 修正为设置权重
		l.SetBiasAndShape(bias.Data, bias.Shape)          // 设置偏置

		result := GetLayerTestResult(script, l, input)
		expected := l.Forward(input)

		if !result.EqualFloat32(expected) {
			t.Errorf("Conv2d layer failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

	// ReLU激活函数测试
	t.Run("relu activation", func(t *testing.T) {
		script := `torch.nn.ReLU()`
		input := tensor.Random([]int{64, 64}, -100, 100)

		l := torch.NewReLULayer()
		result := GetLayerTestResult(script, l, input)
		expected := l.Forward(input)

		// 验证所有负值变为0，正值保持不变
		if !result.EqualFloat32(expected) {
			t.Errorf("ReLU activation failed:\nExpected:\n%v\nGot:\n%v", expected, result)
		}
	})

}
