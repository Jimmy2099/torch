package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	math "github.com/chewxy/math32"
)

// Sigmoid layer struct
type SigmoidLayer struct {
	// Sigmoid 层不需要存储额外的参数，因此这里没有额外的字段
}

func NewSigmoidLayer() *SigmoidLayer {
	return &SigmoidLayer{}
}

// Sigmoid 激活函数
func Sigmoid(x float32) float32 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Sigmoid 导数
func SigmoidDerivative(x float32) float32 {
	s := Sigmoid(x)
	return s * (1.0 - s)
}

// Forward 方法：接受输入张量并应用 Sigmoid 激活函数
func (s *SigmoidLayer) Forward(input *tensor.Tensor) *tensor.Tensor {
	// 获取输入张量的形状
	shape := input.Shape

	// 创建一个与输入张量形状相同的输出张量，初始值为全零
	outputData := make([]float32, len(input.Data)) // 创建一个和输入相同大小的数据切片
	for i := 0; i < len(input.Data); i++ {
		outputData[i] = Sigmoid(input.Data[i]) // 应用 Sigmoid 激活
	}

	// 返回新的张量
	return tensor.NewTensor(outputData, shape)
}

// Backward 方法：计算 Sigmoid 层的梯度
func (s *SigmoidLayer) Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	// 获取梯度输出的形状
	shape := gradOutput.Shape

	// 创建一个新的张量用于存储反向传播的梯度
	gradInputData := make([]float32, len(gradOutput.Data))
	for i := 0; i < len(gradOutput.Data); i++ {
		gradInputData[i] = gradOutput.Data[i] * SigmoidDerivative(gradOutput.Data[i]) // 根据 Sigmoid 导数计算梯度
	}

	// 返回新的张量
	return tensor.NewTensor(gradInputData, shape)
}

// ZeroGrad 方法：清空梯度（对于 Sigmoid 层没有可训练的参数，所以实现为空）
func (s *SigmoidLayer) ZeroGrad() {
	// Sigmoid 层没有可训练的参数，因此此方法无需做任何事情
}

// Parameters 方法：返回 Sigmoid 层的可训练参数（Sigmoid 激活函数没有可训练的参数）
func (s *SigmoidLayer) Parameters() []*tensor.Tensor {
	// Sigmoid 层没有参数，因此返回空切片
	return []*tensor.Tensor{}
}
