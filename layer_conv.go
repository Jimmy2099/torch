package torch

import (
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"math/rand"
)

// ConvLayer 卷积层实现
type ConvLayer struct {
	InChannels  int
	OutChannels int
	KernelSize  int
	Stride      int
	Padding     int
	Weights     *tensor.Tensor
	Bias        *tensor.Tensor
	InputCache  *tensor.Tensor // 添加输入缓存用于反向传播
	GradWeights *tensor.Tensor // 权重梯度
	GradBias    *tensor.Tensor // 偏置梯度
}

// SetWeights 设置权重
func (l *ConvLayer) SetWeights(data []float64) {
	if len(data) != l.OutChannels*l.InChannels*l.KernelSize*l.KernelSize {
		panic("weights data length mismatch")
	}
	l.Weights = tensor.NewTensor(data, []int{l.OutChannels, l.InChannels * l.KernelSize * l.KernelSize})
}

// SetBias 设置偏置
func (l *ConvLayer) SetBias(data []float64) {
	if len(data) != l.OutChannels {
		panic("bias data length mismatch")
	}
	l.Bias = tensor.NewTensor(data, []int{l.OutChannels, 1})
}

func NewConvLayer(inCh, outCh, kSize, stride, pad int) *ConvLayer {
	// 初始化权重和偏置
	weightsData := make([]float64, outCh*inCh*kSize*kSize)
	biasData := make([]float64, outCh)

	// Xavier初始化
	xavierScale := 1.0 / float64(inCh*kSize*kSize)
	for i := range weightsData {
		weightsData[i] = rand.Float64() * xavierScale
	}

	return &ConvLayer{
		InChannels:  inCh,
		OutChannels: outCh,
		KernelSize:  kSize,
		Stride:      stride,
		Padding:     pad,
		Weights:     tensor.NewTensor(weightsData, []int{outCh, inCh * kSize * kSize}),
		Bias:        tensor.NewTensor(biasData, []int{outCh, 1}),
		GradWeights: tensor.NewTensor(make([]float64, outCh*inCh*kSize*kSize), []int{outCh, inCh * kSize * kSize}),
		GradBias:    tensor.NewTensor(make([]float64, outCh), []int{outCh, 1}),
	}
}

func (c *ConvLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	// 保存输入用于反向传播
	c.InputCache = x.Clone()

	// 执行卷积操作 (修改为接收两个返回值)
	convOut, err := x.Conv2D(c.Weights, c.KernelSize, c.Stride, c.Padding)
	if err != nil {
		panic(err)
	}

	// 广播偏置到与convOut相同的维度 (修改为接收两个返回值)
	biasBroadcast := c.Bias.Repeat(1, convOut.Shape[1])

	// 添加偏置 (修改为接收两个返回值)
	result := convOut.Add(biasBroadcast)

	return result
}
func (c *ConvLayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// 计算权重梯度
	gradWeights, err := c.InputCache.Conv2DGradWeights(gradOutput, c.KernelSize, c.Stride, c.Padding)
	if err != nil {
		panic(err)
	}
	c.GradWeights = gradWeights

	// 计算偏置梯度
	gradBias := gradOutput.SumByDim(1) // 沿通道维度求和
	c.GradBias = gradBias

	// 计算输入梯度
	gradInput, err := gradOutput.Conv2DGradInput(c.Weights, c.KernelSize, c.Stride, c.Padding)
	if err != nil {
		panic(err)
	}

	return gradInput
}

func (c *ConvLayer) BackwardWithLR(gradOutput *tensor.Tensor, learningRate float64) *tensor.Tensor {
	// 先计算梯度
	gradInput := c.Backward(gradOutput)

	// 立即更新参数
	c.UpdateParameters(learningRate)

	return gradInput
}

func (c *ConvLayer) UpdateParameters(learningRate float64) {
	// 更新权重
	c.Weights = c.Weights.Sub(c.GradWeights.MulScalar(learningRate))

	// 更新偏置
	c.Bias = c.Bias.Sub(c.GradBias.MulScalar(learningRate))
}

func (c *ConvLayer) ZeroGrad() {
	c.GradWeights = tensor.NewTensor(
		make([]float64, c.OutChannels*c.InChannels*c.KernelSize*c.KernelSize),
		[]int{c.OutChannels, c.InChannels * c.KernelSize * c.KernelSize},
	)
	c.GradBias = tensor.NewTensor(
		make([]float64, c.OutChannels),
		[]int{c.OutChannels, 1},
	)
}
