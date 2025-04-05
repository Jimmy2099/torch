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

func (l *ConvLayer) GetWeights() *tensor.Tensor {
	return l.Weights
}

func (l *ConvLayer) GetBias() *tensor.Tensor {
	return l.Bias
}

// SetWeights 设置权重
func (l *ConvLayer) SetWeights(data []float64) {
	if len(data) != l.OutChannels*l.InChannels*l.KernelSize*l.KernelSize {
		panic("Weights data length mismatch")
	}

	// 创建新数组并拷贝数据
	copiedData := make([]float64, len(data))
	copy(copiedData, data) // 深拷贝

	l.Weights = tensor.NewTensor(copiedData, []int{l.OutChannels, l.InChannels * l.KernelSize * l.KernelSize})
}

// SetBias 设置偏置
func (l *ConvLayer) SetBias(data []float64) {
	if len(data) != l.OutChannels {
		panic("bias data length mismatch")
	}

	// 深拷贝偏置数据
	copiedData := make([]float64, len(data))
	copy(copiedData, data)

	l.Bias = tensor.NewTensor(copiedData, []int{l.OutChannels, 1})
}

func (l *ConvLayer) SetWeightsAndShape(data []float64, shape []int) {
	l.SetWeights(data)
	l.Weights.Reshape(shape)
}

func (l *ConvLayer) SetBiasAndShape(data []float64, shape []int) {
	l.SetBias(data)
	l.Bias.Reshape(shape)
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

	// 执行卷积操作
	convOut, err := x.Conv2D(c.Weights, c.KernelSize, c.Stride, c.Padding)
	if err != nil {
		panic(err)
	}

	// 处理偏置广播 - 改进版
	var biasBroadcast *tensor.Tensor
	switch len(convOut.Shape) {
	case 1: // 1D输出 (out_channels,)
		biasBroadcast = c.Bias
	case 2: // 2D输出 (out_channels, out_size)
		// 将偏置从(out_channels,1)扩展到(out_channels,out_size)
		biasBroadcast = c.Bias.Repeat(1, convOut.Shape[1])
	case 4: // 4D输出 (batch, out_channels, height, width)
		// 改进的4D广播方式 - 先reshape再expand
		biasBroadcast = c.Bias.Reshape([]int{1, c.OutChannels, 1, 1})
		// 使用Expand代替Repeat
		biasBroadcast = biasBroadcast.Expand([]int{
			convOut.Shape[0], // batch
			c.OutChannels,    // channels
			convOut.Shape[2], // height
			convOut.Shape[3], // width
		})
	default:
		panic("unsupported output shape from convolution")
	}

	// 添加偏置
	result := convOut.Add(biasBroadcast)

	return result
}

func (c *ConvLayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	// 计算权重梯度 (支持批处理)
	gradWeights, err := c.InputCache.Conv2DGradWeights(gradOutput, c.KernelSize, c.Stride, c.Padding)
	if err != nil {
		panic(err)
	}
	c.GradWeights = gradWeights

	// 计算偏置梯度 (支持批处理)
	gradBias := gradOutput.SumByDim(0) // 沿批处理维度求和
	gradBias = gradBias.SumByDim(1)    // 沿空间维度求和
	c.GradBias = gradBias

	// 计算输入梯度 (支持批处理)
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
