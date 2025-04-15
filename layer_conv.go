package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math/rand"
)

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

func (l *ConvLayer) SetWeights(data []float32) {
	if len(data) != l.OutChannels*l.InChannels*l.KernelSize*l.KernelSize {
		panic("Weights data length mismatch")
	}

	copiedData := make([]float32, len(data))
	copy(copiedData, data) // 深拷贝

	l.Weights = tensor.NewTensor(copiedData, []int{l.OutChannels, l.InChannels * l.KernelSize * l.KernelSize})
}

func (l *ConvLayer) SetBias(data []float32) {
	if len(data) != l.OutChannels {
		panic("bias data length mismatch")
	}

	copiedData := make([]float32, len(data))
	copy(copiedData, data)

	l.Bias = tensor.NewTensor(copiedData, []int{l.OutChannels, 1})
}

func (l *ConvLayer) SetWeightsAndShape(data []float32, shape []int) {
	l.SetWeights(data)
	l.Weights.Reshape(shape)
}

func (l *ConvLayer) SetBiasAndShape(data []float32, shape []int) {
	l.SetBias(data)
	l.Bias.Reshape(shape)
}

func NewConvLayer(inCh, outCh, kSize, stride, pad int) *ConvLayer {
	weightsData := make([]float32, outCh*inCh*kSize*kSize)
	biasData := make([]float32, outCh)

	xavierScale := 1.0 / float32(inCh*kSize*kSize)
	for i := range weightsData {
		weightsData[i] = float32(rand.Float32()) * xavierScale
	}

	return &ConvLayer{
		InChannels:  inCh,
		OutChannels: outCh,
		KernelSize:  kSize,
		Stride:      stride,
		Padding:     pad,
		Weights:     tensor.NewTensor(weightsData, []int{outCh, inCh * kSize * kSize}),
		Bias:        tensor.NewTensor(biasData, []int{outCh, 1}),
		GradWeights: tensor.NewTensor(make([]float32, outCh*inCh*kSize*kSize), []int{outCh, inCh * kSize * kSize}),
		GradBias:    tensor.NewTensor(make([]float32, outCh), []int{outCh, 1}),
	}
}

func (c *ConvLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	c.InputCache = x.Clone()

	convOut, err := x.Conv2D(c.Weights, c.KernelSize, c.Stride, c.Padding)
	if err != nil {
		panic(err)
	}

	var biasBroadcast *tensor.Tensor
	switch len(convOut.Shape) {
	case 1: // 1D输出 (out_channels,)
		biasBroadcast = c.Bias
	case 2: // 2D输出 (out_channels, out_size)
		biasBroadcast = c.Bias.Repeat(1, convOut.Shape[1])
	case 4: // 4D输出 (batch, out_channels, height, width)
		biasBroadcast = c.Bias.Reshape([]int{1, c.OutChannels, 1, 1})
		biasBroadcast = biasBroadcast.Expand([]int{
			convOut.Shape[0], // batch
			c.OutChannels,    // channels
			convOut.Shape[2], // height
			convOut.Shape[3], // width
		})
	default:
		panic("unsupported output shape from convolution")
	}

	result := convOut.Add(biasBroadcast)

	return result
}

func (c *ConvLayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	gradWeights, err := c.InputCache.Conv2DGradWeights(gradOutput, c.KernelSize, c.Stride, c.Padding)
	if err != nil {
		panic(err)
	}
	c.GradWeights = gradWeights

	gradBias := gradOutput.SumByDim(0) // 沿批处理维度求和
	gradBias = gradBias.SumByDim(1)    // 沿空间维度求和
	c.GradBias = gradBias

	gradInput, err := gradOutput.Conv2DGradInput(c.Weights, c.KernelSize, c.Stride, c.Padding)
	if err != nil {
		panic(err)
	}

	return gradInput
}

func (c *ConvLayer) BackwardWithLR(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	gradInput := c.Backward(gradOutput)

	c.UpdateParameters(learningRate)

	return gradInput
}

func (c *ConvLayer) UpdateParameters(learningRate float32) {
	c.Weights = c.Weights.Sub(c.GradWeights.MulScalar(learningRate))

	c.Bias = c.Bias.Sub(c.GradBias.MulScalar(learningRate))
}

func (c *ConvLayer) ZeroGrad() {
	c.GradWeights = tensor.NewTensor(
		make([]float32, c.OutChannels*c.InChannels*c.KernelSize*c.KernelSize),
		[]int{c.OutChannels, c.InChannels * c.KernelSize * c.KernelSize},
	)
	c.GradBias = tensor.NewTensor(
		make([]float32, c.OutChannels),
		[]int{c.OutChannels, 1},
	)
}
