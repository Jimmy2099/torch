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
	InputCache  *tensor.Tensor
	GradWeights *tensor.Tensor
	GradBias    *tensor.Tensor
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
	copy(copiedData, data)

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

	convOut, err := x.Conv2D(c.Weights, c.KernelSize, c.Stride, c.Padding, c.Padding)
	if err != nil {
		panic(err)
	}

	var biasBroadcast *tensor.Tensor
	switch len(convOut.GetShape()) {
	case 1:
		biasBroadcast = c.Bias
	case 2:
		biasBroadcast = c.Bias.Repeat(1, convOut.GetShape()[1])
	case 4:
		biasBroadcast = c.Bias.Reshape([]int{1, c.OutChannels, 1, 1})
		biasBroadcast = biasBroadcast.Expand([]int{
			convOut.GetShape()[0],
			c.OutChannels,
			convOut.GetShape()[2],
			convOut.GetShape()[3],
		})
	default:
		panic("unsupported output shape from convolution")
	}

	result := convOut.Add(biasBroadcast)

	return result
}

func (c *ConvLayer) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	return nil
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
