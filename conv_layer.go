package torch

import "github.com/Jimmy2099/torch/data_struct/matrix"

// ConvLayer 卷积层实现
type ConvLayer struct {
	inChannels  int
	outChannels int
	kernelSize  int
	stride      int
	pad         int
	weights     *matrix.Matrix
	bias        *matrix.Matrix
	inputCache  *matrix.Matrix // 添加输入缓存用于反向传播
	gradWeights *matrix.Matrix // 权重梯度
	gradBias    *matrix.Matrix // 偏置梯度
	// ... 其他字段如输入缓存等
}

func (c *ConvLayer) SetWeights(data [][]float64) {
	c.weights = matrix.NewMatrixFromSlice(data)
}
func (c *ConvLayer) SetBias(data [][]float64) {
	c.bias = matrix.NewMatrixFromSlice(data)
}
func NewConvLayer(inCh, outCh, kSize, stride, pad int) *ConvLayer {
	// Xavier初始化
	w := matrix.NewRandomMatrix(outCh, inCh*kSize*kSize).MulScalar(1.0 / float64(inCh*kSize*kSize))
	b := matrix.NewMatrix(outCh, 1)
	return &ConvLayer{
		inChannels:  inCh,
		outChannels: outCh,
		kernelSize:  kSize,
		stride:      stride,
		pad:         pad,
		weights:     w,
		bias:        b,
		gradWeights: matrix.NewMatrix(outCh, inCh*kSize*kSize),
		gradBias:    matrix.NewMatrix(outCh, 1),
	}
}

func (c *ConvLayer) Forward(x *matrix.Matrix) *matrix.Matrix {
	// 缓存输入用于反向传播
	c.inputCache = x.Clone()
	// 实现卷积操作
	out := x.Conv2D(c.weights, c.kernelSize, c.stride, c.pad)
	out.Add(c.bias)
	return out
}

// Backward 反向传播
func (c *ConvLayer) Backward(gradOutput *matrix.Matrix) *matrix.Matrix {
	// 计算权重梯度
	c.gradWeights = c.inputCache.Conv2DGradWeights(gradOutput, c.kernelSize, c.stride, c.pad)

	// 计算偏置梯度
	c.gradBias = gradOutput.SumByDim(1) // 沿通道维度求和

	// 计算输入梯度
	gradInput := gradOutput.Conv2DGradInput(c.weights, c.kernelSize, c.stride, c.pad)

	return gradInput
}

// BackwardWithLR 带学习率的反向传播，计算梯度并立即更新参数
func (c *ConvLayer) BackwardWithLR(gradOutput *matrix.Matrix, learningRate float64) *matrix.Matrix {
	// 先计算梯度
	gradInput := c.Backward(gradOutput)

	// 立即更新参数
	c.UpdateParameters(learningRate)

	return gradInput
}

// UpdateParameters 更新参数
func (c *ConvLayer) UpdateParameters(learningRate float64) {
	// 更新权重
	c.weights = c.weights.Sub(c.gradWeights.MulScalar(learningRate))

	// 更新偏置
	c.bias = c.bias.Sub(c.gradBias.MulScalar(learningRate))
}

// ZeroGrad 梯度清零
func (c *ConvLayer) ZeroGrad() {
	c.gradWeights = matrix.NewMatrix(c.outChannels, c.inChannels*c.kernelSize*c.kernelSize)
	c.gradBias = matrix.NewMatrix(c.outChannels, 1)
}
