package torch

import (
	"github.com/Jimmy2099/torch/data_struct/matrix"
	"math"
	"math/rand"
)

// LinearLayer implements a fully connected linear layer
type LinearLayer struct {
	InputDim    int
	OutputDim   int
	Weights     *matrix.Matrix
	Bias        *matrix.Matrix
	Input       *matrix.Matrix
	Output      *matrix.Matrix
	GradInput   *matrix.Matrix
	WeightDecay float64        // L2正则化系数
	Momentum    float64        // 动量系数
	VWeights    *matrix.Matrix // 权重动量
	VBias       *matrix.Matrix // 偏置动量
}

func (l *LinearLayer) Parameters() []*matrix.Matrix {
	//TODO implement me
	panic("implement me")
}

// NewLinearLayer creates a new linear layer with random weights
func NewLinearLayer(inputDim, outputDim int) *LinearLayer {
	weights := matrix.NewRandomMatrix(outputDim, inputDim)
	bias := matrix.NewMatrix(outputDim, 1)

	// Initialize bias with small random values
	for i := 0; i < outputDim; i++ {
		bias.Data[i][0] = rand.Float64()*0.2 - 0.1
	}

	return &LinearLayer{
		InputDim:  inputDim,
		OutputDim: outputDim,
		Weights:   weights,
		Bias:      bias,
		VWeights:  matrix.NewMatrix(outputDim, inputDim),  // 初始化权重动量
		VBias:     matrix.NewMatrix(outputDim, 1),         // 初始化偏置动量
		WeightDecay: 0.001,  // 添加默认L2正则化系数
		Momentum:    0.9,    // 添加默认动量系数
	}
}

// Forward performs forward pass through the linear layer
func (l *LinearLayer) Forward(input *matrix.Matrix) *matrix.Matrix {
	l.Input = input
	// Y = W * X + b
	l.Output = l.Weights.Multiply(input)

	// Add bias to each column
	for i := 0; i < l.OutputDim; i++ {
		for j := 0; j < input.Cols; j++ {
			l.Output.Data[i][j] += l.Bias.Data[i][0]
		}
	}

	return l.Output
}

// Backward performs backward pass through the linear layer
func (l *LinearLayer) Backward(gradOutput *matrix.Matrix, learningRate float64) *matrix.Matrix {
	// Compute gradients
	inputT := l.Input.Transpose()

	// Gradient of weights: dW = dY * X^T
	dWeights := gradOutput.Multiply(inputT)

	// Gradient of bias: db = sum(dY, dim=1)
	dBias := matrix.NewMatrix(l.OutputDim, 1)
	for i := 0; i < l.OutputDim; i++ {
		sum := 0.0
		for j := 0; j < gradOutput.Cols; j++ {
			sum += gradOutput.Data[i][j]
		}
		dBias.Data[i][0] = sum
	}

	// Gradient of input: dX = W^T * dY
	weightsT := l.Weights.Transpose()
	l.GradInput = weightsT.Multiply(gradOutput)

	// Update weights and bias
	for i := 0; i < l.Weights.Rows; i++ {
		for j := 0; j < l.Weights.Cols; j++ {
			// L2正则化梯度
			regGrad := l.WeightDecay * l.Weights.Data[i][j]
			// 动量更新
			l.VWeights.Data[i][j] = l.Momentum*l.VWeights.Data[i][j] -
				learningRate*(dWeights.Data[i][j]+regGrad)
			l.Weights.Data[i][j] += l.VWeights.Data[i][j]
		}
	}

	for i := 0; i < l.Bias.Rows; i++ {
		l.VBias.Data[i][0] = l.Momentum*l.VBias.Data[i][0] -
			learningRate*dBias.Data[i][0]
		l.Bias.Data[i][0] += l.VBias.Data[i][0]
	}

	return l.GradInput
}

func (l *LinearLayer) ZeroGrad() {
	// Reset gradients
	l.GradInput = nil
	// 重置动量
	l.VWeights = matrix.NewMatrix(l.OutputDim, l.InputDim)
	l.VBias = matrix.NewMatrix(l.OutputDim, 1)
}

// CrossEntropyLoss 交叉熵损失函数
func CrossEntropyLoss(pred, target *matrix.Matrix) float64 {
	// 实现softmax交叉熵
	exp := pred.Apply(math.Exp)
	sum := exp.Sum()
	prob := exp.DivScalar(sum)
	return -prob.Log().Multiply(target).Mean()
}

// XavierInit 新增初始化方法
func (l *LinearLayer) XavierInit() {
	fanIn := float64(l.InputDim)
	scale := math.Sqrt(2.0 / fanIn)
	for i := 0; i < l.Weights.Rows; i++ {
		for j := 0; j < l.Weights.Cols; j++ {
			l.Weights.Data[i][j] = rand.NormFloat64() * scale
		}
	}
}

// 新增参数保存方法
func (l *LinearLayer) SaveParams(path string) error {
	// 保存权重和偏置到文件
	// ... 实现文件保存逻辑 ...
	return nil
}

// 新增参数加载方法
func (l *LinearLayer) LoadParams(path string) error {
	// 从文件加载权重和偏置
	// ... 实现文件加载逻辑 ...
	return nil
}

// 新增方法：获取参数数量
func (l *LinearLayer) NumParams() int {
	return l.Weights.Rows*l.Weights.Cols + l.Bias.Rows
}
