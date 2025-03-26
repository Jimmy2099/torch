package torch

import (
	"github.com/Jimmy2099/torch/data_struct/matrix"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"math"
	"math/rand"
)

// LinearLayer 实现全连接线性层
type LinearLayer struct {
	InputDim    int
	OutputDim   int
	Weights     *tensor.Tensor
	Bias        *tensor.Tensor
	Input       *tensor.Tensor
	Output      *tensor.Tensor
	GradInput   *tensor.Tensor
	WeightDecay float64        // L2正则化系数
	Momentum    float64        // 动量系数
	VWeights    *tensor.Tensor // 权重动量
	VBias       *tensor.Tensor // 偏置动量
}

// SetWeights 设置权重
func (l *LinearLayer) SetWeights(data []float64) {
    if len(data) != l.OutputDim*l.InputDim {
        panic("weights data length mismatch")
    }
    l.Weights = tensor.NewTensor(data, []int{l.OutputDim, l.InputDim})
}

// SetBias 设置偏置
func (l *LinearLayer) SetBias(data []float64) {
    if len(data) != l.OutputDim {
        panic("bias data length mismatch")
    }
    l.Bias = tensor.NewTensor(data, []int{l.OutputDim, 1})
}


// Parameters 返回所有可训练参数
func (l *LinearLayer) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{l.Weights, l.Bias}
}

// NewLinearLayer 创建新的线性层
func NewLinearLayer(inputDim, outputDim int) *LinearLayer {
	// 初始化权重和偏置
	weightsData := make([]float64, outputDim*inputDim)
	biasData := make([]float64, outputDim)

	// Xavier初始化
	xavierScale := math.Sqrt(2.0 / float64(inputDim))
	for i := range weightsData {
		weightsData[i] = rand.NormFloat64() * xavierScale
	}

	return &LinearLayer{
		InputDim:    inputDim,
		OutputDim:   outputDim,
		Weights:     tensor.NewTensor(weightsData, []int{outputDim, inputDim}),
		Bias:        tensor.NewTensor(biasData, []int{outputDim, 1}),
		VWeights:    tensor.NewTensor(make([]float64, outputDim*inputDim), []int{outputDim, inputDim}),
		VBias:       tensor.NewTensor(make([]float64, outputDim), []int{outputDim, 1}),
		WeightDecay: 0.001,
		Momentum:    0.9,
	}
}

// updateParameters 参数更新逻辑
func (l *LinearLayer) updateParameters(dWeights, dBias *tensor.Tensor, learningRate float64) {
	// 更新权重
	for i := 0; i < l.Weights.Shape[0]; i++ {
		for j := 0; j < l.Weights.Shape[1]; j++ {
			// L2正则化梯度
			regGrad := l.WeightDecay * l.Weights.Data[i*l.Weights.Shape[1]+j]
			// 动量更新
			l.VWeights.Data[i*l.VWeights.Shape[1]+j] = l.Momentum*l.VWeights.Data[i*l.VWeights.Shape[1]+j] -
				learningRate*(dWeights.Data[i*dWeights.Shape[1]+j]+regGrad)
			l.Weights.Data[i*l.Weights.Shape[1]+j] += l.VWeights.Data[i*l.VWeights.Shape[1]+j]
		}
	}

	// 更新偏置
	for i := 0; i < l.Bias.Shape[0]; i++ {
		l.VBias.Data[i] = l.Momentum*l.VBias.Data[i] - learningRate*dBias.Data[i]
		l.Bias.Data[i] += l.VBias.Data[i]
	}
}

// ZeroGrad 梯度清零
func (l *LinearLayer) ZeroGrad() {
	l.GradInput = nil
	l.VWeights = tensor.NewTensor(make([]float64, l.OutputDim*l.InputDim), []int{l.OutputDim, l.InputDim})
	l.VBias = tensor.NewTensor(make([]float64, l.OutputDim), []int{l.OutputDim, 1})
}

// XavierInit Xavier初始化
func (l *LinearLayer) XavierInit() {
	fanIn := float64(l.InputDim)
	scale := math.Sqrt(2.0 / fanIn)
	for i := 0; i < l.Weights.Shape[0]; i++ {
		for j := 0; j < l.Weights.Shape[1]; j++ {
			l.Weights.Data[i*l.Weights.Shape[1]+j] = rand.NormFloat64() * scale
		}
	}
}

// NumParams 返回参数数量
func (l *LinearLayer) NumParams() int {
	return l.Weights.Shape[0]*l.Weights.Shape[1] + l.Bias.Shape[0]
}

func (l *LinearLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	// 检查输入维度
	if len(x.Shape) != 2 || x.Shape[1] != l.InputDim {
		panic("input dimension mismatch")
	}

	// 矩阵乘法: Wx + b
	weightsMatrix := &matrix.Matrix{
		Data: make([][]float64, l.Weights.Shape[0]),
		Rows: l.Weights.Shape[0],
		Cols: l.Weights.Shape[1],
	}
	for i := 0; i < l.Weights.Shape[0]; i++ {
		weightsMatrix.Data[i] = l.Weights.Data[i*l.Weights.Shape[1] : (i+1)*l.Weights.Shape[1]]
	}

	inputMatrix := &matrix.Matrix{
		Data: make([][]float64, x.Shape[0]),
		Rows: x.Shape[0],
		Cols: x.Shape[1],
	}
	for i := 0; i < x.Shape[0]; i++ {
		inputMatrix.Data[i] = x.Data[i*x.Shape[1] : (i+1)*x.Shape[1]]
	}

	// 执行矩阵乘法
	output := weightsMatrix.Dot(inputMatrix)

	// 添加偏置
	for i := 0; i < output.Rows; i++ {
		for j := 0; j < output.Cols; j++ {
			output.Data[i][j] += l.Bias.Data[i]
		}
	}

	// 保存输入用于反向传播
	l.Input = x

	// 将结果转换为Tensor
	flatOutput := make([]float64, 0, output.Rows*output.Cols)
	for _, row := range output.Data {
		flatOutput = append(flatOutput, row...)
	}

	l.Output = tensor.NewTensor(flatOutput, []int{output.Rows, output.Cols})
	return l.Output
}
