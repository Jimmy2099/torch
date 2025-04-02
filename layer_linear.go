package torch

import (
	"fmt"
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

func (l *LinearLayer) GetWeights() *tensor.Tensor {
	return l.Weights
}

func (l *LinearLayer) GetBias() *tensor.Tensor {
	return l.Bias
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

func (l *LinearLayer) SetWeightsAndShape(data []float64, shape []int) {
	l.SetWeights(data)
	l.Weights.Reshape(shape)
}

func (l *LinearLayer) SetBiasAndShape(data []float64, shape []int) {
	l.SetBias(data)
	l.Bias.Reshape(shape)
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

func (l *LinearLayer) Backward(gradOutput *tensor.Tensor, lr float64) *tensor.Tensor {
	// 添加空指针检查
	if l.Input == nil || l.Input.Data == nil {
		panic("前向传播未正确保存输入数据")
	}

	batchSize := gradOutput.Shape[0]
	dWeights := make([]float64, l.InputDim*l.OutputDim)
	dBias := make([]float64, l.OutputDim)
	gradInput := make([]float64, batchSize*l.InputDim)

	// 计算权重梯度
	for b := 0; b < batchSize; b++ {
		for out := 0; out < l.OutputDim; out++ {
			grad := gradOutput.Data[b*l.OutputDim+out]
			// 添加索引范围检查
			if out >= l.OutputDim || b >= batchSize {
				panic("梯度索引越界")
			}
			for in := 0; in < l.InputDim; in++ {
				dWeights[out*l.InputDim+in] += l.Input.Data[b*l.InputDim+in] * grad
			}
			dBias[out] += grad
		}
	}

	// 计算输入梯度
	for b := 0; b < batchSize; b++ {
		for in := 0; in < l.InputDim; in++ {
			sum := 0.0
			for out := 0; out < l.OutputDim; out++ {
				sum += gradOutput.Data[b*l.OutputDim+out] * l.Weights.Data[out*l.InputDim+in]
			}
			gradInput[b*l.InputDim+in] = sum
		}
	}

	// 参数更新（添加动量初始化检查）
	if l.VWeights == nil || l.VBias == nil {
		l.VWeights = tensor.NewTensor(make([]float64, len(l.Weights.Data)), l.Weights.Shape)
		l.VBias = tensor.NewTensor(make([]float64, len(l.Bias.Data)), l.Bias.Shape)
	}

	// 应用动量更新
	for i := range l.Weights.Data {
		l.VWeights.Data[i] = l.Momentum*l.VWeights.Data[i] - lr*(dWeights[i]/float64(batchSize)+l.WeightDecay*l.Weights.Data[i])
		l.Weights.Data[i] += l.VWeights.Data[i]
	}

	for i := range l.Bias.Data {
		l.VBias.Data[i] = l.Momentum*l.VBias.Data[i] - lr*(dBias[i]/float64(batchSize))
		l.Bias.Data[i] += l.VBias.Data[i]
	}

	return tensor.NewTensor(gradInput, []int{batchSize, l.InputDim})
}

func (l *LinearLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	// 检查输入维度
	if len(x.Shape) == 1 {
		// 如果是一维输入，转换为二维 [1, n]
		x = tensor.NewTensor(x.Data, []int{1, x.Shape[0]})
	} else if len(x.Shape) != 2 {
		panic(fmt.Sprintf("输入必须为 1D 或 2D 张量，实际 %v", x.Shape))
	}

	// 确保输入的第二维与 InputDim 匹配（关键兼容性修复）
	if x.Shape[1] != l.InputDim {
		// 尝试转置输入矩阵
		if x.Shape[0] == l.InputDim {
			x = tensor.Transpose(x)
		} else {
			panic(fmt.Sprintf("输入维度不匹配：实际 %v，期望 [?,%d]", x.Shape, l.InputDim))
		}
	}

	batchSize := x.Shape[0]
	outputData := make([]float64, batchSize*l.OutputDim)

	// 保存转置处理后的输入用于反向传播
	l.Input = x.Clone()

	// 手动实现矩阵乘法（保持与 Forward1 逻辑一致）
	for b := 0; b < batchSize; b++ {
		for out := 0; out < l.OutputDim; out++ {
			sum := l.Bias.Data[out]
			for in := 0; in < l.InputDim; in++ {
				// 注意：权重矩阵形状应为 [OutputDim, InputDim]
				sum += l.Input.Data[b*l.InputDim+in] * l.Weights.Data[out*l.InputDim+in]
			}
			outputData[b*l.OutputDim+out] = sum
		}
	}

	l.Output = tensor.NewTensor(outputData, []int{batchSize, l.OutputDim})
	return l.Output
}
