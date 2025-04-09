package torch

import (
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/viterin/vek/vek32"
	"sync"
)

// TODO add bias=False
// Linear(in_features=2048, out_features=8192, bias=False)

// LinearLayer 实现全连接线性层
type LinearLayer struct {
	InputDim          int
	OutputDim         int
	Weights           *tensor.Tensor
	Bias              *tensor.Tensor
	Input             *tensor.Tensor
	Output            *tensor.Tensor
	GradInput         *tensor.Tensor
	WeightDecay       float32        // L2正则化系数
	Momentum          float32        // 动量系数
	VWeights          *tensor.Tensor // 权重动量
	VBias             *tensor.Tensor // 偏置动量
	WeightsTransposed bool
}

func (l *LinearLayer) GetWeights() *tensor.Tensor {
	return l.Weights
}

func (l *LinearLayer) GetBias() *tensor.Tensor {
	return l.Bias
}

// SetWeights 设置权重
func (l *LinearLayer) SetWeights(data []float32) {
	if len(data) != l.OutputDim*l.InputDim {
		panic("Weights data length mismatch")
	}

	// 创建新数组并拷贝数据
	copiedData := make([]float32, len(data))
	copy(copiedData, data) // 深拷贝

	l.Weights = tensor.NewTensor(copiedData, []int{l.OutputDim, l.InputDim})
}

// SetBias 设置偏置
func (l *LinearLayer) SetBias(data []float32) {
	if len(data) != l.OutputDim {
		panic("bias data length mismatch")
	}

	// 深拷贝偏置数据
	copiedData := make([]float32, len(data))
	copy(copiedData, data)

	l.Bias = tensor.NewTensor(copiedData, []int{l.OutputDim, 1})
}

func (l *LinearLayer) SetWeightsAndShape(data []float32, shape []int) {
	l.SetWeights(data)
	l.Weights.Reshape(shape)
}

func (l *LinearLayer) SetBiasAndShape(data []float32, shape []int) {
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
	weightsData := make([]float32, outputDim*inputDim)
	biasData := make([]float32, outputDim)

	//// Xavier初始化
	//xavierScale := math.Sqrt(2.0 / float32(inputDim))
	//for i := range weightsData {
	//	weightsData[i] = float32(rand.NormFloat64()) * xavierScale
	//}

	return &LinearLayer{
		InputDim:          inputDim,
		OutputDim:         outputDim,
		Weights:           tensor.NewTensor(weightsData, []int{outputDim, inputDim}),
		Bias:              tensor.NewTensor(biasData, []int{outputDim, 1}),
		VWeights:          tensor.NewTensor(make([]float32, outputDim*inputDim), []int{outputDim, inputDim}),
		VBias:             tensor.NewTensor(make([]float32, outputDim), []int{outputDim, 1}),
		WeightDecay:       0.001,
		Momentum:          0.9,
		WeightsTransposed: false,
	}
}

// updateParameters 参数更新逻辑
func (l *LinearLayer) updateParameters(dWeights, dBias *tensor.Tensor, learningRate float32) {
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
	l.VWeights = tensor.NewTensor(make([]float32, l.OutputDim*l.InputDim), []int{l.OutputDim, l.InputDim})
	l.VBias = tensor.NewTensor(make([]float32, l.OutputDim), []int{l.OutputDim, 1})
}

// NumParams 返回参数数量
func (l *LinearLayer) NumParams() int {
	return l.Weights.Shape[0]*l.Weights.Shape[1] + l.Bias.Shape[0]
}

func (l *LinearLayer) Backward(gradOutput *tensor.Tensor, lr float32) *tensor.Tensor {
	// 添加空指针检查
	if l.Input == nil || l.Input.Data == nil {
		panic("前向传播未正确保存输入数据")
	}

	batchSize := gradOutput.Shape[0]
	dWeights := make([]float32, l.InputDim*l.OutputDim)
	dBias := make([]float32, l.OutputDim)
	gradInput := make([]float32, batchSize*l.InputDim)

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
			var sum float32
			for out := 0; out < l.OutputDim; out++ {
				sum += gradOutput.Data[b*l.OutputDim+out] * l.Weights.Data[out*l.InputDim+in]
			}
			gradInput[b*l.InputDim+in] = sum
		}
	}

	// 参数更新（添加动量初始化检查）
	if l.VWeights == nil || l.VBias == nil {
		l.VWeights = tensor.NewTensor(make([]float32, len(l.Weights.Data)), l.Weights.Shape)
		l.VBias = tensor.NewTensor(make([]float32, len(l.Bias.Data)), l.Bias.Shape)
	}

	// 应用动量更新
	for i := range l.Weights.Data {
		l.VWeights.Data[i] = l.Momentum*l.VWeights.Data[i] - lr*(dWeights[i]/float32(batchSize)+l.WeightDecay*l.Weights.Data[i])
		l.Weights.Data[i] += l.VWeights.Data[i]
	}

	for i := range l.Bias.Data {
		l.VBias.Data[i] = l.Momentum*l.VBias.Data[i] - lr*(dBias[i]/float32(batchSize))
		l.Bias.Data[i] += l.VBias.Data[i]
	}

	return tensor.NewTensor(gradInput, []int{batchSize, l.InputDim})
}

func (l *LinearLayer) Forward(x *tensor.Tensor) *tensor.Tensor {
	return l.ForwardSIMD(x)
}

func (l *LinearLayer) ForwardSignalThread(x *tensor.Tensor) *tensor.Tensor {
	// 保存原始形状并检查最后一维
	originalShape := x.ShapeCopy()
	if len(originalShape) == 0 {
		panic("输入张量形状不能为空")
	}
	inputDim := originalShape[len(originalShape)-1]
	if inputDim != l.InputDim {
		panic(fmt.Sprintf("输入维度不匹配：最后一维为%d，期望%d", inputDim, l.InputDim))
	}

	// 展平前部维度
	flattenedBatch := 1
	for _, dim := range originalShape[:len(originalShape)-1] {
		flattenedBatch *= dim
	}
	reshapedX := x.Reshape([]int{flattenedBatch, l.InputDim})

	// 保存输入并执行计算
	l.Input = reshapedX.Clone()
	batchSize := reshapedX.Shape[0]
	outputData := make([]float32, batchSize*l.OutputDim)

	// 矩阵乘法
	for b := 0; b < batchSize; b++ {
		for out := 0; out < l.OutputDim; out++ {
			sum := l.Bias.Data[out]
			for in := 0; in < l.InputDim; in++ {
				sum += l.Input.Data[b*l.InputDim+in] * l.Weights.Data[out*l.InputDim+in]
			}
			outputData[b*l.OutputDim+out] = sum
		}
	}

	// 恢复原始形状
	newShape := make([]int, len(originalShape))
	copy(newShape, originalShape)
	newShape[len(newShape)-1] = l.OutputDim
	output := tensor.NewTensor(outputData, []int{batchSize, l.OutputDim}).Reshape(newShape)
	l.Output = output

	return output
}

func (l *LinearLayer) ForwardMultiThread(x *tensor.Tensor) *tensor.Tensor {
	workers := 15
	originalShape := x.ShapeCopy()
	if len(originalShape) == 0 {
		panic("输入张量形状不能为空")
	}
	inputDim := originalShape[len(originalShape)-1]
	if inputDim != l.InputDim {
		panic(fmt.Sprintf("输入维度不匹配：最后一维为%d，期望%d", inputDim, l.InputDim))
	}

	// 展平前部维度
	flattenedBatch := 1
	for _, dim := range originalShape[:len(originalShape)-1] {
		flattenedBatch *= dim
	}
	reshapedX := x.Reshape([]int{flattenedBatch, l.InputDim})

	l.Input = reshapedX.Clone()
	batchSize := reshapedX.Shape[0]
	outputData := make([]float32, batchSize*l.OutputDim)

	var wg sync.WaitGroup
	chunkSize := (batchSize + workers - 1) / workers

	for w := 0; w < workers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > batchSize {
			end = batchSize
		}
		if start >= end {
			break
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for b := s; b < e; b++ {
				for out := 0; out < l.OutputDim; out++ {
					sum := l.Bias.Data[out]
					ptr := out * l.InputDim
					for in := 0; in < l.InputDim; in++ {
						sum += l.Input.Data[b*l.InputDim+in] * l.Weights.Data[ptr+in]
					}
					outputData[b*l.OutputDim+out] = sum
				}
			}
		}(start, end)
	}
	wg.Wait()

	newShape := make([]int, len(originalShape))
	copy(newShape, originalShape)
	newShape[len(newShape)-1] = l.OutputDim
	return tensor.NewTensor(outputData, []int{batchSize, l.OutputDim}).Reshape(newShape)
}

func (l *LinearLayer) ForwardSIMD(x *tensor.Tensor) *tensor.Tensor {
	// ===== 1. 输入形状验证与展平 =====
	originalShape := x.ShapeCopy()
	if len(originalShape) == 0 {
		panic("输入张量形状不能为空")
	}
	inputDim := originalShape[len(originalShape)-1]
	if inputDim != l.InputDim {
		panic(fmt.Sprintf("输入维度不匹配：最后一维为%d，期望%d", inputDim, l.InputDim))
	}

	// 展平前部维度
	flattenedBatch := 1
	for _, dim := range originalShape[:len(originalShape)-1] {
		flattenedBatch *= dim
	}
	reshapedX := x.Reshape([]int{flattenedBatch, l.InputDim})

	// ===== 2. 核心计算优化 =====
	l.Input = reshapedX.Clone()
	batchSize := reshapedX.Shape[0]

	// 2.1 转置权重矩阵为 [input_dim, output_dim]
	if l.WeightsTransposed == false {
		l.Weights = l.Weights.Transpose()
		l.WeightsTransposed = true
	}

	// 2.2 SIMD矩阵乘法
	matmulResult := vek32.MatMul(
		l.Input.Data,   // [batch, input_dim]
		l.Weights.Data, // [input_dim, output_dim]
		l.InputDim,     // 公共维度（input_dim）
	)

	// 2.3 偏置广播（手动实现）
	broadcastBias := make([]float32, batchSize*l.OutputDim)
	for b := 0; b < batchSize; b++ {
		copy(
			broadcastBias[b*l.OutputDim:(b+1)*l.OutputDim],
			l.Bias.Data,
		)
	}

	// 2.4 SIMD向量加法
	outputData := vek32.Add(matmulResult, broadcastBias)

	// ===== 3. 恢复输出形状 =====
	newShape := make([]int, len(originalShape))
	copy(newShape, originalShape)
	newShape[len(newShape)-1] = l.OutputDim
	output := tensor.NewTensor(outputData, newShape)
	l.Output = output

	return output
}
