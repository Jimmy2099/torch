package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"math"
	"math/rand"
)

type Embedding struct {
	Weights     *tensor.Tensor
	GradWeights *tensor.Tensor // 独立维护梯度张量
	VocabSize   int
	EmbDim      int
	LastIndices []int // 缓存前向传播的整数索引
}

func (e *Embedding) SetBiasAndShape(data []float64, shape []int) {
}

func (e *Embedding) GetWeights() *tensor.Tensor {
	return e.Weights
}

func (e *Embedding) GetBias() *tensor.Tensor {
	return &tensor.Tensor{
		Data:  make([]float64, 0),
		Shape: make([]int, 0),
	}
}

func NewEmbedding(vocabSize, embDim int) *Embedding {
	data := make([]float64, vocabSize*embDim)
	scale := math.Sqrt(2.0 / float64(embDim))
	for i := range data {
		data[i] = rand.NormFloat64() * scale
	}

	return &Embedding{
		Weights:     tensor.NewTensor(data, []int{vocabSize, embDim}),
		GradWeights: tensor.NewTensor(make([]float64, vocabSize*embDim), []int{vocabSize, embDim}),
		VocabSize:   vocabSize,
		EmbDim:      embDim,
	}
}

func (e *Embedding) Forward(indices *tensor.Tensor) *tensor.Tensor {
	if len(indices.Shape) != 2 {
		panic(fmt.Sprintf("Embedding expects 2D input, got %dD", len(indices.Shape)))
	}

	// 直接使用Data字段（已经是[]float64类型）
	floatIndices := indices.Data
	intIndices := make([]int, len(floatIndices))
	for i, v := range floatIndices {
		if v != math.Trunc(v) {
			panic(fmt.Sprintf("Non-integer index at position %d: %v", i, v))
		}
		intIndices[i] = int(v)
	}
	e.LastIndices = intIndices // 缓存整数索引

	batchSize, seqLen := indices.Shape[0], indices.Shape[1]
	outputShape := []int{batchSize, seqLen, e.EmbDim}
	outputData := make([]float64, batchSize*seqLen*e.EmbDim)

	weightsData := e.Weights.Data
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			idx := intIndices[b*seqLen+s]
			if idx < 0 || idx >= e.VocabSize {
				panic(fmt.Sprintf("Index %d out of range [0, %d)", idx, e.VocabSize))
			}

			srcStart := idx * e.EmbDim
			dstStart := (b*seqLen + s) * e.EmbDim
			copy(outputData[dstStart:dstStart+e.EmbDim], weightsData[srcStart:srcStart+e.EmbDim])
		}
	}

	return tensor.NewTensor(outputData, outputShape)
}

func (e *Embedding) Backward(gradOutput *tensor.Tensor, learningRate float64) *tensor.Tensor {
	gradData := gradOutput.Data
	batchSize := gradOutput.Shape[0]
	seqLen := gradOutput.Shape[1]

	// 重置梯度
	for i := range e.GradWeights.Data {
		e.GradWeights.Data[i] = 0
	}

	// 累积梯度
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			idx := e.LastIndices[b*seqLen+s]
			if idx < 0 || idx >= e.VocabSize {
				continue
			}

			gradStart := (b*seqLen + s) * e.EmbDim
			weightStart := idx * e.EmbDim

			for i := 0; i < e.EmbDim; i++ {
				e.GradWeights.Data[weightStart+i] += gradData[gradStart+i]
			}
		}
	}

	// 更新权重
	for i := range e.Weights.Data {
		e.Weights.Data[i] -= learningRate * e.GradWeights.Data[i]
	}

	return nil
}

func (e *Embedding) ZeroGrad() {
	// 手动清零梯度
	for i := range e.GradWeights.Data {
		e.GradWeights.Data[i] = 0
	}
}

func (e *Embedding) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{e.Weights}
}

func (e *Embedding) SetWeightsAndShape(data []float64, shape []int) {
	if len(shape) != 2 || shape[0] != e.VocabSize || shape[1] != e.EmbDim {
		panic(fmt.Sprintf("Invalid embedding shape: expected [%d,%d], got %v",
			e.VocabSize, e.EmbDim, shape))
	}

	// 创建新权重并保留梯度结构
	e.Weights = tensor.NewTensor(data, shape)
	// 重置梯度张量
	e.GradWeights = tensor.NewTensor(make([]float64, len(data)), shape)
}
