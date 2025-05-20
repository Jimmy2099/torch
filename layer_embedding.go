package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
	"math/rand"
)

type Embedding struct {
	Weights     *tensor.Tensor
	GradWeights *tensor.Tensor
	VocabSize   int
	EmbDim      int
	LastIndices []int
}

func (e *Embedding) SetBiasAndShape(data []float32, shape []int) {
}

func (e *Embedding) GetWeights() *tensor.Tensor {
	return e.Weights
}

func (e *Embedding) GetBias() *tensor.Tensor {
	return tensor.NewEmptyTensor()
}

func NewEmbedding(vocabSize, embDim int) *Embedding {
	data := make([]float32, vocabSize*embDim)
	scale := math.Sqrt(2.0 / float32(embDim))
	for i := range data {
		data[i] = float32(rand.NormFloat64()) * scale
	}

	return &Embedding{
		Weights:     tensor.NewTensor(data, []int{vocabSize, embDim}),
		GradWeights: tensor.NewTensor(make([]float32, vocabSize*embDim), []int{vocabSize, embDim}),
		VocabSize:   vocabSize,
		EmbDim:      embDim,
	}
}

func (e *Embedding) Forward(indices *tensor.Tensor) *tensor.Tensor {
	if len(indices.GetShape()) != 2 {
		panic(fmt.Sprintf("Embedding expects 2D input, got %dD", len(indices.GetShape())))
	}

	floatIndices := indices.Data
	intIndices := make([]int, len(floatIndices))
	for i, v := range floatIndices {
		if v != math.Trunc(v) {
			panic(fmt.Sprintf("Non-integer index at position %d: %v", i, v))
		}
		intIndices[i] = int(v)
	}
	e.LastIndices = intIndices

	batchSize, seqLen := indices.GetShape()[0], indices.GetShape()[1]
	outputShape := []int{batchSize, seqLen, e.EmbDim}
	outputData := make([]float32, batchSize*seqLen*e.EmbDim)

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

func (e *Embedding) Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	return nil

}

func (e *Embedding) ZeroGrad() {
	for i := range e.GradWeights.Data {
		e.GradWeights.Data[i] = 0
	}
}

func (e *Embedding) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{e.Weights}
}

func (e *Embedding) SetWeightsAndShape(data []float32, shape []int) {
	if len(shape) != 2 || shape[0] != e.VocabSize || shape[1] != e.EmbDim {
		panic(fmt.Sprintf("Invalid embedding shape: expected [%d,%d], got %v",
			e.VocabSize, e.EmbDim, shape))
	}

	e.Weights = tensor.NewTensor(data, shape)
	e.GradWeights = tensor.NewTensor(make([]float32, len(data)), shape)
}
