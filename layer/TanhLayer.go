package layer

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
)

type TanhLayer struct {
	outputCache *tensor.Tensor
}

func (r *TanhLayer) GetWeights() *tensor.Tensor {
	return &tensor.Tensor{
		Data:  make([]float32, 0),
		Shape: make([]int, 0),
	}
}

func (r *TanhLayer) GetBias() *tensor.Tensor {
	return &tensor.Tensor{
		Data:  make([]float32, 0),
		Shape: make([]int, 0),
	}
}

func NewTanhLayer() *TanhLayer {
	return &TanhLayer{}
}

func (t *TanhLayer) Forward(input *tensor.Tensor) *tensor.Tensor {
	outputData := make([]float32, len(input.Data))
	for i, val := range input.Data {
		outputData[i] = math.Tanh(val)
	}
	output := tensor.NewTensor(outputData, input.Shape)

	t.outputCache = output

	fmt.Println("TanhLayer Forward Pass")
	return output
}

func (t *TanhLayer) Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	if t.outputCache == nil {
		panic("Backward called before Forward or output cache is nil for TanhLayer")
	}
	if !gradOutput.ShapesMatch(t.outputCache) {
		panic(fmt.Sprintf("Shapes of gradOutput %v and cached output %v do not match in TanhLayer Backward", gradOutput.Shape, t.outputCache.Shape))
	}

	gradInputData := make([]float32, len(gradOutput.Data))
	for i := range gradOutput.Data {
		tanhOutput := t.outputCache.Data[i]
		gradInputData[i] = gradOutput.Data[i] * (1.0 - tanhOutput*tanhOutput)
	}

	gradInput := tensor.NewTensor(gradInputData, gradOutput.Shape)
	return gradInput
}

func (t *TanhLayer) ZeroGrad() {
}

func (t *TanhLayer) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

func (t *TanhLayer) SetWeights(data [][]float32) {
}

func (t *TanhLayer) SetBias(data [][]float32) {
}
