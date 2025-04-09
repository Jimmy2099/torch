// layer/TanhLayer.go
package layer

import (
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
)

// TanhLayer implements the hyperbolic tangent activation function.
type TanhLayer struct {
	// Cache output for backward pass
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

// NewTanhLayer creates a new TanhLayer.
func NewTanhLayer() *TanhLayer {
	return &TanhLayer{}
}

// Forward applies the Tanh function element-wise.
// input: Input tensor
// Returns: Output tensor with Tanh applied
func (t *TanhLayer) Forward(input *tensor.Tensor) *tensor.Tensor {
	outputData := make([]float32, len(input.Data))
	for i, val := range input.Data {
		outputData[i] = math.Tanh(val)
	}
	output := tensor.NewTensor(outputData, input.Shape)

	// Cache the output for backward pass
	t.outputCache = output

	fmt.Println("TanhLayer Forward Pass")
	return output
}

// Backward calculates the gradient through the Tanh layer.
// gradOutput: Gradient of the loss with respect to the output of this layer.
// learningRate: Not used by activation layers.
// Returns: Gradient of the loss with respect to the input of this layer.
func (t *TanhLayer) Backward(gradOutput *tensor.Tensor, learningRate float32) *tensor.Tensor {
	if t.outputCache == nil {
		panic("Backward called before Forward or output cache is nil for TanhLayer")
	}
	if !gradOutput.ShapesMatch(t.outputCache) {
		panic(fmt.Sprintf("Shapes of gradOutput %v and cached output %v do not match in TanhLayer Backward", gradOutput.Shape, t.outputCache.Shape))
	}

	gradInputData := make([]float32, len(gradOutput.Data))
	for i := range gradOutput.Data {
		// Derivative of tanh(x) is 1 - tanh(x)^2
		tanhOutput := t.outputCache.Data[i]
		gradInputData[i] = gradOutput.Data[i] * (1.0 - tanhOutput*tanhOutput)
	}

	gradInput := tensor.NewTensor(gradInputData, gradOutput.Shape)
	return gradInput
}

// ZeroGrad does nothing for TanhLayer as it has no trainable parameters.
func (t *TanhLayer) ZeroGrad() {
	// No parameters to zero gradients for
}

// Parameters returns an empty slice as TanhLayer has no trainable parameters.
func (t *TanhLayer) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{}
}

// SetWeights does nothing for TanhLayer.
func (t *TanhLayer) SetWeights(data [][]float32) {
	// No weights to set
}

// SetBias does nothing for TanhLayer.
func (t *TanhLayer) SetBias(data [][]float32) {
	// No bias to set
}
