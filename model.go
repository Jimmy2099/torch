package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor"
)

// Model 定义模型接口
type ModelInterface interface {
	Forward(input *tensor.Tensor) *tensor.Tensor
	Backward(target *tensor.Tensor, learningRate float64)
	ZeroGrad()
}

type Model struct {
	_      ModelInterface
	_      TrainerInterface
	Layers []Layer
	//
	LayerIndex2Name map[int]string
	LayerName2Index map[string]int
	//
}

func NewModel() *Model {
	return &Model{
		Layers:          []Layer{},
		LayerIndex2Name: map[int]string{},
		LayerName2Index: map[string]int{},
	}
}

func (m *Model) AddLayer(layer Layer) {
	m.Layers = append(m.Layers, layer)
	m.LayerIndex2Name[len(m.Layers)-1] = fmt.Sprint(len(m.Layers) - 1)
	m.LayerName2Index[fmt.Sprint(len(m.Layers)-1)] = len(m.Layers) - 1
}

func (m *Model) Forward(input *tensor.Tensor) *tensor.Tensor {
	return m.Forward(input)
}
func (m *Model) Backward(target *tensor.Tensor, learningRate float64) {

}
func (m *Model) PrintModel() {

}

func (m *Model) ZeroGrad() {
	for i := 0; i < len(m.Layers); i++ {
		m.Layers[i].ZeroGrad()
	}
}

// Trainer 定义训练器接口
type TrainerInterface interface {
	Train(model ModelInterface, inputs, targets *tensor.Tensor, epochs int, learningRate float64)
	Validate(model ModelInterface, inputs, targets *tensor.Tensor) float64
}
