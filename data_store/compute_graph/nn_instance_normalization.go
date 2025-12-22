package compute_graph

import (
	"fmt"
	"math"

	"github.com/Jimmy2099/torch/data_store/tensor"
)

type InstanceNormalization struct {
	*OPSNode
	OPSTensor
	epsilon  float32
	mean     *tensor.Tensor
	variance *tensor.Tensor
}

func (m *InstanceNormalization) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	shape := input.GetShape()

	if len(shape) != 4 {
		panic("InstanceNormalization only supports 4D tensors")
	}

	n := shape[0]
	c := shape[1]
	h := shape[2]
	w := shape[3]
	total := h * w

	meanData := make([]float32, len(input.Data))
	varianceData := make([]float32, len(input.Data))
	outputData := make([]float32, len(input.Data))

	for i := 0; i < n; i++ {
		for j := 0; j < c; j++ {
			start := i*c*h*w + j*h*w
			end := start + h*w
			slice := input.Data[start:end]

			sum := float32(0.0)
			for _, val := range slice {
				sum += val
			}
			mean := sum / float32(total)

			variance := float32(0.0)
			for _, val := range slice {
				diff := val - mean
				variance += diff * diff
			}
			variance /= float32(total)

			stddev := float32(math.Sqrt(float64(variance + m.epsilon)))

			for k, val := range slice {
				idx := start + k
				meanData[idx] = mean
				varianceData[idx] = variance
				outputData[idx] = (val - mean) / stddev
			}
		}
	}

	m.mean = tensor.NewTensor(meanData, shape)
	m.variance = tensor.NewTensor(varianceData, shape)
	m.output.value = tensor.NewTensor(outputData, shape)
	m.output.computed = true
	return m.output.value
}

func (t *GraphTensor) InstanceNormalization(epsilon float32, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("instancenorm_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewInstanceNormalization(name, t, epsilon)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(t.GetShape())

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewInstanceNormalization(name string, input *GraphTensor, epsilon float32) *InstanceNormalization {
	return &InstanceNormalization{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "InstanceNormalization",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{input},
		},
		epsilon: epsilon,
	}
}

func (m *InstanceNormalization) GetOutput() *tensor.Tensor {
	return m.output.value
}
