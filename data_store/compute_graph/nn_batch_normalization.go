package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type BatchNormalization struct {
	*OPSNode
	OPSTensor
	gamma       *GraphTensor
	beta        *GraphTensor
	epsilon     float32
	momentum    float32
	runningMean *tensor.Tensor
	runningVar  *tensor.Tensor
}

func (bn *BatchNormalization) Forward() *tensor.Tensor {
	if bn.output.computed {
		return bn.output.value
	}

	input := bn.Children[0].Node.Forward()
	gamma := bn.gamma.Node.Forward()
	beta := bn.beta.Node.Forward()

	sum := float32(0.0)
	for _, v := range input.Data {
		sum += v
	}
	meanVal := sum / float32(len(input.Data))
	meanTensor := tensor.NewTensor([]float32{meanVal}, []int{1})

	varianceVal := float32(0.0)
	for _, v := range input.Data {
		diff := v - meanVal
		varianceVal += diff * diff
	}
	varianceVal /= float32(len(input.Data))
	varianceTensor := tensor.NewTensor([]float32{varianceVal}, []int{1})

	if bn.runningMean == nil {
		bn.runningMean = tensor.NewTensor([]float32{meanVal}, []int{1})
		bn.runningVar = tensor.NewTensor([]float32{varianceVal}, []int{1})
	} else {
		newRunningMean := bn.runningMean.MulScalar(bn.momentum).Add(meanTensor.MulScalar(1 - bn.momentum))
		newRunningVar := bn.runningVar.MulScalar(bn.momentum).Add(varianceTensor.MulScalar(1 - bn.momentum))
		bn.runningMean = newRunningMean
		bn.runningVar = newRunningVar
	}

	stdDev := varianceTensor.AddScalar(bn.epsilon).Sqrt()
	normalized := input.Sub(meanTensor).Div(stdDev)
	result := normalized.Mul(gamma).Add(beta)

	bn.output.value = result
	bn.output.computed = true
	return result
}

func (bn *BatchNormalization) Backward(grad *tensor.Tensor) {
	input := bn.Children[0].value
	gamma := bn.gamma.value
	beta := bn.beta.value

	normalized := bn.output.value.Sub(beta).Div(gamma)

	stdDev := normalized.AddScalar(bn.epsilon).Sqrt()
	invStd := tensor.Ones(stdDev.GetShape()).Div(stdDev)

	N := float32(len(input.Data))
	dNormalized := grad.Mul(gamma)

	dNormalizedMean := dNormalized.MeanTensor()
	dNormalizedMulNormalizedMean := dNormalized.Mul(normalized).MeanTensor()

	dInput := dNormalized.Mul(invStd)

	term2 := normalized.Mul(dNormalizedMulNormalizedMean.MulScalar(-2.0 / N))
	dInput = dInput.Add(term2)

	dInput = dInput.Add(dNormalizedMean.MulScalar(-1.0))

	dGamma := grad.Mul(normalized).ReduceSum()
	dBeta := grad.ReduceSum()

	bn.Children[0].Node.Backward(dInput)
	bn.gamma.Node.Backward(dGamma)
	bn.beta.Node.Backward(dBeta)
}

func (t *GraphTensor) BatchNormalization(gamma, beta *GraphTensor, epsilon, momentum float32, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("batchnorm_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != gamma.Graph || t.Graph != beta.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	node := NewBatchNormalization(name, t, gamma, beta, epsilon, momentum)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(t.Shape())

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewBatchNormalization(name string, input, gamma, beta *GraphTensor, epsilon, momentum float32) *BatchNormalization {
	return &BatchNormalization{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Sub",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{input},
		},
		gamma:    gamma,
		beta:     beta,
		epsilon:  epsilon,
		momentum: momentum,
	}
}

func (m *BatchNormalization) GetOutput() *GraphTensor {
	return m.output
}
