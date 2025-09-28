package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Conv struct {
	*OPSNode
	OPSTensor
	StrideH     int
	StrideW     int
	PadH        int
	PadW        int
	kernelShape []int
}

func (m *Conv) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	weight := m.Children[1].Node.Forward()

	if len(input.GetShape()) != 4 || len(weight.GetShape()) != 4 {
		panic(fmt.Sprintf("convolution expects 4D tensors, got input %v and weight %v",
			input.GetShape(), weight.GetShape()))
	}

	if input.GetShape()[1] != weight.GetShape()[1] {
		panic(fmt.Sprintf("input channels (%d) must match weight channels (%d)",
			input.GetShape()[1], weight.GetShape()[1]))
	}

	m.kernelShape = weight.GetShape()

	kernelH := m.kernelShape[2]
	kernelW := m.kernelShape[3]
	if kernelH != kernelW {
		panic("only square kernels are supported with current Conv2D")
	}

	result := input.Conv2D(
		weight,
		kernelH,
		m.StrideH,
		m.PadH,
		m.PadW,
	)

	m.output.value = result
	m.output.computed = true
	return result
}

func (m *Conv) Backward(grad *tensor.Tensor) {
	input := m.Children[0].Node.Forward()
	weight := m.Children[1].Node.Forward()

	kernelH := m.kernelShape[2]
	kernelW := m.kernelShape[3]

	gradInput := grad.Conv2DTranspose(
		weight,
		kernelH,
		kernelW,
		m.StrideH,
		m.StrideW,
		m.PadH,
		m.PadW,
	)

	gradWeight := computeKernelGradient(
		input,
		grad,
		kernelH,
		kernelW,
		m.StrideH,
		m.StrideW,
		m.PadH,
		m.PadW,
	)

	m.Children[0].Node.Backward(gradInput)
	m.Children[1].Node.Backward(gradWeight)
}

func (t *GraphTensor) Conv(weight *GraphTensor, stride, padding []int, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("conv_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != weight.Graph {
		panic("tensors belong to different graphs")
	}

	sH, sW, padH, padW := processConvParams(stride, padding)

	node := NewConv(name, sH, sW, padH, padW, t, weight)

	outputTensor := &GraphTensor{
		Name:  name,
		value: nil,
		grad:  nil,
		Graph: t.Graph,
		Node:  node,
	}
	outputTensor.SetShape([]int{0, 0, 0, 0})

	if _, exists := t.Graph.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}

	t.Graph.Tensors[name] = outputTensor
	node.output = outputTensor
	t.Graph.Nodes = append(t.Graph.Nodes, node)
	return outputTensor
}

func processConvParams(stride, padding []int) (sH, sW, padH, padW int) {
	if len(stride) == 0 {
		stride = []int{1}
	}

	switch len(stride) {
	case 1:
		sH = stride[0]
		sW = stride[0]
	case 2:
		sH = stride[0]
		sW = stride[1]
	default:
		panic("stride must have 1 or 2 elements")
	}

	if len(padding) == 0 {
		padding = []int{0}
	}

	switch len(padding) {
	case 1:
		padH = padding[0]
		padW = padding[0]
	case 2:
		padH = padding[0]
		padW = padding[1]
	default:
		panic("padding must have 1 or 2 elements")
	}
	return
}

func NewConv(name string, strideH, strideW, padH, padW int, input, weight *GraphTensor) *Conv {
	if len(weight.value.GetShape()) != 4 {
		panic("weight tensor must be 4D [out_channels, in_channels, kernelH, kernelW]")
	}

	return &Conv{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Conv",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{input, weight},
		},
		StrideH:     strideH,
		StrideW:     strideW,
		PadH:        padH,
		PadW:        padW,
		kernelShape: weight.value.GetShape(),
	}
}

func computeKernelGradient(input, grad *tensor.Tensor, kernelH, kernelW, strideH, strideW, padH, padW int) *tensor.Tensor {
	inputShape := input.GetShape()
	gradShape := grad.GetShape()

	batch := inputShape[0]
	inChannels := inputShape[1]
	inH := inputShape[2]
	inW := inputShape[3]

	outChannels := gradShape[1]
	outH := gradShape[2]
	outW := gradShape[3]

	gradWeightData := make([]float32, outChannels*inChannels*kernelH*kernelW)
	gradWeight := tensor.NewTensor(gradWeightData, []int{outChannels, inChannels, kernelH, kernelW})

	for b := 0; b < batch; b++ {
		for oc := 0; oc < outChannels; oc++ {
			for ic := 0; ic < inChannels; ic++ {
				for kh := 0; kh < kernelH; kh++ {
					for kw := 0; kw < kernelW; kw++ {
						var sum float32
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								iH := oh*strideH + kh - padH
								iW := ow*strideW + kw - padW

								if iH >= 0 && iH < inH && iW >= 0 && iW < inW {
									inputVal := input.Get([]int{b, ic, iH, iW})
									gradVal := grad.Get([]int{b, oc, oh, ow})
									sum += inputVal * gradVal
								}
							}
						}
						current := gradWeight.Get([]int{oc, ic, kh, kw})
						gradWeight.Set(current+sum, oc, ic, kh, kw)
					}
				}
			}
		}
	}
	return gradWeight
}

func (m *Conv) GetOutput() *GraphTensor {
	return m.output
}
