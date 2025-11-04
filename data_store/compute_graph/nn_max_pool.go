package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type MaxPool struct {
	*OPSNode
	OPSTensor
	kernel       []int
	stride       []int
	padding      []int
	maxPositions [][][][]int
}

func (m *MaxPool) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	shape := input.GetShape()
	if len(shape) != 4 {
		panic("MaxPool requires 4D input")
	}

	B, C, H, W := shape[0], shape[1], shape[2], shape[3]
	kH, kW := m.kernel[0], m.kernel[1]
	sH, sW := m.stride[0], m.stride[1]
	padH, padW := m.padding[0], m.padding[1]

	Hout := (H+2*padH-kH)/sH + 1
	Wout := (W+2*padW-kW)/sW + 1
	outSize := B * C * Hout * Wout
	outputData := make([]float32, outSize)
	maxPositions := make([][][][]int, B)

	for b := range maxPositions {
		maxPositions[b] = make([][][]int, C)
		for c := range maxPositions[b] {
			maxPositions[b][c] = make([][]int, Hout)
			for i := range maxPositions[b][c] {
				maxPositions[b][c][i] = make([]int, Wout)
			}
		}
	}

	for b := 0; b < B; b++ {
		for c := 0; c < C; c++ {
			for i := 0; i < Hout; i++ {
				for j := 0; j < Wout; j++ {
					maxVal := float32(-1e9)
					maxH, maxW := -1, -1
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							h := i*sH + kh - padH
							w := j*sW + kw - padW
							if h >= 0 && h < H && w >= 0 && w < W {
								idx := b*(C*H*W) + c*(H*W) + h*W + w
								val := input.Data[idx]
								if val > maxVal {
									maxVal = val
									maxH = h
									maxW = w
								}
							}
						}
					}
					if maxH == -1 || maxW == -1 {
						outputData[b*(C*Hout*Wout)+c*(Hout*Wout)+i*Wout+j] = 0
						maxPositions[b][c][i][j] = -1
					} else {
						outputData[b*(C*Hout*Wout)+c*(Hout*Wout)+i*Wout+j] = maxVal
						maxPositions[b][c][i][j] = maxH*W + maxW
					}
				}
			}
		}
	}

	m.output.value = tensor.NewTensor(outputData, []int{B, C, Hout, Wout})
	m.output.computed = true
	m.maxPositions = maxPositions
	return m.output.value
}

func (m *MaxPool) Backward(grad *tensor.Tensor) {
	if grad == nil {
		panic("nil gradient in MaxPool backward pass")
	}

	input := m.Children[0].value
	shape := input.GetShape()
	B, C, H, W := shape[0], shape[1], shape[2], shape[3]
	Hout, Wout := grad.GetShape()[2], grad.GetShape()[3]

	gradInput := make([]float32, len(input.Data))

	for b := 0; b < B; b++ {
		for c := 0; c < C; c++ {
			for i := 0; i < Hout; i++ {
				for j := 0; j < Wout; j++ {
					pos := m.maxPositions[b][c][i][j]
					if pos != -1 {
						gradVal := grad.Data[b*(C*Hout*Wout)+c*(Hout*Wout)+i*Wout+j]
						gradInput[b*(C*H*W)+c*(H*W)+pos] += gradVal
					}
				}
			}
		}
	}

	m.Children[0].Node.Backward(tensor.NewTensor(gradInput, input.GetShape()))
}

func (t *GraphTensor) MaxPool(kernel, stride, padding []int, name string) *GraphTensor {
	if name == "" {
		name = fmt.Sprintf("maxpool_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}
	if t.Graph == nil {
		panic("tensor not in graph")
	}

	node := NewMaxPool(name, t, kernel, stride, padding)
	outputShape := calculatePoolOutputShape(t.GetShape(), kernel, stride, padding)
	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: t.Graph,
		Node:  node,
	}
	outputTensor.SetShape(outputShape)

	t.Graph.Tensors[name] = outputTensor
	node.output = outputTensor
	t.Graph.Nodes = append(t.Graph.Nodes, node)
	return outputTensor
}

func NewMaxPool(name string, input *GraphTensor, kernel, stride, padding []int) *MaxPool {
	return &MaxPool{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "MaxPool",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{input},
		},
		kernel:  kernel,
		stride:  stride,
		padding: padding,
	}
}

func (m *MaxPool) GetOutput() *tensor.Tensor {
	return m.output.value
}
