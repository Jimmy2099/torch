package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type AveragePool struct {
	OPS
	kernel     []int
	stride     []int
	padding    []int
	kernelArea float32
}

func (m *AveragePool) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	shape := input.GetShape()
	if len(shape) != 4 {
		panic("AveragePool requires 4D input")
	}

	B, C, H, W := shape[0], shape[1], shape[2], shape[3]
	kH, kW := m.kernel[0], m.kernel[1]
	sH, sW := m.stride[0], m.stride[1]
	padH, padW := m.padding[0], m.padding[1]

	Hout := (H+2*padH-kH)/sH + 1
	Wout := (W+2*padW-kW)/sW + 1
	outSize := B * C * Hout * Wout
	outputData := make([]float32, outSize)

	for b := 0; b < B; b++ {
		for c := 0; c < C; c++ {
			for i := 0; i < Hout; i++ {
				for j := 0; j < Wout; j++ {
					sum := float32(0.0)
					count := float32(0.0)
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							h := i*sH + kh - padH
							w := j*sW + kw - padW
							if h >= 0 && h < H && w >= 0 && w < W {
								idx := b*(C*H*W) + c*(H*W) + h*W + w
								sum += input.Data[idx]
								count++
							}
						}
					}
					if count == 0 {
						outputData[b*(C*Hout*Wout)+c*(Hout*Wout)+i*Wout+j] = 0
					} else {
						outputData[b*(C*Hout*Wout)+c*(Hout*Wout)+i*Wout+j] = sum / count
					}
				}
			}
		}
	}

	m.output.value = tensor.NewTensor(outputData, []int{B, C, Hout, Wout})
	m.output.computed = true
	return m.output.value
}

func (m *AveragePool) Backward(grad *tensor.Tensor) {
	if grad == nil {
		panic("nil gradient in AveragePool backward pass")
	}

	input := m.Children[0].value
	shape := input.GetShape()
	B, C, H, W := shape[0], shape[1], shape[2], shape[3]
	kH, kW := m.kernel[0], m.kernel[1]
	sH, sW := m.stride[0], m.stride[1]
	padH, padW := m.padding[0], m.padding[1]
	Hout, Wout := grad.GetShape()[2], grad.GetShape()[3]

	gradInput := make([]float32, len(input.Data))

	for b := 0; b < B; b++ {
		for c := 0; c < C; c++ {
			for i := 0; i < Hout; i++ {
				for j := 0; j < Wout; j++ {
					gradVal := grad.Data[b*(C*Hout*Wout)+c*(Hout*Wout)+i*Wout+j]
					for kh := 0; kh < kH; kh++ {
						for kw := 0; kw < kW; kw++ {
							h := i*sH + kh - padH
							w := j*sW + kw - padW
							if h >= 0 && h < H && w >= 0 && w < W {
								idx := b*(C*H*W) + c*(H*W) + h*W + w
								area := float32(kH * kW)
								gradInput[idx] += gradVal / area
							}
						}
					}
				}
			}
		}
	}

	m.Children[0].Node.Backward(tensor.NewTensor(gradInput, input.GetShape()))
}

func (t *GraphTensor) AveragePool(kernel, stride, padding []int, name string) *GraphTensor {
	if name == "" {
		name = fmt.Sprintf("avgpool_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}
	if t.Graph == nil {
		panic("tensor not in graph")
	}

	node := NewAveragePool(name, t, kernel, stride, padding)
	outputShape := calculatePoolOutputShape(t.Shape, kernel, stride, padding)
	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Shape: outputShape,
		Graph: t.Graph,
		Node:  node,
	}

	t.Graph.Tensors[name] = outputTensor
	node.output = outputTensor
	t.Graph.Nodes = append(t.Graph.Nodes, node)
	return outputTensor
}

func NewAveragePool(name string, input *GraphTensor, kernel, stride, padding []int) *AveragePool {
	return &AveragePool{
		OPS: OPS{
			Name:     name,
			Children: []*GraphTensor{input},
		},
		kernel:  kernel,
		stride:  stride,
		padding: padding,
	}
}

func calculatePoolOutputShape(shape []int, kernel, stride, padding []int) []int {
	Hout := (shape[2]+2*padding[0]-kernel[0])/stride[0] + 1
	Wout := (shape[3]+2*padding[1]-kernel[1])/stride[1] + 1
	return []int{shape[0], shape[1], Hout, Wout}
}
