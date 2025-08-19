package compute_graph

import (
	"fmt"
	"math"

	"github.com/Jimmy2099/torch/data_store/tensor"
)

type LpPool struct {
	*OPSNode
	OPSTensor
	kernel  []int
	strides []int
	p       float32
}

func (m *LpPool) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	input := m.Children[0].Node.Forward()
	shape := input.GetShape()
	if len(shape) != 4 {
		panic("LpPool only supports 4D tensors")
	}

	n := shape[0]
	c := shape[1]
	h := shape[2]
	w := shape[3]
	kernelH := m.kernel[0]
	kernelW := m.kernel[1]
	strideH := m.strides[0]
	strideW := m.strides[1]

	outH := (h-kernelH)/strideH + 1
	outW := (w-kernelW)/strideW + 1
	outShape := []int{n, c, outH, outW}
	outputData := make([]float32, n*c*outH*outW)

	for i := 0; i < n; i++ {
		for j := 0; j < c; j++ {
			for k := 0; k < outH; k++ {
				for l := 0; l < outW; l++ {
					startH := k * strideH
					startW := l * strideW
					endH := startH + kernelH
					endW := startW + kernelW
					sum := float32(0.0)

					for x := startH; x < endH; x++ {
						for y := startW; y < endW; y++ {
							idx := i*c*h*w + j*h*w + x*w + y
							val := float32(math.Pow(math.Abs(float64(input.Data[idx])), float64(m.p)))
							sum += val
						}
					}

					poolIdx := i*c*outH*outW + j*outH*outW + k*outW + l
					outputData[poolIdx] = float32(math.Pow(float64(sum), 1/float64(m.p)))
				}
			}
		}
	}

	m.output.value = tensor.NewTensor(outputData, outShape)
	m.output.computed = true
	return m.output.value
}

func (m *LpPool) Backward(grad *tensor.Tensor) {
	input := m.Children[0].value
	shape := input.GetShape()
	n := shape[0]
	c := shape[1]
	h := shape[2]
	w := shape[3]
	kernelH := m.kernel[0]
	kernelW := m.kernel[1]
	strideH := m.strides[0]
	strideW := m.strides[1]
	outH := (h-kernelH)/strideH + 1
	outW := (w-kernelW)/strideW + 1

	gradInput := make([]float32, len(input.Data))

	for i := 0; i < n; i++ {
		for j := 0; j < c; j++ {
			for k := 0; k < outH; k++ {
				for l := 0; l < outW; l++ {
					startH := k * strideH
					startW := l * strideW
					endH := startH + kernelH
					endW := startW + kernelW
					poolIdx := i*c*outH*outW + j*outH*outW + k*outW + l
					poolVal := m.output.value.Data[poolIdx]
					gradVal := grad.Data[poolIdx]

					for x := startH; x < endH; x++ {
						for y := startW; y < endW; y++ {
							idx := i*c*h*w + j*h*w + x*w + y
							xVal := input.Data[idx]
							absX := float32(math.Abs(float64(xVal)))
							sign := float32(1.0)
							if xVal < 0 {
								sign = -1.0
							}

							derivative := float32(0.0)
							if poolVal > 0 {
								derivative = gradVal *
									float32(math.Pow(float64(absX), float64(m.p-1))) *
									sign *
									float32(math.Pow(float64(poolVal), float64(1-m.p)))
							}
							gradInput[idx] += derivative
						}
					}
				}
			}
		}
	}

	m.Children[0].Node.Backward(tensor.NewTensor(gradInput, shape))
}

func (t *GraphTensor) LpPool(kernel, strides []int, p float32, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("lppool_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	g := t.Graph
	node := NewLpPool(name, t, kernel, strides, p)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewLpPool(name string, input *GraphTensor, kernel, strides []int, p float32) *LpPool {
	return &LpPool{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "LpPool",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{input},
		},
		kernel:  kernel,
		strides: strides,
		p:       p,
	}
}

func (m *LpPool) GetOutput() *GraphTensor {
	return m.output
}
