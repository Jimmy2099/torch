package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

type Gemm struct {
	*OPSNode
	OPSTensor
	TransA bool
	TransB bool
	Alpha  float32
	Beta   float32
}

func (m *Gemm) Forward() *tensor.Tensor {
	if m.output.IsComputed() {
		return m.output.Value()
	}

	a := m.Children[0].Node.Forward()
	b := m.Children[1].Node.Forward()
	c := m.Children[2].Node.Forward()

	result := gemm(a, b, c, m.TransA, m.TransB, m.Alpha, m.Beta)
	m.output.SetValue(result)
	m.output.SetComputed(true)
	return result
}

func (t *GraphTensor) Gemm(b, c *GraphTensor, transA, transB bool, alpha, beta float32, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("gemm_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != b.Graph || t.Graph != c.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	node := NewGemm(name, transA, transB, alpha, beta, t, b, c)

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor([]float32{}, []int{0}),
		grad:  tensor.NewTensor([]float32{}, []int{0}),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape([]int{0})

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewGemm(name string, transA, transB bool, alpha, beta float32, a, b, c *GraphTensor) *Gemm {
	return &Gemm{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "Gemm",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{a, b, c},
		},
		TransA: transA,
		TransB: transB,
		Alpha:  alpha,
		Beta:   beta,
	}
}

func (m *Gemm) GetOutput() *tensor.Tensor {
	return m.output.value
}

func gemm(a, b, c *tensor.Tensor, transA, transB bool, alpha, beta float32) *tensor.Tensor {
	aOrigShape := a.GetShape()
	bOrigShape := b.GetShape()
	cShape := c.GetShape()

	aRows := aOrigShape[0]
	aCols := aOrigShape[1]
	if transA {
		aRows = aOrigShape[1]
		aCols = aOrigShape[0]
	}

	bRows := bOrigShape[0]
	bCols := bOrigShape[1]
	if transB {
		bRows = bOrigShape[1]
		bCols = bOrigShape[0]
	}

	if aCols != bRows {
		panic(fmt.Sprintf("incompatible shapes for GEMM: aCols %d, bRows %d", aCols, bRows))
	}
	M := aRows
	N := bCols
	K := aCols

	if M != cShape[0] || N != cShape[1] {
		panic(fmt.Sprintf("incompatible output shape for GEMM: expected %dx%d, got %v", M, N, cShape))
	}

	resultData := make([]float32, M*N)

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)
			for k := 0; k < K; k++ {
				var aIdx, bIdx int
				if transA {
					aIdx = k*aOrigShape[1] + i
				} else {
					aIdx = i*aOrigShape[1] + k
				}
				if transB {
					bIdx = j*bOrigShape[1] + k
				} else {
					bIdx = k*bOrigShape[1] + j
				}
				sum += a.Data[aIdx] * b.Data[bIdx]
			}
			cIdx := i*N + j
			resultData[i*N+j] = alpha*sum + beta*c.Data[cIdx]
		}
	}
	return tensor.NewTensor(resultData, []int{M, N})
}

func gemmGradA(grad, b *tensor.Tensor, aShape []int, transA, transB bool, alpha float32) *tensor.Tensor {
	gradShape := grad.GetShape()
	bShape := b.GetShape()

	M := aShape[0]
	N := aShape[1]
	K := gradShape[1]

	result := make([]float32, M*N)

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)
			for k := 0; k < K; k++ {
				gradIdx := i*gradShape[1] + k
				bIdx := j*bShape[1] + k
				sum += grad.Data[gradIdx] * b.Data[bIdx]
			}
			result[i*N+j] = sum * alpha
		}
	}
	return tensor.NewTensor(result, aShape)
}

func gemmGradB(grad, a *tensor.Tensor, bShape []int, transA, transB bool, alpha float32) *tensor.Tensor {
	gradShape := grad.GetShape()
	aShape := a.GetShape()

	M := bShape[0]
	N := bShape[1]
	K := gradShape[0]

	result := make([]float32, M*N)

	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0.0)
			for k := 0; k < K; k++ {
				gradIdx := k*gradShape[1] + j
				aIdx := k*aShape[1] + i
				sum += a.Data[aIdx] * grad.Data[gradIdx]
			}
			result[i*N+j] = sum * alpha
		}
	}
	return tensor.NewTensor(result, bShape)
}
