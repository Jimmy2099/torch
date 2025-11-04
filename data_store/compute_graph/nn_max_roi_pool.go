package compute_graph

import (
	"fmt"
	"math"

	"github.com/Jimmy2099/torch/data_store/tensor"
)

type MaxRoiPool struct {
	*OPSNode
	OPSTensor
	PooledHeight int
	PooledWidth  int
	argmax       []int
}

func (m *MaxRoiPool) Forward() *tensor.Tensor {
	if m.output.computed {
		return m.output.value
	}

	featureMap := m.Children[0].Node.Forward()
	rois := m.Children[1].Node.Forward()

	if len(featureMap.GetShape()) != 4 || len(rois.GetShape()) != 2 || rois.GetShape()[1] != 5 {
		panic("invalid tensor shapes for MaxRoiPool: featureMap must be 4D [N,H,W,C], rois must be 2D [N,5]")
	}

	_, H, W, C := featureMap.GetShape()[0], featureMap.GetShape()[1], featureMap.GetShape()[2], featureMap.GetShape()[3]

	outShape := []int{
		rois.GetShape()[0],
		m.PooledHeight,
		m.PooledWidth,
		C,
	}

	totalElements := outShape[0] * outShape[1] * outShape[2] * outShape[3]
	resultData := make([]float32, totalElements)
	m.argmax = make([]int, totalElements)

	for roiIdx := 0; roiIdx < rois.GetShape()[0]; roiIdx++ {
		roi := rois.Data[roiIdx*5 : (roiIdx+1)*5]
		batchIdx := int(roi[0])
		x1, y1, x2, y2 := roi[1], roi[2], roi[3], roi[4]

		absX1 := int(math.Round(float64(x1 * float32(W))))
		absY1 := int(math.Round(float64(y1 * float32(H))))
		absX2 := int(math.Round(float64(x2 * float32(W))))
		absY2 := int(math.Round(float64(y2 * float32(H))))

		absX1 = clamp(absX1, 0, W-1)
		absY1 = clamp(absY1, 0, H-1)
		absX2 = clamp(absX2, absX1+1, W)
		absY2 = clamp(absY2, absY1+1, H)

		binWidth := float32(absX2-absX1) / float32(m.PooledWidth)
		binHeight := float32(absY2-absY1) / float32(m.PooledHeight)

		for ph := 0; ph < m.PooledHeight; ph++ {
			for pw := 0; pw < m.PooledWidth; pw++ {
				hStart := int(float32(absY1) + float32(ph)*binHeight)
				hEnd := int(float32(absY1) + float32(ph+1)*binHeight)
				wStart := int(float32(absX1) + float32(pw)*binWidth)
				wEnd := int(float32(absX1) + float32(pw+1)*binWidth)

				hStart = clamp(hStart, absY1, absY2-1)
				hEnd = clamp(hEnd, hStart+1, absY2)
				wStart = clamp(wStart, absX1, absX2-1)
				wEnd = clamp(wEnd, wStart+1, absX2)

				for c := 0; c < C; c++ {
					maxVal := float32(math.SmallestNonzeroFloat32)
					maxIdx := -1

					for y := hStart; y < hEnd; y++ {
						for x := wStart; x < wEnd; x++ {
							featureIdx := batchIdx*(H*W*C) + y*(W*C) + x*C + c
							val := featureMap.Data[featureIdx]

							if val > maxVal {
								maxVal = val
								maxIdx = featureIdx
							}
						}
					}

					outIdx := roiIdx*(m.PooledHeight*m.PooledWidth*C) +
						ph*(m.PooledWidth*C) + pw*C + c

					resultData[outIdx] = maxVal
					m.argmax[outIdx] = maxIdx
				}
			}
		}
	}

	result := tensor.NewTensor(resultData, outShape)
	m.output.value = result
	m.output.computed = true
	return result
}

func (m *MaxRoiPool) Backward(grad *tensor.Tensor) {
	featureMap := m.Children[0].value
	rois := m.Children[1].value

	gradFeatureMap := tensor.NewTensor(
		make([]float32, len(featureMap.Data)),
		featureMap.GetShape(),
	)

	gradRois := tensor.NewTensor(
		make([]float32, len(rois.Data)),
		rois.GetShape(),
	)

	for i, inputIdx := range m.argmax {
		gradFeatureMap.Data[inputIdx] += grad.Data[i]
	}

	m.Children[0].Node.Backward(gradFeatureMap)
	m.Children[1].Node.Backward(gradRois)
}

func clamp(value, min, max int) int {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

func (t *GraphTensor) MaxRoiPool(rois *GraphTensor, pooledHeight, pooledWidth int, names ...string) *GraphTensor {
	var name string
	if len(names) > 0 {
		name = names[0]
	} else {
		name = fmt.Sprintf("max_roi_pool_%d", t.Graph.NodeCount)
		t.Graph.NodeCount++
	}

	if t.Graph != rois.Graph {
		panic("tensors belong to different graphs")
	}
	g := t.Graph

	node := NewMaxRoiPool(name, t, rois, pooledHeight, pooledWidth)
	outputShape := []int{
		rois.GetShape()[0],
		pooledHeight,
		pooledWidth,
		t.GetShape()[3],
	}

	totalElements := outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3]

	outputTensor := &GraphTensor{
		Name:  name,
		value: tensor.NewTensor(make([]float32, totalElements), outputShape),
		grad:  tensor.NewTensor(make([]float32, totalElements), outputShape),
		Graph: g,
		Node:  node,
	}
	outputTensor.SetShape(outputShape)

	if _, exists := g.Tensors[name]; exists {
		panic("tensor name already exists: " + name)
	}
	g.Tensors[name] = outputTensor
	node.output = outputTensor
	g.Nodes = append(g.Nodes, node)
	return outputTensor
}

func NewMaxRoiPool(name string, featureMap, rois *GraphTensor, pooledHeight, pooledWidth int) *MaxRoiPool {
	return &MaxRoiPool{
		OPSNode: NewOPSNode(OPSNode{
			ONNXName:           "MaxRoiPool",
			ONNXProducedTensor: true,
		}),
		OPSTensor: OPSTensor{
			Name:     name,
			Children: []*GraphTensor{featureMap, rois},
		},
		PooledHeight: pooledHeight,
		PooledWidth:  pooledWidth,
	}
}

func (m *MaxRoiPool) GetOutput() *tensor.Tensor {
	return m.output.value
}
