package compute_graph

import (
	"fmt"
	onnx_ir "github.com/Jimmy2099/torch/thirdparty/onnx-go/ir"
	v1proto "github.com/golang/protobuf/proto"
	"io/ioutil"
)

type ONNX struct {
	model *onnx_ir.ModelProto
}

func (g *ComputationalGraph) ToONNXModel() (*ONNX, error) {
	model := &onnx_ir.ModelProto{}
	model.IrVersion = 7 // ONNX IR version
	model.ProducerName = "Torch Go"
	model.ProducerVersion = "0.1"

	// Create operator set import info
	opset := &onnx_ir.OperatorSetIdProto{
		Domain:  "",
		Version: 11, // ONNX opset version
	}
	model.OpsetImport = []*onnx_ir.OperatorSetIdProto{opset}

	// Create graph structure
	graph := &onnx_ir.GraphProto{
		Name: "computational_graph",
	}

	// Track tensors produced by operations
	producedTensors := make(map[string]bool)
	for _, node := range g.Nodes {
		switch n := node.(type) {
		case *Multiply:
			producedTensors[n.output.Name] = true
		case *Add:
			producedTensors[n.output.Name] = true
		case *Sub:
			producedTensors[n.output.Name] = true
		case *Div:
			producedTensors[n.output.Name] = true
		}
	}

	// 1. Add ONLY initial tensors as inputs (exclude intermediates)
	for name, t := range g.Tensors {
		if producedTensors[name] {
			continue // Skip tensors produced by operations
		}

		// Create type info
		tensorType := &onnx_ir.TypeProto_Tensor{
			ElemType: int32(onnx_ir.TensorProto_FLOAT),
			Shape: &onnx_ir.TensorShapeProto{
				Dim: make([]*onnx_ir.TensorShapeProto_Dimension, len(t.Shape)),
			},
		}

		for i, dim := range t.Shape {
			tensorType.Shape.Dim[i] = &onnx_ir.TensorShapeProto_Dimension{
				Value: &onnx_ir.TensorShapeProto_Dimension_DimValue{
					DimValue: int64(dim),
				},
			}
		}

		// Create value info
		valueInfo := &onnx_ir.ValueInfoProto{
			Name: name,
			Type: &onnx_ir.TypeProto{
				Value: &onnx_ir.TypeProto_TensorType{
					TensorType: tensorType,
				},
			},
		}

		// Add to graph inputs
		graph.Input = append(graph.Input, valueInfo)
	}

	// Node name counter for uniqueness
	nodeCounter := make(map[string]int)

	// 2. Add all operation nodes
	for _, node := range g.Nodes {
		var onnxNode *onnx_ir.NodeProto
		var nodeType string

		switch n := node.(type) {
		case *Multiply:
			nodeType = "Mul"
			onnxNode = &onnx_ir.NodeProto{
				OpType: "Mul",
				Input:  []string{n.Children[0].Name, n.Children[1].Name},
				Output: []string{n.output.Name},
			}
		case *Add:
			nodeType = "Add"
			onnxNode = &onnx_ir.NodeProto{
				OpType: "Add",
				Input:  []string{n.Children[0].Name, n.Children[1].Name},
				Output: []string{n.output.Name},
			}
		case *Sub:
			nodeType = "Sub"
			onnxNode = &onnx_ir.NodeProto{
				OpType: "Sub",
				Input:  []string{n.Children[0].Name, n.Children[1].Name},
				Output: []string{n.output.Name},
			}
		case *Div:
			nodeType = "Div"
			onnxNode = &onnx_ir.NodeProto{
				OpType: "Div",
				Input:  []string{n.Children[0].Name, n.Children[1].Name},
				Output: []string{n.output.Name},
			}
		case *InputNode:
			// Skip input nodes (no operation needed)
			continue
		default:
			return nil, fmt.Errorf("unsupported node type: %T", node)
		}

		// Generate unique node name
		count := nodeCounter[nodeType]
		nodeCounter[nodeType] = count + 1
		onnxNode.Name = fmt.Sprintf("%s_%d", nodeType, count)

		graph.Node = append(graph.Node, onnxNode)
	}

	// 3. Add output tensor
	if g.output != nil {
		outputInfo := &onnx_ir.ValueInfoProto{
			Name: g.output.Name, // Use original name
			Type: &onnx_ir.TypeProto{
				Value: &onnx_ir.TypeProto_TensorType{
					TensorType: &onnx_ir.TypeProto_Tensor{
						ElemType: int32(onnx_ir.TensorProto_FLOAT),
						Shape: &onnx_ir.TensorShapeProto{
							Dim: make([]*onnx_ir.TensorShapeProto_Dimension, len(g.output.Shape)),
						},
					},
				},
			},
		}

		// Set output shape
		for i, dim := range g.output.Shape {
			outputInfo.Type.GetTensorType().Shape.Dim[i] = &onnx_ir.TensorShapeProto_Dimension{
				Value: &onnx_ir.TensorShapeProto_Dimension_DimValue{
					DimValue: int64(dim),
				},
			}
		}

		graph.Output = append(graph.Output, outputInfo)
	}

	model.Graph = graph
	return &ONNX{model: model}, nil
}

func (m *ONNX) SaveONNX(filename string) error {
	data, err := v1proto.Marshal(m.model)
	if err != nil {
		return fmt.Errorf("failed to marshal ONNX model: %w", err)
	}

	if err := ioutil.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write ONNX file: %w", err)
	}
	return nil
}
