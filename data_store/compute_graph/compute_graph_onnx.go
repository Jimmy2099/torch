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

	// 创建操作集导入信息
	opset := &onnx_ir.OperatorSetIdProto{
		Domain:  "",
		Version: 11, // ONNX opset version
	}
	model.OpsetImport = []*onnx_ir.OperatorSetIdProto{opset}

	// 创建图结构
	graph := &onnx_ir.GraphProto{
		Name: "computational_graph",
	}

	// 1. 添加所有张量作为输入
	for name, t := range g.tensors {
		// 创建类型信息
		tensorType := &onnx_ir.TypeProto_Tensor{
			ElemType: int32(onnx_ir.TensorProto_FLOAT),
			Shape: &onnx_ir.TensorShapeProto{
				Dim: make([]*onnx_ir.TensorShapeProto_Dimension, len(t.shape)),
			},
		}

		for i, dim := range t.shape {
			tensorType.Shape.Dim[i] = &onnx_ir.TensorShapeProto_Dimension{
				Value: &onnx_ir.TensorShapeProto_Dimension_DimValue{
					DimValue: int64(dim),
				},
			}
		}

		// 创建值信息
		valueInfo := &onnx_ir.ValueInfoProto{
			Name: name,
			Type: &onnx_ir.TypeProto{
				Value: &onnx_ir.TypeProto_TensorType{
					TensorType: tensorType,
				},
			},
		}

		// 添加到图输入
		graph.Input = append(graph.Input, valueInfo)
	}

	// 2. 添加所有操作节点
	for _, node := range g.nodes {
		switch n := node.(type) {
		case *Multiply:
			onnxNode := &onnx_ir.NodeProto{
				OpType: "Mul",
				Input:  []string{n.Children[0].Name, n.Children[1].Name},
				Output: []string{n.output.Name},
				Name:   n.Name,
			}
			graph.Node = append(graph.Node, onnxNode)

		case *Add:
			onnxNode := &onnx_ir.NodeProto{
				OpType: "Add",
				Input:  []string{n.Children[0].Name, n.Children[1].Name},
				Output: []string{n.output.Name},
				Name:   n.Name,
			}
			graph.Node = append(graph.Node, onnxNode)

		case *InputNode:
			// 输入节点不需要特殊操作，已作为输入添加
		}
	}

	// 3. 添加输出张量
	if g.output != nil {
		outputInfo := &onnx_ir.ValueInfoProto{
			Name: g.output.Name,
			Type: &onnx_ir.TypeProto{
				Value: &onnx_ir.TypeProto_TensorType{
					TensorType: &onnx_ir.TypeProto_Tensor{
						ElemType: int32(onnx_ir.TensorProto_FLOAT),
						Shape: &onnx_ir.TensorShapeProto{
							Dim: make([]*onnx_ir.TensorShapeProto_Dimension, len(g.output.shape)),
						},
					},
				},
			},
		}

		for i, dim := range g.output.shape {
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
