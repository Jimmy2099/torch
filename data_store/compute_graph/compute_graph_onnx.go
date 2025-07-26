package compute_graph

import (
	"fmt"
	onnx_ir "github.com/Jimmy2099/torch/thirdparty/onnx-go/ir"
	v1proto "github.com/golang/protobuf/proto"
	"io/ioutil"
)

var ONNXOperators = []string{
	//ai.onnx
	"Abs",
	"Acos",
	"Acosh",
	"Add",
	"AffineGrid",
	"And",
	"ArgMax",
	"ArgMin",
	"Asin",
	"Asinh",
	"Atan",
	"Atanh",
	"Attention",
	"AveragePool",
	"BatchNormalization",
	"Bernoulli",
	"BitShift",
	"BitwiseAnd",
	"BitwiseNot",
	"BitwiseOr",
	"BitwiseXor",
	"BlackmanWindow",
	"Cast",
	"CastLike",
	"Ceil",
	"Celu",
	"CenterCropPad",
	"Clip",
	"Col2Im",
	"Compress",
	"Concat",
	"ConcatFromSequence",
	"Constant",
	"ConstantOfShape",
	"Conv",
	"ConvInteger",
	"ConvTranspose",
	"Cos",
	"Cosh",
	"CumSum",
	"DFT",
	"DeformConv",
	"DepthToSpace",
	"DequantizeLinear",
	"Det",
	"Div",
	"Dropout",
	"DynamicQuantizeLinear",
	"Einsum",
	"Elu",
	"Equal",
	"Erf",
	"Exp",
	"Expand",
	"EyeLike",
	"Flatten",
	"Floor",
	"GRU",
	"Gather",
	"GatherElements",
	"GatherND",
	"Gelu",
	"Gemm",
	"GlobalAveragePool",
	"GlobalLpPool",
	"GlobalMaxPool",
	"Greater",
	"GreaterOrEqual",
	"GridSample",
	"GroupNormalization",
	"HammingWindow",
	"HannWindow",
	"HardSigmoid",
	"HardSwish",
	"Hardmax",
	"Identity",
	"If",
	"ImageDecoder",
	"InstanceNormalization",
	"IsInf",
	"IsNaN",
	"LRN",
	"LSTM",
	"LayerNormalization",
	"LeakyRelu",
	"Less",
	"LessOrEqual",
	"Log",
	"LogSoftmax",
	"Loop",
	"LpNormalization",
	"LpPool",
	"MatMul",
	"MatMulInteger",
	"Max",
	"MaxPool",
	"MaxRoiPool",
	"MaxUnpool",
	"Mean",
	"MeanVarianceNormalization",
	"MelWeightMatrix",
	"Min",
	"Mish",
	"Mod",
	"Mul",
	"Multinomial",
	"Neg",
	"NegativeLogLikelihoodLoss",
	"NonMaxSuppression",
	"NonZero",
	"Not",
	"OneHot",
	"Optional",
	"OptionalGetElement",
	"OptionalHasElement",
	"Or",
	"PRelu",
	"Pad",
	"Pow",
	"QLinearConv",
	"QLinearMatMul",
	"QuantizeLinear",
	"RMSNormalization",
	"RNN",
	"RandomNormal",
	"RandomNormalLike",
	"RandomUniform",
	"RandomUniformLike",
	"Range",
	"Reciprocal",
	"ReduceL1",
	"ReduceL2",
	"ReduceLogSum",
	"ReduceLogSumExp",
	"ReduceMax",
	"ReduceMean",
	"ReduceMin",
	"ReduceProd",
	"ReduceSum",
	"ReduceSumSquare",
	"RegexFullMatch",
	"Relu",
	"Reshape",
	"Resize",
	"ReverseSequence",
	"RoiAlign",
	"RotaryEmbedding",
	"Round",
	"STFT",
	"Scan",
	"Scatter",
	"ScatterElements",
	"ScatterND",
	"Selu",
	"SequenceAt",
	"SequenceConstruct",
	"SequenceEmpty",
	"SequenceErase",
	"SequenceInsert",
	"SequenceLength",
	"SequenceMap",
	"Shape",
	"Shrink",
	"Sigmoid",
	"Sign",
	"Sin",
	"Sinh",
	"Size",
	"Slice",
	"Softmax",
	"SoftmaxCrossEntropyLoss",
	"Softplus",
	"Softsign",
	"SpaceToDepth",
	"Split",
	"SplitToSequence",
	"Sqrt",
	"Squeeze",
	"StringConcat",
	"StringNormalizer",
	"StringSplit",
	"Sub",
	"Sum",
	"Swish",
	"Tan",
	"Tanh",
	"TensorScatter",
	"TfIdfVectorizer",
	"ThresholdedRelu",
	"Tile",
	"TopK",
	"Transpose",
	"Trilu",
	"Unique",
	"Unsqueeze",
	"Upsample",
	"Where",
	"Xor",
	//ai.onnx.ml
	"ArrayFeatureExtractor",
	"Binarizer",
	"CastMap",
	"CategoryMapper",
	"DictVectorizer",
	"FeatureVectorizer",
	"Imputer",
	"LabelEncoder",
	"LinearClassifier",
	"LinearRegressor",
	"Normalizer",
	"OneHotEncoder",
	"SVMClassifier",
	"SVMRegressor",
	"Scaler",
	"TreeEnsemble",
	"TreeEnsembleClassifier",
	"TreeEnsembleRegressor",
	"ZipMap",
	//ai.onnx.preview.training
	"Adagrad",
	"Adam",
	"Gradient",
	"Momentum",
}

type ONNX struct {
	model *onnx_ir.ModelProto
}

type ONNXNodeInfo struct {
	Name           string
	ProducedTensor bool
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

	// 1. Add ONLY initial tensors as inputs (exclude intermediates)
	for name, t := range g.Tensors {
		if t.Node.GetONNXNodeInfo().ProducedTensor {
			continue
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
		//
		//switch n := node.(type) {
		//case *Multiply:
		//	nodeType = "Mul"
		//	onnxNode = &onnx_ir.NodeProto{
		//		OpType: "Mul",
		//		Input:  []string{n.Children[0].Name, n.Children[1].Name},
		//		Output: []string{n.output.Name},
		//	}
		//case *Add:
		//	nodeType = "Add"
		//	onnxNode = &onnx_ir.NodeProto{
		//		OpType: "Add",
		//		Input:  []string{n.Children[0].Name, n.Children[1].Name},
		//		Output: []string{n.output.Name},
		//	}
		//case *Sub:
		//	nodeType = "Sub"
		//	onnxNode = &onnx_ir.NodeProto{
		//		OpType: "Sub",
		//		Input:  []string{n.Children[0].Name, n.Children[1].Name},
		//		Output: []string{n.output.Name},
		//	}
		//case *Div:
		//	nodeType = "Div"
		//	onnxNode = &onnx_ir.NodeProto{
		//		OpType: "Div",
		//		Input:  []string{n.Children[0].Name, n.Children[1].Name},
		//		Output: []string{n.output.Name},
		//	}
		//case *InputNode:
		//	// Skip input nodes (no operation needed)
		//	continue
		//default:
		//	return nil, fmt.Errorf("unsupported node type: %T", node)
		//}
		//
		if node.GetONNXNodeInfo().ProducedTensor != true {
			continue
		}
		nodeType = node.GetONNXNodeInfo().Name
		onnxNode = &onnx_ir.NodeProto{
			OpType: nodeType,
			Input:  []string{node.GetChildren()[0].GetName(), node.GetChildren()[1].GetName()},
			Output: []string{node.GetOutput().Name},
		}

		//log.Println(onnxNode)

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
