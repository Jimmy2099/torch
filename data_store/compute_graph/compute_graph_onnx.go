package compute_graph

import (
	"fmt"
	onnx_ir "github.com/Jimmy2099/torch/thirdparty/onnx-go/ir"
	v1proto "github.com/golang/protobuf/proto"
	"io/ioutil"
)

type ONNXOperator struct {
	Name         string
	InputPCount  int
	OutputPCount int
}

var ONNXOperators = []ONNXOperator{
	//ai.onnx
	{Name: "Abs", InputPCount: 1, OutputPCount: 1},
	{Name: "Acos", InputPCount: 1, OutputPCount: 1},
	{Name: "Acosh", InputPCount: 1, OutputPCount: 1},
	{Name: "Add", InputPCount: 1, OutputPCount: 1},
	{Name: "AffineGrid", InputPCount: 1, OutputPCount: 1},
	{Name: "And", InputPCount: 1, OutputPCount: 1},
	{Name: "ArgMax", InputPCount: 1, OutputPCount: 1},
	{Name: "ArgMin", InputPCount: 1, OutputPCount: 1},
	{Name: "Asin", InputPCount: 1, OutputPCount: 1},
	{Name: "Asinh", InputPCount: 1, OutputPCount: 1},
	{Name: "Atan", InputPCount: 1, OutputPCount: 1},
	{Name: "Atanh", InputPCount: 1, OutputPCount: 1},
	{Name: "Attention", InputPCount: 1, OutputPCount: 1},
	{Name: "AveragePool", InputPCount: 1, OutputPCount: 1},
	{Name: "BatchNormalization", InputPCount: 1, OutputPCount: 1},
	{Name: "Bernoulli", InputPCount: 1, OutputPCount: 1},
	{Name: "BitShift", InputPCount: 1, OutputPCount: 1},
	{Name: "BitwiseAnd", InputPCount: 1, OutputPCount: 1},
	{Name: "BitwiseNot", InputPCount: 1, OutputPCount: 1},
	{Name: "BitwiseOr", InputPCount: 1, OutputPCount: 1},
	{Name: "BitwiseXor", InputPCount: 1, OutputPCount: 1},
	{Name: "BlackmanWindow", InputPCount: 1, OutputPCount: 1},
	{Name: "Cast", InputPCount: 1, OutputPCount: 1},
	{Name: "CastLike", InputPCount: 1, OutputPCount: 1},
	{Name: "Ceil", InputPCount: 1, OutputPCount: 1},
	{Name: "Celu", InputPCount: 1, OutputPCount: 1},
	{Name: "CenterCropPad", InputPCount: 1, OutputPCount: 1},
	{Name: "Clip", InputPCount: 1, OutputPCount: 1},
	{Name: "Col2Im", InputPCount: 1, OutputPCount: 1},
	{Name: "Compress", InputPCount: 1, OutputPCount: 1},
	{Name: "Concat", InputPCount: 1, OutputPCount: 1},
	{Name: "ConcatFromSequence", InputPCount: 1, OutputPCount: 1},
	{Name: "Constant", InputPCount: 1, OutputPCount: 1},
	{Name: "ConstantOfShape", InputPCount: 1, OutputPCount: 1},
	{Name: "Conv", InputPCount: 1, OutputPCount: 1},
	{Name: "ConvInteger", InputPCount: 1, OutputPCount: 1},
	{Name: "ConvTranspose", InputPCount: 1, OutputPCount: 1},
	{Name: "Cos", InputPCount: 1, OutputPCount: 1},
	{Name: "Cosh", InputPCount: 1, OutputPCount: 1},
	{Name: "CumSum", InputPCount: 1, OutputPCount: 1},
	{Name: "DFT", InputPCount: 1, OutputPCount: 1},
	{Name: "DeformConv", InputPCount: 1, OutputPCount: 1},
	{Name: "DepthToSpace", InputPCount: 1, OutputPCount: 1},
	{Name: "DequantizeLinear", InputPCount: 1, OutputPCount: 1},
	{Name: "Det", InputPCount: 1, OutputPCount: 1},
	{Name: "Div", InputPCount: 1, OutputPCount: 1},
	{Name: "Dropout", InputPCount: 1, OutputPCount: 1},
	{Name: "DynamicQuantizeLinear", InputPCount: 1, OutputPCount: 1},
	{Name: "Einsum", InputPCount: 1, OutputPCount: 1},
	{Name: "Elu", InputPCount: 1, OutputPCount: 1},
	{Name: "Equal", InputPCount: 1, OutputPCount: 1},
	{Name: "Erf", InputPCount: 1, OutputPCount: 1},
	{Name: "Exp", InputPCount: 1, OutputPCount: 1},
	{Name: "Expand", InputPCount: 1, OutputPCount: 1},
	{Name: "EyeLike", InputPCount: 1, OutputPCount: 1},
	{Name: "Flatten", InputPCount: 1, OutputPCount: 1},
	{Name: "Floor", InputPCount: 1, OutputPCount: 1},
	{Name: "GRU", InputPCount: 1, OutputPCount: 1},
	{Name: "Gather", InputPCount: 1, OutputPCount: 1},
	{Name: "GatherElements", InputPCount: 1, OutputPCount: 1},
	{Name: "GatherND", InputPCount: 1, OutputPCount: 1},
	{Name: "Gelu", InputPCount: 1, OutputPCount: 1},
	{Name: "Gemm", InputPCount: 1, OutputPCount: 1},
	{Name: "GlobalAveragePool", InputPCount: 1, OutputPCount: 1},
	{Name: "GlobalLpPool", InputPCount: 1, OutputPCount: 1},
	{Name: "GlobalMaxPool", InputPCount: 1, OutputPCount: 1},
	{Name: "Greater", InputPCount: 1, OutputPCount: 1},
	{Name: "GreaterOrEqual", InputPCount: 1, OutputPCount: 1},
	{Name: "GridSample", InputPCount: 1, OutputPCount: 1},
	{Name: "GroupNormalization", InputPCount: 1, OutputPCount: 1},
	{Name: "HammingWindow", InputPCount: 1, OutputPCount: 1},
	{Name: "HannWindow", InputPCount: 1, OutputPCount: 1},
	{Name: "HardSigmoid", InputPCount: 1, OutputPCount: 1},
	{Name: "HardSwish", InputPCount: 1, OutputPCount: 1},
	{Name: "Hardmax", InputPCount: 1, OutputPCount: 1},
	{Name: "Identity", InputPCount: 1, OutputPCount: 1},
	{Name: "If", InputPCount: 1, OutputPCount: 1},
	{Name: "ImageDecoder", InputPCount: 1, OutputPCount: 1},
	{Name: "InstanceNormalization", InputPCount: 1, OutputPCount: 1},
	{Name: "IsInf", InputPCount: 1, OutputPCount: 1},
	{Name: "IsNaN", InputPCount: 1, OutputPCount: 1},
	{Name: "LRN", InputPCount: 1, OutputPCount: 1},
	{Name: "LSTM", InputPCount: 1, OutputPCount: 1},
	{Name: "LayerNormalization", InputPCount: 1, OutputPCount: 1},
	{Name: "LeakyRelu", InputPCount: 1, OutputPCount: 1},
	{Name: "Less", InputPCount: 1, OutputPCount: 1},
	{Name: "LessOrEqual", InputPCount: 1, OutputPCount: 1},
	{Name: "Log", InputPCount: 1, OutputPCount: 1},
	{Name: "LogSoftmax", InputPCount: 1, OutputPCount: 1},
	{Name: "Loop", InputPCount: 1, OutputPCount: 1},
	{Name: "LpNormalization", InputPCount: 1, OutputPCount: 1},
	{Name: "LpPool", InputPCount: 1, OutputPCount: 1},
	{Name: "MatMul", InputPCount: 1, OutputPCount: 1},
	{Name: "MatMulInteger", InputPCount: 1, OutputPCount: 1},
	{Name: "Max", InputPCount: 1, OutputPCount: 1},
	{Name: "MaxPool", InputPCount: 1, OutputPCount: 1},
	{Name: "MaxRoiPool", InputPCount: 1, OutputPCount: 1},
	{Name: "MaxUnpool", InputPCount: 1, OutputPCount: 1},
	{Name: "Mean", InputPCount: 1, OutputPCount: 1},
	{Name: "MeanVarianceNormalization", InputPCount: 1, OutputPCount: 1},
	{Name: "MelWeightMatrix", InputPCount: 1, OutputPCount: 1},
	{Name: "Min", InputPCount: 1, OutputPCount: 1},
	{Name: "Mish", InputPCount: 1, OutputPCount: 1},
	{Name: "Mod", InputPCount: 1, OutputPCount: 1},
	{Name: "Mul", InputPCount: 1, OutputPCount: 1},
	{Name: "Multinomial", InputPCount: 1, OutputPCount: 1},
	{Name: "Neg", InputPCount: 1, OutputPCount: 1},
	{Name: "NegativeLogLikelihoodLoss", InputPCount: 1, OutputPCount: 1},
	{Name: "NonMaxSuppression", InputPCount: 1, OutputPCount: 1},
	{Name: "NonZero", InputPCount: 1, OutputPCount: 1},
	{Name: "Not", InputPCount: 1, OutputPCount: 1},
	{Name: "OneHot", InputPCount: 1, OutputPCount: 1},
	{Name: "Optional", InputPCount: 1, OutputPCount: 1},
	{Name: "OptionalGetElement", InputPCount: 1, OutputPCount: 1},
	{Name: "OptionalHasElement", InputPCount: 1, OutputPCount: 1},
	{Name: "Or", InputPCount: 1, OutputPCount: 1},
	{Name: "PRelu", InputPCount: 1, OutputPCount: 1},
	{Name: "Pad", InputPCount: 1, OutputPCount: 1},
	{Name: "Pow", InputPCount: 1, OutputPCount: 1},
	{Name: "QLinearConv", InputPCount: 1, OutputPCount: 1},
	{Name: "QLinearMatMul", InputPCount: 1, OutputPCount: 1},
	{Name: "QuantizeLinear", InputPCount: 1, OutputPCount: 1},
	{Name: "RMSNormalization", InputPCount: 1, OutputPCount: 1},
	{Name: "RNN", InputPCount: 1, OutputPCount: 1},
	{Name: "RandomNormal", InputPCount: 1, OutputPCount: 1},
	{Name: "RandomNormalLike", InputPCount: 1, OutputPCount: 1},
	{Name: "RandomUniform", InputPCount: 1, OutputPCount: 1},
	{Name: "RandomUniformLike", InputPCount: 1, OutputPCount: 1},
	{Name: "Range", InputPCount: 1, OutputPCount: 1},
	{Name: "Reciprocal", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceL1", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceL2", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceLogSum", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceLogSumExp", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceMax", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceMean", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceMin", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceProd", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceSum", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceSumSquare", InputPCount: 1, OutputPCount: 1},
	{Name: "RegexFullMatch", InputPCount: 1, OutputPCount: 1},
	{Name: "Relu", InputPCount: 1, OutputPCount: 1},
	{Name: "Reshape", InputPCount: 1, OutputPCount: 1},
	{Name: "Resize", InputPCount: 1, OutputPCount: 1},
	{Name: "ReverseSequence", InputPCount: 1, OutputPCount: 1},
	{Name: "RoiAlign", InputPCount: 1, OutputPCount: 1},
	{Name: "RotaryEmbedding", InputPCount: 1, OutputPCount: 1},
	{Name: "Round", InputPCount: 1, OutputPCount: 1},
	{Name: "STFT", InputPCount: 1, OutputPCount: 1},
	{Name: "Scan", InputPCount: 1, OutputPCount: 1},
	{Name: "Scatter", InputPCount: 1, OutputPCount: 1},
	{Name: "ScatterElements", InputPCount: 1, OutputPCount: 1},
	{Name: "ScatterND", InputPCount: 1, OutputPCount: 1},
	{Name: "Selu", InputPCount: 1, OutputPCount: 1},
	{Name: "SequenceAt", InputPCount: 1, OutputPCount: 1},
	{Name: "SequenceConstruct", InputPCount: 1, OutputPCount: 1},
	{Name: "SequenceEmpty", InputPCount: 1, OutputPCount: 1},
	{Name: "SequenceErase", InputPCount: 1, OutputPCount: 1},
	{Name: "SequenceInsert", InputPCount: 1, OutputPCount: 1},
	{Name: "SequenceLength", InputPCount: 1, OutputPCount: 1},
	{Name: "SequenceMap", InputPCount: 1, OutputPCount: 1},
	{Name: "Shape", InputPCount: 1, OutputPCount: 1},
	{Name: "Shrink", InputPCount: 1, OutputPCount: 1},
	{Name: "Sigmoid", InputPCount: 1, OutputPCount: 1},
	{Name: "Sign", InputPCount: 1, OutputPCount: 1},
	{Name: "Sin", InputPCount: 1, OutputPCount: 1},
	{Name: "Sinh", InputPCount: 1, OutputPCount: 1},
	{Name: "Size", InputPCount: 1, OutputPCount: 1},
	{Name: "Slice", InputPCount: 1, OutputPCount: 1},
	{Name: "Softmax", InputPCount: 1, OutputPCount: 1},
	{Name: "SoftmaxCrossEntropyLoss", InputPCount: 1, OutputPCount: 1},
	{Name: "Softplus", InputPCount: 1, OutputPCount: 1},
	{Name: "Softsign", InputPCount: 1, OutputPCount: 1},
	{Name: "SpaceToDepth", InputPCount: 1, OutputPCount: 1},
	{Name: "Split", InputPCount: 1, OutputPCount: 1},
	{Name: "SplitToSequence", InputPCount: 1, OutputPCount: 1},
	{Name: "Sqrt", InputPCount: 1, OutputPCount: 1},
	{Name: "Squeeze", InputPCount: 1, OutputPCount: 1},
	{Name: "StringConcat", InputPCount: 1, OutputPCount: 1},
	{Name: "StringNormalizer", InputPCount: 1, OutputPCount: 1},
	{Name: "StringSplit", InputPCount: 1, OutputPCount: 1},
	{Name: "Sub", InputPCount: 1, OutputPCount: 1},
	{Name: "Sum", InputPCount: 1, OutputPCount: 1},
	{Name: "Swish", InputPCount: 1, OutputPCount: 1},
	{Name: "Tan", InputPCount: 1, OutputPCount: 1},
	{Name: "Tanh", InputPCount: 1, OutputPCount: 1},
	{Name: "TensorScatter", InputPCount: 1, OutputPCount: 1},
	{Name: "TfIdfVectorizer", InputPCount: 1, OutputPCount: 1},
	{Name: "ThresholdedRelu", InputPCount: 1, OutputPCount: 1},
	{Name: "Tile", InputPCount: 1, OutputPCount: 1},
	{Name: "TopK", InputPCount: 1, OutputPCount: 1},
	{Name: "Transpose", InputPCount: 1, OutputPCount: 1},
	{Name: "Trilu", InputPCount: 1, OutputPCount: 1},
	{Name: "Unique", InputPCount: 1, OutputPCount: 1},
	{Name: "Unsqueeze", InputPCount: 1, OutputPCount: 1},
	{Name: "Upsample", InputPCount: 1, OutputPCount: 1},
	{Name: "Where", InputPCount: 1, OutputPCount: 1},
	{Name: "Xor", InputPCount: 1, OutputPCount: 1},
	//ai.onnx.ml
	{Name: "ArrayFeatureExtractor", InputPCount: 1, OutputPCount: 1},
	{Name: "Binarizer", InputPCount: 1, OutputPCount: 1},
	{Name: "CastMap", InputPCount: 1, OutputPCount: 1},
	{Name: "CategoryMapper", InputPCount: 1, OutputPCount: 1},
	{Name: "DictVectorizer", InputPCount: 1, OutputPCount: 1},
	{Name: "FeatureVectorizer", InputPCount: 1, OutputPCount: 1},
	{Name: "Imputer", InputPCount: 1, OutputPCount: 1},
	{Name: "LabelEncoder", InputPCount: 1, OutputPCount: 1},
	{Name: "LinearClassifier", InputPCount: 1, OutputPCount: 1},
	{Name: "LinearRegressor", InputPCount: 1, OutputPCount: 1},
	{Name: "Normalizer", InputPCount: 1, OutputPCount: 1},
	{Name: "OneHotEncoder", InputPCount: 1, OutputPCount: 1},
	{Name: "SVMClassifier", InputPCount: 1, OutputPCount: 1},
	{Name: "SVMRegressor", InputPCount: 1, OutputPCount: 1},
	{Name: "Scaler", InputPCount: 1, OutputPCount: 1},
	{Name: "TreeEnsemble", InputPCount: 1, OutputPCount: 1},
	{Name: "TreeEnsembleClassifier", InputPCount: 1, OutputPCount: 1},
	{Name: "TreeEnsembleRegressor", InputPCount: 1, OutputPCount: 1},
	{Name: "ZipMap", InputPCount: 1, OutputPCount: 1},
	//ai.onnx.preview.training
	{Name: "Adagrad", InputPCount: 1, OutputPCount: 1},
	{Name: "Adam", InputPCount: 1, OutputPCount: 1},
	{Name: "Gradient", InputPCount: 1, OutputPCount: 1},
	{Name: "Momentum", InputPCount: 1, OutputPCount: 1},
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
