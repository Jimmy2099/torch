package compute_graph

import (
	"fmt"
	onnx_ir "github.com/Jimmy2099/torch/thirdparty/onnx-go/ir"
	v1proto "github.com/golang/protobuf/proto"
	"io/ioutil"
)

type ONNXOperator struct {
	Name             string
	InputPCount      int
	OutputPCount     int
	Ignore           bool
	AliasList        []string
	NodeRegistryFunc func(name string, children []*GraphTensor, output *GraphTensor) Node
}

var ONNXOperators = []ONNXOperator{
	{Name: "Input", InputPCount: 0, OutputPCount: 1, Ignore: false, AliasList: []string{}, NodeRegistryFunc: func(name string, children []*GraphTensor, output *GraphTensor) Node {
		return &InputNode{Name: name, output: output}
	}},
	{Name: "Abs", InputPCount: 1, OutputPCount: 1},
	{Name: "Acos", InputPCount: 1, OutputPCount: 1},
	{Name: "Acosh", InputPCount: 1, OutputPCount: 1},
	{Name: "Add", InputPCount: 2, OutputPCount: 1, NodeRegistryFunc: func(name string, children []*GraphTensor, output *GraphTensor) Node {
		m := NewAdd(name, children[0], children[1])
		m.output = output
		return m
	}},
	{Name: "AffineGrid", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "And", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "ArgMax", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "ArgMin", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Asin", InputPCount: 1, OutputPCount: 1},
	{Name: "Asinh", InputPCount: 1, OutputPCount: 1},
	{Name: "Atan", InputPCount: 1, OutputPCount: 1},
	{Name: "Atanh", InputPCount: 1, OutputPCount: 1},
	{Name: "Attention", InputPCount: 3, OutputPCount: 1},
	{Name: "AveragePool", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "BatchNormalization", InputPCount: 5, OutputPCount: 3},
	{Name: "Bernoulli", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "BitShift", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "BitwiseAnd", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "BitwiseNot", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "BitwiseOr", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "BitwiseXor", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "BlackmanWindow", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Cast", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "CastLike", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Ceil", InputPCount: 1, OutputPCount: 1},
	{Name: "Celu", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "CenterCropPad", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Clip", InputPCount: 3, OutputPCount: 1},
	{Name: "Col2Im", InputPCount: 4, OutputPCount: 1},
	{Name: "Compress", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Concat", InputPCount: -1, OutputPCount: 1},
	{Name: "ConcatFromSequence", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Constant", InputPCount: 0, OutputPCount: 1},
	{Name: "ConstantOfShape", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Conv", InputPCount: 3, OutputPCount: 1},
	{Name: "ConvInteger", InputPCount: 4, OutputPCount: 1},
	{Name: "ConvTranspose", InputPCount: 3, OutputPCount: 1},
	{Name: "Cos", InputPCount: 1, OutputPCount: 1},
	{Name: "Cosh", InputPCount: 1, OutputPCount: 1},
	{Name: "CumSum", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "DFT", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "DeformConv", InputPCount: 4, OutputPCount: 1},
	{Name: "DepthToSpace", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "DequantizeLinear", InputPCount: 3, OutputPCount: 1},
	{Name: "Det", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Div", InputPCount: 2, OutputPCount: 1},
	{Name: "Dropout", InputPCount: 1, OutputPCount: 2},
	{Name: "DynamicQuantizeLinear", InputPCount: 1, OutputPCount: 3},
	{Name: "Einsum", InputPCount: -1, OutputPCount: 1, Ignore: true},
	{Name: "Elu", InputPCount: 1, OutputPCount: 1},
	{Name: "Equal", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Erf", InputPCount: 1, OutputPCount: 1},
	{Name: "Exp", InputPCount: 1, OutputPCount: 1},
	{Name: "Expand", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "EyeLike", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Flatten", InputPCount: 1, OutputPCount: 1},
	{Name: "Floor", InputPCount: 1, OutputPCount: 1},
	{Name: "GRU", InputPCount: 6, OutputPCount: 2},
	{Name: "Gather", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "GatherElements", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "GatherND", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Gelu", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Gemm", InputPCount: 3, OutputPCount: 1},
	{Name: "GlobalAveragePool", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "GlobalLpPool", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "GlobalMaxPool", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Greater", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "GreaterOrEqual", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "GridSample", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "GroupNormalization", InputPCount: 3, OutputPCount: 1},
	{Name: "HammingWindow", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "HannWindow", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "HardSigmoid", InputPCount: 1, OutputPCount: 1},
	{Name: "HardSwish", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Hardmax", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Identity", InputPCount: 1, OutputPCount: 1},
	{Name: "If", InputPCount: -1, OutputPCount: -1},
	{Name: "ImageDecoder", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "InstanceNormalization", InputPCount: 3, OutputPCount: 1},
	{Name: "IsInf", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "IsNaN", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "LRN", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "LSTM", InputPCount: 8, OutputPCount: 3},
	{Name: "LayerNormalization", InputPCount: 3, OutputPCount: 1},
	{Name: "LeakyRelu", InputPCount: 1, OutputPCount: 1},
	{Name: "Less", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "LessOrEqual", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Log", InputPCount: 1, OutputPCount: 1},
	{Name: "LogSoftmax", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Loop", InputPCount: -1, OutputPCount: -1},
	{Name: "LpNormalization", InputPCount: 1, OutputPCount: 1},
	{Name: "LpPool", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "MatMul", InputPCount: 2, OutputPCount: 1},
	{Name: "MatMulInteger", InputPCount: 4, OutputPCount: 1},
	{Name: "Max", InputPCount: -1, OutputPCount: 1, Ignore: true},
	{Name: "MaxPool", InputPCount: 1, OutputPCount: 2},
	{Name: "MaxRoiPool", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "MaxUnpool", InputPCount: 3, OutputPCount: 1},
	{Name: "Mean", InputPCount: -1, OutputPCount: 1},
	{Name: "MeanVarianceNormalization", InputPCount: 1, OutputPCount: 1},
	{Name: "MelWeightMatrix", InputPCount: 3, OutputPCount: 1},
	{Name: "Min", InputPCount: -1, OutputPCount: 1},
	{Name: "Mish", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Mod", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Mul", InputPCount: 2, OutputPCount: 1, AliasList: []string{"Multiply"}, NodeRegistryFunc: func(name string, children []*GraphTensor, output *GraphTensor) Node {
		m := NewMultiply(name, children[0], children[1])
		m.output = output
		return m
	}},
	{Name: "Multinomial", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Neg", InputPCount: 1, OutputPCount: 1},
	{Name: "NegativeLogLikelihoodLoss", InputPCount: 3, OutputPCount: 1},
	{Name: "NonMaxSuppression", InputPCount: 5, OutputPCount: 1},
	{Name: "NonZero", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Not", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "OneHot", InputPCount: 3, OutputPCount: 1},
	{Name: "Optional", InputPCount: 0, OutputPCount: 1},
	{Name: "OptionalGetElement", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "OptionalHasElement", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Or", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "PRelu", InputPCount: 2, OutputPCount: 1},
	{Name: "Pad", InputPCount: 3, OutputPCount: 1},
	{Name: "Pow", InputPCount: 2, OutputPCount: 1},
	{Name: "QLinearConv", InputPCount: 9, OutputPCount: 1},
	{Name: "QLinearMatMul", InputPCount: 8, OutputPCount: 1},
	{Name: "QuantizeLinear", InputPCount: 3, OutputPCount: 1},
	{Name: "RMSNormalization", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "RNN", InputPCount: 6, OutputPCount: 2},
	{Name: "RandomNormal", InputPCount: 0, OutputPCount: 1},
	{Name: "RandomNormalLike", InputPCount: 1, OutputPCount: 1},
	{Name: "RandomUniform", InputPCount: 0, OutputPCount: 1},
	{Name: "RandomUniformLike", InputPCount: 1, OutputPCount: 1},
	{Name: "Range", InputPCount: 3, OutputPCount: 1},
	{Name: "Reciprocal", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceL1", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceL2", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceLogSum", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceLogSumExp", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceMax", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceMean", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceMin", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceProd", InputPCount: 1, OutputPCount: 1},
	{Name: "ReduceSum", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "ReduceSumSquare", InputPCount: 1, OutputPCount: 1},
	{Name: "RegexFullMatch", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Relu", InputPCount: 1, OutputPCount: 1},
	{Name: "Reshape", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Resize", InputPCount: 4, OutputPCount: 1},
	{Name: "ReverseSequence", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "RoiAlign", InputPCount: 3, OutputPCount: 1},
	{Name: "RotaryEmbedding", InputPCount: 3, OutputPCount: 1},
	{Name: "Round", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "STFT", InputPCount: 4, OutputPCount: 1},
	{Name: "Scan", InputPCount: -1, OutputPCount: -1},
	{Name: "Scatter", InputPCount: 3, OutputPCount: 1},
	{Name: "ScatterElements", InputPCount: 3, OutputPCount: 1},
	{Name: "ScatterND", InputPCount: 3, OutputPCount: 1},
	{Name: "Selu", InputPCount: 1, OutputPCount: 1},
	{Name: "SequenceAt", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "SequenceConstruct", InputPCount: -1, OutputPCount: 1},
	{Name: "SequenceEmpty", InputPCount: 0, OutputPCount: 1},
	{Name: "SequenceErase", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "SequenceInsert", InputPCount: 3, OutputPCount: 1},
	{Name: "SequenceLength", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "SequenceMap", InputPCount: -1, OutputPCount: -1},
	{Name: "Shape", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Shrink", InputPCount: 1, OutputPCount: 1},
	{Name: "Sigmoid", InputPCount: 1, OutputPCount: 1, NodeRegistryFunc: func(name string, children []*GraphTensor, output *GraphTensor) Node {
		m := NewSigmoid(name, children[0])
		m.output = output
		return m
	}},
	{Name: "Sign", InputPCount: 1, OutputPCount: 1},
	{Name: "Sin", InputPCount: 1, OutputPCount: 1},
	{Name: "Sinh", InputPCount: 1, OutputPCount: 1},
	{Name: "Size", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Slice", InputPCount: 5, OutputPCount: 1},
	{Name: "Softmax", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "SoftmaxCrossEntropyLoss", InputPCount: 3, OutputPCount: 1},
	{Name: "Softplus", InputPCount: 1, OutputPCount: 1},
	{Name: "Softsign", InputPCount: 1, OutputPCount: 1},
	{Name: "SpaceToDepth", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Split", InputPCount: 2, OutputPCount: -1},
	{Name: "SplitToSequence", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Sqrt", InputPCount: 1, OutputPCount: 1},
	{Name: "Squeeze", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "StringConcat", InputPCount: -1, OutputPCount: 1},
	{Name: "StringNormalizer", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "StringSplit", InputPCount: 1, OutputPCount: 2},
	{Name: "Sub", InputPCount: 2, OutputPCount: 1},
	{Name: "Sum", InputPCount: -1, OutputPCount: 1},
	{Name: "Swish", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Tan", InputPCount: 1, OutputPCount: 1},
	{Name: "Tanh", InputPCount: 1, OutputPCount: 1},
	{Name: "TensorScatter", InputPCount: 3, OutputPCount: 1},
	{Name: "TfIdfVectorizer", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "ThresholdedRelu", InputPCount: 1, OutputPCount: 1},
	{Name: "Tile", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "TopK", InputPCount: 2, OutputPCount: 2},
	{Name: "Transpose", InputPCount: 1, OutputPCount: 1},
	{Name: "Trilu", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Unique", InputPCount: 1, OutputPCount: 4},
	{Name: "Unsqueeze", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Upsample", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Where", InputPCount: 3, OutputPCount: 1},
	{Name: "Xor", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "ArrayFeatureExtractor", InputPCount: 2, OutputPCount: 1, Ignore: true},
	{Name: "Binarizer", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "CastMap", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "CategoryMapper", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "DictVectorizer", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "FeatureVectorizer", InputPCount: -1, OutputPCount: 1},
	{Name: "Imputer", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "LabelEncoder", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "LinearClassifier", InputPCount: 1, OutputPCount: 2},
	{Name: "LinearRegressor", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Normalizer", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "OneHotEncoder", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "SVMClassifier", InputPCount: 1, OutputPCount: 2},
	{Name: "SVMRegressor", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Scaler", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "TreeEnsemble", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "TreeEnsembleClassifier", InputPCount: 1, OutputPCount: 2},
	{Name: "TreeEnsembleRegressor", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "ZipMap", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Adagrad", InputPCount: 5, OutputPCount: 3},
	{Name: "Adam", InputPCount: 6, OutputPCount: 3},
	{Name: "Gradient", InputPCount: 1, OutputPCount: 1, Ignore: true},
	{Name: "Momentum", InputPCount: 5, OutputPCount: 3},
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
	model.IrVersion = 7
	model.ProducerName = "BioTorch"
	model.ProducerVersion = "0.1"

	opset := &onnx_ir.OperatorSetIdProto{
		Domain:  "",
		Version: 11,
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
		nodeChildren := node.GetChildren()
		// Dynamically collect input names
		inputNames := make([]string, len(nodeChildren))
		for i, child := range nodeChildren {
			inputNames[i] = child.GetName()
		}

		onnxNode = &onnx_ir.NodeProto{
			OpType: nodeType,
			Input:  inputNames, // Use dynamically created slice
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
