package compute_graph

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/network"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/testing"
	ort "github.com/yalue/onnxruntime_go"
	_ "image/gif"
	_ "image/jpeg"
	"os"
	"path/filepath"
	"runtime"
)

type ONNXRuntime struct {
}

func getDefaultSharedLibPath() string {
	var libPatterns []string

	switch runtime.GOOS {
	case "windows":
		libPatterns = []string{"onnxruntime*.dll"}
	case "darwin":
		libPatterns = []string{"onnxruntime*.dylib"}
	case "linux":
		libPatterns = []string{"onnxruntime*.so"}
	default:
		fmt.Printf("Unsupported OS: %s\n", runtime.GOOS)
		return ""
	}

	pathDirs := filepath.SplitList(os.Getenv("PATH"))

	var foundLibs []string
	for _, dir := range pathDirs {
		for _, pattern := range libPatterns {
			matches, _ := filepath.Glob(filepath.Join(dir, pattern))
			for _, m := range matches {
				if _, err := os.Stat(m); err == nil {
					foundLibs = append(foundLibs, m)
				}
			}
		}
	}

	for _, libPath := range foundLibs {
		{
			ort.SetSharedLibraryPath(libPath)
			e := ort.InitializeEnvironment()
			if e == nil {
				fmt.Println("onnxruntime path: ", libPath, "init successes ", "version: ", ort.GetVersion())
				ort.DestroyEnvironment()
				return libPath
			}
			fmt.Println("onnxruntime path: ", libPath, "init failed: ", e.Error())
			ort.DestroyEnvironment()
		}
	}

	panic("No ONNXRuntime candidates found")
}

var ONNXRuntimeVal *ONNXRuntime

func NewOnnx() *ONNXRuntime {
	if ONNXRuntimeVal != nil {
		return ONNXRuntimeVal
	}
	ort.SetSharedLibraryPath(getDefaultSharedLibPath())
	ort.DestroyEnvironment()
	e := ort.InitializeEnvironment()
	if e != nil {
		panic(fmt.Errorf("error initializing the onnxruntime library: %w\n", e))
	}
	ONNXRuntimeVal = &ONNXRuntime{}
	return ONNXRuntimeVal
}

func (m *ONNXRuntime) Destroy() {
	ort.DestroyEnvironment()
}

func (m *ONNXRuntime) NewOneTimeSessionTest(graph *ComputationalGraph) (outPutTensorList []*tensor.Tensor) {
	var inputNameList []string
	var outputNameList []string
	var inputNameOrtList []ort.ArbitraryTensor
	var outputNameOrtList []ort.ArbitraryTensor
	var inputTensorList []*ort.Tensor[float32]
	var outTensorList []*ort.Tensor[float32]
	{
		inputNameNodeList := graph.Network.GetInput()
		outputNameNodeList := graph.Network.GetOutput()
		for _, inputName := range inputNameNodeList {
			inputNameList = append(inputNameList, inputName.Name)
			t := graph.GetTensorByName(inputName.Name)
			var shape []int64
			{
				for _, s := range t.value.Shape() {
					shape = append(shape, int64(s))
				}
			}
			xTmp, err := ort.NewTensor(ort.NewShape(shape...), t.value.Data)
			if err != nil {
				panic(err)
			}
			inputTensorList = append(inputTensorList, xTmp)
			inputNameOrtList = append(inputNameOrtList, xTmp)
		}

		for _, outputName := range outputNameNodeList {
			outputNameList = append(outputNameList, outputName.Name)
			var shape []int64
			{
				t := graph.GetTensorByName(outputName.Name)
				{
					for _, s := range t.value.Shape() {
						shape = append(shape, int64(s))
					}
				}
			}

			xTmp, err := ort.NewEmptyTensor[float32](ort.NewShape(shape...))
			if err != nil {
				panic(err)
			}
			outTensorList = append(outTensorList, xTmp)
			outputNameOrtList = append(outputNameOrtList, xTmp)
		}

	}

	onnxModel, err := graph.ToONNXModel()
	if err != nil {
		panic(err)
	}

	tempFileName := testing.CreateTempFileName("onnx_runtime.*.onnx")
	err = onnxModel.SaveONNX(tempFileName)
	if err != nil {
		panic(err)
	}
	fmt.Println(tempFileName)

	{
		fmt.Println("-----ONNX-----")
		fmt.Print("INPUT: ")
		for i := 0; i < len(inputNameOrtList); i++ {
			fmt.Print(inputNameList[i], ": ", inputNameOrtList[i].GetShape(), "\t")
		}
		fmt.Println()
		fmt.Print("OUTPUT: ")
		for i := 0; i < len(outputNameOrtList); i++ {
			fmt.Print(outputNameList[i], ": ", outputNameOrtList[i].GetShape(), "\t")
		}
		fmt.Println()
		fmt.Println("-----ONNX END-----")
	}

	session, e := ort.NewAdvancedSession(tempFileName, inputNameList, outputNameList, inputNameOrtList, outputNameOrtList, nil)
	if e != nil {
		panic(fmt.Sprintf("error creating  network session: %v\n", e))
	}

	e = session.Run()
	if e != nil {
		panic(fmt.Sprintf("error running the  network: %v\n", e))
	}

	{
		for i := 0; i < len(outTensorList); i++ {
			var shape []int
			s := outTensorList[i].GetShape()
			for ii := 0; ii < len(s); ii++ {
				shape = append(shape, int(s[ii]))
			}
			fmt.Println("shape:", shape)
			fmt.Println("data:", outTensorList[i].GetData())
			originalData := outTensorList[i].GetData()
			outPutTensorList = append(outPutTensorList, tensor.NewTensor(originalData, shape))
		}
	}

	{

		for i := 0; i < len(inputNameOrtList); i++ {
			inputNameOrtList[i].Destroy()
		}
		for i := 0; i < len(outputNameOrtList); i++ {
			outputNameOrtList[i].Destroy()
		}

	}
	session.Destroy()
	return
}

func (m *ONNXRuntime) NewOneTimeSessionTestByNode(inGraph *ComputationalGraph, inNode *network.Node) (outPutTensorList []*tensor.Tensor) {
	var outGraph *ComputationalGraph
	{
		outGraph = NewComputationalGraph()
		outGraph.ONNXAttributePool = inGraph.ONNXAttributePool
		nodeCurrent := outGraph.Network.NewNode()
		nodeCurrent.Name = inNode.Name
		nodeCurrent.Type = inNode.Type
		{
			for _, v := range inNode.Inputs {
				node := outGraph.Network.NewNode()
				node.Name = v.Name
				node.Type = "Tensor_Input"
				nodeCurrent.AddInput(node)
				outGraph.Network.AddInput(node)
			}

			for _, v := range inNode.Outputs {
				node := outGraph.Network.NewNode()
				node.Name = v.Name
				node.Type = "Tensor_Output"
				nodeCurrent.AddOutput(node)
				outGraph.Network.AddOutput(node)
			}
		}

		{
			for _, v := range inNode.GetInputName() {
				outGraph.Tensors[v] = inGraph.GetTensorByName(v)
			}

			for _, v := range inNode.GetOutputName() {
				outGraph.Tensors[v] = inGraph.GetTensorByName(v)
			}
		}

		{
			for _, v := range inNode.GetInputName() {
				outGraph.Nodes = append(outGraph.Nodes, inGraph.GetNodeByName(v))
			}

			for _, v := range inNode.GetOutputName() {
				outGraph.Nodes = append(outGraph.Nodes, inGraph.GetNodeByName(v))
			}

			outGraph.Nodes = append(outGraph.Nodes, inGraph.GetNodeByName(inNode.Name))
		}

	}

	return m.NewOneTimeSessionTest(outGraph)
}
