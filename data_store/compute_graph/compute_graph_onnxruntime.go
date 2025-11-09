package compute_graph

import (
	"fmt"
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
			ort.DestroyEnvironment()
			if e == nil {
				fmt.Println("onnxruntime path: ", libPath)
				return libPath
			}
		}
	}

	panic("No ONNXRuntime candidates found")
}

func NewOnnx() *ONNXRuntime {
	ort.SetSharedLibraryPath(getDefaultSharedLibPath())
	e := ort.InitializeEnvironment()
	if e != nil {
		panic(fmt.Errorf("error initializing the onnxruntime library: %w\n", e))
	}
	return &ONNXRuntime{}
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

	session, e := ort.NewAdvancedSession(tempFileName, //"c:\\onnxruntime_go_examples\\model.onnx",
		inputNameList, outputNameList,
		inputNameOrtList, outputNameOrtList, nil)
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
			outPutTensorList = append(outPutTensorList, tensor.NewTensor(outTensorList[i].GetData(), shape))
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
