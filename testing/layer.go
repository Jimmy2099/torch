package testing

import (
	"fmt"
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"os"
	"path/filepath"
	"strings"
)

func GetLayerTestResult(inPyScript string, layer torch.LayerForTesting, inTensor *tensor.Tensor) *tensor.Tensor {
	return GetLayerTestResult64(inPyScript, layer, inTensor, 64)
}
func GetLayerTestResult32(inPyScript string, layer torch.LayerForTesting, inTensor *tensor.Tensor) *tensor.Tensor {
	return GetLayerTestResult64(inPyScript, layer, inTensor, 32)
}
func GetLayerTestResult64(inPyScript string, layer torch.LayerForTesting, inTensor *tensor.Tensor, dataType int) *tensor.Tensor {
	var layerWeights *os.File
	var layerBias *os.File
	var err error
	{
		layerWeights, err = os.CreateTemp("", "layer_weights.*.csv")
		if err != nil {
			panic(err)
		}
		layerWeights.Close()
		//defer os.Remove(layerWeights.Name())
		layerBias, err = os.CreateTemp("", "layer_bias.*.csv")
		if err != nil {
			panic(err)
		}
		layerBias.Close()
		//defer os.Remove(layerBias.Name())

		WeightsTensor := layer.GetWeights()
		BiasTensor := layer.GetBias()

		err = WeightsTensor.SaveToCSV(layerWeights.Name())
		if err != nil {
			panic(err)
		}
		err = BiasTensor.SaveToCSV(layerBias.Name())
		if err != nil {
			panic(err)
		}
	}

	var inFile2 *os.File
	{
		inFile2, err = os.CreateTemp("", "input_tensor.*.csv")
		if err != nil {
			panic(err)
		}
		inFile2.Close()
		//defer os.Remove(inFile2.Name())
		err = inTensor.SaveToCSV(inFile2.Name())
		if err != nil {
			panic(err)
		}
	}

	var outFile *os.File
	{
		outFile, err = os.CreateTemp("", "out_tensor.*.csv")
		if err != nil {
			panic(err)
		}
		outFile.Close()
		//defer os.Remove(outFile.Name())
	}

	layerWeightsPath := filepath.ToSlash(layerWeights.Name())
	layerBiasPath := filepath.ToSlash(layerBias.Name())

	inTensorPath := filepath.ToSlash(inFile2.Name())

	outPutPath := filepath.ToSlash(outFile.Name())
	inPyScript = strings.TrimSpace(inPyScript)
	// Python 脚本
	pythonScript := fmt.Sprintf(`
import numpy as np
import torch

def save_tensor_to_csv(tensor, file_path):
    with open(file_path, 'w') as f:
        f.write("Shape," + ",".join(map(str, tensor.shape)) + "\n")
        tensor = tensor.reshape(-1, tensor.shape[0])
        np.savetxt(f, tensor.numpy(), delimiter="," , fmt="%%.16f")

def load_tensor_from_csv(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        if not header.startswith("Shape,"):
            raise ValueError("Invalid CSV format: missing shape header")
        
        shape = list(map(int, header.split(",")[1:]))
        data = np.loadtxt(f, delimiter=",")
    
    flattened = data.flatten()
    return torch.tensor(flattened, dtype=torch.float%d).reshape(*shape)

weight=None
bias=None
try:
    weight = load_tensor_from_csv("%s")
except Exception as e:
    print(e)
try:
    bias = load_tensor_from_csv("%s")
except Exception as e:
    print(e)

in1 = load_tensor_from_csv("%s")

def process_data(weight,bias,in1):
    out = None
    #out = in1*2
    #return out
    layer = %s
    if not weight==None:
        layer.weight.data = weight
    if not bias==None:
        layer.bias.data = bias
    out = layer(in1)
    return out.detach()

out = process_data(weight,bias,in1)

save_tensor_to_csv(out,"%s")
`, dataType, layerWeightsPath, layerBiasPath, inTensorPath, inPyScript, outPutPath)

	RunPyScript(pythonScript)

	// 读取计算结果
	outTensor, err := tensor.LoadFromCSV(outPutPath)
	if err != nil {
		panic(err)
	}
	return outTensor
}
