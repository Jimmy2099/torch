package testing

import (
	"github.com/Jimmy2099/torch"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
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
		layerBias, err = os.CreateTemp("", "layer_bias.*.csv")
		if err != nil {
			panic(err)
		}
		layerBias.Close()

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
	}

	layerWeightsPath := filepath.ToSlash(layerWeights.Name())
	layerBiasPath := filepath.ToSlash(layerBias.Name())

	inTensorPath := filepath.ToSlash(inFile2.Name())

	outPutPath := filepath.ToSlash(outFile.Name())
	inPyScript = strings.TrimSpace(inPyScript)
	pythonScript := fmt.Sprintf(`
import numpy as np
import torch
torch.set_default_dtype(torch.float%d)

def save_tensor_to_csv(tensor, file_path):
    with open(file_path, 'w') as f:
        f.write("shape," + ",".join(map(str, tensor.shape)) + "\n")
        tensor = tensor.detach().numpy().astype(np.float32)
        np.savetxt(f, tensor.reshape(-1, tensor.shape[0]), 
                  delimiter=",", fmt="%%.16f")

def load_tensor_from_csv(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        if not header.startswith("shape,"):
            raise ValueError("Invalid CSV format: missing shape header")
        
        shape = list(map(int, header.split(",")[1:]))
        data = np.loadtxt(f, delimiter=",")
    
    flattened = data.flatten()#, dtype=torch.floatd
    return torch.tensor(flattened).reshape(*shape)

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
    if hasattr(layer, 'weight') and layer.weight is not None and not weight==None:
        layer.weight.data = weight
    if hasattr(layer, 'bias')   and  layer.bias  is not None and not bias==None:
        layer.bias.data = bias
    out = layer(in1)
    return out.detach()

out = process_data(weight,bias,in1)

save_tensor_to_csv(out,"%s")
`, dataType, layerWeightsPath, layerBiasPath, inTensorPath, inPyScript, outPutPath)

	RunPyScript(pythonScript)

	outTensor, err := tensor.LoadFromCSV(outPutPath)
	if err != nil {
		panic(err)
	}
	return outTensor
}

func GetPytorchInitData(pyScript string) *tensor.Tensor {

	var outFile *os.File
	{
		var err error
		outFile, err = os.CreateTemp("", "out_tensor.*.csv")
		if err != nil {
			panic(err)
		}
		outFile.Close()
	}

	outPutPath := filepath.ToSlash(outFile.Name())
	pythonScript := fmt.Sprintf(`
import numpy as np
import torch

def save_tensor_to_csv(tensor, file_path):
    with open(file_path, 'w') as f:
        f.write("shape," + ",".join(map(str, tensor.shape)) + "\n")
        tensor = tensor.reshape(-1, tensor.shape[0])
        np.savetxt(f, tensor.numpy(), delimiter="," , fmt="%%.16f")

def load_tensor_from_csv(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        if not header.startswith("shape,"):
            raise ValueError("Invalid CSV format: missing shape header")
        
        shape = list(map(int, header.split(",")[1:]))
        data = np.loadtxt(f, delimiter=",")
    
    flattened = data.flatten()
    return torch.tensor(flattened, dtype=torch.float32).reshape(*shape)

%s
out=out.detach()
save_tensor_to_csv(out,"%s")
`, pyScript, outPutPath)

	RunPyScript(pythonScript)

	outTensor, err := tensor.LoadFromCSV(outPutPath)
	if err != nil {
		panic(err)
	}
	return outTensor
}

func LayerTest(inPyScript string, inTensor *tensor.Tensor, dataType string) *tensor.Tensor {
	var err error
	var inFile2 *os.File
	{
		inFile2, err = os.CreateTemp("", "input_tensor.*.csv")
		if err != nil {
			panic(err)
		}
		inFile2.Close()
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
	}

	inTensorPath := filepath.ToSlash(inFile2.Name())

	outPutPath := filepath.ToSlash(outFile.Name())
	inPyScript = strings.TrimSpace(inPyScript)
	pythonScript := fmt.Sprintf(`
import numpy as np
import torch
torch.set_default_dtype(%s)

def save_tensor_to_csv(tensor, file_path):
    with open(file_path, 'w') as f:
        f.write("shape," + ",".join(map(str, tensor.shape)) + "\n")
        tensor = tensor.detach().numpy().astype(np.float32)
        np.savetxt(f, tensor.reshape(-1, tensor.shape[0]), 
                  delimiter=",", fmt="%%.16f")

def load_tensor_from_csv_with_shape(file_path,shape):
    with open(file_path, 'r') as f:
        data = np.loadtxt(f, delimiter=",")
    
    flattened = data.flatten()
    return torch.tensor(flattened).reshape(shape)

def load_tensor_from_csv(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        if not header.startswith("shape,"):
            raise ValueError("Invalid CSV format: missing shape header")
        
        shape = list(map(int, header.split(",")[1:]))
        data = np.loadtxt(f, delimiter=",")
    
    flattened = data.flatten()#, dtype=torch.floatd
    return torch.tensor(flattened).reshape(*shape)


in1 = load_tensor_from_csv("%s")

%s

out = out.detach().cpu().float()
save_tensor_to_csv(out,"%s")
`, dataType, inTensorPath, inPyScript, outPutPath)

	RunPyScript(pythonScript)

	outTensor, err := tensor.LoadFromCSV(outPutPath)
	if err != nil {
		panic(err)
	}
	return outTensor
}

func LayerTest1(inPyScript string, inTensors []*tensor.Tensor, dataType string) *tensor.Tensor {
	var err error
	var inFiles []*os.File
	for i := 0; i < len(inTensors); i++ {
		inFile2, err := os.CreateTemp("", "input_tensor.*.csv")
		if err != nil {
			panic(err)
		}
		inFile2.Close()
		err = inTensors[i].SaveToCSV(inFile2.Name())
		if err != nil {
			panic(err)
		}
		inFiles = append(inFiles, inFile2)
	}

	var outFile *os.File
	{
		outFile, err = os.CreateTemp("", "out_tensor.*.csv")
		if err != nil {
			panic(err)
		}
		outFile.Close()
	}

	textpy := ""

	for i := 0; i < len(inFiles); i++ {
		textpy += fmt.Sprintf(`in%d = load_tensor_from_csv("%s")`, i+1, filepath.ToSlash(inFiles[i].Name()))
		textpy += fmt.Sprintln()
	}

	outPutPath := filepath.ToSlash(outFile.Name())
	inPyScript = strings.TrimSpace(inPyScript)
	pythonScript := fmt.Sprintf(`
import numpy as np
import torch
torch.set_default_dtype(%s)

def save_tensor_to_csv(tensor, file_path):
    with open(file_path, 'w') as f:
        f.write("shape," + ",".join(map(str, tensor.shape)) + "\n")
        tensor = tensor.detach().numpy().astype(np.float32)
        np.savetxt(f, tensor.reshape(-1, tensor.shape[0]), 
                  delimiter=",", fmt="%%.16f")

def load_tensor_from_csv_with_shape(file_path,shape):
    with open(file_path, 'r') as f:
        data = np.loadtxt(f, delimiter=",")
    
    flattened = data.flatten()
    return torch.tensor(flattened).reshape(shape)

def load_tensor_from_csv(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        if not header.startswith("shape,"):
            raise ValueError("Invalid CSV format: missing shape header")
        
        shape = list(map(int, header.split(",")[1:]))
        data = np.loadtxt(f, delimiter=",")
    
    flattened = data.flatten()#, dtype=torch.floatd
    return torch.tensor(flattened).reshape(*shape)


%s

%s

out = out.detach().cpu().float()
save_tensor_to_csv(out,"%s")
`, dataType, textpy, inPyScript, outPutPath)

	RunPyScript(pythonScript)

	outTensor, err := tensor.LoadFromCSV(outPutPath)
	if err != nil {
		panic(err)
	}
	return outTensor
}
