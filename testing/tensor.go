package testing

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"os"
	"path/filepath"
	"strings"
)

func GetTensorTestResult(inPyScript string, inTensor1 *tensor.Tensor, inTensor2 *tensor.Tensor) *tensor.Tensor {
	var inFile1 *os.File
	var err error
	{
		inFile1, err = os.CreateTemp("", "input_tensor1.*.csv")
		if err != nil {
			panic(err)
		}
		inFile1.Close()
		defer os.Remove(inFile1.Name())
		err = inTensor1.SaveToCSV(inFile1.Name())
		if err != nil {
			panic(err)
		}
	}

	var inFile2 *os.File
	{
		inFile2, err = os.CreateTemp("", "input_tensor2.*.csv")
		if err != nil {
			panic(err)
		}
		inFile2.Close()
		defer os.Remove(inFile2.Name())
		err = inTensor2.SaveToCSV(inFile2.Name())
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
		defer os.Remove(outFile.Name())
	}

	inputPath1 := filepath.ToSlash(inFile1.Name())
	inputPath2 := filepath.ToSlash(inFile2.Name())
	outPutPath := filepath.ToSlash(outFile.Name())
	inPyScript = strings.TrimSpace(inPyScript)
	// Python 脚本
	pythonScript := fmt.Sprintf(`
import numpy as np
import torch

def save_tensor_to_csv(tensor, file_path):
    with open(file_path, 'w') as f:
        f.write("Shape," + ",".join(map(str, tensor.shape)) + "\n")
        np.savetxt(f, tensor.numpy().reshape(-1, tensor.shape[-1]), delimiter=",")

def load_tensor_from_csv(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        if not header.startswith("Shape,"):
            raise ValueError("Invalid CSV format: missing shape header")
        
        shape = list(map(int, header.split(",")[1:]))
        data = np.loadtxt(f, delimiter=",")
    
    flattened = data.flatten()
    return torch.tensor(flattened).reshape(*shape)

in1 = load_tensor_from_csv("%s")
in2 = load_tensor_from_csv("%s")

def process_data(in1,in2):
    out = None
    %s
    return out

out = process_data(in1,in2)

save_tensor_to_csv(out,"%s")
`, inputPath1, inputPath2, inPyScript, outPutPath)

	RunPyScript(pythonScript)

	// 读取计算结果
	outTensor, err := tensor.LoadFromCSV(outPutPath)
	if err != nil {
		panic(err)
	}
	return outTensor
}
