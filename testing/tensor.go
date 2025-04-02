package testing

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"io"
	"os"
	"os/exec"
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

	runPyscript(pythonScript)

	// 读取计算结果
	outTensor, err := tensor.LoadFromCSV(outPutPath)
	if err != nil {
		panic(err)
	}
	return outTensor
}

func runPyscript(pythonScript string) {
	// 打印嵌入的 Python 脚本内容
	fmt.Println("Embedded Python script:\n", pythonScript)

	// 将嵌入的 Python 脚本写入临时文件
	file, err := os.CreateTemp("", "script.py")
	if err != nil {
		panic(err)
	}
	defer os.Remove(file.Name()) // 程序结束时删除临时文件

	// 将嵌入的 Python 脚本内容写入临时文件
	_, err = file.WriteString(pythonScript)
	if err != nil {
		panic(err)
	}

	// 执行 Python 脚本
	cmd := exec.Command("python", file.Name())

	// 实时打印输出
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(fmt.Sprintln("Error creating Stdout pipe:", err))
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		panic(fmt.Sprintln("Error creating Stderr pipe:", err))
	}

	// 启动命令
	if err := cmd.Start(); err != nil {
		panic(fmt.Sprintln("Error starting Python script:", err))
	}

	// 使用 Goroutines 来实时打印标准输出和错误输出
	go func() {
		_, err := io.Copy(os.Stdout, stdout)
		if err != nil {
			panic(fmt.Sprintln("Error copying stdout:", err))
		}
	}()

	go func() {
		_, err := io.Copy(os.Stderr, stderr)
		if err != nil {
			panic(fmt.Sprintln("Error copying stderr:", err))
		}
	}()

	// 等待命令执行完成
	if err := cmd.Wait(); err != nil {
		panic(fmt.Sprintln("Error waiting for Python script to finish:", err))
	}
}
