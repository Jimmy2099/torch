package mnist

import (
	_ "embed"
	"fmt"
	"io"
	"os"
	"os/exec"
)

//go:embed load_dataset.py
var pythonScript string

func LoadDataset() {
	// 打印嵌入的 Python 脚本内容
	fmt.Println("Embedded Python script:\n", pythonScript)

	// 将嵌入的 Python 脚本写入临时文件
	file, err := os.CreateTemp("", "script.py")
	if err != nil {
		fmt.Println("Error creating temporary file:", err)
		return
	}
	defer os.Remove(file.Name()) // 程序结束时删除临时文件

	// 将嵌入的 Python 脚本内容写入临时文件
	_, err = file.WriteString(pythonScript)
	if err != nil {
		fmt.Println("Error writing to temporary file:", err)
		return
	}

	// 执行 Python 脚本
	cmd := exec.Command("python", file.Name())

	// 实时打印输出
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		fmt.Println("Error creating Stdout pipe:", err)
		return
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		fmt.Println("Error creating Stderr pipe:", err)
		return
	}

	// 启动命令
	if err := cmd.Start(); err != nil {
		fmt.Println("Error starting Python script:", err)
		return
	}

	// 使用 Goroutines 来实时打印标准输出和错误输出
	go func() {
		_, err := io.Copy(os.Stdout, stdout)
		if err != nil {
			fmt.Println("Error copying stdout:", err)
		}
	}()

	go func() {
		_, err := io.Copy(os.Stderr, stderr)
		if err != nil {
			fmt.Println("Error copying stderr:", err)
		}
	}()

	// 等待命令执行完成
	if err := cmd.Wait(); err != nil {
		fmt.Println("Error waiting for Python script to finish:", err)
	}
}
