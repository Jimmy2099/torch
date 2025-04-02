package testing

import (
	"fmt"
	"io"
	"os"
	"os/exec"
)

func RunPyScript(pythonScript string) {
	// 打印嵌入的 Python 脚本内容
	//fmt.Println("Embedded Python script:\n", pythonScript)

	// 将嵌入的 Python 脚本写入临时文件
	file, err := os.CreateTemp("", "script.*.py")
	if err != nil {
		panic(err)
	}
	//defer os.Remove(file.Name()) // 程序结束时删除临时文件

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
		panic(fmt.Sprintln(file.Name(), "Error waiting for Python script to finish:", err))
	}
}
