package mnist

import (
	_ "embed"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"io"
	"os"
	"os/exec"
)

var pythonScript string

func loadDataset() {
	fmt.Println("Embedded Python script:\n", pythonScript)

	file, err := os.CreateTemp("", "script.py")
	if err != nil {
		panic(err)
	}
	defer os.Remove(file.Name()) // 程序结束时删除临时文件

	_, err = file.WriteString(pythonScript)
	if err != nil {
		panic(err)
	}

	cmd := exec.Command("python", file.Name())

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		panic(fmt.Sprintln("Error creating Stdout pipe:", err))
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		panic(fmt.Sprintln("Error creating Stderr pipe:", err))
	}

	if err := cmd.Start(); err != nil {
		panic(fmt.Sprintln("Error starting Python script:", err))
	}

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

	if err := cmd.Wait(); err != nil {
		panic(fmt.Sprintln("Error waiting for Python script to finish:", err))
	}
}
