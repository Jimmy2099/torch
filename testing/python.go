package testing

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/pkg/log"
	"io"
	"os"
	"os/exec"
)

func RunPyScript(pythonScript string) {
	file, err := os.CreateTemp("", "script.*.py")
	log.Println("pyScript Path: ", file.Name())
	if err != nil {
		panic(err)
	}

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
		panic(fmt.Sprintln(file.Name(), "Error waiting for Python script to finish:", err))
	}
}
