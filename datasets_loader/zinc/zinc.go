package zinc

import (
	_ "embed"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/testing"
)

//go:embed load_dataset.py
var pythonScript string

func LoadZINC(subset bool) {
	subSetString := "False"
	if subset == true {
		subSetString = "True"
	}
	testing.RunPyScript(fmt.Sprintf(pythonScript, subSetString, subSetString, subSetString))
}
