package mnist

import (
	_ "embed"
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/Jimmy2099/torch/testing"
	"os"
)

type MNISTData struct {
	Images *tensor.Tensor
	Labels *tensor.Tensor
}

//go:embed load_dataset.py
var pythonScript string

func LoadMNIST(imageFile, labelFile string) (*MNISTData, error) {
	{
		_, err := os.Stat(imageFile)
		_, err1 := os.Stat(labelFile)
		if err != nil || err1 != nil {
			testing.RunPyScript(pythonScript)
		}
	}

	images, err := loadImages(imageFile)
	if err != nil {
		return nil, err
	}
	if images == nil {
		return nil, fmt.Errorf("images data is nil")
	}

	labels, err := loadLabels(labelFile)
	if err != nil {
		return nil, err
	}
	if labels == nil {
		return nil, fmt.Errorf("labels data is nil")
	}

	imageMatrix := tensor.NewTensorFromSlice(images)
	labelMatrix := tensor.NewTensorFromSlice(labels)

	return &MNISTData{Images: imageMatrix, Labels: labelMatrix}, nil
}
