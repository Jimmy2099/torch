package mnist

import (
	"testing"
)

func TestLoadMNIST(t *testing.T) {
	imageFile := "./dataset/MNIST/raw/train-images-idx3-ubyte"
	labelFile := "./dataset/MNIST/raw/train-labels-idx1-ubyte"

	data, err := LoadMNIST(imageFile, labelFile)
	if err != nil {
		t.Fatalf("Failed to load MNIST data: %v", err)
	}

	if data.Images == nil {
		t.Error("Images tensor is nil")
	}

	if data.Labels == nil {
		t.Error("Labels tensor is nil")
	}

	if data.Images.GetShape()[0] != data.Labels.GetShape()[0] {
		t.Errorf("Mismatched number of images (%v,%d) and labels (%v,%d)", data.Images.GetShape(), data.Images.Size(), data.Labels.GetShape(), data.Labels.Size())
	}

	expectedImageShape := []int{28, 28}
	if data.Images.GetShape()[1] != expectedImageShape[0]*expectedImageShape[1] {
		t.Errorf("Unexpected image shape: got %v, expected %v", data.Images.GetShape(), expectedImageShape)
	}
}
