package tensor

import (
	"encoding/gob"
	"os"
)

type Tensor struct {
	Data  []float32
	Shape []int // e.g., [batch_size, channels, height, width]
}

func NewTensor(data []float32, shape []int) *Tensor {
	t := &Tensor{Data: data, Shape: shape}
	if shape == nil {
		shape = []int{1, len(data)}
		t.Reshape(shape)
	}
	return t
}

func (m *Tensor) TensorData() []float32 {
	return m.Data
}

// Save 将Tensor结构体使用gob编码保存到指定文件
func (t *Tensor) Save(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(t); err != nil {
		return err
	}
	return nil
}

func LoadTensorFromGobFile(filename string) (*Tensor, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	t := &Tensor{}
	if err = decoder.Decode(t); err != nil {
		return nil, err
	}
	return t, nil
}
