package tensor

import (
	"encoding/gob"
	"os"
)

type DeviceType int

const (
	CPU DeviceType = iota
	GPU
)

type Device struct {
	Type  DeviceType
	Index int
}

type Tensor struct {
	Data   []float32
	shape  []int
	Device Device
}

func NewTensor(data []float32, shape []int) *Tensor {
	t := &Tensor{
		Data:  data,
		shape: shape,
		Device: Device{
			Type:  CPU,
			Index: 0,
		},
	}
	if shape == nil {
		shape = []int{1, len(data)}
		t.Reshape(shape)
	}
	return t
}

func (m *Tensor) TensorData() []float32 {
	return m.Data
}

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

func (t *Tensor) GetShape() []int {
	return t.shape
}
func (t *Tensor) GetShapeByNum(num int) int {
	return t.shape[num]
}

func NewEmptyTensor() *Tensor {
	return &Tensor{
		Data:  make([]float32, 0),
		shape: make([]int, 0),
	}
}
