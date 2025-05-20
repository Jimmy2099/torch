package tensor

import (
	"encoding/gob"
	"os"
)

type Tensor struct {
	Data   []float32
	shape  []int
	Device Device
	//
	Grad []float32
}

func NewTensor(data []float32, shape []int) *Tensor {
	if shape == nil {
		shape = []int{1, len(data)}
	}

	sum := ShapeSum(shape)

	if len(data) != sum {
		panic("shape length mismatch")
	}

	t := &Tensor{
		Data:  data,
		shape: shape,
	}
	t.Device = GetDefaultDevice()
	return t
}

func ShapeSum(shape []int) (result int) {
	if len(shape) == 0 {
		return 0
	}
	result = 1
	for i := 0; i < len(shape); i++ {
		if shape[i] == 1 {
			continue
		}
		result *= shape[i]
	}
	return result
}

func (t *Tensor) TensorData() []float32 {
	return t.Data
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

func (t *Tensor) ToDevice(device Device) {
	if device == nil {
		panic("device cannot be nil")
	}
	//TODO
}
