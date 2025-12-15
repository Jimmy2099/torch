package screen

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"math/rand"
	"testing"
	"time"
)

func TestScreenTensor(t *testing.T) {
	s := NewScreen()
	defer s.Close()

	// NewTensor [Height=100, Width=100]
	height, width := 100, 100
	grayData := make([]float32, height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			val := float32(x) / float32(width) * 255
			grayData[y*width+x] = val
		}
	}
	grayTensor := tensor.NewTensor(grayData, []int{height, width})

	// Random color Tensor [Height=50, Width=50, Channels=3]
	h2, w2 := 50, 50
	colorData := make([]float32, h2*w2*3)
	for i := 0; i < len(colorData); i++ {
		colorData[i] = float32(rand.Intn(255))
	}
	colorTensor := tensor.NewTensor(colorData, []int{h2, w2, 3})

	for {

		s.DrawTensor(grayTensor, 200, 100)

		s.DrawTensor(colorTensor, 350, 100)

		time.Sleep(time.Millisecond)
	}
}
