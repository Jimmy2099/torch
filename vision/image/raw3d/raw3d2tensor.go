package raw3d

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_store/tensor"
)

func Raw3DToTensor(input *tensor.Tensor, c int, r int) *tensor.Tensor {
	if input == nil {
		panic("input tensor is nil")
	}

	shape := input.Shape()
	if len(shape) != 4 {
		panic(fmt.Errorf("expected 4D tensor [D, H, W, C], got %v", shape))
	}

	depth, h, w, ch := shape[0], shape[1], shape[2], shape[3]

	if c*r != depth {
		panic(fmt.Errorf("grid size (r=%d, c=%d, total=%d) does not match input depth %d", r, c, c*r, depth))
	}

	outH := h * r
	outW := w * c
	newData := make([]float32, outH*outW*ch)

	strideRowIn := w * ch
	strideSliceIn := h * strideRowIn

	strideRowOut := outW * ch

	for row := 0; row < r; row++ {
		for col := 0; col < c; col++ {
			z := row*c + col

			srcBase := z * strideSliceIn

			dstBase := (row * h * strideRowOut) + (col * w * ch)

			for y := 0; y < h; y++ {
				srcStart := srcBase + (y * strideRowIn)
				srcEnd := srcStart + strideRowIn

				dstStart := dstBase + (y * strideRowOut)

				copy(newData[dstStart:], input.Data[srcStart:srcEnd])
			}
		}
	}

	return tensor.NewTensor(newData, []int{outH, outW, ch})
}
