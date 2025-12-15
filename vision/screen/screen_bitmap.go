//go:build windows

package screen

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"unsafe"
)

var (
	procSetDIBitsToDevice = gdi32.NewProc("SetDIBitsToDevice")
)

type bitmapInfo struct {
	BiSize          uint32
	BiWidth         int32
	BiHeight        int32
	BiPlanes        uint16
	BiBitCount      uint16
	BiCompression   uint32
	BiSizeImage     uint32
	BiXPelsPerMeter int32
	BiYPelsPerMeter int32
	BiClrUsed       uint32
	BiClrImportant  uint32
	Colors          [1]uint32
}

func (s *Screen) DrawTensorBitmap(t *tensor.Tensor, x, y int) {
	if len(t.GetShape()) < 2 {
		return
	}
	h := int32(t.GetShape()[0])
	w := int32(t.GetShape()[1])
	channels := 1
	if len(t.GetShape()) > 2 {
		channels = t.GetShape()[2]
	}

	pixelData := make([]uint8, w*h*4)

	tensorIdx := 0
	bufIdx := 0
	totalPixels := int(w * h)

	for i := 0; i < totalPixels; i++ {
		var r, g, b uint8

		if channels == 1 {
			val := uint8(t.Data[tensorIdx])
			r, g, b = val, val, val
			tensorIdx++
		} else {
			r = uint8(t.Data[tensorIdx])
			g = uint8(t.Data[tensorIdx+1])
			b = uint8(t.Data[tensorIdx+2])
			tensorIdx += 3
		}

		pixelData[bufIdx] = b
		pixelData[bufIdx+1] = g
		pixelData[bufIdx+2] = r
		pixelData[bufIdx+3] = 255
		bufIdx += 4
	}

	bmi := bitmapInfo{}
	{
		bmi.BiSize = uint32(unsafe.Sizeof(bmi) - 4)
		bmi.BiWidth = w
		bmi.BiHeight = -h
		bmi.BiPlanes = 1
		bmi.BiBitCount = 32
		bmi.BiCompression = 0
	}

	procSetDIBitsToDevice.Call(
		uintptr(s.hdc),
		uintptr(x),
		uintptr(y),
		uintptr(w),
		uintptr(h),
		0,
		0,
		0,
		uintptr(h),
		uintptr(unsafe.Pointer(&pixelData[0])),
		uintptr(unsafe.Pointer(&bmi)),
		0,
	)
}
