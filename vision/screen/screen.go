//go:build windows

package screen

import (
	"syscall"
)

var (
	user32 = syscall.NewLazyDLL("user32.dll")
	gdi32  = syscall.NewLazyDLL("gdi32.dll")

	procGetDC     = user32.NewProc("GetDC")
	procReleaseDC = user32.NewProc("ReleaseDC")

	procSetPixel = gdi32.NewProc("SetPixel")
	procGetPixel = gdi32.NewProc("GetPixel")
)

type Screen struct {
	hdc syscall.Handle
}

func NewScreen() *Screen {
	hdc, _, _ := procGetDC.Call(0)
	return &Screen{
		hdc: syscall.Handle(hdc),
	}
}

func (s *Screen) Close() {
	if s.hdc != 0 {
		procReleaseDC.Call(0, uintptr(s.hdc))
		s.hdc = 0
	}
}

func (s *Screen) SetPixel(x, y int, r, g, b uint8) {
	colorref := uint32(r) |
		uint32(g)<<8 |
		uint32(b)<<16

	procSetPixel.Call(
		uintptr(s.hdc),
		uintptr(x),
		uintptr(y),
		uintptr(colorref),
	)
}

func (s *Screen) GetPixel(x, y int) (r, g, b uint8) {
	ret, _, _ := procGetPixel.Call(
		uintptr(s.hdc),
		uintptr(x),
		uintptr(y),
	)

	col := uint32(ret)
	r = uint8(col & 0xFF)
	g = uint8((col >> 8) & 0xFF)
	b = uint8((col >> 16) & 0xFF)
	return
}

func drawBlock(s *Screen, x, y, size int, r, g, b uint8) {
	//s.SetPixel(100, 100, 255, 0, 0)
	//s.SetPixel(101, 100, 0, 255, 0)
	//s.SetPixel(102, 100, 0, 0, 255)
	for dy := 0; dy < size; dy++ {
		for dx := 0; dx < size; dx++ {
			s.SetPixel(x+dx, y+dy, r, g, b)
		}
	}
}
