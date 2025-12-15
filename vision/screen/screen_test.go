package screen

import (
	"testing"
	"time"
)

func TestScreen(t *testing.T) {
	s := NewScreen()
	defer s.Close()

	for {
		{
			drawBlock(s, 100, 100, 30, 255, 0, 0) // R
			drawBlock(s, 140, 100, 30, 0, 255, 0) // G
			drawBlock(s, 180, 100, 30, 0, 0, 255) // B
		}
		time.Sleep(time.Millisecond)
	}

}
