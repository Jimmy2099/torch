package float16

import "fmt"

var _ fmt.Formatter = Float16(0)

// Format implements [fmt.Formatter].
func (x Float16) Format(s fmt.State, verb rune) {
	if x.IsNaN() {
		s.Write([]byte("NaN"))
		return
	}

	var prefix []byte
	var data []byte

	// sign
	if x&signMask16 != 0 {
		prefix = append(prefix, '-')
		x &^= signMask16
	} else {
		if s.Flag('+') {
			prefix = append(prefix, '+')
		} else if s.Flag(' ') {
			prefix = append(prefix, ' ')
		}
	}

	switch verb {
	case 'b':
		data = x.appendBin(data)
	case 'f':
		if prec, ok := s.Precision(); ok {
			data = x.Append(data, byte(verb), prec)
		} else {
			data = x.Append(data, byte(verb), -1)
		}
	case 'e', 'E':
		if prec, ok := s.Precision(); ok {
			data = x.Append(data, byte(verb), prec)
		} else {
			data = x.Append(data, byte(verb), -1)
		}
	case 'g', 'G':
		if prec, ok := s.Precision(); ok {
			data = x.Append(data, byte(verb), prec)
		} else {
			data = x.Append(data, byte(verb), -1)
		}
	case 'x', 'X':
		if prec, ok := s.Precision(); ok {
			data = x.Append(data, byte(verb), prec)
		} else {
			data = x.Append(data, byte(verb), -1)
		}
	case 'v':
		data = x.Append(data, 'g', -1)
	}

	if w, ok := s.Width(); ok {
		var buf [1]byte
		if s.Flag('-') {
			s.Write(prefix)
			s.Write(data)
			buf[0] = ' '
			for i := len(data); i < w; i++ {
				s.Write(buf[:1])
			}
		} else {
			buf[0] = ' '
			for i := len(data); i < w; i++ {
				s.Write(buf[:1])
			}
			s.Write(prefix)
			s.Write(data)
		}
		return
	}

	if len(prefix) > 0 {
		s.Write(prefix)
	}
	s.Write(data)
}
