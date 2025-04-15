package matrix

import math "github.com/chewxy/math32"

func (m *Matrix) Conv2D(weights *Matrix, kernelSize, stride, pad int) *Matrix {

	outHeight := (m.Rows+2*pad-kernelSize)/stride + 1
	outWidth := (m.Cols+2*pad-kernelSize)/stride + 1

	unfolded := m.im2col(kernelSize, stride, pad)

	result := weights.Multiply(unfolded)

	return result.Reshape(weights.Rows, outHeight*outWidth)
}

func im2col_get_pixel(im []float32, height, width, channels int,
	row, col, channel, pad int) float32 {
	row -= pad
	col -= pad

	if row < 0 || col < 0 || row >= height || col >= width {
		return 0
	}
	return im[col+width*(row+height*channel)]
}

func (m *Matrix) im2col(kernelSize, stride, pad int) *Matrix {
	channels := m.Rows
	height := int(math.Sqrt(float32(m.Cols)))
	width := height

	height_col := (height+2*pad-kernelSize)/stride + 1
	width_col := (width+2*pad-kernelSize)/stride + 1
	channels_col := channels * kernelSize * kernelSize

	im := make([]float32, 0, channels*height*width)
	for c := 0; c < channels; c++ {
		im = append(im, m.Data[c]...)
	}

	cols := NewMatrix(channels_col, height_col*width_col)
	data_col := make([]float32, channels_col*height_col*width_col)

	for c := 0; c < channels_col; c++ {
		w_offset := c % kernelSize
		h_offset := (c / kernelSize) % kernelSize
		c_im := c / kernelSize / kernelSize

		h := 0
		for h < height_col {
			w := 0
			for w < width_col {
				im_row := h_offset + h*stride
				im_col := w_offset + w*stride
				col_index := (c*height_col+h)*width_col + w
				data_col[col_index] = im2col_get_pixel(im, height, width, channels,
					im_row, im_col, c_im, pad)
				w++
			}
			h++
		}
	}

	for c := 0; c < channels_col; c++ {
		start := c * height_col * width_col
		end := start + height_col*width_col
		cols.Data[c] = data_col[start:end]
	}

	return cols
}

func (m *Matrix) Pad2D(pad int) *Matrix {
	if pad == 0 {
		return m.Clone()
	}

	channels := m.Rows
	size := int(math.Sqrt(float32(m.Cols)))
	newSize := size + 2*pad

	padded := NewMatrix(channels, newSize*newSize)

	for c := 0; c < channels; c++ {
		for i := 0; i < size; i++ {
			for j := 0; j < size; j++ {
				padded.Data[c][(i+pad)*newSize+(j+pad)] = m.Data[c][i*size+j]
			}
		}
	}

	return padded
}

func (m *Matrix) Repeat(rowRepeat, colRepeat int) *Matrix {
	newRows := m.Rows * rowRepeat
	newCols := m.Cols * colRepeat
	result := NewMatrix(newRows, newCols)

	for i := 0; i < newRows; i++ {
		for j := 0; j < newCols; j++ {
			result.Data[i][j] = m.Data[i%m.Rows][j%m.Cols]
		}
	}
	return result
}

func (m Matrix) Conv2DGradWeights(gradOutput *Matrix, kernelSize, stride, pad int) *Matrix {

	unfolded := m.im2col(kernelSize, stride, pad)

	return gradOutput.Multiply(unfolded.Transpose())
}

func (m *Matrix) Conv2DGradInput(weights *Matrix, kernelSize, stride, pad int) *Matrix {

	wT := weights.Transpose()

	result := wT.Multiply(m)

	return result.col2im(kernelSize, stride, pad, m.Rows, m.Cols)
}

func (m *Matrix) col2im(kernelSize, stride, pad, inHeight, inWidth int) *Matrix {
	origHeight := inHeight + 2
	origWidth := inWidth + 2

	output := NewMatrix(origHeight, origWidth)

	for i := 0; i < m.Cols; i++ {
		h := (i / origWidth) * stride
		w := (i % origWidth) * stride

		patch := m.GetCol(i).Reshape(kernelSize, kernelSize)

		for dh := 0; dh < kernelSize; dh++ {
			for dw := 0; dw < kernelSize; dw++ {
				output.Data[h+dh][w+dw] += patch.Data[dh][dw]
			}
		}
	}

	return output.GetRows(pad, origHeight-pad).GetCols(pad, origWidth-pad)
}

func (m *Matrix) Pad2D1(pad int) *Matrix {
	if pad == 0 {
		return m.Clone()
	}

	newRows := m.Rows + 2*pad
	newCols := m.Cols + 2*pad
	padded := NewMatrix(newRows, newCols)

	for i := 0; i < m.Rows; i++ {
		copy(padded.Data[i+pad][pad:], m.Data[i])
	}
	return padded
}

func (m *Matrix) Flatten() *Matrix {
	return m.Reshape(m.Cols*m.Rows, 1)
}

func (m *Matrix) FlattenByDim(startDim, endDim int) *Matrix {
	if startDim < 0 || startDim >= m.Dimensions() {
		panic("Invalid startDim")
	}
	if endDim < -1 || endDim >= m.Dimensions() {
		panic("Invalid endDim")
	}

	if endDim == -1 {
		endDim = m.Dimensions() - 1
	}

	rows := 1
	cols := 1

	for i := startDim; i <= endDim; i++ {
		rows *= m.DimSize(i)
	}

	for i := endDim + 1; i < m.Dimensions(); i++ {
		cols *= m.DimSize(i)
	}

	return m.Reshape(rows, cols)
}

func (m *Matrix) Dimensions() int {
	return 2
}

func (m *Matrix) DimSize(dim int) int {
	if dim == 0 {
		return m.Rows
	} else if dim == 1 {
		return m.Cols
	}
	panic("invalid dimension")
}

func (m *Matrix) Clone() *Matrix {
	return Copy(m)
}

func (m *Matrix) GetCols(start, end int) *Matrix {
	if start < 0 || end > m.Cols || start >= end {
		panic("invalid column range")
	}

	result := NewMatrix(m.Rows, end-start)
	for i := 0; i < m.Rows; i++ {
		for j := start; j < end; j++ {
			result.Data[i][j-start] = m.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) SetCol(colIdx int, data *Matrix) {
	if data.Rows != m.Rows || data.Cols != 1 {
		panic("invalid column data dimensions")
	}

	for i := 0; i < m.Rows; i++ {
		m.Data[i][colIdx] = data.Data[i][0]
	}
}

func (m *Matrix) GetCol(colIdx int) *Matrix {
	result := NewMatrix(m.Rows, 1)
	for i := 0; i < m.Rows; i++ {
		result.Data[i][0] = m.Data[i][colIdx]
	}
	return result
}

func (m *Matrix) SumByDim(dim int) *Matrix {
	if dim == 0 {
		result := NewMatrix(1, m.Cols)
		for j := 0; j < m.Cols; j++ {
			var sum float32
			for i := 0; i < m.Rows; i++ {
				sum += m.Data[i][j]
			}
			result.Data[0][j] = sum
		}
		return result
	} else if dim == 1 {
		result := NewMatrix(m.Rows, 1)
		for i := 0; i < m.Rows; i++ {
			var sum float32
			for j := 0; j < m.Cols; j++ {
				sum += m.Data[i][j]
			}
			result.Data[i][0] = sum
		}
		return result
	}
	panic("invalid dimension for sum")
}

func (m *Matrix) Pad(padding int) *Matrix {
	newRows := m.Rows + 2*padding
	newCols := m.Cols + 2*padding
	paddedData := make([][]float32, newRows)
	for i := range paddedData {
		paddedData[i] = make([]float32, newCols)
	}
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			paddedData[i+padding][j+padding] = m.Data[i][j]
		}
	}
	return &Matrix{
		Rows: newRows,
		Cols: newCols,
		Data: paddedData,
	}
}

func (m *Matrix) Crop(padding int) *Matrix {
	if padding == 0 {
		return m
	}
	newRows := m.Rows - 2*padding
	newCols := m.Cols - 2*padding
	croppedData := make([][]float32, newRows)
	for i := 0; i < newRows; i++ {
		croppedData[i] = make([]float32, newCols)
		for j := 0; j < newCols; j++ {
			croppedData[i][j] = m.Data[i+padding][j+padding]
		}
	}
	return &Matrix{
		Rows: newRows,
		Cols: newCols,
		Data: croppedData,
	}
}
