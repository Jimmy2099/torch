package matrix

import "math"

// Conv2D 实现二维卷积操作
func (m *Matrix) Conv2D(weights *Matrix, kernelSize, stride, pad int) *Matrix {
	// 输入维度：(1, height, width) 对于MNIST单通道图像
	// 权重维度：(out_channels, kernelSize*kernelSize)
	// 输出维度：(out_channels, out_height, out_width)

	// 计算输出空间尺寸
	outHeight := (m.Rows+2*pad-kernelSize)/stride + 1
	outWidth := (m.Cols+2*pad-kernelSize)/stride + 1

	// 执行im2col展开
	unfolded := m.im2col(kernelSize, stride, pad)

	// 矩阵乘法：weights * unfolded
	result := weights.Multiply(unfolded)

	// 重新排列为输出形状
	return result.Reshape(weights.Rows, outHeight*outWidth)
}

// im2col_get_pixel 实现边界检查的像素获取
func im2col_get_pixel(im []float64, height, width, channels int,
	row, col, channel, pad int) float64 {
	row -= pad
	col -= pad

	if row < 0 || col < 0 || row >= height || col >= width {
		return 0
	}
	return im[col+width*(row+height*channel)]
}

// im2col 基于Caffe实现的高效版本，使用一维数组和前置递增
func (m *Matrix) im2col(kernelSize, stride, pad int) *Matrix {
	channels := m.Rows
	height := int(math.Sqrt(float64(m.Cols))) // 假设是方阵
	width := height

	height_col := (height+2*pad-kernelSize)/stride + 1
	width_col := (width+2*pad-kernelSize)/stride + 1
	channels_col := channels * kernelSize * kernelSize

	// 将输入数据展平为一维数组
	im := make([]float64, 0, channels*height*width)
	for c := 0; c < channels; c++ {
		im = append(im, m.Data[c]...)
	}

	// 创建一维输出数组
	cols := NewMatrix(channels_col, height_col*width_col)
	data_col := make([]float64, channels_col*height_col*width_col)

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

	// 将一维数组转换回Matrix格式
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
	size := int(math.Sqrt(float64(m.Cols))) // 原始尺寸
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

// Repeat 将矩阵沿行和列方向重复
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

// Conv2DGradWeights 计算权重梯度
func (m Matrix) Conv2DGradWeights(gradOutput *Matrix, kernelSize, stride, pad int) *Matrix {
	// 输入梯度维度：(out_channels, out_hout_w)
	// 输出梯度维度：(out_channels, in_channelskernelSizekernelSize)

	// 执行im2col展开输入
	unfolded := m.im2col(kernelSize, stride, pad)

	// 矩阵乘法：gradOutput * unfolded^T
	return gradOutput.Multiply(unfolded.Transpose())
}

// Conv2DGradInput 计算输入梯度
func (m *Matrix) Conv2DGradInput(weights *Matrix, kernelSize, stride, pad int) *Matrix {
	// 输入梯度维度：(out_channels, out_hout_w)
	// 输出梯度维度：(in_channels, in_hin_w)

	// 转置权重矩阵
	wT := weights.Transpose()

	// 矩阵乘法：wT * gradOutput
	result := wT.Multiply(m)

	// 执行col2im操作
	return result.col2im(kernelSize, stride, pad, m.Rows, m.Cols)
}

// col2im 将展开的列重新排列为图像格式
func (m *Matrix) col2im(kernelSize, stride, pad, inHeight, inWidth int) *Matrix {
	// 计算原始尺寸（包含padding）
	origHeight := inHeight + 2
	origWidth := inWidth + 2

	// 初始化输出矩阵
	output := NewMatrix(origHeight, origWidth)

	// 遍历所有列
	for i := 0; i < m.Cols; i++ {
		// 计算原始位置
		h := (i / origWidth) * stride
		w := (i % origWidth) * stride

		// 获取当前patch并reshape
		patch := m.GetCol(i).Reshape(kernelSize, kernelSize)

		// 累加到对应位置
		for dh := 0; dh < kernelSize; dh++ {
			for dw := 0; dw < kernelSize; dw++ {
				output.Data[h+dh][w+dw] += patch.Data[dh][dw]
			}
		}
	}

	// 去除padding
	return output.GetRows(pad, origHeight-pad).GetCols(pad, origWidth-pad)
}

// Pad2D 实现二维padding
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

// Flatten 展平矩阵为列向量
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
		endDim = m.Dimensions() - 1 // -1 代表最后一个维度
	}

	// 计算展平后矩阵的维度
	rows := 1
	cols := 1

	// 计算从 startDim 到 endDim 展平的维度
	for i := startDim; i <= endDim; i++ {
		rows *= m.DimSize(i)
	}

	// 计算剩余的维度
	for i := endDim + 1; i < m.Dimensions(); i++ {
		cols *= m.DimSize(i)
	}

	// 调用 Reshape 进行展平
	return m.Reshape(rows, cols)
}

// Dimensions 获取矩阵的维度数量
func (m *Matrix) Dimensions() int {
	// 对于二维矩阵，只有行和列两维
	return 2
}

// DimSize 获取指定维度的大小，dim = 0 时返回行数，dim = 1 时返回列数
func (m *Matrix) DimSize(dim int) int {
	if dim == 0 {
		return m.Rows
	} else if dim == 1 {
		return m.Cols
	}
	panic("invalid dimension")
}

// Clone 深拷贝矩阵
func (m *Matrix) Clone() *Matrix {
	return Copy(m)
}

// GetCols 获取指定列范围的子矩阵
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

// SetCol 设置指定列的数据
func (m *Matrix) SetCol(colIdx int, data *Matrix) {
	if data.Rows != m.Rows || data.Cols != 1 {
		panic("invalid column data dimensions")
	}

	for i := 0; i < m.Rows; i++ {
		m.Data[i][colIdx] = data.Data[i][0]
	}
}

// GetCol 获取指定列的数据
func (m *Matrix) GetCol(colIdx int) *Matrix {
	result := NewMatrix(m.Rows, 1)
	for i := 0; i < m.Rows; i++ {
		result.Data[i][0] = m.Data[i][colIdx]
	}
	return result
}

// Sum 沿指定维度求和
func (m *Matrix) SumByDim(dim int) *Matrix {
	if dim == 0 { // 沿列求和，返回行向量
		result := NewMatrix(1, m.Cols)
		for j := 0; j < m.Cols; j++ {
			sum := 0.0
			for i := 0; i < m.Rows; i++ {
				sum += m.Data[i][j]
			}
			result.Data[0][j] = sum
		}
		return result
	} else if dim == 1 { // 沿行求和，返回列向量
		result := NewMatrix(m.Rows, 1)
		for i := 0; i < m.Rows; i++ {
			sum := 0.0
			for j := 0; j < m.Cols; j++ {
				sum += m.Data[i][j]
			}
			result.Data[i][0] = sum
		}
		return result
	}
	panic("invalid dimension for sum")
}

// Pad 在矩阵四周添加 padding 数量的零填充
func (m *Matrix) Pad(padding int) *Matrix {
	newRows := m.Rows + 2*padding
	newCols := m.Cols + 2*padding
	// 初始化全零矩阵
	paddedData := make([][]float64, newRows)
	for i := range paddedData {
		paddedData[i] = make([]float64, newCols)
	}
	// 将原矩阵拷贝到中间位置
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

// Crop 裁剪掉矩阵四周 padding 数量的边界
func (m *Matrix) Crop(padding int) *Matrix {
	if padding == 0 {
		return m
	}
	newRows := m.Rows - 2*padding
	newCols := m.Cols - 2*padding
	croppedData := make([][]float64, newRows)
	for i := 0; i < newRows; i++ {
		croppedData[i] = make([]float64, newCols)
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
