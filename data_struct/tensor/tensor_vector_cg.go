package tensor

import (
	"fmt"
	"math"
)

func NewVec3(x, y, z float64) *Tensor {
	return NewTensor([]float64{x, y, z}, []int{3})
}

//tensor vector computer graphic

func (m *Tensor) X() float64 {
	return m.Data[0]
}

func (m *Tensor) Y() float64 {
	return m.Data[1]
}

func (m *Tensor) Z() float64 {
	return m.Data[2]
}

//

func (t *Tensor) IsMatrix() bool {
	return len(t.Shape) == 2
}

func (t *Tensor) IsVector() bool {
	return len(t.Shape) == 1
}

func (t *Tensor) checkMatrix() {
	if !t.IsMatrix() {
		panic("Operation requires matrix")
	}
}

func (t *Tensor) checkSquareMatrix() {
	t.checkMatrix()
	if t.Shape[0] != t.Shape[1] {
		panic("Operation requires square matrix")
	}
}

func (t *Tensor) checkVector() {
	if !t.IsVector() {
		panic("Operation requires vector")
	}
}

// 矩阵行列式（仅限4x4）
func (t *Tensor) Determinant() float64 {
	t.checkSquareMatrix()
	if t.Shape[0] != 4 {
		panic("Determinant currently only supported for 4x4 matrices")
	}

	m := t.Data
	return m[0]*m[5]*m[10]*m[15] - m[0]*m[5]*m[11]*m[14] +
		m[0]*m[6]*m[11]*m[13] - m[0]*m[6]*m[9]*m[15] +
		m[0]*m[7]*m[9]*m[14] - m[0]*m[7]*m[10]*m[13] -
		m[1]*m[6]*m[11]*m[12] + m[1]*m[6]*m[8]*m[15] -
		m[1]*m[7]*m[8]*m[14] + m[1]*m[7]*m[10]*m[12] +
		m[1]*m[4]*m[10]*m[15] - m[1]*m[4]*m[11]*m[14] +
		m[2]*m[7]*m[8]*m[13] - m[2]*m[7]*m[9]*m[12] +
		m[2]*m[4]*m[9]*m[15] - m[2]*m[4]*m[11]*m[13] -
		m[2]*m[5]*m[8]*m[15] + m[2]*m[5]*m[11]*m[12] -
		m[3]*m[4]*m[9]*m[14] + m[3]*m[4]*m[10]*m[13] -
		m[3]*m[5]*m[10]*m[12] + m[3]*m[5]*m[8]*m[14] +
		m[3]*m[6]*m[8]*m[13] - m[3]*m[6]*m[9]*m[12]
}

// 逆矩阵（4x4专用实现）
func (t *Tensor) Inverse() *Tensor {
	t.checkSquareMatrix()
	if t.Shape[0] != 4 {
		panic("Inverse currently only supported for 4x4 matrices")
	}

	m := t.Data
	inv := make([]float64, 16)
	det := t.Determinant()

	inv[0] = (m[5]*m[10]*m[15] - m[5]*m[11]*m[14] + m[7]*m[9]*m[14] - m[6]*m[9]*m[15] + m[6]*m[11]*m[13] - m[7]*m[10]*m[13]) / det
	// 完整实现需要计算所有16个元素，此处为示例
	// 实际需要实现完整4x4逆矩阵计算

	return NewTensor(inv, []int{4, 4})
}

func Identity() *Tensor {
	return NewTensor([]float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1}, []int{4, 4})
}

func TranslateMatrix(v *Tensor) *Tensor {
	v.checkVector()
	if len(v.Data) != 3 {
		panic("Translation vector must be 3D")
	}

	return NewTensor([]float64{
		1, 0, 0, v.Data[0],
		0, 1, 0, v.Data[1],
		0, 0, 1, v.Data[2],
		0, 0, 0, 1}, []int{4, 4})
}

func Rotate(axis *Tensor, angle float64) *Tensor {
	axis = axis.Normalize()
	s := math.Sin(angle)
	c := math.Cos(angle)
	m := 1 - c

	x := axis.Data[0]
	y := axis.Data[1]
	z := axis.Data[2]

	return NewTensor([]float64{
		m*x*x + c, m*x*y + z*s, m*x*z - y*s, 0,
		m*x*y - z*s, m*y*y + c, m*y*z + x*s, 0,
		m*x*z + y*s, m*y*z - x*s, m*z*z + c, 0,
		0, 0, 0, 1}, []int{4, 4})
}

func Perspective(fovy, aspect, near, far float64) *Tensor {
	f := 1.0 / math.Tan(fovy/2)
	return NewTensor([]float64{
		f / aspect, 0, 0, 0,
		0, f, 0, 0,
		0, 0, (far + near) / (near - far), (2 * far * near) / (near - far),
		0, 0, -1, 0}, []int{4, 4})
}

func LookAt(eye, center, up *Tensor) *Tensor {
	z := eye.Sub(center).Normalize()
	x := up.Cross(z).Normalize()
	y := z.Cross(x)

	return NewTensor([]float64{
		x.Data[0], x.Data[1], x.Data[2], -x.Dot(eye),
		y.Data[0], y.Data[1], y.Data[2], -y.Dot(eye),
		z.Data[0], z.Data[1], z.Data[2], -z.Dot(eye),
		0, 0, 0, 1}, []int{4, 4})
}

func (t *Tensor) Dot(other *Tensor) float64 {
	if !t.IsVector() || !other.IsVector() {
		panic("Dot product requires vectors")
	}
	if len(t.Data) != len(other.Data) {
		panic("Vectors must have same length")
	}

	sum := 0.0
	for i := range t.Data {
		sum += t.Data[i] * other.Data[i]
	}
	return sum
}

func (t *Tensor) Cross(other *Tensor) *Tensor {
	if !t.IsVector() || !other.IsVector() || len(t.Data) != 3 || len(other.Data) != 3 {
		panic("Cross product requires 3D vectors")
	}

	a := t.Data
	b := other.Data
	return NewTensor([]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0]}, []int{3})
}

func (t *Tensor) Normalize() *Tensor {
	if !t.IsVector() {
		panic("Normalization requires vector")
	}

	length := 0.0
	for _, v := range t.Data {
		length += v * v
	}
	length = math.Sqrt(length)

	data := make([]float64, len(t.Data))
	for i := range data {
		data[i] = t.Data[i] / length
	}
	return NewTensor(data, t.Shape)
}

func (t *Tensor) Homogeneous() *Tensor {
	if !t.IsVector() || len(t.Data) != 3 {
		panic("Requires 3D vector")
	}
	return NewTensor(append(t.Data, 1.0), []int{4})
}

// 生成旋转矩阵
func RotateTensor(axis *Tensor, angle float64) *Tensor {
	// 验证输入参数
	if !axis.IsVector() || len(axis.Data) != 3 {
		panic("Rotate requires 3D vector axis")
	}

	// 归一化轴向量
	normAxis := axis.Normalize()
	x := normAxis.Data[0]
	y := normAxis.Data[1]
	z := normAxis.Data[2]

	// 计算三角函数值
	s := math.Sin(angle)
	c := math.Cos(angle)
	m := 1 - c

	// 构建4x4旋转矩阵
	data := []float64{
		m*x*x + c, m*x*y + z*s, m*x*z - y*s, 0,
		m*x*y - z*s, m*y*y + c, m*y*z + x*s, 0,
		m*x*z + y*s, m*y*z - x*s, m*z*z + c, 0,
		0, 0, 0, 1,
	}

	return NewTensor(data, []int{4, 4})
}

// 矩阵乘法
func (a *Tensor) MatMulMatrix(b *Tensor) *Tensor {
	// 验证矩阵形状
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("Matrix multiplication requires 2D tensors")
	}
	if a.Shape[1] != b.Shape[0] {
		panic(fmt.Sprintf("Shape mismatch: %v vs %v", a.Shape, b.Shape))
	}

	// 获取矩阵维度
	m := a.Shape[0] // 结果行数
	n := b.Shape[1] // 结果列数
	k := a.Shape[1] // 公共维度

	// 初始化结果矩阵数据
	result := make([]float64, m*n)

	// 执行矩阵乘法
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for p := 0; p < k; p++ {
				sum += a.Data[i*k+p] * b.Data[p*n+j]
			}
			result[i*n+j] = sum
		}
	}

	return NewTensor(result, []int{m, n})
}

// 链式旋转方法（类似原Rotate方法）
func (a *Tensor) Rotate(axis *Tensor, angle float64) *Tensor {
	rotation := RotateTensor(axis, angle)
	return rotation.MatMulMatrix(a)
}

// Viewport 创建视口变换矩阵
func Viewport(x, y, w, h float64) *Tensor {
	// 计算视口边界
	l := x
	b := y
	r := x + w
	t := y + h

	// 构建4x4视口变换矩阵（行主序）
	data := []float64{
		(r - l) / 2, 0, 0, (r + l) / 2,
		0, (t - b) / 2, 0, (t + b) / 2,
		0, 0, 0.5, 0.5,
		0, 0, 0, 1,
	}

	return NewTensor(data, []int{4, 4})
}

func (m *Tensor) MulPosition(v *Tensor) *Tensor {
	// 验证矩阵为4x4
	if !m.IsMatrix() || m.Shape[0] != 4 || m.Shape[1] != 4 {
		panic("MulPosition需要4x4矩阵")
	}

	// 验证输入为3D向量
	if !v.IsVector() || len(v.Data) != 3 {
		panic("输入向量需要是3D向量")
	}

	// 转换为齐次坐标 [x, y, z, 1]
	homoData := make([]float64, 4)
	copy(homoData, v.Data)
	homoData[3] = 1.0
	homogeneous := NewTensor(homoData, []int{4})

	// 执行矩阵变换（优化后的直接计算）
	result := make([]float64, 4)
	mData := m.Data
	vData := homogeneous.Data

	// 直接展开循环优化性能
	result[0] = mData[0]*vData[0] + mData[1]*vData[1] + mData[2]*vData[2] + mData[3]*vData[3]
	result[1] = mData[4]*vData[0] + mData[5]*vData[1] + mData[6]*vData[2] + mData[7]*vData[3]
	result[2] = mData[8]*vData[0] + mData[9]*vData[1] + mData[10]*vData[2] + mData[11]*vData[3]
	result[3] = mData[12]*vData[0] + mData[13]*vData[1] + mData[14]*vData[2] + mData[15]*vData[3]

	// 返回前三个分量（忽略w分量）
	return NewTensor(result[:3], []int{3})
}
