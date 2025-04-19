package tensor

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	math "github.com/chewxy/math32"
)

func NewVec3(x, y, z float32) *Tensor {
	return NewTensor([]float32{x, y, z}, []int{3})
}

func (m *Tensor) X() float32 {
	return m.Data[0]
}

func (m *Tensor) Y() float32 {
	return m.Data[1]
}

func (m *Tensor) Z() float32 {
	return m.Data[2]
}

func (t *Tensor) IsMatrix() bool {
	return len(t.shape) == 2
}

func (t *Tensor) IsVector() bool {
	return len(t.shape) == 1
}

func (t *Tensor) checkMatrix() {
	if !t.IsMatrix() {
		panic("Operation requires matrix")
	}
}

func (t *Tensor) checkSquareMatrix() {
	t.checkMatrix()
	if t.shape[0] != t.shape[1] {
		panic("Operation requires square matrix")
	}
}

func (t *Tensor) checkVector() {
	if !t.IsVector() {
		panic("Operation requires vector")
	}
}

func (t *Tensor) Determinant() float32 {
	t.checkSquareMatrix()
	if t.shape[0] != 4 {
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

func (t *Tensor) Inverse() *Tensor {
	t.checkSquareMatrix()
	if t.shape[0] != 4 {
		panic("Inverse currently only supported for 4x4 matrices")
	}

	m := t.Data
	inv := make([]float32, 16)
	det := t.Determinant()

	inv[0] = (m[5]*m[10]*m[15] - m[5]*m[11]*m[14] + m[7]*m[9]*m[14] - m[6]*m[9]*m[15] + m[6]*m[11]*m[13] - m[7]*m[10]*m[13]) / det

	return NewTensor(inv, []int{4, 4})
}

func Identity() *Tensor {
	return NewTensor([]float32{
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

	return NewTensor([]float32{
		1, 0, 0, v.Data[0],
		0, 1, 0, v.Data[1],
		0, 0, 1, v.Data[2],
		0, 0, 0, 1}, []int{4, 4})
}

func Rotate(axis *Tensor, angle float32) *Tensor {
	axis = axis.Normalize()
	s := math.Sin(angle)
	c := math.Cos(angle)
	m := 1 - c

	x := axis.Data[0]
	y := axis.Data[1]
	z := axis.Data[2]

	return NewTensor([]float32{
		m*x*x + c, m*x*y + z*s, m*x*z - y*s, 0,
		m*x*y - z*s, m*y*y + c, m*y*z + x*s, 0,
		m*x*z + y*s, m*y*z - x*s, m*z*z + c, 0,
		0, 0, 0, 1}, []int{4, 4})
}

func Perspective(fovy, aspect, near, far float32) *Tensor {
	f := 1.0 / math.Tan(fovy/2)
	return NewTensor([]float32{
		f / aspect, 0, 0, 0,
		0, f, 0, 0,
		0, 0, (far + near) / (near - far), (2 * far * near) / (near - far),
		0, 0, -1, 0}, []int{4, 4})
}

func LookAt(eye, center, up *Tensor) *Tensor {
	z := eye.Sub(center).Normalize()
	x := up.Cross(z).Normalize()
	y := z.Cross(x)

	return NewTensor([]float32{
		x.Data[0], x.Data[1], x.Data[2], -x.Dot(eye),
		y.Data[0], y.Data[1], y.Data[2], -y.Dot(eye),
		z.Data[0], z.Data[1], z.Data[2], -z.Dot(eye),
		0, 0, 0, 1}, []int{4, 4})
}

func (t *Tensor) Dot(other *Tensor) float32 {
	if !t.IsVector() || !other.IsVector() {
		panic("Dot product requires vectors")
	}
	if len(t.Data) != len(other.Data) {
		panic("Vectors must have same length")
	}

	var sum float32
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
	return NewTensor([]float32{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0]}, []int{3})
}

func (t *Tensor) Normalize() *Tensor {
	if !t.IsVector() {
		panic("Normalization requires vector")
	}

	var length float32
	for _, v := range t.Data {
		length += v * v
	}
	length = math.Sqrt(length)

	data := make([]float32, len(t.Data))
	for i := range data {
		data[i] = t.Data[i] / length
	}
	return NewTensor(data, t.shape)
}

func (t *Tensor) Homogeneous() *Tensor {
	if !t.IsVector() || len(t.Data) != 3 {
		panic("Requires 3D vector")
	}
	return NewTensor(append(t.Data, 1.0), []int{4})
}

func RotateTensor(axis *Tensor, angle float32) *Tensor {
	if !axis.IsVector() || len(axis.Data) != 3 {
		panic("Rotate requires 3D vector axis")
	}

	normAxis := axis.Normalize()
	x := normAxis.Data[0]
	y := normAxis.Data[1]
	z := normAxis.Data[2]

	s := math.Sin(angle)
	c := math.Cos(angle)
	m := 1 - c

	data := []float32{
		m*x*x + c, m*x*y + z*s, m*x*z - y*s, 0,
		m*x*y - z*s, m*y*y + c, m*y*z + x*s, 0,
		m*x*z + y*s, m*y*z - x*s, m*z*z + c, 0,
		0, 0, 0, 1,
	}

	return NewTensor(data, []int{4, 4})
}

func (a *Tensor) MatMulMatrix(b *Tensor) *Tensor {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		panic("Matrix multiplication requires 2D tensors")
	}
	if a.shape[1] != b.shape[0] {
		panic(fmt.Sprintf("shape mismatch: %v vs %v", a.shape, b.shape))
	}

	m := a.shape[0]
	n := b.shape[1]
	k := a.shape[1]

	result := make([]float32, m*n)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a.Data[i*k+p] * b.Data[p*n+j]
			}
			result[i*n+j] = sum
		}
	}

	return NewTensor(result, []int{m, n})
}

func (a *Tensor) Rotate(axis *Tensor, angle float32) *Tensor {
	rotation := RotateTensor(axis, angle)
	return rotation.MatMulMatrix(a)
}

func Viewport(x, y, w, h float32) *Tensor {
	l := x
	b := y
	r := x + w
	t := y + h

	data := []float32{
		(r - l) / 2, 0, 0, (r + l) / 2,
		0, (t - b) / 2, 0, (t + b) / 2,
		0, 0, 0.5, 0.5,
		0, 0, 0, 1,
	}

	return NewTensor(data, []int{4, 4})
}

func (m *Tensor) MulPosition(v *Tensor) *Tensor {
	if !m.IsMatrix() || m.shape[0] != 4 || m.shape[1] != 4 {
		panic("MulPosition requires a 4x4 matrix")
	}

	if !v.IsVector() || len(v.Data) != 3 {
		panic("Input vector needs to be a 3D vector")
	}

	homoData := make([]float32, 4)
	copy(homoData, v.Data)
	homoData[3] = 1.0
	homogeneous := NewTensor(homoData, []int{4})

	result := make([]float32, 4)
	mData := m.Data
	vData := homogeneous.Data

	result[0] = mData[0]*vData[0] + mData[1]*vData[1] + mData[2]*vData[2] + mData[3]*vData[3]
	result[1] = mData[4]*vData[0] + mData[5]*vData[1] + mData[6]*vData[2] + mData[7]*vData[3]
	result[2] = mData[8]*vData[0] + mData[9]*vData[1] + mData[10]*vData[2] + mData[11]*vData[3]
	result[3] = mData[12]*vData[0] + mData[13]*vData[1] + mData[14]*vData[2] + mData[15]*vData[3]

	return NewTensor(result[:3], []int{3})
}
