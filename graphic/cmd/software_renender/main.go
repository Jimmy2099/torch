package main

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"github.com/disintegration/imaging"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/sheenobu/go-obj/obj"
	glm "gitlab.com/brickhill/site/fauxgl"
	"gonum.org/v1/plot"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"net/http"
	_ "net/http/pprof"
	"os"
	"softrender/ui"
	"time"
)

// glm "gitlab.com/brickhill/site/fauxgl"

type Render struct {
	p          *plot.Plot
	width      int
	height     int
	frameBuff  *image.RGBA
	GFrameBuff *ebiten.Image
	camera     *tensor.Tensor
	lookAt     *tensor.Tensor
	up         *tensor.Tensor
	fovy       float32
	near       float32
	far        float32
	scale      float32
	ZBuffer    [][]float32
}

func (m *Render) Init() {
	m.p = plot.New()
	m.height = 1080
	m.width = 1920
	{
		m.fovy = 45
		m.near = 1
		m.far = 10
		m.scale = 8.0

		m.camera = tensor.NewVec3(0, 1, 1)
		m.lookAt = tensor.NewVec3(0, 0, 0)
		m.up = tensor.NewVec3(0, 1, 0)
	}
	{
		m.ZBuffer = make([][]float32, m.width)
		for i := range m.ZBuffer {
			m.ZBuffer[i] = make([]float32, m.height)
			for j := range m.ZBuffer[i] {
				m.ZBuffer[i][j] = 1
			}
		}
	}
	upLeft := image.Point{0, 0}
	lowRight := image.Point{m.width, m.height}
	m.frameBuff = image.NewRGBA(image.Rectangle{upLeft, lowRight})
	if true {
		background := color.RGBA{0, 0, 0, 255}
		draw.Draw(m.frameBuff, m.frameBuff.Bounds(), &image.Uniform{background}, image.ZP, draw.Src)
	}
}

func main() {
	//for test
	glm.Identity()

	//pprof
	go func() {
		http.ListenAndServe("localhost:6060", nil)
	}()

	var m *Render
	g := ui.UIMain(nil, nil)

	m = &Render{}
	m.Init()

	// load our OBJ
	f, err := os.Open("./model/teapot.obj")
	if err != nil {
		panic(err)
	}
	objData, err := obj.NewReader(f).Read()
	if err != nil {
		panic(err)
	}

	var data3 []*tensor.Tensor
	for _, face := range objData.Faces {
		data3 = append(data3, m.drawTriangle(NewVec3(face.Points[0].Vertex.X, face.Points[0].Vertex.Y, face.Points[0].Vertex.Z),
			NewVec3(face.Points[1].Vertex.X, face.Points[1].Vertex.Y, face.Points[1].Vertex.Z),
			NewVec3(face.Points[2].Vertex.X, face.Points[2].Vertex.Y, face.Points[2].Vertex.Z))...)
	}
	//	model := glm.Identity()
	model := tensor.Identity()
	projection := tensor.Perspective(m.fovy, float32(m.width)/float32(m.height), m.near, m.far)
	viewport := tensor.Viewport(0, 0, float32(1), float32(1))
	view := tensor.LookAt(m.camera, m.lookAt, m.up)
	var matrix *tensor.Tensor
	go func() {
		for {
			view = tensor.LookAt(m.camera, m.lookAt, m.up)
			//model = model.Rotate(tensor.NewTensor([]float32{1, 0, 0}, []int{3}), glm.Radians(5))
			matrix = projection.MatMulMatrix(view).MatMulMatrix(viewport).MatMulMatrix(model)
			data4 := Camera(matrix, data3)
			//Multithreading Optimize
			//data4 := CameraMultithreading(&matrix, data3)

			data4 = m.Scaling(data4)
			m.AddVecData(data4)
			m.GFrameBuff = ebiten.NewImageFromImage(imaging.FlipV(m.frameBuff))
			//img := canvas.NewImageFromImage(imaging.FlipV(m.frameBuff))
			if g != nil {
				g.FrameBuff = m.GFrameBuff
			}
			//w.SetContent(img)
			//time.Sleep(time.Millisecond * 10)
			{
				upLeft := image.Point{0, 0}
				lowRight := image.Point{m.width, m.height}
				m.frameBuff = image.NewRGBA(image.Rectangle{upLeft, lowRight})
				background := color.RGBA{0, 0, 0, 255}
				draw.Draw(m.frameBuff, m.frameBuff.Bounds(), &image.Uniform{background}, image.ZP, draw.Src)
			}
			{
				for i := range m.ZBuffer {
					for j := range m.ZBuffer[i] {
						m.ZBuffer[i][j] = 1
					}
				}
			}
		}
	}()
	g.FrameBuff = m.GFrameBuff
	g.Camera = m.camera
	g.RunGame()

}

func (m *Render) AddVecData(data []*tensor.Tensor) {
	fill := color.RGBA{R: 255, G: 0, B: 0, A: 255} // 设置绘制颜色为红色
	for i := 0; i < len(data); i++ {
		m.frameBuff.Set(int(data[i].X()), int(data[i].Y()), fill) // 绘制点到缓冲区
	}
}

func (m *Render) Draw() {
	f, _ := os.Create(fmt.Sprint(time.Now().Unix()) + ".png")
	png.Encode(f, imaging.FlipV(m.frameBuff))
}

func NewVec3(x, y, z float32) *tensor.Tensor {
	return tensor.NewVec3(x, y, z)
}

func (r *Render) drawLine(v0, v1 *tensor.Tensor) []*tensor.Tensor {
	var result []*tensor.Tensor
	for t := float32(0.0); t < 1.0; t += 0.01 {
		x := v0.X() + (v1.X()-v0.X())*t
		y := v0.Y() + (v1.Y()-v0.Y())*t
		z := v0.Z() + (v1.Z()-v0.Z())*t
		result = append(result, NewVec3(x, y, z))
	}
	return result
}

func (m *Render) drawLineWithoutZBuff(v0 *tensor.Tensor, v1 *tensor.Tensor) (result []*tensor.Tensor) {
	for t := float32(0); t < 1; t += 0.01 {
		x := v0.X() + (v1.X()-v0.X())*t
		y := v0.Y() + (v1.Y()-v0.Y())*t
		result = append(result, NewVec3(x, y, 0))
	}
	return
}

func (m *Render) drawTriangle(t0 *tensor.Tensor, t1 *tensor.Tensor, t2 *tensor.Tensor) (result []*tensor.Tensor) {
	result = append(result, m.drawLine(t0, t1)...)
	result = append(result, m.drawLine(t1, t2)...)
	result = append(result, m.drawLine(t2, t0)...)
	return
}

func Camera(matrix *tensor.Tensor, v []*tensor.Tensor) (result []*tensor.Tensor) {
	for i := 0; i < len(v); i++ {
		vx := matrix.MulPosition(v[i])
		result = append(result, vx)
	}
	return
}

func (m *Render) Scaling(v []*tensor.Tensor) (result []*tensor.Tensor) {
	for i := 0; i < len(v); i++ {
		result = append(result, NewVec3(float32(m.width)*((v[i].X()-0.5)/m.scale+0.5),
			float32(m.height)*((v[i].Y()-0.5)/m.scale+0.5), v[i].Z()))
	}
	return
}
