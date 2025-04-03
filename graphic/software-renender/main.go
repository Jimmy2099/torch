package main

import (
	"fmt"
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

// 渲染器结构体，包含绘图参数和缓冲区信息
type Render struct {
	p         *plot.Plot
	width     int
	height    int
	frameBuff *image.RGBA
	camera    *glm.Vector
	lookAt    *glm.Vector
	up        *glm.Vector
	fovy      float64
	near      float64
	far       float64
	scale     float64
	zBuffer   [][]float64
}

// 初始化渲染器，设置默认参数和缓冲区
func (m *Render) Init() {
	m.p = plot.New()
	m.height = 1080
	m.width = 1920
	{
		// 设置相机参数
		m.camera = NewVec3(0, 0, 1)
		m.lookAt = NewVec3(0, 0, 0)
		m.up = NewVec3(0, 1, 0)
		m.fovy = 40
		m.near = 1
		m.far = 10
		m.scale = 12
	}
	{
		// 初始化深度缓冲区
		m.zBuffer = make([][]float64, m.width)
		for i := range m.zBuffer {
			m.zBuffer[i] = make([]float64, m.height)
			for j := range m.zBuffer[i] {
				m.zBuffer[i][j] = 1 // 深度范围默认为 0 到 1
			}
		}
	}
	// 初始化帧缓冲区
	upLeft := image.Point{0, 0}
	lowRight := image.Point{m.width, m.height}
	m.frameBuff = image.NewRGBA(image.Rectangle{upLeft, lowRight})
	if true {
		// 设置背景颜色为黑色
		background := color.RGBA{0, 0, 0, 255}
		draw.Draw(m.frameBuff, m.frameBuff.Bounds(), &image.Uniform{background}, image.ZP, draw.Src)
	}
}

// 添加矢量数据到帧缓冲区
func (m *Render) AddVecData(data []*glm.Vector) {
	fill := color.RGBA{R: 255, G: 0, B: 0, A: 255} // 设置绘制颜色为红色
	for i := 0; i < len(data); i++ {
		m.frameBuff.Set(int(data[i].X), int(data[i].Y), fill) // 绘制点到缓冲区
	}
}

// 将帧缓冲区内容保存为 PNG 图像
func (m *Render) Draw() {
	f, _ := os.Create(fmt.Sprint(time.Now().Unix()) + ".png") // 使用当前时间戳命名文件
	png.Encode(f, imaging.FlipV(m.frameBuff))                 // 垂直翻转图像后保存
}

// 创建一个新的三维向量
func NewVec3(x, y, z float64) *glm.Vector {
	vec3 := glm.Vector{X: x, Y: y, Z: z}
	return &vec3
}

// 绘制两点之间的线条（包含深度信息）
func (r *Render) drawLine(v0, v1 *glm.Vector) []*glm.Vector {
	var result []*glm.Vector
	for t := 0.0; t < 1.0; t += 0.01 {
		x := v0.X + (v1.X-v0.X)*t
		y := v0.Y + (v1.Y-v0.Y)*t
		z := v0.Z + (v1.Z-v0.Z)*t
		result = append(result, NewVec3(x, y, z))
	}
	return result
}

// 绘制两点之间的线条（忽略深度信息）
func (m *Render) drawLineWithoutZBuff(v0 *glm.Vector, v1 *glm.Vector) (result []*glm.Vector) {
	for t := float64(0); t < 1; t += 0.01 {
		x := v0.X + (v1.X-v0.X)*t
		y := v0.Y + (v1.Y-v0.Y)*t
		result = append(result, NewVec3(x, y, 0))
	}
	return
}

// 绘制三角形边框
func (m *Render) drawTriangle(t0 *glm.Vector, t1 *glm.Vector, t2 *glm.Vector) (result []*glm.Vector) {
	result = append(result, m.drawLine(t0, t1)...) // t0 到 t1 的线段
	result = append(result, m.drawLine(t1, t2)...) // t1 到 t2 的线段
	result = append(result, m.drawLine(t2, t0)...) // t2 到 t0 的线段
	return
}

var m *Render

func main() {
	//pprof
	go func() {
		http.ListenAndServe("localhost:6060", nil)
	}()
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

	//Vec2i(10, 70),   Vec2i(50, 160),  Vec2i(70, 80)
	//t1 := NewVec3(10, 70, 0)
	//t2 := NewVec3(50, 160, 0)
	//t3 := NewVec3(70, 80, 0)
	var data3 []*glm.Vector
	for _, face := range objData.Faces {
		data3 = append(data3, m.drawTriangle(NewVec3(face.Points[0].Vertex.X, face.Points[0].Vertex.Y, face.Points[0].Vertex.Z),
			NewVec3(face.Points[1].Vertex.X, face.Points[1].Vertex.Y, face.Points[1].Vertex.Z),
			NewVec3(face.Points[2].Vertex.X, face.Points[2].Vertex.Y, face.Points[2].Vertex.Z))...)
	}
	model := glm.Identity()
	projection := glm.Perspective(m.fovy, float64(m.width)/float64(m.height), m.near, m.far)
	viewport := glm.Viewport(0, 0, float64(1), float64(1))
	view := glm.LookAt(*m.camera, *m.lookAt, *m.up)
	var matrix glm.Matrix
	if false {
		//record exec time
		model = model.Rotate(glm.Vector{Y: 1, X: 0, Z: 0}, glm.Radians(5))
		matrix = projection.Mul(view).Mul(viewport).Mul(model)
		recordExecTime(&matrix, data3)
		os.Exit(0)
	}
	//m.Draw()
	//a := app.New()
	//w := a.NewWindow("Images")
	go func() {
		for {
			view = glm.LookAt(*m.camera, *m.lookAt, *m.up)
			model = model.Rotate(glm.Vector{Y: 1, X: 0, Z: 0}, glm.Radians(5))
			matrix = projection.Mul(view).Mul(viewport).Mul(model)
			data4 := Camera(&matrix, data3)
			//Multithreading Optimize
			//data4 := CameraMultithreading(&matrix, data3)

			data4 = m.Scaling(data4)
			m.AddVecData(data4)
			frameBuff = ebiten.NewImageFromImage(imaging.FlipV(m.frameBuff))
			//img := canvas.NewImageFromImage(imaging.FlipV(m.frameBuff))
			if g != nil {
				g.FrameBuff = frameBuff
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
				for i := range m.zBuffer {
					for j := range m.zBuffer[i] {
						m.zBuffer[i][j] = 1
					}
				}
			}
		}
	}()
	g.FrameBuff = frameBuff
	g.Camera = m.camera
	g.RunGame()
	//w.Resize(fyne.NewSize(640, 480))
	//w.ShowAndRun()
}

var frameBuff *ebiten.Image

// 应用视图变换到点集
func Camera(matrix *glm.Matrix, v []*glm.Vector) (result []*glm.Vector) {
	for i := 0; i < len(v); i++ {
		vx := matrix.MulPosition(*v[i])
		result = append(result, &vx)
	}
	return
}

// 表示是否已经初始化了多线程处理的标志变量
var CameraMultithreadingInit = false

// 用于传递待处理向量数据的通道
var CameraMultithreadingChanel = make(chan []*glm.Vector, 100)

// 用于返回多线程处理结果的通道
var CameraMultithreadingResultChannel = make(chan []*glm.Vector, 100)

// 定义多线程处理时的线程数量
var threadingNum = 5

func CameraMultithreading(matrix *glm.Matrix, vDataList []*glm.Vector) (result []*glm.Vector) {
	// 检查并初始化多线程的运行环境
	if CameraMultithreadingInit == false {
		for i := 0; i < threadingNum; i++ {
			go func() {
				for {
					var resultTmp []*glm.Vector
					// 从通道中接收数据片段
					v := <-CameraMultithreadingChanel
					for j := 0; j < len(v); j++ {
						// 应用矩阵变换处理每个向量
						vx := matrix.MulPosition(*v[j])
						resultTmp = append(resultTmp, &vx)
					}
					// 将处理结果发送回结果通道
					CameraMultithreadingResultChannel <- resultTmp
				}
			}()
		}
		CameraMultithreadingInit = true // 确保只初始化一次
	}

	// 分割数据为多个部分，分发给工作线程
	pos := splitArrayIntoNParts(len(vDataList), threadingNum)
	for i := 0; i < len(pos); i++ {
		CameraMultithreadingChanel <- vDataList[pos[i][0]:pos[i][1]]
	}

	// 收集所有线程的计算结果
	for i := 0; i < threadingNum; i++ {
		resultTmp := <-CameraMultithreadingResultChannel
		result = append(result, resultTmp...)
	}
	return
}

// 将点集均匀分割成多部分
func splitArrayIntoNParts(length int, n int) [][]int {
	result := make([][]int, 0, n)
	partSize := length / n
	remainder := length % n

	start := 0
	for i := 0; i < n; i++ {
		end := start + partSize
		if i < remainder {
			end++
		}

		result = append(result, []int{start, end})
		start = end
	}
	return result
}

// 缩放点集到帧缓冲区范围
func (m *Render) Scaling(v []*glm.Vector) (result []*glm.Vector) {
	for i := 0; i < len(v); i++ {
		result = append(result, NewVec3(float64(m.width)*((v[i].X-0.5)/m.scale+0.5),
			float64(m.height)*((v[i].Y-0.5)/m.scale+0.5), v[i].Z))
	}
	return
}

// 测量函数执行时间
func recordFunctionTime(f func(), filename string, iterations int) {
	file, err := os.Create(filename)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close()

	for i := 0; i < iterations; i++ {
		start := time.Now()
		f()
		duration := time.Since(start)
		_, err := file.WriteString(fmt.Sprintf("%d\n", duration.Microseconds()))
		if err != nil {
			fmt.Println("Error writing to file:", err)
			return
		}
	}
}

// 测量视图变换的执行时间
func recordExecTime(matrix *glm.Matrix, vDataList []*glm.Vector) {
	recordFunctionTime(func() {
		Camera(matrix, vDataList)
	}, "times_single-threading.txt", 100)

	recordFunctionTime(func() {
		CameraMultithreading(matrix, vDataList)
	}, "times_multi-threading.txt", 100)
}
