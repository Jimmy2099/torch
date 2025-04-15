package ui

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	imgui "github.com/gabstv/cimgui-go"
	ebimgui "github.com/gabstv/ebiten-imgui/v3"
	"github.com/gabstv/ebiten-imgui/v3/imcolor"
	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"image/color"
)

func UIMain(frameBuff *ebiten.Image, camera *tensor.Tensor) *G {
	gg := &G{
		name:       "Torch Graphic",
		clearColor: [3]float32{0, 0, 0},
		FrameBuff:  frameBuff,
		Camera:     camera,
	}

	ebiten.SetWindowSize(1200, 900)
	ebiten.SetWindowTitle(gg.name)
	mgr = ebimgui.GlobalManager()
	return gg
}

func (g *G) RunGame() {
	ebiten.RunGame(g)
}

type G struct {
	clearColor [3]float32
	floatVal   float32
	counter    int
	name       string
	FrameBuff  *ebiten.Image
	Camera     *tensor.Tensor
}

func (g *G) Draw(screen *ebiten.Image) {
	screen.Fill(color.RGBA{uint8(g.clearColor[0] * 255), uint8(g.clearColor[1] * 255), uint8(g.clearColor[2] * 255), 255})
	ebitenutil.DebugPrint(screen, fmt.Sprintf("TPS: %.2f", ebiten.CurrentTPS()))
	ebimgui.Draw(screen)
}

func InputText(label string, buf *string) bool {
	return imgui.InputTextWithHint(label, "", buf, 0, nil)
}

func (g *G) Update() error {
	ebimgui.Update(1.0 / 60.0)
	ebimgui.BeginFrame()
	defer func() {
		ebimgui.EndFrame()
		//mgr.Cache.RemoveTexture(imgui.TextureID(&myImageIDRef))
	}()

	imgui.Text("ภาษาไทย测试조선말")                        // To display these, you'll need to register a compatible font
	imgui.Text("Hello, world!")                       // Display some text
	imgui.SliderFloat("float", &g.floatVal, 0.0, 1.0) // Edit 1 float using a slider from 0.0f to 1.0f
	imgui.ColorEdit3("clear color", &g.clearColor)    // Edit 3 floats representing a color

	//imgui.Checkbox("Demo Window", &showDemoWindow) // Edit bools storing our window open/close state
	//imgui.Checkbox("Go Demo Window", &showGoDemoWindow)
	//imgui.Checkbox("Another Window", &showAnotherWindow)

	if imgui.Button("Button") { // Buttons return true when clicked (most widgets return true when edited/activated)
		g.counter++
	}
	imgui.SameLine()
	imgui.Text(fmt.Sprintf("counter = %d", g.counter))

	if InputText("Window title", &g.name) {
		ebiten.SetWindowTitle(g.name)
	}

	xcol := imcolor.ToVec4(color.RGBA{
		R: 0xFF,
		G: 0x00,
		B: 0xFF,
		A: 0x99,
	})

	imgui.PushStyleColorVec4(imgui.ColText, xcol)
	imgui.Text(fmt.Sprintf("fps = %f", ebiten.CurrentFPS()))
	imgui.PopStyleColor()
	{
		imgui.Begin("Engine")
		imgui.Text("Engine")
		if g.FrameBuff != nil {
			mgr.Cache.SetTexture(imgui.TextureID(&myImageIDRef), g.FrameBuff) // Texture ID 10 will contain this example image
			Image(imgui.TextureID(&myImageIDRef), imgui.Vec2{X: 600, Y: 480})
		}
		imgui.End()
	}
	Setting(g.Camera)
	return nil
}

func Setting(camera *tensor.Tensor) {
	imgui.Begin("Settings")
	cx := float32(camera.X())
	cy := float32(camera.Y())
	cz := float32(camera.Z())
	imgui.Text("Camera")
	imgui.SliderFloat("X", &cx, -10.0, 10)
	imgui.SliderFloat("Y", &cy, -10.0, 10)
	imgui.SliderFloat("Z", &cz, -10.0, 10)
	camera.Data = []float32{float32(cx), float32(cy), float32(cz)}
	imgui.End()
}

var mgr *ebimgui.Manager

var (
	myImageIDRef int = 10
)

func Image(tid imgui.TextureID, size imgui.Vec2) {
	uv0 := imgui.NewVec2(0, 0)
	uv1 := imgui.NewVec2(1, 1)
	border_col := imgui.NewVec4(0, 0, 0, 0)
	tint_col := imgui.NewVec4(1, 1, 1, 1)

	imgui.ImageV(tid, size, uv0, uv1, tint_col, border_col)
}

func (g *G) Layout(outsideWidth, outsideHeight int) (int, int) {
	ebimgui.SetDisplaySize(float32(1200), float32(900))
	return 1200, 900
}
