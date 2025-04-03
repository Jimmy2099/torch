module softrender

go 1.23.7

require (
	github.com/Jimmy2099/torch v0.0.0-20250403061702-b4f02e7526ec
	github.com/disintegration/imaging v1.6.2
	github.com/gabstv/cimgui-go v0.0.0-20231031221758-68bd718f94cc
	github.com/gabstv/ebiten-imgui/v3 v3.0.0
	github.com/hajimehoshi/ebiten/v2 v2.8.6
	github.com/sheenobu/go-obj/obj v0.0.0-20190106231111-fb5ef7341b74
	gitlab.com/brickhill/site/fauxgl v0.0.0-20200818143847-27cddc103802
	gonum.org/v1/plot v0.15.2
)

require (
	codeberg.org/go-fonts/liberation v0.4.1 // indirect
	codeberg.org/go-latex/latex v0.0.1 // indirect
	codeberg.org/go-pdf/fpdf v0.10.0 // indirect
	git.sr.ht/~sbinet/gg v0.6.0 // indirect
	github.com/ajstarks/svgo v0.0.0-20211024235047-1546f124cd8b // indirect
	github.com/campoy/embedmd v1.0.0 // indirect
	github.com/ebitengine/gomobile v0.0.0-20240911145611-4856209ac325 // indirect
	github.com/ebitengine/hideconsole v1.0.0 // indirect
	github.com/ebitengine/purego v0.8.0 // indirect
	github.com/fogleman/simplify v0.0.0-20170216171241-d32f302d5046 // indirect
	github.com/golang/freetype v0.0.0-20170609003504-e2365dfdc4a0 // indirect
	github.com/jezek/xgb v1.1.1 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	golang.org/x/image v0.24.0 // indirect
	golang.org/x/sync v0.11.0 // indirect
	golang.org/x/sys v0.30.0 // indirect
	golang.org/x/text v0.22.0 // indirect
)

replace (
	gitlab.com/brickhill/site/fauxgl v0.0.0-20200818143847-27cddc103802 => github.com/thedenbruh/brickgl v0.0.0-20231013201946-d440f553eda2
	github.com/Jimmy2099/torch v0.0.0-20250403061702-b4f02e7526ec => ../../../
)
