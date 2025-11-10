package compute_graph

type ComputationalGraphSettings struct {
	ComputationalGraphRunMode
}

func NewComputationalGraphSettings() *ComputationalGraphSettings {
	return &ComputationalGraphSettings{
		ComputationalGraphRunMode: ComputationalGraphRunModeDebug,
	}
}

type ComputationalGraphRunMode int

const (
	ComputationalGraphRunModeUnDefine   ModeAvailable = iota
	ComputationalGraphRunModeDebug                    = 1
	ComputationalGraphRunModeProduction               = 2
)
