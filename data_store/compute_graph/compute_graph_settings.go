package compute_graph

type ComputationalGraphSettings struct {
	ComputationalGraphRunMode
}

func NewComputationalGraphSettings(mode ComputationalGraphRunMode) *ComputationalGraphSettings {
	return &ComputationalGraphSettings{
		ComputationalGraphRunMode: mode,
	}
}

type ComputationalGraphRunMode int

const (
	ComputationalGraphRunModeUnDefine   ModeAvailable = iota
	ComputationalGraphRunModeDebug                    = 1
	ComputationalGraphRunModeProduction               = 2
)

func (m *ComputationalGraphSettings) IsDebugMode() bool {
	if m.ComputationalGraphRunMode == ComputationalGraphRunModeDebug {
		return true
	}
	return false
}
