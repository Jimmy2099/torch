package torch

import (
	"github.com/Jimmy2099/torch/data_store/tensor"
	"github.com/Jimmy2099/torch/pkg/fmt"
	"sync/atomic"
	"time"
)


type CalcStruct struct {
	Data        *tensor.Tensor
	StartAndEnd []uint64
}
type LayerLinearMC struct {
	*LinearLayer
	calcChannel chan *CalcStruct

	counter     uint64 // 处理总数
	totalTime   uint64 // 总耗时（纳秒）
	printTicker *time.Ticker
}

func NewLayerLinearMC(inputDim int, outputDim int) *LayerLinearMC {
	layer := &LayerLinearMC{
		LinearLayer: NewLinearLayer(inputDim, outputDim),
		calcChannel: make(chan *CalcStruct, 512),
		printTicker: time.NewTicker(time.Second),
	}

	for i := 0; i < 15; i++ {
		go layer.RunCalculation()
	}
	go layer.monitorPerformance()

	return layer
}

func (m *LayerLinearMC) monitorPerformance() {
	for range m.printTicker.C {
		count := atomic.LoadUint64(&m.counter)
		total := atomic.LoadUint64(&m.totalTime)

		if count == 0 {
			fmt.Println("[Perf] No operations completed in last second")
			continue
		}

		avgNs := total / count
		opsPerSec := count
		fmt.Printf("[Perf] Ops/s: %d | Avg Time: %s\n",
			opsPerSec,
			time.Duration(avgNs).Round(time.Microsecond))

		atomic.StoreUint64(&m.counter, 0)
		atomic.StoreUint64(&m.totalTime, 0)
	}
}


func (m *LayerLinearMC) CreateCalculationTask() {
	for i := 0; i < 15; i++ {
		go m.RunCalculation()
	}
}

func (m *LayerLinearMC) RunCalculation() {
	size := uint64(100)
	for {
		data := <-m.calcChannel
		start := time.Now()
		result := m.Forward(data.Data)
		elapsed := time.Since(start)
		atomic.AddUint64(&m.counter, size)
		atomic.AddUint64(&m.totalTime, uint64(elapsed.Nanoseconds()))

		_ = result // 实际使用时需要处理结果
	}
}
