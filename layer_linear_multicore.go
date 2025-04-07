package torch

import (
	"fmt"
	"github.com/Jimmy2099/torch/data_struct/tensor"
	"sync/atomic"
	"time"
)

//cpu calculation TODO
//Single Instruction Multiple Data SIMD TODO

type CalcStruct struct {
	Data        *tensor.Tensor
	StartAndEnd []uint64
}
type LayerLinearMC struct {
	*LinearLayer
	calcChannel chan *CalcStruct
	//resultChannel chan []float32 //[posStart,posEnd,Data]

	// 性能统计字段
	counter     uint64 // 处理总数
	totalTime   uint64 // 总耗时（纳秒）
	printTicker *time.Ticker
}

func NewLayerLinearMC(inputDim int, outputDim int) *LayerLinearMC {
	layer := &LayerLinearMC{
		LinearLayer: NewLinearLayer(inputDim, outputDim),
		calcChannel: make(chan *CalcStruct, 512),
		//resultChannel: make(chan []float32, 100),
		printTicker: time.NewTicker(time.Second),
	}

	for i := 0; i < 15; i++ {
		go layer.RunCalculation()
	}
	// 启动性能监控
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

		// 重置计数器
		atomic.StoreUint64(&m.counter, 0)
		atomic.StoreUint64(&m.totalTime, 0)
	}
}

//func init() {
//	x := NewLayerLinearMC(10, 10)
//	for {
//		x.calcChannel <- &CalcStruct{Data: tensor.Ones([]int{10, 10}), StartAndEnd: []uint64{0, 100}}
//	}
//	//x.CreateCalculationTask()
//	fmt.Println(x)
//}

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
		// 记录性能数据
		elapsed := time.Since(start)
		atomic.AddUint64(&m.counter, size)
		atomic.AddUint64(&m.totalTime, uint64(elapsed.Nanoseconds()))

		// 处理结果（示例）
		_ = result // 实际使用时需要处理结果
	}
}
