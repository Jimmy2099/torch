package testing

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	"os"
	"strings"
	"time"
)

// IsTesting 检测当前是否在 go test 环境中运行
func IsTesting() bool {
	for _, arg := range os.Args {
		if strings.HasPrefix(arg, "-test.") {
			return true
		}
	}
	return false
}

// 智能时间格式化函数（核心逻辑）
func formatDuration(d time.Duration) string {
	switch {
	case d >= time.Hour:
		return fmt.Sprintf("%.2fh", d.Hours())
	case d >= time.Minute:
		return fmt.Sprintf("%.2fm", d.Minutes())
	case d >= time.Second:
		return fmt.Sprintf("%.3fs", d.Seconds())
	case d >= time.Millisecond:
		return fmt.Sprintf("%.3fms", float64(d)/float64(time.Millisecond))
	case d >= time.Microsecond:
		return fmt.Sprintf("%.3fµs", float64(d)/float64(time.Microsecond))
	default:
		return fmt.Sprintf("%dns", d.Nanoseconds())
	}
}

func TimeMeasure(start time.Time) string {
	//time.Sleep(350*time.Millisecond + 458*time.Microsecond)
	elapsed := time.Since(start)
	return fmt.Sprintf("Execution time: %s \n", formatDuration(elapsed))
}
