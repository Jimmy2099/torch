package testing

import (
	"os"
	"strings"
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
