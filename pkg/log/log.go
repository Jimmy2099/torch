package log

import (
	"github.com/Jimmy2099/torch/pkg/fmt"
	"log"
)

func Println(a ...any) {
	if !fmt.EnableLog {
		return
	}
	log.Println(a...)
}

func Printf(format string, a ...any) (n int, err error) {
	if !fmt.EnableLog {
		return
	}
	return fmt.Printf(format, a...)
}

func Sprint(a ...any) string {
	return fmt.Sprint(a...)
}

func Sprintf(format string, a ...any) string {
	return fmt.Sprintf(format, a...)
}

func Fatalf(format string, v ...any) {
	log.Fatalf(format, v...)
}

func Panicf(format string, v ...any) {
	log.Panicf(format, v...)
}
