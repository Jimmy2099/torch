package fmt

import (
	"fmt"
	"io"
	"os"
)

var (
	EnableLog = true
)

func init() {
	if os.Getenv("Enable_Log") == "false" {
		EnableLog = false
	}
}

func MustPrintln(a ...any) (n int, err error) {
	return fmt.Println(a...)
}

func MustPrintf(format string, a ...any) (n int, err error) {
	return fmt.Printf(format, a...)
}

func Sprintln(a ...any) string {
	return fmt.Sprintln(a...)
}

func Errorf(format string, a ...any) (err error) {
	if !EnableLog {
		return
	}
	return fmt.Errorf(format, a...)
}

func Println(a ...any) {
	if !EnableLog {
		return
	}
	fmt.Println(a...)
}

func Printf(format string, a ...any) (n int, err error) {
	if !EnableLog {
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

func Fprintf(w io.Writer, format string, a ...any) (n int, err error) {
	return fmt.Fprintf(w, format, a...)
}
