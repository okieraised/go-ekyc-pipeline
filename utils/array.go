package utils

import (
	"unsafe"
)

func BytesToT32[T int32 | float32](arr []byte) []T {
	if len(arr) == 0 {
		return nil
	}

	l := len(arr) / 4
	ptr := unsafe.Pointer(&arr[0])
	return (*[1 << 26]T)(ptr)[:l:l]
}

func BytesToT64[T int64 | float64](arr []byte) []T {
	if len(arr) == 0 {
		return nil
	}

	l := len(arr) / 8
	ptr := unsafe.Pointer(&arr[0])
	return (*[1 << 26]T)(ptr)[:l:l]
}
