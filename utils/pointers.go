package utils

func RefPointer[T string | bool | int | int8 | int16 | int32 | int64 | float32 | float64](val T) *T {
	return &val
}

func DerefPointer[T string | bool | int | int8 | int16 | int32 | int64 | float32 | float64](val *T) T {
	return *val
}
