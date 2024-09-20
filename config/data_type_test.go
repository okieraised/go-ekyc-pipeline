package config

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
	"testing"
)

func TestConvertMetadataToTensors(t *testing.T) {
	lmk := &FaceLandmark{
		LeftEye: Coordinate2D{
			X: 266.14566,
			Y: 220.05692,
		},
		RightEye: Coordinate2D{
			X: 397.149,
			Y: 221.45383,
		},
		Nose: Coordinate2D{
			X: 332.13202,
			Y: 258.6127,
		},
		LeftMouth: Coordinate2D{
			X: 284.05356,
			Y: 294.186,
		},
		RightMouth: Coordinate2D{
			X: 380.22375,
			Y: 294.31165,
		},
	}

	actual := ConvertMetadataToTensors(lmk)

	expect := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking([]float32{
			266.14566, 220.05692,
			397.149, 221.45383,
			332.13202, 258.6127,
			284.05356, 294.186,
			380.22375, 294.31165,
		}),
	)
	assert.Equal(t, expect, actual)

	fmt.Println(actual)
}
