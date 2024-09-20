package utils

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
	"testing"
)

func TestTensorToPoints(t *testing.T) {

	lmk := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(5, 2),
		tensor.WithBacking(
			[]float32{
				173.57267761, 191.85157776,
				450.2043457, 210.12382507,
				309.74865723, 302.90393066,
				180.64160156, 377.55731201,
				418.83895874, 392.7986145,
			}),
	)

	pts, err := TensorToPoints(lmk)
	assert.NoError(t, err)
	fmt.Println(pts)
}
