package modules

import (
	"fmt"
	"github.com/okieraised/go-ekyc-pipeline/config"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/stretchr/testify/assert"
	"gocv.io/x/gocv"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"testing"
)

func TestFaceIDClient_InferBatch(t *testing.T) {

	triton, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	far, err := genTestFarData()
	assert.NoError(t, err)

	mid, err := genTestMidData()
	assert.NoError(t, err)

	near, err := genTestNearData()
	assert.NoError(t, err)

	faceIDClient, err := NewFaceIDClient(
		triton,
		config.DefaultFaceIDParams,
	)
	assert.NoError(t, err)

	v, err := faceIDClient.InferBatch([][]gocv.Mat{{*far, *mid, *near}})
	assert.NoError(t, err)
	fmt.Println("v", len(v), v)
}
