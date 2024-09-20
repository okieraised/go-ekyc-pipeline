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

func TestNewFaceDetectionClient_InferBatch(t *testing.T) {

	triton, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	helperClient, err := NewFaceDetectionClient(triton, config.DefaultFaceDetectionParams)
	assert.NoError(t, err)

	img, err := genTestMidData()
	assert.NoError(t, err)

	outputs, err := helperClient.InferBatch([][]gocv.Mat{{*img}})
	assert.NoError(t, err)

	fmt.Println("outputs", outputs)
}

func TestNewFaceDetectionClient_InferSingle(t *testing.T) {

	triton, err := gotritonclient.NewTritonGRPCClient(
		tritonTestURL,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{PermitWithoutStream: true}),
	)
	assert.NoError(t, err)

	helperClient, err := NewFaceDetectionClient(triton, config.DefaultFaceDetectionParams)
	assert.NoError(t, err)

	img, err := genTestMidData()
	assert.NoError(t, err)

	outputs, err := helperClient.InferSingle([]gocv.Mat{*img})
	assert.NoError(t, err)

	fmt.Println("outputs", outputs)
}
