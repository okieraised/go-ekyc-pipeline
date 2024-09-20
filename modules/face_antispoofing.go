package modules

import (
	"github.com/okieraised/go-ekyc-pipeline/config"
	"github.com/okieraised/go-ekyc-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/okieraised/go-triton-client/triton_proto"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"image"
)

type FaceAntiSpoofingClient struct {
	tritonClient *gotritonclient.TritonGRPCClient
	ModelParams  *config.FaceAntiSpoofingParams
	ModelConfig  *triton_proto.ModelConfigResponse
}

func NewFaceAntiSpoofingClient(triton *gotritonclient.TritonGRPCClient, cfg *config.FaceAntiSpoofingParams) (*FaceAntiSpoofingClient, error) {

	inferenceConfig, err := triton.GetModelConfiguration(cfg.Timeout, cfg.ModelName, "")
	if err != nil {
		return nil, err
	}

	return &FaceAntiSpoofingClient{
		tritonClient: triton,
		ModelParams:  cfg,
		ModelConfig:  inferenceConfig,
	}, nil
}

func (c *FaceAntiSpoofingClient) preprocess(img gocv.Mat) (*tensor.Dense, error) {
	var err error
	resizedImg := gocv.NewMat()
	defer func(resizedImg *gocv.Mat) {
		cErr := resizedImg.Close()
		if cErr != nil && err == nil {
			err = cErr
		}
	}(&resizedImg)
	gocv.Resize(
		img,
		&resizedImg,
		image.Point{
			X: c.ModelParams.ImgSize,
			Y: c.ModelParams.ImgSize,
		},
		0.0,
		0.0,
		gocv.InterpolationLinear,
	)

	imgTensors := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(
			resizedImg.Cols(),
			resizedImg.Rows(),
			3,
		),
	)

	for z := range 3 {
		for y := range resizedImg.Cols() {
			for x := range resizedImg.Rows() {
				err := imgTensors.SetAt(
					(float32(resizedImg.GetVecbAt(y, x)[z])/255.0-float32(c.ModelParams.Mean[z]))/float32(c.ModelParams.STD[z]),
					y, x, z)
				if err != nil {
					return nil, err
				}
			}
		}
	}
	err = imgTensors.T(2, 0, 1)
	if err != nil {
		return nil, err
	}
	newShape := []int{1}
	newShape = append(newShape, imgTensors.Shape()...)
	err = imgTensors.Reshape(newShape...)

	return imgTensors, nil

}

func (c *FaceAntiSpoofingClient) InferSingle(imgFar, imgMid, imgNear gocv.Mat) (float32, error) {
	var asScore float32

	far, err := c.preprocess(imgFar)
	if err != nil {
		return asScore, err
	}
	mid, err := c.preprocess(imgMid)
	if err != nil {
		return asScore, err
	}
	near, err := c.preprocess(imgNear)
	if err != nil {
		return asScore, err
	}

	inputTensors := [][]float32{
		far.Float32s(),
		mid.Float32s(),
		near.Float32s(),
	}

	modelInputs := make([]*triton_proto.ModelInferRequest_InferInputTensor, 3)
	modelRequest := &triton_proto.ModelInferRequest{
		ModelName: c.ModelParams.ModelName,
	}

	for idx, inputCfg := range c.ModelConfig.Config.Input {
		modelInput := &triton_proto.ModelInferRequest_InferInputTensor{
			Name:     inputCfg.Name,
			Datatype: inputCfg.DataType.String()[5:],
			Shape:    inputCfg.Dims,
			Contents: &triton_proto.InferTensorContents{
				Fp32Contents: inputTensors[idx],
			},
		}
		modelInputs[idx] = modelInput
	}
	modelRequest.Inputs = modelInputs

	inferResp, err := c.tritonClient.ModelGRPCInfer(c.ModelParams.Timeout, modelRequest)
	if err != nil {
		return asScore, err
	}

	for oIdx, output := range inferResp.GetOutputs() {
		outputShape := make([]int, 0, len(output.Shape))
		for _, shp := range output.Shape {
			outputShape = append(outputShape, int(shp))
		}
		content := utils.BytesToT32[float32](inferResp.RawOutputContents[oIdx])
		for idx, score := range content {
			if idx == 1 {
				asScore = score
			}
		}
	}

	return asScore, nil
}
