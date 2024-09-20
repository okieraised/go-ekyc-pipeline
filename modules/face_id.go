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

type FaceIDClient struct {
	tritonClient *gotritonclient.TritonGRPCClient
	ModelParams  *config.FaceIDParams
	ModelConfig  *triton_proto.ModelConfigResponse
}

func NewFaceIDClient(triton *gotritonclient.TritonGRPCClient, cfg *config.FaceIDParams) (*FaceIDClient, error) {

	inferenceConfig, err := triton.GetModelConfiguration(cfg.Timeout, cfg.ModelName, "")
	if err != nil {
		return nil, err
	}

	return &FaceIDClient{
		tritonClient: triton,
		ModelParams:  cfg,
		ModelConfig:  inferenceConfig,
	}, nil
}

func (c *FaceIDClient) preprocess(rawInputTensors []gocv.Mat) error {
	return nil
}

func (c *FaceIDClient) preprocessBatch(rawInputTensors [][]gocv.Mat) ([]*tensor.Dense, []config.Size, error) {
	inputs := rawInputTensors[0]
	outputs := make([]*tensor.Dense, 0)
	sizes := make([]config.Size, 0)

	for _, input := range inputs {
		resizedImg := gocv.NewMat()
		defer resizedImg.Close()
		gocv.Resize(
			input,
			&resizedImg,
			image.Point{
				X: c.ModelParams.ImgSize,
				Y: c.ModelParams.ImgSize,
			},
			0.0,
			0.0,
			gocv.InterpolationLinear,
		)
		imgH, imgW := resizedImg.Size()[0], resizedImg.Size()[1]
		sizes = append(sizes, config.Size{
			Width:  imgW,
			Height: imgH,
		})

		imgTensors := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(
				int(c.ModelConfig.Config.Input[0].Dims[1]),
				int(c.ModelConfig.Config.Input[0].Dims[2]),
				int(c.ModelConfig.Config.Input[0].Dims[0]),
			),
		)

		for z := range int(c.ModelConfig.Config.Input[0].Dims[0]) {
			for y := range int(c.ModelConfig.Config.Input[0].Dims[1]) {
				for x := range int(c.ModelConfig.Config.Input[0].Dims[2]) {
					err := imgTensors.SetAt((float32(resizedImg.GetVecbAt(y, x)[z])-float32(c.ModelParams.Mean))*float32(c.ModelParams.Scale), y, x, z)
					if err != nil {
						return nil, nil, err
					}
				}
			}
		}
		err := imgTensors.T(2, 0, 1)
		if err != nil {
			return nil, nil, err
		}
		newShape := []int{1}
		newShape = append(newShape, imgTensors.Shape()...)
		err = imgTensors.Reshape(newShape...)
		if err != nil {
			return nil, nil, err
		}
		outputs = append(outputs, imgTensors)
	}

	return outputs, sizes, nil
}

func (c *FaceIDClient) postprocessBatch(rawOutputs [][]*tensor.Dense, sizes []config.Size) ([]*tensor.Dense, error) {
	return rawOutputs[0], nil
}

func (c *FaceIDClient) InferBatch(rawInputTensors [][]gocv.Mat) ([]*tensor.Dense, error) {
	inputTensors, sizes, err := c.preprocessBatch(rawInputTensors)
	if err != nil {
		return nil, err
	}

	outputs := make([][]*tensor.Dense, len(c.ModelConfig.Config.Output))
	for idx := 0; idx < len(c.ModelConfig.Config.Output); idx++ {
		outputs[idx] = make([]*tensor.Dense, 0)
	}

	for idx := 0; idx < len(inputTensors); idx++ {
		modelRequest := &triton_proto.ModelInferRequest{
			ModelName: c.ModelParams.ModelName,
		}

		modelInputs := make([]*triton_proto.ModelInferRequest_InferInputTensor, 0)
		for _, inputCfg := range c.ModelConfig.Config.Input {
			modelInput := &triton_proto.ModelInferRequest_InferInputTensor{
				Name:     inputCfg.Name,
				Datatype: inputCfg.DataType.String()[5:],
				Shape:    []int64{1, inputCfg.Dims[0], inputCfg.Dims[1], inputCfg.Dims[2]},
				Contents: &triton_proto.InferTensorContents{
					Fp32Contents: inputTensors[idx].Float32s(),
				},
			}
			modelInputs = append(modelInputs, modelInput)
		}

		modelRequest.Inputs = modelInputs
		inferResp, err := c.tritonClient.ModelGRPCInfer(c.ModelParams.Timeout, modelRequest)
		if err != nil {
			return nil, err
		}

		for oIdx, output := range inferResp.GetOutputs() {
			outputShape := make([]int, 0, len(output.Shape))
			for _, shp := range output.Shape {
				outputShape = append(outputShape, int(shp))
			}
			var tensors *tensor.Dense
			content := utils.BytesToT32[float32](inferResp.RawOutputContents[oIdx])
			tensors = tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(outputShape...),
				tensor.WithBacking(content),
			)
			outputs[oIdx] = append(outputs[oIdx], tensors)
		}
	}

	faceIDOutputs, err := c.postprocessBatch(outputs, sizes)
	if err != nil {
		return nil, err
	}
	return faceIDOutputs, nil
}
