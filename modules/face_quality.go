package modules

import (
	"github.com/okieraised/go-ekyc-pipeline/config"
	"github.com/okieraised/go-ekyc-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/okieraised/go-triton-client/triton_proto"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
)

type FaceQualityClient struct {
	tritonClient *gotritonclient.TritonGRPCClient
	ModelParams  *config.FaceQualityParams
	ModelConfig  *triton_proto.ModelConfigResponse
}

func NewFaceQualityClient(triton *gotritonclient.TritonGRPCClient, cfg *config.FaceQualityParams) (*FaceQualityClient, error) {

	inferenceConfig, err := triton.GetModelConfiguration(cfg.Timeout, cfg.ModelName, "")
	if err != nil {
		return nil, err
	}

	return &FaceQualityClient{
		tritonClient: triton,
		ModelParams:  cfg,
		ModelConfig:  inferenceConfig,
	}, nil
}

func (c *FaceQualityClient) preprocess(rawInputTensors []gocv.Mat) ([]*tensor.Dense, []config.Size, error) {
	outputs := make([]*tensor.Dense, 0)
	sizes := make([]config.Size, 0)

	for _, input := range rawInputTensors {
		imgH, imgW := input.Size()[0], input.Size()[1]
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
					err := imgTensors.SetAt((float32(input.GetVecbAt(y, x)[z])-float32(c.ModelParams.Mean[z]))*float32(c.ModelParams.Scale[z]), y, x, z)
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
		outputs = append(outputs, imgTensors)
	}
	preprocessed, err := outputs[0].Stack(0, outputs[1:]...)
	if err != nil {
		return nil, nil, err
	}

	return []*tensor.Dense{preprocessed}, sizes, nil
}

func (c *FaceQualityClient) preprocessBatch(rawInputTensors [][]gocv.Mat) ([]*tensor.Dense, []config.Size, error) {
	inputs := rawInputTensors[0]
	outputs := make([]*tensor.Dense, 0)
	sizes := make([]config.Size, 0)

	for _, input := range inputs {
		imgH, imgW := input.Size()[0], input.Size()[1]
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
					err := imgTensors.SetAt((float32(input.GetVecbAt(y, x)[z])-float32(c.ModelParams.Mean[z]))*float32(c.ModelParams.Scale[z]), y, x, z)
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
		outputs = append(outputs, imgTensors)
	}
	preprocessed, err := outputs[0].Stack(0, outputs[1:]...)
	if err != nil {
		return nil, nil, err
	}

	return []*tensor.Dense{preprocessed}, sizes, nil
}

func (c *FaceQualityClient) postprocessBatch(rawOutputs []*tensor.Dense, sizes []config.Size) ([]*tensor.Dense, error) {
	output := make([]*tensor.Dense, 0)
	for _, o := range rawOutputs {
		quality, err := o.Slice(nil, tensor.S(2))
		if err != nil {
			return output, err
		}
		result := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(quality.(*tensor.Dense).Shape()...))
		err = tensor.Copy(result, quality)
		if err != nil {
			return output, err
		}

		output = append(output, result)
	}

	return output, nil
}

func (c *FaceQualityClient) postprocess(rawOutputs []*tensor.Dense, sizes []config.Size) ([]*tensor.Dense, error) {
	output := make([]*tensor.Dense, 0)
	for _, o := range rawOutputs {
		quality, err := o.Slice(nil, tensor.S(2))
		if err != nil {
			return output, err
		}
		result := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(quality.(*tensor.Dense).Shape()...))
		err = tensor.Copy(result, quality)
		if err != nil {
			return output, err
		}

		output = append(output, result)
	}

	return output, nil
}

func (c *FaceQualityClient) InferSingle(rawInputTensors []gocv.Mat) ([]*tensor.Dense, error) {
	inputTensors, sizes, err := c.preprocess(rawInputTensors)
	if err != nil {
		return nil, err
	}
	outputs := make([]*tensor.Dense, 0)

	modelRequest := &triton_proto.ModelInferRequest{
		ModelName: c.ModelParams.ModelName,
	}

	modelInputs := make([]*triton_proto.ModelInferRequest_InferInputTensor, 0)
	for _, inputCfg := range c.ModelConfig.Config.Input {
		tShapes := inputTensors[0].Shape()
		inputShapes := make([]int64, 0)
		for _, s := range tShapes {
			inputShapes = append(inputShapes, int64(s))
		}

		modelInput := &triton_proto.ModelInferRequest_InferInputTensor{
			Name:     inputCfg.Name,
			Datatype: inputCfg.DataType.String()[5:],
			Shape:    inputShapes,
			Contents: &triton_proto.InferTensorContents{
				Fp32Contents: inputTensors[0].Float32s(),
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
		outputs = append(outputs, tensors)
	}
	qualityOutputs, err := c.postprocess(outputs, sizes)
	if err != nil {
		return nil, err
	}
	return qualityOutputs, nil
}

func (c *FaceQualityClient) InferBatch(rawInputTensors [][]gocv.Mat) ([]*tensor.Dense, error) {
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
			tShapes := inputTensors[idx].Shape()
			inputShapes := make([]int64, 0)
			for _, s := range tShapes {
				inputShapes = append(inputShapes, int64(s))
			}

			modelInput := &triton_proto.ModelInferRequest_InferInputTensor{
				Name:     inputCfg.Name,
				Datatype: inputCfg.DataType.String()[5:],
				Shape:    inputShapes,
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

	concatenatedOutputs := make([]*tensor.Dense, len(outputs))
	for i, output := range outputs {
		concatenated, err := output[0].Concat(0, output[1:]...)
		if err != nil {
			return nil, err
		}
		concatenatedOutputs[i] = concatenated
	}

	qualityOutputs, err := c.postprocessBatch(concatenatedOutputs, sizes)
	if err != nil {
		return nil, err
	}
	return qualityOutputs, nil
}
