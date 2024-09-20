package modules

import (
	"github.com/okieraised/go-ekyc-pipeline/config"
	"github.com/okieraised/go-ekyc-pipeline/utils"
	gotritonclient "github.com/okieraised/go-triton-client"
	"github.com/okieraised/go-triton-client/triton_proto"
	"gocv.io/x/gocv"
	"gorgonia.org/tensor"
	"image"
	"slices"
)

type FaceDetectionClient struct {
	tritonClient *gotritonclient.TritonGRPCClient
	ModelConfig  *triton_proto.ModelConfigResponse
	ModelParams  *config.FaceDetectionParams
}

func NewFaceDetectionClient(triton *gotritonclient.TritonGRPCClient, cfg *config.FaceDetectionParams) (*FaceDetectionClient, error) {

	inferenceConfig, err := triton.GetModelConfiguration(cfg.Timeout, cfg.ModelName, "")
	if err != nil {
		return nil, err
	}

	return &FaceDetectionClient{
		tritonClient: triton,
		ModelParams:  cfg,
		ModelConfig:  inferenceConfig,
	}, nil
}

func (c *FaceDetectionClient) preprocess(rawInputTensors []gocv.Mat) ([]*tensor.Dense, []config.Size, error) {
	outputs := make([]*tensor.Dense, 0)
	input := rawInputTensors[0]
	imgH, imgW := input.Size()[0], input.Size()[1]
	imgRatio := float64(imgW) / float64(imgH)
	sizes := make([]config.Size, 0)
	sizes = append(sizes, config.Size{
		Width:  imgW,
		Height: imgH,
	})

	modelRatio := float64(c.ModelConfig.Config.Input[0].Dims[2]) / float64(c.ModelConfig.Config.Input[0].Dims[1])

	var newWidth, newHeight int64
	if imgRatio > modelRatio {
		newWidth = c.ModelConfig.Config.Input[0].Dims[2]
		newHeight = int64(float64(newWidth) / imgRatio)
	} else {
		newHeight = c.ModelConfig.Config.Input[0].Dims[1]
		newWidth = int64(float64(newHeight) * imgRatio)
	}

	resizedImg := gocv.NewMat()
	defer resizedImg.Close()
	gocv.Resize(input, &resizedImg, image.Point{X: int(newWidth), Y: int(newHeight)}, 0.0, 0.0, gocv.InterpolationLinear)

	scaledImg := gocv.NewMatWithSizesWithScalar(
		[]int{
			int(c.ModelConfig.Config.Input[0].Dims[1]),
			int(c.ModelConfig.Config.Input[0].Dims[2]),
		},
		gocv.MatTypeCV8UC3,
		gocv.NewScalar(0, 0, 0, 0),
	)
	defer scaledImg.Close()

	roi := scaledImg.Region(image.Rect(0, 0, int(newWidth), int(newHeight)))

	gocv.Resize(resizedImg, &roi, image.Point{X: roi.Size()[1], Y: roi.Size()[0]}, 0, 0, gocv.InterpolationLinear)
	imgTensors := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(
			int(c.ModelConfig.Config.Input[0].Dims[0]),
			int(c.ModelConfig.Config.Input[0].Dims[1]),
			int(c.ModelConfig.Config.Input[0].Dims[2]),
		),
	)

	for z := range 3 {
		for y := range int(c.ModelConfig.Config.Input[0].Dims[1]) {
			for x := range int(c.ModelConfig.Config.Input[0].Dims[2]) {
				err := imgTensors.SetAt((float32(scaledImg.GetVecbAt(y, x)[z])-float32(c.ModelParams.Mean))*float32(c.ModelParams.Scale), z, y, x)
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
	return outputs, sizes, nil
}

func (c *FaceDetectionClient) preprocessBatch(rawInputTensors [][]gocv.Mat) ([]*tensor.Dense, []config.Size, error) {
	inputs := rawInputTensors[0]
	outputs := make([]*tensor.Dense, 0)
	sizes := make([]config.Size, 0)

	for _, input := range inputs {
		imgH, imgW := input.Size()[0], input.Size()[1]
		imgRatio := float64(imgW) / float64(imgH)
		sizes = append(sizes, config.Size{
			Width:  imgW,
			Height: imgH,
		})

		modelRatio := float64(c.ModelConfig.Config.Input[0].Dims[2]) / float64(c.ModelConfig.Config.Input[0].Dims[1])

		var newWidth, newHeight int64
		if imgRatio > modelRatio {
			newWidth = c.ModelConfig.Config.Input[0].Dims[2]
			newHeight = int64(float64(newWidth) / imgRatio)
		} else {
			newHeight = c.ModelConfig.Config.Input[0].Dims[1]
			newWidth = int64(float64(newHeight) * imgRatio)
		}

		resizedImg := gocv.NewMat()
		defer resizedImg.Close()
		gocv.Resize(input, &resizedImg, image.Point{X: int(newWidth), Y: int(newHeight)}, 0.0, 0.0, gocv.InterpolationLinear)

		scaledImg := gocv.NewMatWithSizesWithScalar(
			[]int{
				int(c.ModelConfig.Config.Input[0].Dims[1]),
				int(c.ModelConfig.Config.Input[0].Dims[2]),
			},
			gocv.MatTypeCV8UC3,
			gocv.NewScalar(0, 0, 0, 0),
		)

		roi := scaledImg.Region(image.Rect(0, 0, int(newWidth), int(newHeight)))

		gocv.Resize(resizedImg, &roi, image.Point{X: roi.Size()[1], Y: roi.Size()[0]}, 0, 0, gocv.InterpolationLinear)
		imgTensors := tensor.New(
			tensor.Of(tensor.Float32),
			tensor.WithShape(
				int(c.ModelConfig.Config.Input[0].Dims[0]),
				int(c.ModelConfig.Config.Input[0].Dims[1]),
				int(c.ModelConfig.Config.Input[0].Dims[2]),
			),
		)

		for z := range 3 {
			for y := range int(c.ModelConfig.Config.Input[0].Dims[1]) {
				for x := range int(c.ModelConfig.Config.Input[0].Dims[2]) {
					err := imgTensors.SetAt((float32(scaledImg.GetVecbAt(y, x)[z])-float32(c.ModelParams.Mean))*float32(c.ModelParams.Scale), z, y, x)
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
	return outputs, sizes, nil
}

func (c *FaceDetectionClient) postprocessBatch(rawOutputs []*tensor.Dense, sizes []config.Size) ([][]config.FaceDetectionOutput, error) {

	results := make([][]config.FaceDetectionOutput, 0)

	for b := range rawOutputs[0].Shape()[0] {
		res := make([]config.FaceDetectionOutput, 0)
		numDets, err := rawOutputs[0].Slice(tensor.S(b))
		if err != nil {
			return nil, err
		}
		boxes, err := rawOutputs[1].Slice(tensor.S(b))
		if err != nil {
			return nil, err
		}
		scores, err := rawOutputs[2].Slice(tensor.S(b))
		if err != nil {
			return nil, err
		}
		classes, err := rawOutputs[3].Slice(tensor.S(b))
		if err != nil {
			return nil, err
		}
		landmarks, err := rawOutputs[4].Slice(tensor.S(b))
		if err != nil {
			return nil, err
		}

		scale := slices.Max(c.ModelConfig.Config.Input[0].Dims)
		if len(sizes) > 0 {
			scale = int64(sizes[b].Max())
		}
		for i := range numDets.Size() {
			score, err := scores.Slice(tensor.S(i))
			if err != nil {
				return nil, err
			}
			classID, err := classes.Slice(tensor.S(i))
			if err != nil {
				return nil, err
			}
			box, err := boxes.Slice(tensor.S(i))
			if err != nil {
				return nil, err
			}
			scaledBox, err := box.Apply(func(x float32) float32 {
				return x * float32(scale)
			})
			if err != nil {
				return nil, err
			}

			landmark, err := landmarks.Slice(tensor.S(i))
			if err != nil {
				return nil, err
			}
			scaledLandmark, err := landmark.Apply(func(x float32) float32 {
				return x * float32(scale)
			})
			if err != nil {
				return nil, err
			}

			res = append(res, config.FaceDetectionOutput{
				Box:      scaledBox.(*tensor.Dense),
				Score:    score.(*tensor.Dense),
				ClassID:  classID.(*tensor.Dense),
				Landmark: scaledLandmark.(*tensor.Dense),
			})
		}
		results = append(results, res)

	}

	return results, nil
}

func (c *FaceDetectionClient) postprocess(rawOutputs []*tensor.Dense, sizes []config.Size) ([]config.FaceDetectionOutput, error) {

	results := make([]config.FaceDetectionOutput, 0)
	numDets, err := rawOutputs[0].Slice(tensor.S(0))
	if err != nil {
		return nil, err
	}
	boxes, err := rawOutputs[1].Slice(tensor.S(0))
	if err != nil {
		return nil, err
	}
	scores, err := rawOutputs[2].Slice(tensor.S(0))
	if err != nil {
		return nil, err
	}
	classes, err := rawOutputs[3].Slice(tensor.S(0))
	if err != nil {
		return nil, err
	}
	landmarks, err := rawOutputs[4].Slice(tensor.S(0))
	if err != nil {
		return nil, err
	}

	scale := slices.Max(c.ModelConfig.Config.Input[0].Dims)
	if len(sizes) > 0 {
		scale = int64(sizes[0].Max())
	}
	for i := range numDets.Size() {
		score, err := scores.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}
		classID, err := classes.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}
		box, err := boxes.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}
		scaledBox, err := box.Apply(func(x float32) float32 {
			return x * float32(scale)
		})
		if err != nil {
			return nil, err
		}

		landmark, err := landmarks.Slice(tensor.S(i))
		if err != nil {
			return nil, err
		}
		scaledLandmark, err := landmark.Apply(func(x float32) float32 {
			return x * float32(scale)
		})
		if err != nil {
			return nil, err
		}

		results = append(results, config.FaceDetectionOutput{
			Box:      scaledBox.(*tensor.Dense),
			Score:    score.(*tensor.Dense),
			ClassID:  classID.(*tensor.Dense),
			Landmark: scaledLandmark.(*tensor.Dense),
		})
	}
	return results, nil
}

func (c *FaceDetectionClient) InferBatch(rawInputTensors [][]gocv.Mat) ([][]config.FaceDetectionOutput, error) {

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
			switch output.Datatype {
			case "FP32":
				content := utils.BytesToT32[float32](inferResp.RawOutputContents[oIdx])
				tensors = tensor.New(
					tensor.Of(tensor.Float32),
					tensor.WithShape(outputShape...),
					tensor.WithBacking(content),
				)

			case "INT32":
				content := utils.BytesToT32[int32](inferResp.RawOutputContents[oIdx])
				tensors = tensor.New(
					tensor.Of(tensor.Int),
					tensor.WithShape(outputShape...),
					tensor.WithBacking(content),
				)

			}
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
	detOutputs, err := c.postprocessBatch(concatenatedOutputs, sizes)
	if err != nil {
		return nil, err
	}
	return detOutputs, nil

}

func (c *FaceDetectionClient) InferSingle(rawInputTensors []gocv.Mat) ([]config.FaceDetectionOutput, error) {
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
		modelInput := &triton_proto.ModelInferRequest_InferInputTensor{
			Name:     inputCfg.Name,
			Datatype: inputCfg.DataType.String()[5:],
			Shape:    []int64{1, inputCfg.Dims[0], inputCfg.Dims[1], inputCfg.Dims[2]},
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
		switch output.Datatype {
		case "FP32":
			content := utils.BytesToT32[float32](inferResp.RawOutputContents[oIdx])
			tensors = tensor.New(
				tensor.Of(tensor.Float32),
				tensor.WithShape(outputShape...),
				tensor.WithBacking(content),
			)

		case "INT32":
			content := utils.BytesToT32[int32](inferResp.RawOutputContents[oIdx])
			tensors = tensor.New(
				tensor.Of(tensor.Int),
				tensor.WithShape(outputShape...),
				tensor.WithBacking(content),
			)

		}
		outputs = append(outputs, tensors)
	}
	detOutputs, err := c.postprocess(outputs, sizes)
	if err != nil {
		return nil, err
	}
	return detOutputs, nil
}
