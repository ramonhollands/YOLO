from pathlib import Path
from typing import Dict, List, Optional

from yolo.config.config import Config
from yolo.model.yolo import YOLO
from yolo.utils.logger import logger


class ModelExporter:
    def __init__(self, cfg: Config, model: YOLO, format: str, model_path: Optional[str] = None):
        self.model = model
        self.cfg = cfg
        self.class_num = cfg.dataset.class_num
        self.format = format
        if cfg.weight == True:
            cfg.weight = Path("weights") / f"{cfg.model.name}.pt"

        if model_path:
            self.model_path = model_path
        else:
            extention = self.format
            if self.format == "coreml":
                extention = "mlpackage"

            self.model_path = f"{Path(self.cfg.weight).stem}.{extention}"

        self.output_names: List[str] = [
            "1_class_scores_small",
            "2_box_features_small",
            "3_bbox_deltas_small",
            "4_class_scores_medium",
            "5_box_features_medium",
            "6_bbox_deltas_medium",
            "7_class_scores_large",
            "8_box_features_large",
            "9_bbox_deltas_large",
        ]

    def export_onnx(self, dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None, model_path: Optional[str] = None):
        logger.info(f":package: Exporting model to onnx format")
        import torch

        dummy_input = torch.ones((1, 3, *self.cfg.image_size))

        if model_path:
            onnx_model_path = model_path
        else:
            onnx_model_path = self.model_path

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_model_path,
            input_names=["input"],
            output_names=self.output_names,
            dynamic_axes=dynamic_axes,
        )

        logger.info(f":inbox_tray: ONNX model saved to {onnx_model_path}")

        return onnx_model_path

    def export_tflite(self):
        logger.info(f":package: Exporting model to tflite format")

        import torch

        self.model.eval()
        example_inputs = (torch.rand(1, 3, *self.cfg.image_size),)

        import ai_edge_torch

        edge_model = ai_edge_torch.convert(self.model, example_inputs)
        edge_model.export(self.model_path)

        logger.info(f":white_check_mark: Model exported to tflite format")

    def export_coreml(self):
        logger.info(f":package: Exporting model to coreml format")

        import torch

        self.model.eval()
        example_inputs = (torch.rand(1, 3, *self.cfg.image_size),)
        # model = IOSDetectModel(self.model, example_inputs[0])
        exported_program = torch.export.export(self.model, example_inputs)

        import logging

        import coremltools as ct

        # Convert to Core ML program using the Unified Conversion API.
        logging.getLogger("coremltools").disabled = True

        self.output_names: List[str] = ["preds_cls", "preds_box"]
        # self.output_names: List[str] = [f"output_{i}" for i in range(6)]
        ct_model = ct.convert(
            exported_program,
            inputs=[ct.ImageType("x", shape=example_inputs[0].shape, scale=1/255., bias=[0,0,0])],
            outputs=[ct.TensorType(name=name) for name in self.output_names], convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
        )

        # int8 quantization
        # import coremltools.optimize.coreml as cto
        # op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=8, weight_threshold=512)
        # config = cto.OptimizationConfig(global_config=op_config)
        # ct_model = cto.palettize_weights(ct_model, config=config)
        
        ct_model_32 = ct.convert(
            exported_program,
            inputs=[ct.ImageType("x", shape=example_inputs[0].shape, scale=1/255., bias=[0,0,0])],
            outputs=[ct.TensorType(name=name) for name in self.output_names], convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
        )
        
        # ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, 16, 'linear')
        fp_16_model_path = self.model_path.replace(".mlpackage", "_fp16.mlpackage")
        fp_32_model_path = self.model_path.replace(".mlpackage", "_fp32.mlpackage")
        ct_model.save(fp_16_model_path)
        ct_model_32.save(fp_32_model_path)
        logger.info(f":white_check_mark: Model exported to coreml format {self.model_path}")
        
        # run both model and check the output
        import numpy as np
        import torch
        
        from coremltools.models import MLModel
        # Load saved models
        mlmodel_16 = MLModel(fp_16_model_path)
        mlmodel_32 = MLModel(fp_32_model_path)

        # Prepare input in Core ML format
        torch_input = example_inputs[0]
        np_input = torch_input.squeeze(0).permute(1, 2, 0).numpy() * 255  # Convert to HWC and scale to [0,255]
        np_input = np_input.astype(np.uint8)
        
        from PIL import Image
        pil_image = Image.fromarray(np_input)

        coreml_input = {"x": pil_image}

        # Run both models
        out_16 = mlmodel_16.predict(coreml_input)
        out_32 = mlmodel_32.predict(coreml_input)
        
        print("Comparing model_float16 and model_float32 outputs:")
        for name in self.output_names:
            if name == "preds_anc":
                continue
            output16 = out_16[name]
            output32 = out_32[name]
            print("Maximum value in output16:", name, output16.max())
            print("Maximum value in output32:", name, output32.max())
            print("Minimum value in output16:", name, output16.min())
            print("Minimum value in output32:", name, output32.min())
            diff = np.abs(output16 - output32)
            print(f"{name}: mean abs diff = {diff.mean():.6f}, max diff = {diff.max():.6f}")

        


# import torch
# class IOSDetectModel(torch.nn.Module):
#     """Wrap an Ultralytics YOLO model for Apple iOS CoreML export."""

#     def __init__(self, model, im):
#         """Initialize the IOSDetectModel class with a YOLO model and example image."""
#         super().__init__()
#         _, _, h, w = im.shape  # batch, channel, height, width
#         self.model = model
#         self.nc = 80  # number of classes
#         if w == h:
#             self.normalize = 1.0 / w  # scalar
#         else:
#             self.normalize = torch.tensor([1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h])  # broadcast (slower, smaller)

#     def forward(self, x):
#         """Normalize predictions of object detection model with input size-dependent factors."""
#         xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
#         return cls, xywh * self.normalize  # confidence (3780, 80), coordinates (3780, 4)