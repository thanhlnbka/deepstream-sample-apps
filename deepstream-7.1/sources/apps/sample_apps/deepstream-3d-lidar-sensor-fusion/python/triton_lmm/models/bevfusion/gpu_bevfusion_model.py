################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

import logging
import os
from typing import Optional

import libpybev
import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

from triton_lmm.common.model import INPUT_IMAGE, INPUT_LIDAR, OUTPUT_3D_BBOX, IModel

# from .tensor_load import tensor_load

logger = logging.getLogger(__name__)

import torch
from torch.utils.dlpack import to_dlpack

class BevFusionModel(IModel):
    """ BEV Fusion Model

    Model setup:
        Follow https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-BEVFusion
        to build and setup the model
        export BEVFUSION_MODEL=bevfusion/model/resnet50int8
        export BEVFUSION_PRECISION=int8
    """

    def __init__(
            self, model_name: str, model_dir: Optional[str] = None, precision: Optional[str] = None,
            log_level: int = logging.INFO):
        super().__init__(model_name)
        logger.setLevel(log_level)
        if model_dir:
            self._model_dir = model_dir
        else:
            self._model_dir = os.environ.get("BEVFUSION_MODEL", "model")
        if precision:
            self._precision = precision
        else:
            self._precision = os.environ.get("BEVFUSION_PRECISION", "int8")
        self._image_num = 6
        self._core = libpybev.load_bevfusion(
            f"{self._model_dir}/build/camera.backbone.plan", f"{self._model_dir}/build/camera.vtransform.plan",
            f"{self._model_dir}/lidar.backbone.xyz.onnx", f"{self._model_dir}/build/fuser.plan",
            f"{self._model_dir}/build/head.bbox.plan", self._precision)
        self._image_names = [INPUT_IMAGE + "_" + str(i) for i in range(self._image_num)]
        assert self._core

        self.results = []
        self.frame_no = 0

    def start(self, **kwargs):
        # import ipdb
        # ipdb.set_trace()
        camera_intrinsics = np.zeros(shape=(1, 6, 4, 4), dtype=np.float32)
        camera2lidar = np.zeros(shape=(1, 6, 4, 4), dtype=np.float32)
        lidar2image = np.zeros(shape=(1, 6, 4, 4), dtype=np.float32)
        img_aug_matrix = np.zeros(shape=(1, 6, 4, 4), dtype=np.float32)
        camera_intrinsics = kwargs.get("camera_intrinsics", camera_intrinsics)
        camera2lidar = kwargs.get("camera2lidar", camera2lidar)
        lidar2image = kwargs.get("lidar2image", lidar2image)
        img_aug_matrix = kwargs.get("img_aug_matrix", img_aug_matrix)


        logger.info(
            f"camera_intrinsics shape: {camera_intrinsics.shape}, {camera_intrinsics.dtype}, data\n: {camera_intrinsics}."
        )
        logger.info(f"camera2lidar shape: {camera2lidar.shape} {camera2lidar.dtype}, data\n: {camera2lidar}.")
        logger.info(f"lidar2image shape: {lidar2image.shape} {lidar2image.dtype}, data\n: {lidar2image}.")
        logger.info(f"img_aug_matrix shape: {img_aug_matrix.shape} {img_aug_matrix.dtype}, data\n: {img_aug_matrix}.")

        assert self._core
        self._core.update(camera2lidar, camera_intrinsics, lidar2image, img_aug_matrix)
        if logger.level <= logging.INFO:
            self._core.print()

    def _debug_inputs(self, images: list, lidar: np.array):
        from PIL import Image
        for i, data in enumerate(images):
            img = Image.fromarray(data[0], 'RGB')
            img.save(f"input_image_{i}.jpg")

    def stop(self):
        if self._core:
            del self._core
        self._core = None

    def _infer(self, inputs):
        # import ipdb
        # ipdb.set_trace()

        # input image shape (1, 900, 1600, 4)
        images = [inputs[name][:, :, :, :3] for name in self._image_names]
        # input lidar shape (N, 4)
        lidar = inputs[INPUT_LIDAR].cuda()
        # self._debug_inputs(images, lidar)

        images = torch.cat(images, dim=0).cuda()
        images = images.unsqueeze(0)

        num_points = lidar.shape[0]
        lidar = torch.cat([lidar.half(), torch.zeros(num_points, 1, dtype=torch.float16)], dim=1).cuda()

        logger.debug(f"keys: {inputs.keys()}")
        logger.debug(f"images.shape: {images.shape}, datatype: {images.dtype}")
        logger.debug(f"lidar.shape: {lidar.shape}, datatype: {lidar.dtype}")

        # forward images shape (1, batch, 900, 1600, 3)
        # forward lidar shape (N, 5)

        # Arguments in order: images, lidar, do normalization (true/false), is dlpack (true/false)
        # normalization must be false if dlpack is true
        boxes = self._core.forward(to_dlpack(images), to_dlpack(lidar), False, True)

        if logger.level <= logging.DEBUG:
            self._core.print()

        np.set_printoptions(3, suppress=True, linewidth=300)

        boxes = np.expand_dims(boxes, axis=0)
        logger.debug(f"bboxes.shape: {boxes.shape}, datatype: {boxes.dtype}")
        return {OUTPUT_3D_BBOX: boxes}

    def bind_model(self, triton: Triton):
        logger.info(f"binding model: {self.name} to triton")
        inputs = [Tensor(name=img_name, dtype=np.uint8, shape=(1, 900, 1600, 4)) for img_name in self._image_names]
        inputs.append(Tensor(name=INPUT_LIDAR, dtype=np.float32, shape=(242180, 4)))

        triton.bind(
            model_name=self.name,
            infer_func=self._infer,
            inputs=inputs,
            outputs=[
                Tensor(name=OUTPUT_3D_BBOX, dtype=np.float32, shape=(1, -1, -1)),    # 200, 11
            ],
            config=ModelConfig(batching=False))