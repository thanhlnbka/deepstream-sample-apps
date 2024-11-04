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

# set from docker_build_bevfusion_image.sh
ARG BASE_IMAGE=nvcr.io/nvidia/deepstream:7.0.0-triton-multiarch

FROM ${BASE_IMAGE} as base_debug

ARG WORKSPACE=/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-3d-lidar-sensor-fusion

WORKDIR ${WORKSPACE}

# install dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt update && \
    apt install -y --no-install-recommends cmake

RUN pip3 install gdown onnx nvidia-pytriton

# setup bevfusion environment
RUN mkdir -p ${WORKSPACE}/bevfusion
ENV BEVFUSION_MODEL=bevfusion/model_root/model/resnet50int8
ENV BEVFUSION_PRECISION=int8
ENV PYTHONPATH=$PYTHONPATH:${WORKSPACE}/python:{WORKSPACE}/bevfusion/Lidar_AI_Solution/CUDA-BEVFusion/build:

ARG COMMIT_ID=master
# clone source code
RUN cd ${WORKSPACE}/bevfusion && \
    git clone --recursive --depth 1 https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution.git \
    -b ${COMMIT_ID} --single-branch Lidar_AI_Solution || exit 1

# check bevfusion specific patch
ENV WORKSPACE=${WORKSPACE}
# this is not required after github fix
RUN cd ${WORKSPACE}/bevfusion && \
    if [ -e local/CUDA-BEVFusion_patch_for_x86_cuda12-2.tgz ]; then \
        tar -pxvf local/CUDA-BEVFusion_patch_for_x86_cuda12-2.tgz -C Lidar_AI_Solution; \
    fi && \
    cd ${WORKSPACE}/bevfusion/Lidar_AI_Solution/CUDA-BEVFusion && \
    sed -i '/^set(CMAKE_CXX_FLAGS_RELEASE/d' CMakeLists.txt && \
    sed -i '/^set(CUDA_NVCC_FLAGS_RELEASE/d' CMakeLists.txt && \
    cd ${WORKSPACE}/bevfusion/Lidar_AI_Solution && \
    if [ -e libraries/3DSparseConvolution/libspconv/lib/x86_64/libspconv.so_x86_cuda12.2 ]; then \
        mv libraries/3DSparseConvolution/libspconv/lib/x86_64/libspconv.so_x86_cuda12.2 libraries/3DSparseConvolution/libspconv/lib/x86_64/libspconv.so; \
    fi

# set nvcc build flags for different gpu platforms
ARG CUDA_NVCC_FLAGS="-gencode arch=compute_90,code=compute_90 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_70,code=sm_70"

ARG CMAKE_CXX_FLAGS_RELEASE="-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB"
ARG CUDA_NVCC_FLAGS_RELEASE="-Werror=all-warnings -Xcompiler -std=c++17,-Wextra,-Wall,-Wno-deprecated-declarations,-O3 -DENABLE_TEXT_BACKEND_STB"

ARG CUDASM

# set protobuf build along with deepstream-triton release
ARG CMAKE_PROTOBUF_ARGS="-DProtobuf_INCLUDE_DIR=/opt/tritonclient/include \
    -DProtobuf_LIBRARY=/opt/tritonclient/lib/libprotobuf.a \
    -DProtobuf_PROTOC_EXECUTABLE=/opt/proto/bin/protoc \
    -DProtobuf_PROTOC_LIBRARY=/opt/tritonclient/lib/libprotoc.a"

# set protobuf build along with Ubuntu apt-install version
RUN apt update && DEBIAN_FRONTEND=noninteractive apt update && \
    apt install -y --no-install-recommends libprotobuf-dev protobuf-compiler
ARG CMAKE_PROTOBUF_ARGS="-DProtobuf_PROTOC_EXECUTABLE=/usr/bin/protoc"

# build github cuda-bevfusion python-bindings from source
RUN export USE_Python=ON && \
    export CUDASM=${CUDASM} && \
    export CUDA_Lib=/usr/local/cuda/lib64 && \
    export CUDA_Inc=/usr/local/cuda/include && \
    export CUDA_Bin=/usr/local/cuda/bin && \
    export Python_Inc=`python3 -c "import sysconfig;print(sysconfig.get_path('include'))"`  && \
    export Python_Lib=`python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"`  && \
    export Python_Soname=`python3 -c "import sysconfig;import re;print(re.sub('.a', '.so', sysconfig.get_config_var('LIBRARY')))"`  && \
    export PATH=$CUDA_Bin:$PATH && \
    mkdir -p ${WORKSPACE}/bevfusion/Lidar_AI_Solution/CUDA-BEVFusion/build && \
    cd ${WORKSPACE}/bevfusion/Lidar_AI_Solution/CUDA-BEVFusion/build && \
    cmake -DCUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS}" \
    -DCMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE}" \
    -DCUDA_NVCC_FLAGS_RELEASE="${CUDA_NVCC_FLAGS_RELEASE}" \
    ${CMAKE_PROTOBUF_ARGS} \
    .. && make  VERBOSE=1 -j || exit 1

# install deepstream dependencies
RUN cd /opt/nvidia/deepstream/deepstream && ./install.sh && \
    (if [[ -f ./user_additional_install.sh ]]; then bash ./user_additional_install.sh; fi)

WORKDIR ${WORKSPACE}

# It is useful to setup nuscene dataset for ds3d sensor fusion
RUN pip3 install nuscenes-devkit