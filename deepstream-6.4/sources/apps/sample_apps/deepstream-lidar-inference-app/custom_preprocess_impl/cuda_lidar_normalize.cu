/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <assert.h>
#include <cuda_runtime.h>

extern "C" void ds3dCustomCudaLidarNormalize(
    float* in, float* out, int points, float offset, float scale, cudaStream_t stream);

static __global__ void
cuda_lidar_normalize_c4(float4* in, float4* out, int points, float offset, float scale)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= points)
        return;
    float4 p = in[idx];
    p.w = (p.w - offset) * scale;
    out[idx] = p;
}

extern "C" void
ds3dCustomCudaLidarNormalize(
    float* in, float* out, int points, float offset, float scale, cudaStream_t stream)
{
    size_t threads = 64;
    size_t blocks = (points + threads - 1) / threads;
    cuda_lidar_normalize_c4<<<blocks, threads, 0, stream>>>(
        (float4*)in, (float4*)out, points, offset, scale);
}