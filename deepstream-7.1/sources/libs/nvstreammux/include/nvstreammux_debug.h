/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __NVSTREAMMUX_DEBUG_H__
#define __NVSTREAMMUX_DEBUG_H__

class INvStreammuxDebug
{
    public:
    virtual void DebugPrint(const char* format, ... ) = 0;
};

#endif /**< __NVSTREAMMUX_DEBUG_H__ */
