/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include "buffer_probe.hpp"

namespace deepstream {

class MsgMetaGenerator : public BufferProbe::IBatchMetadataOperator {
public:
  MsgMetaGenerator() {}
  virtual ~MsgMetaGenerator() {}

  virtual probeReturn handleData(BufferProbe& probe, BatchMetadata& data) {
    int frame_interval = 0;
    probe.getProperty("frame-interval", frame_interval);
    FrameMetadata::Iterator frame_itr;
    for (data.initiateIterator(frame_itr); !frame_itr->done(); frame_itr->next())
    {
      ObjectMetadata::Iterator obj_itr;
      for ((*frame_itr)->initiateIterator(obj_itr); !obj_itr->done(); obj_itr->next()) {
        if (frames_ % frame_interval == 0)
        {
          EventMessageUserMetadata event_user_meta;
          if (data.acquire(event_user_meta)) {
            event_user_meta.generate(**obj_itr, **frame_itr);
            (*frame_itr)->append(event_user_meta);
          }
        }
      }
      frames_++;
    }

    return probeReturn::Probe_Ok;
  }

 protected:
  int frames_ = 0;
};

}
