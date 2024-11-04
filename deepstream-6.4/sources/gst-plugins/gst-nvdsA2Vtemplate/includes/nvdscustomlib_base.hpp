/**
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __NVDSCUSTOMLIB_BASE_HPP__
#define __NVDSCUSTOMLIB_BASE_HPP__

#include <gst/base/gstbasetransform.h>

#include "nvdscustomlib_interface.hpp"

/* Buffer Pool Configuration Parameters */
struct BufferPoolConfig {
  gint cuda_mem_type;
  guint gpu_id;
  guint max_buffers;
  gint batch_size;
};

class DSCustomLibraryBase : public IDSCustomLibrary
{
public:
    explicit DSCustomLibraryBase(GstElement* bscope = nullptr);

    /* Set Init Parameters */
    virtual bool SetInitParams(DSCustom_CreateParams *params);

    virtual ~DSCustomLibraryBase();

    /* Set Custom Properties  of the library */
    virtual bool SetProperty(Property &prop) = 0;

    virtual bool HandleEvent (GstEvent *event) = 0;
    // TODO: Add getProperty as well

    virtual char* QueryProperties () = 0;

    virtual BufferResult ProcessBuffer(GstAudio2Video * base, GstBuffer * audio, GstVideoFrame * video) = 0;

public:
    /* Gstreamer dsexaple2 plugin's base class reference */
    GstElement *m_element;

    /** GPU ID on which we expect to execute the algorithm */
};


DSCustomLibraryBase::DSCustomLibraryBase(GstElement* bscope) : m_element(bscope) {
}

bool DSCustomLibraryBase::SetInitParams(DSCustom_CreateParams *params) {
    m_element = params->m_element;

    return true;
}

DSCustomLibraryBase::~DSCustomLibraryBase() {
}

#endif
