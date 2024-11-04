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

#ifndef __GST_NVDSA2VTEMPLATE_H__
#define __GST_NVDSA2VTEMPLATE_H__

#include <vector>
#include "gstaudio2video.h"
#include "nvdscustomlib_factory.hpp"
#include "nvdscustomlib_interface.hpp"

G_BEGIN_DECLS

/* Package and library details required for plugin_init */
#define PACKAGE "nvdsA2Vtemplate"
#define VERSION "1.0"
#define LICENSE "Proprietary"
#define DESCRIPTION "Nvidia nvdsA2Vtemplate plugin for handling audio in to video out related use cases \0"
#define BINARY_PACKAGE "NVIDIA DeepStream A2Vtemplate Plugin"
#define URL "http://nvidia.com/"

/* Standard boilerplate stuff */
#define GST_TYPE_NVDSA2VTEMPLATE            (gst_nvdsA2Vtemplate_get_type())
#define GST_NVDSA2VTEMPLATE(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSA2VTEMPLATE,GstNvDsA2Vtemplate))
#define GST_NVDSA2VTEMPLATE_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSA2VTEMPLATE,GstNvDsA2VtemplateClass))
#define GST_NVDSA2VTEMPLATE_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_NVDSA2VTEMPLATE, GstNvDsA2VtemplateClass))
#define GST_IS_NVDSA2VTEMPLATE(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSA2VTEMPLATE))
#define GST_IS_NVDSA2VTEMPLATE_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSA2VTEMPLATE))
#define GST_NVDSA2VTEMPLATE_CAST(obj)       ((GstNvDsA2Vtemplate *)(obj))

typedef struct _GstNvDsA2Vtemplate GstNvDsA2Vtemplate;
typedef struct _GstNvDsA2VtemplateClass GstNvDsA2VtemplateClass;

struct _GstNvDsA2Vtemplate
{
  GstAudio2Video parent;
  GstPadEventFunction parent_sink_event_fn;

  /** Custom Library Factory and Interface */
  DSCustomLibrary_Factory *algo_factory;
  IDSCustomLibrary *algo_ctx;

  /** Custom Library Name and output caps string */
  gchar* custom_lib_name;

  /* Store custom lib property values */
  std::vector<Property> *vecProp;
  gchar *custom_prop_string;

  gboolean gpu_on;

  /* < private > */
  GstBufferPool *pool;
};

struct _GstNvDsA2VtemplateClass
{
  GstAudio2VideoClass parent_class;
  GstStateChangeReturn (* parent_change_state_fn) (GstElement * element,
    GstStateChange transition);
};

GType gst_nvdsA2Vtemplate_get_type (void);

G_END_DECLS
#endif /* __GST_NVDSA2VTEMPLATE_H__ */
