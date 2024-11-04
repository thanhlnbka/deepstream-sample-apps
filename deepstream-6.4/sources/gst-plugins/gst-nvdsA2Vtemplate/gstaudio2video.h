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

#ifndef __GST_AUDIO2VIDEO_H__
#define __GST_AUDIO2VIDEO_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

#include <gst/video/video.h>
#include <gst/audio/audio.h>
#include <gst/base/gstadapter.h>
#include <gst/pbutils/pbutils-prelude.h>

G_BEGIN_DECLS
#define GST_TYPE_AUDIO2VIDEO            (gst_audio2video_get_type())
#define GST_AUDIO2VIDEO(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_AUDIO2VIDEO,GstAudio2Video))
#define GST_AUDIO2VIDEO_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_AUDIO2VIDEO,GstAudio2VideoClass))
#define GST_AUDIO2VIDEO_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),GST_TYPE_AUDIO2VIDEO,GstAudio2VideoClass))
typedef struct _GstAudio2Video GstAudio2Video;
typedef struct _GstAudio2VideoClass GstAudio2VideoClass;
typedef struct _GstAudio2VideoPrivate GstAudio2VideoPrivate;

struct _GstAudio2Video
{
  GstElement parent;

  guint req_spf;                /* min samples per frame wanted by the subclass */

  /* video state */
  GstVideoInfo vinfo;

  /* audio state */
  GstAudioInfo ainfo;

  /*< private >*/
  GstAudio2VideoPrivate *priv;
};

struct _GstAudio2VideoClass
{
  /*< private >*/
  GstElementClass parent_class;

  /*< public >*/
  /* virtual function, called whenever the format changes */
  gboolean (*setup) (GstAudio2Video * scope);

  /* virtual function for rendering a frame */
  gboolean (*render) (GstAudio2Video * scope, GstBuffer * audio, GstVideoFrame * video);

  gboolean (*decide_allocation)   (GstAudio2Video * scope, GstQuery *query);
};

GST_PBUTILS_API
GType gst_audio2video_get_type (void);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (GstAudio2Video, gst_object_unref)

G_END_DECLS
#endif /* __GST_AUDIO2VIDEO_H__ */
