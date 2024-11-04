/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __GST_DS_NVMULTIURISRC_BIN_H__
#define __GST_DS_NVMULTIURISRC_BIN_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include "gst-nvmultiurisrcbincreator.h"
#include "gstdsnvurisrcbin.h"

G_BEGIN_DECLS

enum
{
  MULTIURIBIN_PROP_0,
  MULTIURIBIN_PROP_URI_LIST,
  MULTIURIBIN_PROP_SENSOR_ID_LIST,
  MULTIURIBIN_PROP_SENSOR_NAME_LIST,
  MULTIURIBIN_PROP_MODE,
  MULTIURIBIN_PROP_HTTP_PORT,
  MULTIURIBIN_PROP_MAX_BATCH_SIZE,
  MULTIURIBIN_PROP_HTTP_IP,
  MULTIURIBIN_PROP_NUM_EXTRA_SURF,
  MULTIURIBIN_PROP_GPU_DEVICE_ID,
  MULTIURIBIN_PROP_DEC_SKIP_FRAMES,
  MULTIURIBIN_PROP_SOURCE_TYPE,
  MULTIURIBIN_PROP_CUDADEC_MEM_TYPE,
  MULTIURIBIN_PROP_DROP_FRAME_INTERVAL,
  MULTIURIBIN_PROP_RTP_PROTOCOL,
  MULTIURIBIN_PROP_FILE_LOOP,
  MULTIURIBIN_PROP_SMART_RECORD,
  MULTIURIBIN_PROP_SMART_RECORD_DIR_PATH,
  MULTIURIBIN_PROP_SMART_RECORD_FILE_PREFIX,
  MULTIURIBIN_PROP_SMART_RECORD_VIDEO_CACHE,
  MULTIURIBIN_PROP_SMART_RECORD_CACHE,
  MULTIURIBIN_PROP_SMART_RECORD_CONTAINER,
  MULTIURIBIN_PROP_SMART_RECORD_MODE,
  MULTIURIBIN_PROP_SMART_RECORD_DEFAULT_DURATION,
  MULTIURIBIN_PROP_SMART_RECORD_STATUS,
  MULTIURIBIN_PROP_RTSP_RECONNECT_INTERVAL,
  MULTIURIBIN_PROP_RTSP_RECONNECT_ATTEMPTS,
  MULTIURIBIN_PROP_LATENCY,
  MULTIURIBIN_PROP_SOURCE_ID,
  MULTIURIBIN_PROP_UDP_BUFFER_SIZE,
  MULTIURIBIN_PROP_DISABLE_PASSTHROUGH,
  MULTIURIBIN_PROP_DISABLE_AUDIO,
  MULTIURIBIN_PROP_EXTRACT_SEI_TYPE5_DATA_DEC,
  MULTIURIBIN_PROP_SEI_UUID,
  MULTIURIBIN_PROP_LOW_LATENCY_MODE,
  //nvstreammux props:

  PROP_BATCH_SIZE,
  PROP_BATCHED_PUSH_TIMEOUT,
  PROP_WIDTH,
  PROP_HEIGHT,
  PROP_ENABLE_PADDING,
  PROP_LIVE_SOURCE,
  PROP_NUM_SURFACES_PER_FRAME,
  PROP_NVBUF_MEMORY_TYPE,
  PROP_COMPUTE_HW,
  PROP_INTERPOLATION_METHOD,
  PROP_BUFFER_POOL_SIZE,
  PROP_ATTACH_SYS_TIME_STAMP,
  PROP_SYNC_INPUTS,
  PROP_MAX_LATNECY,
  PROP_FRAME_NUM_RESET_ON_EOS,
  PROP_FRAME_NUM_RESET_ON_STREAM_RESET,
  PROP_FRAME_DURATION,
  PROP_ASYNC_PROCESS,
  PROP_NO_PIPELINE_EOS,
  PROP_CONFIG_FILE_PATH,
  PROP_EXTRACT_SEI_TYPE5_DATA_MUX,
  MULTIURIBIN_PROP_LAST
};

typedef struct _GstDsNvMultiUriBin
{
  GstBin bin;

  GMutex bin_lock;

  /** source config that will be used for all the N uri sources
   * that may be added into this bin */
  GstDsNvUriSrcConfig *config;
  GstDsNvStreammuxConfig* muxConfig;
  gchar* uriList;
  gchar** uriListV;
  gchar* sensorIdList;
  gchar** sensorIdListV;
  gchar* sensorNameList;
  gchar** sensorNameListV;
  NvDsMultiUriMode mode;
  NvDst_Handle_NvMultiUriSrcCreator nvmultiurisrcbinCreator;
  guint sourceIdCounter;
  GstPad* bin_src_pad;
  void* restServer;
  gchar* httpIp;
  gchar* httpPort;
} GstDsNvMultiUriBin;

typedef struct _GstDsNvMultiUriBinClass
{
  GstBinClass parent_class;
} GstDsNvMultiUriBinClass;


/* Standard GStreamer boilerplate */
#define GST_TYPE_DS_NVMULTIURISRC_BIN (gst_ds_nvmultiurisrc_bin_get_type())
#define GST_DS_NVMULTIURISRC_BIN(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_DS_NVMULTIURISRC_BIN,GstDsNvMultiUriBin))
#define GST_DS_NVMULTIURISRC_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_DS_NVMULTIURISRC_BIN,GstDsNvMultiUriBinClass))
#define GST_DS_NVMULTIURISRC_BIN_GET_CLASS(obj) (G_TYPE_INSTANCE_GET_CLASS((obj), GST_TYPE_DS_NVMULTIURISRC_BIN, GstDsNvMultiUriBinClass))
#define GST_IS_DS_NVMULTIURISRC_BIN(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_DS_NVMULTIURISRC_BIN))
#define GST_IS_DS_NVMULTIURISRC_BIN_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_DS_NVMULTIURISRC_BIN))
#define GST_DS_NVMULTIURISRC_BIN_CAST(obj)  ((GstDsNvMultiUriBin *)(obj))

GType gst_ds_nvmultiurisrc_bin_get_type (void);

G_END_DECLS
#endif /* __GST_DS_NVMULTIURISRC_BIN_H__ */
