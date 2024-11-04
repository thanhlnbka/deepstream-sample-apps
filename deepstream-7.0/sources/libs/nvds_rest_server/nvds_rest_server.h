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

#ifndef _NVDS_SERVER_H_
#define _NVDS_SERVER_H_

#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include "gst-nvdscustomevent.h"
#include "gst-nvdscommonconfig.h"
#include <json/json.h>
#define UNKNOWN_STRING "unknown"
#define EMPTY_STRING ""

typedef enum
{
  DROP_FRAME_INTERVAL = 1 << 0,
  SKIP_FRAMES = 1 << 1,
  LOW_LATENCY_MODE = 1 << 2,
} NvDsServerDecPropFlag;

typedef enum
{
  BITRATE = 1 << 0,
  FORCE_IDR = 1 << 1,
  FORCE_INTRA = 1 << 2,
  IFRAME_INTERVAL = 1 << 3,
} NvDsServerEncPropFlag;

typedef enum
{
  SRC_CROP = 1 << 0,
  DEST_CROP = 1 << 1,
  FLIP_METHOD = 1 << 2,
  INTERPOLATION_METHOD = 1 << 3,
} NvDsServerConvPropFlag;

typedef enum
{
  BATCHED_PUSH_TIMEOUT = 1 << 0,
  MAX_LATENCY = 1 << 1,
} NvDsServerMuxPropFlag;

typedef enum
{
  INFER_INTERVAL = 1 << 0,
} NvDsServerInferPropFlag;

typedef enum
{
  INFERSERVER_INTERVAL = 1 << 0,
} NvDsServerInferServerPropFlag;

typedef enum
{
  GET_LIVE_STREAM_INFO = 1 << 0,
} NvDsServerGetRequestPropFlag;

typedef enum
{
  PROCESS_MODE = 1 << 0,
} NvDsServerOsdPropFlag;

typedef enum
{
  ROI_UPDATE = 1 << 0,
} NvDsServerRoiPropFlag;

typedef enum
{
  QUIT_APP = 1 << 0,
} NvDsServerAppInstanceFlag;

typedef enum
{
  QUIT_SUCCESS = 0,
  QUIT_FAIL,
} NvDsServerAppInstanceStatus;

typedef enum
{
  STREAM_ADD_SUCCESS = 0,
  STREAM_ADD_FAIL,
  STREAM_REMOVE_SUCCESS,
  STREAM_REMOVE_FAIL,
} NvDsServerStreamStatus;

typedef enum
{
  GET_LIVE_STREAM_INFO_SUCCESS = 0,
  GET_LIVE_STREAM_INFO_FAIL,
} NvDsServerGetRequestStatus;

typedef enum
{
  ROI_UPDATE_SUCCESS = 0,
  ROI_UPDATE_FAIL,
} NvDsServerRoiStatus;

typedef enum
{
  DROP_FRAME_INTERVAL_UPDATE_SUCCESS = 0,
  DROP_FRAME_INTERVAL_UPDATE_FAIL,
  SKIP_FRAMES_UPDATE_SUCCESS,
  SKIP_FRAMES_UPDATE_FAIL,
  LOW_LATENCY_MODE_UPDATE_SUCCESS,
  LOW_LATENCY_MODE_UPDATE_FAIL,
} NvDsServerDecStatus;

typedef enum
{
  BITRATE_UPDATE_SUCCESS = 0,
  BITRATE_UPDATE_FAIL,
  FORCE_IDR_UPDATE_SUCCESS,
  FORCE_IDR_UPDATE_FAIL,
  FORCE_INTRA_UPDATE_SUCCESS,
  FORCE_INTRA_UPDATE_FAIL,
  IFRAME_INTERVAL_UPDATE_SUCCESS,
  IFRAME_INTERVAL_UPDATE_FAIL,
} NvDsServerEncStatus;

typedef enum
{
  DEST_CROP_UPDATE_SUCCESS = 0,
  DEST_CROP_UPDATE_FAIL,
  SRC_CROP_UPDATE_SUCCESS,
  SRC_CROP_UPDATE_FAIL,
  INTERPOLATION_METHOD_UPDATE_SUCCESS,
  INTERPOLATION_METHOD_UPDATE_FAIL,
  FLIP_METHOD_UPDATE_SUCCESS,
  FLIP_METHOD_UPDATE_FAIL,
} NvDsServerConvStatus;

typedef enum
{
  BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS = 0,
  BATCHED_PUSH_TIMEOUT_UPDATE_FAIL,
  MAX_LATENCY_UPDATE_SUCCESS,
  MAX_LATENCY_UPDATE_FAIL,
} NvDsServerMuxStatus;

typedef enum
{
  INFER_INTERVAL_UPDATE_SUCCESS = 0,
  INFER_INTERVAL_UPDATE_FAIL,
} NvDsServerInferStatus;

typedef enum
{
  INFERSERVER_INTERVAL_UPDATE_SUCCESS = 0,
  INFERSERVER_INTERVAL_UPDATE_FAIL,
} NvDsServerInferServerStatus;

typedef enum
{
  PROCESS_MODE_UPDATE_SUCCESS = 0,
  PROCESS_MODE_UPDATE_FAIL,
} NvDsServerOsdStatus;

typedef enum
{
  StatusOk = 0,                         // HTTP error code : 200
  StatusAccepted,                       // HTTP error code : 202
  StatusBadRequest,                     // HTTP error code : 400
  StatusUnauthorized,                   // HTTP error code : 401
  StatusForbidden,                      // HTTP error code : 403
  StatusMethodNotAllowed,               // HTTP error code : 405
  StatusNotAcceptable,                  // HTTP error code : 406
  StatusProxyAuthenticationRequired,    // HTTP error code : 407
  StatusRequestTimeout,                 // HTTP error code : 408
  StatusPreconditionFailed,             // HTTP error code : 412
  StatusPayloadTooLarge,                // HTTP error code : 413
  StatusUriTooLong,                     // HTTP error code : 414
  StatusUnsupportedMediaType,           // HTTP error code : 415
  StatusInternalServerError,            // HTTP error code : 500
  StatusNotImplemented                  // HTTP error code : 501
} NvDsServerStatusCode;

typedef struct NvDsServerErrorInfo
{
  std::pair < int, std::string > err_log;
  NvDsServerStatusCode code;
} NvDsServerErrorInfo;

typedef struct NvDsServerDecInfo
{
  std::string root_key;
  std::string stream_id;
  guint drop_frame_interval;
  guint skip_frames;
  gboolean low_latency_mode;
  NvDsServerDecStatus status;
  NvDsServerDecPropFlag dec_flag;
  std::string dec_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerDecInfo;

typedef struct NvDsServerEncInfo
{
  std::string root_key;
  std::string stream_id;
  guint bitrate;
  gboolean force_idr;
  gboolean force_intra;
  guint iframeinterval;
  NvDsServerEncStatus status;
  NvDsServerEncPropFlag enc_flag;
  std::string enc_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerEncInfo;

typedef struct NvDsServerConvInfo
{
  std::string root_key;
  std::string stream_id;
  std::string src_crop;
  std::string dest_crop;
  guint flip_method;
  guint interpolation_method;
  NvDsServerConvStatus status;
  NvDsServerConvPropFlag conv_flag;
  std::string conv_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerConvInfo;

typedef struct NvDsServerMuxInfo
{
  std::string root_key;
  gint batched_push_timeout;
  guint max_latency;
  NvDsServerMuxStatus status;
  NvDsServerMuxPropFlag mux_flag;
  std::string mux_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerMuxInfo;

typedef struct NvDsServerRoiInfo
{
  std::string root_key;
  std::string stream_id;
  guint roi_count;
  std::vector < RoiDimension > vect;
  NvDsServerRoiStatus status;
  NvDsServerRoiPropFlag roi_flag;
  std::string roi_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerRoiInfo;

typedef struct NvDsServerStreamInfo
{
  std::string key;
  std::string value_camera_id;
  std::string value_camera_name;
  std::string value_camera_url;
  std::string value_change;

  std::string metadata_resolution;
  std::string metadata_codec;
  std::string metadata_framerate;

  std::string headers_source;
  std::string headers_created_at;
  NvDsServerStreamStatus status;
  std::string stream_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerStreamInfo;

typedef struct NvDsGetRequestInfo
{
  std::string root_key;
  std::string stream_id;
  NvDsServerGetRequestStatus status;
  NvDsServerGetRequestPropFlag get_request_flag;
  std::string get_request_log;
  std::string uri;
  Json::Value stream_info;
  std::vector<NvDsSensorInfo*> sensorInfo_vec;
  NvDsServerErrorInfo err_info;
} NvDsServerGetRequestInfo;

typedef struct NvDsServerInferInfo
{
  std::string root_key;
  std::string stream_id;
  guint interval;
  NvDsServerInferStatus status;
  NvDsServerInferPropFlag infer_flag;
  std::string infer_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerInferInfo;

typedef struct NvDsServerOsdInfo
{
  std::string root_key;
  std::string stream_id;
  guint process_mode;
  NvDsServerOsdStatus status;
  NvDsServerOsdPropFlag osd_flag;
  std::string osd_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerOsdInfo;

typedef struct NvDsServerAppInstanceInfo
{
  std::string root_key;
  gboolean app_quit;
  NvDsServerAppInstanceStatus status;
  NvDsServerAppInstanceFlag appinstance_flag;
  std::string app_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerAppInstanceInfo;

typedef struct NvDsServerInferServerInfo
{
  std::string root_key;
  std::string stream_id;
  guint interval;
  NvDsServerInferServerStatus status;
  NvDsServerInferServerPropFlag inferserver_flag;
  std::string inferserver_log;
  std::string uri;
  NvDsServerErrorInfo err_info;
} NvDsServerInferServerInfo;

typedef struct NvDsServerResponseInfo
{
  std::string status;
  std::string reason;
  Json::Value stream_info;
} NvDsServerResponseInfo;

typedef struct NvDsServerConfig
{
  std::string ip;
  std::string port;
} NvDsServerConfig;

using cb_func = std::function < NvDsServerStatusCode (const Json::Value & req_info,
      const Json::Value & in,
      Json::Value & out, struct mg_connection * conn, void *ctx)>;

typedef struct NvDsServerCallbacks
{
  std::function < void (NvDsServerRoiInfo * roi_info, void *ctx) > roi_cb;
  std::function < void (NvDsServerDecInfo * dec_info, void *ctx) > dec_cb;
  std::function < void (NvDsServerEncInfo * enc_info, void *ctx) > enc_cb;
  std::function < void (NvDsServerStreamInfo * stream_info,
    void *ctx) > stream_cb;
  std::function < void (NvDsServerInferInfo * infer_info,
    void *ctx) > infer_cb;
  std::function < void (NvDsServerConvInfo * conv_info, void *ctx) > conv_cb;
  std::function < void (NvDsServerMuxInfo * mux_info, void *ctx) > mux_cb;
  std::function < void (NvDsServerInferServerInfo * inferserver_info,
    void *ctx) > inferserver_cb;
  std::function < void (NvDsServerOsdInfo * osd_info, void *ctx) > osd_cb;
  std::function < void (NvDsServerAppInstanceInfo * appinstance_info,
    void *ctx) > appinstance_cb;
  std::function < void (NvDsServerGetRequestInfo * get_request_info,
    void *ctx) > get_request_cb;
  std::unordered_map <std::string, cb_func> custom_cb_endpt;
} NvDsServerCallbacks;

class NvDsRestServer;
NvDsRestServer* nvds_rest_server_start (NvDsServerConfig * server_config, NvDsServerCallbacks * server_cb);
void nvds_rest_server_stop (NvDsRestServer *ctx);
bool iequals (const std::string & a, const std::string & b);

#endif
