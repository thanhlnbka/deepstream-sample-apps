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

#include "nvds_rest_server.h"
#include "nvds_parse.h"
#include <iostream>
#include <unistd.h>
#include <memory>
#include "CivetServer.h"

#define stringify( name ) #name

/* ---------------------------------------------------------------------------
**  http callback
** -------------------------------------------------------------------------*/
class NvDsRestServer:public CivetServer
{
public:
  typedef std::function < NvDsServerStatusCode (const Json::Value &,
      const Json::Value &, Json::Value &,
      struct mg_connection *conn) > httpFunction;

    NvDsRestServer (const std::vector < std::string > &options);

  void addRequestHandler (std::map < std::string, httpFunction > &func);
  std::map < std::string, void *> requestList;
};

std::pair < int,
    std::string >
NvDsServerStatusCodeToHttpStatusCode (NvDsServerStatusCode code);

bool iequals (const std::string & a, const std::string & b);

int log_message (const struct mg_connection *conn, const char *message);

const struct CivetCallbacks *getCivetCallbacks ();

static bool log_api_info (const std::string & api_name);

NvDsServerStatusCode VersionInfo (Json::Value & response, struct mg_connection *conn);

NvDsServerStatusCode
handleCustomReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < NvDsServerStatusCode (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn, void *ctx) > custom_cb);

NvDsServerStatusCode
handleUpdateROI (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerRoiInfo * roi_ctx, void *ctx) > roi_cb,
    std::string uri);

NvDsServerStatusCode
handleOsdReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerOsdInfo * osd_ctx, void *ctx) > osd_cb,
    std::string uri);

NvDsServerStatusCode
handleEncReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerEncInfo * enc_ctx, void *ctx) > enc_cb,
    std::string uri);

NvDsServerStatusCode
handleConvReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerConvInfo * conv_ctx, void *ctx) > conv_cb,
    std::string uri);

NvDsServerStatusCode
handleMuxReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerMuxInfo * mux_ctx, void *ctx) > mux_cb,
    std::string uri);

NvDsServerStatusCode
handleAppReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerAppInstanceInfo * appinstance_ctx,
        void *ctx) > appinstance_cb, std::string uri);

NvDsServerStatusCode
handleDecReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerDecInfo * dec_ctx, void *ctx) > dec_cb,
    std::string uri);

NvDsServerStatusCode
handleAddStream (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerStreamInfo * stream_ctx, void *ctx) > stream_cb,
    std::string uri);

NvDsServerStatusCode
handleRemoveStream (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerStreamInfo * stream_ctx, void *ctx) > stream_cb,
    std::string uri);

NvDsServerStatusCode
handleGetRequest (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerGetRequestInfo * get_request_ctx, void *ctx) > get_request_cb,
    std::string uri);

NvDsServerStatusCode
handleInferReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerInferInfo * infer_ctx, void *ctx) > infer_cb,
    std::string uri);

NvDsServerStatusCode
handleInferServerReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerInferServerInfo * inferserver_ctx,
        void *ctx) > inferserver_cb, std::string uri);

std::pair < int,
    std::string >
NvDsServerStatusCodeToHttpStatusCode (NvDsServerStatusCode code)
{
  switch ((int) code) {
    case StatusOk:
      return std::make_pair (200, "OK");
    case StatusAccepted:
      return std::make_pair (202, "Accepted");
    case StatusBadRequest:
      return std::make_pair (400, "Bad Request");
    case StatusUnauthorized:
      return std::make_pair (401, "Unauthorized");
    case StatusForbidden:
      return std::make_pair (403, "Forbidden");
    case StatusMethodNotAllowed:
      return std::make_pair (405, "Method Not Allowed");
    case StatusNotAcceptable:
      return std::make_pair (406, "Not Acceptable");
    case StatusProxyAuthenticationRequired:
      return std::make_pair (407, "Proxy Authentication Required");
    case StatusRequestTimeout:
      return std::make_pair (408, "Request Timout");
    case StatusPreconditionFailed:
      return std::make_pair (412, "Precondition Failed");
    case StatusPayloadTooLarge:
      return std::make_pair (413, "Payload Too Large");
    case StatusUriTooLong:
      return std::make_pair (414, "Uri Too Long");
    case StatusUnsupportedMediaType:
      return std::make_pair (415, "Unsupported Media Type");
    case StatusInternalServerError:
      return std::make_pair (500, "Internal Server Error");
    case StatusNotImplemented:
      return std::make_pair (501, "Not Implemented");
    default:
      return std::make_pair (501, "Not Implemented");
  }
}

bool
iequals (const std::string & a, const std::string & b)
{
  return std::equal (a.begin (), a.end (),
      b.begin (), b.end (),[](char str1, char str2) {
        return std::tolower (str1) == std::tolower (str2);
      }
  );
}


int
log_message (const struct mg_connection *conn, const char *message)
{
  fprintf (stderr, "%s\n", message);
  // LOG(verbose) << "HTTP SERVER: " << message << endl;
  return 0;
}

static struct CivetCallbacks _callbacks;
const struct CivetCallbacks *
getCivetCallbacks ()
{
  //memset(&_callbacks, 0, sizeof(_callbacks));
  _callbacks.log_message = &log_message;
  return &_callbacks;
}

static bool
log_api_info (const std::string & api_name)
{
  if ((api_name == "/api/stream/status") || (api_name == "/api/stream/stats")) {
    return false;
  }
  return true;
}

/* ---------------------------------------------------------------------------
**  Civet HTTP callback
** -------------------------------------------------------------------------*/
class RequestHandler:public CivetHandler
{
public:
  RequestHandler (std::string uri,
      NvDsRestServer::httpFunction & func):m_uri (std::move (uri))
  , m_func (func)
  {
  }

  bool handle (CivetServer * server, struct mg_connection *conn)
  {
    bool ret = false;
    Json::Value response;
    Json::Value req;
    NvDsServerStatusCode result;
    const struct mg_request_info *req_info = mg_get_request_info (conn);
    if (req_info == NULL) {
      std::cout << "req_info is NULL " << std::endl;
      return ret;
    }

    if (log_api_info (req_info->request_uri)) {
      std::cout << "uri:" << req_info->request_uri << std::endl;
      std::cout << "method:" << req_info->request_method << std::endl;
    }

    if (m_uri.back () != '*') {
      if (m_uri != req_info->request_uri) {
        std::cout << "Wrong API uri:" << req_info->request_uri <<
            " Please use correct uri: " << m_uri << std::endl;
        return ret;
      }
    }
    // read input
    Json::Value in;
    result = this->getInputMessage (req_info, conn, in);
    req["url"] = req_info->request_uri;
    req["method"] = req_info->request_method;
    req["query"] =
        req_info->query_string != NULL ? req_info->query_string : "";
    req["remote_addr"] = req_info->remote_addr;
    req["remote_user"] =
        req_info->remote_user != NULL ? req_info->remote_user : "";
    //Invoke API implementation (both cases i.e. valid & invalid Json input
    result = m_func (req, in, response, conn);

    return httpResponseHandler (result, response, conn);
  }

  bool httpResponseHandler (NvDsServerStatusCode & result, Json::Value & response,
      struct mg_connection *conn)
  {
    if (result == NvDsServerStatusCode::StatusOk) {
      mg_printf (conn, "HTTP/1.1 200 OK\r\n");
    } else {
      std::pair < int, std::string > http_err_code =
          NvDsServerStatusCodeToHttpStatusCode (result);
      std::string response =
          std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
          " " + http_err_code.second;
      mg_printf (conn, "%s\r\n", response.c_str ());
    }
    mg_printf (conn, "Access-Control-Allow-Origin: *\r\n");
    std::string content_type;
    if (response.isObject ()) {
      content_type = response.get ("content_type", "").asString ();
    }
    std::string answer;
    if (content_type.empty () == false) {
      answer = response.get ("data", "").asString ();
      mg_printf (conn, "Content-Type: image/jpeg\r\n");
    } else {
      std::string ans (Json::writeString (m_writerBuilder, response));
      answer = std::move (ans);
      mg_printf (conn, "Content-Type: text/plain\r\n");
    }
    mg_printf (conn, "Content-Length: %zd\r\n", answer.size ());
    mg_printf (conn, "Connection: close\r\n");
    mg_printf (conn, "\r\n");
    mg_write (conn, answer.c_str (), answer.size ());
    return true;
  }

  bool handleGet (CivetServer * server, struct mg_connection *conn)
  {
    return handle (server, conn);
  }
  bool handlePost (CivetServer * server, struct mg_connection *conn)
  {
    return handle (server, conn);
  }
  bool handlePut (CivetServer * server, struct mg_connection *conn)
  {
    return handle (server, conn);
  }
  bool handleDelete (CivetServer * server, struct mg_connection *conn)
  {
    return handle (server, conn);
  }

private:
  std::string m_uri;
  NvDsRestServer::httpFunction m_func;
  Json::StreamWriterBuilder m_writerBuilder;
  Json::CharReaderBuilder m_readerBuilder;

  NvDsServerStatusCode getInputMessage (const struct mg_request_info *req_info,
      struct mg_connection *conn, Json::Value & out)
  {
    //Return if content length is zero otherwise procede to check content type
    if (req_info == NULL || conn == NULL) {
      out = Json::nullValue;
      std::string error_message = "Request Information is null";
      std::cout << error_message << std::endl;

      return NvDsServerStatusCode::StatusBadRequest;
    }
    long long tlen = req_info->content_length;
    if (tlen > 0) {
      std::string body;
      unsigned long long rlen;
      long long nlen = 0;
      char buf[1024] = "";
      while (nlen < tlen) {
        rlen = tlen - nlen;
        if (rlen > sizeof (buf)) {
          rlen = sizeof (buf);
        }
        rlen = mg_read (conn, buf, (size_t) rlen);
        if (rlen <= 0) {
          break;
        }
        try {
          body.append (buf, rlen);
        }
        catch (const std::exception & e)
        {
          std::cout << "Exception while fetching content data" << std::endl;
          break;
        }
        nlen += rlen;
      }
      // parse in
      std::unique_ptr < Json::CharReader >
          reader (m_readerBuilder.newCharReader ());
      std::string errors;
      if (!reader->parse (body.c_str (), body.c_str () + body.size (), &out,
              &errors)) {
        out = Json::nullValue;
        std::string error_message =
            std::string ("Received unknown message:") + body +
            std::string (" errors:") + errors;
        std::cout << error_message << std::endl;
        return NvDsServerStatusCode::StatusBadRequest;
      }
    }
    return NvDsServerStatusCode::StatusOk;
  }
};

/* ---------------------------------------------------------------------------
**  Constructor
** -------------------------------------------------------------------------*/
NvDsRestServer::NvDsRestServer (const std::vector < std::string > &options)
    :
CivetServer (options, getCivetCallbacks ())
{
  std::cout << "Civetweb version: v" << mg_version () << std::endl;
}

NvDsServerStatusCode
VersionInfo (Json::Value & response, struct mg_connection *conn)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  response["version"] = "DeepStream-SDK 7.0 - REST API Version v1";
  return ret;
}

void __attribute__((constructor)) nvds_rest_server_init (void);
void __attribute__((destructor)) nvds_rest_server_deinit (void);

void __attribute__((constructor)) nvds_rest_server_init (void)
{
  mg_init_library (0);
}

void __attribute__((destructor)) nvds_rest_server_deinit (void)
{
  mg_exit_library ();
}

void
nvds_rest_server_stop (NvDsRestServer * handler)
{
  std::cout << "Stopping the server..!! \n";

  if (handler) {
    delete handler;
  }

  std::cout << "Stopped the server..!! \n";

}

NvDsServerStatusCode
handleInferReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerInferInfo * infer_ctx, void *ctx) > infer_cb,
    std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {
    NvDsServerInferInfo infer_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    infer_info.uri = uri;

    void *custom_ctx = NULL;

    if (request_api.find ("set-interval") != std::string::npos) {
      infer_info.infer_flag = INFER_INTERVAL;
    }
    if (nvds_rest_infer_parse (in, &infer_info) && (infer_cb)) {
      infer_cb (&infer_info, &custom_ctx);

      switch (infer_info.infer_flag) {
        case INFER_INTERVAL:
          http_err_code = NvDsServerStatusCodeToHttpStatusCode(infer_info.err_info.code);
          break;
        default:
          break;
      }
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(infer_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
          " " + http_err_code.second;

    res_info.reason = infer_info.infer_log;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
  }

  return ret;
}

NvDsServerStatusCode
handleInferServerReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerInferServerInfo * inferserver_ctx,
        void *ctx) > inferserver_cb, std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {
    NvDsServerInferServerInfo inferserver_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    inferserver_info.uri = uri;

    void *custom_ctx = NULL;
    if (request_api.find ("set-interval") != std::string::npos) {
      inferserver_info.inferserver_flag = INFERSERVER_INTERVAL;
    }

    if (nvds_rest_inferserver_parse (in, &inferserver_info) && (inferserver_cb)) {
      inferserver_cb (&inferserver_info, &custom_ctx);
      switch (inferserver_info.inferserver_flag) {
        case INFERSERVER_INTERVAL:
          http_err_code = NvDsServerStatusCodeToHttpStatusCode(inferserver_info.err_info.code);
          break;
        default:
          break;
      }
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(inferserver_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
          " " + http_err_code.second;

    res_info.reason = inferserver_info.inferserver_log;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
  }

  return ret;
}

NvDsServerStatusCode
handleDecReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerDecInfo * dec_ctx, void *ctx) > dec_cb,
    std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {
    NvDsServerDecInfo dec_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    dec_info.uri = uri;

    void *custom_ctx = NULL;
    if (request_api.find ("drop-frame-interval") != std::string::npos) {
      dec_info.dec_flag = DROP_FRAME_INTERVAL;
    }
    if (request_api.find ("skip-frames") != std::string::npos) {
      dec_info.dec_flag = SKIP_FRAMES;
    }
    if (request_api.find ("low-latency-mode") != std::string::npos) {
      dec_info.dec_flag = LOW_LATENCY_MODE;
    }

    if (nvds_rest_dec_parse (in, &dec_info) && (dec_cb)) {
      dec_cb (&dec_info, &custom_ctx);
      switch (dec_info.dec_flag) {
        case DROP_FRAME_INTERVAL:
        case SKIP_FRAMES:
        case LOW_LATENCY_MODE:
          http_err_code = NvDsServerStatusCodeToHttpStatusCode(dec_info.err_info.code);
          break;
        default:
          break;
      }
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(dec_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
          " " + http_err_code.second;

    res_info.reason = dec_info.dec_log;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
  }

  return ret;
}

NvDsServerStatusCode
handleEncReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerEncInfo * enc_ctx, void *ctx) > enc_cb,
    std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {
    NvDsServerEncInfo enc_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    enc_info.uri = uri;

    void *custom_ctx = NULL;
    if (request_api.find ("bitrate") != std::string::npos) {
      enc_info.enc_flag = BITRATE;
    }
    if (request_api.find ("force-idr") != std::string::npos) {
      enc_info.enc_flag = FORCE_IDR;
    }
    if (request_api.find ("force-intra") != std::string::npos) {
      enc_info.enc_flag = FORCE_INTRA;
    }
    if (request_api.find ("iframe-interval") != std::string::npos) {
      enc_info.enc_flag = IFRAME_INTERVAL;
    }
    if (nvds_rest_enc_parse (in, &enc_info) && (enc_cb)) {
      enc_cb (&enc_info, &custom_ctx);
      switch (enc_info.enc_flag) {
        case BITRATE:
        case FORCE_IDR:
        case FORCE_INTRA:
        case IFRAME_INTERVAL:
          http_err_code = NvDsServerStatusCodeToHttpStatusCode(enc_info.err_info.code);
          break;
        default:
          break;
      }
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(enc_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
      " " + http_err_code.second;

    res_info.reason = enc_info.enc_log;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
  }

  return ret;
}

NvDsServerStatusCode
handleConvReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerConvInfo * conv_ctx, void *ctx) > conv_cb,
    std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {
    NvDsServerConvInfo conv_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    conv_info.uri = uri;

    void *custom_ctx = NULL;
    if (request_api.find ("srccrop") != std::string::npos) {
      conv_info.conv_flag = SRC_CROP;
    }
    if (request_api.find ("destcrop") != std::string::npos) {
      conv_info.conv_flag = DEST_CROP;
    }
    if (request_api.find ("flip-method") != std::string::npos) {
      conv_info.conv_flag = FLIP_METHOD;
    }
    if (request_api.find ("interpolation-method") != std::string::npos) {
      conv_info.conv_flag = INTERPOLATION_METHOD;
    }
    if (nvds_rest_conv_parse (in, &conv_info) && (conv_cb)) {
      conv_cb (&conv_info, &custom_ctx);
      switch (conv_info.conv_flag) {
        case SRC_CROP:
        case DEST_CROP:
        case FLIP_METHOD:
        case INTERPOLATION_METHOD:
          http_err_code = NvDsServerStatusCodeToHttpStatusCode(conv_info.err_info.code);
          break;
        default:
          break;
      }
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(conv_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
          " " + http_err_code.second;

    res_info.reason = conv_info.conv_log;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
  }

  return ret;
}

NvDsServerStatusCode
handleMuxReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerMuxInfo * mux_ctx, void *ctx) > mux_cb,
    std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {
    NvDsServerMuxInfo mux_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    mux_info.uri = uri;

    if (request_api.find ("batched-push-timeout") != std::string::npos) {
      mux_info.mux_flag = BATCHED_PUSH_TIMEOUT;
    }
    if (request_api.find ("max-latency") != std::string::npos) {
      mux_info.mux_flag = MAX_LATENCY;
    }

    void *custom_ctx = NULL;
    if (nvds_rest_mux_parse (in, &mux_info) && (mux_cb)) {
      mux_cb (&mux_info, &custom_ctx);
      switch (mux_info.mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
        case MAX_LATENCY:
          http_err_code = NvDsServerStatusCodeToHttpStatusCode(mux_info.err_info.code);
          break;
        default:
          break;
      }
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(mux_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
    " " + http_err_code.second;

    res_info.reason = mux_info.mux_log;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
  }

  return ret;
}

NvDsServerStatusCode
handleOsdReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerOsdInfo * osd_ctx, void *ctx) > osd_cb,
    std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {
    NvDsServerOsdInfo osd_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    osd_info.uri = uri;

    if (request_api.find ("process-mode") != std::string::npos) {
      osd_info.osd_flag = PROCESS_MODE;
    }

    void *custom_ctx = NULL;
    if (nvds_rest_osd_parse (in, &osd_info) && (osd_cb)) {
      osd_cb (&osd_info, &custom_ctx);
      switch (osd_info.osd_flag) {
        case PROCESS_MODE:
          http_err_code = NvDsServerStatusCodeToHttpStatusCode(osd_info.err_info.code);
          break;
        default:
          break;
      }
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(osd_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
      " " + http_err_code.second;

    res_info.reason = osd_info.osd_log;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
  }
  return ret;
}

NvDsServerStatusCode
handleAppReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerAppInstanceInfo * appinstance_ctx,
        void *ctx) > appinstance_cb, std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {
    NvDsServerAppInstanceInfo appinstance_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    appinstance_info.uri = uri;

    if (request_api.find ("quit") != std::string::npos) {
      appinstance_info.appinstance_flag = QUIT_APP;
    }

    void *custom_ctx = NULL;
    if (nvds_rest_app_instance_parse (in, &appinstance_info)
        && (appinstance_cb)) {
      appinstance_cb (&appinstance_info, &custom_ctx);

      switch (appinstance_info.appinstance_flag) {
        case QUIT_APP:
          http_err_code = NvDsServerStatusCodeToHttpStatusCode(appinstance_info.err_info.code);
          break;
        default:
          break;
      }
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(appinstance_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
          " " + http_err_code.second;

    res_info.reason = appinstance_info.app_log;
    if (res_info.reason == "")
      res_info.reason = "NA";

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
  }

  return ret;
}

NvDsServerStatusCode
handleUpdateROI (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerRoiInfo * roi_ctx, void *ctx) > roi_cb,
    std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {

    NvDsServerRoiInfo roi_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    roi_info.uri = uri;

    void *custom_ctx = NULL;

    if (request_api.find ("update") != std::string::npos) {
      roi_info.roi_flag = ROI_UPDATE;
    }
    if (nvds_rest_roi_parse (in, &roi_info) && (roi_cb)) {
      roi_cb (&roi_info, &custom_ctx);
      switch (roi_info.roi_flag) {
        case ROI_UPDATE:
          http_err_code = NvDsServerStatusCodeToHttpStatusCode(roi_info.err_info.code);
          break;
        default:
          break;
      }
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(roi_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
          " " + http_err_code.second;

    res_info.reason = roi_info.roi_log;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;

  }
  return ret;
}

NvDsServerStatusCode
handleAddStream (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerStreamInfo * stream_ctx, void *ctx) > stream_cb,
    std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {

    NvDsServerStreamInfo stream_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    stream_info.uri = uri;

    void *custom_ctx = NULL;

    if (nvds_rest_stream_parse (in, &stream_info) && (stream_cb)) {
      stream_cb (&stream_info, &custom_ctx);
        http_err_code = NvDsServerStatusCodeToHttpStatusCode(stream_info.err_info.code);
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(stream_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
          " " + http_err_code.second;
    res_info.reason = stream_info.stream_log;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
  }
  return ret;
}

NvDsServerStatusCode
handleCustomReq (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < NvDsServerStatusCode (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn, void *ctx) > custom_cb)
{
  NvDsServerStatusCode ret =  NvDsServerStatusCode::StatusOk;
  void *custom_ctx;
  ret = custom_cb(req_info, in, response, conn, &custom_ctx);
  return ret;
}

NvDsServerStatusCode
handleRemoveStream (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerStreamInfo * stream_ctx, void *ctx) > stream_cb,
    std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {

  }

  if (iequals (request_method, "post")) {

    NvDsServerStreamInfo stream_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    stream_info.uri = uri;

    void *custom_ctx = NULL;

    if (nvds_rest_stream_parse (in, &stream_info) && (stream_cb)) {
      stream_cb (&stream_info, &custom_ctx);
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(stream_info.err_info.code);
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(stream_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
          " " + http_err_code.second;
    res_info.reason = stream_info.stream_log;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
  }
  return ret;
}

NvDsServerStatusCode
handleGetRequest (const Json::Value & req_info, const Json::Value & in,
    Json::Value & response, struct mg_connection *conn,
    std::function < void (NvDsServerGetRequestInfo * get_request_ctx, void *ctx) > get_request_cb,
    std::string uri)
{
  NvDsServerStatusCode ret = NvDsServerStatusCode::StatusOk;
  const std::string request_api =
      req_info.get ("url", EMPTY_STRING).asString ();
  const std::string request_method =
      req_info.get ("method", UNKNOWN_STRING).asString ();
  const std::string query_string =
      req_info.get ("query", EMPTY_STRING).asString ();

  if (request_api.empty () || request_method == UNKNOWN_STRING) {
    std::cout << "Malformed HTTP request" << std::endl;
    return NvDsServerStatusCode::StatusBadRequest;
  }

  if (iequals (request_method, "get")) {
    NvDsServerGetRequestInfo get_request_info = { };
    NvDsServerResponseInfo res_info = { };
    std::pair < int, std::string > http_err_code(0,"");
    get_request_info.uri = uri;

    void *custom_ctx = NULL;
    if (request_api.find ("get-stream-info") != std::string::npos) {
      get_request_info.get_request_flag = GET_LIVE_STREAM_INFO;
    }
    if (get_request_cb) {
      get_request_cb (&get_request_info, &custom_ctx);
      switch (get_request_info.get_request_flag) {
        case GET_LIVE_STREAM_INFO:
          http_err_code = NvDsServerStatusCodeToHttpStatusCode(get_request_info.err_info.code);
          break;
        default:
          break;
      }
    } else {
      http_err_code = NvDsServerStatusCodeToHttpStatusCode(get_request_info.err_info.code);
    }
    res_info.status = std::string ("HTTP/1.1 ") + std::to_string (http_err_code.first) +
          " " + http_err_code.second;
    res_info.reason = get_request_info.get_request_log;
    res_info.stream_info = get_request_info.stream_info;

    response["status"] = res_info.status;
    response["reason"] = res_info.reason;
    response["stream-info"] = res_info.stream_info;
  }

  if (iequals (request_method, "post")) {

  }
  return ret;
}

NvDsRestServer *
nvds_rest_server_start (NvDsServerConfig * server_config,
    NvDsServerCallbacks * server_cb)
{
  auto roi_cb = server_cb->roi_cb;
  auto dec_cb = server_cb->dec_cb;
  auto enc_cb = server_cb->enc_cb;
  auto stream_cb = server_cb->stream_cb;
  auto infer_cb = server_cb->infer_cb;
  auto conv_cb = server_cb->conv_cb;
  auto mux_cb = server_cb->mux_cb;
  auto inferserver_cb = server_cb->inferserver_cb;
  auto osd_cb = server_cb->osd_cb;
  auto appinstance_cb = server_cb->appinstance_cb;
  auto get_request_cb = server_cb->get_request_cb;

  const char *options[] = {
    "listening_ports", server_config->port.c_str (), 0
  };

  std::vector < std::string > cpp_options;
  for (long unsigned int i = 0;
      i < (sizeof (options) / sizeof (options[0]) - 1); i++) {
    cpp_options.push_back (options[i]);
    // std::cout<< "cpp option : "<<cpp_options[i]<<"\n";
  }

  NvDsRestServer *httpServerHandler = new NvDsRestServer (cpp_options);

  std::map < std::string, NvDsRestServer::httpFunction > m_func;
  std::map < std::string, std::string > m_versions;

  /* Add supported uri(s) in the map m_versions
   */
  m_versions.insert ( {
      "/api/v1/stream/add", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/stream/remove", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/roi/update", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/dec/drop-frame-interval", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/dec/skip-frames", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/dec/drop-frame-interval", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/dec/low-latency-mode", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/enc/bitrate", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/enc/force-idr", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/enc/force-intra", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/enc/iframe-interval", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/infer/set-interval", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/inferserver/set-interval", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/conv/srccrop", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/conv/destcrop", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/conv/interpolation-method", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/conv/flip-method", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/mux/batched-push-timeout", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/mux/max-latency", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/osd/process-mode", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/app/quit", "v1"}
  );
  m_versions.insert ( {
      "/api/v1/stream/get-stream-info", "v1"}
  );

  /* To query the supported REST API version(s) */
  m_func["/version"] =
      [server_cb] (const Json::Value & req_info, const Json::Value & in,
      Json::Value & out, struct mg_connection * conn) {
    return VersionInfo (out, conn);
  };

  for (auto itr = m_versions.begin (); itr != m_versions.end (); itr++) {
    std::string uri = itr->first;
    if (uri.find ("/stream/add") != std::string::npos) {
      /* Stream Management Specific */
      m_func[uri] =
          [stream_cb, uri] (const Json::Value & req_info,
          const Json::Value & in, Json::Value & out,
          struct mg_connection * conn) {
        return handleAddStream (req_info, in, out, conn, stream_cb, uri);
      };
    } else if (uri.find ("/stream/remove") != std::string::npos) {
      /* Stream Management Specific */
      m_func[uri] =
          [stream_cb, uri] (const Json::Value & req_info,
          const Json::Value & in, Json::Value & out,
          struct mg_connection * conn) {
        return handleRemoveStream (req_info, in, out, conn, stream_cb, uri);
      };
    } else if (uri.find ("/stream/get-stream-info") != std::string::npos) {
      /* GET Requests Specific */
      m_func[uri] =
          [get_request_cb, uri] (const Json::Value & req_info,
          const Json::Value & in, Json::Value & out,
          struct mg_connection * conn) {
        return handleGetRequest (req_info, in, out, conn, get_request_cb, uri);
      };
    }
    else if (uri.find ("/roi/update") != std::string::npos) {
      /* Pre-Process Specific */
      m_func[uri] =
          [roi_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleUpdateROI (req_info, in, out, conn, roi_cb, uri);
      };
    }
    else if (uri.find ("/dec/drop-frame-interval") != std::string::npos) {
      /* Decoder Specific */
      m_func[uri] =
          [dec_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleDecReq (req_info, in, out, conn, dec_cb, uri);
      };
    } else if (uri.find ("/dec/skip-frames") != std::string::npos) {
      /* Decoder Specific */
      m_func[uri] =
          [dec_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleDecReq (req_info, in, out, conn, dec_cb, uri);
      };
    } else if (uri.find ("/dec/low-latency-mode") != std::string::npos) {
      /* Decoder Specific */
      m_func[uri] =
          [dec_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleDecReq (req_info, in, out, conn, dec_cb, uri);
      };
    }
    else if (uri.find ("/enc/bitrate") != std::string::npos) {
      /* Encoder Specific */
      m_func[uri] =
          [enc_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleEncReq (req_info, in, out, conn, enc_cb, uri);
      };
    } else if (uri.find ("/enc/force-idr") != std::string::npos) {
      /* Encoder Specific */
      m_func[uri] =
          [enc_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleEncReq (req_info, in, out, conn, enc_cb, uri);
      };
    } else if (uri.find ("/enc/force-intra") != std::string::npos) {
      /* Encoder Specific */
      m_func[uri] =
          [enc_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleEncReq (req_info, in, out, conn, enc_cb, uri);
      };
    } else if (uri.find ("/enc/iframe-interval") != std::string::npos) {
      /* Encoder Specific */
      m_func[uri] =
          [enc_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleEncReq (req_info, in, out, conn, enc_cb, uri);
      };
    }
    else if (uri.find ("/infer/set-interval") != std::string::npos) {
      /* Inference Specific */
      m_func[uri] =
          [infer_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleInferReq (req_info, in, out, conn, infer_cb, uri);
      };
    } else if (uri.find ("/inferserver/set-interval") != std::string::npos) {
      /* Inference Specific */
      m_func[uri] =
          [inferserver_cb, uri] (const Json::Value & req_info,
          const Json::Value & in, Json::Value & out,
          struct mg_connection * conn) {
        return handleInferServerReq (req_info, in, out, conn, inferserver_cb,
            uri);
      };
    }
    else if (uri.find ("/conv/destcrop") != std::string::npos) {
      /* video convert Specific */
      m_func[uri] =
          [conv_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleConvReq (req_info, in, out, conn, conv_cb, uri);
      };
    } else if (uri.find ("/conv/srccrop") != std::string::npos) {
      /* video convert Specific */
      m_func[uri] =
          [conv_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleConvReq (req_info, in, out, conn, conv_cb, uri);
      };
    } else if (uri.find ("/conv/interpolation-method") != std::string::npos) {
      /* video convert Specific */
      m_func[uri] =
          [conv_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleConvReq (req_info, in, out, conn, conv_cb, uri);
      };
    } else if (uri.find ("/conv/flip-method") != std::string::npos) {
      /* video convert Specific */
      m_func[uri] =
          [conv_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleConvReq (req_info, in, out, conn, conv_cb, uri);
      };
    }
    else if (uri.find ("/mux/batched-push-timeout") != std::string::npos) {
      /* Mux Specific */
      m_func[uri] =
          [mux_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleMuxReq (req_info, in, out, conn, mux_cb, uri);
      };
    } else if (uri.find ("/mux/max-latency") != std::string::npos) {
      /* Mux Specific */
      m_func[uri] =
          [mux_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleMuxReq (req_info, in, out, conn, mux_cb, uri);
      };
    }
    else if (uri.find ("/osd/process-mode") != std::string::npos) {
      /* Osd Specific */
      m_func[uri] =
          [osd_cb, uri] (const Json::Value & req_info, const Json::Value & in,
          Json::Value & out, struct mg_connection * conn) {
        return handleOsdReq (req_info, in, out, conn, osd_cb, uri);
      };
    }
    else if (uri.find ("/app/quit") != std::string::npos) {
      /* App Instance Specific */
      m_func[uri] =
          [appinstance_cb, uri] (const Json::Value & req_info,
          const Json::Value & in, Json::Value & out,
          struct mg_connection * conn) {
        return handleAppReq (req_info, in, out, conn, appinstance_cb, uri);
      };
    }

  }

  std::map < std::string, NvDsRestServer::httpFunction >::iterator it
      = m_func.begin ();

  while (it != m_func.end ()) {
    httpServerHandler->addHandler (it->first, new RequestHandler (it->first,
            it->second));
    it++;
  }

  for (auto &it: server_cb->custom_cb_endpt){
    auto cb_func = it.second;
    m_func[it.first] =
          [cb_func] (const Json::Value & req_info, const Json::Value & in,
        Json::Value & out, struct mg_connection * conn) {
      return handleCustomReq (req_info, in, out, conn, cb_func);
    };
  }

for (auto it:m_func) {
    RequestHandler *reqHandle = new RequestHandler(it.first, it.second);
    httpServerHandler->requestList[it.first] = static_cast<void*>(reqHandle);
    httpServerHandler->addHandler (it.first, reqHandle);
  }

  std::cout << "Server running at port: " << server_config->port << "\n";

  return httpServerHandler;
}
