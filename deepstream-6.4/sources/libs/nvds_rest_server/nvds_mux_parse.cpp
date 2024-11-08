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
#include "nvds_rest_server.h"
#include "nvds_parse.h"

#define EMPTY_STRING ""

bool
nvds_rest_mux_parse (const Json::Value & in, NvDsServerMuxInfo * mux_info)
{
  if (mux_info->uri.find ("/api/v1/") != std::string::npos) {
    for (Json::ValueConstIterator it = in.begin (); it != in.end (); ++it) {

      std::string root_val = it.key ().asString ().c_str ();
      mux_info->root_key = root_val;

      const Json::Value sub_root_val = in[root_val];      //object values of root_key

      if (mux_info->mux_flag == BATCHED_PUSH_TIMEOUT) {
        mux_info->batched_push_timeout =
            sub_root_val.get ("batched_push_timeout", -1).asInt ();
        if (mux_info->batched_push_timeout < -1
            || mux_info->batched_push_timeout > INT_MAX) {
          mux_info->mux_log =
              "BATCHED_PUSH_TIMEOUT_UPDATE_FAIL, batched_push_timeout value not parsed correctly,  Range: -1 - 2147483647";
          mux_info->status = BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
          mux_info->err_info.code = StatusBadRequest;
          return false;
        }
      }
      if (mux_info->mux_flag == MAX_LATENCY) {
        mux_info->max_latency = sub_root_val.get ("max_latency", 0).asUInt ();
      }
    }
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return true;
}
