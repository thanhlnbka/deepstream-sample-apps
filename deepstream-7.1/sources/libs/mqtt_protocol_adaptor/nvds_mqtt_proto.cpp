/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "mqtt_protocol.h"
#include "mosquitto.h"
#include <errno.h>
#include "glib.h"

#include <string.h>
#include <string>
#include <unordered_map>
#include "nvds_logger.h"
#include "nvds_msgapi.h"
#include "nvds_utils.h"
#include <sstream>
#include <vector>
#include <algorithm>
#include <thread>
#include <unistd.h>
#include <openssl/sha.h>
using namespace std;

#define MAX_FIELD_LEN 1024
#define UNUSED(A) (void)(A)
#define NVDS_MQTT_LOG_CAT "DSLOG:NVDS_MQTT_PROTO"
#define CONFIG_GROUP_MSG_BROKER_MQTT_USERNAME_CFG "username"
#define CONFIG_GROUP_MSG_BROKER_MQTT_PASSWORD_CFG "password"
#define CONFIG_GROUP_MSG_BROKER_MQTT_CLIENT_ID_CFG "client-id"
#define CONFIG_GROUP_MSG_BROKER_MQTT_LOOP_TIMEOUT_CFG "loop-timeout"
#define CONFIG_GROUP_MSG_BROKER_MQTT_KEEP_ALIVE_CFG "keep-alive"
#define CONFIG_GROUP_MSG_BROKER_MQTT_CAFILE_CFG "tls-cafile"
#define CONFIG_GROUP_MSG_BROKER_MQTT_CAPATH_CFG "tls-capath"
#define CONFIG_GROUP_MSG_BROKER_MQTT_CERTFILE_CFG "tls-certfile"
#define CONFIG_GROUP_MSG_BROKER_MQTT_KEYFILE_CFG "tls-keyfile"

#define NVDS_MSGAPI_VERSION "4.0"
#define NVDS_MSGAPI_PROTOCOL "MQTT"

#define DEFAULT_LOOP_TIMEOUT 2000
#define DEFAULT_KEEP_ALIVE 60

#define PASSWORD_VAR "PASSWORD_MQTT"
#define USERNAME_VAR "USER_MQTT"

/* Message details:
 * send_callback = user callback func
 * user_ptr = user pointer passed by async send
 */
struct send_msg_info_t {
  nvds_msgapi_send_cb_t send_callback;
  void *user_ptr;
};

/* Details of mqtt connection handle:
 * mosq : mosquitto client object
 * sub_callback : user subscription callback func
 * connect_cb : user connection callback func 
 * user_ctx : user pointer passed by sub
 * username: username for login to server
 * password: password for login to server
 * client_id: name of MQTT client
 * loop_timeout : time in ms for the call to loop to wait for network activity
 * keep_alive : number of seconds after which broker should send PING if no messages have been exchanged
 * subscription_on : Flag to check if subscription is ON
 * send_msg_info_map : map message info to id assigned by mosquitto broker
 * enable_tls : flag to check if TLS encryption is enabled by the broker
 * cafile : path to a TLS certificate authority file
 * capath : path to a directory containing TLS CA files
 * certfile : path to the client TLS certificate file
 * keyfile : path to the client TLS key file
 */
typedef struct {
   struct mosquitto *mosq = NULL;
   nvds_msgapi_subscribe_request_cb_t  sub_callback;
   nvds_msgapi_connect_cb_t connect_cb;
   void* user_ctx;
   char username[MAX_FIELD_LEN] = {0};
   char password[MAX_FIELD_LEN] = {0};
   char client_id[MAX_FIELD_LEN] = {0};
   int loop_timeout = DEFAULT_LOOP_TIMEOUT;
   int keep_alive = DEFAULT_KEEP_ALIVE;
   bool subscription_on = false;
   std::unordered_map<int , send_msg_info_t> send_msg_info_map;
   bool enable_tls = false;
   char cafile[MAX_FIELD_LEN] = {0};
   char capath[MAX_FIELD_LEN] = {0};
   char certfile[MAX_FIELD_LEN] = {0};
   char keyfile[MAX_FIELD_LEN] = {0};
} NvDsMqttClientHandle;

NvDsMsgApiErrorType nvds_mqtt_read_config(NvDsMqttClientHandle *mh, char *config_path);
void mosq_mqtt_log_callback(struct mosquitto *mosq, void *obj, int level, const char *str);
void my_disconnect_callback(struct mosquitto *mosq, void *obj, int rc, const mosquitto_property *properties);
void my_connect_callback(struct mosquitto *mosq, void *obj, int result, int flags, const mosquitto_property *properties);
void my_publish_callback(struct mosquitto *mosq, void *obj, int mid, int reason_code, const mosquitto_property *properties);
bool is_valid_mqtt_connection_str(char *connection_str, string &burl, string &bport);

/* Parse the config file into the mosquitto handle
*/
NvDsMsgApiErrorType nvds_mqtt_read_config(NvDsMqttClientHandle *mh, char *config_path)
{
    char cfg_uname[MAX_FIELD_LEN]="", cfg_pwd[MAX_FIELD_LEN]="", cfg_client_id[MAX_FIELD_LEN]="";
    char cfg_timeout[MAX_FIELD_LEN]="", cfg_keep_alive[MAX_FIELD_LEN]="";

    const gchar* var_user = NULL;
    const gchar* var_pwd = NULL;

    // Read config to fetch username -- to be deprecated in favor of env variable
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_USERNAME_CFG, cfg_uname, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_uname,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->username, cfg_uname, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT username = %s \n", mh->username);
    }
    else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Username not specified in cfg.\n");
    }
    // Read config to fetch password -- to be deprecated in favor of env variable
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_PASSWORD_CFG, cfg_pwd, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_pwd,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->password, cfg_pwd, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT password = %s \n", mh->password);
    }
    else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Password not specified in cfg.\n");
    }
    // Read config to fetch client ID
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_CLIENT_ID_CFG, cfg_client_id, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_client_id,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->client_id, cfg_client_id, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT Client ID = %s \n", mh->client_id);
    }
    else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Client ID not specified in cfg.\n");
    }
    // Read config to fetch loop_timeout value. Uses default if not present.
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_LOOP_TIMEOUT_CFG, cfg_timeout, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_timeout,"", MAX_FIELD_LEN)) {
        string timeout(cfg_timeout);
        try {
          mh->loop_timeout = stoi(timeout);
        } catch (const std::invalid_argument & e) {
          mh->loop_timeout = DEFAULT_LOOP_TIMEOUT;
          g_print("parse_config: loop_timeout stoi exception %s. Using default of %d\n", e.what(), mh->loop_timeout);
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "parse_config: loop_timeout stoi exception %s. Using default of %d\n", e.what(), mh->loop_timeout);
        }
        catch (const std::out_of_range & e) {
          mh->loop_timeout = DEFAULT_LOOP_TIMEOUT;
          g_print("parse_config: loop_timeout stoi exception %s. Using default of %d\n", e.what(), mh->loop_timeout);
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "parse_config: loop_timeout stoi exception %s. Using default of %d\n", e.what(), mh->loop_timeout);
        }
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "Loop timeout = %d \n", mh->loop_timeout);
    }
    else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Loop timeout not specified in cfg. Using default of %d\n", mh->loop_timeout);
    }
    // Read config to fetch keep-alive value. Uses default if not present.
    if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_KEEP_ALIVE_CFG, cfg_keep_alive, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_keep_alive,"", MAX_FIELD_LEN)) {
        string keep_alive(cfg_keep_alive);
        try {
          mh->keep_alive = stoi(keep_alive);
        } catch (const std::invalid_argument & e) {
          mh->keep_alive = DEFAULT_KEEP_ALIVE;
          g_print("parse_config: keep_alive stoi exception %s. Using default of %d\n", e.what(), mh->keep_alive);
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "parse_config: keep_alive stoi exception %s. Using default of %d\n", e.what(), mh->keep_alive);
        }
        catch (const std::out_of_range & e) {
          mh->keep_alive = DEFAULT_KEEP_ALIVE;
          g_print("parse_config: keep_alive stoi exception %s. Using default of %d\n", e.what(), mh->keep_alive);
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "parse_config: keep_alive stoi exception %s. Using default of %d\n", e.what(), mh->keep_alive);
        }
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "keep_alive = %d \n", mh->keep_alive);
    }
    else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Keep alive interval not specified in cfg. Using default of %d\n", mh->keep_alive);
    }
    // Read config file to check if TLS is enabled
    char enable_tls[16]="";
    if(fetch_config_value(config_path, "enable-tls", enable_tls, sizeof(enable_tls), NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) {
        if (strncmp(enable_tls, "1", 1) == 0) {
          mh->enable_tls = true;
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "TLS encryption enabled");
        }
        else {
          mh->enable_tls = false;
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "TLS encryption disabled");
        }
    }
    else {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "parse_config: Error parsing mqtt enable-tls config option. Disabled by default.");
    }
    // If TLS is enabled, read config to fetch TLS CA, cert, and key file paths
    if (mh->enable_tls) {
      char cfg_cafile[MAX_FIELD_LEN]="", cfg_capath[MAX_FIELD_LEN]="", cfg_certfile[MAX_FIELD_LEN]="", cfg_keyfile[MAX_FIELD_LEN]="";
      if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_CAFILE_CFG, cfg_cafile, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_cafile,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->cafile, cfg_cafile, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS CA certificate filepath = %s \n", mh->cafile);
      }
      else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "CA certificate filepath is not provided\n");
      }

      if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_CAPATH_CFG, cfg_capath, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_capath,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->capath, cfg_capath, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS CA certificate directory path = %s \n", mh->capath);
      }
      else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "CA certificate directory path is not provided\n");
      }

      if (strncmp(mh->cafile,"", MAX_FIELD_LEN) == 0 && strncmp(mh->capath,"", MAX_FIELD_LEN) == 0) {
        g_print("parse_config: Error: TLS encyption is enabled but neither CA certificate filepath nor directory path were provided\n");
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR,  "parse_config: Error: TLS encyption is enabled but neither CA certificate filepath nor directory path were provided\n");
        return NVDS_MSGAPI_ERR;
      }

      if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_CERTFILE_CFG, cfg_certfile, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_certfile,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->certfile, cfg_certfile, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS client certificate filepath = %s \n", mh->certfile);
      }
      else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS client certificate not provided.\n");
      }

      if((fetch_config_value(config_path, CONFIG_GROUP_MSG_BROKER_MQTT_KEYFILE_CFG, cfg_keyfile, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) == NVDS_MSGAPI_OK) && strncmp(cfg_keyfile,"", MAX_FIELD_LEN)) {
        g_strlcpy(mh->keyfile, cfg_keyfile, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS client key filepath = %s \n", mh->keyfile);
      }
      else {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT TLS client key not provided.\n");
      }
    }

    // Fetch username and password from environmental variable
    var_user = g_getenv(USERNAME_VAR);
    var_pwd = g_getenv(PASSWORD_VAR);
    // Check that username and password do not exceed maximum length
    // Override user and password with environmental variable if given
    if (var_user != NULL) {
      if (MAX_FIELD_LEN < (int) strlen(var_user) + 1) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Username length exceeds capacity of %d", MAX_FIELD_LEN);
        return NVDS_MSGAPI_ERR;
      }
      else{
        g_strlcpy(mh->username, var_user, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT username = %s \n", mh->username);
      }
    } 
    else {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Username not provided through environmental variable.\n");
    }

    if (var_pwd != NULL) {
      if(MAX_FIELD_LEN < (int) strlen(var_pwd) + 1) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Password length exceeds capacity of %d", MAX_FIELD_LEN);
        return NVDS_MSGAPI_ERR;
      }
      else{
        g_strlcpy(mh->password, var_pwd, MAX_FIELD_LEN);
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "MQTT password = %s \n", mh->password);
      }
    }
    else {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Password not provided through environmental variable.\n");
    }

  return NVDS_MSGAPI_OK;
}

/* Mosquitto log callback
*/
void mosq_mqtt_log_callback(struct mosquitto *mosq, void *obj, int level, const char *str)
{
  UNUSED(mosq);
  UNUSED(obj);
  UNUSED(level);

  g_print("[%s] %s\n", __func__, str);
  nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "[%s] %s\n", __func__, str);
}

/* Mosquitto disconnect callback, called when disconnect is received from the broker
*/
void my_disconnect_callback(struct mosquitto *mosq, void *obj, int rc, const mosquitto_property *properties)
{
  UNUSED(mosq);
  UNUSED(obj);
  UNUSED(rc);
  UNUSED(properties);

  if(rc == 0){
    g_print("mqtt disconnected\n");
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "mqtt disconnected\n");
  }
  else {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "mqtt disconnect failed rc=%d\n", rc);
  }
}

/* Mosquitto connect callback, called when CONNACK is recieved from the broker
*/
void my_connect_callback(struct mosquitto *mosq, void *obj, int result, int flags, const mosquitto_property *properties)
{
  UNUSED(obj);
  UNUSED(flags);
  UNUSED(properties);
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)mosquitto_userdata(mosq);

  // When connection succeeds, call the user connect callback with success flag
  if(!result){
    g_print("mqtt connection success; ready to send data\n");
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "mqtt connection success; ready to send data\n");
    if (mqttH->connect_cb)
      mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SUCCESS);
  // If connection failed, call the user connect callback with failure flag
  }else{
    if(result){
      { //we are always using MQTT v5
        if(result == MQTT_RC_UNSUPPORTED_PROTOCOL_VERSION){
          g_print("Connection error: %s. Try connecting to an MQTT v5 broker; Other versions are not supported\n", mosquitto_reason_string(result));
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Connection error: %s. Try connecting to an MQTT v5 broker; Other versions are not supported\n", mosquitto_reason_string(result));
        }else{
          g_print("Connection error: %s\n", mosquitto_reason_string(result));
          nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Connection error: %s\n", mosquitto_reason_string(result));
        }
        if (mqttH->connect_cb)
          mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
      }
    }
  }
}

/* Mosquitto publish callback, called on PUBACK from broker
*/
void my_publish_callback(struct mosquitto *mosq, void *obj, int mid, int reason_code, const mosquitto_property *properties)
{
  UNUSED(obj);
  UNUSED(properties);

  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)mosquitto_userdata(mosq);

  g_print("Publish callback with reason code: %s.\n", mosquitto_reason_string(reason_code));
  nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "Publish callback with reason code: %s.\n", mosquitto_reason_string(reason_code));

  // Retrieve the message info corresponding to the message id
  std::unordered_map<int , send_msg_info_t>::iterator it = mqttH->send_msg_info_map.find(mid);
  if(it != mqttH->send_msg_info_map.end()) {
    send_msg_info_t& msg_info = it->second;
    // Call the user send callback function associated with the message ID to communicate publish status
    //MQTTv5 reason code check:
    if(reason_code > 127){
      g_print("Warning: Publish %d failed: %s.\n", mid, mosquitto_reason_string(reason_code));
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Warning: Publish %d failed: %s.\n", mid, mosquitto_reason_string(reason_code));
      msg_info.send_callback(msg_info.user_ptr, NVDS_MSGAPI_ERR);
    } else {
      msg_info.send_callback(msg_info.user_ptr, NVDS_MSGAPI_OK);
    }
    // Send is now complete, remove the msg_info from the map
    mqttH->send_msg_info_map.erase(it);
  }
}

/* Mosquitto message callback, called when a message is received from the broker
*/
static void my_message_callback(struct mosquitto *mosq, void *obj, const struct mosquitto_message *message, const mosquitto_property *properties)
{
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)mosquitto_userdata(mosq);
  // Call the user sub callback with the message and user pointer
  if(mqttH->sub_callback) {
    mqttH->sub_callback(NVDS_MSGAPI_OK, message->payload,message->payloadlen, message->topic, mqttH->user_ctx);
  }
}

/* Mosquitto subscribe callback, called when SUBACK is received from the broker
*/
static void my_subscribe_callback(struct mosquitto *mosq, void *obj, int mid, int qos_count, const int *granted_qos, const mosquitto_property *properties)
{
  int i;
  UNUSED(obj);
  for(i=0; i<qos_count; i++){
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "Subscription granted qos %d", granted_qos[i]);
  }
  return;
}

/*
 * Validate mqtt connection string format
 * Valid format host;port or (host;port;topic to support backward compatibility)
 */
bool is_valid_mqtt_connection_str(char *connection_str, string &burl, string &bport) {
  if(connection_str == NULL) {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "mqtt connection string cant be NULL");
    return false;
  }

  string str(connection_str);
  size_t n = count(str.begin(), str.end(), ';');
  if(n>2) {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "mqtt connection string format is invalid");
    return false;
  }

  istringstream iss(connection_str);
  getline(iss, burl, ';');
  getline(iss, bport,';');

  if(burl =="" || bport == "") {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "mqtt connection string is invalid. hostname or port is empty\n");
      return false;
  }
  return true;
}

/* nvds_msgapi function for connect to broker
*/
NvDsMsgApiHandle nvds_msgapi_connect(char *connection_str,
        nvds_msgapi_connect_cb_t connect_cb, char *config_path)
{
  int rc;
  string burl="", bport="";
  int port = 0;
  char* err;

  // Retrieve host and port from connection string
  if(!is_valid_mqtt_connection_str(connection_str, burl, bport))
    return NULL;

  try {
    port = stoi(bport);
  } catch (const std::invalid_argument & e) {
    g_print("stoi exception %s\n", e.what());
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "stoi exception %s\n", e.what());
  }
  catch (const std::out_of_range & e) {
    g_print("stoi exception %s\n", e.what());
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "stoi exception %s\n", e.what());
  }

  if(port <= 0)
  {
    return NULL;
  }

  // Mosquitto library initialization, must be called before any other mosquitto function
  mosquitto_lib_init();
  // Create handle
  NvDsMqttClientHandle* mqttH = new NvDsMqttClientHandle;
  // Set user connect callback
  mqttH->connect_cb = connect_cb;
  // Read config
  if (config_path) {
    if(nvds_mqtt_read_config(mqttH, config_path) == NVDS_MSGAPI_ERR) {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "MQTT config parsing failed. Connection failed \n");
      delete mqttH;
      return NULL;
    }
  }
  if (mqttH->client_id[0] != 0) // If client ID is set
    mqttH->mosq = mosquitto_new(mqttH->client_id, true, NULL); // Instantiate client
  else
    mqttH->mosq = mosquitto_new(NULL, true, NULL); // Instantiate client with NULL client ID -- random client ID will be generated
  if(!mqttH->mosq){
    switch(errno){
      case ENOMEM:
        g_print("Error: Out of memory.\n");
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error: Out of memory.\n");
        break;
      case EINVAL:
        g_print("Error: Invalid id.\n");
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error: Invalid id.\n");
        break;
    }
    delete mqttH;
    return NULL;
  }

  // Set TLS if enabled
  if (mqttH->enable_tls) {
    // Set cafile and capath to null if empty
    char* cafile = (mqttH->cafile[0] != 0) ? (char*)mqttH->cafile : NULL;
    char* capath = (mqttH->capath[0] != 0) ? (char*)mqttH->capath : NULL;
    // certfile and keyfile must both be provided or both be NULL
    char* certfile = (mqttH->certfile[0] != 0 && mqttH->keyfile[0] != 0) ? (char*)mqttH->certfile : NULL;
    char* keyfile = (mqttH->certfile[0] != 0 && mqttH->keyfile[0] != 0) ? (char*)mqttH->keyfile : NULL;
    rc = mosquitto_tls_set(mqttH->mosq, cafile, capath, certfile, keyfile, NULL);
    if(rc){
      if(rc == MOSQ_ERR_INVAL){
        g_print("Error: Problem setting TLS options: File not found.\n");
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error: Problem setting TLS options: File not found.\n");
			}
      else{
        g_print("Error: Problem setting TLS options: %s.\n", mosquitto_strerror(rc));
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error: Problem setting TLS options: %s.\n", mosquitto_strerror(rc));
      }
      delete mqttH;
      return NULL;
    }
  }

  // Start thread for continuous calls to mosquitto_loop, managed by mosquitto lib
  rc = mosquitto_loop_start(mqttH->mosq);
  if(rc>0){
    printf("error=%d\n", rc);
    if(rc == MOSQ_ERR_ERRNO){
      err = strerror(errno);
      g_print("Error: %s\n", err);
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR,  "Error: %s\n", err);

    }else{
      g_print("Unable to start mosquitto loop: (%s).\n", mosquitto_strerror(rc));
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR,  "Unable to start mosquitto loop: (%s).\n", mosquitto_strerror(rc));
    }
    if (mqttH->connect_cb)
      mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
    delete mqttH;
    return NULL;
  }

  mosquitto_user_data_set(mqttH->mosq, (void*)mqttH); // Set user data so handle can be retrieved in mosquitto callbacks
  // Set mosquitto callbacks
  mosquitto_log_callback_set(mqttH->mosq, mosq_mqtt_log_callback);
  mosquitto_connect_v5_callback_set(mqttH->mosq, my_connect_callback);
  mosquitto_disconnect_v5_callback_set(mqttH->mosq, my_disconnect_callback);
  mosquitto_publish_v5_callback_set(mqttH->mosq, my_publish_callback);
  mosquitto_subscribe_v5_callback_set(mqttH->mosq, my_subscribe_callback);
  mosquitto_message_v5_callback_set(mqttH->mosq, my_message_callback);

  // Authenticate with user credentials if provided
  if (strncmp(mqttH->username,"", MAX_FIELD_LEN) || strncmp(mqttH->password, "", MAX_FIELD_LEN)) {
    if(mosquitto_username_pw_set(mqttH->mosq, mqttH->username, mqttH->password) == MOSQ_ERR_SUCCESS) {
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO,  "Successfully set MQTT username = %s and password = %s\n", mqttH->username, mqttH->password);
    }
  }
  // Call connect
  rc = mosquitto_connect_bind_v5(mqttH->mosq, burl.c_str(), port, mqttH->keep_alive, NULL, NULL);
  if(rc>0){
    printf("error=%d\n", rc);
    if(rc == MOSQ_ERR_ERRNO){
      err = strerror(errno);
      g_print("Error: %s\n", err);
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR,  "Error: %s\n", err);

    }else{
      g_print("Unable to connect (%s).\n", mosquitto_strerror(rc));
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR,  "Unable to connect (%s).\n", mosquitto_strerror(rc));
    }
    if (mqttH->connect_cb)
      mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
    delete mqttH;
    return NULL;
  }

  return (NvDsMsgApiHandle)mqttH;
}

/* nvds_msgapi function for synchronous send, not supported for MQTT
*/
NvDsMsgApiErrorType nvds_msgapi_send(NvDsMsgApiHandle h_ptr, char *topic, const uint8_t *payload, size_t nbuf)
{
  g_print("nvds_msgapi_send() is not currently supported for MQTT.");
  nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR,  "nvds_msgapi_send() is not currently supported for MQTT.");
  return NVDS_MSGAPI_ERR;
}

/* nvds_msgapi function for asynchronous send
*/
NvDsMsgApiErrorType nvds_msgapi_send_async(NvDsMsgApiHandle h_ptr, char  *topic, const uint8_t *payload, size_t nbuf, nvds_msgapi_send_cb_t send_callback, void *user_ptr)
{
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)(h_ptr);
  int rc = MOSQ_ERR_SUCCESS;
  
  // Create msg_info from the user send callback and pointer
  struct send_msg_info_t msg_info = {send_callback, user_ptr};

  // message id, will be assigned by call to publish
  int mid;

  if(mqttH == NULL)
  {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "mqtt connection handle passed for nvds_msgapi_send_async() = NULL. Send failed\n");
    return NVDS_MSGAPI_ERR;
  }
  if(!topic)
  {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Topic not specified for send. Send failed\n");
    return NVDS_MSGAPI_ERR;
  }
  if(!payload || nbuf <= 0)
  {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Message not specified for send. Send failed\n");
    return NVDS_MSGAPI_ERR;
  }
  if(nbuf <= 0)
  {
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Invalid message size (<=0) for send. Send failed\n");
    return NVDS_MSGAPI_ERR;
  }

  // Call mosquitto publish
  rc = mosquitto_publish_v5(mqttH->mosq, &mid, topic, nbuf, payload, 0, false, NULL);
  if(rc != MOSQ_ERR_SUCCESS){
    g_print("Error sending repeat publish: %s", mosquitto_strerror(rc));
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error sending repeat publish: %s\n", mosquitto_strerror(rc));
    if (mqttH->connect_cb && rc == MOSQ_ERR_NO_CONN) {
      mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
    }
  }
  // Map msg_info to message id
  else
    mqttH->send_msg_info_map[mid] = msg_info;

  return (rc == MOSQ_ERR_SUCCESS) ? NVDS_MSGAPI_OK : NVDS_MSGAPI_ERR;
}

/* nvds_msgapi function for subscribe to topics
*/
NvDsMsgApiErrorType nvds_msgapi_subscribe (NvDsMsgApiHandle h_ptr, char ** topics, int num_topics, nvds_msgapi_subscribe_request_cb_t  cb, void *user_ctx)
{
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle *)h_ptr;
    if(mqttH == NULL ) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "mqtt connection handle passed for nvds_msgapi_subscribe() = NULL. Subscribe failed\n");
        return NVDS_MSGAPI_ERR;
    }

    if(topics == NULL || num_topics <=0 ) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Topics not specified for subscribe. Subscription failed\n");
        return NVDS_MSGAPI_ERR;
    }

    if(!cb) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Subscribe callback cannot be NULL. Subscription failed\n");
        return NVDS_MSGAPI_ERR;
    }

    if(mqttH->subscription_on) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "MQTT Subscription already exists. Ignoring this subscription call\n\n");
        return NVDS_MSGAPI_ERR;
    }

  // Set user callback and pointer
  mqttH->sub_callback = cb;
  mqttH->user_ctx = user_ctx;

  // Subscribe to each topic
  for (int i = 0 ; i < num_topics ; i++){
    int rc;
    // Subscribe with QoS 0 and default options
    rc = mosquitto_subscribe_v5(mqttH->mosq, NULL, topics[i], 0, 0, NULL);
    if(rc != MOSQ_ERR_SUCCESS) {
      g_print("Error subscribing: %s", mosquitto_strerror(rc));
      nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "mqtt: Failed to subscribe to topic %s. Error: %s\n", string(topics[i]), mosquitto_strerror(rc));
      if (mqttH->connect_cb && (rc == MOSQ_ERR_NO_CONN || rc == MOSQ_ERR_CONN_LOST)) {
          mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
      }
      return NVDS_MSGAPI_ERR;
    }
  }

  mqttH->subscription_on=true;
  return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for performing work on thread in nvmsgbroker plugin. In the case of mqtt, we call mosquitto_loop to send outgoing messages
* and read incoming messages
*/
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr)
{
  int rc;
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)(h_ptr);
  // mosquitto_loop is called to process incoming data and send outgoing messages
  rc = mosquitto_loop(mqttH->mosq, mqttH->loop_timeout, 1);
  if (mqttH->connect_cb && (rc == MOSQ_ERR_NO_CONN || rc == MOSQ_ERR_CONN_LOST)) {
    mqttH->connect_cb(mqttH, NVDS_MSGAPI_EVT_SERVICE_DOWN);
  }
}

/* nvds_msgapi function for disconnect from broker
*/
NvDsMsgApiErrorType nvds_msgapi_disconnect(NvDsMsgApiHandle h_ptr)
{
  int rc;
  NvDsMqttClientHandle* mqttH = (NvDsMqttClientHandle*)(h_ptr);
  // Call disconnect
  rc = mosquitto_disconnect_v5(mqttH->mosq, 0, NULL);
  if(rc>0){
    g_print("Error disconnecting: (%s).\n", mosquitto_strerror(rc));
    nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR, "Error disconnecting: (%s).\n", mosquitto_strerror(rc));
  }

  mosquitto_loop(mqttH->mosq, mqttH->loop_timeout, 1); // Make sure disconnect message is sent out before stopping loop
  // mosquitto_loop_stop must be called AFTER mosquitto_disconnect
  mosquitto_loop_stop(mqttH->mosq, false);
  // Standard mosquitto clean-up procedure
  mosquitto_destroy(mqttH->mosq);
  mosquitto_lib_cleanup();
  delete mqttH;
  return NVDS_MSGAPI_OK;
}

/* nvds_msgapi function for retrieving nvds_msgapi version
*/
char *nvds_msgapi_getversion() {
  return (char *) NVDS_MSGAPI_VERSION;
}

/* nvds_msgapi function for retrieving protocol name
*/
char *nvds_msgapi_get_protocol_name(void) {
  return (char *)NVDS_MSGAPI_PROTOCOL;
}

/* nvds_msgapi function for retrieving connection signature, used for connection sharing
*/
NvDsMsgApiErrorType nvds_msgapi_connection_signature(char *broker_str, char *cfg, char *output_str, int max_len) {
  g_strlcpy(output_str,"", max_len);

    //check if share-connection config option is turned ON
    char reuse_connection[16]="";
    if(fetch_config_value(cfg, "share-connection", reuse_connection, sizeof(reuse_connection), NVDS_MQTT_LOG_CAT) != NVDS_MSGAPI_OK) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing mqtt share-connection config option");
        return NVDS_MSGAPI_ERR;
    }
    if(strncmp(reuse_connection, "1", 1)) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_INFO, "nvds_msgapi_connection_signature: mqtt connection sharing disabled. Hence connection signature cant be returned");
        return NVDS_MSGAPI_OK;
    }

    if(broker_str == NULL && cfg == NULL) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Both mqtt broker_str and path to cfg cant be NULL");
        return NVDS_MSGAPI_ERR;
    }
    if(max_len < 2 * SHA256_DIGEST_LENGTH + 1) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: output string length allocated not sufficient");
        return NVDS_MSGAPI_ERR;
    }

    char mqtt_connection_str[MAX_FIELD_LEN]="";
    char uname[MAX_FIELD_LEN]="", pwd[MAX_FIELD_LEN]="";

    string burl="", bport="";
    if(!is_valid_mqtt_connection_str(broker_str, burl, bport)) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Invalid connection string");
        return NVDS_MSGAPI_ERR;
    }

    if((fetch_config_value(cfg, CONFIG_GROUP_MSG_BROKER_MQTT_USERNAME_CFG, uname, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) != NVDS_MSGAPI_OK)) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing mqtt username");
    }

    if((fetch_config_value(cfg, CONFIG_GROUP_MSG_BROKER_MQTT_PASSWORD_CFG, pwd, MAX_FIELD_LEN, NVDS_MQTT_LOG_CAT) != NVDS_MSGAPI_OK)) {
        nvds_log(NVDS_MQTT_LOG_CAT, LOG_ERR , "nvds_msgapi_connection_signature: Error parsing mqtt password");
    }

    // Generate connection signature from connection string and user credentials
    g_snprintf(mqtt_connection_str, sizeof (mqtt_connection_str), "%s;%s;%s;%s", burl.c_str(),bport.c_str(),uname,pwd);
    string hashval = generate_sha256_hash(string(mqtt_connection_str, max_len));
    g_strlcpy(output_str, hashval.c_str(), max_len);
    return NVDS_MSGAPI_OK;
}
