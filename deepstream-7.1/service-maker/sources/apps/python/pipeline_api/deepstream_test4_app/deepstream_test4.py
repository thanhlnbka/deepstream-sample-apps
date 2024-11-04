#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from pyservicemaker import Pipeline
import sys
import platform
import os

PIPELINE_NAME = "deepstream-test4"
CONFIG_FILE_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_pgie_config.yml"
CONFIG_MSGCONV_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-test4/dstest4_msgconv_config.yml"
KAFKA_PROTO_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_kafka_proto.so"
MSGBROKER_CONN_STR = "localhost;9092"
MGSBROKER_TOPIC = "test4app"

def main(file_path):
    file_ext = os.path.splitext(file_path)[1]

    if file_ext in [".yaml", ".yml"]:
        Pipeline(PIPELINE_NAME, file_path).start().wait()
    else:
        (Pipeline(PIPELINE_NAME).add("filesrc", "src", {"location": file_path}).add("h264parse", "parser").add("nvv4l2decoder", "decoder")
            .add("nvstreammux", "mux", {"batch-size": 1, "width": 1280, "height": 720})
            .add("nvinfer", "infer", {"config-file-path": CONFIG_FILE_PATH})
            .add("nvosdbin", "osd").add("tee", "tee").add("queue", "queue1").add("queue", "queue2")
            .add("nvmsgconv", "msgconv", {"config": CONFIG_MSGCONV_PATH})
            .add("nvmsgbroker", "msgbroker", {"conn-str": MSGBROKER_CONN_STR, "proto-lib": KAFKA_PROTO_LIB, "sync": False, "topic": MGSBROKER_TOPIC})
            .add("nv3dsink" if platform.processor() == "aarch64" else "nveglglessink", "sink")
            .link("src", "parser", "decoder").link(("decoder", "mux"), ("", "sink_%u")).link("mux","infer", "osd", "tee", "queue1", "sink")
            .link("tee", "queue2", "msgconv", "msgbroker")
            .attach("osd", "add_message_meta_probe", "metadata generator")
            .start().wait())

if __name__ == '__main__':
    # Check input arguments
    if len(sys.argv) != 2:
        sys.stderr.write("usage: %s <H264 filename> OR <YAML config file>\n" % sys.argv[0])
        sys.exit(1)

    main(sys.argv[1])