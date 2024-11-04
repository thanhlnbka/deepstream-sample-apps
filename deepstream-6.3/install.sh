################################################################################
# Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

#!/bin/bash

TARGET_DEVICE=$(uname -m)
OS=$(cat /etc/os-release | awk -F= '$1=="ID"{print $2}' | sed 's/"//g')

if [ "${TARGET_DEVICE}" = "x86_64" ]; then
    if [ "${OS}" = "rhel" ]; then
        mkdir -p /usr/lib/x86_64-linux-gnu/libv4l/plugins/
        ln -sf /opt/nvidia/deepstream/deepstream-6.3/lib/libv4l/plugins/libcuvidv4l2_plugin.so /usr/lib/x86_64-linux-gnu/libv4l/plugins/libcuvidv4l2_plugin.so
        BASE_LIB_DIR="/usr/lib64/"
    elif [ "${OS}" = "ubuntu" ]; then
        BASE_LIB_DIR="/usr/lib/x86_64-linux-gnu/"
    else
        echo "Unsupported OS" 2>&1
        exit 1
    fi
elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
    BASE_LIB_DIR="/usr/lib/aarch64-linux-gnu"
    if [ -d "$BASE_LIB_DIR/nvidia" ]; then
        NVIDIA_LIB_DIR="$BASE_LIB_DIR/nvidia"
    else
        NVIDIA_LIB_DIR="$BASE_LIB_DIR/tegra"
    fi
fi

while getopts "d:" option; do
    case "${option}" in
    d)
        BASE_LIB_DIR="${OPTARG}"
        ;;
    *)
        echo "ERROR! Unsupported option!"
        exit 1
        ;;
    esac
done

if [ -n $TARGET_DEVICE ]; then
    if [ "${TARGET_DEVICE}" = "x86_64" ]; then
        update-alternatives --install $BASE_LIB_DIR/gstreamer-1.0/deepstream deepstream-plugins /opt/nvidia/deepstream/deepstream-6.3/lib/gst-plugins 63
        update-alternatives --install $BASE_LIB_DIR/libv4l/plugins/libcuvidv4l2_plugin.so deepstream-v4l2plugin /opt/nvidia/deepstream/deepstream-6.3/lib/libv4l/plugins/libcuvidv4l2_plugin.so 63
        update-alternatives --install /usr/bin/deepstream-appsrc-cuda-test deepstream-appsrc-cuda-test /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-appsrc-cuda-test 63
        update-alternatives --install /usr/bin/deepstream-avsync-app deepstream-avsync-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-avsync-app 63
        update-alternatives --install /usr/bin/deepstream-multigpu-nvlink-test deepstream-multigpu-nvlink-test /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-multigpu-nvlink-test 63
        update-alternatives --install $BASE_LIB_DIR/libv4l2.so.0.0.99999 deepstream-v4l2library /opt/nvidia/deepstream/deepstream-6.3/lib/libnvv4l2.so 63
        update-alternatives --install $BASE_LIB_DIR/libv4lconvert.so.0.0.99999 deepstream-v4lconvert /opt/nvidia/deepstream/deepstream-6.3/lib/libnvv4lconvert.so 63
	update-alternatives --install /usr/bin/deepstream-ucx-test1-app-server deepstream-ucx-test1-app-server /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-ucx-test1-app-server 63
	update-alternatives --install /usr/bin/deepstream-ucx-test1-app-client deepstream-ucx-test1-app-client /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-ucx-test1-app-client 63
	update-alternatives --install /usr/bin/deepstream-ucx-test2-app-server deepstream-ucx-test2-app-server /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-ucx-test2-app-server 63
	update-alternatives --install /usr/bin/deepstream-ucx-test2-app-client deepstream-ucx-test2-app-client /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-ucx-test2-app-client 63
	update-alternatives --install /usr/bin/deepstream-ucx-test3-app-server deepstream-ucx-test3-app-server /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-ucx-test3-app-server 63
	update-alternatives --install /usr/bin/deepstream-ucx-test3-app-client deepstream-ucx-test3-app-client /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-ucx-test3-app-client 63
        update-alternatives --install /usr/bin/deepstream-3d-lidar-sensor-fusion deepstream-3d-lidar-sensor-fusion /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-3d-lidar-sensor-fusion 63
    elif [ "${TARGET_DEVICE}" = "aarch64" ]; then
        update-alternatives --install $BASE_LIB_DIR/gstreamer-1.0/deepstream deepstream-plugins /opt/nvidia/deepstream/deepstream-6.3/lib/gst-plugins 63
        ln -sf $NVIDIA_LIB_DIR/libnvbufsurface.so /opt/nvidia/deepstream/deepstream-6.3/lib/libnvbufsurface.so
        ln -sf $NVIDIA_LIB_DIR/libnvbufsurftransform.so /opt/nvidia/deepstream/deepstream-6.3/lib/libnvbufsurftransform.so
        ln -sf $NVIDIA_LIB_DIR/libnvdsbufferpool.so /opt/nvidia/deepstream/deepstream-6.3/lib/libnvdsbufferpool.so
        ln -sf $NVIDIA_LIB_DIR/libgstnvcustomhelper.so /opt/nvidia/deepstream/deepstream-6.3/lib/libgstnvcustomhelper.so
        echo "/opt/nvidia/deepstream/deepstream-6.3/lib" > /etc/ld.so.conf.d/deepstream.conf
        echo "/opt/nvidia/deepstream/deepstream-6.3/lib/gst-plugins" >> /etc/ld.so.conf.d/deepstream.conf
    fi
fi
update-alternatives --install /usr/bin/deepstream-app deepstream-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-app 63
update-alternatives --install /usr/bin/deepstream-audio deepstream-audio /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-audio 63
update-alternatives --install /usr/bin/deepstream-asr-app deepstream-asr-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-asr-app 63
update-alternatives --install /usr/bin/deepstream-asr-tts-app deepstream-asr-tts-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-asr-tts-app 63
update-alternatives --install /usr/bin/deepstream-test1-app deepstream-test1-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-test1-app 63
update-alternatives --install /usr/bin/deepstream-test2-app deepstream-test2-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-test2-app 63
update-alternatives --install /usr/bin/deepstream-test3-app deepstream-test3-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-test3-app 63
update-alternatives --install /usr/bin/deepstream-test4-app deepstream-test4-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-test4-app 63
update-alternatives --install /usr/bin/deepstream-test5-app deepstream-test5-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-test5-app 63
update-alternatives --install /usr/bin/deepstream-testsr-app deepstream-testsr-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-testsr-app 63
update-alternatives --install /usr/bin/deepstream-transfer-learning-app deepstream-transfer-learning-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-transfer-learning-app 63
update-alternatives --install /usr/bin/deepstream-user-metadata-app deepstream-user-metadata-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-user-metadata-app 63
update-alternatives --install /usr/bin/deepstream-dewarper-app deepstream-dewarper-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-dewarper-app 63
update-alternatives --install /usr/bin/deepstream-nvof-app deepstream-nvof-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-nvof-app 63
update-alternatives --install /usr/bin/deepstream-image-decode-app deepstream-image-decode-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-image-decode-app 63
update-alternatives --install /usr/bin/deepstream-gst-metadata-app deepstream-gst-metadata-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-gst-metadata-app 63
update-alternatives --install /usr/bin/deepstream-opencv-test deepstream-opencv-test /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-opencv-test 63
update-alternatives --install /usr/bin/deepstream-preprocess-test deepstream-preprocess-test /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-preprocess-test 63
update-alternatives --install /usr/bin/deepstream-segmentation-app deepstream-segmentation-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-segmentation-app 63
update-alternatives --install /usr/bin/deepstream-image-meta-test deepstream-image-meta-test /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-image-meta-test 63
update-alternatives --install /usr/bin/deepstream-appsrc-test deepstream-appsrc-test /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-appsrc-test 63
update-alternatives --install /usr/bin/deepstream-can-orientation-app deepstream-can-orientation-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-can-orientation-app 63
update-alternatives --install /usr/bin/deepstream-nvdsanalytics-test deepstream-nvdsanalytics-test /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-nvdsanalytics-test 63
update-alternatives --install /usr/bin/deepstream-3d-action-recognition deepstream-3d-action-recognition /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-3d-action-recognition 63
update-alternatives --install /usr/bin/deepstream-3d-depth-camera deepstream-3d-depth-camera /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-3d-depth-camera 63
update-alternatives --install /usr/bin/deepstream-lidar-inference-app deepstream-lidar-inference-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-lidar-inference-app 63
update-alternatives --install /usr/bin/deepstream-nmos-app deepstream-nmos-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-nmos-app 63
update-alternatives --install /usr/bin/deepstream-server-app deepstream-server-app /opt/nvidia/deepstream/deepstream-6.3/bin/deepstream-server-app 63
ldconfig
rm -rf /home/*/.cache/gstreamer-1.0/
rm -rf /root/.cache/gstreamer-1.0/
