################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################

# this Makefile is to be used to build the mqtt_proto protocol adaptor .so
CXX:=g++

PKGS:= glib-2.0 

DS_LIB:=/opt/nvidia/deepstream/deepstream/lib/

SRCS:=  nvds_mqtt_proto.cpp ../nvds_msgapi_common_src/nvds_utils.cpp
TARGET_LIB:= libnvds_mqtt_proto.so

CFLAGS:= -fPIC -Wall

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

LIBS:= $(shell pkg-config --libs $(PKGS))
LDFLAGS:= -shared

DS_INC:= ../../includes
MOSQ_INC:=/usr/local/include/

INC_PATHS:= -I $(DS_INC) -I $(MOSQ_INC)
CFLAGS+= $(INC_PATHS)

LIBS+= -L/$(DS_LIB) -lnvds_logger -lmosquitto -lpthread -lglib-2.0 -lcrypto
LDFLAGS+= -Wl,-rpath=$(DS_LIB)

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCS)
	$(CXX) -o $@ $^ $(CFLAGS) $(LIBS) $(LDFLAGS)

install: $(TARGET_LIB)
	cp -rv $(TARGET_LIB) $(DS_LIB)

clean:
	rm -rf $(TARGET_LIB)