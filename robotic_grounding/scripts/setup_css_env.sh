#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Source this file to set CSS (PDX) credentials for list_css_sequences.py.
#
# Usage:
#   source scripts/setup_css_env.sh

export CSS_ENDPOINT_URL="https://pdx.s8k.io"
export CSS_ACCESS_KEY="v2p:AUTH_team-isaac"
# export CSS_SECRET_KEY="[API_KEY_HERE]"
export CSS_REGION="us-east-1"

echo "CSS environment configured."
