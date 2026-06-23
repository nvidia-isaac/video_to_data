#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
#
# Shared SSH setup for osmo workflows.
# Expects osmo credentials at /tmp/.ssh (set via `credentials: sshd-keys`).

set -ex

# Install sshd
apt update
apt install -y openssh-server

# Configure SSH to allow root login
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Setup SSH keys for root
mkdir -p /root/.ssh
chmod 700 /root/.ssh

# Setup authorized keys to allow logging in with key
CLIENT_PUBLIC_KEY=$(cat /tmp/.ssh/client-public-key)
echo "$CLIENT_PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# Setup sshd server private key so the client will recognize it
cp /tmp/.ssh/server-private-key /etc/ssh/ssh_host_ed25519_key
cp /tmp/.ssh/server-public-key /etc/ssh/ssh_host_ed25519_key.pub
chmod 600 /etc/ssh/ssh_host_ed25519_key
chmod 644 /etc/ssh/ssh_host_ed25519_key.pub

# Start sshd
mkdir -p /run/sshd
/usr/sbin/sshd

echo "SSH server started - root login enabled"
