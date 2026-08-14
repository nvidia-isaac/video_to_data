# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

from isaaclab.app import AppLauncher

# Launch the simulator in headless mode for unit tests
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
APP_IS_READY = True

if __name__ == "__main__":
    unittest.main(exit=False)
    simulation_app.close()
