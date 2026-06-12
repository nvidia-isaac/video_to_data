# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Code adapted from https://github.com/stack-of-tasks/pinocchio/blob/devel/bindings/python/pinocchio/visualize/viser_visualizer.py"""

# ruff: noqa
# mypy: ignore-errors

from pathlib import Path

import hppfcl
import numpy as np

import pinocchio as pin
from pinocchio.visualize.base_visualizer import BaseVisualizer

import trimesh
import viser

MESH_TYPES = (hppfcl.BVHModelBase, hppfcl.HeightFieldOBBRSS, hppfcl.HeightFieldAABB)


class ViserVisualizer(BaseVisualizer):
    """A Pinocchio visualizer using Viser."""

    def __init__(
        self,
        viser_server: viser.ViserServer,
        model=pin.Model(),
        collision_model=None,
        visual_model=None,
        copy_models=False,
        data=None,
        collision_data=None,
        visual_data=None,
    ):
        super().__init__(
            model,
            collision_model,
            visual_model,
            copy_models,
            data,
            collision_data,
            visual_data,
        )
        self.static_objects = []
        self.viser_server = viser_server
        self.frames = {}
        self.loadViewerModel()

    def loadViewerModel(
        self,
        rootNodeName="pinocchio",
        collision_color=None,
        visual_color=None,
        frame_axis_length=0.018,
        frame_axis_radius=0.0008,
    ):
        """Load the robot in a Viser viewer.

        Parameters:
            rootNodeName: name to give to the robot in the viewer
            collision_color: optional, color to give to the collision model of
                the robot. Format is a list of four RGBA floating-point numbers
                (between 0 and 1)
            visual_color: optional, color to give to the visual model of
                the robot. Format is a list of four RGBA floating-point numbers
                (between 0 and 1)
            frame_axis_length: optional, length of frame axes if displaying frames.
            frame_axis_radius: optional, radius of frame axes if displaying frames.
        """
        self.viewerRootNodeName = rootNodeName

        # Create root frames to help toggle visibility in the Viser UI.
        self.visualRootNodeName = rootNodeName + "/visual"
        self.visualRootFrame = self.viser_server.scene.add_frame(
            self.visualRootNodeName, show_axes=False
        )
        self.collisionRootNodeName = rootNodeName + "/collision"
        self.collisionRootFrame = self.viser_server.scene.add_frame(
            self.collisionRootNodeName, show_axes=False
        )
        self.framesRootNodeName = rootNodeName + "/frames"
        self.framesRootFrame = self.viser_server.scene.add_frame(
            self.framesRootNodeName, show_axes=False
        )

        # Load visual model
        if (visual_color is not None) and (len(visual_color) != 4):
            raise RuntimeError("visual_color must have 4 elements for RGBA.")
        if self.visual_model is not None:
            for visual in self.visual_model.geometryObjects:
                self.loadViewerGeometryObject(
                    visual, self.visualRootNodeName, visual_color
                )
        self.displayVisuals(True)

        # Load collision model
        if (collision_color is not None) and (len(collision_color) != 4):
            raise RuntimeError("collision_color must have 4 elements for RGBA.")
        if self.collision_model is not None:
            for collision in self.collision_model.geometryObjects:
                self.loadViewerGeometryObject(
                    collision, self.collisionRootNodeName, collision_color
                )
        self.displayCollisions(False)

        # Load frames
        for frame in self.model.frames:
            frame_name = self.framesRootNodeName + "/" + frame.name
            self.frames[frame_name] = self.viser_server.scene.add_frame(
                frame_name,
                show_axes=True if "site" in frame_name else False,
                axes_length=frame_axis_length,
                axes_radius=frame_axis_radius,
            )
        self.displayFrames(True)

    def loadViewerGeometryObject(self, geometry_object, prefix="", color=None):
        """Loads a single geometry object."""
        name = geometry_object.name
        if prefix:
            name = prefix + "/" + name
        geom = geometry_object.geometry
        color_override = color or geometry_object.meshColor

        if isinstance(geom, hppfcl.Box):
            frame = self.viser_server.scene.add_box(
                name,
                dimensions=geom.halfSide * 2.0,
                color=color_override[:3],
                opacity=color_override[3],
            )
        elif isinstance(geom, hppfcl.Sphere):
            frame = self.viser_server.scene.add_icosphere(
                name,
                radius=geom.radius,
                color=color_override[:3],
                opacity=color_override[3],
            )
        elif isinstance(geom, hppfcl.Cylinder):
            mesh = trimesh.creation.cylinder(
                radius=geom.radius,
                height=geom.halfLength * 2.0,
            )
            frame = self.viser_server.scene.add_mesh_simple(
                name,
                mesh.vertices,
                mesh.faces,
                color=color_override[:3],
                opacity=color_override[3],
            )
        elif isinstance(geom, MESH_TYPES):
            frame = self._add_mesh_from_path(
                name, geometry_object.meshPath, color_override
            )
        elif isinstance(geom, hppfcl.Convex):
            if len(geometry_object.meshPath) > 0:
                frame = self._add_mesh_from_path(
                    name, geometry_object.meshPath, color_override
                )
            else:
                frame = self._add_mesh_from_convex(name, geom, color_override)
        else:
            raise RuntimeError(f"Unsupported geometry type for {name}: {type(geom)}")

        self.frames[name] = frame

    def _add_mesh_from_path(self, name, mesh_path, color):
        """
        Load a mesh from a file.
        """
        extension = Path(mesh_path).suffix.lower()
        if extension == ".dae" or color is None:
            mesh = trimesh.load_scene(mesh_path)
            return self.viser_server.scene.add_mesh_trimesh(name, mesh)
        else:
            mesh = trimesh.load_mesh(mesh_path)
            return self.viser_server.scene.add_mesh_simple(
                name,
                mesh.vertices,
                mesh.faces,
                color=color[:3],
                opacity=color[3],
            )

    def _add_mesh_from_convex(self, name, geom, color):
        """
        Load a mesh from triangles stored inside a hppfcl.Convex.
        """
        num_tris = geom.num_polygons
        call_triangles = geom.polygons
        call_vertices = geom.points

        vertices = call_vertices()
        vertices = vertices.astype(np.float32)
        faces = np.empty((num_tris, 3), dtype=int)
        for k in range(num_tris):
            tri = call_triangles(k)
            faces[k] = [tri[i] for i in range(3)]

        return self.viser_server.scene.add_mesh_simple(
            name,
            vertices,
            faces,
            color=color[:3],
            opacity=color[3],
        )

    def display(self, q=None):
        """
        Display the robot at configuration q in the viewer by placing all the bodies
        """
        if q is not None:
            pin.forwardKinematics(self.model, self.data, q)

        if self.collisionRootFrame.visible:
            self.updatePlacements(pin.GeometryType.COLLISION)

        if self.visualRootFrame.visible:
            self.updatePlacements(pin.GeometryType.VISUAL)

        if self.framesRootFrame.visible:
            self.updateFrames()

    def displayCollisions(self, visibility):
        self.collisionRootFrame.visible = visibility
        self.updatePlacements(pin.GeometryType.COLLISION)

    def displayVisuals(self, visibility):
        self.visualRootFrame.visible = visibility
        self.updatePlacements(pin.GeometryType.VISUAL)

    def displayFrames(self, visibility):
        self.framesRootFrame.visible = visibility
        self.updateFrames()

    def drawFrameVelocities(self, *args, **kwargs):
        raise NotImplementedError("drawFrameVelocities is not yet implemented.")

    def updatePlacements(self, geometry_type):
        if geometry_type == pin.GeometryType.VISUAL:
            geom_model = self.visual_model
            geom_data = self.visual_data
            prefix = self.viewerRootNodeName + "/visual"
        else:
            geom_model = self.collision_model
            geom_data = self.collision_data
            prefix = self.viewerRootNodeName + "/collision"

        pin.updateGeometryPlacements(self.model, self.data, geom_model, geom_data)
        for geom_id, geometry_object in enumerate(geom_model.geometryObjects):
            # Get mesh pose.
            M = geom_data.oMg[geom_id]

            # Update viewer configuration.
            frame_name = prefix + "/" + geometry_object.name
            frame = self.frames[frame_name]
            frame.position = M.translation * geometry_object.meshScale
            frame.wxyz = pin.Quaternion(M.rotation).coeffs()[
                [3, 0, 1, 2]
            ]  # Pinocchio uses xyzw

    def updateFrames(self):
        pin.updateFramePlacements(self.model, self.data)
        for frame_id, frame in enumerate(self.model.frames):
            # Get frame pose.
            M = self.data.oMf[frame_id]

            # Update viewer configuration.
            viser_frame_name = self.framesRootNodeName + "/" + frame.name
            viser_frame = self.frames[viser_frame_name]
            viser_frame.position = M.translation
            viser_frame.wxyz = pin.Quaternion(M.rotation).coeffs()[
                [3, 0, 1, 2]
            ]  # Pinocchio uses xyzw

    def captureImage(self, w=None, h=None, client_id=None, transport_format="jpeg"):
        """
        Capture an image from the Viser viewer and return an RGB array.

        Parameters:
            w: The width of the captured image. If None, uses the actual camera width.
            h: The height of the captured image. If None, uses the actual camera height.
            client_id: The ID of the Viser client handle.
                If None, uses the first available client.
            transport_format: The transport format to use for the captured image.
                Can be "jpeg" (default) or "png".
        """
        clients = self.viewer.get_clients()
        if len(clients) == 0:
            raise RuntimeError("Viser server has no attached clients!")

        if client_id is None:
            cli = next(iter(clients.values()))
        elif client_id not in clients:
            raise RuntimeError(
                f"Viser server does not have a client with ID '{client_id}'"
            )
        else:
            cli = clients[client_id]

        height = h or cli.camera.image_height
        width = w or cli.camera.image_width
        return cli.get_render(
            height=height, width=width, transport_format=transport_format
        )

    def setBackgroundColor(self, preset_name: str = "gray", col_top=None, col_bot=None):
        raise NotImplementedError("setBackgroundColor is not yet implemented.")

    def setCameraTarget(self, target: np.ndarray):
        raise NotImplementedError("setCameraTarget is not yet implemented.")

    def setCameraPosition(self, position: np.ndarray):
        raise NotImplementedError("setCameraPosition is not yet implemented.")

    def setCameraZoom(self, zoom: float):
        raise NotImplementedError("setCameraZoom is not yet implemented.")

    def setCameraPose(self, pose: np.ndarray = np.eye(4)):
        raise NotImplementedError("setcameraPose is not yet implemented.")

    def disableCameraControl(self):
        raise NotImplementedError("disableCameraControl is not yet implemented.")

    def enableCameraControl(self):
        raise NotImplementedError("enableCameraControl is not yet implemented.")
