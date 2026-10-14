import bpy
import os
import numpy as np
from ..utils.camera_eval import evaluate_camera, render_resolution
from ..utils.colmap import (
    SPARSE_FOLDER,
    colmap_pose,
    write_cameras_bin,
    write_images_bin,
    write_points3d_bin,
)
from ..utils.output_paths import render_image_extension, rendered_frames


class EXPORT_OT_colmap(bpy.types.Operator):
    """Export cameras as a COLMAP sparse model (sparse/0/*.bin)"""

    bl_idname = "export.colmap"
    bl_label = "Export COLMAP Model"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return (
            context.scene.camera
            and context.scene.frame_end >= context.scene.frame_start
            and context.scene.output_folder.strip()
        )

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        # Same frames and file names Render Animation produces
        frames = rendered_frames(scene)
        width, height = render_resolution(scene)
        ext = render_image_extension(scene)
        matrices, intrinsics = evaluate_camera(scene, camera, frames)

        # One COLMAP camera per distinct set of intrinsics (usually just one)
        cameras = []
        camera_ids = {}
        images = []
        for image_id, frame_idx in enumerate(frames, start=1):
            params = intrinsics[frame_idx]
            key = tuple(round(p, 6) for p in params)
            if key not in camera_ids:
                camera_ids[key] = len(camera_ids) + 1
                cameras.append((camera_ids[key], width, height, params))
            qvec, tvec = colmap_pose(matrices[frame_idx])
            images.append((image_id, qvec, tvec, camera_ids[key], f"{frame_idx:04d}{ext}"))

        sparse_dir = os.path.join(bpy.path.abspath(scene.output_folder), SPARSE_FOLDER)
        os.makedirs(sparse_dir, exist_ok=True)
        write_cameras_bin(os.path.join(sparse_dir, "cameras.bin"), cameras)
        write_images_bin(os.path.join(sparse_dir, "images.bin"), images)

        # Keep a model valid before the point cloud exists, without clobbering
        # points written by Generate Point Cloud
        points_path = os.path.join(sparse_dir, "points3D.bin")
        if not os.path.exists(points_path):
            write_points3d_bin(points_path, np.empty((0, 3)), np.empty((0, 3)))

        if ext == ".exr":
            self.report(
                {"WARNING"},
                f"Exported {len(images)} images to {sparse_dir}, but most COLMAP trainers can't read EXR frames",
            )
        else:
            self.report(
                {"INFO"},
                f"Exported {len(images)} images, {len(cameras)} camera(s) to {sparse_dir}",
            )
        return {"FINISHED"}
