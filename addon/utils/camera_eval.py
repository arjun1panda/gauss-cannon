from mathutils import Matrix, Vector, Euler
import numpy as np


def iter_action_fcurves(animation_data):
    """
    Yield F-curves from animation_data.

    F-curves live in Channelbags inside Strips inside Layers (Slotted
    Actions), keyed by the animation_data's action_slot.
    """
    if not animation_data or not animation_data.action:
        return
    action = animation_data.action

    slot = getattr(animation_data, "action_slot", None)
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            cb = None
            if slot is not None and hasattr(strip, "channelbag"):
                try:
                    cb = strip.channelbag(slot)
                except Exception:
                    cb = None
            if cb is None:
                # No slot match — fall back to any channelbags on the strip
                for cb_alt in getattr(strip, "channelbags", []):
                    for fc in cb_alt.fcurves:
                        yield fc
                continue
            for fc in cb.fcurves:
                yield fc


def _is_animated(id_data):
    """True if an ID's properties can change per frame: action F-curves,
    drivers or NLA strips."""
    anim = id_data.animation_data
    return bool(
        anim
        and (
            any(True for _ in iter_action_fcurves(anim))
            or len(anim.drivers) > 0
            or len(anim.nla_tracks) > 0
        )
    )


def build_fast_path_evaluator(camera):
    """
    Build a frame -> world-matrix evaluator that bypasses scene.frame_set.

    Reads keyframe values directly from the camera's location/rotation_euler
    F-curves so we don't trigger a full depsgraph evaluation for every frame
    (which is slow when the scene has rigged characters, simulations, etc.).

    Returns a callable on success, or None when the camera transform can't
    be reconstructed in isolation — caller should then fall back to
    scene.frame_set + camera.matrix_world.
    """
    # Anything that makes matrix_world depend on more than the camera's own
    # keyframed local transform forces the slow path.
    if camera.parent is not None:
        return None
    if len(camera.constraints) > 0:
        return None
    if camera.rotation_mode != "XYZ":
        return None
    if any(camera.delta_location) or any(camera.delta_rotation_euler):
        return None
    if tuple(camera.delta_scale) != (1.0, 1.0, 1.0):
        return None
    if _is_animated(camera.data):
        # Animated intrinsics (lens, sensor, clip): we'd need depsgraph eval
        return None
    anim = camera.animation_data
    if not anim or not anim.action:
        return None
    if len(anim.drivers) > 0 or len(anim.nla_tracks) > 0:
        return None
    if anim.action_influence != 1.0 or anim.action_blend_type != "REPLACE":
        return None

    needed = {
        ("location", 0): None,
        ("location", 1): None,
        ("location", 2): None,
        ("rotation_euler", 0): None,
        ("rotation_euler", 1): None,
        ("rotation_euler", 2): None,
    }
    for fc in iter_action_fcurves(anim):
        key = (fc.data_path, fc.array_index)
        if key in needed:
            needed[key] = fc
        elif fc.data_path.startswith("delta_"):
            return None
    if any(fc is None for fc in needed.values()):
        return None
    # Keyframe values are only what renders if nothing reshapes the curve
    if any(fc.mute or len(fc.modifiers) > 0 for fc in needed.values()):
        return None

    def to_dict(fc):
        return {int(round(kp.co.x)): kp.co.y for kp in fc.keyframe_points}

    loc_maps = [to_dict(needed[("location", i)]) for i in range(3)]
    rot_maps = [to_dict(needed[("rotation_euler", i)]) for i in range(3)]

    scale_mat = Matrix.Diagonal(camera.scale).to_4x4()

    def evaluate(frame_idx):
        try:
            loc = Vector(
                (loc_maps[0][frame_idx], loc_maps[1][frame_idx], loc_maps[2][frame_idx])
            )
            rot = Euler(
                (rot_maps[0][frame_idx], rot_maps[1][frame_idx], rot_maps[2][frame_idx]),
                "XYZ",
            )
        except KeyError:
            # Frame missing from keyframes for any channel; let caller fall back.
            return None
        return Matrix.Translation(loc) @ rot.to_matrix().to_4x4() @ scale_mat

    return evaluate


def render_resolution(scene):
    """Rendered image size in pixels, after the resolution percentage."""
    render = scene.render
    return (
        render.resolution_x * render.resolution_percentage // 100,
        render.resolution_y * render.resolution_percentage // 100,
    )


def pinhole_intrinsics(cam_data, width, height, pixel_aspect=(1.0, 1.0)):
    """
    Pixel intrinsics (fx, fy, cx, cy) of a perspective camera, with cx/cy in
    OpenCV image coordinates (origin top-left, y down).

    Follows Blender's viewplane (BKE_camera_params_compute_viewplane): honors
    sensor_fit (unlike deriving fy from sensor_height, which Blender ignores
    under AUTO/HORIZONTAL fit), lens shift and non-square pixels. With square
    pixels and no shift this is LichtFeld's convention:
    focal = 0.5 * resolution / tan(0.5 * fov).

    Args:
        cam_data: Camera data (or anything with sensor_fit, angle, shift_x,
                  shift_y)
        width, height: Rendered image size in pixels
        pixel_aspect: (pixel_aspect_x, pixel_aspect_y) from the render settings
    """
    aspect_x, aspect_y = pixel_aspect
    # Pixel height in units of pixel width
    ycor = aspect_y / aspect_x
    fit = cam_data.sensor_fit
    if fit == "AUTO":
        fit = "HORIZONTAL" if aspect_x * width >= aspect_y * height else "VERTICAL"
    # Image extent, in pixel widths, that the camera angle and shift span
    viewfac = width if fit == "HORIZONTAL" else ycor * height

    fx = 0.5 * viewfac / np.tan(0.5 * cam_data.angle)
    fy = fx / ycor
    # Shift moves the frustum, so the optical axis moves the other way in the
    # image (left for +x; down, i.e. +y in OpenCV rows, for +y)
    cx = 0.5 * width - cam_data.shift_x * viewfac
    cy = 0.5 * height + cam_data.shift_y * viewfac / ycor
    return fx, fy, cx, cy


def evaluate_camera(scene, camera, frames):
    """
    Camera world matrix and pinhole intrinsics at each frame, as rendered.

    Uses the keyframe fast path when possible, otherwise scene.frame_set per
    frame (restoring the current frame afterwards). Intrinsics are evaluated
    per frame when the camera data (lens, sensor, shift) is animated.

    Returns:
        tuple: (dict frame -> Matrix (4x4 world matrix),
                dict frame -> (fx, fy, cx, cy))
    """
    width, height = render_resolution(scene)
    pixel_aspect = (scene.render.pixel_aspect_x, scene.render.pixel_aspect_y)
    animated_data = _is_animated(camera.data)
    # The fast path refuses animated camera data, so frame_set runs before
    # every per-frame intrinsics read below
    fast_eval = build_fast_path_evaluator(camera)
    static_intrinsics = (
        None if animated_data else pinhole_intrinsics(camera.data, width, height, pixel_aspect)
    )

    orig_frame = scene.frame_current
    moved = False
    matrices, intrinsics = {}, {}
    for frame_idx in frames:
        matrix = fast_eval(frame_idx) if fast_eval is not None else None
        if matrix is None:
            scene.frame_set(frame_idx)
            moved = True
            matrix = camera.matrix_world.copy()
        matrices[frame_idx] = matrix
        if animated_data:
            intrinsics[frame_idx] = pinhole_intrinsics(camera.data, width, height, pixel_aspect)
        else:
            intrinsics[frame_idx] = static_intrinsics
    if moved:
        scene.frame_set(orig_frame)
    return matrices, intrinsics
