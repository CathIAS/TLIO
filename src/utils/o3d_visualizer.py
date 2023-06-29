import os
from shutil import rmtree
import numpy as np
import pandas as pd
from collections import defaultdict

import open3d as o3d
from scipy.spatial.transform import Rotation
from .math_utils import inv_SE3
from .alignment import align_inertial_frames
from .from_scipy import compute_euler_from_matrix
from .logging import get_logger

logger = get_logger(__file__)


ALIGN_GT = True  # Whether to align the position and yaw of GT at start.
SHOW_W_FRAME = True
MAX_TRAJ_LEN = 50  # None # Int for fading trajectory (good if really messy) or None for show the whole trajectory

# Scale down the whole scene since O3D clips anythig beyond 2m from the camera (seriously why?!)
# https://github.com/isl-org/Open3D/issues/803
SCALE = 0.1

T_flip = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)

T_rotate_y_up = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)


def scaled_imu_pose_from_vio_ghost(vio_ghost):
    qxyzw_vio = vio_ghost[:, 1:5]
    p_vio = vio_ghost[:, 5:8]

    T_World_Imu_vio = np.zeros((qxyzw_vio.shape[0], 4, 4))
    T_World_Imu_vio[:, :3, :3] = Rotation.from_quat(qxyzw_vio).as_matrix()
    T_World_Imu_vio[:, :3, 3] = p_vio
    T_World_Imu_vio[:, 3, 3] = 1

    return T_World_Imu_vio


class O3dVisualizer:
    def __init__(
        self,
        vio_ghost: np.array = None,
        pointcloud_folder=None,
        save_imgs_path=None,  # "/tmp/vis_frames",
    ):
        o3d.visualization.gui.Application.instance.initialize()
        self.vio_ghost = vio_ghost
        self.realign_next_update = True

        self.save_imgs_path = save_imgs_path
        if self.save_imgs_path is not None:
            if os.path.exists(self.save_imgs_path):
                rmtree(self.save_imgs_path)
            os.makedirs(self.save_imgs_path, exist_ok=True)

        self.vis_intrinsics = np.array(
            [
                [640, 0, 640 / 2],
                [0, 640, 480 / 2],
                [0, 0, 1],
            ]
        )
        self.follow_cam = True
        self.follow_cam_offsets_idx = 0
        self.follow_cam_zoom = 2.0
        self.follow_cam_dzoom = 0.1
        self.follow_cam_offsets = np.array(
            [[1, 1, 0], [-1, 1, 1], [0, 1, 1], [0, 1, -1], [0, 1, 0]], dtype=np.float32
        )

        self.vis = o3d.visualization.O3DVisualizer("TLIO Viewer")
        self.vis.show_ground = False
        self.vis.ground_plane = o3d.visualization.rendering.Scene.GroundPlane.XZ
        self.vis.show_skybox(False)
        self.vis.scene_shader = o3d.visualization.O3DVisualizer.Shader.UNLIT
        self.vis.point_size = 1
        bg_color = np.array([[0.2, 0.2, 0.2, 1]], dtype=np.float32).T
        self.vis.set_background(bg_color, None)
        self.vis.size = o3d.visualization.gui.Size(1920, 1280)

        self.vis.add_action("Follow Cam", self.toggle_follow_cam)
        self.vis.add_action("Follow Cam Direction", self.toggle_follow_cam_direction)
        self.vis.add_action("Follow Cam Zoom In", self.follow_cam_zoom_in)
        self.vis.add_action("Follow Cam Zoom Out", self.follow_cam_zoom_out)
        self.vis.add_action("Realign With GT", self.request_realignment)

        # World frame (Two options here, default o3d axes (super long) or custom one
        if SHOW_W_FRAME:
            W_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            W_frame.scale(SCALE, (0, 0, 0))
            self.vis.add_geometry("world_frame", W_frame)
        # self.vis.show_axes = SHOW_W_FRAME

        self.is_first_update = True

        self.traj_pts_tlio = defaultdict(list)
        self.T_Vio_Tlio = defaultdict(
            lambda: np.eye(4)
        )  # Transforms (pos and yaw only) between TLIOs and VIO frames if not initialized the same.

        # Load raw vio if provided
        if self.vio_ghost is not None:
            self.full_t_us_vio = self.vio_ghost[:, 0]
            self.full_T_World_Imu_vio = scaled_imu_pose_from_vio_ghost(
                self.vio_ghost
            )
            self.full_traj_pts_vio = self.full_T_World_Imu_vio[:, :3, 3]

        self.pointcloud_folder = pointcloud_folder
        self.pointcloud = None
        self.maybe_load_and_realign_point_cloud(np.eye(4))
        self.geometries_last = {}

        # For programmable cam following
        self.num_cam_change = 0
        self.curr_idx = 0
        self.colors = [[0, 0, 1], [1, 0, 0]]
        self.gt_color = [0, 1, 0]

        self.run_once()

    def toggle_follow_cam(self, _):
        self.follow_cam = not self.follow_cam

    def toggle_follow_cam_direction(self, _):
        self.follow_cam_offsets_idx = (self.follow_cam_offsets_idx + 1) % len(
            self.follow_cam_offsets
        )

    def follow_cam_zoom_in(self, _):
        self.follow_cam_zoom -= self.follow_cam_dzoom

    def follow_cam_zoom_out(self, _):
        self.follow_cam_zoom += self.follow_cam_dzoom

    def request_realignment(self, _):
        self.realign_next_update = True

    def maybe_load_and_realign_point_cloud(self, T_Vio_Tlio):
        if self.pointcloud_folder is not None:
            if self.pointcloud is None:
                self.pointcloud = o3d.io.read_point_cloud(
                    self.pointcloud_folder + "/semidense_global_points_color.ply"
                )
                if len(self.pointcloud.points) == 0:
                    self.pointcloud = o3d.io.read_point_cloud(
                        self.pointcloud_folder + "/semidense_global_points.ply"
                    )
                self.pointcloud.scale(SCALE, center=(0, 0, 0))
                self.pointcloud = self.pointcloud.uniform_down_sample(5)

        # assume pointcloud is in the same frame as vio_ghost for alignment
        if self.pointcloud is not None and len(self.pointcloud.points) > 0:
            aligned_pointcloud = o3d.geometry.PointCloud(self.pointcloud.points)
            colors = np.asarray(self.pointcloud.colors)
            # filter red points (no colors)
            colors[colors[:, 0] == 1, 0] = 0
            aligned_pointcloud.colors = self.pointcloud.colors
            aligned_pointcloud.transform(T_rotate_y_up @ inv_SE3(T_Vio_Tlio))
            # aligned_pointcloud.paint_uniform_color((1,0,0))
            if self.vis.get_geometry("pointcloud").name == "pointcloud":
                self.vis.remove_geometry("pointcloud")
            self.vis.add_geometry("pointcloud", aligned_pointcloud)

    def add_traj_to_geometries(
        self,
        name,
        traj_points_World,
        geometries,
        color=[0, 0, 1],
        T_VisWorld_InputWorld=np.eye(4),
        scale=SCALE,
        max_traj_len=2000,
    ):
        if len(traj_points_World) > 1:
            # Cut off the trajectory since it's odometry
            max_traj_len = (
                max_traj_len if max_traj_len is not None else len(traj_points_World)
            )
            if len(traj_points_World) > max_traj_len:
                traj_points_World = traj_points_World[-max_traj_len:]
            traj = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(traj_points_World),
                lines=o3d.utility.Vector2iVector(
                    [[i, i + 1] for i in range(len(traj_points_World) - 1)]
                ),
            )
            traj.paint_uniform_color(color)
            traj.transform(T_rotate_y_up @ T_VisWorld_InputWorld)
            traj.scale(scale, (0, 0, 0))
            geometries[name] = traj

    def add_path_pose_frustrum_to_geometries(
        self,
        name_prefix,
        traj_points_World,
        T_World_Imu,
        color,
        geometries,
        scale=SCALE,
        T_VisWorld_InputWorld=np.eye(4),
        max_traj_len=2000,
        with_frustrum=False,
        size_coord_frame=0.5
    ):
        self.add_traj_to_geometries(
            f"{name_prefix}_path",
            traj_points_World,
            geometries,
            color=color,
            T_VisWorld_InputWorld=T_VisWorld_InputWorld,
            max_traj_len=max_traj_len,
        )
        # Coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size_coord_frame)
        frame.transform(T_rotate_y_up @ T_VisWorld_InputWorld @ T_World_Imu @ T_flip)
        frame.scale(scale, (0, 0, 0))
        geometries[f"{name_prefix}_frame"] = frame

        if with_frustrum:
            T_Imu_World = inv_SE3(T_VisWorld_InputWorld @ T_World_Imu)
            # Camera frustum (VIO ghost only)
            cam = o3d.geometry.LineSet.create_camera_visualization(
                640, 480, self.vis_intrinsics, np.eye(4), scale=size_coord_frame
            )
            cam.paint_uniform_color(color)
            cam.transform(T_rotate_y_up @ T_VisWorld_InputWorld @ T_World_Imu @ T_flip)
            cam.scale(scale, (0, 0, 0))
            geometries[f"{name_prefix}_cam"] = cam

    def run_once(self):
        app = o3d.visualization.gui.Application.instance
        if self.is_first_update:
            # Such an app is interactable without blocking the main thread
            # https://stackoverflow.com/questions/73668731/how-to-create-a-basic-gui-with-pointcloud-updates-using-open3d
            app.add_window(self.vis)
            self.is_first_update = False

        app_still_running = app.run_one_tick()
        if app_still_running:
            self.vis.post_redraw()

            if self.save_imgs_path is not None:
                self.vis.export_current_image(
                    os.path.join(self.save_imgs_path, f"{self.curr_idx:09d}.jpg")
                )

            self.curr_idx += 1

        return app_still_running

    def update(self, ts_us, dict_T_World_Imu, dict_position_since_last_update):
        """this function can take the result of several filter at the same time, their result is passed through the dict variable"""
        assert dict_T_World_Imu.keys() == dict_position_since_last_update.keys()

        geometries = {}
        for i, run_name in enumerate(dict_T_World_Imu.keys()):
            # Add trajectory path
            run_T_World_Imu = dict_T_World_Imu[run_name].copy()
            self.traj_pts_tlio[run_name] += dict_position_since_last_update[run_name]
            self.add_path_pose_frustrum_to_geometries(
                run_name,
                self.traj_pts_tlio[run_name],
                run_T_World_Imu,
                T_VisWorld_InputWorld=self.T_Vio_Tlio[run_name],
                color=self.colors[i],
                geometries=geometries,
                max_traj_len=MAX_TRAJ_LEN,
                size_coord_frame=0.5 if run_name == "tlio" else 0.5
            )

        if self.vio_ghost is not None:
            idx = np.searchsorted(self.full_t_us_vio, ts_us)
            # the treshold is loose here since the trajectory can be at framerate
            if idx > 0 and idx < len(self.full_t_us_vio):
                if idx < len(self.full_t_us_vio) - 1:
                    tk = self.full_t_us_vio[idx]
                    tkm1 = self.full_t_us_vio[idx - 1]
                    time_discrepancy_s = (tk - ts_us) * 1e-6
                    DT = (
                        self.full_T_World_Imu_vio[idx]
                        - self.full_T_World_Imu_vio[idx - 1]
                    )
                    T_World_Imu_vio = (
                        self.full_T_World_Imu_vio[idx - 1]
                        + float(ts_us - tkm1) / float(tk - tkm1) * DT
                    )
                    if time_discrepancy_s < 0.1:
                        if (
                            ALIGN_GT
                            and self.realign_next_update
                            and time_discrepancy_s < 0.05
                        ):
                            for run_name in dict_T_World_Imu.keys():
                                # Zero out the position and yaw offset between VIO and TLIO first frame.
                                run_T_World_Imu = dict_T_World_Imu[run_name].copy()
                                self.T_Vio_Tlio[run_name] = align_inertial_frames(
                                    run_T_World_Imu, T_World_Imu_vio
                                )
                            self.realign_next_update = False

                    self.add_path_pose_frustrum_to_geometries(
                        "vio",
                        self.full_traj_pts_vio[:idx, :],
                        T_World_Imu_vio,
                        color=self.gt_color,
                        geometries=geometries,
                        with_frustrum=True,
                        max_traj_len=20,
                        size_coord_frame=0.4
                    )
            else:
                logger.warning("Could not find close enough VIO timestamp for viz")

        # Adjust the camera if needed
        if self.follow_cam:
            run_name_followed = "tlio"
            T_World_Imu = dict_T_World_Imu[run_name_followed].copy()
            # align to VIO
            T_World_Imu = (
                T_rotate_y_up @ self.T_Vio_Tlio[run_name_followed] @ T_World_Imu
            )
            p = T_World_Imu[:3, 3]
            new_cam_p = (
                T_World_Imu[:3, 3]
                + self.follow_cam_zoom
                * self.follow_cam_offsets[self.follow_cam_offsets_idx]
            )
            # logger.info(f"Moving camera to {new_cam_p.tolist()}")
            self.vis.setup_camera(90, SCALE * p, SCALE * new_cam_p, [0, 1, 0])
            self.num_cam_change += 1

        # Scene is complete, update it
        for gname in self.geometries_last.keys():
            self.vis.remove_geometry(gname)

        self.geometries_last = geometries

        for gname, g in geometries.items():
            self.vis.add_geometry(gname, g)

        # Adjust all line widths
        for gname in geometries.keys():
            if "path" in gname or "cam" in gname:
                mat = self.vis.get_geometry_material(gname)
                mat.shader = "unlitLine"
                mat.line_width = 10
                self.vis.modify_geometry_material(gname, mat)
        
        self.run_once()
