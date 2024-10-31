import torch
import torch.nn as nn
import math
import numpy as np

def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return H, W, intrinsics, c2w


def to_viewpoint_camera(camera):
    """
    Parse a camera of intrinsic and c2w into a Camera Object
    """
    device = camera.device
    Hs, Ws, intrinsics, c2ws = parse_camera(camera.unsqueeze(0))
    camera = Camera(width=int(Ws[0]), height=int(Hs[0]), intrinsic=intrinsics[0], c2w=c2ws[0])
    return camera

class Camera(nn.Module):
    def __init__(self, width, height, intrinsic, c2w, znear=0.1, zfar=100., trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super(Camera, self).__init__()
        device = c2w.device
        self.znear = znear
        self.zfar = zfar
        self.focal_x, self.focal_y = intrinsic[0, 0], intrinsic[1, 1]
        self.FoVx = focal2fov(self.focal_x, width)
        self.FoVy = focal2fov(self.focal_y, height)
        self.image_width = int(width)
        self.image_height = int(height)
        self.world_view_transform = torch.linalg.inv(c2w).permute(1,0)
        self.intrinsic = intrinsic
        self.c2w = c2w
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(device)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def camera_path_creator(camera):
    # Get Intrisic camera parameters
    H, W, intrinsic, c2w = parse_camera(camera[None,...])

    # Calculate camera position in 3d around an ellipse
    cam_position = circle_path(theta=45, r=2, num=300) 
    # Calculate the poses

    poses = [compute_pose(cam_position[:, i].T) for i in range(cam_position.shape[1])]

    # Convert the C2W in the opencv format. 
    poses = [convert_c2w_to_opencvg_format(pose) for pose in poses]

    cameras = []

    for pose in poses:

        cameras.append(torch.concatenate(([H,W,
                                           intrinsic.flatten(),
                                           torch.from_numpy(pose.flatten()).to(torch.float32).to(H.device)])))
    return cameras

def circle_path(theta=45, r=5, num=100):
    # r^2=s^2+t^2
    theta = np.deg2rad(theta) 
    phi = np.deg2rad(np.linspace(0,360,num))
    x = r*np.cos(phi)
    y = r/np.sin(theta)*np.sin(phi)
    z = r/np.cos(theta)*np.sin(phi)

    v = np.vstack((x,y,z))
    return v

def compute_pose(position, target=np.array([0.0, 0.0, 0.0]), up=np.array([0.0, 0.0, 1.0])):
    """
    Compute the camera pose matrix given the camera position, target, and up vector.
    """


    # Compute the forward vector (camera direction)
    forward = (target - position)
    forward /= np.linalg.norm(forward,axis=0)

    # Compute the right vector
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        # Up vector is parallel to forward vector; choose a different up vector
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    # Recompute the true up vector
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    # Build the rotation matrix
    rotation = np.vstack([right, up, -forward]).T

        # Swap Y and Z axes: New Y = Z, New Z = -Y
    swap_matrix = np.array([
        [1, 0,  0],
        [0, 0, -1],
        [0, 1,  0]
    ])

    # theta = np.radians(-90)  # Convert degrees to radians
    # R_x_neg90 = np.array([
    #     [1, 0,            0],
    #     [0, np.cos(theta), -np.sin(theta)],
    #     [0, np.sin(theta),  np.cos(theta)]
    # ])

    # rotation = R_x_neg90 @ rotation
    
    # Build the pose matrix
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position

    return pose

def convert_c2w_to_opencvg_format(c2w):
    w2c_blender = np.linalg.inv(c2w)
    w2c_opencv = w2c_blender
    w2c_opencv[1:3] *= -1
    c2w_opencv = np.linalg.inv(w2c_opencv)

    return c2w_opencv