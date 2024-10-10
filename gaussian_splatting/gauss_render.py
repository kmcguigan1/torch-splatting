import pdb
import torch
import torch.nn as nn
import math
from einops import reduce

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance
    # symm = strip_symmetric(actual_covariance)
    # return symm

def build_covariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.3
    return cov2d[:, :2, :2] + filter[None]

def build_covariance_3d_projected(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # The following models the steps outlined by equations 29
	# and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	# Additionally considers aspect / scaling of viewport.
	# Transposes used to account for row-/column-major conventions.
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]

    # truncate the influences of gaussians far outside the frustum.
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx*1.3, max=tan_fovx*1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy*1.3, max=tan_fovy*1.3) * t[..., 2]
    tz = t[..., 2]

    # we need to get the transformed covariance first
    # to do so we get the view matrix and transform the cov3d to be on the 
    # same basis as the view matrix
    # this is imperfect since the x, y covariance of this changes at the depths as well based on focal 
    # stuff but this is an okay approximate for now
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix

    # Eq.29 locally affine transform 
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    # J[..., 2, 0] = tx / t.norm(dim=-1) # discard
    # J[..., 2, 1] = ty / t.norm(dim=-1) # discard
    # J[..., 2, 2] = tz / t.norm(dim=-1) # discard
    cov3d_proj = J @ W @ cov3d @ W.T @ J.permute(0,2,1)
    
    # add low pass filter here according to E.q. 32
    filter = torch.eye(3,3).to(cov3d_proj) * 0.3
    return cov3d_proj + filter[None]

def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask

@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


from .utils.sh_utils import eval_sh
import torch.autograd.profiler as profiler
USE_PROFILE = False
import contextlib

class GaussRenderer(nn.Module):
    """
    A gaussian splatting renderer

    >>> gaussModel = GaussModel.create_from_pcd(pts)
    >>> gaussRender = GaussRenderer()
    >>> out = gaussRender(pc=gaussModel, camera=camera)
    """

    def __init__(self, active_sh_degree=3, white_bkgd=True, ignore_negatives:bool=False, **kwargs):
        super(GaussRenderer, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.ignore_negatives = ignore_negatives
        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(128), torch.arange(128), indexing='xy'), dim=-1).to('cuda')
        
    
    # def build_color(self, means3D, shs, camera):
    #     rays_o = camera.camera_center
    #     rays_d = means3D - rays_o
    #     color = eval_sh(self.active_sh_degree, shs.permute(0,2,1), rays_d)
    #     color = (color + 0.5).clip(min=0.0)
    #     return color
    
    def render(self, camera, means2D, cov2d, color, opacity, depths, 
               neg_means2D, neg_cov3d, neg_opacity, neg_depths):
        # print("cov2d: ", cov2d)
        radii = get_radius(cov2d)
        # print("radii: ", radii)
        rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
        # print("rect: ", rect)

        if(not self.ignore_negatives):
            # print(f"N Negs before radii, {neg_cov3d.shape}")
            neg_radii = get_radius(neg_cov3d[..., :2, :2])
            neg_rect = get_rect(neg_means2D, neg_radii, width=camera.image_width, height=camera.image_height)
            # print(f"N Negs after radii: {neg_radii.shape}")
        
        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to('cuda')
        self.render_depth = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')

        TILE_SIZE = 32
        tile_stats = {'tile_negs':0, 'tiles':0, 'negs':0, 'pos':0}
        for h in range(0, camera.image_height, TILE_SIZE):
            for w in range(0, camera.image_width, TILE_SIZE):
                tile_stats['tiles'] += 1
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                
                P = in_mask.sum()
                if not in_mask.sum() > 0:
                    continue
                # else:
                    # print("Found gaussian")

                tile_stats['pos'] += P.detach().cpu().numpy()

                # get the tile coordinates and sort the positive gaussians by depth
                # this is done for traditional rasterization
                # sort all our variables as well
                tile_coord = self.pix_coord[h:h+TILE_SIZE, w:w+TILE_SIZE].flatten(0,-2)
                sorted_depths, index = torch.sort(depths[in_mask])
                sorted_means2D = means2D[in_mask][index]
                sorted_cov2d = cov2d[in_mask][index] # P 2 2
                sorted_conic = sorted_cov2d.inverse() # inverse of variance
                sorted_opacity = opacity[in_mask][index]
                sorted_color = color[in_mask][index]

                # get the distance from each gaussian to each pixel coordinate in our
                # pixel space, then calculate the gaussian weight
                # each pixel, each gaussian, xy
                dx = (tile_coord[:,None,:] - sorted_means2D[None,:]) # Im P+ 2 (B P 2 old)
                gauss_weight = torch.exp(-0.5 * (
                    dx[:, :, 0]**2 * sorted_conic[:, 0, 0] 
                    + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]
                    + dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]))
                
                # alpha = (gauss_weight[..., None] * sorted_opacity[None]).clip(max=0.99) # Im P+ 1 (B P 1 old)
                alpha = (gauss_weight[..., None] * sorted_opacity[None]) # Im P+ 1 (B P 1 old)


                if(not self.ignore_negatives):
                    # check for negative gaussians that are in the tile
                    neg_over_tl = neg_rect[0][..., 0].clip(min=w), neg_rect[0][..., 1].clip(min=h)
                    neg_over_br = neg_rect[1][..., 0].clip(max=w+TILE_SIZE-1), neg_rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                    neg_in_mask = (neg_over_br[0] > neg_over_tl[0]) & (neg_over_br[1] > neg_over_tl[1]) # 3D gaussian in the tile 
                    
                    # if and only if we have more than zero negative gaussians then we need to adjust
                    # the gaus weights based on the spatial positioning of these gaussians
                    NP = neg_in_mask.sum()
                    tile_stats['negs'] += NP.detach().cpu().numpy()
                    if(NP == 0): continue

                    tile_stats['tile_negs'] += 1

                    # container for the negative gaussian impact
                    negative_impact_per_positive_gaussian = torch.zeros_like(alpha)

                    # tile in the depth dimension
                    N_NEGS_PER_TILE = 2
                    depth_tile_count = math.ceil(NP / N_NEGS_PER_TILE)

                    # Get indecies of the neg gaussian [0] is to get out of tuple
                    neg_in_mask_indecies = torch.where(neg_in_mask)[0]

                    for depth_idx in range(depth_tile_count):
                        lower = depth_idx * N_NEGS_PER_TILE
                        upper = (depth_idx + 1) * N_NEGS_PER_TILE
                        upper = min(upper, NP)
                        neg_in_mask_in_depth = neg_in_mask_indecies[lower:upper]

                        # select the negative gaussians in the tile
                        sel_neg_means2D = neg_means2D[neg_in_mask_in_depth]
                        sel_neg_cov3d = neg_cov3d[neg_in_mask_in_depth]
                        sel_neg_depths = neg_depths[neg_in_mask_in_depth]
                        sel_neg_opacity = neg_opacity[neg_in_mask_in_depth]
                        neg_conic = sel_neg_cov3d.inverse() # inverse of variance

                        # get the distances used for calculating the gaussian impact
                        dx = (tile_coord[:,None,:] - sel_neg_means2D[None,:]) # Im N- 2 (N P 2 old)
                        dx = dx.unsqueeze(1).expand(-1, P, -1, -1) # Im P+ N- 2 (N B P 2 old)
                        dy = sorted_depths[:, None] - sel_neg_depths[None, :] # P+ N-
                        dy = dy.unsqueeze(dim=0).unsqueeze(dim=-1) # Im(1) P+ N- 1
                        dy = dy.expand(dx.shape[0], -1, -1, -1) # Im P+ N- 1
                        dx = torch.cat((dx, dy), dim=-1) # Im P+ N- 3 (N B P 3 old)

                        # get the negative point spread
                        neg_gauss_weight = torch.exp(-0.5 * (
                            dx[...,0]**2 * neg_conic[:, 0, 0] 
                            + dx[...,1]**2 * neg_conic[:, 1, 1]
                            + dx[...,2]**2 * neg_conic[:, 2, 2]
                            + dx[...,0]*dx[...,1] * neg_conic[:, 0, 1]
                            + dx[...,0]*dx[...,2] * neg_conic[:, 0, 2]
                            + dx[...,1]*dx[...,0] * neg_conic[:, 1, 0]
                            + dx[...,1]*dx[...,2] * neg_conic[:, 1, 2]
                            + dx[...,2]*dx[...,0] * neg_conic[:, 2, 0]
                            + dx[...,2]*dx[...,1] * neg_conic[:, 2, 1]
                        )) # Im P+ N- (N B P old)

                        # calculate the negative alpha of the gaussians
                        # neg_alpha = (neg_gauss_weight[..., None] * sel_neg_opacity[None]).clip(max=0.99) # Im P+ N- 1 (N B P 1 old)
                        neg_alpha = (neg_gauss_weight[..., None] * sel_neg_opacity[None]) # Im P+ N- 1 (N B P 1 old)

                        neg_alpha = neg_alpha.sum(dim=2, keepdims=False) # Im P+ 1  (B P 1 old)

                        negative_impact_per_positive_gaussian += neg_alpha # Im P+ 1

                    # apply the negative gaussian alpha to the positive gaussians
                    alpha = alpha - negative_impact_per_positive_gaussian
                    # alpha = alpha.clip(min=0.01)
                        
                alpha = alpha.clip(max=0.99)
                T = torch.cat([torch.ones_like(alpha[:,:1]), 1-alpha[:,:-1]], dim=1).cumprod(dim=1)
                acc_alpha = (alpha * T).sum(dim=1)
                tile_color = (T * alpha * sorted_color[None]).sum(dim=1) + (1-acc_alpha) * (1 if self.white_bkgd else 0)
                tile_depth = ((T * alpha) * sorted_depths[None,:,None]).sum(dim=1)
                self.render_color[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_color.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_depth[h:h+TILE_SIZE, w:w+TILE_SIZE] = tile_depth.reshape(TILE_SIZE, TILE_SIZE, -1)
                self.render_alpha[h:h+TILE_SIZE, w:w+TILE_SIZE] = acc_alpha.reshape(TILE_SIZE, TILE_SIZE, -1)
        
        # print(tile_stats)
        return {
            "render": self.render_color,
            "depth": self.render_depth,
            "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii,
            "tile_stats": tile_stats
        }

    def forward(self, camera, pc, **kwargs):
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        color = pc.get_colors

        # negative variables
        if(not self.ignore_negatives):
            neg_means3D = pc.get_negative_xyz
            neg_opacity = pc.get_negative_opacity
            neg_scales = pc.get_negative_scaling
            neg_rotations = pc.get_negative_rotation
            # print(f"N Negs from model {neg_means3D.shape}")

        # print(f"Initial number of points: {means3D.shape[0]}")
        
        if USE_PROFILE:
            prof = profiler.record_function
        else:
            prof = contextlib.nullcontext
            
        # if we are rendering and not training we should filter out the negative gaussians with low
        # alpha values to make it faster to render

        
        with prof("positive projection"):
            # get the positive projections
            mean_ndc, mean_view, in_mask = projection_ndc(means3D, 
                    viewmatrix=camera.world_view_transform, 
                    projmatrix=camera.projection_matrix)
            
            # print("mean_ndc: ", mean_ndc)
            # print("mean_view: ", mean_view)
            # print("in_mask: ", in_mask)
            mean_ndc = mean_ndc[in_mask]
            mean_view = mean_view[in_mask]
            depths = mean_view[:,2]

        if(not self.ignore_negatives):
            with prof("negative projection"):
                # get the negative projections
                neg_mean_ndc, neg_mean_view, neg_in_mask = projection_ndc(neg_means3D, 
                        viewmatrix=camera.world_view_transform, 
                        projmatrix=camera.projection_matrix)
                neg_mean_ndc = neg_mean_ndc[neg_in_mask]
                neg_mean_view = neg_mean_view[neg_in_mask]
                neg_depths = neg_mean_view[:,2]
                # print(f"N Negs from projection {neg_mean_ndc.shape}")

        # every point should be in the view since the default is that chair
        # with a wide FOV camera angle
        # print(f"projected number of points: {mean_ndc.shape[0]}")
        
        # with prof("build color"):
        #     color = self.build_color(means3D=means3D, shs=shs, camera=camera)
        
        with prof("build cov3d"):
            cov3d = build_covariance_3d(scales, rotations)
            # print("cov3d: ", cov3d)

        if(not self.ignore_negatives):
            with prof("build negative cov3d"):
                neg_cov3d = build_covariance_3d(neg_scales, neg_rotations)
                
        with prof("build cov2d"):
            cov2d = build_covariance_2d(
                mean3d=means3D, 
                cov3d=cov3d, 
                viewmatrix=camera.world_view_transform,
                fov_x=camera.FoVx, 
                fov_y=camera.FoVy, 
                focal_x=camera.focal_x, 
                focal_y=camera.focal_y)
            
            # print("cov2d: ", cov2d)

            mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
            mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
            means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
            # print("means2d: ", means2D)


        if(not self.ignore_negatives):
            with prof("build negative cov2d"):
                neg_cov3d_proj = build_covariance_3d_projected(
                    mean3d=neg_means3D, 
                    cov3d=neg_cov3d, 
                    viewmatrix=camera.world_view_transform,
                    fov_x=camera.FoVx, 
                    fov_y=camera.FoVy, 
                    focal_x=camera.focal_x, 
                    focal_y=camera.focal_y)

                neg_mean_coord_x = ((neg_mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
                neg_mean_coord_y = ((neg_mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
                neg_means2D = torch.stack([neg_mean_coord_x, neg_mean_coord_y], dim=-1)
                # print(f"N Negs from cov2d {neg_means2D.shape}")
                # print(f"N Negs from conv2d projection with 3d {neg_cov3d_proj.shape}")
        
        if(self.ignore_negatives):
            neg_means2D = None
            neg_cov3d_proj = None
            neg_opacity = None
            neg_depths = None

        with prof("render"):
            rets = self.render(
                camera = camera, 
                means2D=means2D,
                cov2d=cov2d,
                color=color,
                opacity=opacity, 
                depths=depths,
                neg_means2D=neg_means2D,
                neg_cov3d=neg_cov3d_proj,
                neg_opacity=neg_opacity, 
                neg_depths=neg_depths,
            )
        return rets
