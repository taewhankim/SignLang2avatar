import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_rasterizer(type = 'pytorch3d'):
    if type == 'pytorch3d':
        global Meshes, load_obj, rasterize_meshes
        # from pytorch3d.structures import Meshes
        # from pytorch3d.io import load_obj
        # from pytorch3d.renderer.mesh import rasterize_meshes
    elif type == 'standard':
        global standard_rasterize, load_obj
        import os
        # Use JIT Compiling Extensions
        # ref: https://pytorch.org/tutorials/advanced/cpp_extension.html
        from torch.utils.cpp_extension import load, CUDA_HOME
        curr_dir = os.path.dirname(__file__)
        standard_rasterize_cuda = \
            load(name='standard_rasterize_cuda',
                sources=[f'{curr_dir}/rasterizer/standard_rasterize_cuda.cpp', f'{curr_dir}/rasterizer/standard_rasterize_cuda_kernel.cu'],
                extra_cuda_cflags = ['-std=c++14', '-ccbin=$$(which gcc-7)']) # cuda10.2 is not compatible with gcc9. Specify gcc 7

class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    This class implements methods for rasterizing a batch of heterogenous Meshes.
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin':  None,
            'perspective_correct': False,
        }
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone(); attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        return pixel_vals



class SRenderY(nn.Module):
    def __init__(self, image_size, obj_filename, uv_size=256, rasterizer_type='standard'):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size

        if rasterizer_type == 'pytorch3d':
            self.rasterizer = Pytorch3dRasterizer(image_size)
            self.uv_rasterizer = Pytorch3dRasterizer(uv_size)
            verts, faces, aux = load_obj(obj_filename)
            uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
            uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
            faces = faces.verts_idx[None, ...]
        # elif rasterizer_type == 'standard':
        #     self.rasterizer = StandardRasterizer(image_size)
        #     self.uv_rasterizer = StandardRasterizer(uv_size)
        #     verts, uvcoords, faces, uvfaces = load_obj(obj_filename)
        #     verts = verts[None, ...]
        #     uvcoords = uvcoords[None, ...]
        #     faces = faces[None, ...]
        #     uvfaces = uvfaces[None, ...]
        else:
            NotImplementedError

        # faces
        dense_triangles = generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None, :, :])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords * 2 - 1;
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max() + 1, 1).float() / 255.
        face_colors = face_vertices(colors, faces)
        self.register_buffer('vertex_colors', colors)
        self.register_buffer('face_colors', face_colors)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor(
            [1 / np.sqrt(4 * pi), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), \
             ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))), (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), \
             (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))), (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
             (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi)))]).float()
        self.register_buffer('constant_factor', constant_factor)

    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point', background=None, h=None,
                w=None):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], rnage:[-1,1], projected vertices, in image space, for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights:
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        batch_size = vertices.shape[0]
        # normalize z to 10-90 for raterization (in pytorch3d, near far: 0-100)
        transformed_vertices = transformed_vertices.clone()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] / transformed_vertices[:, :, 2].max()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] * 80 + 10

        # attributes
        face_vertice = face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = vertex_normals(vertices, self.faces.expand(batch_size, -1, -1));
        face_normals = face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1));
        transformed_face_normals = face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        attributes = torch.cat([self.face_uvcoords.expand(batch_size, -1, -1, -1),
                                transformed_face_normals.detach(),
                                face_vertice.detach(),
                                face_normals],
                               -1)

        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)

        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :];
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type == 'point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(vertice_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                  lights)
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
                else:
                    shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]),
                                                      lights)
                    shading_images = shading.reshape(
                        [batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3, 1, 2)
            images = albedo_images * shading_images
        else:
            images = albedo_images
            shading_images = images.detach() * 0.

        if background is None:
            images = images * alpha_images + torch.ones_like(images).to(vertices.device) * (1 - alpha_images)
        else:
            # background = F.interpolate(background, [self.image_size, self.image_size])
            images = images * alpha_images + background.contiguous() * (1 - alpha_images)

        outputs = {
            'images': images,
            'albedo_images': albedo_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images,
            'transformed_normals': transformed_normals,
        }

        return outputs

    # def add_SHlight(self, normal_images, sh_coeff):
    #     '''
    #         sh_coeff: [bz, 9, 3]
    #     '''
    #     N = normal_images
    #     sh = torch.stack([
    #         N[:, 0] * 0. + 1., N[:, 0], N[:, 1], \
    #         N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
    #         N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
    #     ],
    #         1)  # [bz, 9, h, w]
    #     sh = sh * self.constant_factor[None, :, None, None]
    #     shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
    #     return shading
    #
    # def add_pointlight(self, vertices, normals, lights):
    #     '''
    #         vertices: [bz, nv, 3]
    #         lights: [bz, nlight, 6]
    #     returns:
    #         shading: [bz, nv, 3]
    #     '''
    #     light_positions = lights[:, :, :3];
    #     light_intensities = lights[:, :, 3:]
    #     directions_to_lights = F.normalize(light_positions[:, :, None, :] - vertices[:, None, :, :], dim=3)
    #     # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
    #     normals_dot_lights = (normals[:, None, :, :] * directions_to_lights).sum(dim=3)
    #     shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
    #     return shading.mean(1)
    #
    # def add_directionlight(self, normals, lights):
    #     '''
    #         normals: [bz, nv, 3]
    #         lights: [bz, nlight, 6]
    #     returns:
    #         shading: [bz, nv, 3]
    #     '''
    #     light_direction = lights[:, :, :3];
    #     light_intensities = lights[:, :, 3:]
    #     directions_to_lights = F.normalize(light_direction[:, :, None, :].expand(-1, -1, normals.shape[1], -1), dim=3)
    #     # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
    #     # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
    #     normals_dot_lights = torch.clamp((normals[:, None, :, :] * directions_to_lights).sum(dim=3), 0., 1.)
    #     shading = normals_dot_lights[:, :, :, None] * light_intensities[:, :, None, :]
    #     return shading.mean(1)

    def render_shape(self, vertices, transformed_vertices, colors=None, background=None, detail_normal_images=None,
                     lights=None, return_grid=False, uv_detail_normals=None, h=None, w=None):
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]
        if lights is None:
            light_positions = torch.tensor(
                [
                    [-5, 5, -5],
                    [5, 5, -5],
                    [-5, -5, -5],
                    [5, -5, -5],
                    [0, 0, -5],
                ]
            )[None, :, :].expand(batch_size, -1, -1).float()

            light_intensities = torch.ones_like(light_positions).float() * 1.7
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        # normalize z to 10-90 for raterization (in pytorch3d, near far: 0-100)
        transformed_vertices = transformed_vertices.clone()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] / transformed_vertices[:, :, 2].max()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] * 80 + 10

        # Attributes
        face_vertice = face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = vertex_normals(vertices, self.faces.expand(batch_size, -1, -1));
        face_normals = face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1));
        transformed_face_normals = face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        if colors is None:
            colors = self.face_colors.expand(batch_size, -1, -1, -1)
        attributes = torch.cat([colors,
                                transformed_face_normals.detach(),
                                face_vertice.detach(),
                                face_normals,
                                self.face_uvcoords.expand(batch_size, -1, -1, -1)],
                               -1)
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images
        if uv_detail_normals is not None:
            uvcoords_images = rendering[:, 12:15, :, :];
            grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
            detail_normal_images = F.grid_sample(uv_detail_normals, grid, align_corners=False)
            normal_images = detail_normal_images

        shading = self.add_directionlight(normal_images.permute(0, 2, 3, 1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0, 3,
                                                                                                                  1,
                                                                                                                  2).contiguous()
        shaded_images = albedo_images * shading_images

        if background is None:
            shape_images = shaded_images * alpha_images + torch.ones_like(shaded_images).to(vertices.device) * (
                        1 - alpha_images)
        else:
            # background = F.interpolate(background, [self.image_size, self.image_size])
            shape_images = shaded_images * alpha_images + background.contiguous() * (1 - alpha_images)

        if return_grid:
            uvcoords_images = rendering[:, 12:15, :, :];
            grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
            return shape_images, normal_images, grid
        else:
            return shape_images

    def render_depth(self, transformed_vertices):
        '''
        -- rendering depth
        '''
        transformed_vertices = transformed_vertices.clone()
        batch_size = transformed_vertices.shape[0]

        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        z = -transformed_vertices[:, :, 2:].repeat(1, 1, 3)
        z = z - z.min()
        z = z / z.max()
        # Attributes
        attributes = face_vertices(z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images

    def render_colors(self, transformed_vertices, colors, h=None, w=None):
        '''
        -- rendering colors: could be rgb color/ normals, etc
            colors: [bz, num of vertices, 3]
        '''
        transformed_vertices = transformed_vertices.clone()
        batch_size = colors.shape[0]
        # normalize z to 10-90 for raterization (in pytorch3d, near far: 0-100)
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] - transformed_vertices[:, :, 2].min()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] / transformed_vertices[:, :, 2].max()
        transformed_vertices[:, :, 2] = transformed_vertices[:, :, 2] * 80 + 10
        # Attributes
        attributes = face_vertices(colors, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h=h, w=w)
        ####
        alpha_images = rendering[:, [-1], :, :].detach()
        images = rendering[:, :3, :, :] * alpha_images
        return images

    def world2uv(self, vertices):
        '''
        project vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''
        batch_size = vertices.shape[0]
        face_vertice = face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(self.uvcoords.expand(batch_size, -1, -1),
                                         self.uvfaces.expand(batch_size, -1, -1), face_vertice)[:, :3]
        return uv_vertices






def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn

def generate_triangles(h, w, mask = None):
    '''
    quad layout:
        0 1 ... w-1
        w w+1
        .
        w*h
    '''
    triangles = []
    margin=0
    for x in range(margin, w-1-margin):
        for y in range(margin, h-1-margin):
            triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
            triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:,[0,2,1]]
    return triangles

def face_vertices(vertices, faces):
    """
    borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def vertex_normals(vertices, faces):
    """
    borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals