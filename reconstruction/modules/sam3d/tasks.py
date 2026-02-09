"""
Celery tasks for SAM3D service.
Wraps functions from functions/ module with Celery task decorators.
"""
from modules.sam3d.worker import celery_app
from modules.sam3d.functions.image_to_mesh import image_to_mesh as image_to_mesh_func
from modules.sam3d.functions.render_debug_image import render_debug_image as render_debug_image_func

@celery_app.task(queue='sam3d.render_debug_image')
def render_debug_image(image_path: str, mesh_path: str, transform_path: str, intrinsics_path: str, output_image_path: str, num_vertices_to_use: int = 5000):
    """Render a debug image of a mesh"""
    return render_debug_image_func(image_path, mesh_path, transform_path, intrinsics_path, output_image_path, num_vertices_to_use)

@celery_app.task(queue='sam3d.image_to_mesh')
def image_to_mesh(image_path: str, mask_path: str, mesh_path: str, transform_path: str, intrinsics_path: str,
                 seed: int = None,
                 stage1_only: bool = False,
                 with_mesh_postprocess: bool = False,
                 with_texture_baking: bool = False,
                 with_layout_postprocess: bool = False,
                 use_vertex_color: bool = True,
                 stage1_inference_steps: int = None):
    """Process an image with mask to generate 3D mesh and save outputs to files."""
    return image_to_mesh_func(
        image_path, mask_path, mesh_path, transform_path, intrinsics_path,
        seed=seed,
        stage1_only=stage1_only,
        with_mesh_postprocess=with_mesh_postprocess,
        with_texture_baking=with_texture_baking,
        with_layout_postprocess=with_layout_postprocess,
        use_vertex_color=use_vertex_color,
        stage1_inference_steps=stage1_inference_steps
    )

