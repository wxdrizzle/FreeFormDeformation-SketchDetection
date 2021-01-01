import manimlib.config
import manimlib.constants
import manimlib.extract_scene
import numpy as np


def generate_video(original_mesh, current_mesh, mesh_spacing, file="ffd_visualization.py", low_quality=True, resolution="700,700",
                   scene_names=None, preview=False):
    """
    generate video using manim libarary with customized parameters

    Args:
        original_mesh: original original_mesh
        current_mesh: current original_mesh
        mesh_spacing: original_mesh spacing
        file: file that defines animations
        low_quality:
        resolution: size of generated video
        scene_names: video types
        preview: automatically open the saved file once its done

    Returns:
        video path
    """

    if scene_names is None:
        scene_names = ['FFDSquare']
    args = manimlib.config.parse_cli()
    args.file = file
    args.low_quality = low_quality
    args.resolution = resolution
    args.scene_names = scene_names
    args.preview = preview
    config = manimlib.config.get_configuration(args)
    manimlib.constants.initialize_directories(config)

    mesh_trans_flipped = flip_mesh(original_mesh, current_mesh)
    return manimlib.extract_scene.main(config, original_mesh, mesh_trans_flipped, mesh_spacing)


def flip_mesh(original_mesh, current_mesh):
    """
    flip current original_mesh in order to make original_mesh in the FFD module of GUI and that in the video are the same

    Args:
        original_mesh: original original_mesh
        current_mesh: current original_mesh

    Returns:

    """
    disp = current_mesh - original_mesh
    disp[:, :original_mesh.shape[1] - 1, :original_mesh.shape[1] - 1] = \
        np.flip(disp[:, :original_mesh.shape[1] - 1, :original_mesh.shape[1] - 1], 2)
    disp[1, :, :] = -disp[1, :, :]
    return original_mesh + disp


if __name__ == "__main__":
    mesh_size = 5
    delta = 40.
    mesh = np.ones((2, mesh_size + 3, mesh_size + 3))
    for i in range(mesh_size + 3):
        for j in range(mesh_size + 3):
            mesh[:, i, j] = [(i - 1) * delta, (j - 1) * delta]
    displacement = 0.3 * delta * np.random.randn(np.size(mesh, 0), np.size(mesh, 1), np.size(mesh, 2))
    mesh_trans = mesh + displacement
    print(generate_video(mesh, mesh_trans, delta))
