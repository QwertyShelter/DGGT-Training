import os
import sys
import glob
import argparse
import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import rendering

def load_ply(path):
    p = o3d.io.read_point_cloud(path)
    if not p.has_colors():
        # 如果没有颜色，设为灰色
        colors = np.ones((np.asarray(p.points).shape[0], 3), dtype=np.float64) * 0.8
        p.colors = o3d.utility.Vector3dVector(colors)
    return p

def animate(ply_dir, out_video=None, fps=10, point_size=2.0, window_size=(1024,768), loop=False):
    files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
    if not files:
        raise FileNotFoundError(f"No .ply files found in {ply_dir}")

    width, height = window_size
    renderer = rendering.OffscreenRenderer(width, height)

    # simple material for points
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = point_size

    writer = None
    if out_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))

    try:
        # 初次添加并设置相机
        p0 = load_ply(files[0])
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("pc", p0, mat)

        bbox = p0.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        # extent = np.linalg.norm(bbox.get_extent())
        # if extent == 0:
        #     extent = -0.1
        # eye = center + np.array([0.0, 0.0, extent * 2.0])
        eye = np.array([0.0, -0.1, 0.3])     # -1.0
        up = np.array([0.0, -1.0, 0.0])
        cam = renderer.scene.camera
        cam.look_at(center, eye, up)

        while True:
            for f in files:
                pc = load_ply(f)
                # 更新场景中的点云（移除并重新添加）
                try:
                    renderer.scene.remove_geometry("pc")
                except Exception:
                    pass
                renderer.scene.add_geometry("pc", pc, mat)

                img = renderer.render_to_image()
                arr = np.asarray(img)
                # Open3D 输出通常为 uint8 RGB
                if arr.dtype != np.uint8:
                    arr = (arr * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                cv2.imwrite("test.png", img_bgr)

                if writer is not None:
                    if img_bgr.shape[1] != width or img_bgr.shape[0] != height:
                        img_bgr = cv2.resize(img_bgr, (width, height))
                    writer.write(img_bgr)

            if not loop:
                break
    finally:
        if writer is not None:
            writer.release()
        # renderer.release()

# ...existing code...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate PLY sequence with Open3D")
    parser.add_argument("--ply_dir", default='result/dog', help="Directory with .ply frames (sorted lexicographically)")
    parser.add_argument("--out", default='dog.mp4', help="Output MP4 path (optional)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args()

    animate(args.ply_dir, out_video=args.out, fps=args.fps, point_size=args.point_size, window_size=(args.width, args.height), loop=args.loop)

# export EGL_PLATFORM=surfaceless