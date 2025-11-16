import argparse
from pathlib import Path

from ultralytics import YOLO
import coremltools as ct


def export_to_coreml_ane(
    weights_path: str,
    imgsz: int = 640,
    out_dir: str = "coreml_ane",
):
    weights_path = Path(weights_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[convert] Loading YOLO weights from: {weights_path}")
    model = YOLO(str(weights_path))

    # 1) Export to CoreML (.mlpackage) with static input size
    #    - imgsz: fixed size (ANE does not like dynamic shapes)
    #    - half: FP16 (ANE-friendly)
    print(f"[convert] Exporting to CoreML (imgsz={imgsz}x{imgsz}, half=True)")
    coreml_pkg = model.export(
        format="coreml",
        imgsz=imgsz,
        half=True,
        dynamic=False,
    )
    # Ultralytics returns a path-like object
    coreml_pkg = Path(coreml_pkg)
    print(f"[convert] CoreML package created: {coreml_pkg}")

    # 2) Load with coremltools and save as .mlmodel (simpler single file)
    print("[convert] Loading CoreML model with coremltools...")
    mlmodel = ct.models.MLModel(coreml_pkg)

    # Ensure spec revision supports ANE well (CoreML 4+)
    spec = mlmodel.get_spec()
    # You can bump spec.version if needed; generally coremltools handles this.

    # Save as .mlmodel
    out_model_path = out_dir / f"{weights_path.stem}_ane_{imgsz}.mlmodel"
    mlmodel.save(out_model_path)
    print(f"[convert] Saved ANE-optimized CoreML model: {out_model_path}")

    print(
        "\nNext step: In your runtime, load this model with "
        "ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)\n"
    )
    return out_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLO .pt weights (e.g. yolov8l.pt or yolo11s.pt)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square input size (e.g. 640). Must match runtime preprocessing.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="coreml_ane",
        help="Directory to save the resulting .mlmodel",
    )

    args = parser.parse_args()
    export_to_coreml_ane(args.weights, imgsz=args.imgsz, out_dir=args.out_dir)
