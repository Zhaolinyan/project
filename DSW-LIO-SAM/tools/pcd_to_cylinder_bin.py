"""Convert Newer College/Ouster binary PCD scans to Cylinder3D demo .bin files."""

import argparse
import os
from pathlib import Path

import numpy as np


PCD_TYPE_TO_DTYPE = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("U", 1): "u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("I", 1): "i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
}


def read_pcd_xyzi(path, drop_invalid=True, normalize_intensity=True):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: missing DATA line")
            decoded = line.decode("ascii", errors="replace").strip()
            header_lines.append(decoded)
            if decoded.startswith("DATA"):
                if decoded != "DATA binary":
                    raise ValueError(f"{path}: only binary PCD is supported, got '{decoded}'")
                break
        payload = f.read()

    header = {}
    for line in header_lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        header[parts[0]] = parts[1:]

    fields = header["FIELDS"]
    sizes = [int(x) for x in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(x) for x in header.get("COUNT", ["1"] * len(fields))]
    points = int(header["POINTS"][0])

    names = []
    formats = []
    offsets = []
    offset = 0
    for name, size, typ, count in zip(fields, sizes, types, counts):
        if count != 1:
            raise ValueError(f"{path}: field '{name}' has unsupported COUNT={count}")
        dtype = PCD_TYPE_TO_DTYPE.get((typ, size))
        if dtype is None:
            raise ValueError(f"{path}: unsupported PCD field type {typ} size {size}")
        names.append(name)
        formats.append(dtype)
        offsets.append(offset)
        offset += size

    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": offset})
    data = np.frombuffer(payload, dtype=dtype, count=points)

    missing = [name for name in ("x", "y", "z") if name not in data.dtype.names]
    if missing:
        raise ValueError(f"{path}: missing required fields {missing}")

    intensity_name = next(
        (name for name in ("intensity", "signal", "reflectivity") if name in data.dtype.names),
        None,
    )

    xyzi = np.zeros((points, 4), dtype=np.float32)
    xyzi[:, 0] = data["x"].astype(np.float32)
    xyzi[:, 1] = data["y"].astype(np.float32)
    xyzi[:, 2] = data["z"].astype(np.float32)
    if intensity_name:
        xyzi[:, 3] = data[intensity_name].astype(np.float32)

    if drop_invalid:
        valid = np.isfinite(xyzi).all(axis=1)
        valid &= np.linalg.norm(xyzi[:, :3], axis=1) > 1e-3
        xyzi = xyzi[valid]

    if normalize_intensity and xyzi.size:
        max_intensity = float(np.max(xyzi[:, 3]))
        if max_intensity > 1.0:
            xyzi[:, 3] /= max_intensity
    return xyzi


def convert_folder(input_dir, output_dir, max_frames, drop_invalid, normalize_intensity):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd_paths = sorted(input_dir.glob("*.pcd"))
    if max_frames > 0:
        pcd_paths = pcd_paths[:max_frames]
    if not pcd_paths:
        raise RuntimeError(f"No .pcd files found in {input_dir}")

    for index, pcd_path in enumerate(pcd_paths):
        xyzi = read_pcd_xyzi(
            pcd_path,
            drop_invalid=drop_invalid,
            normalize_intensity=normalize_intensity,
        )
        out_path = output_dir / f"{index:06d}.bin"
        xyzi.tofile(out_path)
        print(f"[{index}] {pcd_path.name}: {xyzi.shape[0]} points -> {out_path}")

    print(f"Done. Converted {len(pcd_paths)} scans to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder containing .pcd scans")
    parser.add_argument("--output-dir", required=True, help="Folder for Cylinder3D .bin files")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--keep-invalid", action="store_true", help="Keep zero/invalid organized-cloud points")
    parser.add_argument("--raw-intensity", action="store_true", help="Do not normalize intensity to 0..1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_folder(
        args.input_dir,
        args.output_dir,
        args.max_frames,
        drop_invalid=not args.keep_invalid,
        normalize_intensity=not args.raw_intensity,
    )
