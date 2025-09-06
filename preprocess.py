import os
import re
import glob
import torch
import numpy as np
from PIL import Image

def load_sequence_as_tensor(seq_dir, normalize_per_frame=True):
    """
    Loads all tXXX.tif frames in seq_dir, sorts by index, converts to grayscale,
    normalizes to [0,1], stacks into shape [D, H, W], then returns [1, D, H, W].
    """
    # Find frames like t000.tif, t001.tif, ...
    frame_paths = glob.glob(os.path.join(seq_dir, "t*.tif"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {seq_dir}")

    # Sort by numeric index in filename
    def frame_index(p):
        m = re.search(r"t(\d+)\.tif$", os.path.basename(p))
        return int(m.group(1)) if m else 0

    frame_paths.sort(key=frame_index)

    tensor_stack = []
    ref_hw = None

    for fp in frame_paths:
        img = Image.open(fp)

        # Ensure single-channel (grayscale); DIC is grayscale anyway.
        if img.mode not in ("I;16", "I", "F", "L"):
            img = img.convert("L")

        arr = np.asarray(img, dtype=np.float32)

        # Normalize to [0, 1]
        if normalize_per_frame:
            amin = float(arr.min())
            amax = float(arr.max())
            if amax > amin:
                arr = (arr - amin) / (amax - amin)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
        else:
            print("Not implemented!") 
            assert False

        # Ensure consistent shape across frames
        h, w = arr.shape[:2]
        if ref_hw is None:
            ref_hw = (h, w)
        else:
            if (h, w) != ref_hw:
                raise ValueError(f"Frame {fp} has shape {(h, w)}; expected {ref_hw}")

        tensor_stack.append(torch.from_numpy(arr).unsqueeze(0))  # [1,H,W]

    volume = torch.cat(tensor_stack, dim=0).contiguous()         # [D,H,W]
    volume = volume.unsqueeze(0)                                 # [1,D,H,W]
    return volume

def save_sequence(volume, out_base_path, key_name):
    """
    Saves the tensor in your expected format:
      path: {out_base_path}_tensor
      content: { key_name: volume }
    where volume has shape [1, D, H, W] and dtype float32 in [0,1].
    """
    os.makedirs(os.path.dirname(out_base_path), exist_ok=True)
    out_path = f"{out_base_path}_tensor"
    torch.save({key_name: volume}, out_path)
    return out_path

if __name__ == "__main__":
    # Configure paths
    dataset_root = "Fluo-N2DL-HeLa"     # folder that contains '01', '02', ..
    sequences = ["01", "02"]
    out_root = "data/dataset"

    for seq in sequences:
        seq_dir = os.path.join(dataset_root, seq)
        key_name = f"Fluo-N2DL-HeLa_{seq}_source"
        out_base = os.path.join(out_root, f"{key_name}")

        volume = load_sequence_as_tensor(seq_dir, normalize_per_frame=True)  # [1,D,H,W]
        saved_path = save_sequence(volume, out_base, key_name)

        d, h, w = volume.shape[1:]
        print(f"{seq}: saved to {saved_path}")
        print(f"  shape={tuple(volume.shape)}, dtype={volume.dtype}, "
              f"min={float(volume.min()):.4f}, max={float(volume.max()):.4f}, "
              f"mean={float(volume.mean()):.4f}, std={float(volume.std()):.4f}")


