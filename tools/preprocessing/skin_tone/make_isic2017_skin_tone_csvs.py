#!/usr/bin/env python3
"""
Build ISIC 2017 skin-tone labels (binary) from ITA analysis.

Logic matches `ita_analysis.py`:
  - Load image + lesion mask (ISIC convention: ISIC_XXXX.jpg -> ISIC_XXXX_segmentation.png)
  - Ignore "vintage black" / vignetting via Lab lightness threshold (L < lightness_threshold)
  - Ignore the lesion itself via the mask
  - Compute ITA on remaining pixels and map mean ITA to categories
  - skin_tone = 1 if category in {"Intermediate","Tan","Brown","Dark"}, else 0

Additionally, this script crops each sample to the lesion bounding box (from the mask)
before computing ITA and saving image/mask crops.

Outputs:
  - train.csv: (training + validation combined)
  - test.csv:  (test split only)
Each CSV has columns:
  image_path,mask_path,skin_tone
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def list_images_in_dir(images_dir: str) -> list[str]:
    files = []
    for fn in os.listdir(images_dir):
        ext = os.path.splitext(fn)[-1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(images_dir, fn))
    files.sort()
    if not files:
        raise RuntimeError(f"No images found in {images_dir}")
    return files


def get_mask_path(
    image_path: str, masks_dir: str, mask_suffix: str = "_segmentation"
) -> Optional[str]:
    if not masks_dir:
        return None
    basename = os.path.basename(image_path)
    name, _ = os.path.splitext(basename)
    mask_name = f"{name}{mask_suffix}.png"
    mask_path = os.path.join(masks_dir, mask_name)
    if os.path.exists(mask_path):
        return mask_path
    return None


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image in [0,1] to CIE L*a*b*.
    Uses skimage if available; falls back to OpenCV otherwise.
    """
    try:
        from skimage import color  # type: ignore

        return color.rgb2lab(img)
    except Exception:
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Need either scikit-image or opencv-python installed for RGB->LAB conversion. "
                "Install with: pip install scikit-image OR pip install opencv-python"
            ) from e

        # OpenCV expects uint8 BGR in [0,255]
        bgr = (img[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # OpenCV L in [0,255], a,b in [0,255] with 128 center
        L = lab[..., 0] * (100.0 / 255.0)
        a = lab[..., 1] - 128.0
        b = lab[..., 2] - 128.0
        return np.stack([L, a, b], axis=-1)


def compute_ita_from_lab(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Common ITA definition: ITA = arctan2(L - 50, b) * 180 / pi."""
    return np.degrees(np.arctan2((L - 50.0), b + 1e-8))


def ita_to_fitzpatrick(ita_mean: float) -> str:
    if np.isnan(ita_mean):
        return "Unknown"
    if ita_mean > 55.0:
        return "Very Light"
    if ita_mean > 41.0:
        return "Light"
    if ita_mean > 28.0:
        return "Intermediate"
    if ita_mean > 10.0:
        return "Tan"
    if ita_mean > -30.0:
        return "Brown"
    return "Dark"


@dataclass(frozen=True)
class ItaResult:
    ita_mean: float
    category: str
    skin_tone: int


def compute_lesion_bbox(
    lesion_mask: np.ndarray, pad_frac: float = 0.05
) -> Tuple[int, int, int, int]:
    """
    Returns PIL crop box (left, upper, right, lower) in pixel coordinates.
    If lesion_mask has no foreground pixels, returns full image bbox.
    """
    if lesion_mask.dtype != bool:
        lesion_mask = lesion_mask.astype(bool)

    h, w = lesion_mask.shape[:2]
    ys, xs = np.where(lesion_mask)
    if len(ys) == 0:
        return (0, 0, w, h)

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    box_h = y1 - y0 + 1
    box_w = x1 - x0 + 1
    pad_y = int(box_h * pad_frac)
    pad_x = int(box_w * pad_frac)

    left = max(0, x0 - pad_x)
    right = min(w, x1 + pad_x + 1)
    upper = max(0, y0 - pad_y)
    lower = min(h, y1 + pad_y + 1)
    return (left, upper, right, lower)


def per_image_ita_stats_from_full_image_and_mask(
    img_pil: Image.Image,
    mask_pil: Optional[Image.Image],
    *,
    resize: Optional[int],
    ita_clip: Optional[Tuple[float, float]],
    lightness_threshold: float,
) -> ItaResult:
    """
    Compute ITA using the same logic as `ita_analysis.py`:
      - ignore lesion pixels using the lesion mask (mask > 127)
      - ignore "vintage black"/vignetting pixels using Lab L channel threshold (L < lightness_threshold)
      - compute ITA from remaining (healthy) pixels
    """
    if resize is not None:
        img_pil = img_pil.resize((resize, resize), Image.BILINEAR)
    img_arr = np.asarray(img_pil, dtype=np.float32) / 255.0

    if mask_pil is not None:
        if resize is not None:
            mask_pil = mask_pil.resize((resize, resize), Image.NEAREST)
        mask_np = np.asarray(mask_pil.convert("L"), dtype=np.uint8)
        lesion_mask = mask_np > 127
    else:
        lesion_mask = np.zeros(img_arr.shape[:2], dtype=bool)

    lab = rgb_to_lab(img_arr)
    L = lab[..., 0]
    b = lab[..., 2]

    vignette_mask = L < lightness_threshold
    valid_mask = ~(lesion_mask | vignette_mask)

    L_valid = L[valid_mask]
    b_valid = b[valid_mask]

    if L_valid.size == 0:
        return ItaResult(ita_mean=float("nan"), category="Unknown", skin_tone=0)

    ita = compute_ita_from_lab(L_valid, b_valid)
    if ita_clip is not None:
        lo, hi = ita_clip
        ita = np.clip(ita, lo, hi)

    ita_mean = float(np.mean(ita))
    category = ita_to_fitzpatrick(ita_mean)
    dark_categories = {"Tan", "Brown", "Dark"}
    skin_tone = 1 if category in dark_categories else 0
    return ItaResult(ita_mean=ita_mean, category=category, skin_tone=skin_tone)


def per_image_ita_stats_from_pil_crops(
    img_crop_pil: Image.Image,
    mask_crop_pil: Image.Image,
    *,
    resize: Optional[int],
    ita_clip: Optional[Tuple[float, float]],
    lightness_threshold: float,
) -> ItaResult:
    if resize is not None:
        img_crop_pil = img_crop_pil.resize((resize, resize), Image.BILINEAR)
        mask_crop_pil = mask_crop_pil.resize((resize, resize), Image.NEAREST)

    arr = np.asarray(img_crop_pil, dtype=np.float32) / 255.0  # HxWx3, [0,1]
    lab = rgb_to_lab(arr)
    L = lab[..., 0]
    b = lab[..., 2]

    mask_np = np.asarray(mask_crop_pil.convert("L"), dtype=np.uint8)
    lesion_mask = mask_np > 127

    # Ignore vintage-black/vignetting via lightness threshold
    vignette_mask = L < lightness_threshold

    # Valid healthy skin pixels = not lesion AND not vignette black corners
    valid_mask = ~(lesion_mask | vignette_mask)

    L_valid = L[valid_mask]
    b_valid = b[valid_mask]

    if L_valid.size == 0:
        return ItaResult(ita_mean=float("nan"), category="Unknown", skin_tone=0)

    ita = compute_ita_from_lab(L_valid, b_valid)
    if ita_clip is not None:
        lo, hi = ita_clip
        ita = np.clip(ita, lo, hi)

    ita_mean = float(np.mean(ita))
    category = ita_to_fitzpatrick(ita_mean)
    dark_categories = {"Tan", "Brown", "Dark"}
    skin_tone = 1 if category in dark_categories else 0
    return ItaResult(ita_mean=ita_mean, category=category, skin_tone=skin_tone)


def save_crop_pair(
    img_crop_pil: Image.Image,
    mask_crop_pil: Image.Image,
    out_img_path: str,
    out_mask_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
    img_crop_pil.save(out_img_path)
    mask_crop_pil.save(out_mask_path)


def process_split(
    *,
    images_dir: str,
    masks_dir: str,
    split_name: str,
    mask_suffix: str,
    resize: int,
    lightness_threshold: float,
    ita_clip_min: float,
    ita_clip_max: float,
    crop_pad_frac: float,
    max_images: Optional[int],
    require_mask: bool,
    save_crops: bool,
    out_images_dir: Optional[str],
    out_masks_dir: Optional[str],
    csv_writer: csv.writer,
    warn_prefix: str = "",
) -> None:
    image_paths = list_images_in_dir(images_dir)
    if max_images is not None:
        image_paths = image_paths[:max_images]

    ita_clip = (ita_clip_min, ita_clip_max)

    for image_path in tqdm(image_paths, desc=f"{split_name} ITA"):
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        mask_path = get_mask_path(image_path, masks_dir, mask_suffix=mask_suffix)
        if mask_path is None and require_mask:
            raise FileNotFoundError(f"Missing mask for {image_path} in {masks_dir}")

        img_pil = Image.open(image_path).convert("RGB")

        mask_pil: Optional[Image.Image]
        if mask_path is not None:
            mask_pil = Image.open(mask_path).convert("L")
        else:
            mask_pil = None

        # Compute ITA skin tone on the FULL image/mask (matches `ita_analysis.py`),
        # then optionally save lesion crops for training.
        ita_res = per_image_ita_stats_from_full_image_and_mask(
            img_pil,
            mask_pil,
            resize=resize,
            ita_clip=ita_clip,
            lightness_threshold=lightness_threshold,
        )

        if not save_crops:
            csv_writer.writerow([image_path, mask_path or "", ita_res.skin_tone])
            continue

        if out_images_dir is None or out_masks_dir is None:
            raise ValueError("save_crops=True requires out_images_dir and out_masks_dir.")

        # Save crops around the lesion bounding box (computed from the ORIGINAL mask size).
        if mask_pil is None:
            # No mask: write resized full image as a degenerate crop.
            crop_box = (0, 0, img_pil.size[0], img_pil.size[1])
            img_crop_pil = img_pil.crop(crop_box)
            mask_crop_pil = Image.fromarray(np.zeros((img_pil.size[1], img_pil.size[0]), dtype=np.uint8))
        else:
            mask_np = np.asarray(mask_pil.convert("L"), dtype=np.uint8)
            lesion_mask = mask_np > 127
            crop_box = compute_lesion_bbox(lesion_mask, pad_frac=crop_pad_frac)
            img_crop_pil = img_pil.crop(crop_box)
            mask_crop_pil = mask_pil.crop(crop_box)

        out_img_path = os.path.join(out_images_dir, f"{image_id}.png")
        out_mask_path = os.path.join(out_masks_dir, f"{image_id}.png")

        save_crop_pair(
            img_crop_pil.resize((resize, resize), Image.BILINEAR),
            mask_crop_pil.resize((resize, resize), Image.NEAREST),
            out_img_path=out_img_path,
            out_mask_path=out_mask_path,
        )

        csv_writer.writerow([out_img_path, out_mask_path, ita_res.skin_tone])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make ISIC 2017 ITA-based skin tone CSVs.")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to ISIC 2017 root containing training/val/test images and masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write train.csv/test.csv (and optionally cropped images/masks).",
    )
    parser.add_argument("--mask-suffix", type=str, default="_segmentation")
    parser.add_argument("--ita-resize", type=int, default=224)
    parser.add_argument("--lightness-threshold", type=float, default=20.0)
    parser.add_argument("--ita-clip-min", type=float, default=-90.0)
    parser.add_argument("--ita-clip-max", type=float, default=90.0)
    parser.add_argument("--crop-pad-frac", type=float, default=0.05)
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="If set, also save lesion-cropped/resized images/masks and point the CSV at them.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit for quick debugging (applies per split).",
    )
    parser.add_argument(
        "--require-mask",
        action="store_true",
        help="Fail if a mask file is missing (otherwise create a zero mask fallback).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_images_dir = os.path.join(args.data_root, "training_images")
    train_masks_dir = os.path.join(args.data_root, "training_masks")
    val_images_dir = os.path.join(args.data_root, "val_images")
    val_masks_dir = os.path.join(args.data_root, "val_masks")
    test_images_dir = os.path.join(args.data_root, "test_images")
    test_masks_dir = os.path.join(args.data_root, "test_masks")

    os.makedirs(args.output_dir, exist_ok=True)

    train_img_out = os.path.join(args.output_dir, "train_images_cropped")
    train_mask_out = os.path.join(args.output_dir, "train_masks_cropped")
    test_img_out = os.path.join(args.output_dir, "test_images_cropped")
    test_mask_out = os.path.join(args.output_dir, "test_masks_cropped")

    if args.save_crops:
        os.makedirs(train_img_out, exist_ok=True)
        os.makedirs(train_mask_out, exist_ok=True)
        os.makedirs(test_img_out, exist_ok=True)
        os.makedirs(test_mask_out, exist_ok=True)

    train_csv_path = os.path.join(args.output_dir, "train.csv")
    test_csv_path = os.path.join(args.output_dir, "test.csv")

    for p in (train_csv_path, test_csv_path):
        if os.path.exists(p):
            os.remove(p)

    # ---- Train CSV = training + validation combined ----
    with open(train_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "skin_tone"])

        process_split(
            images_dir=train_images_dir,
            masks_dir=train_masks_dir,
            split_name="train",
            mask_suffix=args.mask_suffix,
            resize=args.ita_resize,
            lightness_threshold=args.lightness_threshold,
            ita_clip_min=args.ita_clip_min,
            ita_clip_max=args.ita_clip_max,
            crop_pad_frac=args.crop_pad_frac,
            max_images=args.max_images,
            require_mask=args.require_mask,
            save_crops=args.save_crops,
            out_images_dir=(train_img_out if args.save_crops else None),
            out_masks_dir=(train_mask_out if args.save_crops else None),
            csv_writer=writer,
        )

        # Validation appended after training split.
        process_split(
            images_dir=val_images_dir,
            masks_dir=val_masks_dir,
            split_name="val(added)",
            mask_suffix=args.mask_suffix,
            resize=args.ita_resize,
            lightness_threshold=args.lightness_threshold,
            ita_clip_min=args.ita_clip_min,
            ita_clip_max=args.ita_clip_max,
            crop_pad_frac=args.crop_pad_frac,
            max_images=args.max_images,
            require_mask=args.require_mask,
            save_crops=args.save_crops,
            out_images_dir=(train_img_out if args.save_crops else None),
            out_masks_dir=(train_mask_out if args.save_crops else None),
            csv_writer=writer,
        )

    # ---- Test CSV ----
    with open(test_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "skin_tone"])

        process_split(
            images_dir=test_images_dir,
            masks_dir=test_masks_dir,
            split_name="test",
            mask_suffix=args.mask_suffix,
            resize=args.ita_resize,
            lightness_threshold=args.lightness_threshold,
            ita_clip_min=args.ita_clip_min,
            ita_clip_max=args.ita_clip_max,
            crop_pad_frac=args.crop_pad_frac,
            max_images=args.max_images,
            require_mask=args.require_mask,
            save_crops=args.save_crops,
            out_images_dir=(test_img_out if args.save_crops else None),
            out_masks_dir=(test_mask_out if args.save_crops else None),
            csv_writer=writer,
        )

    print("Saved:")
    print(f"  {train_csv_path}")
    print(f"  {test_csv_path}")


if __name__ == "__main__":
    main()

'''
python preprocessing/skin_tone/make_isic2017_skin_tone_csvs.py \
  --data-root /path/to/ISIC_2017 \
  --output-dir /path/to/ISIC_2017
'''

