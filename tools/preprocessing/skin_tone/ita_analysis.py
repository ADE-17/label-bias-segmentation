import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import cv2 as cv
from PIL import Image

# Import from the file you provided
from kmeans_skin_color_estimator import find_dominant_color, get_ita_angle

import faiss
import numpy as np
import cv2 as cv

# from cuml.cluster import KMeans

def kmeans_dominant_color_lab_faiss(processed_img, k):
    processed_img_lab = cv.cvtColor(processed_img, cv.COLOR_BGR2LAB)
    pixel_values = processed_img_lab.reshape((-1, 3))
    pixel_values = pixel_values[np.where(pixel_values[:, 0] > 0)]
    
    # Faiss requires float32 contiguous arrays
    pixel_values = np.ascontiguousarray(pixel_values, dtype=np.float32)

    # Initialize Faiss K-Means
    # d=3 (for L, a, b channels), k=number of clusters
    kmeans = faiss.Kmeans(d=3, k=k, niter=100, nredo=10, gpu=True)
    
    # Train it
    kmeans.train(pixel_values)
    
    # The cluster centers
    centers = kmeans.centroids
    
    # To get the "compactness" (inertia) and labels, we map the pixels to the centers
    distances, labels = kmeans.index.search(pixel_values, 1)
    compactness = np.sum(distances)
    
    labels = labels.flatten()
    dominant_label = np.argmax(np.bincount(labels))
    
    dominant_color = centers[dominant_label]
    dominant_color = np.round(dominant_color).astype(int)
    
    # Add back L channel
    dominant_color = cv.cvtColor(np.uint8([[dominant_color]]), cv.COLOR_LAB2RGB)
    
    return dominant_color, compactness

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def list_images(root: str) -> List[str]:
    """Recursively traverse the directory and return all image paths."""
    image_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = os.path.splitext(filename)[-1].lower()
            if ext in IMG_EXTS:
                image_paths.append(os.path.join(dirpath, filename))
    image_paths.sort()
    return image_paths

def get_mask_path(image_path: str, masks_dir: str, mask_suffix: str = "_segmentation") -> str | None:
    if not masks_dir: return None
    basename = os.path.basename(image_path)
    name, _ = os.path.splitext(basename)
    mask_path = os.path.join(masks_dir, f"{name}{mask_suffix}.png")
    return mask_path if os.path.exists(mask_path) else None

def ita_to_fitzpatrick(ita_mean: float) -> str:
    if np.isnan(ita_mean): return "Unknown"
    if ita_mean > 55.0: return "Very Light"
    if ita_mean > 41.0: return "Light"
    if ita_mean > 28.0: return "Intermediate"
    if ita_mean > 10.0: return "Tan"
    if ita_mean > -30.0: return "Brown"
    return "Dark"

def calculate_ita_ssynth(img_path: str, mask_path: str | None) -> float:
    """
    S-Synth approach: Uses Median + Standard Deviation to aggressively 
    filter out artifacts, vignettes, and stray lesion pixels.
    """
    image = cv.imread(img_path, cv.IMREAD_COLOR)
    
    # 1. Load mask (if available) or create a blank one
    if mask_path:
        # Assuming mask > 0 is lesion
        mask_img = cv.imread(mask_path, cv.IMREAD_COLOR)[:, :, 1]
        mask = (mask_img > 0).astype(int)
    else:
        mask = np.zeros(image.shape[:2], dtype=int)

    # 2. Add Black Vignette to the mask (L < 20)
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    L_full = lab[:, :, 0] * (100.0 / 255.0) # OpenCV L is 0-255, convert to 0-100
    vignette_mask = (L_full < 20.0).astype(int)
    
    # Combine lesion mask and vignette mask (1 = bad pixel, 0 = good skin)
    combined_mask = np.bitwise_or(mask, vignette_mask)

    # 3. Extract valid channels
    L_channel = lab[:, :, 0] * (100.0 / 255.0) 
    b_channel = lab[:, :, 2] - 128.0 # OpenCV b is 0-255, shift to -128 to 128
    
    valid_L = L_channel[combined_mask == 0]
    valid_b = b_channel[combined_mask == 0]

    if len(valid_L) == 0:
        return float('nan')

    # 4. Statistical Filtering (Median +/- 1 Std Dev)
    L_median_temp = np.median(valid_L)
    L_std = np.std(valid_L)
    L_filtered = valid_L[(valid_L >= L_median_temp - L_std) & (valid_L <= L_median_temp + L_std)]
    L_final = np.median(L_filtered) if len(L_filtered) > 0 else L_median_temp

    b_median_temp = np.median(valid_b)
    b_std = np.std(valid_b)
    b_filtered = valid_b[(valid_b >= b_median_temp - b_std) & (valid_b <= b_median_temp + b_std)]
    b_final = np.median(b_filtered) if len(b_filtered) > 0 else b_median_temp

    # 5. Calculate ITA
    ita = np.arctan((L_final - 50) / (b_final + 1e-8)) * (180 / np.pi)
    return float(ita)

def calculate_ita_kmeans(img_path: str, mask_path: str | None) -> float:
    """
    K-Means approach: Uses the custom logic from kmeans_skin_color_estimator.py
    """
    img = cv.imread(img_path)
    
    label = None
    if mask_path:
        mask_img = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        label = (mask_img > 127).astype(np.uint8) * 255 # Format expected by extract_skin

    dominant_color = find_dominant_color(img, label)
    
    # Check for failure (-1, -1, -1)
    if np.array_equal(dominant_color, np.array([-1, -1, -1])):
        return float('nan')
        
    # Squeeze needed because find_dominant_color might return a nested array
    ita = get_ita_angle(dominant_color.squeeze())
    return float(ita)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ITA stats for skin lesions.")
    parser.add_argument("--images-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, default=None)
    parser.add_argument("--mask-suffix", type=str, default="_segmentation")
    parser.add_argument("--method", type=str, choices=['ssynth', 'kmeans'], default='ssynth', help="Method for ITA calculation.")
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--grid-output", type=str, default=None)
    parser.add_argument("--grid-per-type", type=int, default=10)
    return parser.parse_args()

def main():
    from tqdm import tqdm

    args = parse_args()
    paths = list_images(args.images_dir)
    if args.max_images:
        paths = paths[:args.max_images]

    print(f"Found {len(paths)} images. Using '{args.method}' method.")

    ita_means, categories = [], []
    
    for i, p in enumerate(tqdm(paths, desc="Processing", unit="img"), start=1):
        mask_path = get_mask_path(p, args.masks_dir, args.mask_suffix)

        if args.method == 'ssynth':
            ita = calculate_ita_ssynth(p, mask_path)
        else:
            ita = calculate_ita_kmeans(p, mask_path)

        ita_means.append(ita)
        categories.append(ita_to_fitzpatrick(ita))
        
        if i % 50 == 0 or i == len(paths):
            print(f"[{i}/{len(paths)}] {os.path.basename(p)}: ITA={ita:.2f} ({categories[-1]})")

    # CSV Output
    out_csv = args.out_csv or os.path.join(args.images_dir, f"ita_stats_{args.method}.csv")
    import csv
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "ita_mean", "category"])
        for p, m, c in zip(paths, ita_means, categories):
            writer.writerow([p, m, c])
    print(f"\nSaved per-image ITA stats to {out_csv}")

    # Grid Output
    if args.grid_output is not None:
        by_cat = {}
        for p, c in zip(paths, categories):
            if c != "Unknown":
                by_cat.setdefault(c, []).append(p)

        cats_order = ["Very Light", "Light", "Intermediate", "Tan", "Brown", "Dark"]
        thumb_size = (224, 224)
        rows = []
        for cat in cats_order:
            ps = by_cat.get(cat, [])[: args.grid_per_type]
            if not ps: continue
            thumbs = [Image.open(p).convert("RGB").resize(thumb_size) for p in ps]
            row_img = Image.new("RGB", (thumb_size[0] * len(thumbs), thumb_size[1]))
            for idx, im in enumerate(thumbs):
                row_img.paste(im, (idx * thumb_size[0], 0))
            rows.append((cat, row_img))

        if rows:
            max_w = max(row.size[0] for _, row in rows)
            grid = Image.new("RGB", (max_w, thumb_size[1] * len(rows)), color=(0, 0, 0))
            for y, (cat, row) in enumerate(rows):
                grid.paste(row, ((max_w - row.size[0]) // 2, y * thumb_size[1]))
            grid.save(args.grid_output)
            print(f"Saved ITA category grid to {args.grid_output}")

if __name__ == "__main__":
    main()

    '''
python ita_analysis.py \
  --images-dir /path/to/ISIC_2017/val_images \
  --masks-dir /path/to/ISIC_2017/val_masks \
  --method kmeans \
  --grid-output ./ita_grid_kmeans_val.png \
  --max-images 2000 \
  --out-csv ./ita_stats_kmeans_val.csv 

python ita_analysis.py \
  --images-dir /path/to/ISIC_2017/training_images \
  --masks-dir /path/to/ISIC_2017/training_masks \
  --method ssynth \
  --grid-output ./ita_grid_ssynth.png \
  --max-images 100 \
  --out-csv ./ita_stats_ssynth.csv
    '''