from typing import Optional, Literal

import cv2 as cv
import numpy as np
import kneed
import matplotlib.pyplot as plt

import faiss
import numpy as np
import cv2 as cv

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

def find_dominant_color(img: np.ndarray, label: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Finds the dominant color of the skin in the image.

    Args:
        img: The BGR image to find the dominant color of.
        label: The label of the image. If provided, the dominant color will be found only in the skin area.
    
    Returns:
        The dominant color of the skin in the image as (R, G, B) or (-1, -1, -1) if the dominant color could not be found.
    """
    processed_img = extract_skin(img, label)
    try:
      dominant_color = kmeans_with_kneed_dominant_color(processed_img)
    except Exception as e:
      print('Error processing file: ', e)
      return np.array((-1, -1, -1))
    return dominant_color

def get_fitzpatrick_type(ita_angle: float) -> Literal[12, 34, 56]:
    if ita_angle >= 41:
        return 12
    elif ita_angle >= 19:
        return 34
    else:
        return 56

def get_ita_angle(color_rgb: np.ndarray) -> float:
  color_lab = cv.cvtColor(np.uint8([[color_rgb]]), cv.COLOR_RGB2LAB)[0][0]
  return np.arctan((color_lab[0] - 50) / color_lab[2]) * 180 / np.pi

def extract_skin(img: np.ndarray, label: Optional[np.ndarray] = None) -> np.ndarray:
  """
  Extracts the skin from the image using the label and additional processing
  including DullRazor, CLAHE and thresholding.
  """
  clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
  lab_planes = list(cv.split(lab))
  lab_planes[0] = clahe.apply(lab_planes[0])
  lab = cv.merge(lab_planes)
  clahe_img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)

  hair_mask = get_hair_mask(clahe_img)
  if label is None:
    label = np.zeros_like(hair_mask)
  mask_all = 255 - np.bitwise_or(hair_mask, label)
  masked_img = cv.bitwise_and(clahe_img, clahe_img, mask=mask_all)

  # threshold to remove pigmentations
  hsv = cv.cvtColor(masked_img, cv.COLOR_BGR2HSV)
  _, _, v = cv.split(hsv)
  v = cv.GaussianBlur(v, (5, 5), 0)
  _, v_thresh = cv.threshold(v, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
  thresh = v_thresh

  # expand
  kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
  thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=5)
  thresh = cv.dilate(thresh, kernel, iterations=5)
  thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=5)

  thresh = cv.bitwise_or(thresh, hair_mask)
  final_image = cv.bitwise_and(img, img, mask=255 - thresh)
  return final_image

def get_hair_mask(img):
  grayscale = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  kernel = cv.getStructuringElement(1, (3,3))
  blackhat = cv.morphologyEx(grayscale, cv.MORPH_BLACKHAT, kernel)
  blurred = cv.GaussianBlur(blackhat, (3,3), cv.BORDER_DEFAULT)
  _, hair_mask = cv.threshold(blurred, 25, 255, cv.THRESH_BINARY)
  return hair_mask

def dull_razor(img):
  """
  Applies the DullRazor algorithm to the image.
  img should be an RGB numpy array of shape (H, W, C) between 0 and 255.
  """
  #Gray scale
  grayScale = cv.cvtColor(img, cv.COLOR_RGB2GRAY )
  #Black hat filter
  kernel = cv.getStructuringElement(1,(9,9)) 
  blackhat = cv.morphologyEx(grayScale, cv.MORPH_BLACKHAT, kernel)
  #Gaussian filter
  bhg= cv.GaussianBlur(blackhat,(3,3),cv.BORDER_DEFAULT)
  #Binary thresholding (MASK)
  ret,mask = cv.threshold(bhg,10,255,cv.THRESH_BINARY)
  #Replace pixels of the mask
  dst = cv.inpaint(img,mask,6,cv.INPAINT_TELEA)
  return dst 

def kmeans_dominant_color(processed_img, k):
  processed_img_rgb = cv.cvtColor(processed_img, cv.COLOR_BGR2RGB)
  pixel_values = processed_img_rgb.reshape((-1, 3))
  pixel_values = np.float32(pixel_values)
  pixel_values = pixel_values[np.where(pixel_values[:, 0] > 0)]
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
  compactness, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
  centers = np.uint8(centers)
  labels = labels.flatten()
  dominant_label = np.argmax(np.bincount(labels))
  dominant_color = centers[dominant_label]
  dominant_color = np.array(dominant_color).astype(int)
  return dominant_color, compactness

def kmeans_dominant_color_lab(processed_img, k):
  processed_img_lab = cv.cvtColor(processed_img, cv.COLOR_BGR2LAB)
  pixel_values = processed_img_lab.reshape((-1, 3))
  # remove black pixels
  pixel_values = pixel_values[np.where(pixel_values[:, 0] > 0)]
  # keep only a and b channels
  #pixel_values = pixel_values[:, 1:]
  pixel_values = np.float32(pixel_values)

  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
  compactness, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
  centers = np.uint8(centers)
  labels = labels.flatten()

  dominant_label = np.argmax(np.bincount(labels))
  dominant_color = centers[dominant_label]
  dominant_color = np.array(dominant_color)
  dominant_color = np.round(dominant_color).astype(int)
  # add back L channel
  dominant_color = cv.cvtColor(np.uint8([[dominant_color]]), cv.COLOR_LAB2RGB)
  return dominant_color, compactness

def mean_dominant_color(processed_img):
  processed_img_rgb = cv.cvtColor(processed_img, cv.COLOR_BGR2RGB)
  pixel_values = processed_img_rgb.reshape((-1, 3))
  pixel_values = np.float32(pixel_values)
  pixel_values = pixel_values[np.where(pixel_values[:, 0] > 0)]
  mean_color = np.mean(pixel_values, axis=0)
  mean_color = np.array(mean_color).astype(int)
  return mean_color

def kmeans_with_kneed_dominant_color(processed_img):
  ks = range(2, 10)
  colors, compactnesses = zip(*[kmeans_dominant_color_lab_faiss(processed_img, k) for k in ks])
  kneedle = kneed.KneeLocator(ks, compactnesses, S=1.0, curve='convex', direction='decreasing')
  dominant_color = colors[kneedle.elbow]
  return dominant_color
