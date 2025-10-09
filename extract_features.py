import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
import hashlib

# -----------------------------
# Feature extraction functions
# -----------------------------
def get_leaf_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([25, 40, 40]), np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def color_feats(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []
    for arr in [hsv, lab]:
        pix = arr[mask > 0]
        for ch in range(arr.shape[2]):
            feats.append(pix[:,ch].mean() if len(pix) else 0)
            feats.append(pix[:,ch].std() if len(pix) else 0)
    pix_gray = gray[mask > 0]
    feats += [pix_gray.mean() if len(pix_gray) else 0,
              pix_gray.std() if len(pix_gray) else 0]
    hist = cv2.calcHist([hsv], [0], mask, [16], [0,180]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    feats += hist.tolist()
    return np.array(feats[:30]) if len(feats) >= 30 else np.pad(feats, (0,30-len(feats)), 'constant')

def texture_feats(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    # GLCM
    try:
        quantized = (masked / 4).astype(np.uint8)
        glcm = graycomatrix(quantized, [5], [0], levels=64, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0,0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
        energy = graycoprops(glcm, 'energy')[0,0]
        correlation = graycoprops(glcm, 'correlation')[0,0]
    except:
        contrast = homogeneity = energy = correlation = 0
    # LBP
    lbp = local_binary_pattern(masked, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp[mask>0], bins=np.arange(0,18), range=(0,18))
    hist = hist.astype("float") / (hist.sum() + 1e-6)
    # HOG
    try:
        hog_desc = hog(masked, pixels_per_cell=(16,16), cells_per_block=(1,1), feature_vector=True)
        hog_mean, hog_std = np.mean(hog_desc), np.std(hog_desc)
    except:
        hog_mean = hog_std = 0
    # Edge density
    try:
        edges = cv2.Canny(masked, 50, 150)
        edge_density = edges.sum() / (mask.sum() + 1e-6)
    except:
        edge_density = 0
    feats = [contrast, homogeneity, energy, correlation] + hist.tolist() + [hog_mean, hog_std, edge_density]
    return np.array(feats[:25]) if len(feats) >= 25 else np.pad(feats, (0,25-len(feats)), 'constant')

def shape_feats(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return np.array([0,0,0,0])
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4*np.pi*area / (perimeter**2 + 1e-6)
    x,y,w,h = cv2.boundingRect(cnt)
    aspect = w / (h+1e-6)
    return np.array([area, perimeter, circularity, aspect])

def extract_features(img_path):
    img = cv2.imread(img_path)
    mask = get_leaf_mask(img)
    feats = np.hstack([color_feats(img, mask), texture_feats(img, mask), shape_feats(mask)])
    return feats if len(feats)==59 else np.pad(feats, (0, 59-len(feats)), 'constant')

#==========hashing function to avoid duplicate images==========
def get_image_hash(img_path):
    """Return MD5 hash of image content"""
    
    try:
        with open(img_path, "rb") as f:
            data = f.read()
            return hashlib.md5(data).hexdigest()
    except:
        return None
    

# -----------------------------
# Main extraction
# -----------------------------
if __name__ == "__main__":
    dataset_dir = "dataset/train_data"  # dataset folder
    csv_path = "feature_train.csv"

    # Load existing CSV if exists
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)


        if 'hash' not in df_existing.columns:
            # Compute hash for existing images
            df_existing['hash'] = df_existing['path'].apply(get_image_hash)


        existing_hashes = set(df_existing['hash'].tolist()) if 'hash' in df_existing.columns else set()


        
    else:
        df_existing = pd.DataFrame()
        existing_hashes = set()

    rows = []
    total = sum([len(files) for r,d,files in os.walk(dataset_dir)])
    count = 0

    for label in os.listdir(dataset_dir):
        folder = os.path.join(dataset_dir, label)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            path = os.path.join(folder, f)

            # Compute image hash
            img_hash = get_image_hash(path)
            if img_hash in existing_hashes:
                count += 1
                continue  # skip duplicate image

            try:
                feats = extract_features(path)
                rows.append([path, label, img_hash] + feats.tolist())
            except Exception as e:
                print("Error:", path, e)

            count += 1
            if count % 10 == 0 or count == total:
                print(f"[{count}/{total}] Extracted...")

    # Column names
    color_cols = [
        "h_mean","h_std","s_mean","s_std","v_mean","v_std",
        "l_mean","l_std","a_mean","a_std","b_mean","b_std",
        "gray_mean","gray_std"
    ] + [f"h_hist_{i}" for i in range(16)]

    texture_cols = [
        "glcm_contrast","glcm_homogeneity","glcm_energy","glcm_correlation"
    ] + [f"lbp_{i}" for i in range(18)] + ["hog_mean","hog_std","edge_density"]

    shape_cols = ["area","perimeter","circularity","aspect_ratio"]

    cols = ["path","label","hash"] + color_cols + texture_cols + shape_cols

    # Merge new rows with existing
    if rows:
        df_new = pd.DataFrame(rows, columns=cols)
        df_new['label'] = df_new['label'].replace({'non leaf_1': 'Non-leaf', 'non leaf_2': 'Non-leaf'})
        df_existing = pd.concat([df_existing, df_new], ignore_index=True)
        df_existing.to_csv(csv_path, index=False)
        print(f"✓ Features extracted and saved to {csv_path}")
    else:
        print("✓ No new images to extract. CSV is up-to-date.")