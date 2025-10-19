import cv2, os, random
import numpy as np

input_dir = "dataset/raw_data"
output_dir = "dataset/test_data"
os.makedirs(output_dir, exist_ok=True)

def augment_image(img):
    # Random rotation
    angle = random.randint(-30, 30)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))

    # Random flip
    if random.choice([True, False]):
        img = cv2.flip(img, 1)

    # Random brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = random.randint(-30, 30)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + value, 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img

for label in os.listdir(input_dir):
    in_path = os.path.join(input_dir, label)
    out_path = os.path.join(output_dir, label)
    os.makedirs(out_path, exist_ok=True)

    for img_name in os.listdir(in_path):
        img = cv2.imread(os.path.join(in_path, img_name))
        if img is None:
            continue

        for i in range(5):  # number of augmentations
            aug = augment_image(img)
            save_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(out_path, save_name), aug)
