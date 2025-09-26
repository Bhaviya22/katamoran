import cv2, os

def save_crop(img, out_dir, prefix="face"):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{prefix}.jpg")
    cv2.imwrite(filename, img)
    return filename
