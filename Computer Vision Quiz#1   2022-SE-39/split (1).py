import cv2
import numpy as np
import os

def split_receipts_master():
    # 1. Load Image
    image_path = "rs.jpg"
    img = cv2.imread(image_path)
    if img is None:
        print("Error: rs.jpg not found!")
        return

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. ADAPTIVE THRESHOLD (The Secret Sauce)
    # This ignores global brightness and looks for local edges/shadows
    # If it still gives 1 file, change the '11' to a higher odd number like '21' or '31'
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 10)

    # 4. CLEAN UP NOISE
    # We remove the text and only keep the large paper shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. FIND INDIVIDUAL PIECES
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create output folder
    output_dir = "extracted_receipts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Detecting pieces...")
    
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # FILTER: Ignore tiny shapes or shapes that are the whole image
        # This ensures we get the 5 separate pieces
        img_h, img_w = img.shape[:2]
        if (w > img_w * 0.1) and (h > img_h * 0.1) and (w < img_w * 0.9):
            count += 1
            # Crop with a tiny margin
            crop = img[y:y+h, x:x+w]
            cv2.imwrite(f"{output_dir}/receipt_{count}.jpg", crop)
            print(f"Saved receipt {count}")

    # 6. DEBUG: This helps you see why it's failing
    # It saves a 'mask.jpg' showing what the computer "saw"
    cv2.imwrite("debug_mask.jpg", close)
    
    if count <= 1:
        print("\n[!] Warning: Only found 1 or 0 receipts.")
        print("Check 'debug_mask.jpg' in your folder. If it's all white, we need to adjust the sensitivity.")
    else:
        print(f"\nSuccess! Found {count} receipts in the '{output_dir}' folder.")

if __name__ == "__main__":
    split_receipts_master()