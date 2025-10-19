import cv2
import numpy as np
import os

# Allow some low values in other channels to increase tolerance
# white range
LOWER_WHITE = np.array([100, 100, 100])
UPPER_WHITE = np.array([255, 255, 255])

# red range (captures all pixels from dark red to bright red)
LOWER_RED = np.array([0, 0, 100])
UPPER_RED = np.array([100, 100, 255])

# blue range
LOWER_BLUE = np.array([100, 0, 0])
UPPER_BLUE = np.array([255, 100, 100])

# green range
LOWER_GREEN = np.array([0, 80, 0])
UPPER_GREEN = np.array([100, 255, 100])


# target colors
TARGET_FOR_WHITE = np.array([0, 255, 0])   # green
TARGET_FOR_RED = np.array([255, 0, 0])     # blue
TARGET_FOR_BLUE_GREEN = np.array([0, 0, 255]) # red

os.makedirs('new_data_masks/', exist_ok=True)

save_path = 'new_data_masks/'
os.makedirs(save_path, exist_ok=True)
image_path = '../../Lap_Images/patient4/masks_original/'
print("Processing images num:", len(os.listdir(image_path)))

for image_name in os.listdir(image_path):
    i = 1
    image = cv2.imread(os.path.join(image_path, image_name))
    print(f"Processing image: {image_name}")
    if image is None:
        print(f"no file: {image_name}")
        continue

    output_image = image.copy()
    kernel = np.ones((3, 3), np.uint8)


    white_mask = cv2.inRange(image, LOWER_WHITE, UPPER_WHITE)
    white_mask_cleaned = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    output_image[white_mask_cleaned > 0] = TARGET_FOR_WHITE

    red_mask = cv2.inRange(image, LOWER_RED, UPPER_RED)
    red_mask_cleaned = cv2.morphologyEx(red_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
    output_image[red_mask_cleaned] = TARGET_FOR_RED

    blue_mask = cv2.inRange(image, LOWER_BLUE, UPPER_BLUE)
    blue_mask_cleaned = cv2.morphologyEx(blue_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
    output_image[blue_mask_cleaned] = TARGET_FOR_BLUE_GREEN

    green_mask = cv2.inRange(image, LOWER_GREEN, UPPER_GREEN)
    green_mask_cleaned = cv2.morphologyEx(green_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
    output_image[green_mask_cleaned] = TARGET_FOR_BLUE_GREEN

    # split filename and extension to save new image
    base_name, ext = os.path.splitext(image_name)
    new_image_name = f"{base_name}_convert.png"
    cv2.imwrite(f"{save_path}{new_image_name}", output_image)


    print(f"processed, saved as '{image_name}_convert.png'")