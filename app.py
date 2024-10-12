from PIL import Image, ImageDraw
import requests
import os
import easyocr
import csv
from io import BytesIO
import streamlit as st
import re
import cv2
import pandas as pd  # To display tables
from room_segment import segment_rooms
import numpy as np

# Global variables and pattern
numeric_pattern = re.compile(r'\d+[,.]?\d*')

def send_to_roboflow(image):
    API_URL = "https://detect.roboflow.com/floor_plan_analysis/4"
    API_KEY = "JXQBztxWh7ccmVxM3TgG"
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = requests.post(f"{API_URL}?api_key={API_KEY}", files={"file": img_byte_arr})
    return response.json() if response.status_code == 200 else None


def draw_detections(image, predictions):
    for pred in predictions:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        left, top = int(x - width / 2), int(y - height / 2)
        right, bottom = int(x + width / 2), int(y + height / 2)
        cv2.rectangle(image, (left, top),(right, bottom), (0, 255, 0), 2)
    return image

def get_cropped_image(image, pred):
    x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
    left, top = int(x - width / 2), int(y - height / 2)
    right, bottom = int(x + width / 2), int(y + height / 2)

    crop = image.crop((left, top, right, bottom))
    if crop.height > crop.width:
        crop = crop.rotate(90, expand=True)
    
    return crop, pred

def perform_ocr_on_crops(crops, ocr_output_folder):
    reader = easyocr.Reader(['en', 'ru'], gpu=True)
    ocr_csv_path = os.path.join(ocr_output_folder, "ocr_results.csv")
    total_sum = 0

    with open(ocr_csv_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Class', 'Area', 'Living Room'])

        for idx, (crop, pred) in enumerate(crops):
            image_path = os.path.join(ocr_output_folder, f"crop_{idx}.png")
            crop.save(image_path)
            area = extract_numeric_area(reader, image_path)
            total_sum += area
            room_type = pred['class']
            is_living_room = "Yes" if room_type in ['sofa', 'bed'] else "No"
            writer.writerow([f"crop_{idx}", room_type, area, is_living_room])

    return ocr_csv_path, total_sum


def extract_numeric_area(reader, image_path):
    result = reader.readtext(image_path)
    detected_text = " ".join([res[1] for res in result])
    numeric_values = re.findall(numeric_pattern, detected_text)
    return sum(float(val.replace(',', '.')) for val in numeric_values) if numeric_values else 0.0


def save_detection_results(predictions, csv_path):
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Attributes'])  
        for idx, pred in enumerate(predictions):
            attributes = f"{pred['class']}, ({pred['x']}, {pred['y']}, {pred['width']}, {pred['height']})"
            writer.writerow([f"crop_{idx}", attributes])      


def crop_non_living_room_objects(predictions):
    coords = []
    non_living_objects = ['bath', 'sink', 'toilet', 'washing-machine', 'kitchen', 'kitchen-table']
    living_objects = ['sofa', 'bed']
    for pred in predictions:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        if pred['class'] in non_living_objects:
            coords.append((x, y, width, height, False)) # non living room
        elif pred['class'] in living_objects:
            coords.append((x, y, width, height, True))  # living room

    return coords


def is_coordinate_in_contour(coord, contour):
    x, y, width, height, _ = coord

    # point = np.array([[x, y]]) # or use w, h ??
    distance = cv2.pointPolygonTest(contour[0], (x, y), True)

    return distance >= 0


def crop_class_10_objects(image, predictions):
    """Crop objects of class '10' and return cropped images with predictions."""
    crops = []
    for pred in predictions:
        if pred['class'] == "10":
            crop, pred_data = get_cropped_image(image, pred)
            crops.append((crop, pred_data))

    # Remove the largest crop by resolution if any
    areas = [crop[0].size[0] * crop[0].size[1] for crop in crops]
    if areas:
        smallest_area = min(areas)

        # Drop crops where the area is more than twice the smallest crop
        crops = [crop for crop, area in zip(crops, areas) if area <= 2 * smallest_area]

    return crops



def perform_ocr_on_crops(crops, ocr_output_folder):
    """Perform OCR on cropped images and save results in CSV."""
    reader = easyocr.Reader(['en', 'ru'], gpu=True)
    ocr_csv_path = os.path.join(ocr_output_folder, "ocr_results.csv")
    total_sum = 0  # Initialize total area sum

    # Open the CSV file only once to ensure correct writing
    with open(ocr_csv_path, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Area', 'Coordinates'])  # Write headers

        # Iterate over each cropped image and process OCR
        for idx, (crop, pred) in enumerate(crops):
            image_path = os.path.join(ocr_output_folder, f"crop_{idx}.png")
            crop.save(image_path)  # Save cropped image

            # Extract area from OCR result
            area = extract_numeric_area(reader, image_path)
            total_sum += area  # Accumulate the total area

            # Prepare coordinates string from predictions
            coordinates = f"{pred['x']}, {pred['y']}, {pred['width']}, {pred['height']}"

            # Write each row with crop ID, area, and coordinates
            writer.writerow([f"crop_{idx}", area, coordinates])

    return ocr_csv_path, total_sum  # Return path and total area

def extract_numeric_area(reader, image_path):
    """Extract numeric area values from OCR results."""
    result = reader.readtext(image_path)
    detected_text = " ".join([res[1] for res in result])
    
    numeric_values = re.findall(numeric_pattern, detected_text)

    # Convert extracted values to floats and sum them
    if numeric_values:
        area = sum(float(val.replace(',', '.')) for val in numeric_values)
    else:
        area = 0.0  # Default to 0 if no numeric values are found

    return area

def main():
    st.set_page_config(layout="wide")  # Ensure full-width layout

    st.title("Floor Plan Analyze")

    # Define two vertical sections using st.columns() with equal width
    left_col, right_col = st.columns([1, 1])

    with left_col:
        uploaded_file = st.file_uploader("Upload a floor plan image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        right_col.image(image, caption="Uploaded Floor Plan", use_column_width=True)

        with left_col:
            st.write("### Room Segmentation")
            with st.spinner("Looking for rooms..."):
                cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                segmented_rooms, image_with_rooms = segment_rooms(cv2_image)
            right_col.image(image_with_rooms, caption="Rooms Segmented", use_column_width=True)

        with left_col:
            st.write("### Detection Inference")
            with st.spinner("Running inference..."):
                roboflow_result = send_to_roboflow(image)
                if roboflow_result:
                    predictions = roboflow_result['predictions']
                    image_with_detections = draw_detections(image_with_rooms.copy(), predictions)
                    save_detection_results(predictions, "model_results.csv")
                    right_col.image(image_with_detections, caption="Detections on Floor Plan", use_column_width=True)
                    crops = crop_class_10_objects(image, predictions)

        with left_col:
            st.write("### OCR Processing")
            obj_coords = crop_non_living_room_objects(predictions)

            ocr_output_folder = "ocr_crop"
            os.makedirs(ocr_output_folder, exist_ok=True)

            with st.spinner("Performing OCR..."):
                ocr_csv_path, total_sum = perform_ocr_on_crops(crops, ocr_output_folder)

        with right_col:
            st.write("### Detected Rooms and Areas")
            st.table(pd.read_csv(ocr_csv_path))  # Display the OCR results table

            st.write(f"## Total Living Area: {total_sum}")

# Run the app
if __name__ == "__main__":
    main()
