from PIL import Image, ImageDraw, ImageEnhance
import requests
import os
import easyocr
import time
import csv
from io import BytesIO
import streamlit as st
import re
import cv2
import pandas as pd  # To display tables
from room_segment import segment_rooms
import numpy as np
import typing
import concurrent.futures
from ocr import crop_and_process_ocr, full_image_ocr


# Global variables and pattern
numeric_pattern = re.compile(r'\d+[,.]?\d*')

def send_to_roboflow(image):
    API_URL = "https://detect.roboflow.com/floor_plan_analysis/4"
    API_KEY = "JXQBztxWh7ccmVxM3TgG"
    img_byte_arr = BytesIO()
    try:
        image.save(img_byte_arr, format='PNG')
    except:
        image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = requests.post(f"{API_URL}?api_key={API_KEY}", files={"file": img_byte_arr})
    return response.json() if response.status_code == 200 else None

def draw_detections(image, predictions):
    for pred in predictions:
        x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
        left, top = int(x - width / 2), int(y - height / 2)
        right, bottom = int(x + width / 2), int(y + height / 2)
        cv2.rectangle(image, (left, top),(right, bottom), (255, 255, 0), 2)
    return image

def get_cropped_image(image, pred):
    x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']
    left, top = int(x - width / 2)-5, int(y - height / 2)-5
    right, bottom = int(x + width / 2)+5, int(y + height / 2)+5

    crop = image.crop((left, top, right, bottom))
    if crop.height > crop.width:
        crop = crop.rotate(90, expand=True)
    
    return crop, pred

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

def display_detection_summary(detection_file):
    """Reads detection results and displays the counts of relevant attributes in a beautiful layout."""
    # Load the CSV file into a DataFrame
    df = pd.read_csv(detection_file)

    # Define the objects of interest and their corresponding emojis
    objects = {
        "inner-door": ("🚪", "Doors"),
        "window": ("🪟", "Window"),
        "bath": ("🛁", "Baths"),
        "sink": ("🚰", "Sinks"),
        "toilet": ("🚽", "Toilets"),
    }

    # Initialize counters
    counters = {key: 0 for key in objects.keys()}

    # toilets minimum is 1
    counters["toilet"] = 1

    # Count occurrences in the DataFrame
    for attribute in df["Attributes"]:
        for key in counters:
            if key in attribute:
                counters[key] += 1

    # Display summary with style
    st.markdown("---")
    st.markdown("## 🔍 Detection Summary")
    st.write("The following items were detected in the floor plan:")

    # Create horizontal layout using columns
    col1, col2, col3, col4, col5 = st.columns(5)

    # Style for each counter block
    counter_style = """
    <div style="
        border: 2px solid #4CAF50; 
        border-radius: 15px; 
        padding: 20px; 
        text-align: center; 
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1); 
        background-color: #f9f9f9;
    ">
        <h1 style="color: #4CAF50; font-size: 48px;">{count}</h1>
        <p style="font-size: 24px;">{emoji} {label}</p>
    </div>
    """

    # Populate columns with styled counters
    for (key, (emoji, label)), col in zip(objects.items(), [col1, col2, col3, col4, col5]):
        with col:
            html = counter_style.format(count=counters[key], emoji=emoji, label=label)
            st.markdown(html, unsafe_allow_html=True)


def main():
    st.set_page_config(layout="wide")
    st.title("🏢 Floor Plan Analyzer with Streamlit")

    # Upload Section
    uploaded_file = st.file_uploader("📁 Upload a Floor Plan Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file:
        with st.spinner("Processing the uploaded floor plan..."):
            time.sleep(3) 

        image = Image.open(uploaded_file)

        # Row 1: Display Uploaded Image
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("## 1️⃣ Uploaded Floor Plan")
                st.write(
                    "The original floor plan image uploaded for analysis. "
                    "This image will undergo room segmentation, object detection, and OCR processing."
                )
            with col2:
                st.image(image, caption="Uploaded Floor Plan", use_column_width=True)
        time.sleep(1)
        # Row 2: Room Segmentation
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("## 2️⃣ Room Segmentation")
                st.write(
                    "In this step, the floor plan is segmented to detect individual rooms. "
                    "Each detected room is highlighted with a unique color to differentiate them."
                )
                with st.spinner("🔍 Segmenting rooms..."):
                    cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    segmented_rooms, image_with_rooms = segment_rooms(cv2_image)
            with col2:
                st.image(image_with_rooms, caption="Segmented Rooms", use_column_width=True)

        # Row 3: Object Detection Inference
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("## 3️⃣ Object Detection Inference")
                st.write(
                    "This step uses the **RoboFlow API** to detect objects such as furniture and utilities. "
                    "The detections are marked on the floor plan, helping to identify specific objects within the rooms."
                )
                with st.spinner("⚙️ Running inference..."):
                    roboflow_result = send_to_roboflow(image)
                    if roboflow_result:
                        predictions = roboflow_result['predictions']
                        image_with_detections = draw_detections(image_with_rooms.copy(), predictions)
                        save_detection_results(predictions, "outputs/model_results.csv")
                        obj_coords = crop_non_living_room_objects(predictions)
            with col2:
                st.image(image_with_detections, caption="Detected Objects", use_column_width=True)

        # Row 4: OCR Processing
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("## 4️⃣ OCR Processing")
                st.write(
                    "After detecting objects, **Optical Character Recognition (OCR)** is applied to extract any numeric "
                    "information from the detected objects, such as room sizes or furniture dimensions."
                )

                with st.spinner("📝 Performing OCR..."):
                    image_copy = image.copy()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_crop = executor.submit(crop_and_process_ocr, image, uploaded_file.name, predictions) 
                        future_full_image = executor.submit(full_image_ocr, image_copy, uploaded_file.name)

                        ocr_csv_path, total_area_1, total_area_2 = future_crop.result()
                        total_area_3 = future_full_image.result()

            with col2:
                st.table(pd.read_csv(ocr_csv_path)) 

        # Row 5: Total Living Area Calculation
        with st.container():

            living_rooms_count = 0
            for segmented_room in segmented_rooms:
                living_room = False
                for obj_coord in obj_coords:
                    in_room = is_coordinate_in_contour(obj_coord, segmented_room)
                    if in_room and obj_coord[-1]: # if living type object inside room 
                        living_room = True
                        living_rooms_count += 1
                        break # proceed to next room
                    elif in_room and not obj_coord[-1]: # if not living type object inside room
                        break

            living_rooms_count = max(1, living_rooms_count)
            
            # print(living_rooms_count)

            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("## 5️⃣ Total Living Area Calculation")
                st.write(
                    "Finally, the total living area is calculated based on the numeric data extracted via OCR. "
                    "This result can provide useful insights for architectural planning or real estate purposes."
                )
            with col2:
                st.write(f"## 🏠 Total Area Predict 1: {total_area_1} sq. units")
                st.write(f"## 🏠 Total Area Predict 2: {total_area_2} sq. units")
                st.write(f"## 🏠 Total Area Predict 3: {total_area_3} sq. units")
                st.write(f"## 🛋️ Total living rooms: {living_rooms_count}")
                display_detection_summary('outputs/model_results.csv')

    
# Run the app
if __name__ == "__main__":
    main()
