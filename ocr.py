import re
from PIL import Image
import requests
from io import BytesIO
import os
import easyocr
import csv
import numpy as np
import cv2
import concurrent.futures

# Clean detected numbers and convert to float
def clean_detected_number(text):
    text = re.sub(r'м2|м²|м', '', text)
    text = text.replace(',', '.')
    text = re.sub(r'(\d)\s+(\d)', r'\1.\2', text)
    
    if '/' in text:
        numbers = text.split('/')
        try:
            return float(numbers[1])
        except ValueError:
            return None  
    else:
        cleaned_text = re.sub(r'[^\d.]', '', text)
        try:
            return float(cleaned_text)
        except ValueError:
            return None  

# Crop processing function
def crop_and_process_ocr(image, image_path, predictions, output_folder="crop", csv_file="ocr_results.csv"):
    if not isinstance(image, Image.Image):
        raise ValueError("The image passed to the function is not valid.")

    reader = easyocr.Reader(['en', 'ru'], gpu=True)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    areas = []  # To store crop areas
    crop_results = []
    unique_id = 0

    # First pass: Extract areas and detected numbers, but don't decide status yet
    for idx, pred in enumerate(predictions):
        if pred['class'] == "10":
            x, y, width, height = pred['x'], pred['y'], pred['width'], pred['height']

            # Calculate bounding box
            left = int(x - width / 2) - 5
            top = int(y - height / 2) - 3
            right = int(x + width / 2) + 5
            bottom = int(y + height / 2) + 3

            left = max(0, left)
            top = max(0, top)
            right = min(image.width, right)
            bottom = min(image.height, bottom)

            # Crop the image based on the bounding box
            cropped_img = image.crop((left, top, right, bottom))
            cropped_img_np = np.array(cropped_img)
            gray_img = cv2.cvtColor(cropped_img_np, cv2.COLOR_RGB2GRAY)
            _, binary_img = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((2, 2), np.uint8)
            dilated_img = cv2.dilate(binary_img, kernel, iterations=1)
            dilated_img_pil = Image.fromarray(cv2.bitwise_not(dilated_img))

            # Calculate the area of the crop
            crop_area = (right - left) * (bottom - top)
            areas.append(crop_area)

            # Run OCR on the dilated image
            result = reader.readtext(np.array(dilated_img_pil))
            detected_numbers = []

            for res in result:
                # Convert detected numbers to float and clean them
                cleaned_text = clean_detected_number(res[1])
                if cleaned_text is not None:
                    detected_numbers.append(float(cleaned_text))  # Ensure the detected numbers are float

            detected_area = sum(detected_numbers)

            # Save the initial crop data, without the status
            crop_results.append({
                'unique_id': unique_id,
                'image_path': image_path,
                'detected_area': detected_area,
                'coordinates': (x, y, width, height),
                'crop_area': crop_area,
                'dilated_img_pil': dilated_img_pil  # Save the image for later
            })

            unique_id += 1

    # Second pass: Decide status (good or max) based on crop area
    if areas:
        min_area = min(areas)  # Get the minimum area

    for crop_result in crop_results:
        crop_area = crop_result['crop_area']
        detected_area = crop_result['detected_area']

        # Determine status
        if abs(crop_area - min_area) <= min_area//2:  # Close enough to the minimum area
            status = "good"
        else:
            status = "max"

        # Adjust detected area for large "good" areas
        if status == "good" and detected_area > 120:
            detected_area = detected_area / 10  # Adjust detected area

        # Save the crop image
        crop_filename = f"{output_folder}/crop_{image_path}_{crop_result['unique_id']}.png"
        crop_result['dilated_img_pil'].save(crop_filename)

        # Update the crop result with the final status and adjusted area
        crop_result['status'] = status
        crop_result['detected_area'] = detected_area

    # Write the processed crops to the CSV file
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Unique ID', 'Image Path', 'Area', 'Coordinates', 'Crop Area', 'Status'])

        for crop_result in crop_results:
            writer.writerow([
                crop_result['unique_id'],
                crop_result['image_path'],
                crop_result['detected_area'],
                crop_result['coordinates'],
                crop_result['crop_area'],
                crop_result['status']
            ])

    # Summing up the "good" and "max" areas
    total_good_area = sum([r['detected_area'] for r in crop_results if r['status'] == 'good'])
    max_area = max([r['detected_area'] for r in crop_results if r['status'] == 'max'], default=0)

    # Append the total good area and max area to the CSV
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Total Good Area', total_good_area])
        writer.writerow(['Max Area', max_area])
    print("total_good_area", total_good_area)
    print("Max Area", max_area)

    return csv_file, total_good_area, max_area


def extract_numbers_from_text(text):
    text = re.sub(r'м2|м²|м', '', text)
    text = text.replace(',', '.')
    cleaned_text = re.findall(r'[\d.]+', text)
    
    if cleaned_text:
        try:
            return float(cleaned_text[0])  
        except ValueError:
            return None
    return None

def full_image_ocr(image, image_path):
    predictions_with_numbers = []

    reader = easyocr.Reader(['en', 'ru'], gpu=True)
    image_np = np.array(image)
    results = reader.readtext(image_np)

    # print(f"OCR results for {image_path}:")
    # print(results)

    total_good_area = 0.0
    total_max_area = 0.0
    max_area = 0.0
    areas = []  

    for result in results:
        coordinates = result[0]  
        text = result[1]  
        confidence = result[2]  

        extracted_number = extract_numbers_from_text(text)

        if extracted_number is not None:
            left = min(coordinates, key=lambda x: x[0])[0]
            right = max(coordinates, key=lambda x: x[0])[0]
            top = min(coordinates, key=lambda x: x[1])[1]
            bottom = max(coordinates, key=lambda x: x[1])[1]

            crop_area = (right - left) * (bottom - top)
            areas.append(crop_area)  

            status = ""

            min_area = min(areas) if areas else 0

            if abs(crop_area - min_area) <= min_area//2:
                status = "good"
            else:
                status = "max"
                
            if status == "good" and extracted_number > 120:
                extracted_number = extracted_number / 10  

            predictions_with_numbers.append({
                'Area': extracted_number,
                'Coordinates': coordinates,
                'Status': status,
                'Crop Area': crop_area
            })

            # print(f"Status: {status}, Coordinates: {coordinates}, Area: {extracted_number}, Crop Area: {crop_area}")
    
    for row in predictions_with_numbers:
        detected_area = float(row['Area'])
        status = row['Status']

        if status == 'good':
            total_good_area += detected_area
        elif status == 'max':
            max_area = max(max_area, detected_area)
            total_max_area += detected_area
                
    total_area = max_area if max_area and total_good_area > max_area else total_good_area

    # print(f"Final Total Area: {total_area}")
    
    return total_area
