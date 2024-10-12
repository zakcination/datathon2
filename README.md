<<<<<<< HEAD
# 🏠 Floor Plan Analyzer with Room Segmentation and OCR

This repository contains a **Streamlit web application** that performs the following operations:

1. **Room segmentation** with unique colors drawn on each room.
2. **Object detection** on floor plans using RoboFlow.
3. **OCR processing** using EasyOCR to extract text from detected objects.
4. **Living area calculation** and display of results in a dynamic table.

## 🎯 Features

- **Upload Floor Plan Image:** Users can upload PNG, JPG, or JPEG files.
- **Room Segmentation:** Paints room borders in unique colors.
- **Detection Inference:** Uses RoboFlow API to detect objects and display them on the original image.
- **OCR Extraction:** Extracts numeric data and calculates the total living area.
- **Interactive UI:** Displays segmented images, detection results, and a detailed OCR table.

## 🛠️ Technologies Used

- **Python**  
- **Streamlit** – For building the interactive web interface  
- **EasyOCR** – For extracting text from images  
- **RoboFlow API** – For object detection on floor plans  
- **OpenCV** – For image manipulation and room segmentation  
- **Pandas** – For tabular data handling  

---

## 📑 Prerequisites

Make sure you have the following installed:

- **Python 3.8+**
- **Pip (Python package installer)**

Install the required libraries using the command:

```bash
pip install -r requirements.txt
=======
# datathon_team_name
A Streamlit-based Floor Plan Analyzer that segments rooms, performs object detection using RoboFlow, and extracts text from detected areas with EasyOCR. This solution visualizes room segmentation, detection results, and calculates the total living area, providing an interactive web interface with vertically aligned content.
>>>>>>> f23c791fb2a5f224268fe4504e9fc47f7208c4df
