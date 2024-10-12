# 🏢 Floor Plan Analyzer with Streamlit, RoboFlow, and EasyOCR

## 🎯 Datathon Solution: Unlocking Space Optimization with AI 🏆

**Problem:** Accurate floor plan analysis is crucial for architecture, real estate, and smart building systems, but manual measurement is time-consuming and error-prone.

**Our Solution:** 
We present an **AI-powered Floor Plan Analyzer** that leverages **room segmentation, object detection, and OCR** to extract room details from floor plans, enabling smart decision-making for space optimization.

---

## ✨ Features

- **Room Segmentation:** Automatically detects and paints room boundaries with unique colors.
- **Object Detection:** Uses RoboFlow to detect objects in the floor plan and overlays bounding boxes.
- **OCR Processing:** Extracts numeric values (like room sizes) with EasyOCR.
- **Living Area Calculation:** Accurately computes total living area for optimization.
- **Interactive Web Interface:** A user-friendly, vertically aligned UI for smooth navigation.

---

## 🚀 How It Works

1. **Upload a Floor Plan Image:** 
   - Upload PNG, JPG, or JPEG images using the file uploader.

2. **Room Segmentation:** 
   - The model identifies rooms and draws colored boundaries.

3. **Object Detection:** 
   - Runs inference using RoboFlow and overlays detected objects on the floor plan.

4. **OCR Processing:** 
   - Extracts numeric values from objects and calculates the living area.

5. **Results Display:** 
   - A table with room details (ID, Class, Area, and Room Type) and the total living area.
