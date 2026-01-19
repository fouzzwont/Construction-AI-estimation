# Construction AI Estimation Pipeline

**YOLOv8-powered automated quantity takeoff from construction PDF blueprints.** Converts multi-page drawings into high-resolution images, detects structural elements (rebar, beams, columns, concrete forms), and generates precise cost estimates for construction bidding.

## Features
- PDF blueprint processing with PyMuPDF
- Custom YOLOv8 object detection for construction elements
- Automatic scale detection and real-world measurement conversion
- Material quantity calculation and cost estimation
- CSV reports and PDF summaries for clients
- Production-ready folder structure

## Tech Stack
-Python 3.11
-Ultralytics YOLOv8 (object detection)
-PyMuPDF (PDF processing)
-OpenCV (image preprocessing)
-Pandas (data analysis)
-PyYAML (configuration)
-tqdm (progress bars)

## Sample Output
text
Rebar (Ø16mm): 245 meters    → ₹1,47,000
Rebar (Ø12mm): 180 meters    → ₹72,000
Concrete (M25): 120 m³       → ₹9,60,000
Formwork: 450 m²             → ₹2,25,000
----------------------------------------
TOTAL ESTIMATE: ₹13,32,000
----------------------------------------

