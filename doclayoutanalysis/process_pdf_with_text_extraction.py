import os
import cv2
import json
from pdf2image import convert_from_path
from doclayout_yolo import YOLOv10
import pytesseract  # For OCR
from huggingface_hub import hf_hub_download

# Step 1: Load the YOLO Model
def load_model(model_path):
    return YOLOv10(model_path)

# Step 2: Convert PDF to Images
def pdf_to_images(pdf_path, output_folder, dpi=200):
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i+1}.jpg")
        image.save(image_path, "JPEG")
        image_paths.append(image_path)
    return image_paths

# Step 3: Extract Text for Each Label
def extract_text_for_label(image, coordinates):
    x_min, y_min, x_max, y_max = coordinates
    cropped = image[y_min:y_max, x_min:x_max]  # Crop the region of interest
    text = pytesseract.image_to_string(cropped)  # Perform OCR
    return text.strip()

# Step 4: Perform Layout Detection and Extract Text
def detect_and_extract_text(model, image_path, class_mapping, conf=0.2, imgsz=1024, device="cuda:0"):
    # Perform detection
    det_res = model.predict(
        image_path,
        imgsz=imgsz,
        conf=conf,
        device=device
    )
    
    # Load the image for OCR
    image = cv2.imread(image_path)

    # Prepare results for JSON
    structured_content = []
    for box in det_res[0].boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())  # # Bounding box coordinates
        print("box.xyxy:", box.xyxy)
        label_index = int(box.cls)  # Class label index
        confidence = float(box.conf)  # Confidence score

        # Get label name using the class mapping
        label_name = class_mapping.get(label_index, f"Unknown_{label_index}")

        # Extract text for the detected region
        extracted_text = extract_text_for_label(image, (x_min, y_min, x_max, y_max))

        # Add the structured content
        structured_content.append({
            "label": label_name,
            "coordinates": [x_min, y_min, x_max, y_max],
            "confidence": confidence,
            "extracted_text": extracted_text
        })

    return structured_content

# Step 5: Save Structured Results to JSON
def save_results_to_json(pdf_path, results, output_folder):
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    json_path = os.path.join(output_folder, f"{pdf_name}_structured_results.json")
    with open(json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to: {json_path}")

# Step 6: Process the Entire PDF
def process_pdf(pdf_path, model_path, output_folder, class_mapping, dpi=200, conf=0.2, imgsz=1024, device="cpu"):
    # Load the YOLO model
    model = load_model(model_path)
    
    # Convert PDF to images
    temp_image_folder = "./temp_images"
    image_paths = pdf_to_images(pdf_path, temp_image_folder, dpi=dpi)

    # Detect layout and extract text for each image
    structured_results = []
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        structured_content = detect_and_extract_text(
            model, image_path, class_mapping, conf=conf, imgsz=imgsz, device=device
        )
        structured_results.append({
            "page": os.path.basename(image_path),
            "content": structured_content
        })

    # Save all results to JSON
    save_results_to_json(pdf_path, structured_results, output_folder)

    # Clean up temporary images
    for image_path in image_paths:
        os.remove(image_path)
    os.rmdir(temp_image_folder)

# Step 7: Entry Point
if __name__ == "__main__":
    # Paths
    pdf_path = "/workspaces/Document-Processing-System/doclayoutanalysis/inputs/2024020622.pdf"  
    model_path = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt"
    )  
    output_folder = "./output_jsons"           # Output folder for JSON results

    # Class mapping 
    class_mapping = {
        0: 'title', 
        1: 'plain text',
        2: 'abandon', 
        3: 'figure', 
        4: 'figure_caption', 
        5: 'table', 
        6: 'table_caption', 
        7: 'table_footnote', 
        8: 'isolate_formula', 
        9: 'formula_caption'
}


    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process the PDF with structured results
    process_pdf(pdf_path, model_path, output_folder, class_mapping)
