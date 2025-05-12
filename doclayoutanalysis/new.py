# import os
# import cv2
# import json
# from pdf2image import convert_from_path
# from doclayout_yolo import YOLOv10
# import pytesseract  # For OCR
# from huggingface_hub import hf_hub_download

# # Step 1: Load the YOLO Model
# def load_model(model_path):
#     return YOLOv10(model_path)

# # Step 2: Convert PDF to Images
# def pdf_to_images(pdf_path, output_folder, dpi=200):
#     os.makedirs(output_folder, exist_ok=True)
#     images = convert_from_path(pdf_path, dpi=dpi)
#     image_paths = []
#     for i, image in enumerate(images):
#         image_path = os.path.join(output_folder, f"page_{i+1}.jpg")
#         image.save(image_path, "JPEG")
#         image_paths.append(image_path)
#     return image_paths

# # Step 3: Extract Text for Each Label
# def extract_text_for_label(image, coordinates):
#     x_min, y_min, x_max, y_max = coordinates
#     cropped = image[y_min:y_max, x_min:x_max]  # Crop the region of interest
#     text = pytesseract.image_to_string(cropped)  # Perform OCR
#     return text.strip()

# # Step 4: Draw Bounding Boxes and Save Annotated Image
# def annotate_and_save_image(image, annotations, output_path):
#     for annotation in annotations:
#         x_min, y_min, x_max, y_max = annotation["coordinates"]
#         label = annotation["label"]
#         confidence = annotation["confidence"]

#         # Draw bounding box
#         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box

#         # Add label and confidence as text
#         label_text = f"{label} ({confidence:.2f})"
#         cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Save annotated image
#     cv2.imwrite(output_path, image)

# # Step 5: Perform Layout Detection and Extract Text
# def detect_and_extract_text(model, image_path, class_mapping, conf=0.2, imgsz=1024, device="cuda:0"):
#     # Perform detection
#     det_res = model.predict(
#         image_path,
#         imgsz=imgsz,
#         conf=conf,
#         device=device
#     )
    
#     # Load the image for OCR and annotation
#     image = cv2.imread(image_path)

#     # Prepare results for JSON
#     structured_content = []
#     for box in det_res[0].boxes:
#         # Extract bounding box coordinates
#         if isinstance(box.xyxy.tolist()[0], list):
#             # Handle nested list case
#             x_min, y_min, x_max, y_max = map(int, box.xyxy.tolist()[0])
#         else:
#             # Handle flat list case
#             x_min, y_min, x_max, y_max = map(int, box.xyxy.tolist())

#         label_index = int(box.cls)  # Class label index
#         confidence = float(box.conf)  # Confidence score

#         # Get label name using the class mapping
#         label_name = class_mapping.get(label_index, f"Unknown_{label_index}")

#         # Extract text for the detected region
#         extracted_text = extract_text_for_label(image, (x_min, y_min, x_max, y_max))

#         # Add the structured content
#         structured_content.append({
#             "label": label_name,
#             "coordinates": [x_min, y_min, x_max, y_max],
#             "confidence": confidence,
#             "extracted_text": extracted_text
#         })

#     return structured_content, image

# # Step 6: Save Structured Results to JSON
# def save_results_to_json(pdf_path, results, output_folder):
#     pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
#     json_path = os.path.join(output_folder, f"{pdf_name}_structured_results.json")
#     with open(json_path, "w") as json_file:
#         json.dump(results, json_file, indent=4)
#     print(f"Results saved to: {json_path}")

# # Step 7: Process the Entire PDF
# def process_pdf(pdf_path, model_path, output_folder, class_mapping, dpi=200, conf=0.2, imgsz=1024, device="cpu"):
#     # Load the YOLO model
#     model = load_model(model_path)
    
#     # Convert PDF to images
#     temp_image_folder = "./temp_images"
#     image_paths = pdf_to_images(pdf_path, temp_image_folder, dpi=dpi)

#     # Create folder for annotated images
#     annotated_image_folder = os.path.join(output_folder, "annotated_images")
#     os.makedirs(annotated_image_folder, exist_ok=True)

#     # Detect layout and extract text for each image
#     structured_results = []
#     for image_path in image_paths:
#         print(f"Processing {image_path}...")
#         structured_content, annotated_image = detect_and_extract_text(
#             model, image_path, class_mapping, conf=conf, imgsz=imgsz, device=device
#         )

#         # Save annotated image
#         annotated_image_path = os.path.join(annotated_image_folder, os.path.basename(image_path))
#         annotate_and_save_image(annotated_image, structured_content, annotated_image_path)

#         # Append structured content to results
#         structured_results.append({
#             "page": os.path.basename(image_path),
#             "content": structured_content
#         })

#     # Save all results to JSON
#     save_results_to_json(pdf_path, structured_results, output_folder)

#     # Clean up temporary images
#     for image_path in image_paths:
#         os.remove(image_path)
#     os.rmdir(temp_image_folder)

# # Step 8: Entry Point
# if __name__ == "__main__":
#     # Paths
#     pdf_path = "/workspaces/Document-Processing-System/doclayoutanalysis/inputs/2024020637.pdf"  # Replace with your PDF file
#     model_path = hf_hub_download(
#         repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
#         filename="doclayout_yolo_docstructbench_imgsz1024.pt"
#     )  # Replace with your YOLO model weights
#     output_folder = "./output_jsons"  # Output folder for JSON results and annotated images

#     # Class mapping (update with your YOLO class labels)
#     class_mapping = {
#         0: 'title', 
#         1: 'plain text',
#         2: 'abandon', 
#         3: 'figure', 
#         4: 'figure_caption', 
#         5: 'table', 
#         6: 'table_caption', 
#         7: 'table_footnote', 
#         8: 'isolate_formula', 
#         9: 'formula_caption'
#     }

#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Process the PDF with structured results
#     process_pdf(pdf_path, model_path, output_folder, class_mapping)
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

# Step 4: Draw Bounding Boxes and Save Annotated Image
def annotate_and_save_image(image, annotations, output_path):
    for annotation in annotations:
        x_min, y_min, x_max, y_max = annotation["coordinates"]
        label = annotation["label"]
        confidence = annotation["confidence"]

        # Use a consistent bounding box color (e.g., green)
        color = (0, 255, 0)

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Add label and confidence as text
        label_text = f"{label} ({confidence:.2f})"
        cv2.putText(image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save annotated image
    cv2.imwrite(output_path, image)

# Step 5: Sort and Maintain Content Order
def sort_content_by_order(structured_content):
    # Sort content by vertical position (y_min) to maintain top-to-bottom order
    return sorted(structured_content, key=lambda item: item["coordinates"][1])

# Step 6: Perform Layout Detection and Extract Text
def detect_and_extract_text(model, image_path, class_mapping, conf=0.4, imgsz=1024, device="cuda:0"):
    # Perform detection
    det_res = model.predict(
        image_path,
        imgsz=imgsz,
        conf=conf,
        # conf=conf,  # Confidence threshold
        iou=0.5,  
        device=device
    )
    
    # Load the image for OCR and annotation
    image = cv2.imread(image_path)

    # Prepare results for JSON
    structured_content = []
    for box in det_res[0].boxes:
        # Extract bounding box coordinates
        # x_min, y_min, x_max, y_max = map(int, box.xyxy.tolist())
        # ...existing code...
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
# ...existing code...

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

    # Sort content to maintain order
    structured_content = sort_content_by_order(structured_content)

    return structured_content, image

# Step 7: Save Structured Results to JSON
def save_results_to_json(pdf_path, results, output_folder):
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    json_path = os.path.join(output_folder, f"{pdf_name}_structured_results2.json")
    with open(json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to: {json_path}")

# Step 8: Process the Entire PDF
def process_pdf(pdf_path, model_path, output_folder, class_mapping, dpi=200, conf=0.4, imgsz=1024, device="cpu"):
    # Load the YOLO model
    model = load_model(model_path)
    
    # Convert PDF to images
    temp_image_folder = "./temp_images"
    image_paths = pdf_to_images(pdf_path, temp_image_folder, dpi=dpi)

    # Create folder for annotated images
    annotated_image_folder = os.path.join(output_folder, "annotated_images2")
    os.makedirs(annotated_image_folder, exist_ok=True)

    # Detect layout and extract text for each image
    structured_results = []
    for image_path in image_paths:
        print(f"Processing {image_path}...")
        structured_content, annotated_image = detect_and_extract_text(
            model, image_path, class_mapping, conf=conf, imgsz=imgsz, device=device
        )

        # Save annotated image
        annotated_image_path = os.path.join(annotated_image_folder, os.path.basename(image_path))
        annotate_and_save_image(annotated_image, structured_content, annotated_image_path)

        # Append structured content to results
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

# Step 9: Entry Point
if __name__ == "__main__":
    # Paths
    pdf_path = "/workspaces/Document-Processing-System/doclayoutanalysis/inputs/2024020637.pdf"  # Replace with your PDF file
    model_path = hf_hub_download(
        repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
        filename="doclayout_yolo_docstructbench_imgsz1024.pt"
    )  # Replace with your YOLO model weights
    output_folder = "./output_jsons"  # Output folder for JSON results and annotated images

    # Class mapping (update with your YOLO class labels)
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