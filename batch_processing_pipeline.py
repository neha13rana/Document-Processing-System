import os
import cv2
from convert_pdfs_to_images import pdf_to_images
from layout_detection import detect_layout
from text_extraction import extract_text_from_blocks
from output_to_json import save_to_json

def process_pdf(pdf_path, model_path, output_folder):
    # Step 1: Convert PDF to images
    images_folder = pdf_to_images(pdf_path, "temp_images")
    
    # Step 2: Process each page
    pdf_data = []
    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)
        
        # Step 3: Detect layout
        layout = detect_layout(image_path, model_path)
        
        # Step 4: Extract text
        text_data = extract_text_from_blocks(image, layout)
        pdf_data.append(text_data)
    
    # Step 5: Save individual PDF data to JSON
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    json_output_path = os.path.join(output_folder, f"{pdf_name}.json")
    save_to_json(pdf_data, json_output_path)

def batch_process_pdfs(folder_path, model_path, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each PDF in the folder
    for pdf_name in os.listdir(folder_path):
        if pdf_name.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, pdf_name)
            print(f"Processing: {pdf_path}")
            process_pdf(pdf_path, model_path, output_folder)

# Run the batch pipeline
batch_process_pdfs(
    folder_path="path_to_pdfs",  # Replace with your folder path
    model_path="weights/yolo_model.pt",  # Replace with your YOLO model path
    output_folder="output_jsons"  # Replace with your desired output folder
)