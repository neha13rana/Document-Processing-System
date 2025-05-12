import pytesseract

def extract_text_from_blocks(image, blocks):
    extracted_text = {}
    for block in blocks:
        x, y, w, h = block['bbox']
        cropped = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(cropped)
        extracted_text[block['label']] = text
    return extracted_text