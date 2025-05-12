import json

def save_to_json(data, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# Sample usage
data = {
    "text_blocks": ["Text 1", "Text 2"],
    "tables": ["Table Data"],
}
save_to_json(data, 'output.json')