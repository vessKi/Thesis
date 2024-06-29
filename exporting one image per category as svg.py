import ndjson
import numpy as np
import os
import svgwrite
from tqdm import tqdm

def get_one_image_per_category(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(data_dir):
        if file.endswith('.ndjson'):
            category = file.split('.')[0]
            with open(os.path.join(data_dir, file)) as f:
                content = f.read()
                print(f"Content of {file}: {content[:200]}")  # Print first 200 characters of the file content
                data = ndjson.loads(content)
                if data:
                    drawing = data[0]['drawing']
                    dwg = svgwrite.Drawing(os.path.join(output_dir, f'{category}.svg'), size=(256, 256))
                    for stroke in drawing:
                        points = list(zip(stroke[0], stroke[1]))
                        dwg.add(dwg.polyline(points, stroke=svgwrite.rgb(0, 0, 0, '%'), fill='none', stroke_width=2))
                    dwg.save()
                    print(f'Saved {category}.svg')

# Directory containing .ndjson files
data_dir = r'C:\Users\user\AppData\Local\Google\Cloud SDK\quickdraw_dataset\quickdraw_dataset'
# Directory to save output images
output_dir = r'C:\Users\user\AppData\Local\Google\Cloud SDK\quickdraw_dataset\output'

# Now extract one image per category
get_one_image_per_category(data_dir, output_dir)
