import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path

def generate_synthetic_data(num_images=20, img_size=(256, 256)):
    output_dir = Path('data/synthetic_images')
    output_dir.mkdir(exist_ok=True)

    for i in range(num_images):
        # Create a blank image with a random background color
        background_color = tuple(np.random.randint(200, 256, 3))
        img = Image.new('RGB', img_size, color=background_color)
        draw = ImageDraw.Draw(img)

        # Draw a random shape (rectangle or circle) as the main object
        shape_type = np.random.choice(['rectangle', 'circle'])
        shape_color = tuple(np.random.randint(100, 200, 3))
        
        if shape_type == 'rectangle':
            x1 = np.random.randint(10, 50)
            y1 = np.random.randint(10, 50)
            x2 = np.random.randint(200, 240)
            y2 = np.random.randint(200, 240)
            draw.rectangle([x1, y1, x2, y2], fill=shape_color)
        else: # circle
            x = np.random.randint(100, 150)
            y = np.random.randint(100, 150)
            r = np.random.randint(50, 100)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=shape_color)

        # Create a corresponding mask
        mask = Image.new('L', img_size, color=0)
        mask_draw = ImageDraw.Draw(mask)

        # Add a "defect" (a smaller shape) to the image and mask
        defect_type = np.random.choice(['rectangle', 'circle'])
        defect_color = tuple(np.random.randint(0, 100, 3))
        defect_x = np.random.randint(50, 200)
        defect_y = np.random.randint(50, 200)
        defect_size = np.random.randint(10, 30)

        if defect_type == 'rectangle':
            draw.rectangle(
                [defect_x, defect_y, defect_x + defect_size, defect_y + defect_size], 
                fill=defect_color
            )
            mask_draw.rectangle(
                [defect_x, defect_y, defect_x + defect_size, defect_y + defect_size], 
                fill=255
            )
        else: # circle
            draw.ellipse(
                (defect_x - defect_size, defect_y - defect_size, defect_x + defect_size, defect_y + defect_size), 
                fill=defect_color
            )
            mask_draw.ellipse(
                (defect_x - defect_size, defect_y - defect_size, defect_x + defect_size, defect_y + defect_size), 
                fill=255
            )
            
        # Add some noise to the image
        img_array = np.array(img)
        noise = np.random.randint(-10, 10, img_array.shape, dtype='int16')
        img_array = np.clip(img_array.astype('int16') + noise, 0, 255).astype('uint8')
        img = Image.fromarray(img_array)

        # Save the image and mask
        img.save(output_dir / f'img_{i:04d}.png')
        mask.save(output_dir / f'mask_{i:04d}.png')

if __name__ == '__main__':
    generate_synthetic_data()
    print(f"Generated synthetic data in {Path('data/synthetic_images').resolve()}")