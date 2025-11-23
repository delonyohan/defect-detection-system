
import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path
import noise

def generate_procedural_data(num_images=20, img_size=(256, 256)):
    output_dir = Path('data/procedural_images')
    output_dir.mkdir(exist_ok=True)

    for i in range(num_images):
        # Generate a procedural metal texture using Perlin noise
        scale = 100.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        
        world = np.zeros(img_size)
        for x in range(img_size[0]):
            for y in range(img_size[1]):
                world[x][y] = noise.pnoise2(x/scale, 
                                            y/scale, 
                                            octaves=octaves, 
                                            persistence=persistence, 
                                            lacunarity=lacunarity, 
                                            repeatx=img_size[0], 
                                            repeaty=img_size[1], 
                                            base=0)
        
        # Normalize the texture and convert to a grayscale image
        world_min = np.min(world)
        world_max = np.max(world)
        world = (world - world_min) / (world_max - world_min)
        img_array = (world * 255).astype(np.uint8)
        img = Image.fromarray(img_array, 'L').convert('RGB')
        draw = ImageDraw.Draw(img)

        # Create a corresponding mask
        mask = Image.new('L', img_size, color=0)
        mask_draw = ImageDraw.Draw(mask)

        # Add a "defect" (a scratch or a patch)
        defect_type = np.random.choice(['scratch', 'patch'])
        
        if defect_type == 'scratch':
            x1 = np.random.randint(10, img_size[0] - 10)
            y1 = np.random.randint(10, img_size[1] - 10)
            x2 = x1 + np.random.randint(-50, 50)
            y2 = y1 + np.random.randint(-50, 50)
            width = np.random.randint(1, 3)
            color = tuple(np.random.randint(0, 50, 3))
            draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
            mask_draw.line([(x1, y1), (x2, y2)], fill=255, width=width)
        else: # patch
            x = np.random.randint(50, 200)
            y = np.random.randint(50, 200)
            size = np.random.randint(10, 40)
            color = tuple(np.random.randint(100, 150, 3))
            draw.ellipse((x - size, y - size, x + size, y + size), fill=color)
            mask_draw.ellipse((x - size, y - size, x + size, y + size), fill=255)

        # Save the image and mask
        img.save(output_dir / f'img_{i:04d}.png')
        mask.save(output_dir / f'mask_{i:04d}.png')

if __name__ == '__main__':
    generate_procedural_data()
    print(f"Generated procedural data in {Path('data/procedural_images').resolve()}")
