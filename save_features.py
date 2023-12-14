from PIL import Image
import h5py
import torch
from feature_extraction import *
from torchvision import transforms, models
import torch.nn as nn

import os

# Directory containing your images
image_dir = r"E:\Artificial_Intelligence\Video Summarization\Datasets\tvsum\vid1"

# Directory to save the HDF5 file
output_dir = r"E:\Artificial_Intelligence\Video Summarization\Datasets\tvsum\features"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)






# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for frame_number in range(10):  # Assuming images are named frame0000.jpg to frame5970.jpg
    image_path = os.path.join(image_dir, f'frame{frame_number:04d}.png')
    
    # Load and preprocess the image
    image = Image.open(image_path)
    input_tensor = resnet_transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

    # Extract features
    with torch.no_grad():
        res5c, pool5 = resnet_feature_model(input_batch.cuda())

    # Save features to a new HDF5 file for each image
    h5_file_path = os.path.join(output_dir, f'features_{frame_number}.h5')
    with h5py.File(h5_file_path, 'w') as h5_file:
        h5_file.create_dataset('res5c', data=res5c.cpu().numpy())
        h5_file.create_dataset('pool5', data=pool5.cpu().numpy())

    print(f"Features saved to {h5_file_path}")
        
