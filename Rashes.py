import os
import csv

# Set the folder where images are stored
image_folder = "Lupus dataset"  # Update this to your actual folder

# Output CSV file
csv_filename = "Lupus_Data.csv"

# Get all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Create and write CSV file
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow(["Filename", "Path", "Size (KB)"])

    # Write image details
    for image in image_files:
        image_path = os.path.join(image_folder, image)
        image_size = round(os.path.getsize(image_path) / 1024, 2)  # Convert size to KB
        writer.writerow([image, image_path, image_size])

print(f"CSV file '{csv_filename}' created successfully with {len(image_files)} images.")
