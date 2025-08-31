import os
import shutil

# Base directories
base_dir = 'coco'
image_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'labels')

output_dir = os.path.join(base_dir, 'images2025')

# Dataset splits
splits = ['train2017', 'val2017']

# COCO ID to target class name mapping
coco_id_to_name = {
    0: 'person',
    2: 'vehicle',
    3: 'vehicle',
    7: 'vehicle',
    16: 'bird',
    18: 'dog'
}

# Target classes list (fixed order)
target_classes = ['person', 'vehicle', 'dog', 'bird', 'fox', 'cat', 'squirrel']

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Count total files for progress display
total_files = sum(len(os.listdir(os.path.join(label_dir, split))) for split in splits)
processed_files = 0
kept_files = 0

# Write classes.txt into output dir
classes_file_path = os.path.join(output_dir, 'classes.txt')
with open(classes_file_path, 'w') as f:
    for cls in target_classes:
        f.write(f"{cls}\n")

print(f"Processing {total_files} annotation files...\n")

# Process annotations
for split in splits:
    label_split_dir = os.path.join(label_dir, split)
    image_split_dir = os.path.join(image_dir, split)

    for label_file in os.listdir(label_split_dir):
        label_path = os.path.join(label_split_dir, label_file)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            original_id = int(parts[0])
            class_name = coco_id_to_name.get(original_id, None)

            # Skip if not mapped to a target class
            if class_name not in target_classes:
                continue

            new_id = target_classes.index(class_name)
            parts[0] = str(new_id)
            new_lines.append(" ".join(parts))

        if new_lines:
            # Copy image
            image_filename = os.path.splitext(label_file)[0] + '.jpg'
            src_image_path = os.path.join(image_split_dir, image_filename)
            dest_image_path = os.path.join(output_dir, image_filename)

            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dest_image_path)
            else:
                print(f"⚠️ Image {image_filename} not found in {image_split_dir}")

            # Write new annotation next to the image
            dest_label_path = os.path.join(output_dir, label_file)
            with open(dest_label_path, 'w') as f:
                f.write("\n".join(new_lines) + "\n")

            kept_files += 1

        processed_files += 1

        # Progress every 500 files
        if processed_files % 500 == 0 or processed_files == total_files:
            progress = (processed_files / total_files) * 100
            print(f"Processed {processed_files}/{total_files} files ({progress:.2f}%)")

print(f"\nDone. Kept {kept_files} images and annotations matching target classes.")
print(f"Output written to: {output_dir}")
