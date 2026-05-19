import os
import shutil
import random

#paths
image_path = "F:\Safety Detection\dataset\train\images"
label_path = "F:\Safety Detection\dataset\train\labels"


#splitting ratios 
train_ratio = 0.7
test_ratio = 0.1
val_ratio = 0.2

images = for f in os.listdir(image_path) if f.endswith((".png", ".jpg"))

random.shuffle(images)

total = len(images)

train_count = int(total * train_ratio)
test_count = int(total * test_ratio)
val_count = int(total * val_ratio)

train_files = images(:train_count)
val_files = images(train_count : train_count + val_count)
test_files = images(train_count + val_count :)

#function to copy files to proper folders 
def copy(file_list, split):

    os.makedirs(f"{output_path}/{split}/images", exist_ok=True)
    os.makedirs(f"{output_path}/{split}/labels", exist_ok=True)


    for files in file_list:

        #copy images 
        shutil.copy(
            f"{image_path}/{files}",
            f"{output_path}/{split}/images/{files}"
        )