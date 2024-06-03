import os
import shutil
from_path = "torch_data/train"
to_path ="torch_data/test"

for dir in os.listdir(from_path):
    from_dir_path=os.path.join(from_path,dir)
    to_dir_path=os.path.join(to_path,dir)

    if not os.path.exists(to_dir_path):
        os.mkdir(to_dir_path)
    cnt=0
    for image in os.listdir(from_dir_path):
        from_image_path=os.path.join(from_dir_path,image)
        to_image_path = os.path.join(to_dir_path,image)
        shutil.move(from_image_path,to_image_path)
        cnt=cnt+1
        if cnt == 35:
            break


