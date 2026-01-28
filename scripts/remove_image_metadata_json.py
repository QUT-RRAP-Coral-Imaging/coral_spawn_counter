import glob
import os

# Change this to your target directory
# target_dir = '/Volumes/DT4TB/cslics_2023_datasets/2023_Dec_Spawning/20231205_alor_tank4_cslics08/images'
# target_dir = '/Volumes/DT4TB/cslics_2023_datasets/2023_Dec_Spawning/20231205_alor_tank4_cslics09/images'
target_dir = '/Volumes/DT4TB/cslics_2023_datasets/2023_Dec_Spawning/20231205_alor_tank4_cslics01/images'

# Recursively find all .json files
json_files = glob.glob(os.path.join(target_dir, '**', '*.json'), recursive=True)

print(f"Found {len(json_files)} JSON files.")

ginput = input("Do you want to delete these files? (y/n): ")

i = 0
if ginput.lower() == 'y':
    for file_path in json_files:
        os.remove(file_path)
        i += 1
else:
    print("No files deleted.")
    
    # print some of the json file names so I know where they are:
    max_files_to_show = 5
    # avoid the error where max_files_to_show is greater than len(json_files)
    max_files_to_show = min(max_files_to_show, len(json_files))
    for file_path in json_files[:max_files_to_show]:
        print(f" - {file_path}")

print(f"Deleted {i} JSON files.")


print(f"Done")