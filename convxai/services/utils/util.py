import os

def create_folder(folder_list):   #[model_dir, log_dir, result_dir, cache_dir]
    for folder in folder_list:
        if not os.path.isdir(folder):
            os.makedirs(folder)
    print("Created folders:", folder_list)
    


