import os
import shutil
import time

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'data')

    for subfolder_name in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder_name)
        if os.path.isdir(subfolder_path) and subfolder_name != 'backup':
            shutil.rmtree(subfolder_path)
            print("folder '{}' has been deleted.".format(subfolder_name))
            time.sleep(1)  # Add a small delay to avoid issues

if __name__ == "__main__":
    main()
