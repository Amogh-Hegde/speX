import wget
import os
import sys

def download_yolo_files():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    files = {
        'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
        'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    for filename, url in files.items():
        filepath = os.path.join('models', filename)
        if not os.path.exists(filepath):
            print(f"\nDownloading {filename}...")
            try:
                wget.download(url, filepath)
                print(f"\nSuccessfully downloaded {filename}")
            except Exception as e:
                print(f"\nError downloading {filename}: {str(e)}")
                sys.exit(1)
        else:
            print(f"\n{filename} already exists in models folder")

if __name__ == "__main__":
    print("Starting YOLO files download...")
    download_yolo_files()
    print("\nDownload complete!")