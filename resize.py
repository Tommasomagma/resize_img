from PIL import Image
import cv2
import pandas as pd
import requests
from io import BytesIO

def resize(img, width, height):

    img_resized = img.resize((width, height))  # You can specify a smaller width and height

    return img_resized

def gray(url):

    img = Image.open(url)
    
    return img.convert('L')  # Convert to grayscale

def down_sample(url):

    img = cv2.imread(url)
    img_downsampled = cv2.pyrDown(img)  # Downsamples the image
    
    return img_downsampled

def compress(url):

    img = Image.open(url)
    path = f'{url}_compressed.png'
    img.save(path, optimize=True, quality=85)

    return path

def binary_thr(url):

    img = cv2.imread(url)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    return binary


if __name__ == '__main__':

    df = pd.read_csv('test_img.csv')
    img_ls = list(df['image'])[:5]
    for i,img in enumerate(img_ls):
        url = "https://s3.eu-central-1.amazonaws.com/prod.solutions/"+img
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(f'{i}.png')
        half_width = img.width // 2
        half_height = img.height // 2
        resized = resize(img, half_width, half_height)
        resized.save(f'{i}_resized.png')
        print(resized)