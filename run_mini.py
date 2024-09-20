#Setup
import requests
import json
import math
import os
import pandas as pd
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import shutil
import re
import numpy as np
from openai import OpenAI
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.model_selection import train_test_split
from resize import binary_thr, gray, resize, compress, down_sample

def get_url():

    resize_url_ls = []
    url_ls = []

    # Replace with your GitHub repo details
    github_username = "Tommasomagma"
    repo_name = "resize_img"
    branch = "main"  # Change if you're using a different branch
    # Path to the folder where your images are stored (relative to your local workspace)
    resize_image_folder = "resized"
    image_folder = "original"

    # List all image files in the local folder
    resize_image_files = [f for f in os.listdir(resize_image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Generate URLs for each image
    for image_file in image_files:
        image_url = f"https://raw.githubusercontent.com/{github_username}/{repo_name}/{branch}/{image_file}"
        url_ls.append(image_url)
    
    for image_file in resize_image_files:
        image_url = f"https://raw.githubusercontent.com/{github_username}/{repo_name}/{branch}/{image_file}"
        resize_url_ls.append(image_url)

    return url_ls, resize_url_ls



def check_mini(df):

    df = df[df['LLM']!='X']
    df = df[df['mini']!='X']
    df = df[df['digitsProb'] != 0]
    LLM_labels = list(df['LLM'].astype(int))
    mini_labels = list(df['mini'].astype(int))
    print(f'mean response time: {np.mean(list(df['time'].astype(float)))}')

    for i in df.index:
        if float(df['time'][i]) <= 1.5:
            print(df['string'][i])
            print(df['problem'][i])
            print(df['answer'][i])
            print(df['time'][i])
            print(f'{int(df['LLM'][i])}/{df['mini'][i]}')
            print('---')
            url = df['image'][i]
            url = "https://s3.eu-central-1.amazonaws.com/prod.solutions/"+url
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            image_size = image.size  # This returns a tuple (width, height)
            print(f"Image size: {image_size}")
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.show(block=False)
            go = input('next:')
            plt.close()


def run_mini(df):

    LLM_labels = list(df['LLM_fix'].astype(int))
    label_ls = []
    time_ls = []
    tag_ls = []
    fail_count = 0
    i = 0

    stop_ls = [100,200,400,600,800,1000]
    url_ls, resize_url = get_url() 
    all_url = url_ls + resize_url

    # for i in df.index:
    for i, url in enumerate(all_url):

        print(url)
        go = input('next')

        if url in url_ls:
            tag = 'original'
        if url in resize_url:
            tag = 'resize'

        if i % 10 == 0:
            print(f'{i} of {len(df)} done')

        if i in stop_ls:

            save_df = pd.DataFrame(df.head(len(label_ls)))
            save_df['mini_str'] = label_ls
            save_df['time'] = time_ls
            save_df['size'] = tag_ls
            save_df.to_csv('check_img.csv')

        try:

            #url = df.image[index]
            #url_prompt = 'https://raw.githubusercontent.com/Tommasomagma/resize_img/refs/heads/main/2.png?token=GHSAT0AAAAAACURNTLUXWYSB6Z35APM5EAYZXMJABA'
            # answer = df.answer_fix[0]
            # problem = df.problem_fix[0]
            client = OpenAI(api_key='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')
            #url_prompt = "https://s3.eu-central-1.amazonaws.com/prod.solutions/"+url
            #prompt = f' You are given a math problem: {problem}, the correct answer {answer}, and a students handwritten solution in the image. Your task is to classify whether the solution is "good" (label = 1) or "bad" (label = 0). For the handwritten solution to be good, the student has showed how he solved the math problem with the handwritten solution. If the solution contains a mathematical expression that is relevant to the problem description and results in the answer (although the answer might not be written in the solution) the solution is good and the classification should be 1. If the student has written out the problem describtion and answer, the classification should be 1. If the answer is the only thing present in the hand written solution, the classification should be 0. If the mathematical expressions in the solution are not at all relevant for solving the problem the solution is bad and the classification should be 0. If the student has written very unclear or incorrect calculations for the answer, such as “1+2=5”, the solution is bad and the classification should be 0. Keep in mind that the student is a child and might have a bit unclear handwriting. In your output, only include a single integer: 1 if you classify it as positive, 0 if you classify it as negative.'
            prompt = 'return a string representation of the mathematical operations performed in the image of a hand written math solution'
            start_time = time.time()
            response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", 
                    "text": prompt},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                    },
                    },
                ],
                }
            ],
            max_tokens=300,
            )
            end_time = time.time()
            t = end_time - start_time
            time_ls.append(t)
            label_ls.append(int(response.choices[0].message.content))
            tag_ls.append(tag)
            i += 1
        except Exception as e:
            print('fucked up')
            fail_count += 1
            label_ls.append('X')
            time_ls.append('X')
            tag_ls.append('X')
            i += 1
            print(e)

    print(np.max(time_ls))
    print(np.min(time_ls))
    df['mini_str'] = label_ls
    df['time'] = time_ls

    return df

if __name__ == "__main__":

    df = pd.read_csv('test_img.csv')
    df = pd.DataFrame(df.head(3))
    run_mini(df)
    df.to_csv('check_img.csv')
    

