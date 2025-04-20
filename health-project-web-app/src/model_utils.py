import torch
import h5py
import matplotlib.pyplot as plt
from .unet.unet_module import UnetModule
import fastmri
from fastmri.data import transforms as T
import boto3
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger

# h5_file_path = "data/file1002549.h5"
# model_path = "models/l1_ssim baseline_no_roi.ckpt"

async def predict_models(q):
    hf = h5py.File(q.client.file_path)
    volume_kspace = hf['kspace'][()]
    logger.info(f"Original kspace shape: {volume_kspace.shape}")

    kspace_tensor = T.to_tensor(volume_kspace)
    logger.info(f"Tensor kspace tensor shape: {kspace_tensor.shape}")

    image = fastmri.ifft2c(kspace_tensor)
    image_abs = fastmri.complex_abs(image)
    image_abs = fastmri.rss(image_abs, dim=0)
    logger.info(f"Image shape: {image_abs.shape}")

    image_abs = image_abs.unsqueeze(0)
    logger.info(f"Input shape after adding dimensions: {image_abs.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_abs = image_abs / image_abs.max()
    image_abs = image_abs.to(device)
    image = image_abs.cpu().numpy().squeeze()
    image = add_annotation(q, image)

    os.makedirs("./predictions", exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f'./predictions/input_{q.client.file_name}.png', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    q.client.predictions_files = []
    if q.client.model_picker:
        for model_name in q.client.model_picker:
            try:
                model_path = f"./models/{model_name}.ckpt"
                await download_model_from_s3(model_name)
                model = UnetModule.load_from_checkpoint(model_path)
                model.eval()
                model = model.to(device)
                logger.info(f"Model {model_name} loaded and moved to {device}")
                with torch.no_grad():
                    reconstructed = model(image_abs)

                reconstructed_np = reconstructed.cpu().numpy().squeeze()
                # reconstructed_np = add_annotation(q, reconstructed_np)
                plt.figure(figsize=(10, 10))
                plt.imshow(reconstructed_np, cmap='gray')
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(f'predictions/{model_name}_{q.client.file_name}.png', bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close()
                q.client.predictions_files.append(f'predictions/{model_name}_{q.client.file_name}.png')
                os.remove(model_path)

            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
        
    os.remove(q.client.file_path)
    await q.page.save()
    return


async def download_model_from_s3(model_name):
    S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
    KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    s3 = boto3.client(
        's3',
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET_KEY,
        region_name='us-east-1'
    )

    try:
        local_dir = "./models"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f"{model_name}.ckpt")
        logger.info(f"Downloading model {model_name} from S3 from the bucket {S3_BUCKET} to {local_path}")
        s3.download_file(S3_BUCKET, f"models/{model_name}.ckpt", local_path)
        logger.info(f"Model {model_name} downloaded from S3")
        return True
    except Exception as e:
        logger.error(f"Error downloading model {model_name} from S3: {str(e)}")
        return False


async def download_data_from_s3(file_name):
    S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
    KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    s3 = boto3.client(
        's3',
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET_KEY,
        region_name='us-east-1'
    )
    
    try:
        local_dir = "./data"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, file_name)
        logger.info(f"Downloading data {file_name} from S3 from the bucket {S3_BUCKET} to {local_path}")
        s3.download_file(S3_BUCKET, f"data/{file_name}", local_path)
        logger.info(f"Data {file_name} downloaded from S3")
        return True
    except Exception as e:
        logger.error(f"Error downloading data {file_name} from S3: {str(e)}")
        return False
        

def add_annotation(q, image_np):
    df = pd.read_csv('annotations/knee.csv')
    name = q.client.file_name.split('.')[0]
    df = df[df['file'] == name]

    if df.empty:
        return image_np
    
    df_sample = df.head(1)
    image_2d_scaled = (np.maximum(image_np,0) / image_np.max()) * 255.0
    image_2d_scaled = Image.fromarray(np.uint8(image_2d_scaled))
    for _, row in df_sample.iterrows():
        x0, y0, w, h, label_txt = row['x'], row['y'], row['width'], row['height'], row['label']
        x1 = x0 + w
        y1 = y0 + h
        plotted_image = ImageDraw.Draw(image_2d_scaled)
        plotted_image.rectangle(((x0,y0), (x1,y1)), outline="white")
        plotted_image.text((x0, max(0, y0 - h)), label_txt, fill= "white")

    del df, df_sample
    return np.array(image_2d_scaled)
        