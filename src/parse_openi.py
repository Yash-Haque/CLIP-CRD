import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from typing import List

def parse_Data(clip_model, preprocess, caption_path: str, image_path: str, device, out_path: str):
    # Loading Caption 
    data = []
    with open(caption_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print("%0d captions loaded from json " % len(data))

    # Embeddings and Captions
    all_embeddings = []
    all_captions = []

    # Processing each captions and images
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["id"]
        filename = f"{int(img_id)}.jpg"
        path = os.path.join(image_path, filename)
        if not os.path.isfile(path):
            print(f"File {path} does not exist in the directory './data/images/'")
        image = io.imread(path)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)

        # Periodic save every 10,000 entries
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

        # Final Save
        with open(out_path, 'wb') as f:
            pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    return all_embeddings, all_captions

def main(clip_model_type: str):
    # Setting up device
    device = torch.device("cuda:0")

    # Model and Preprocess Initialization
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    out_path = f"./outputs/openi_{clip_model_name}_train.pkl"
    
    train_captions = f"./data/Captions/Train.jsonl"
    test_captions = f"./data/Captions/Test.jsonl"
    valid_captions = f"./data/Captions/Valid.jsonl"

    image_path = f"./data/images/"

    # train_captions = f"D:/WORKING/TRACKED-PROJECTS/GITHUB/MedCLIPCap/data/Captions/Train.jsonl"
    # test_captions = f"D:/WORKING/TRACKED-PROJECTS/GITHUB/MedCLIPCap/data/Captions/Test.jsonl"
    # valid_captions = f"D:/WORKING/TRACKED-PROJECTS/GITHUB/MedCLIPCap/data/Captions/Valid.jsonl"
    # image_path = f"D:/WORKING/TRACKED-PROJECTS/GITHUB/MedCLIPCap/data/images/"

    
    all_embeddings, all_captions = parse_Data(
        clip_model=clip_model,
        preprocess=preprocess, 
        caption_path=train_captions,
        image_path=image_path,
        device=device,
        out_path=out_path
    )
    
    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))