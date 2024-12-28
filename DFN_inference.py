import os
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel


def read_img_files(img_files):
    """
    Reads all image files and returns them as PIL Image objects.
    
    Args:
        img_files (list): List of paths to image files.
    
    Returns:
        imgs (list): List of PIL Image objects.
    """
    imgs = []
    for file in img_files:
        if os.path.exists(file):
            try:
                img = Image.open(file)
                imgs.append(img)
            except Exception as e:
                print(f"Error loading image {file}: {e}")
        else:
            print(f"File not found: {file}")
    return imgs

def read_txt_files(txt_files):
    """
    Reads all text files and returns their contents as a list of strings.
    
    Args:
        txt_files (list): List of paths to text files.
    
    Returns:
        texts: Contents of each text file as strings.
    """
    texts = []
    for file in txt_files:
        if os.path.exists(file):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            except Exception as e:
                print(f"Error reading file {file}: {e}")
        else:
            print(f"File not found: {file}")
    return texts

# Model Initialization
def initialize_model_DFN(HF_model_name="XudongShen/DFN-public"):
    """
    Initialize the CLIP model and processor.

    Returns:
        model: huggingface model .
        preprocesser: huggingface preprocesser.
    """
    model = CLIPModel.from_pretrained(HF_model_name, ignore_mismatched_sizes=True)
    preprocesser = CLIPProcessor.from_pretrained("XudongShen/DFN-public")
    
    return model, preprocesser


def compute_DFN_score(model, processor, imgs, texts):
    """
    DFN score should be a value between -1 and 1, with higher being better
    precisely, DFN computes the cosine similarity between text and image embeddings,
    embeddings are processed by the DFN model, ie, a specially trained CLIP model

    Args:
        model: CLIP model.
        processor: CLIP processor.
        imgs: List of PIL images.
        texts: List of texts.

    Returns:
        DFN_score: list of DFN scores.
    """
    # Prepare inputs
    inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True, truncation=True)

    # Get embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeddings = outputs.image_embeds  # Image embeddings
        text_embeddings = outputs.text_embeds    # Text embeddings

    # Normalize the embeddings
    image_embeddings = image_embeddings / image_embeddings.norm(p=2, dim=-1, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity
    cosine_similarity = (text_embeddings * image_embeddings).sum(dim=-1)

    return cosine_similarity.tolist()

if __name__ == "__main__":
    
    # set number of images caption pairs to process
    number_imgs = 10

    # list .jpg files using os.scandir
    folder_path = os.path.join( os.path.dirname(os.path.abspath(__file__)), "00000000_tar")
    jpg_files = sorted([os.path.join(folder_path, entry.name) for entry in os.scandir(folder_path) if entry.is_file() and entry.name.endswith(".jpg")])

    jpg_files = jpg_files[:number_imgs]
    txt_files = [path.replace(".jpg", ".txt") for path in jpg_files]

    imgs = read_img_files(jpg_files)
    captions = read_txt_files(txt_files)

    # Initialize model and processor
    model, preprocesser = initialize_model_DFN()
    model.eval()

    # Compute cosine similarity
    DFN_score = compute_DFN_score(model, preprocesser, imgs, captions)

    print("DFN Scores:")
    print(DFN_score)