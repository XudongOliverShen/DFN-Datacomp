import os
from PIL import Image

import torch
from transformers import CLIPProcessor, set_seed
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

import time


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
    preprocessor = CLIPProcessor.from_pretrained(HF_model_name)

    text_model_proj = CLIPTextModelWithProjection.from_pretrained(HF_model_name)
    vision_model_proj = CLIPVisionModelWithProjection.from_pretrained(HF_model_name)

    
    return text_model_proj, vision_model_proj, preprocessor


def compute_DFN_score(text_model_proj, vision_model_proj, processor, imgs, texts):
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

    text_model_proj.eval()
    vision_model_proj.eval()

    # Get text embeddings
    text_embeddings = []
    for text in texts:
        # Tokenize the text
        tokens = processor.tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0)

        # Split into chunks of processor.tokenizer.model_max_length tokens
        token_chunks = [
            tokens[i : i + processor.tokenizer.model_max_length]
            for i in range(0, len(tokens), processor.tokenizer.model_max_length)
        ]

        # Compute embeddings for each chunk and average them
        chunk_embeddings = []
        for chunk in token_chunks:
            chunk_text = processor.tokenizer.decode(chunk, skip_special_tokens=True)
            text_inputs = processor(text=[chunk_text], return_tensors="pt", padding="max_length", truncation=True)
            with torch.no_grad():
                chunk_embedding = text_model_proj(**text_inputs)[0]
                chunk_embedding = chunk_embedding / chunk_embedding.norm(p=2, dim=-1, keepdim=True)

                chunk_embeddings.append(chunk_embedding)

        mean_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
        if len(chunk_embeddings) > 1:
            mean_embedding = mean_embedding / mean_embedding.norm(p=2, dim=-1, keepdim=True)
        text_embeddings.append(mean_embedding)

    text_embeddings = torch.cat(text_embeddings)

    # Get image embeddings
    image_embeddings = []
    for img in imgs:
        image_inputs = processor(images=[img], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            image_embedding = vision_model_proj(**image_inputs)[0]

        image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
        image_embeddings.append(image_embedding)

    image_embeddings = torch.cat(image_embeddings)

    # Compute cosine similarity
    cosine_similarity = (text_embeddings * image_embeddings).sum(dim=-1)

    return cosine_similarity.tolist()

if __name__ == "__main__":
    
    # set number of images caption pairs to process
    number_imgs = 5

    # must set random seed
    set_seed(abs(hash("gata")) % (10 ** 8))

    # list .jpg files using os.scandir
    # folder_path = os.path.join( os.path.dirname(os.path.abspath(__file__)), "00000000_tar")
    folder_path = "/root/Documents/datadrive/datacomp_dataset/xsmall/shards/00000000_tar"
    jpg_files = sorted([os.path.join(folder_path, entry.name) for entry in os.scandir(folder_path) if entry.is_file() and entry.name.endswith(".jpg")])

    jpg_files = jpg_files[:number_imgs]
    txt_files = [path.replace(".jpg", ".txt") for path in jpg_files]

    imgs = read_img_files(jpg_files)
    captions = read_txt_files(txt_files)

    # Initialize model and processor
    text_model_proj, vision_model_proj, preprocessor = initialize_model_DFN()

    # Compute cosine similarity
    DFN_score = compute_DFN_score(text_model_proj, vision_model_proj, preprocessor, imgs, captions)

    print("DFN Scores:")
    print(DFN_score)
    # check if results match
    # [0.12383436411619186, 0.24514636397361755, 0.2799367904663086, 0.06613056361675262, 0.06913130730390549]
    # [0.12383435666561127, 0.24514639377593994, 0.279936820268631, 0.06613057106733322, 0.06913133710622787]