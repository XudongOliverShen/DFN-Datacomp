import os
from PIL import Image

import torch
from transformers import CLIPProcessor, set_seed
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection

import time
import itertools



def compute_accuracy(pair_similarity):
    # Ensure the pair_similarity is a square matrix
    assert pair_similarity.shape[0] == pair_similarity.shape[1], "The input must be a square matrix."

    # Get the size of the square matrix
    n = pair_similarity.shape[0]

    # Compute the accuracy
    correct_count = 0
    for i in range(n):
        # Check if the diagonal element is the largest in the row
        if pair_similarity[i, i] == torch.max(pair_similarity[i]):
            correct_count += 1

    # Calculate average accuracy
    accuracy = correct_count / n
    return accuracy


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

def pad_to_square(img, fill_color=(0,0,0)):
    """
    Pad a PIL Image so that its width and height become equal.
    The shorter side is padded with the specified fill_color (default is black).
    """
    width, height = img.size
    max_dim = max(width, height)
    
    # Create a new square image with the max dimension
    new_img = Image.new(mode='RGB', size=(max_dim, max_dim), color=fill_color)
    
    # To center the original image, compute the offsets
    offset_x = (max_dim - width) // 2
    offset_y = (max_dim - height) // 2
    
    # Paste the original image onto the square canvas
    new_img.paste(img, (offset_x, offset_y))
    
    return new_img

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
        image_inputs = processor(images=[img], return_tensors="pt")
        with torch.no_grad():
            image_embedding = vision_model_proj(**image_inputs)[0]

        image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
        image_embeddings.append(image_embedding)

    image_embeddings = torch.cat(image_embeddings)

    # Compute cosine similarity
    cosine_similarity = (text_embeddings * image_embeddings).sum(dim=-1)

    A = compute_accuracy( torch.matmul(text_embeddings, image_embeddings.T) )

    # no padding
    # 100
    # 0.22
    # 200
    # 0.185
    # 500
    # 0.128
    # 1000
    # 0.11
    # 2000
    # 0.11

    # 2000
    # 0.07953976988494248

    # padding 
    # 100
    # 0.25
    # 200
    # 0.175
    # 500
    # 0.126
    # 1000
    # 0.1
    # 2000
    # 0.1

    # 2000
    # 0.0830415207603802





    # no padding
    # 10: 0.18 
    # 11: 0.22 1
    # 12: 0.14
    # 13: 0.17
    # 14: 0.15
    # 15: 0.14
    # 16: 0.21
    # 18: 0.17
    # 19: 0.13 1
    # 20: 0.23 1
    # 21: 0.28
    # 21: 0.19 1

    # padding 
    # 10: 0.19 1
    # 11: 0.21
    # 12: 0.14
    # 13: 0.19 1
    # 14: 0.16 1
    # 15: 0.17 1
    # 16: 0.24 1
    # 18: 0.17
    # 19: 0.1
    # 20: 0.22
    # 21: 0.28
    # 22: 0.18

    # print(cosine_similarity)
    # tensor([0.1238, 0.2451, 0.2799, 0.0661, 0.0691, 0.1284, 0.1985, 0.2215, 0.0797,
    #     0.2521])

#     print(cosine_similarity)
# tensor([0.1238, 0.2451, 0.2799, 0.0661, 0.0691, 0.1284, 0.1985, 0.2215, 0.0797,
#         0.2521])

    return cosine_similarity.tolist()

if __name__ == "__main__":
    
    # set number of images caption pairs to process
    number_imgs = 10

    # must set random seed
    set_seed(abs(hash("gata")) % (10 ** 8))

    # list .jpg files using os.scandir
    # folder_path = os.path.join( os.path.dirname(os.path.abspath(__file__)), "00000000_tar")
    folder_path = "/root/Documents/datadrive/datacomp_dataset/xsmall/shards/00000000_tar"
    jpg_files = sorted([os.path.join(folder_path, entry.name) for entry in os.scandir(folder_path) if entry.is_file() and entry.name.endswith(".jpg")])

    # i = 1
    # jpg_files = jpg_files[number_imgs*i:number_imgs*(i+1)]
    jpg_files = jpg_files[:number_imgs]
    txt_files = [path.replace(".jpg", ".txt") for path in jpg_files]

    imgs = read_img_files(jpg_files)
    captions = read_txt_files(txt_files)

    imgs_valid = []
    captions_valid = []
    for img, caption in itertools.zip_longest(imgs, captions):
        if img.size[0] > 1 and img.size[1] > 1:
            imgs_valid.append(img)
            captions_valid.append(caption)

    imgs = imgs_valid
    captions = captions_valid

    # Initialize model and processor
    text_model_proj, vision_model_proj, preprocessor = initialize_model_DFN()


    # texts = ['a photo of a car', 'a photo of a football match']
    # text_inputs = preprocessor(text=texts, return_tensors="pt", padding="max_length", truncation=True)
    # with torch.no_grad():
    #     text_embeddings = text_model_proj(**text_inputs)[0]
    # # 测试1: 确保js输出是一致的
    # # print(text_embeddings[0,:5].tolist())
    # # [-0.0008133882656693459, -0.002951593603938818, 0.007977750152349472, 0.010096581652760506, 0.010253100655972958]

    # # image_inputs = preprocessor(images=[imgs[0]], return_tensors="pt", padding=True, truncation=True)
    # image_inputs = preprocessor(images=[imgs[0]], return_tensors="pt")
    # # image_inputs['pixel_values'].norm()
    # # tensor(608.6111)

    # image_embedding = vision_model_proj(**image_inputs)[0]
    # # 测试2: 确保js输出是一致的
    # # print(image_embedding[0,:5].tolist())
    # # [-0.00812644325196743, 0.01711365208029747, 0.016149630770087242, -0.004290394484996796, -0.024714656174182892]









    imgs = [pad_to_square(img) for img in imgs]
    # image_inputs = preprocessor(images=[imgs[0]], return_tensors="pt")
    # # image_inputs["pixel_values"].norm()
    # # tensor(635.2685)

    # image_embedding = vision_model_proj(**image_inputs)[0]
    # print(image_embedding[0,:5].tolist())
    # # [-0.004965551197528839, -0.012129547074437141, 0.01617734134197235, -0.01127003412693739, -0.013096662238240242]
    # A = 1
    # # [-0.00812644325196743, 0.01711365208029747, 0.016149630770087242, -0.004290394484996796, -0.024714656174182892]







    # Compute cosine similarity
    DFN_score = compute_DFN_score(text_model_proj, vision_model_proj, preprocessor, imgs, captions)

    print("DFN Scores:")
    print(DFN_score)
    # 测试3: 确保js输出是一致的
    # [0.1001691222190857, 0.24619701504707336, 0.279936820268631, 0.04590398445725441, 0.11274418234825134, 0.12837016582489014, 0.14036184549331665, 0.27388766407966614, 0.10299073159694672, 0.2248506247997284]

    
