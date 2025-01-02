import { RawImage, AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection , CLIPVisionModelWithProjection, Processor} from '@huggingface/transformers';

function cosineSimilarity(vecA, vecB) {
    let dot = 0.0;
    let normA = 0.0;
    let normB = 0.0;
  
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
  
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

// Load tokenizer and text model
const processor = await AutoProcessor.from_pretrained('XudongShen/DFN-public');
const tokenizer = await AutoTokenizer.from_pretrained('XudongShen/DFN-public');
const text_model = await CLIPTextModelWithProjection.from_pretrained('XudongShen/DFN-public');
const vision_model = await CLIPVisionModelWithProjection.from_pretrained('XudongShen/DFN-public');

const texts = ['Customizing Windows 7 Setup Please Help Solved'];
// const texts = ['Jack Strong Official Trailer 1 (2015)   Patrick Wilson Drama Thriller HD'];
// const texts = ['WEN Fall Ginger Pumpkin Cleansing Conditioner ~ 16 oz ~  sealed  plus 6 oz Mist'];
const text_inputs = tokenizer(texts, { padding: "max_length", truncation: true });
const text_outputs = await text_model(text_inputs);
// console.log(text_outputs.text_embeds.ort_tensor.cpuData);
// console.log( computeL2Norm(text_outputs.text_embeds.ort_tensor.cpuData));

const image = await RawImage.read('/root/Documents/datadrive/datacomp_dataset/DFN/00000000_tar/000000000000.jpg');
// const image = await RawImage.read('/root/Documents/datadrive/datacomp_dataset/DFN/00000000_tar/000000000001.jpg');
// const image = await RawImage.read('/root/Documents/datadrive/datacomp_dataset/DFN/00000000_tar/000000000003.jpg');
const image_inputs = await processor.image_processor([image]);
const image_outputs = await vision_model(image_inputs);


// Compute and log the cosine similarity
// Assume these are your Float32Array(512) embeddings
const textEmbedding = text_outputs.text_embeds.ort_tensor.cpuData;
const imageEmbedding = image_outputs.image_embeds.ort_tensor.cpuData;
const similarity = cosineSimilarity(textEmbedding, imageEmbedding);
console.log("Cosine similarity:", similarity);