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


/**
 * Normalize a vector (L2 norm).
 * @param {Float32Array|number[]} vec
 * @returns {Float32Array} normalized vector
 */
function normalize(vec) {
  // 1. Compute the L2 norm (magnitude)
  let norm = 0.0;
  for (let i = 0; i < vec.length; i++) {
    norm += vec[i] * vec[i];
  }
  norm = Math.sqrt(norm);

  // 2. Divide each element by the norm
  const result = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) {
    result[i] = vec[i] / norm;
  }

  return result;
}

// Load tokenizer and text model
const processor = await AutoProcessor.from_pretrained('XudongShen/DFN-public');
const tokenizer = await AutoTokenizer.from_pretrained('XudongShen/DFN-public');
const text_model = await CLIPTextModelWithProjection.from_pretrained('XudongShen/DFN-public');
const vision_model = await CLIPVisionModelWithProjection.from_pretrained('XudongShen/DFN-public');

// Run tokenization
// const texts = ['a photo of a car', 'a photo of a football match'];
// const text_inputs = tokenizer(texts, { padding: "max_length", truncation: true });
// const { text_embeds } = await text_model(text_inputs);
// const text_embeds_first_five = text_embeds.data.slice(0, 5);
// console.log(text_embeds_first_five);
// Float32Array(5) [
//     -0.0008134003728628159,
//     -0.002951625734567642,
//     0.007977735251188278,
//     0.010096576064825058,
//     0.010253064334392548
//   ]


// const texts = ['Customizing Windows 7 Setup Please Help Solved'];
const texts = ['Jack Strong Official Trailer 1 (2015)   Patrick Wilson Drama Thriller HD'];
// const texts = ['WEN Fall Ginger Pumpkin Cleansing Conditioner ~ 16 oz ~  sealed  plus 6 oz Mist'];
// const texts = ['Couleur'];
const text_inputs = tokenizer(texts, { padding: "max_length", truncation: true });
const text_outputs = await text_model(text_inputs);
const textEmbedding = normalize( text_outputs.text_embeds.ort_tensor.cpuData );


// const image = await RawImage.read('/root/Documents/datadrive/datacomp_dataset/DFN/00000000_tar/000000000000.jpg');
const image = await RawImage.read('/root/Documents/datadrive/datacomp_dataset/DFN/00000000_tar/000000000001.jpg');
// const image = await RawImage.read('/root/Documents/datadrive/datacomp_dataset/DFN/00000000_tar/000000000003.jpg');
// const image = await RawImage.read('/root/Documents/datadrive/datacomp_dataset/DFN/00000000_tar/000000000004.jpg');
const image_inputs = await processor.image_processor([image]);
const image_outputs = await vision_model(image_inputs);
const imageEmbedding = normalize( image_outputs.image_embeds.ort_tensor.cpuData );

// Compute and log the cosine similarity
// Assume these are your Float32Array(512) embeddings
const similarity = cosineSimilarity(textEmbedding, imageEmbedding);
console.log("Cosine similarity:", similarity);

// Cosine similarity: 0.1144849144155551
// Cosine similarity: 0.24324223623890373
// Cosine similarity: 0.2740907674593308
// Cosine similarity: 0.05521306347289925