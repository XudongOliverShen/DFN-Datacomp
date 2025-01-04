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

const texts = ['Customizing Windows 7 Setup Please Help Solved'];
const text_inputs = tokenizer(texts, { padding: "max_length", truncation: true });
const text_outputs = await text_model(text_inputs);
const textEmbedding = normalize( text_outputs.text_embeds.ort_tensor.cpuData );
console.log("textEmbedding:", textEmbedding.slice(textEmbedding.length / 2, textEmbedding.length/2+5));
// textEmbedding: Float32Array(5) [
//   0.019684800878167152,
//   0.001726271933875978,
//   -0.0503770112991333,
//   0.018079029396176338,
//   -0.16432690620422363
// ]


const image = await RawImage.read('/root/Documents/datadrive/datacomp_dataset/DFN/00000000_tar/000000000000.jpg');
console.log("image:", image);
// image: RawImage {
//   data: Uint8ClampedArray(414720) [
//     147, 149, 162, 171, 173, 186, 172, 174, 186, 171, 173, 185,
//     171, 173, 185, 170, 172, 184, 170, 174, 185, 167, 171, 182,
//     168, 171, 186, 172, 175, 190, 170, 173, 188, 168, 171, 186,
//     169, 175, 189, 165, 171, 185, 165, 171, 183, 171, 174, 189,
//     170, 171, 189, 173, 171, 192, 168, 169, 187, 169, 170, 188,
//     170, 173, 188, 170, 173, 188, 166, 172, 186, 167, 173, 187,
//     166, 172, 188, 167, 173, 189, 171, 174, 193, 166, 169, 188,
//     172, 173, 191, 171, 172, 190, 170, 169, 185, 169, 171, 186,
//     173, 175, 190, 167,
//     ... 414620 more items
//   ],
//   width: 512,
//   height: 270,
//   channels: 3
// }


const image_inputs = await processor.image_processor([image]);
console.log("image_inputs:", image_inputs);
// image_inputs: {
//   pixel_values: Tensor {
//     ort_tensor: Tensor {
//       cpuData: [Float32Array],
//       dataLocation: 'cpu',
//       type: 'float32',
//       dims: [Array],
//       size: 150528
//     }
//   },
//   original_sizes: [ [ 270, 512 ] ],
//   reshaped_input_sizes: [ [ 224, 224 ] ]
// }


console.log("image_inputs:", image_inputs.pixel_values.ort_tensor.cpuData.slice(image_inputs.pixel_values.ort_tensor.cpuData.length / 2, image_inputs.pixel_values.ort_tensor.cpuData.length/2+5));
// image_inputs: Float32Array(5) [
//   -0.2513202428817749,
//   -1.7520971298217773,
//   -1.7520971298217773,
//   -1.7520971298217773,
//   -1.7520971298217773
// ]

const image_outputs = await vision_model(image_inputs);
console.log("image_outputs:", image_outputs.image_embeds.ort_tensor.cpuData.slice(image_outputs.image_embeds.ort_tensor.cpuData.length / 2, image_outputs.image_embeds.ort_tensor.cpuData.length/2+5));
// image_outputs: Float32Array(5) [
//   -0.007455918937921524,
//   0.008535768836736679,
//   0.01811673864722252,
//   0.004406895488500595,
//   -0.0200477484613657
// ]

const imageEmbedding = normalize( image_outputs.image_embeds.ort_tensor.cpuData );
console.log("imageEmbedding:", imageEmbedding.slice(imageEmbedding.length / 2, imageEmbedding.length/2+5));
// imageEmbedding: Float32Array(5) [
//   -0.022664044052362442,
//   0.02594650536775589,
//   0.055070146918296814,
//   0.013395809568464756,
//   -0.060939911752939224
// ]


// Compute and log the cosine similarity
// Assume these are your Float32Array(512) embeddings
const similarity = cosineSimilarity(textEmbedding, imageEmbedding);
console.log("Cosine similarity:", similarity);
// Cosine similarity: 0.1144849144155551