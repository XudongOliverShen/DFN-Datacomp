import { RawImage, AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection , CLIPVisionModelWithProjection, Processor} from '@huggingface/transformers';

// Load tokenizer and text model
const processor = await AutoProcessor.from_pretrained('XudongShen/DFN-public');
const tokenizer = await AutoTokenizer.from_pretrained('XudongShen/DFN-public');
const text_model = await CLIPTextModelWithProjection.from_pretrained('XudongShen/DFN-public');
const vision_model = await CLIPVisionModelWithProjection.from_pretrained('XudongShen/DFN-public');

// Run tokenization
const texts = ['a photo of a car', 'a photo of a football match'];
const text_inputs = tokenizer(texts, { padding: "max_length", truncation: true });

// Compute embeddings
// print the first 5 numbers of text_embeds
const { text_embeds } = await text_model(text_inputs);
// const text_embeds_first_five = text_embeds.data.slice(0, 5);
// console.log(text_embeds_first_five);
// Float32Array(5) [
//     -0.0008134003728628159,
//     -0.002951625734567642,
//     0.007977735251188278,
//     0.010096576064825058,
//     0.010253064334392548
//   ]

const image = await RawImage.read('/root/Documents/datadrive/datacomp_dataset/DFN/00000000_tar/000000000000.jpg');
// const image_inputs = await processor([image], { padding: true, truncation: true });
const image_inputs = await processor.image_processor([image], { padding: true, truncation: true });
console.log(image_inputs.pixel_values.ort_tensor);
// console.log(image_inputs.pixel_values.ort_tensor.cpuData.slice(0,5));

// const image_outputs = await vision_model(image_inputs);
// const arrayData = image_outputs.image_embeds.ort_tensor.cpuData;
// console.log(arrayData.slice(0, 5));
// const image_embeds_first_five = image_embeds;
// console.log(image_embeds_first_five);