import glob
import io
import os
import mediapy as media
import numpy as np
from PIL import Image
from PIL import Image
import tensorflow_text
from tips.pytorch import image_encoder
from tips.pytorch import text_encoder
from tips.scenic.utils import feature_viz
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import logging
import json
import time


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global parameters
IMAGE_MEAN = (0, 0, 0)
IMAGE_STD = (1.0, 1.0, 1.0)
image_size = 448

def get_image_paths(folder_path):
    """Returns an iterable of image paths from a given folder."""
    try:
        logging.info(f"Reading images from folder: {folder_path}")
        image_paths = glob.glob(os.path.join(folder_path, '*'))
        if not image_paths:
            logging.warning(f"No images found in {folder_path}")
        return image_paths
    except Exception as e:
        logging.error(f"Error reading image paths from {folder_path}: {e}")
        return []

def load_image_bytes(file_name):
    """Loads an image from a file and applies preprocessing."""
    try:
        logging.info(f"Loading image: {file_name}")
        with open(file_name, 'rb') as fd:
            image_bytes = io.BytesIO(fd.read())
            pil_image = Image.open(image_bytes).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ])
            return transform(pil_image).unsqueeze(0)
    except Exception as e:
        logging.error(f"Error loading image {file_name}: {e}")
        return None

def save_image(output_folder, original_path, image_tensor):
    """Saves the processed image to the output folder with the same filename."""
    try:
        os.makedirs(output_folder, exist_ok=True)
        image_filename = os.path.basename(original_path)
        output_path = os.path.join(output_folder, image_filename)

        # Convert tensor to numpy array
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # Convert normalization parameters to NumPy arrays
        image_std_np = np.array(IMAGE_STD)
        image_mean_np = np.array(IMAGE_MEAN)

        # Denormalize the image
        image_np = (image_np * image_std_np) + image_mean_np
        image_np = (image_np * 255.0).clip(0, 255).astype('uint8')

        # Convert back to PIL and save
        image_pil = Image.fromarray(image_np)
        image_pil.save(output_path)
        logging.info(f"Image saved: {output_path}")

    except Exception as e:
        logging.error(f"Error saving image {original_path}: {e}")


class EmbeddingProcessor:
    def __init__(self, text_encoder_checkpoint, image_encoder_checkpoint, tokenizer_path, text_prompts, image_size=448):
        """Initialize embedding processor with model checkpoints."""
        logging.info("Initializing EmbeddingProcessor...")
        self.text_encoder_checkpoint = text_encoder_checkpoint
        self.image_encoder_checkpoint = image_encoder_checkpoint
        self.tokenizer_path = tokenizer_path
        self.image_size = image_size  # Store image size as a class attribute

        self.text_prompts = text_prompts
        self.model_text = None
        self.embeddings_text = None
        self.temperature = None
        
        self.model_image = None  # Image model instance

        # Store results
        self.cls_token = None
        self.spatial_features = []
        self.embeddings_image = []
        self.similarity_results = []

        self._load_text_embeddings()
        self._load_image_embeddings()



    def _load_image_bytes(self, file_name):
        """Loads and preprocesses an image."""
        try:
            logging.info(f"Loading image: {file_name}")
            with open(file_name, 'rb') as fd:
                image_bytes = io.BytesIO(fd.read())
                pil_image = Image.open(image_bytes).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0, 0, 0), (1.0, 1.0, 1.0)),
                ])
                return transform(pil_image).unsqueeze(0)
        except Exception as e:
            logging.error(f"Error loading image {file_name}: {e}")
            return None

    def _load_text_embeddings(self):
        """Load and process text embeddings from the checkpoint."""
        try:
            logging.info("Loading text embeddings...")
            with open(self.text_encoder_checkpoint, 'rb') as fin:
                inbuffer = io.BytesIO(fin.read())
            np_weights_text = np.load(inbuffer, allow_pickle=False)

            weights_text = {key: torch.from_numpy(value) for key, value in np_weights_text.items()}
            self.temperature = weights_text.pop('temperature')

            with torch.no_grad():
                variant = "S"  # Adjust variant as needed
                self.model_text = text_encoder.TextEncoder(
                    self._get_text_config(variant),
                    vocab_size=32000,
                )
                self.model_text.load_state_dict(weights_text)
                tokenizer = text_encoder.Tokenizer(self.tokenizer_path)

                
                text_ids, text_paddings = tokenizer.tokenize(self.text_prompts, max_len=64)
                self.embeddings_text = feature_viz.normalize(
                    self.model_text(
                        torch.from_numpy(text_ids), torch.from_numpy(text_paddings)
                    )
                )

            logging.info("Text embeddings successfully loaded.")
        except Exception as e:
            logging.error(f"Error loading text embeddings: {e}")

    def _get_text_config(self, v):
        """Returns text encoder configuration based on variant."""
        return {
            "hidden_size": {"S": 384, "B": 768, "L": 1024, "So400m": 1152, "g": 1536}[v],
            "mlp_dim": {"S": 1536, "B": 3072, "L": 4096, "So400m": 4304, "g": 6144}[v],
            "num_heads": {"S": 6, "B": 12, "L": 16, "So400m": 16, "g": 24}[v],
            "num_layers": {"S": 12, "B": 12, "L": 12, "So400m": 27, "g": 12}[v],
        }

    def _load_image_embeddings(self):
        """Load image encoder from checkpoint."""
        try:
            logging.info("Loading image encoder weights...")
            weights_image = dict(np.load(self.image_encoder_checkpoint, allow_pickle=False))
            for key in weights_image:
                weights_image[key] = torch.tensor(weights_image[key])

            variant = "S"  # Adjust variant as needed
            ffn_layer = "swiglu" if variant == "g" else "mlp"

            logging.info("Initializing vision model...")
            with torch.no_grad():
                self.model_image = image_encoder.vit_small(
                    img_size=self.image_size,
                    patch_size=14,
                    ffn_layer=ffn_layer,
                    block_chunks=0,
                    init_values=1.0,
                    interpolate_antialias=True,
                    interpolate_offset=0.0,
                )
                self.model_image.load_state_dict(weights_image)

            logging.info("Image encoder successfully loaded.")
        except Exception as e:
            logging.error(f"Error loading image encoder: {e}")

    def infer_image_embeddings(self, image_paths):
        """Run inference on image embeddings."""
        try:
            if not self.model_image:
                logging.error("Image encoder model is not loaded.")
                return

            logging.info(f"Running inference on {len(image_paths)} images...")

            with torch.no_grad():
                for image_path in image_paths:
                    input_batch = self._load_image_bytes(image_path)
                    if input_batch is None:
                        logging.warning(f"Skipping {image_path} due to loading error.")
                        continue

                    output = self.model_image(input_batch)

                    cls_token = feature_viz.normalize(output[0][0][0])
                    spatial_feature = torch.reshape(
                        output[2],
                        (1, int(self.image_size / 14), int(self.image_size / 14), -1),
                    )

                    self.cls_token = cls_token
                    self.spatial_features.append(spatial_feature)
                    self.embeddings_image.append(cls_token)

                    logging.info(f"Processed image: {image_path}")

            logging.info("Image inference complete.")
        except Exception as e:
            logging.error(f"Error during image inference: {e}")

    def compute_cosine_similarity(self, output_folder):
        """Compute cosine similarity and save results."""
        logging.info("Computing cosine similarity...")
        similarity_scores = []

        for idx, embedding_image in enumerate(self.embeddings_image):
            cos_sim = F.softmax(
                ((embedding_image.unsqueeze(0) @ self.embeddings_text.T) / self.temperature), dim=-1
            )
            label_idxs = torch.argmax(cos_sim, axis=-1)
            cos_sim_max = torch.max(cos_sim, axis=-1)
            label_predicted = self.text_prompts[label_idxs[0]]
            similarity = cos_sim_max.values[0].item()

            similarity_scores.append({
                "image_index": idx,
                "predicted_label": label_predicted,
                "similarity": similarity
            })

            logging.info(f"Image {idx}: {label_predicted} ({similarity:.2f})")

        # Save to JSON file
        similarity_output_path = os.path.join(output_folder, "similarity_scores.json")
        os.makedirs(output_folder, exist_ok=True)

        with open(similarity_output_path, "w") as f:
            json.dump(similarity_scores, f, indent=4)

        logging.info(f"Similarity scores saved to {similarity_output_path}")

    def generate_pca_visualization(self, image_paths, output_folder):
        """Generate and save PCA visualizations with proper size."""
        logging.info("Generating PCA visualizations...")
        pca_folder = os.path.join(output_folder, "pca_visualizations")
        os.makedirs(pca_folder, exist_ok=True)

        for idx, spatial_feature in enumerate(self.spatial_features):
            pca_obj = feature_viz.PCAVisualizer(spatial_feature)
            image_pca = pca_obj(spatial_feature)[0]  # PCA visualization output

            with open(image_paths[idx], 'rb') as f:
                image = Image.open(f)
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                image = np.array(image).astype(np.float32) / 255.0

            # Convert PCA image to match input image size
            pca_resized = Image.fromarray((image_pca * 255).astype(np.uint8))
            pca_resized = pca_resized.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

            # Save PCA visualization
            pca_output_path = os.path.join(pca_folder, f"pca_{idx}.png")
            pca_resized.save(pca_output_path)

            logging.info(f"PCA visualization saved: {pca_output_path}")

    
    def run_pipeline(self, image_paths, output_folder):
        """Run the full pipeline for processing a sequence of images."""
        logging.info(f"Running pipeline on {len(image_paths)} images...")
        self.infer_image_embeddings(image_paths)
        self.compute_cosine_similarity(output_folder)
        self.generate_pca_visualization(image_paths, output_folder)



if __name__ == "__main__":
    # Set paths
    text_encoder_checkpoint = "/usr/mvl2/jdmcnw/Projects/2025/VOTS/TIPS/tips/pytorch/checkpoints/tips_oss_s14_highres_distilled_text.npz"
    image_encoder_checkpoint = "/usr/mvl2/jdmcnw/Projects/2025/VOTS/TIPS/tips/pytorch/checkpoints/tips_oss_s14_highres_distilled_vision.npz"
    tokenizer_path = "/usr/mvl2/jdmcnw/Projects/2025/VOTS/TIPS/tips/pytorch/checkpoints/tokenizer.model"
    
    image_folder = "images/inputs/basketball/color/"
    output_folder = "images/outputs/basketball/pca/"
    image_paths = get_image_paths(image_folder)

    text_prompts = [
                    "Basketball player", "basketball", "basketball jersey", "basketball hoop", "basketball court", "basketball game", "basketball team", "basketball shoes"]

    start_time = time.time()
    processor = EmbeddingProcessor(text_encoder_checkpoint, image_encoder_checkpoint, tokenizer_path,text_prompts)

    # Run pipeline
    processor.run_pipeline(image_paths, output_folder)

    end_time = time.time()
    inference_time = end_time - start_time
    logging.info(f"Inference time for the sequence: {inference_time:.2f} seconds")
    logging.info("Image path reading, loading, and saving complete.")
