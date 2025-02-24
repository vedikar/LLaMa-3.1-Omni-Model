from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import os
import torch

# Step 1: Log in to Hugging Face
login(token="Hugging face token")

# Step 2: Define the dataset path
audio_dir = r"C:\Users\Vedika\.cache\LibriSpeech\train-clean-100"

# Function to load transcriptions from a `.trans` file
def load_transcriptions_from_file(trans_file_path):
    transcriptions = {}
    try:
        with open(trans_file_path, 'r') as f:
            for line in f:
                # Split each line into the audio file ID and the transcription
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:  # Ensure the line contains both ID and transcription
                    audio_id, transcription = parts
                    transcriptions[audio_id] = transcription
    except FileNotFoundError:
        print(f"Warning: Transcription file not found: {trans_file_path}")
    except Exception as e:
        print(f"Error reading transcription file: {e}")
    return transcriptions

# Step 3: Prepare the dataset
data = []
for root, _, files in os.walk(audio_dir):
    # Identify transcription files in the folder
    trans_files = [file for file in files if file.endswith('.trans.txt')]
    if trans_files:
        # Load transcriptions from the `.trans` file
        trans_file_path = os.path.join(root, trans_files[0])
        transcriptions = load_transcriptions_from_file(trans_file_path)
        
        # Process each `.flac` file in the folder
        for file in files:
            if file.endswith('.flac'):
                audio_id = os.path.splitext(file)[0]  # Remove `.flac` extension
                transcription = transcriptions.get(audio_id, None)
                if transcription:
                    data.append({"sentence": transcription})
                else:
                    print(f"Warning: No transcription found for {audio_id} in {trans_file_path}")

# Step 4: Create the dataset
if len(data) == 0:
    raise ValueError("No valid transcriptions found! Please check the dataset structure.")
dataset = Dataset.from_list(data)

# Step 5: Load the tokenizer and model
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Fix for missing `pad_token`
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# Step 6: Tokenize the dataset
def preprocess_data(batch):
    input_texts = batch["sentence"]
    tokenized = tokenizer(
        input_texts,
        truncation=True,
        padding="longest",  # Dynamic padding for efficiency
        return_tensors="pt"
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

# Apply tokenization
tokenized_dataset = dataset.map(preprocess_data, batched=True, batch_size=32, remove_columns=["sentence"])

# Step 7: Save the tokenized dataset
output_dir = "tokenized_dataset"
os.makedirs(output_dir, exist_ok=True)
tokenized_dataset.save_to_disk(output_dir)
print(f"Dataset tokenization complete. Saved to {output_dir}.")

# Step 8: Verify the size of the tokenized dataset
from datasets import load_from_disk

# Load the tokenized dataset
loaded_tokenized_dataset = load_from_disk(output_dir)

# Print the number of examples
print(f"Number of tokenized examples: {len(loaded_tokenized_dataset)}")

