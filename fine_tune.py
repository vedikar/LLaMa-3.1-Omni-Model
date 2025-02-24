from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, get_peft_model
from bitsandbytes import BitsAndBytesConfig

# Step 1: Load the tokenized dataset
tokenized_dataset = load_from_disk("tokenized_dataset")

# Step 2: Split dataset into training and validation sets (80/20 split)
split_datasets = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# Step 3: Check GPU availability
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"
torch_dtype = torch.float16 if use_gpu else torch.float32

# Step 4: Enable 4-bit QLoRA to fit model in memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # NF4 quantization (more efficient)
    bnb_4bit_compute_dtype=torch_dtype,
    llm_int8_enable_fp32_cpu_offload=True,  # Offload model parts to CPU
)

# Step 5: Load tokenizer and model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # Use 4-bit quantization
    device_map="auto",
)

# Step 6: Apply LoRA for Low-Rank Adaptation (Memory Efficient Fine-Tuning)
lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=16,  # Scaling parameter
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to query & value projections
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Step 7: Fix missing `pad_token`
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# Step 8: Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal Language Modeling (CLM)
)

# Step 9: Define Training Arguments
training_args = TrainingArguments(
    output_dir="fine_tuned_llama",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Reduce batch size for low VRAM
    gradient_accumulation_steps=8,  # Accumulate gradients to simulate larger batch size
    save_steps=500,  # Save model periodically
    save_total_limit=1,
    logging_dir="logs",
    logging_steps=50,
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    eval_accumulation_steps=2,  # Helps with large validation sets
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=use_gpu,  # Enable FP16 if using GPU
    push_to_hub=False,
    report_to="none",
)

# Step 10: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 11: Fine-tune the model
trainer.train()

# Step 12: Save fine-tuned model
model.save_pretrained("fine_tuned_llama")
tokenizer.save_pretrained("fine_tuned_llama")
print("Fine-tuning complete. Model saved to 'fine_tuned_llama'.")
