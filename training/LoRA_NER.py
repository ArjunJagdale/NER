# !pip install transformers datasets seqeval torch accelerate peft matplotlib -q

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import torch

print("✓ All libraries imported successfully!\n")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\n" + "="*80)
print("LOADING FEW-NERD DATASET (FULL)")
print("="*80)
# Load the supervised setting with fine-grained entity types
dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")

print(f"\nDataset splits:")
print(f"  Train:      {len(dataset['train']):,} sentences")
print(f"  Validation: {len(dataset['validation']):,} sentences")
print(f"  Test:       {len(dataset['test']):,} sentences")
print(f"  Total:      {sum(len(dataset[split]) for split in dataset.keys()):,} sentences")

# We'll use a larger subset since training is fast with LoRA
print("\n" + "-"*80)
print("SAMPLING STRATEGY FOR LORA")
print("-"*80)

def sample_dataset(dataset, n_samples, seed=42):
    """Sample n_samples from dataset with fixed seed for reproducibility"""
    if n_samples >= len(dataset):
        return dataset
    indices = np.random.RandomState(seed).choice(
        len(dataset), size=n_samples, replace=False
    )
    return dataset.select(indices)

# Configuration: Since your 20K took only 5 min, let's use much more!
# LoRA is efficient, so we can handle larger datasets
TRAIN_SAMPLES = 100000   # ~76% of full training data
VAL_SAMPLES = 15000      # ~80% of validation data
RANDOM_SEED = 42

dataset['train'] = sample_dataset(dataset['train'], TRAIN_SAMPLES, RANDOM_SEED)
dataset['validation'] = sample_dataset(dataset['validation'], VAL_SAMPLES, RANDOM_SEED)

print(f"\n✓ Dataset configuration (Larger for LoRA efficiency):")
print(f"  Train:      {len(dataset['train']):,} sentences ({100*len(dataset['train'])/131767:.1f}% of original)")
print(f"  Validation: {len(dataset['validation']):,} sentences ({100*len(dataset['validation'])/18824:.1f}% of original)")
print(f"  Test:       {len(dataset['test']):,} sentences (100% - full test set)")
print(f"  Total:      {len(dataset['train']) + len(dataset['validation']):,} training samples")
print(f"\n  Expected training time: ~20-30 minutes (with LoRA on GPU)")

# Examine dataset structure
print(f"\n✓ Example from training set:")
example = dataset['train'][0]
print(f"  Tokens: {example['tokens'][:15]}...")
print(f"  NER Tags: {example['fine_ner_tags'][:15]}...")

# ============================================================================
# SETUP MODEL AND TOKENIZER
# ============================================================================

print("\n" + "="*80)
print("LOADING MODEL AND TOKENIZER")
print("="*80)

MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

# Get fine-grained label list
label_list = dataset['train'].features['fine_ner_tags'].feature.names
num_labels = len(label_list)

print(f"\n✓ Model: {MODEL_NAME}")
print(f"✓ Total entity types: {num_labels}")
print(f"\nEntity type categories:")
print("  • Person (actor, artist, athlete, director, politician, etc.)")
print("  • Location (GPE, body of water, island, mountain, road, etc.)")
print("  • Organization (company, education, government, media, etc.)")
print("  • Building (airport, hospital, hotel, library, restaurant, etc.)")
print("  • Art (broadcast, film, music, painting, written work, etc.)")
print("  • Product (airplane, car, food, game, ship, software, weapon, etc.)")
print("  • Event (attack/battle, disaster, election, protest, sports, etc.)")
print("  • Other (astronomy, biology, chemistry, currency, disease, law, etc.)")

# Create label mappings
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Load base model
base_model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

print(f"\n✓ Base model loaded with {num_labels} entity types")

print("\n" + "="*80)
print("CONFIGURING LORA (PEFT)")
print("="*80)

# LoRA configuration for RoBERTa
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,  # Token classification task
    r=16,                          # Rank of the low-rank matrices
    lora_alpha=32,                 # Scaling parameter (typically 2*r)
    lora_dropout=0.1,              # Dropout probability
    target_modules=["query", "value"],  # Apply LoRA to attention Q and V
    bias="none",                   # Don't train bias parameters
    inference_mode=False           # Training mode
)

print("LoRA Configuration:")
print(f"  Rank (r): {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")
print(f"  Dropout: {lora_config.lora_dropout}")
print(f"  Target modules: {lora_config.target_modules}")
print(f"  Task type: {lora_config.task_type}")

# Apply LoRA to the model
model = get_peft_model(base_model, lora_config)

# Print model parameters
print("\n" + "="*80)
print("MODEL PARAMETER ANALYSIS")
print("="*80)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
trainable_percentage = 100 * trainable_params / total_params

print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters (LoRA): {trainable_params:,} ({trainable_percentage:.2f}%)")
print(f"Frozen Parameters: {total_params - trainable_params:,} ({100-trainable_percentage:.2f}%)")
print(f"\nParameter Efficiency: {total_params / trainable_params:.1f}x fewer trainable params")
print(f"Memory Savings: ~{(1 - trainable_percentage/100) * 100:.1f}% reduction in trainable memory")

model.print_trainable_parameters()

# ============================================================================
# TOKENIZATION AND LABEL ALIGNMENT
# ============================================================================

print("\n" + "="*80)
print("TOKENIZING DATASETS")
print("="*80)

def tokenize_and_align_labels(examples):
    """
    Tokenize texts and align NER labels with subword tokens.
    Few-NERD uses IO tagging, so we handle entity boundaries properly.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding=False,
        max_length=512
    )

    labels = []
    for i, label in enumerate(examples["fine_ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the label
                label_ids.append(label[word_idx])
            else:
                # Other subwords get -100 (ignored)
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize all splits
tokenized_train = dataset['train'].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing train"
)

tokenized_val = dataset['validation'].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset['validation'].column_names,
    desc="Tokenizing validation"
)

tokenized_test = dataset['test'].map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset['test'].column_names,
    desc="Tokenizing test"
)

print(f"\n✓ Tokenization complete!")
print(f"  Train samples: {len(tokenized_train):,}")
print(f"  Validation samples: {len(tokenized_val):,}")
print(f"  Test samples: {len(tokenized_test):,}")

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(eval_pred):
    """Compute precision, recall, and F1 using seqeval"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_labels = [[id2label[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [[id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                        for prediction, label in zip(predictions, labels)]

    # Calculate metrics
    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ============================================================================
# METRICS TRACKING CALLBACK
# ============================================================================

class MetricsCallback(TrainerCallback):
    """Callback to track and store metrics during training"""

    def __init__(self):
        self.eval_losses = []
        self.eval_f1s = []
        self.eval_precisions = []
        self.eval_recalls = []
        self.epochs = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.epochs.append(state.epoch)
            self.eval_losses.append(metrics.get('eval_loss', 0))
            self.eval_f1s.append(metrics.get('eval_f1', 0))
            self.eval_precisions.append(metrics.get('eval_precision', 0))
            self.eval_recalls.append(metrics.get('eval_recall', 0))

metrics_callback = MetricsCallback()

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("TRAINING CONFIGURATION")
print("="*80)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="./roberta-lora-fewnerd-100k",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,              # Higher LR for LoRA (standard practice)
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,              # More epochs since we have more data
    weight_decay=0.01,
    warmup_ratio=0.1,                # Warmup 10% of steps
    logging_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    report_to="none",
    push_to_hub=False,
    fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    gradient_accumulation_steps=2,   # Effective batch size = 32
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback]
)

print(f"\n✓ Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size per device: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Warmup ratio: {training_args.warmup_ratio}")
print(f"  Mixed precision (FP16): {training_args.fp16}")
print(f"  Total training steps: ~{len(tokenized_train) * training_args.num_train_epochs // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "="*80)
print("STARTING LORA TRAINING")
print("="*80)
print(f"Training samples: {len(tokenized_train):,}")
print(f"Validation samples: {len(tokenized_val):,}")
print(f"Entity types: {num_labels}")
print(f"Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}%)")
print("="*80 + "\n")

# Train the model
trainer.train()

print("\n✓ Training completed successfully!")

# ============================================================================
# PLOT TRAINING METRICS
# ============================================================================

print("\n" + "="*80)
print("GENERATING TRAINING PLOTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('LoRA Training Metrics - RoBERTa on Few-NERD', fontsize=16, fontweight='bold')

# Plot 1: Loss
axes[0, 0].plot(metrics_callback.epochs, metrics_callback.eval_losses, 'b-o', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Loss', fontsize=12)
axes[0, 0].set_title('Validation Loss', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: F1 Score
axes[0, 1].plot(metrics_callback.epochs, metrics_callback.eval_f1s, 'g-o', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('F1 Score', fontsize=12)
axes[0, 1].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 1])

# Plot 3: Precision
axes[1, 0].plot(metrics_callback.epochs, metrics_callback.eval_precisions, 'r-o', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Precision', fontsize=12)
axes[1, 0].set_title('Validation Precision', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim([0, 1])

# Plot 4: Recall
axes[1, 1].plot(metrics_callback.epochs, metrics_callback.eval_recalls, 'orange', marker='o', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Recall', fontsize=12)
axes[1, 1].set_title('Validation Recall', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('lora_training_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Training plots saved as 'lora_training_metrics.png'")
plt.show()

# Print essential metrics only
print("\n" + "="*80)
print("TRAINING PROGRESS")
print("="*80)
print(f"Best Validation F1: {max(metrics_callback.eval_f1s):.4f} (Epoch {metrics_callback.epochs[metrics_callback.eval_f1s.index(max(metrics_callback.eval_f1s))]:.0f})")
print(f"Final Validation F1: {metrics_callback.eval_f1s[-1]:.4f}")
print(f"Final Validation Loss: {metrics_callback.eval_losses[-1]:.4f}")

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================

print("\n" + "="*80)
print("EVALUATING ON TEST SET")
print("="*80)

test_results = trainer.evaluate(tokenized_test)

print(f"\n{'='*80}")
print("TEST SET RESULTS")
print("="*80)
print(f"Test F1 Score:   {test_results['eval_f1']:.4f}")
print(f"Test Precision:  {test_results['eval_precision']:.4f}")
print(f"Test Recall:     {test_results['eval_recall']:.4f}")
print(f"Test Loss:       {test_results['eval_loss']:.4f}")

# ============================================================================
# SAVE LORA ADAPTERS
# ============================================================================

print("\n" + "="*80)
print("SAVING LORA ADAPTERS")
print("="*80)

ADAPTER_PATH = "./roberta-lora-fewnerd-adapters"
model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)

# Calculate adapter size
import os
def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

adapter_size = get_folder_size(ADAPTER_PATH)
print(f"✓ LoRA adapters saved to '{ADAPTER_PATH}'")
print(f"✓ Adapter folder size: {adapter_size:.2f} MB")
print(f"\n📦 DEPLOYMENT READY:")
print(f"   - Use these {adapter_size:.2f} MB adapters for AWS deployment")
print(f"   - Load with: PeftModel.from_pretrained(base_model, '{ADAPTER_PATH}')")
print(f"   - Perfect for AWS Lambda (under 250 MB limit)")

# Also save a merged version for easier deployment
print("\n" + "-"*80)
print("SAVING MERGED MODEL (Optional - For Easier Inference)")
print("-"*80)

MERGED_PATH = "./roberta-lora-fewnerd-merged"
merged_model = model.merge_and_unload()
merged_model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)

merged_size = get_folder_size(MERGED_PATH)
print(f"✓ Merged model saved to '{MERGED_PATH}'")
print(f"✓ Merged model size: {merged_size:.2f} MB")
print(f"✓ This can be loaded like a standard transformers model")
print(f"\n💡 For AWS deployment: Use adapters ({adapter_size:.2f} MB) to save space!")

# ============================================================================
# INFERENCE FUNCTION (OPTIMIZED FOR FASTAPI)
# ============================================================================

class NERPredictor:
    """Production-ready NER predictor for FastAPI deployment"""

    def __init__(self, model_path):
        """
        Initialize predictor with model path

        Args:
            model_path: Path to saved model (merged or adapters)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text):
        """
        Predict named entities in text

        Args:
            text: Input text string

        Returns:
            List of dictionaries with entity, label, start, end positions
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

        # Convert tokens to words and align predictions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        entities = []
        current_entity = None

        for idx, (token, pred_idx) in enumerate(zip(tokens, predictions)):
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue

            label = self.model.config.id2label[pred_idx]

            # Handle IO tagging
            if label != 'O':
                # This is an entity token
                if current_entity is None or current_entity["label"] != label:
                    # Start new entity
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "entity": token.replace("Ġ", " ").strip(),
                        "label": label,
                        "start": idx,
                        "end": idx
                    }
                else:
                    # Continue current entity
                    current_entity["entity"] += token.replace("Ġ", " ")
                    current_entity["end"] = idx
            else:
                # O tag - no entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Add last entity if exists
        if current_entity:
            entities.append(current_entity)

        return entities

    def predict_batch(self, texts, batch_size=8):
        """
        Batch prediction for multiple texts

        Args:
            texts: List of text strings
            batch_size: Batch size for processing

        Returns:
            List of entity lists
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = [self.predict(text) for text in batch]
            results.extend(batch_results)
        return results

# ============================================================================
# INFERENCE EXAMPLES
# ============================================================================

print("\n" + "="*80)
print("INFERENCE EXAMPLES")
print("="*80)

# Initialize predictor
predictor = NERPredictor(MERGED_PATH)

test_sentences = [
    "Apple Inc. CEO Tim Cook announced the iPhone 15 at the Apple Park campus in Cupertino.",
    "The movie Inception directed by Christopher Nolan won several Academy Awards.",
    "Microsoft headquarters is located in Redmond, Washington near Lake Washington.",
    "Tesla Model S is an electric vehicle manufactured at the Fremont Factory in California.",
    "Elon Musk founded SpaceX and serves as CEO of Tesla Motors in Palo Alto.",
    "Arjun studies at SITS Pune and works on machine learning projects."
]

for i, text in enumerate(test_sentences, 1):
    print(f"\n{'-'*80}")
    print(f"Example {i}:")
    print(f"Text: {text}")
    print("\nDetected Entities:")

    entities = predictor.predict(text)
    if entities:
        for ent in entities:
            print(f"  • {ent['entity']:<45} [{ent['label']}]")
    else:
        print("  (No entities detected)")

!zip -r roberta-lora-fewnerd-merged.zip /content/roberta-lora-fewnerd-merged

