#!/usr/bin/env python3
"""
M4 RoBERTa Fine-Tuning Script
English-only AI text detection. See README.md for full guide.
"""

import argparse
import json
import os
import re
import random
import numpy as np
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from scipy.special import softmax
from typing import List, Dict, Tuple
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train M4 detector with chunking")

    # Model params
    parser.add_argument("--model", type=str, default="roberta-base", help="Model name or path")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--freeze_base", action="store_true", help="Freeze base layers, train only last N")
    parser.add_argument("--freeze_layers", type=int, default=4, help="Number of layers to unfreeze from top")

    # Data params
    parser.add_argument("--val_generator", type=str, default="gpt2-neo", help="Generator to hold out for validation")
    parser.add_argument("--train_domains", type=str, nargs="+", default=["arxiv", "wikipedia", "reddit"], help="Domains to train on")
    parser.add_argument("--train_generators", type=str, nargs="+", default=["chatGPT", "davinci", "cohere"], help="Generators to train on")
    parser.add_argument("--data_dir", type=str, default="M4/data", help="Path to M4 data directory")

    # Preprocessing params
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length (including special tokens)")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Token overlap between chunks")
    parser.add_argument("--normalize_digits", action="store_true", default=True, help="Replace digits with '1'")

    # Training params
    parser.add_argument("--output_dir", type=str, default="runs/m4-roberta", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--run_name", type=str, default="m4-roberta-base", help="Run name for W&B")

    # Debug params
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples (for testing)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit validation samples (for testing)")

    return parser.parse_args()


def set_all_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    print(f"✅ Random seed set to {seed}")


def normalize_digits(text: str) -> str:
    """Replace all digits with '1' while keeping format"""
    return re.sub(r'\d', '1', text)


def chunk_text_with_overlap(text: str, tokenizer, max_length: int = 512, overlap: int = 50) -> List[str]:
    """
    Chunk text into overlapping segments that fit within max_length tokens.

    Args:
        text: Input text to chunk
        tokenizer: Hugging Face tokenizer
        max_length: Maximum tokens per chunk (including special tokens)
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)

    # Account for special tokens [CLS] and [SEP]
    max_tokens_per_chunk = max_length - 2

    if len(tokens) <= max_tokens_per_chunk:
        # Text fits in one chunk
        return [text]

    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        # Get chunk tokens
        end_idx = min(start_idx + max_tokens_per_chunk, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]

        # Decode chunk back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)

        # Move to next chunk with overlap
        if end_idx >= len(tokens):
            break
        start_idx += max_tokens_per_chunk - overlap

    return chunks


def load_m4_data(data_dir, domains, generators, normalize_digits_flag=True, max_samples=None):
    """
    Load M4 JSONL files and create dataset.

    Handles multiple M4 format variations:
    1. Standard: human_text (str), machine_text (str) - 28 files
    2. Reddit/bloomz: text (str), machine_text (str) - 3 files
    3. PeerRead: human_text (list), machine_text (list) - 5 files
    4. Broken: missing fields - skipped

    Creates TWO training examples from each line:
    - One with human text, label=0 (human)
    - One with machine text, label=1 (machine/AI)
    """
    data = []

    print(f"Loading data: domains={domains}, generators={generators}")

    for domain in domains:
        for generator in generators:
            # Try different naming patterns
            patterns = [
                f"{domain}_{generator}.jsonl",
                f"{domain}_{generator.lower()}.jsonl",
                f"{domain}_{generator.upper()}.jsonl",
                f"{domain}_{generator.capitalize()}.jsonl",
            ]

            # Special case: wikihow uses dolly2 instead of dolly
            if generator.lower() == 'dolly' and domain == 'wikihow':
                patterns.insert(0, f"{domain}_dolly2.jsonl")

            filepath = None
            for pattern in patterns:
                potential_path = os.path.join(data_dir, pattern)
                if os.path.exists(potential_path):
                    filepath = potential_path
                    break

            if filepath is None:
                print(f"⚠️  Warning: Could not find data for {domain}_{generator}")
                continue

            print(f"  Loading {filepath}")
            line_count = 0
            skipped_count = 0

            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())

                        # Extract human and machine text with format handling
                        human_text = None
                        machine_text = None

                        # Format 1: Standard (human_text, machine_text)
                        if 'human_text' in item and 'machine_text' in item:
                            human_text = item['human_text']
                            machine_text = item['machine_text']

                        # Format 2: Reddit/bloomz (text, machine_text)
                        elif 'text' in item and 'machine_text' in item:
                            human_text = item['text']
                            machine_text = item['machine_text']

                        # Format 3: Skip if fields missing
                        else:
                            skipped_count += 1
                            continue

                        # Handle list format (PeerRead files)
                        if isinstance(human_text, list):
                            # Flatten list: join all reviews with newline
                            human_text = '\n\n'.join(str(x) for x in human_text if x)
                        if isinstance(machine_text, list):
                            machine_text = '\n\n'.join(str(x) for x in machine_text if x)

                        # Skip if still empty
                        if not human_text or not machine_text:
                            skipped_count += 1
                            continue

                        # Normalize digits if requested
                        if normalize_digits_flag:
                            human_text = normalize_digits(human_text)
                            machine_text = normalize_digits(machine_text)

                        # Create example 1: Human text with label=0
                        data.append({
                            'text': human_text,
                            'label': 0,  # 0 = human
                            'generator': 'human',
                            'domain': domain,
                            'id': f"{item.get('source_ID', item.get('source_id', line_count))}_human"
                        })

                        # Create example 2: Machine text with label=1
                        data.append({
                            'text': machine_text,
                            'label': 1,  # 1 = machine/AI
                            'generator': generator,
                            'domain': domain,
                            'id': f"{item.get('source_ID', item.get('source_id', line_count))}_machine"
                        })

                        line_count += 1

                    except json.JSONDecodeError:
                        skipped_count += 1
                        continue

            if line_count > 0:
                print(f"    → {line_count} lines → {line_count * 2} examples")
            if skipped_count > 0:
                print(f"    ⚠️  Skipped {skipped_count} lines (missing/invalid format)")

    if max_samples:
        data = random.sample(data, min(max_samples, len(data)))

    print(f"✅ Loaded {len(data)} samples total")
    return Dataset.from_list(data)


def create_chunked_dataset(dataset, tokenizer, max_length=512, overlap=50):
    """
    Create chunked version of dataset for training.
    Each long text becomes multiple training examples (chunks).
    """
    chunked_data = []

    print("Creating chunks for training...")
    for item in tqdm(dataset):
        chunks = chunk_text_with_overlap(
            item['text'],
            tokenizer,
            max_length=max_length,
            overlap=overlap
        )

        # Each chunk becomes a training example with same label
        for chunk_idx, chunk_text in enumerate(chunks):
            chunked_data.append({
                'text': chunk_text,
                'label': item['label'],
                'generator': item['generator'],
                'domain': item['domain'],
                'original_id': item['id'],
                'chunk_id': chunk_idx,
                'num_chunks': len(chunks)
            })

    print(f"✅ Created {len(chunked_data)} chunks from {len(dataset)} texts")
    return Dataset.from_list(chunked_data)


def compute_metrics(pred):
    """Compute metrics including per-generator F1"""
    logits = pred.predictions
    labels = pred.label_ids

    # Get predictions
    probs = softmax(logits, axis=1)[:, 1]
    preds = (probs > 0.5).astype(int)

    # Overall metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return metrics


def freeze_model_layers(model, num_layers_to_unfreeze=4):
    """
    Freeze all layers except the last N transformer layers + classification head.
    For RoBERTa, the layers are in model.roberta.encoder.layer
    """
    # Freeze embeddings
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False

    # Freeze all encoder layers first
    for param in model.roberta.encoder.parameters():
        param.requires_grad = False

    # Unfreeze last N layers
    total_layers = len(model.roberta.encoder.layer)
    layers_to_unfreeze = model.roberta.encoder.layer[-num_layers_to_unfreeze:]

    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    # Classification head always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"🔒 Froze base layers. Unfroze last {num_layers_to_unfreeze}/{total_layers} encoder layers")
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def main():
    args = parse_args()

    # Set seeds
    set_all_seeds(args.seed)

    # Configure W&B to NOT upload model checkpoints (saves space on free tier)
    if args.use_wandb:
        import wandb
        # Prevent uploading large checkpoint files
        os.environ["WANDB_DISABLE_CODE"] = "false"  # Keep code snapshot (small)
        os.environ["WANDB_LOG_MODEL"] = "false"     # Don't upload model checkpoints
        print("🔧 W&B configured: metrics only (no checkpoint upload)")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print(f"\n📦 Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load raw data
    print(f"\n📂 Loading data from {args.data_dir}")
    train_dataset_raw = load_m4_data(
        args.data_dir,
        args.train_domains,
        args.train_generators,
        normalize_digits_flag=args.normalize_digits,
        max_samples=args.max_train_samples
    )

    val_dataset_raw = load_m4_data(
        args.data_dir,
        args.train_domains,
        [args.val_generator],
        normalize_digits_flag=args.normalize_digits,
        max_samples=args.max_val_samples
    )

    print(f"\n📊 Raw data: Train={len(train_dataset_raw)} texts, Val={len(val_dataset_raw)} texts")

    # Create chunked datasets
    print(f"\n✂️  Chunking with max_length={args.max_length}, overlap={args.chunk_overlap}")
    train_dataset = create_chunked_dataset(
        train_dataset_raw,
        tokenizer,
        max_length=args.max_length,
        overlap=args.chunk_overlap
    )

    val_dataset = create_chunked_dataset(
        val_dataset_raw,
        tokenizer,
        max_length=args.max_length,
        overlap=args.chunk_overlap
    )

    print(f"📊 Chunked data: Train={len(train_dataset)} chunks, Val={len(val_dataset)} chunks")

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=args.max_length
        )

    print("\n🔤 Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    # Load model with custom dropout
    print(f"\n🤖 Loading model: {args.model}")
    config = AutoConfig.from_pretrained(args.model)
    config.hidden_dropout_prob = args.dropout
    config.attention_probs_dropout_prob = args.dropout
    config.num_labels = 2

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config
    )

    # Freeze layers if requested
    if args.freeze_base:
        freeze_model_layers(model, num_layers_to_unfreeze=args.freeze_layers)
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"🔓 Full fine-tuning. Total params: {total:,}")

    # Training arguments
    print(f"\n⚙️  Training config:")
    print(f"   Learning rate: {args.lr}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   FP16: {args.fp16}, BF16: {args.bf16}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="wandb" if args.use_wandb else "none",
        run_name=args.run_name if args.use_wandb else None,
        seed=args.seed,
        data_seed=args.seed,
        label_smoothing_factor=0.0,  # No label smoothing per guide
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n🚀 Starting training...")
    train_result = trainer.train()

    # Save model
    print(f"\n💾 Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # Final evaluation
    print("\n📊 Final evaluation on validation set:")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save training config
    config_dict = {
        "model": args.model,
        "lr": args.lr,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_generator": args.val_generator,
        "train_domains": args.train_domains,
        "train_generators": args.train_generators,
        "seed": args.seed,
        "max_length": args.max_length,
        "chunk_overlap": args.chunk_overlap,
        "normalize_digits": args.normalize_digits,
        "freeze_base": args.freeze_base,
        "freeze_layers": args.freeze_layers if args.freeze_base else None,
        "final_metrics": {k: float(v) for k, v in metrics.items()},
    }

    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"📁 Model saved to: {args.output_dir}")
    print(f"🎯 Best F1: {metrics.get('eval_f1', 0):.4f}")
    print(f"📝 Config saved to: {config_path}")


if __name__ == "__main__":
    main()
