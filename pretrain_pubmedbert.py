# finetune_pubmedbert_from_pdfs.py
# End-to-end PubMedBERT fine-tuning from PDFs (MLM)

import os
import re
import pdfplumber
import nltk
import torch
from nltk.tokenize import sent_tokenize

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

############################################
# 0. Setup
############################################

nltk.download("punkt")

PDF_DIR = "pdfs"                      # folder containing PDFs
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
OUTPUT_DIR = "./pubmedbert_pdf_adapted"

MAX_LENGTH = 512
MAX_CHARS_PER_CHUNK = 1500            # ~512 tokens
TEST_SPLIT = 0.05

############################################
# 1. Extract text from PDFs
############################################

def extract_text_from_pdfs(pdf_dir):
    texts = []
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, fname)
            print(f"Extracting: {fname}")
            with pdfplumber.open(path) as pdf:
                pages = []
                for page in pdf.pages:
                    txt = page.extract_text()
                    if txt:
                        pages.append(txt)
                if pages:
                    texts.append("\n".join(pages))
    return texts

############################################
# 2. Clean + chunk text
############################################

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\s+', '', text)        # fix hyphenation
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def chunk_text(text, max_chars=1500):
    sentences = sent_tokenize(text)
    chunks, current = [], ""

    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    return chunks

############################################
# 3. Build dataset
############################################

def build_dataset(pdf_dir):
    raw_docs = extract_text_from_pdfs(pdf_dir)

    all_chunks = []
    for doc in raw_docs:
        doc = clean_text(doc)
        all_chunks.extend(chunk_text(doc))

    print(f"Total text chunks: {len(all_chunks)}")

    dataset = Dataset.from_dict({"text": all_chunks})
    dataset = dataset.train_test_split(test_size=TEST_SPLIT, seed=42)

    return dataset

############################################
# 4. Tokenization
############################################

def tokenize_dataset(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    return tokenized

############################################
# 5. Main training routine
############################################

def main():
    print("Building dataset...")
    dataset = build_dataset(PDF_DIR)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    print("Tokenizing dataset...")
    tokenized = tokenize_dataset(dataset, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=500,
        save_steps=1000,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")

############################################
# 6. Entry point
############################################

if __name__ == "__main__":
    main()
