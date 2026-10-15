"""Fine-tuning LoRA/QLoRA con TRL. Richiede GPU NVIDIA (pip install -r finetune/requirements.txt).

Il fine-tuning NON si fa sul GGUF di LM Studio: serve il modello originale (Hugging Face).
Per 27B servono 24+ GB di VRAM anche in 4-bit; per cominciare usa una variante piccola
(es. 4B-8B). Dopo il training, converti in GGUF (llama.cpp) per caricarlo in LM Studio.
"""
import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="id Hugging Face del modello base, es. una variante Qwen piccola")
    ap.add_argument("--data", default="finetune/train.jsonl")
    ap.add_argument("--out", default="finetune/out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--qlora", action="store_true", help="carica il modello a 4 bit (meno VRAM)")
    a = ap.parse_args()

    model_kwargs = {}
    if a.qlora:
        import torch
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    ds = load_dataset("json", data_files=a.data, split="train")
    trainer = SFTTrainer(
        model=a.base,
        train_dataset=ds,
        args=SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
                       per_device_train_batch_size=1, gradient_accumulation_steps=8,
                       logging_steps=5, save_strategy="epoch", bf16=True,
                       model_init_kwargs=model_kwargs or None),
        peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                               target_modules="all-linear"),
    )
    trainer.train()
    trainer.save_model(a.out)
    print(f"Adattatore LoRA salvato in {a.out}")


if __name__ == "__main__":
    main()
