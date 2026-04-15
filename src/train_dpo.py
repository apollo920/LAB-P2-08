import os
import torch
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

from prepare_dataset import prepare

BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   
OUTPUT_DIR   = Path(__file__).parent.parent / "outputs" / "dpo_model"
DATA_PATH    = Path(__file__).parent.parent / "data" / "hhh_dataset.jsonl"

BETA         = 0.1
MAX_LENGTH   = 512
BATCH_SIZE   = 1         
GRAD_ACCUM   = 8
EPOCHS       = 1
LR           = 5e-5



def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_actor_model(model_name: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_reference_model(model_name: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    for param in model.parameters():
        param.requires_grad = False
    return model



def get_dpo_config() -> DPOConfig:
    return DPOConfig(
        output_dir=str(OUTPUT_DIR),

        beta=BETA,

        optim="paged_adamw_32bit",

        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,

        max_length=MAX_LENGTH,
        max_prompt_length=MAX_LENGTH // 2,

        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="none",

        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),

        seed=42,
    )


def main():
    print("=" * 60)
    print("  Lab 08 — Alinhamento Humano com DPO")
    print("=" * 60)

    # 1. Dataset
    print("\n[1/4] Preparando dataset HHH...")
    dataset = prepare(DATA_PATH)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset  = split["test"]
    print(f"      Treino: {len(train_dataset)} | Avaliação: {len(eval_dataset)}")

    print(f"\n[2/4] Carregando tokenizer de '{BASE_MODEL}'...")
    tokenizer = load_tokenizer(BASE_MODEL)

    print(f"\n[3/4] Carregando modelos (ator + referência)...")
    actor_model     = load_actor_model(BASE_MODEL)
    reference_model = load_reference_model(BASE_MODEL)

    print("\n[4/4] Iniciando treinamento DPO...")
    dpo_config = get_dpo_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = DPOTrainer(
        model=actor_model,
        ref_model=reference_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("\n✅ Treinamento concluído!")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"   Modelo salvo em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()