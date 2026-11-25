"""Optional tiny LoRA fine-tuning demo. Install requirements-training.txt first."""
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--output_dir", default="artifacts/lora-support")
    a = p.parse_args()
    examples = [
        "Customer: I cannot sign in. Agent: Check your email and use password reset.",
        "Customer: Track my order. Agent: Open Orders and select the order to view tracking.",
        "Customer: I forgot my password. Agent: Select Forgot password and follow email instructions.",
        "Customer: I need help with a purchase. Agent: I can create a support ticket."
    ]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    ds = Dataset.from_dict({"text":examples})
    ds = ds.map(lambda b: tok(b["text"], truncation=True, padding="max_length", max_length=128),
                batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(a.model)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                     lora_dropout=0.05, target_modules=["c_attn"], fan_in_fan_out=True)
    model = get_peft_model(model, cfg)
    args = TrainingArguments(output_dir=a.output_dir, num_train_epochs=3,
        per_device_train_batch_size=2, learning_rate=2e-4, logging_steps=1,
        save_strategy="epoch", report_to="none", fp16=False)
    trainer = Trainer(model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()
    model.save_pretrained(a.output_dir); tok.save_pretrained(a.output_dir)
    print("Adapter saved to", a.output_dir)
if __name__ == "__main__": main()
