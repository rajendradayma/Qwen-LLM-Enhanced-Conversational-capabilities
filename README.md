# Qwen-LLM-Enhanced-Conversational-capabilities
Qwen2.5 Fine-Tuning with LoRA

This project demonstrates how to fine-tune the [Qwen2.5 language model](https://huggingface.co/Qwen) using **LoRA** (Low-Rank Adaptation) with PEFT, saving both the **LoRA adapter** and the **full model**, and evaluating responses with a human-readable prompt.

---

## 📦 Contents

- `qwen2.5-lora-adapter/`: Contains only the LoRA adapter weights.
- `qwen2.5-lora-full/`: Contains the full base model with LoRA applied.
- `finetune_lora.ipynb`: Colab notebook used for training.
- `train_loss_plot.png`: Visualization of training loss over steps.

---

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install torch transformers peft datasets accelerate bitsandbytes evaluate matplotlib
```
🧠 Training Summary
Model: Qwen2.5

Fine-Tuning Method: LoRA with PEFT

Device: CUDA / GPU

Loss: Reduced from ~4.7 to ~2.3 over 10 epochs

📉 Training Loss Plot

💾 Saving Models
✅ Save LoRA Adapter
python
Copy
Edit
peft_path = "./qwen2.5-lora-adapter"
model.save_pretrained(peft_path)
tokenizer.save_pretrained(peft_path)
✅ Save Full Model with Adapter
python
Copy
Edit
full_model_path = "./qwen2.5-lora-full"
model.base_model.save_pretrained(full_model_path)
tokenizer.save_pretrained(full_model_path)
🧪 Human Evaluation Prompt
You can test how well your model responds using a human-readable prompt:

python
Copy
Edit
example = "<|im_start|>user\nHow do I improve my Python skills?<|im_end|>\n<|im_start|>assistant\n"

input_ids = tokenizer(example, return_tensors="pt").to(model.device)

output = model.generate(
    input_ids.input_ids,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id
)

response = tokenizer.decode(output[0], skip_special_tokens=False)
print(response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0])
Rate the response for:

✅ Relevance

✅ Coherence

✅ Helpfulness
On a scale of 1–5 stars ⭐️

📥 Downloading Models from Google Colab
python
Copy
Edit
import shutil, zipfile
from google.colab import files

# Zip
shutil.make_archive('qwen2.5-lora-adapter', 'zip', './qwen2.5-lora-adapter')
shutil.make_archive('qwen2.5-lora-full', 'zip', './qwen2.5-lora-full')

# Download
files.download('qwen2.5-lora-adapter.zip')
files.download('qwen2.5-lora-full.zip')
🧩 Model Loading (for inference)

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./qwen2.5-lora-adapter")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./qwen2.5-lora-adapter")
📜 License
MIT License

✨ Acknowledgments
Transformers by HuggingFace

Qwen Model

PEFT Library

