import torch
import os
os.environ["PYTHONUTF8"] = "1"
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ==========================================
# 1. Konfiguracja pod 6GB VRAM (Magia QLoRA)
# ==========================================
# Ściskamy główny model do 4-bitów, żeby nie spalić Twojej karty
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16 # Używamy float16, bo bfloat16 słabo działa na 1660 Ti
)

# ==========================================
# 2. Wczytanie modelu i słownika (Tokenizer)
# ==========================================
print("Ładowanie modelu (to chwilę potrwa)...")
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Ustawiamy padding, żeby model radził sobie z tekstami o różnej długości
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", # Automatycznie wrzuca model na Twoją kartę GPU
    dtype="float32"
)
model.config.torch_dtype = torch.float16

# Przygotowujemy skompresowany model do treningu
model = prepare_model_for_kbit_training(model)

# ==========================================
# 3. Konfiguracja "Plastra Wiedzy" (LoRA)
# ==========================================
# To są te małe wagi, które będziemy fizycznie trenować
lora_config = LoraConfig(
    r=8, # Rozmiar "plastra" - 8 to bezpieczny standard
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # Trenujemy tylko moduły "uwagi" modelu
    bias="none",
    task_type="CAUSAL_LM"
)


# ==========================================
# 4. Wczytanie Twojego pliku JSONL
# ==========================================
dataset = load_dataset("json", data_files="dane_treningowe.jsonl", split="train")

# ==========================================
# 5. Ustawienia Treningu (Zabezpieczenia dla 1660 Ti)
# ==========================================
training_args = SFTConfig(
    output_dir="./qwen_moj_silnik",
    per_device_train_batch_size=1,      # Ładujemy tylko 1 przykład na raz
    gradient_accumulation_steps=4,      # Symulujemy większy batch size zbierając dane co 4 kroki
    max_length=512, # Obcinamy za długie teksty
    optim="paged_adamw_32bit",          # Optymalizator oszczędzający pamięć
    logging_steps=5,                    # Ilość kroków
    learning_rate=2e-4,                 # Szybkość nauki
    max_steps=100,                      # Ustalamy na 100 dla szybkiego testu
    fp16=False,                          # Dodatkowa oszczędność pamięci na kartach GTX
    bf16=False,
    gradient_checkpointing=True         # Zwalnia pamięć kosztem minimalnie dłuższego czasu obliczeń
)

# ==========================================
# 6. Start Treningu!
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args,
)

print("🔥 Odpalamy piece! Rozpoczynamy fine-tuning...")
trainer.train()

# Na sam koniec zapisujemy gotowy, wyuczony model na dysk!
trainer.model.save_pretrained("./qwen_gotowy")
tokenizer.save_pretrained("./qwen_gotowy")
print("✅ Sukces! Twój sfinetunowany model czeka w folderze './qwen_gotowy'")