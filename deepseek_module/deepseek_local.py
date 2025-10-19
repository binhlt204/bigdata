# # -*- coding: utf-8 -*-
# import os
# import pandas as pd
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # ===========================
# # ⚙️  Cấu hình chung
# # ===========================
# DEFAULT_SAVE_DIR = os.path.join('deepseek_module', 'text_prompt')
# DEFAULT_FILE = 'UBC-OCEAN_two_scale_text_prompt.csv'

# CLASS_NAMES = [
#     "clear-cell ovarian carcinoma",
#     "high-grade serous carcinoma",
#     "low-grade serous carcinoma",
#     "endometrioid carcinoma",
#     "mucinous carcinoma"
# ]

# PROMPT_TEMPLATE_LOW = (
#     "Write one concise pathology-style English description for a whole slide image "
#     "of {name} at low resolution. "
#     "Use only one sentence, starting with 'A whole slide image of {name} at low resolution'. "
#     "Do not add any introduction or extra commentary."
# )

# PROMPT_TEMPLATE_HIGH = (
#     "Write one concise pathology-style English description for a whole slide image "
#     "of {name} at high resolution. "
#     "Use only one sentence, starting with 'A whole slide image of {name} at high resolution'. "
#     "Do not add any introduction or extra commentary."
# )

# # ===========================
# # ⚙️  Gọi model DeepSeek thực tế
# # ===========================
# def generate_prompt_descriptions(model_name="deepseek-ai/deepseek-coder-1.3b-base"):
#     print(f"🚀 Loading DeepSeek model: {model_name}")
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
#     model.to(device)

#     texts = []
#     with torch.no_grad():
#         for name in CLASS_NAMES:
#             # low res
#             prompt_low = PROMPT_TEMPLATE_LOW.format(name=name)
#             input_ids = tokenizer.encode(prompt_low, return_tensors="pt").to(device)
#             output = model.generate(input_ids, max_new_tokens=80, do_sample=True, temperature=0.8)
#             text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
#             text = text.split("\n")[-1]  # lấy dòng cuối
#             texts.append(text)
#             print("🧩", text)

#         for name in CLASS_NAMES:
#             # high res
#             prompt_high = PROMPT_TEMPLATE_HIGH.format(name=name)
#             input_ids = tokenizer.encode(prompt_high, return_tensors="pt").to(device)
#             output = model.generate(input_ids, max_new_tokens=80, do_sample=True, temperature=0.8)
#             text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
#             text = text.split("\n")[-1]
#             texts.append(text)
#             print("🧠", text)

#     return texts


# # ===========================
# # 💾 Lưu CSV
# # ===========================
# def ensure_prompts_csv(csv_path=None, model_name="deepseek-ai/deepseek-coder-1.3b-base"):
#     if csv_path is None:
#         os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
#         csv_path = os.path.join(DEFAULT_SAVE_DIR, DEFAULT_FILE)

#     if os.path.exists(csv_path):
#         print(f"✅ Found existing prompt file: {csv_path}")
#         return csv_path

#     print("⚙️ Generating DeepSeek prompts...")
#     os.makedirs(os.path.dirname(csv_path), exist_ok=True)
#     lines = generate_prompt_descriptions(model_name=model_name)
#     pd.Series(lines).to_csv(csv_path, index=False, header=False)
#     print(f"✅ Saved DeepSeek prompt file: {csv_path}")
#     return csv_path


# # ===========================
# # 📂 Copy sang thư mục kết quả
# # ===========================
# def copy_prompt_to_results(csv_path, results_dir):
#     try:
#         import shutil
#         os.makedirs(results_dir, exist_ok=True)
#         dest = os.path.join(results_dir, "prompt_used.csv")
#         shutil.copy(csv_path, dest)
#         print(f"📁 Copied prompt CSV to {dest}")
#     except Exception as e:
#         print(f"[WARN] Could not copy prompt CSV: {e}")


# # ===========================
# # 🔄 Load danh sách prompt
# # ===========================
# def load_two_scale_prompts(csv_path):
#     if not os.path.exists(csv_path):
#         csv_path = ensure_prompts_csv(csv_path)
#     df = pd.read_csv(csv_path, header=None)
#     return df[0].tolist()



# # -*- coding: utf-8 -*-
# import os
# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # ===========================
# # ⚙️ Cấu hình chung
# # ===========================
# DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "text_prompt")
# DEFAULT_FILE = "UBC-OCEAN_two_scale_text_prompt.csv"

# CLASS_NAMES = [
#     "clear-cell ovarian carcinoma",
#     "high-grade serous carcinoma",
#     "low-grade serous carcinoma",
#     "endometrioid carcinoma",
#     "mucinous carcinoma"
# ]

# MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"  # model local (nhẹ, không cần API)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# def main():
#     print("🚀 Loading DeepSeek model...")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
#     model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     trust_remote_code=True,
#     use_safetensors=True
#     ).to(DEVICE)

#     print("✅ Model loaded successfully!")

#     prompts = []
#     for name in CLASS_NAMES:
#         # prompt_text = (
#         #     f"Describe the essential visual characteristics of {name} as seen in a high-resolution whole slide image."
#         # )

#         prompt_text = f"""
# Chỉ mô tả các đặc điểm hình ảnh thiết yếu của {name} khi nhìn thấy trong hình ảnh toàn bộ slide có độ phân giải cao. Lưu ý: không viết bài luận hay báo cáo, chỉ đưa ra thông tin các loại bệnh cho tôi thôi
# Ví dụ: 
# "Hình ảnh toàn bộ của ung thư nội mạc tử cung ở độ phân giải cao cho thấy các tuyến giáp nhau được lót bởi các tế bào hình trụ có nhân phân tầng. Có thể có sự biệt hóa dạng vảy. Sự dị sản nhân thay đổi tùy theo cấp độ khối u."
# Hãy lọc và chỉ in các ví dụ tương ứng với các bệnh ung thư

# """

#         inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
#         outputs = model.generate(**inputs, max_length=500, do_sample=True, temperature=0.7)
#         gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         prompts.append(gen_text)
#         print(f"🧠 Generated description for {name}")

#     # Lưu file CSV
#     os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
#     output_path = os.path.join(DEFAULT_SAVE_DIR, DEFAULT_FILE)
#     pd.DataFrame(prompts, columns=["text"]).to_csv(output_path, index=False, header=False)

#     print(f"✅ Saved to: {output_path}")
#     print(f"📄 Total {len(prompts)} pathology text prompts generated.")


# if __name__ == "__main__":
#     main()





# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===========================
# ⚙️ Cấu hình chung
# ===========================
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), "text_prompt")
DEFAULT_FILE = "UBC_OCEAN_pathology_descriptions.csv"

CLASS_NAMES = [
    "clear-cell ovarian carcinoma",
    "high-grade serous carcinoma",
    "low-grade serous carcinoma",
    "endometrioid carcinoma",
    "mucinous carcinoma"
]

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================
# ✏️ Prompt chuyên biệt
# ===========================
def build_prompt(name):
    return f"""
You are a pathology expert.
Describe only the essential microscopic and visual characteristics of {name}
as seen in a high-resolution whole-slide image.
Write 2–3 concise factual sentences describing tissue architecture, cellular arrangement, and any notable features.
Do not include apologies, explanations, markdown, or links.

### Response:
"""


# ===========================
# 🧹 Hàm làm sạch đầu ra
# ===========================
def clean_description(text: str) -> str:
    # Cắt phần dư nếu model lặp lại prompt
    if "### Response:" in text:
        text = text.split("### Response:")[-1]
    # Xóa các chỉ thị không mong muốn
    text = re.sub(r"Do NOT include.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Do not include.*", "", text, flags=re.IGNORECASE)
    # Loại bỏ HTML hoặc ký tự thừa
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[{}\[\]();#=/<>|`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===========================
# 🧠 Sinh mô tả
# ===========================
def generate_description(tokenizer, model, name):
    prompt = build_prompt(name)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.4,
        do_sample=False,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    desc = clean_description(raw)

    # Giữ lại đoạn bắt đầu bằng “A whole slide image...”
    match = re.findall(r"A whole slide image of .*?(?:\.|\!|\?)", desc)
    if match:
        desc = " ".join(match[:3])
    return desc.strip()


# ===========================
# 🏁 Chạy chính
# ===========================
def main():
    print("🚀 Loading model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        use_safetensors=True,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    print("✅ Model loaded successfully!\n")

    results = []
    for name in CLASS_NAMES:
        print(f"🧬 Generating description for: {name} ...")
        desc = generate_description(tokenizer, model, name)
        print(f"👉 {desc}\n")
        results.append({"Class": name, "Description": desc})

    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(DEFAULT_SAVE_DIR, DEFAULT_FILE)
    pd.DataFrame(results).to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Saved clean results to: {save_path}")
    print(f"📄 Generated {len(results)} class descriptions.")


if __name__ == "__main__":
    main()


# -*- coding: utf-8 -*-
import os, re, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"  # must be the instruct version
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "clear-cell ovarian carcinoma",
    "high-grade serous carcinoma",
    "low-grade serous carcinoma",
    "endometrioid carcinoma",
    "mucinous carcinoma"
]

def build_prompt(name):
    return f"""
You are a **medical pathology expert** specializing in digital histopathology.

Your task:
Describe **only** the essential microscopic and visual features of **{name}** 
as seen in a **high-resolution whole slide image (WSI)**.

Write **2 to 3 short, factual sentences**.
Each sentence must begin with: 
"A whole slide image of {name} at high resolution..."

Focus on:
- tissue architecture
- cellular morphology
- nuclear atypia
- diagnostic features specific to {name}

⚠️ Output rules:
- Do NOT include apologies, self-references, markdown, or disclaimers.
- Do NOT explain AI limitations or training data.
- Return ONLY the factual description sentences.

### Response:
"""



def clean_output(txt):
    txt = txt.split("### Response:")[-1]
    txt = re.sub(r"(?i)(i'?m sorry|do not include|developed by|deepseek|AI model|Deepset|technolog(y|ies)).*", "", txt)
    txt = re.sub(r"<.*?>", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    matches = re.findall(r"A whole slide image of .*?(?:\.|\!|\?)", txt)
    if matches:
        txt = " ".join(matches[:3])
    return txt.strip()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_safetensors=True
    ).to(DEVICE)

    results = []
    for name in CLASS_NAMES:
        prompt = build_prompt(name)
        inputs = tok(prompt, return_tensors="pt").to(DEVICE)
        out = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.3,
            top_p=0.9,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id
        )
        txt = tok.decode(out[0], skip_special_tokens=True)
        desc = clean_output(txt)
        # print(f"{name} → {desc}\n")
        results.append({"Description": desc})

    os.makedirs("text_prompt", exist_ok=True)
    pd.DataFrame(results).to_csv("text_prompt/UBC_OCEAN_fixed.csv",
                                 index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
