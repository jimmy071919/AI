"""
要注意使用，這個會將整個模型下載到
C:\Users\jimmy\.cache\huggingface\hub\之中 如果要處理要去刪掉這邊的東西

這是取用huggung face 的模型by transformers的方法與格式 可以參考
"""



# ===== 導入必要的庫 =====
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch

"""
載入 Transformers 提供的通用接口

AutoTokenizer: 自動處理分詞 - 將文字轉換成模型能理解的數字序列（詞元/token）
AutoModelForCausalLM: 自動載入符合「自回歸語言模型」架構的模型（如GPT類模型）
TextStreamer: 可以即時顯示模型輸出（Streaming 推理），讓使用者不用等待完整生成就能看到結果
"""


# ===== 載入模型和分詞器 =====
# 這個字串是 Hugging Face 的模型 repository 名稱。你也可以下載到本地後改成 model_id = "./MiniMax-M1-40k"
model_id = "MiniMaxAI/MiniMax-M1-40k"

# 載入分詞器 - 負責將文字轉換為模型可理解的數字序列
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 載入語言模型
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # 使用半精度浮點數（FP16）以節省GPU顯存並加速計算
    device_map="auto",        # 自動偵測並使用可用的GPU，若有多張卡會自動分配
    trust_remote_code=True,     # 允許使用模型作者提供的自訂程式碼
)

"""
關於 from_pretrained 方法的詳細說明：

from_pretrained 是 Hugging Face Transformers 庫中的核心方法，用於載入預訓練模型和相關組件。

主要功能：
1. 載入預訓練模型：不需要從頭訓練，直接使用已經在大量數據上訓練好的模型
2. 自動下載與快取：
   - 當提供 Hugging Face Hub 上的模型 ID 時（如「MiniMaxAI/MiniMax-M1-40k」），自動從網路下載
   - 下載的模型會被快取在本地（通常在 ~/.cache/huggingface/hub 或 Windows 上的 C:\Users\USERNAME\.cache\huggingface\hub）
   - 下次使用相同模型時，會直接從快取中載入，不需要再次下載
3. 本地模型載入：也可以載入本地保存的模型，只需提供本地路徑（如「./my_model」）
4. 自動配置：根據模型的配置文件自動設置模型架構和參數

常用參數說明：
- torch_dtype：設置模型的數據類型，如 torch.float16（半精度）可節省顯存
- device_map：控制模型在哪些設備上運行，「auto」表示自動選擇最佳設備
- trust_remote_code：是否信任並執行模型附帶的自定義代碼
- cache_dir：指定模型快取的位置
- force_download：強制重新下載模型，即使本地已有快取
- local_files_only：只使用本地文件，不嘗試下載
- revision：指定要使用的模型版本或分支
"""

"""
詳細說明:

1. tokenizer (分詞器):
   - AutoTokenizer 會根據模型類型自動選擇合適的 tokenizer 類別（例如 LLaMATokenizer）
   - 分詞器將文字切分成「詞元」(tokens)，並轉換為對應的數字ID

2. model (模型):
   - AutoModelForCausalLM 是自回歸語言模型的通用接口，適用 GPT 類架構
   - 自回歸語言模型指的是根據前面的文字來預測下一個詞的模型
   - torch_dtype=torch.float16: 使用16位元浮點數表示模型參數，可節省約一半顯存
   - device_map="auto": 會自動將模型載入到可用 GPU 上（如有多張卡，也能自動分片）
   - trust_remote_code=True: 允許執行模型作者提供的自訂程式碼，某些模型需要這個選項
"""


# ===== 準備輸入並生成回應 =====
# 設定提示詞
prompt = "你是誰"

# 將文字轉換為模型可處理的格式
# return_tensors="pt" 表示返回PyTorch張量格式
# .to(model.device) 將輸入數據移至與模型相同的設備上（CPU或GPU）
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 生成回應
with torch.no_grad():  # 不計算梯度，節省記憶體並加速推理
    outputs = model.generate(
        **inputs,                         # 展開輸入參數
        max_new_tokens=100,               # 最多生成100個新詞元
        streamer=TextStreamer(tokenizer), # 使用串流輸出，即時顯示生成結果
        skip_special_tokens=True          # 跳過特殊標記如<s>、</s>等，只顯示實際文字內容
    )

# 印出完整結果
# 如果使用了streamer，這行可能是多餘的，因為TextStreamer已經會顯示生成的文字
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
