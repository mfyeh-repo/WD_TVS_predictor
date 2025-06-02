#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 匯入必要套件
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os


# In[2]:


# 設定資料夾路徑
data_path = "TVSdata"
# 設定模型資料夾路徑
model_path = "TVSmodel"


# In[7]:


# 輸入元件編號（例如 D5094 或 D9414）
PartNumber = input("輸入元件編號（例如 D5094 或 D9414）：")

# 組合完整檔案路徑
modelpath = os.path.join(model_path,"TVSmodel_"+PartNumber+".h5")
#print(modelpath)


# In[8]:


# 載入模型與標準化器
try:
    model = load_model(modelpath)
    print("✅ 模型載入成功")
except Exception as e:
    print("模型載入失敗，錯誤訊息：", e)

# 檢查 scaler_X 檔案是否存在，並嘗試載入
file_path = os.path.join(model_path,"scaler_X_"+PartNumber+".pkl")
if os.path.isfile(file_path):
    try:
        scaler_X = joblib.load(file_path)
        print("✅ X_標準化器載入成功")
    except Exception as e:
        print("X_標準化器載入失敗，錯誤訊息：", e)
else:
    print("X_標準化器檔案不存在")

# 檢查 scaler_y 檔案是否存在，並嘗試載入
file_path = os.path.join(model_path,"scaler_y_"+PartNumber+".pkl")
if os.path.isfile(file_path):
    try:
        scaler_y = joblib.load(file_path)
        print("✅ y_標準化器載入成功")
    except Exception as e:
        print("y_標準化器載入失敗，錯誤訊息：", e)
else:
    print("y_標準化器檔案不存在")


# In[9]:


# 單筆預測互動迴圈（輸入 x 結束）
print("輸入 Vc_1 和 Ipp_1 進行預測，直到輸入 X 或 x 為止")
while True:
    vcc_input = input("Vc_1 (V): ")
    if vcc_input.strip().lower() == "x":
        print("⛔ 結束預測")
        break

    ipp_input = input("Ipp_1 (A): ")
    if ipp_input.strip().lower() == "x":
        print("⛔ 結束預測")
        break

    try:
        vcc1 = float(vcc_input)
        ipp1 = float(ipp_input)
        X_input = np.array([[vcc1, ipp1]])
        X_scaled = scaler_X.transform(X_input)
        y_scaled_pred = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_scaled_pred)

        print(f"✅ 預測 Vc_2: {y_pred[0][0]:.2f} V")
        print(f"✅ 預測 Ipp_2: {y_pred[0][1]:.2f} A\n")

    except Exception as e:
        print(f"⚠️ 輸入錯誤：{e}\n")


# In[ ]:




