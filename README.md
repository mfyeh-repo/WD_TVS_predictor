# WD_TVS_predictor
<pre> ``` 
主資料夾/
│
├── train.py, test.py  ← 程式碼放這裡
├── TVSdata/           ← 量測數據放這裡
│   ├── TVSdataD5094.xlsx
│   ├── TVSdataD9414.xlsx
│   └── ...
├── TVSmodel/          ← 產生的模型數據放這裡
│   ├── TVSmodel_D5094.h5,scaler_X_D5094.pkl,scaler_y_D5094.pkl
│   ├── TVSmodel_D9414.h5,scaler_X_D9414.pkl,scaler_y_D9414.pkl
│   └── ...
 ``` </pre>

TVS_MLPpredictor_train.ipynb MLP預測模型的訓練過程

1. 載入數據(資料)
2. 數據前處理
3. 建立模型(Sequential模型,compile)
4. 訓練模型(fit)
5. 儲存訓練結果TVSmodel.h5 (模型架構+權重)

TVS_MLPpredictor_test.ipynb 使用以訓練MLP模型來進行預測

1. 載入已訓練的 TVSmodel.h5
2. 載入標準化器 scaler_X.pkl 與 scaler_y.pkl
3. 模擬命令列式的單筆預測輸入流程，直到輸入 X/x 中斷
