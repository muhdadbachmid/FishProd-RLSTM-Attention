# Prediksi Produksi Ikan menggunakan LSTM dengan Residual Connection dan Multi-Head Attention

Repositori ini berisi implementasi model prediksi produksi ikan tahunan menggunakan arsitektur LSTM yang ditingkatkan dengan Residual Connection dan Multi-Head Attention. Model ini dikembangkan untuk memprediksi total produksi tahunan berdasarkan data historis produksi ikan dari tahun 2018-2022.

## 🔧 Instalasi

### Prasyarat
- Python 3.8 atau lebih baru
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

### Langkah Instalasi

1. Klon repositori ini:
```
git clone https://github.com/muhdadbachmid/FishProd-RLSTM-Attention.git
cd FishProd-RLSTM-Attention
```

2. Instal dependensi:
```python
pip install -r requirements.txt
```

## 📁 Struktur Proyek

```
FishProd-RLSTM-Attention/
├── data/
│   └── produksi2018_2022.csv
├── docs/
│   └── readme.md
│   └── workflow-architecture-diagram.mermaid
│   └── workflow-architecture-diagram.png
├── models/
│   └── 
├── notebooks/
│   └── model.ipynb
└── requirements.txt
```

## 🚀 Penggunaan

### Data Input
Data produksi ikan yang digunakan memiliki format sebagai berikut:
- Kolom-kolom fitur (berbagai parameter produksi)
- Kolom `jenis_ikan` (kategori)
- Kolom `Total_Produksi_Tahunan` (target yang diprediksi)

### Memuat dan Praproses Data

```python
import pandas as pd
from utils.data_preprocessing import preprocess_data

# Memuat data
df = pd.read_csv('data/produksi2018_2022.csv')

# Praproses data
X_train, X_test, y_train, y_test = preprocess_data(df)
```

### Melatih Model

```python
from models.lstm_model import build_lstm_model

# Membangun model
model = build_lstm_model(input_shape=(1, X_train.shape[2]))

# Melatih model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)
```

### Evaluasi Model

```python
from utils.evaluation import evaluate_model

# Evaluasi model
rmse, mae, huber_loss = evaluate_model(model, X_test, y_test)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"Huber Loss: {huber_loss:.4f}")
```

## 🏗️ Arsitektur Model

Model ini menggunakan arsitektur yang menggabungkan beberapa teknik deep learning:

1. **LSTM (Long Short-Term Memory)** - Untuk menangkap pola temporal dalam data
2. **Residual Connection** - Untuk mengatasi masalah vanishing gradient
3. **Multi-Head Attention** - Untuk meningkatkan fokus model pada bagian penting dari data

Arsitektur lengkap model:

```
Model: "functional_22"
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer_23      │ (None, 1, 16)     │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ lstm_6 (LSTM)       │ (None, 1, 64)     │     20,736 │ input_layer_23[0] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_69 (Dense)    │ (None, 1, 64)     │      1,088 │ input_layer_23[0] │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ add_3 (Add)         │ (None, 1, 64)     │          0 │ lstm_6[0][0],     │
│                     │                   │            │ dense_69[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ lstm_7 (LSTM)       │ (None, 1, 128)    │     98,816 │ add_3[0][0]       │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ multi_head_attenti… │ (None, 1, 128)    │    263,808 │ lstm_7[0][0],     │
│ (MultiHeadAttentio… │                   │            │ lstm_7[0][0]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_70 (Dense)    │ (None, 1, 64)     │      8,256 │ multi_head_atten… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_71 (Dense)    │ (None, 1, 32)     │      2,080 │ dense_70[0][0]    │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_72 (Dense)    │ (None, 1, 1)      │         33 │ dense_71[0][0]    │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘
 Total params: 394,817 (1.51 MB)
 Trainable params: 394,817 (1.51 MB)
 Non-trainable params: 0 (0.00 B)
```

## 📊 Alur Kerja

Model prediksi ikan ini mengikuti alur kerja berikut:

1. **Instalasi Dependensi**
   - Mengimpor pustaka yang diperlukan (TensorFlow, Pandas, NumPy, dll.)

2. **Analisis Data & Pra-pemrosesan**
   - Memuat data CSV dan memeriksa tipe data
   - Membersihkan data dengan menghapus nilai NaN
   - Memisahkan fitur (X) dan target (y)
   - Mengisi nilai kosong dengan rata-rata
   - Seleksi fitur menggunakan `SelectKBest`
   - Normalisasi data dengan `MinMaxScaler`
   - Membagi data menjadi train-test
   - Mengubah format data agar sesuai dengan LSTM

3. **Arsitektur Model**
   - Membuat model LSTM dengan residual connection dan MultiHeadAttention
   - Menyusun arsitektur model dengan lapisan `LSTM`, `Dense`, dan `MultiHeadAttention`
   - Menggunakan Adam optimizer dan loss function MSE

4. **Pelatihan Model**
   - Melatih model dengan dataset yang sudah diproses
   - Menggunakan 100 epochs dengan batch size 16

5. **Evaluasi Model**
   - Membandingkan hasil prediksi dengan data aktual menggunakan:
     - RMSE (Root Mean Squared Error)
     - MAE (Mean Absolute Error)
     - Huber Loss
   - Memvisualisasikan perbandingan prediksi vs aktual

6. **Validasi Silang**
   - Menggunakan K-Fold Cross-Validation (K=5) untuk mengevaluasi performa model

## 📈 Hasil

### Evaluasi

Hasil evaluasi model pada data pengujian:
- RMSE: 0.0212
- MAE: 0.0089
- Huber Loss: 0.0002

### Validasi Silang

Hasil K-Fold Cross-Validation (5 folds):
- RMSE: 0.0867 ± 0.0363
- MAE: 0.0415 ± 0.0091
- Huber Loss: 0.0044 ± 0.0030

## 🤝 Kontribusi

Kontribusi untuk proyek ini sangat dihargai.

## 📜 Lisensi

Proyek ini dilisensikan di bawah [AGPL-3.0 License](LICENSE).

## 📞 Kontak

- Email: alfarisbachmid@gmail.com
- GitHub: [username](https://github.com/muhdadbachmid)