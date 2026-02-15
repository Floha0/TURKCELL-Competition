import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from pathlib import Path
import sys

# Proje kök dizinini bul
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from config.paths import TRAIN_FILE, MODELS_DIR


def train_rul_model():
    print("🚀 RUL (Kalan Ömür) Modeli Eğitimi Başlıyor...")

    # 1. Veriyi Yükle
    # Sütun isimlerini tanımla (NASA Veriseti standardı)
    cols = ['unit_number', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor_measurement{i}' for i in
                                                                           range(1, 22)]
    data_path = TRAIN_FILE

    if not data_path.exists():
        print(f"❌ HATA: Veri dosyası bulunamadı: {data_path}")
        return

    df = pd.read_csv(data_path, sep='\s+', header=None, names=cols)
    print(f"✅ Veri yüklendi: {df.shape}")

    # 2. RUL (Remaining Useful Life) Hesapla
    # Mantık: Her motorun ulaştığı maksimum döngü ömrüdür.
    # RUL = Max_Cycle - Current_Cycle
    print("⏳ RUL etiketleri hesaplanıyor...")

    # Her motorun max döngüsünü bul
    max_cycles = df.groupby('unit_number')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max']

    # Ana tabloya birleştir
    df = df.merge(max_cycles, on='unit_number', how='left')

    # RUL hesapla
    df['RUL'] = df['max'] - df['cycle']

    # Gereksiz sütunları at (max sütunu artık lazım değil)
    df.drop('max', axis=1, inplace=True)

    # 3. Eğitim İçin Hazırlık
    # Eğitilecek özellikler (Sensörler + Ayarlar)
    features = [c for c in df.columns if c not in ['unit_number', 'cycle', 'RUL']]
    target = 'RUL'

    X = df[features]
    y = df[target]

    # 4. Modeli Eğit (Random Forest Regressor)
    # Bu algoritma sensörler arasındaki karmaşık ilişkileri iyi yakalar
    print("🧠 Model eğitiliyor (Bu işlem birkaç saniye sürebilir)...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)

    # 5. Modeli Kaydet
    save_path = MODELS_DIR / 'rul_model.pkl'
    # Klasör yoksa oluştur
    save_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(rf_model, save_path)
    print(f"🎉 BAŞARILI: Model kaydedildi -> {save_path}")
    print("Örnek Tahmin (Cycle 1):", rf_model.predict(X.iloc[[0]]))


if __name__ == "__main__":
    train_rul_model()