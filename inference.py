import joblib
import numpy as np
import pandas as pd
import os

# --- MODEL VE AYARLARI YÜKLEME (GÜÇLENDİRİLMİŞ YOL BULMA) ---

# 1. Şu anki dosyanın (inference.py) bulunduğu klasörü bul
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Model dosyalarının tam yolunu oluştur (Linux/Windows uyumlu)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_xgb_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model', 'final_features.pkl')
BIAS_PATH = os.path.join(BASE_DIR, 'model', 'bias_correction_factor.pkl')

try:
    model = joblib.load(MODEL_PATH)
    FINAL_FEATURES = joblib.load(FEATURES_PATH)
    
    # Bias faktörünü yüklemeye çalış, yoksa varsayılan olarak 0 al
    if os.path.exists(BIAS_PATH):
        BIAS_FACTOR = joblib.load(BIAS_PATH)
    else:
        BIAS_FACTOR = 0 
        
except FileNotFoundError as e:
    # Hata mesajını detaylandıralım ki sorunu anlayalım
    raise FileNotFoundError(f"Model dosyası bulunamadı! Aranan Yol: {MODEL_PATH}. Hata: {e}")


# --- TAHMİN FONKSİYONLARI ---

def exp_predict(log_value):
    """
    Logaritmik tahmini gerçek sayıya çevirir.
    Varsa Bias düzeltmesini uygular.
    """
    corrected_log_value = log_value + (BIAS_FACTOR / 2)
    return np.expm1(corrected_log_value)

def make_prediction(likes: int, comment_count: int, dislikes: int, title: str, tags: str):
    """
    Kullanıcının girdiği ham veriyi alır, Feature Engineering uygular ve tahmin yapar.
    """
    
    # 1. FEATURE ENGINEERING
    title_length = len(title)
    tag_count = len(tags.split('|')) if tags and tags != 'None' else 0
    
    # Log dönüşümleri
    log_likes = np.log1p(likes)
    log_comment_count = np.log1p(comment_count)

    # DataFrame oluşturma
    input_data = pd.DataFrame([[
        log_likes, 
        log_comment_count, 
        dislikes, 
        title_length, 
        tag_count
    ]], columns=FINAL_FEATURES)
    
    # 2. TAHMİN VE TERS DÖNÜŞÜM
    log_prediction = model.predict(input_data)[0]
    real_prediction = exp_predict(log_prediction)
    
    return max(0, int(real_prediction))