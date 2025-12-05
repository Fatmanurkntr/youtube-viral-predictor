import joblib
import numpy as np
import pandas as pd

# Kaydedilen modeli ve özellikleri yüklüyoruz
# Dikkat: Bu dosyayı çalıştırırken model klasörünün bir üst dizininde olmalısınız!
try:
    model = joblib.load('model/best_xgb_model.pkl')
    FINAL_FEATURES = joblib.load('model/final_features.pkl')
    BIAS_FACTOR = joblib.load('model/bias_correction_factor.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Model dosyaları (best_xgb_model.pkl) bulunamadı. Lütfen 'model' klasörünün doğru konumda olduğundan emin olun.")


# Log dönüşümünü tersine çevirme fonksiyonu (Tahmini gerçek izlenme sayısına çevirir)
def exp_predict(log_value):
    corrected_log_value = log_value + (BIAS_FACTOR / 2)
    return np.expm1(corrected_log_value) # np.expm1(x) = exp(x) - 1

def make_prediction(likes: int, comment_count: int, dislikes: int, title: str, tags: str):
    """
    Kullanıcının girdiği ham veriyi alır, Feature Engineering uygular ve tahmin yapar.
    """
    
    # 1. FEATURE ENGINEERING (5 Özellikli Seti Türetme)
    title_length = len(title)
    tag_count = len(tags.split('|')) if tags and tags != 'None' else 0
    
    # Log dönüşümlerini uygula
    log_likes = np.log1p(likes)
    log_comment_count = np.log1p(comment_count)

    # DataFrame oluşturma (Modelin istediği format)
    # Sıralama ve sütun isimleri, Feature Selection'daki sırayla aynı OLMALIDIR.
    input_data = pd.DataFrame([[
        log_likes, 
        log_comment_count, 
        dislikes, 
        title_length, 
        tag_count
    ]], columns=FINAL_FEATURES)
    
    # 2. TAHMİN YAPMA VE TERS DÖNÜŞÜM
    log_prediction = model.predict(input_data)[0]
    
    # Log'dan gerçek sayıya dönüştür
    real_prediction = exp_predict(log_prediction)
    
    # Tahmin negatif çıkmasın (nadiren olur)
    return max(0, int(real_prediction))

if __name__ == '__main__':
    # Örnek test
    tahmin = make_prediction(
        likes=50000, 
        comment_count=1200, 
        dislikes=100, 
        title="BU BİR TEST BAŞLIĞIDIR!", 
        tags="bootcamp|makine|test"
    )
    print(f"Tahmini İzlenme: {tahmin:,}")