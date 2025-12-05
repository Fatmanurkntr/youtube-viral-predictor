# YouTube Viral İçerik Tahminleyicisi (YouTube Viral Predictor)

## Proje Hakkında

Bu proje, makine öğrenmesi tekniklerini kullanarak YouTube videolarının potansiyel izlenme sayılarını tahmin eden ve içerik üreticilerine **veri odaklı (data-driven)** büyüme stratejileri sunan uçtan uca (End-to-End) bir yapay zeka uygulamasıdır.

**Temel Amaç:** İçerik üreticilerinin deneme-yanılma yöntemine başvurmadan; başlık, etiket ve hedeflenen etkileşim oranlarına göre videolarının viral olma potansiyelini yayın öncesinde simüle edebilmelerini sağlamaktır.

---

## Veri Seti

Proje, YouTube'un ABD (US) bölgesindeki güncel trend videolarını içeren büyük veri seti üzerinde geliştirilmiştir.

- **Veri Kaynağı:** YouTube Trending Video Dataset (Kaggle - Rsrishav)
- **Boyut:** 260.000+ satır, 16 değişken.
- **İçerik:** Video başlığı, kanal adı, yayınlanma tarihi, etiketler, izlenme, beğeni, yorum sayıları.
- **Veri Kalitesi:** Veriler %100 gerçek kullanıcı davranışlarına dayanmaktadır.

> **Not:** Veri seti boyutu GitHub sınırlarını aştığı için repoya dahil edilmemiştir. Projeyi çalıştırmak için Kaggle'dan `US_youtube_trending_data.csv` dosyası indirilip `data/` klasörüne atılmalıdır.

---

## Proje Metodolojisi ve Teknik Kararlar

Proje, veriden ürüne giden yolda şu teknik aşamaları ve stratejik kararları içermektedir:

### 1. Keşifçi Veri Analizi (EDA) ve Temizlik

Veriyi modele hazırlamak için detaylı analizler yapılmıştır.

- **Logaritmik Dönüşüm (Neden Yapıldı?):** Hedef değişken olan `view_count` (izlenme) verisinin aşırı sağa çarpık (right-skewed) olduğu ve viral videoların uç değerler (outliers) oluşturduğu tespit edilmiştir. Modelin bu uç değerlere aşırı odaklanıp hata yapmasını engellemek ve veriyi **Normal Dağılıma** yaklaştırmak için hedef değişkene ve sayısal özelliklere `Log(1+x)` dönüşümü uygulanmıştır.
- **Veri Temizliği:** `description` gibi metin sütunlarındaki eksik veriler doldurulmuş, tarih formatları zaman serisi analizi için `datetime` objesine çevrilmiştir.

### 2. Baseline (Referans) Model Kurulumu

Model başarısını ölçmek için referans noktaları belirlenmiştir.

- **Baseline 1 (Ortalama):** Rastgele tahminin başarısızlığı (R2 ~ 0) kanıtlanmıştır.
- **Baseline 2 (Linear Regression):** Ham verilerle %64 başarı elde edilmiştir.
- **Baseline 3 (Random Forest):** Doğrusal olmayan (non-linear) modelin %65 başarı göstermesi, verideki karmaşık desenleri çözmek için **ağaç tabanlı modellere** geçilmesi gerektiğini kanıtlamıştır.

### 3. Öznitelik Mühendisliği (Feature Engineering)

Modelin tahmin gücünü artırmak için ham veriden yeni öznitelikler türetilmiştir.

- **Zaman Özellikleri:** `publish_hour` (yayın saati) ve `publish_day` (gün) türetilerek, izleyici trafiğinin yoğun olduğu zaman dilimleri modele öğretilmiştir.
- **Metin Özellikleri:** `title_length` (başlık uzunluğu) ve `tag_count` (etiket sayısı) türetilmiş; başlıkta ünlem (!) kullanımının etkisi sayısallaştırılmıştır.
- **Özellik Seçimi (Feature Selection):** Oluşturulan 35+ özellik arasından modele en çok katkı sağlayan **5 kritik özellik** (Log Likes, Log Comments, Dislikes, Title Length, Tag Count) seçilerek modelin hızı ve verimliliği artırılmıştır.

### 4. Model Optimizasyonu

- **Algoritma:** XGBoost Regressor.
- **Neden XGBoost?** Baseline aşamasında ağaç tabanlı modellerin daha başarılı olduğu görüldüğü için, bu ailenin en güçlü, hızlı ve optimize edilebilir algoritması olan XGBoost seçilmiştir.
- **Optimizasyon:** `GridSearchCV` kullanılarak hiperparametre optimizasyonu (`n_estimators`, `max_depth`, `learning_rate`) yapılmıştır.
- **Sonuç:** Modelin başarısı (R2 Skoru) referans noktası olan 0.65'ten **0.78** seviyesine çıkarılmıştır.

### 5. Değerlendirme (Evaluation)

Model, hiç görmediği **Test Seti** üzerinde değerlendirilmiştir.

- **Hata Analizi (Residuals):** Hataların rastgele dağıldığı ve modelin sistematik bir hata (bias) yapmadığı doğrulanmıştır.
- **Model Sınırları:** Modelin genel eğilimi başarıyla yakaladığı, ancak tahmin edilmesi zor olan "Süper Viral" (Outlier) videolarda daha muhafazakar tahminler yaptığı gözlemlenmiştir.

### 6. Canlıya Alma (Deployment)

Geliştirilen model, son kullanıcıların erişebileceği interaktif bir web arayüzüne dönüştürülmüştür.

- **Teknoloji:** Streamlit.
- **Fonksiyon:** Kullanıcılar senaryolarını girerek (Örn: "5000 beğeni alırsam ne olur?") anlık tahmin alabilir ve yapay zeka destekli stratejik tavsiyelere ulaşabilirler.

---

## Model Performansı

| Metrik          | Değer      | Açıklama                                                            |
| :-------------- | :--------- | :------------------------------------------------------------------ |
| **R2 Skoru**    | **0.7802** | Model, izlenmelerdeki değişimin %78'ini başarıyla açıklamaktadır.   |
| **Baseline R2** | 0.6509     | Feature Engineering ile **%13'lük performans artışı** sağlanmıştır. |

---

## İş İçgörüleri (Business Insights)

Veri analizi sonucunda içerik üreticileri için şu stratejik bulgular elde edilmiştir:

1.  **Zamanlama:** En yüksek trend olma potansiyeli **Cuma günleri** ve **14:00 - 19:00** saatleri arasındadır.
2.  **Başlık Yapısı:** 30 ile 70 karakter arasındaki başlıklar ve "!" işareti kullanımı izlenmeyi pozitif etkilemektedir.
3.  **Etkileşim:** İzlenme sayısını belirleyen en kritik faktör "Beğeni" sayısıdır. İzleyiciyi etkileşime geçirmek, algoritma tarafından ödüllendirilmektedir.

---

## 💻 Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1. Gereksinimleri Yükleyin

```bash
pip install -r requirements.txt
```

### 2. Uygulamayı Başlatın

Arayüzü çalıştırmak için ana dizinde (terminalde) şu komutu kullanın:

```bash
streamlit run app.py
```

## 📂 Dosya Yapısı

```text
Youtube_Projesi/
│
├── data/                   # Ham veri dosyaları (Kaggle'dan indirilmelidir)
├── model/                  # Eğitilmiş model (.pkl) ve bias faktörü
├── notebooks/              # Proje aşamaları (Jupyter Notebooks)
│   ├── 1_eda.ipynb         # Veri Analizi ve Temizlik
│   ├── 2_baseline.ipynb    # Referans Model Kurulumu
│   ├── 3_feature_engineering.ipynb # Özellik Türetme
│   ├── 4_model_optimization.ipynb  # XGBoost Optimizasyonu
│   └── 5_model_evaluation.ipynb    # Final Testler
│
├── app.py                  # Streamlit Arayüz Kodu
├── inference.py            # Tahminleme Motoru
├── requirements.txt        # Kütüphane Listesi
└── README.md               # Proje Dokümantasyonu
```
