# 🚀 YouTube Viral İçerik Tahminleyicisi (YouTube Viral Predictor)

Bu proje, Makine Öğrenmesi (XGBoost) kullanarak YouTube videolarının potansiyel izlenme sayılarını tahmin eden ve içerik üreticilerine **veriye dayalı (data-driven)** stratejiler sunan bir yapay zeka uygulamasıdır.

## 🎯 Proje Amacı

İçerik üreticilerinin deneme-yanılma yöntemini bırakıp, videolarını yayınlamadan önce **başlık, etiket ve etkileşim hedeflerini** simüle etmelerini sağlamaktır.

## 📊 Model Başarısı

- **Algoritma:** XGBoost Regressor (GridSearch Optimize)
- **Veri Seti:** 260.000+ Güncel YouTube Videosu (2024)
- **Başarı Skoru (R2):** %78.02
- **Baseline Skoru:** %65.09 (Referans model %13 geçilmiştir)

## 🛠️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için:

1. **Gerekli Kütüphaneleri Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```
