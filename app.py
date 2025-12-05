import streamlit as st
import pandas as pd
import numpy as np
from inference import make_prediction

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="YouTube Viral Stratejist",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- YAN MENÜ (PROJE KİMLİĞİ) ---
with st.sidebar:
    st.header("🤔 Bu Araç Nedir?")
    st.info("Bu proje, **Makine Öğrenmesi (XGBoost)** kullanarak YouTube videolarının potansiyel erişimini tahmin eder.")
    
    st.markdown("""
    **Kime Hitap Eder?**
    * 🎥 İçerik Üreticileri
    * 📢 Sosyal Medya Yöneticileri
    * 📈 Markalar
    
    **Nasıl Çalışır?**
    260.000+ videoluk veri seti üzerinde eğitilmiş modelimiz, girdiğiniz senaryoya göre **viral olma potansiyelinizi** hesaplar.
    """)
    
    st.metric(label="Model Doğruluğu (R2)", value="%78.02", delta="Başarılı")
    st.write("---")
    

# --- ANA BAŞLIK ---
st.title("🚀 YouTube Viral İçerik Simülatörü")
st.markdown("Videonuzu yayınlamadan önce **başlık stratejisini** ve **etkileşim hedeflerini** test edin.")
st.divider()

# --- GİRİŞ ALANLARI (İKİ KOLONLU YAPI) ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. İçerik Stratejisi")
    title = st.text_input(
        "Video Başlığı", 
        value="Bu Video Neden Viral Olacak? | İnanılmaz Sonuçlar!",
        help="Model, başlık uzunluğunu ve içindeki '!', '?' gibi tetikleyicileri analiz eder."
    )
    tags = st.text_area(
        "Etiketler (Tags)", 
        value="vlog|eğlence|challenge|2024|trend",
        help="Etiketleri '|' işareti ile ayırın."
    )

with col2:
    st.subheader("2. Hedeflenen Etkileşim (Senaryo)")
    st.markdown("_Bu video için ne kadar etkileşim bekliyorsunuz?_")
    
    # Slider kullanarak "Simülasyon" hissini güçlendiriyoruz
    likes = st.slider(
        "Hedeflenen Beğeni (Like)", 
        min_value=100, max_value=100000, value=5000, step=100,
        help="İzlenmeyi en çok artıran faktör beğenidir."
    )
    
    comments = st.slider(
        "Hedeflenen Yorum", 
        min_value=10, max_value=10000, value=200, step=10,
        help="Yorum sayısı, izleyici bağlılığını gösterir."
    )
    
    # Dislike'ı "Gelişmiş Ayarlar" içine saklayarak arayüzü temiz tutuyoruz
    with st.expander("Gelişmiş Ayarlar (Dislike Tahmini)"):
        dislikes = st.number_input("Tahmini Dislike", value=int(likes * 0.02), help="Genelde like sayısının %2'si kadardır.")

# --- TAHMİN BUTONU VE SONUÇLAR ---
st.divider()
analyze_button = st.button("✨ Analizi Başlat ve Tahmin Et", type="primary", use_container_width=True)

if analyze_button:
    if not title:
        st.error("Lütfen bir video başlığı girin.")
    else:
        with st.spinner("Yapay Zeka (XGBoost) verileri analiz ediyor..."):
            # Tahmin Fonksiyonunu Çağır
            prediction = make_prediction(likes, comments, dislikes, title, tags)
            
            # --- SONUÇ EKRANI ---
            st.success("✅ Analiz Tamamlandı!")
            
            # 3 Kutu Yan Yana (Metrikler)
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(label="Tahmini İzlenme (Views)", value=f"{prediction:,}")
            
            with m2:
                # Basit bir "Viral Skoru" (Görsel Zenginlik İçin)
                # Formül: (Etkileşim / İzlenme) oranına göre basit bir skor
                engagement_ratio = (likes + comments) / (prediction if prediction > 0 else 1) * 100
                st.metric(label="Tahmini Etkileşim Oranı", value=f"%{engagement_ratio:.2f}")
                
            with m3:
                # Başlık Analizi (Dinamik Geri Bildirim)
                char_len = len(title)
                if 30 <= char_len <= 70:
                    status_text = "Mükemmel Uzunluk"
                    delta_color = "normal"
                else:
                    status_text = "Geliştirilebilir"
                    delta_color = "off" # Gri renk
                    
                st.metric(label="Başlık Uzunluğu", value=f"{char_len} Karakter", delta=status_text, delta_color=delta_color)

            # --- DETAYLI AKSİYON TAVSİYELERİ ---
            st.warning("### 💡 Yapay Zeka Tavsiyeleri")
            
            advice_list = []
            
            # Başlık Analizi
            if len(title) < 30:
                st.write("⚠️ **Başlık Çok Kısa:** Daha açıklayıcı ve anahtar kelime içeren bir başlık (30-70 karakter) kullanın.")
            elif len(title) > 70:
                st.write("⚠️ **Başlık Çok Uzun:** Mobil kullanıcılar için başlığın sonu kesilebilir. Biraz kısaltmayı deneyin.")
            else:
                st.write("✅ **Başlık Uzunluğu:** İdeal aralıkta (30-70 karakter).")
            
            # Heyecan Faktörü
            if "!" in title:
                st.write("🔥 **Heyecan Faktörü:** Başlıkta ünlem (!) kullanmanız dikkat çekiciliği artırıyor.")
            else:
                st.write("💡 **İpucu:** Başlığa ünlem (!) işareti eklemek tıklanma oranını artırabilir.")
                
            # Zamanlama (Statik Veri)
            st.write("📅 **Yayınlama Stratejisi:** Verilerimize göre videonuzu **Cuma günü 14:00 - 19:00** arasında yayınlamak trend olma şansını artırır.")