# ❤️ Heart Disease Risk Prediction (End-to-End ML Project)

Bu proje, kalp hastalığı riskini tahmin eden uçtan uca (end-to-end) bir makine öğrenmesi uygulamasıdır. Proje kapsamında; veri analizi, model eğitimi, feature importance analizi, modelin test edilmesi, FastAPI ile backend geliştirilmesi ve kullanıcı dostu bir frontend arayüzü oluşturulmuştur.

📌 Bu çalışma, Atıl Samancıoğlu – Veri Bilimi ve Makine Öğrenmesi 2025: 100 Günlük Kamp eğitiminin bitirme projesi kapsamında hazırlanmıştır.

# 🚀 Projenin Genel Akışı

Veri analizi ve modelleme (model.ipynb)

Modelin test edilmesi (model_test.py)

FastAPI backend geliştirme (main.py)

Swagger UI ile API testleri

Frontend (kullanıcı dostu arayüz) entegrasyonu

Tüm bu adımlar birbiriyle uyumlu şekilde ilerletilerek gerçek hayata yakın bir ML uygulaması ortaya konmuştur.

# 📊 Model Geliştirme – model.ipynb

Bu dosya, projenin makine öğrenmesi kalbidir.

Yapılan Çalışmalar:

Veri seti üzerinde EDA (Exploratory Data Analysis)

Gerekli feature engineering adımları

Model seçimi ve eğitimi

Pipeline kullanılarak:

Ölçekleme (scaling)

Modelin tek bir yapı altında toplanması

# 🔍 Feature Importance Analizi

Model eğitildikten sonra, hangi değişkenlerin tahmini daha fazla etkilediğini görmek için Feature Importance analizi yapılmıştır.

Özellikle cp (göğüs ağrısı tipi) gibi değişkenlerin modele etkisi görselleştirilmiştir

Bu analiz sayesinde:

Modelin neden böyle tahmin yaptığı daha iyi anlaşılmış

Domain bilgisi sınırlı olsa bile, model davranışı yorumlanabilir hale getirilmiştir

Bu grafik, modelin şeffaflığını artıran önemli bir adımdır.

# 🧪 Model Testi – model_test.py

Bu dosya, eğitilen ve pickle ile kaydedilen modelin bağımsız olarak doğru çalışıp çalışmadığını test etmek için oluşturulmuştur.

Amaçları:

heart_model.pkl dosyasının sorunsuz yüklenip yüklenmediğini kontrol etmek

Modelin beklenen formatta input alıp tahmin üretebildiğini doğrulamak

Notebook dışına çıkıldığında (production öncesi) modelin stabil çalıştığından emin olmak

Bu adım, gerçek projelerde çok kritik bir test aşamasıdır.

# ⚙️ FastAPI Backend – main.py

Bu dosyada, eğitilen model FastAPI kullanılarak bir REST API haline getirilmiştir.

Temel Yapılanlar:

FastAPI uygulamasının oluşturulması

Pickle ile kaydedilen pipeline’ın yüklenmesi

/predict endpoint’inin tanımlanması

# 🔁 String → Numeric Dönüştürme

Kullanılan veri seti zaten encode edilmiş (0–1–2–3 gibi) değerler içeriyordu. Ancak:

❗ Kullanıcıdan bu değerleri sayısal olarak almak kullanıcı dostu değildir.

Bu yüzden:

Kullanıcıdan Türkçe string ifadeler alındı ("erkek", "kadın", "evet", "hayır", "anjinal olmayan ağrı" vb.)

Backend tarafında mapping sözlükleri kullanılarak bu ifadeler modelin beklediği sayısal değerlere dönüştürüldü

Bu yaklaşım sayesinde:

Model yeniden encode edilmedi

Pipeline bozulmadı

API daha anlaşılır ve güvenli hale geldi

# 🧪 Swagger UI ile Test

FastAPI’nin sunduğu Swagger UI kullanılarak:

Farklı senaryolara ait veriler manuel olarak girildi

Modelin verdiği tahminler ve risk oranları test edildi

Eğitim verisinden alınan örneklerle birebir doğrulama yapıldı

Bu aşamada, modelin hem doğru hem de tutarlı sonuçlar verdiği gözlemlendi.

# 🎨 Frontend (Kullanıcı Dostu Arayüz)

Backend tamamlandıktan sonra, Cursor yardımıyla modern ve kullanıcı dostu bir frontend arayüzü oluşturulmuştur.

Frontend tarafında:

Kullanıcıdan serbest metin yerine dropdown / seçim baloncukları ile veri alındı

Tıbbi terimler Türkçeleştirildi

Sonuçlar sade ve anlaşılır şekilde gösterildi

Bu sayede proje:

Sadece teknik değil, gerçek bir kullanıcıya hitap eden bir uygulama haline geldi.

## 📁 Proje Yapısı

```text
├── templates/          # Frontend HTML dosyaları
├── .gitignore
├── heart_model.pkl     # Eğitilmiş ML modeli (pipeline)
├── main.py             # FastAPI backend
├── model.ipynb         # Model geliştirme & feature importance
├── model_test.py       # Model test scripti
└── requirements.txt    # Gerekli kütüphaneler

# 🛠️ Kullanılan Teknolojiler

Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn (EDA & Feature Importance)

FastAPI

Pydantic

Swagger UI

HTML / CSS (Frontend)

# 🎓 Eğitim ve Kazanımlar

Bu proje, Atıl Samancıoğlu’nun Veri Bilimi ve Makine Öğrenmesi 2025: 100 Günlük Kamp eğitimi kapsamında hazırlanmıştır.

Kurs süresince öğrenilen ve bu projede uygulanan başlıca konular:

Python ile veri analizi

Makine öğrenmesi algoritmaları

Model değerlendirme ve yorumlama

Feature importance ve model açıklanabilirliği

Pipeline mantığı

Modelin production’a hazırlanması

FastAPI ile ML model servisleme

Uçtan uca (end-to-end) proje geliştirme yaklaşımı

# 👤 Geliştirici: Gökdeniz Tural
