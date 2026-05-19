# TradeFlow AI ⚡

AI destekli otonom pazarlık simülatörü. Streamlit arayüzü ve Gemini tabanlı ajanlarla alıcı-satıcı müzakeresi.

## 🚀 Hızlı Başlangıç

1. Repoyu klonlayın
2. Sanal ortam oluşturun: `python -m venv venv`
3. Aktif edin: `venv\Scripts\activate` (Windows)
4. Bağımlılıkları yükleyin: `pip install -r requirements.txt`
5. `.env.example` dosyasını `.env` olarak kopyalayıp API anahtarınızı girin
6. `streamlit run app.py` ile çalıştırın

## 📌 Özellikler

- 🤖 Gemini LLM ile akıllı pazarlık ajanları
- 📊 CSV'den canlı fiyat grafikleri
- 🛡️ Policy Engine ile bütçe/piyasa koruması
- 🔍 Observability & XAI debug paneli
- 📱 Mobil uyumlu Streamlit arayüzü
- 📦 Mock modu (API olmadan da çalışır)

## 📁 Dosyalar

- `app.py` – Streamlit arayüzü
- `agent_system.py` – Backend motoru (event-driven)
- `market_listings_big.csv` – Örnek ürün verisi
- `architecture.md` – Sistem mimarisi dökümanı

## 🧪 Test

```bash
python -m pytest tests/
