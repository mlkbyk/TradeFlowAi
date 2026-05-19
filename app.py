import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# .env dosyasını yükle (API key için)
load_dotenv()

# ──────── agent_system import (yeni sınıfları eski isimlerle eşle) ────────
try:
    from agent_system import (
        EnhancedEventBus          as EventBus,
        EnhancedObservabilityEngine as ObservabilityEngine,
        EnhancedProductionRAG     as ProductionRAG,
        EnhancedNegotiationSession as NegotiationSession,
        EventType,
        GeminiEmbeddingProvider
    )
    BACKEND_READY = True
except ImportError as e:
    BACKEND_READY = False
    st.warning(f"⚠️ agent_system.py yüklenemedi: {e}. Mock modu çalışacak.")

# ═══════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════
st.set_page_config(
    page_title="TradeFlow AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════
# CSS (orijinal CSS aynen korunuyor)
# ═══════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { margin:0; padding:0; box-sizing:border-box; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp { background:#070711; }
header, footer, #MainMenu { visibility:hidden; display:none; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display:none; }
[data-testid="stAppViewContainer"] { background:#070711; }

.block-container {
    max-width:400px !important; height:780px !important; background:#111120 !important;
    border-radius:46px !important; border:2px solid #1f1f38 !important;
    overflow-y:auto !important; overflow-x:hidden !important; padding:16px 18px 24px 18px !important;
    margin:20px auto !important;
    box-shadow:0 0 0 7px #070711, 0 0 0 9px #1a1a2d, 0 32px 80px rgba(88,56,255,0.18) !important;
    scrollbar-width:thin; display:flex; flex-direction:column;
}
.block-container::-webkit-scrollbar { width:3px; }
.block-container::-webkit-scrollbar-track { background:#1a1a30; border-radius:10px; }
.block-container::-webkit-scrollbar-thumb { background:#5b5ef4; border-radius:10px; }
[data-testid="stVerticalBlock"] { gap:0!important; display:flex; flex-direction:column; justify-content:space-between; }

.notch-wrap { display:flex; justify-content:center; margin-bottom:8px; }
.notch { width:110px; height:6px; border-radius:999px; background:#1c1c32; }

.status-bar { display:flex; justify-content:space-between; align-items:center; color:white; font-size:11px; font-weight:600; margin-bottom:8px; }

.app-header { display:flex; justify-content:space-between; align-items:center; padding-bottom:10px; border-bottom:1px solid #1d1d35; margin-bottom:12px; }
.logo { font-size:18px; font-weight:800; background:linear-gradient(90deg,#818cf8,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.avatar { width:34px; height:34px; border-radius:50%; background:linear-gradient(135deg,#5b5ef4,#8b5cf6); display:flex; align-items:center; justify-content:center; font-size:16px; }

.greet-small { color:#6b7280; font-size:12px; margin-top:4px; }
.greet-big { color:white; font-size:24px; font-weight:800; margin-top:2px; margin-bottom:16px; }

.cards { display:flex; gap:8px; margin-bottom:20px; }
.card { flex:1; background:rgba(255,255,255,0.03); border:1px solid #1d1d35; border-radius:14px; padding:10px 8px; }
.card-icon { font-size:18px; margin-bottom:6px; }
.card-value { color:white; font-size:18px; font-weight:800; }
.card-title { color:#6b7280; font-size:9px; margin-top:4px; }

.sec-label { font-family:monospace; font-size:10px; color:#5a5a80; letter-spacing:2px; text-transform:uppercase; margin-bottom:8px; margin-top:4px; }

.tx-list { display:flex; flex-direction:column; gap:8px; margin-bottom:20px; }
.tx-item { background:rgba(255,255,255,0.02); border:1px solid #1a1a30; border-radius:14px; padding:10px 12px; display:flex; align-items:center; gap:10px; }
.tx-icon { width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:16px; }
.tx-icon.buy { background:rgba(91,94,244,0.15); }
.tx-icon.sell { background:rgba(16,185,129,0.12); }
.tx-icon.pend { background:rgba(251,191,36,0.12); }
.tx-info { flex:1; }
.tx-name { font-size:12px; font-weight:600; color:#e2e8f0; }
.tx-date { font-size:9px; color:#5a5a80; margin-top:2px; }
.tx-price { font-size:13px; font-weight:700; }
.tx-price.buy { color:#818cf8; }
.tx-price.sell { color:#34d399; }
.tx-price.pend { color:#fbbf24; }

.stTextInput input, .stNumberInput input, .stSelectbox select {
    background:#0f0f22!important; color:white!important; border-radius:12px!important;
    border:1px solid #1d1d35!important; font-size:13px!important; padding:8px 12px!important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label {
    color:#6b7280!important; font-size:11px!important; font-weight:500!important; margin-bottom:4px!important;
}
.stSlider { margin:8px 0; }
.stSlider label { color:#6b7280!important; font-size:11px!important; }

.stButton button {
    width:100%; border:none!important; border-radius:40px!important;
    background:linear-gradient(135deg,#6366f1,#8b5cf6)!important; color:white!important;
    font-weight:500!important; padding:10px 0!important; font-size:13px!important; margin:6px 0!important;
    cursor:pointer; transition:all 0.2s ease!important;
    box-shadow:0 2px 8px rgba(99,102,241,0.25)!important;
}
.stButton button:hover { transform:translateY(-1px); box-shadow:0 4px 12px rgba(99,102,241,0.35)!important; }

.summary-row { display:flex; gap:6px; margin:12px 0; }
.summary-chip { flex:1; padding:8px 6px; background:rgba(255,255,255,0.03); border:1px solid #1d1d35; border-radius:12px; text-align:center; }
.summary-chip-val { font-size:13px; font-weight:700; color:#818cf8; }
.summary-chip-lbl { font-size:8px; color:#6b7280; margin-top:2px; }

.progress-screen { text-align:center; padding:16px 0; }
.progress-emoji { font-size:40px; margin-bottom:8px; }
.progress-title { font-size:16px; font-weight:700; color:#e2e8f0; }
.progress-sub { font-size:11px; color:#6b7280; margin-bottom:16px; }
.progress-steps { display:flex; flex-direction:column; gap:6px; }
.step-item { display:flex; align-items:center; gap:10px; background:rgba(255,255,255,0.02); border:1px solid #1a1a30; border-radius:12px; padding:8px 12px; }
.step-icon { width:28px; height:28px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:13px; }
.step-icon.done { background:rgba(16,185,129,0.15); }
.step-icon.active { background:rgba(91,94,244,0.18); }
.step-icon.wait { background:rgba(255,255,255,0.04); }
.step-text { font-size:11px; font-weight:600; color:#e2e8f0; }
.step-sub { font-size:9px; color:#6b7280; }
.step-badge { margin-left:auto; font-size:8px; padding:2px 6px; border-radius:16px; }
.badge-done { background:rgba(16,185,129,0.15); color:#34d399; }
.badge-live { background:rgba(91,94,244,0.2); color:#818cf8; }
.badge-wait { background:rgba(255,255,255,0.04); color:#6b7280; }

.chat-wrap { display:flex; flex-direction:column; gap:8px; margin:8px 0; }
.msg { display:flex; gap:8px; align-items:flex-end; }
.msg.buyer { flex-direction:row; }
.msg.seller { flex-direction:row-reverse; }
.av { width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; flex-shrink:0; }
.av.buyer { background:linear-gradient(135deg,#5b5ef4,#8b5cf6); }
.av.seller { background:linear-gradient(135deg,#0ea5e9,#06b6d4); }
.bbl { max-width:75%; padding:8px 12px; border-radius:16px; font-size:11px; line-height:1.4; }
.bbl.buyer { background:linear-gradient(135deg,#2d2b7a,#3730a3); color:#e0e7ff; border-bottom-left-radius:3px; }
.bbl.seller { background:rgba(14,165,233,0.1); border:1px solid rgba(14,165,233,0.2); color:#bae6fd; border-bottom-right-radius:3px; }
.meta { font-size:8px; color:#4a4a70; margin-top:3px; }
.meta.r { text-align:right; }

.result-box {
    margin-top:16px; margin-bottom:16px; background:linear-gradient(135deg,rgba(91,94,244,0.12),rgba(139,92,246,0.08));
    border:1px solid rgba(139,92,246,0.28); border-radius:20px; padding:16px; text-align:center;
}
.result-title { color:#818cf8; font-size:10px; font-weight:600; letter-spacing:1px; margin-bottom:8px; }
.result-price { color:white; font-size:32px; font-weight:800; }
.result-original { font-size:10px; color:#6b7280; text-decoration:line-through; margin-top:6px; }
.savings-row { display:flex; justify-content:center; gap:8px; margin-top:10px; }
.savings-chip { padding:3px 10px; border-radius:16px; font-size:9px; font-weight:600; }
.chip-green { background:rgba(16,185,129,0.12); color:#34d399; }
.chip-purple { background:rgba(91,94,244,0.12); color:#818cf8; }

[data-testid="stMetric"] { background:rgba(255,255,255,0.02); border:1px solid #1d1d35; border-radius:14px; padding:10px; }
[data-testid="stMetricLabel"] { font-size:10px!important; }
[data-testid="stMetricValue"] { font-size:18px!important; }

.nav-wrap { margin-top:16px; margin-bottom:0; border-top:1px solid #1d1d35; padding-top:12px; }
hr { display:none; }
.element-container { margin-bottom:0!important; }
.stAlert { margin-top:8px; font-size:12px; }
.stSubheader { color:white!important; font-size:16px!important; font-weight:700!important; margin-bottom:4px!important; }
.caption { color:#6b7280!important; font-size:11px!important; }
[data-testid="stVerticalBlockBorderWrapper"] { padding:12px!important; margin-bottom:8px!important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# HELPERS (Mock)
# ═══════════════════════════════════════
def get_prices_mock(urun, days=90):
    np.random.seed(abs(hash(urun)) % 2**31)
    base = 7000
    dates = [datetime.today() - timedelta(days=i) for i in range(days, 0, -1)]
    prices = base + np.cumsum(np.random.normal(0, base * 0.025, days))
    return pd.DataFrame({"Tarih": dates, "Fiyat": prices.astype(int)})

def mock_chat(butce, gun, urun):
    piyasa = int(butce * random.uniform(1.08, 1.22))
    hedef = int(butce * random.uniform(0.86, 0.95))
    msgs = [
        ("buyer", "🤖 Alıcı Ajan", f"Merhaba! '{urun}' için bütçe: {butce:,} TL, süre: {gun} gün."),
        ("seller", "🏪 Satıcı Ajan", f"Ürünümüz {piyasa:,} TL'den listelendi."),
        ("buyer", "🤖 Alıcı Ajan", f"{int(piyasa*0.88):,} TL ile başlamak istiyorum."),
        ("seller", "🏪 Satıcı Ajan", f"En fazla {int(piyasa*0.96):,} TL yapabilirim."),
        ("buyer", "🤖 Alıcı Ajan", f"Hızlı kapanırsa {int(piyasa*0.91):,} TL teklif ediyorum."),
        ("seller", "🏪 Satıcı Ajan", f"{int(piyasa*0.93):,} son teklifim."),
        ("buyer", "🤖 Alıcı Ajan", f"Orta noktada buluşalım: {hedef:,} TL. Anlaşalım."),
        ("seller", "🏪 Satıcı Ajan", f"✅ Anlaşma! {hedef:,} TL."),
    ]
    return msgs, hedef, piyasa

SON_ISLEMLER = [
    ("🖥️", "buy", "MacBook Air M2", "3 gün önce", "-8.450 ₺"),
    ("📱", "sell", "iPhone 13 Pro", "1 hafta önce", "+12.200 ₺"),
    ("🎮", "pend", "PS5 + Kol", "Bekliyor", "~7.800 ₺"),
]

# ═══════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "Ana"
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []
if "bitti" not in st.session_state:
    st.session_state.bitti = False
if "sonuc" not in st.session_state:
    st.session_state.sonuc = 0
if "piyasa" not in st.session_state:
    st.session_state.piyasa = 0

# ═══════════════════════════════════════
# PHONE TOP
# ═══════════════════════════════════════
st.markdown('<div class="notch-wrap"><div class="notch"></div></div>', unsafe_allow_html=True)
current_time = datetime.now().strftime("%H:%M")
st.markdown(f'<div class="status-bar"><div>{current_time}</div><div>📶 🔋</div></div>', unsafe_allow_html=True)
st.markdown('<div class="app-header"><div class="logo">⚡ TradeFlow AI</div><div class="avatar">👤</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# ANA SAYFA
# ═══════════════════════════════════════
if st.session_state.page == "Ana":
    st.markdown('<div class="greet-small">Merhaba 👋</div>', unsafe_allow_html=True)
    st.markdown('<div class="greet-big">Başak Hanım</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="cards">
        <div class="card">
            <div class="card-icon">🤖</div>
            <div class="card-value">3</div>
            <div class="card-title">Aktif Ajan</div>
        </div>
        <div class="card">
            <div class="card-icon">💰</div>
            <div class="card-value">2.840 ₺</div>
            <div class="card-title">Tasarruf</div>
        </div>
        <div class="card">
            <div class="card-icon">✅</div>
            <div class="card-value">7</div>
            <div class="card-title">Anlaşma</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">📋 Son İşlemler</div>', unsafe_allow_html=True)
    tx_html = '<div class="tx-list">'
    for icon, tip, isim, tarih, fiyat in SON_ISLEMLER:
        tx_html += f"""
        <div class="tx-item">
            <div class="tx-icon {tip}">{icon}</div>
            <div class="tx-info">
                <div class="tx-name">{isim}</div>
                <div class="tx-date">{tarih}</div>
            </div>
            <div class="tx-price {tip}">{fiyat}</div>
        </div>"""
    tx_html += '</div>'
    st.markdown(tx_html, unsafe_allow_html=True)

    if st.button("Yeni Ajan Gönder 🚀", use_container_width=True):
        st.session_state.page = "Ajan"
        st.session_state.bitti = False
        st.session_state.mesajlar = []
        st.rerun()

# ═══════════════════════════════════════
# AJAN SAYFASI (DÜZELTİLMİŞ)
# ═══════════════════════════════════════
elif st.session_state.page == "Ajan":

    st.subheader("Yeni Pazarlık Görevi")
    st.caption("AI Ajan'ın pazarlık yapması için bilgi ver.")

    @st.cache_data
    def get_market_data(csv_path="market_listings_big.csv"):
        if not os.path.exists(csv_path):
            return pd.DataFrame(), ["MacBook Air M2", "iPhone 14", "PlayStation 5"]
        df = pd.read_csv(csv_path)
        prod_col = "product" if "product" in df.columns else "baslik"
        urunler = df[prod_col].unique().tolist()
        return df, urunler

    df_market, urun_listesi = get_market_data("market_listings_big.csv")
    urun = st.selectbox("Hangi ürünü arıyorsun?", urun_listesi)

    butce = st.number_input("Maksimum Bütçe (TL)", min_value=500, max_value=500000, value=7000, step=500)
    gun = st.slider("Kaç gün içinde teslim lazım?", 1, 30, 7)

    if not df_market.empty:
        prod_col = "product" if "product" in df_market.columns else "baslik"
        price_col = "listing_price" if "listing_price" in df_market.columns else "fiyat"
        date_col = "date" if "date" in df_market.columns else "Tarih"

        prod_df = df_market[df_market[prod_col] == urun].copy()
        if date_col in prod_df.columns:
            prod_df = prod_df.sort_values(date_col).tail(30)

        if not prod_df.empty:
            avg_price = prod_df[price_col].mean()
            st.markdown(f"<div style='color: #818cf8; font-size: 11px; font-weight: 600; margin-bottom: 4px;'>📉 {urun} - Son 30 Günlük Fiyat Trendi (Ortalama: {avg_price:,.0f} ₺)</div>", unsafe_allow_html=True)
            chart_data = prod_df[[price_col]].copy()
            if date_col in prod_df.columns:
                chart_data.index = prod_df[date_col]
            st.line_chart(chart_data, height=150)

    st.markdown(f"""
    <div class="summary-row">
        <div class="summary-chip"><div class="summary-chip-val">{butce:,} ₺</div><div class="summary-chip-lbl">Bütçe</div></div>
        <div class="summary-chip"><div class="summary-chip-val">{gun} gün</div><div class="summary-chip-lbl">Süre</div></div>
        <div class="summary-chip"><div class="summary-chip-val">~{int(butce*0.82):,} ₺</div><div class="summary-chip-lbl">Hedef Fiyat</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        basla = st.button("🚀 Pazarlığı Başlat", use_container_width=True)

    if basla and not urun:
        st.warning("⚠️ Lütfen aradığın ürünü yazınız!")

    elif basla and urun:
        # ── PROGRESS ──
        ph_progress = st.empty()
        adimlar = [
            ("🔍", "Ürün Taraması", "Piyasada ürün aranıyor..."),
            ("🤝", "Satıcı Bulunuyor", "En uygun satıcı eşleştirileniyor..."),
            ("💬", "Pazarlık Başladı", "Ajan müzakereye giriyor..."),
        ]
        for idx, (icon, baslik, alt) in enumerate(adimlar):
            steps_html = '<div class="progress-steps">'
            for i, (ic, bas, al) in enumerate(adimlar):
                if i < idx:
                    durum = "done"; badge_cls = "badge-done"; badge_txt = "✓"
                elif i == idx:
                    durum = "active"; badge_cls = "badge-live"; badge_txt = "●"
                else:
                    durum = "wait"; badge_cls = "badge-wait"; badge_txt = "○"
                steps_html += f"""
                <div class="step-item">
                    <div class="step-icon {durum}">{ic}</div>
                    <div><div class="step-text">{bas}</div><div class="step-sub">{al}</div></div>
                    <span class="step-badge {badge_cls}">{badge_txt}</span>
                </div>"""
            steps_html += '</div>'
            ph_progress.markdown(f"""
            <div class="progress-screen">
                <div class="progress-emoji">⚡</div>
                <div class="progress-title">Ajan Devrede</div>
                <div class="progress-sub">En iyi fiyat aranıyor...</div>
                {steps_html}
            </div>""", unsafe_allow_html=True)
            time.sleep(1.2)
        ph_progress.empty()

        # ── PAZARLIK (GERÇEK / MOCK) ──
        sonuc_fiyat = None
        piyasa_fiyati = 0
        is_mock_run = False                     # ← bayrak burada tanımlandı

        with st.container(border=True):
            st.markdown("💬 Ajanlar Arasındaki Pazarlık")

            if BACKEND_READY and os.path.exists("market_listings_big.csv"):
                try:
                    csv_path = "market_listings_big.csv"
                    embedding_provider = GeminiEmbeddingProvider()
                    bus = EventBus()
                    obs = ObservabilityEngine(bus)
                    rag = ProductionRAG(csv_path, embedding_provider)

                    product_list = rag.retrieve_product(urun)
                    if not product_list:
                        st.error(f"'{urun}' için eşleşen ürün bulunamadı.")
                    else:
                        product = product_list[0]          # en iyi eşleşme
                        piyasa_fiyati = product["price"]

                        taban_limit = piyasa_fiyati * 0.75
                        if butce < taban_limit:
                            st.error(f"🛑 **İşlem İptal:** Girdiğiniz bütçe ({butce:,.0f} ₺), '{product['product']}' için piyasa ortalamasının ({piyasa_fiyati:,.0f} ₺) çok altında. Ajanlar bu makası kapatamayacağı için pazarlık başlatılmadı.")
                        else:
                            session = NegotiationSession(
                                session_id=f"TRD-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                                product=product,
                                kullanici_butcesi=butce,
                                event_bus=bus,
                                obs=obs,
                                embedding_provider=embedding_provider
                            )

                            status_ph = st.empty()
                            chat_ph = st.empty()
                            chat_state = {'html': '<div class="chat-wrap">'}
                            streaming_ph = st.empty()

                            def ui_event_listener(event):
                                if event["type"] == "AGENT_THOUGHT":
                                    status_ph.info(f"🧠 {event['payload']['agent']}: {event['payload']['thought'][:100]}...")
                                elif event["type"] == "AGENT_MESSAGE":
                                    status_ph.empty()
                                    agent = event['payload']['agent']
                                    message = event['payload']['message']
                                    words = message.split()
                                    streaming_text = ""
                                    for word in words:
                                        streaming_text += word + " "
                                        streaming_ph.write(streaming_text)
                                        time.sleep(0.05)
                                    streaming_ph.empty()
                                    msg_class = "buyer" if agent == "Alıcı" else "seller"
                                    icon_ = "🤖" if agent == "Alıcı" else "🏪"
                                    chat_state['html'] += f"""
                                    <div class="msg {msg_class}">
                                        <div class="av {msg_class}">{icon_}</div>
                                        <div><div class="bbl {msg_class}">{message}</div><div class="meta {'r' if agent == 'Satıcı' else ''}">{agent}</div></div>
                                    </div>"""
                                    chat_ph.markdown(chat_state['html'] + '</div>', unsafe_allow_html=True)
                                    time.sleep(1.5)

                            bus.subscribe(EventType.AGENT_THOUGHT, ui_event_listener)
                            bus.subscribe(EventType.AGENT_MESSAGE, ui_event_listener)

                            session.run()
                            status_ph.empty()

                            # Sonuç fiyatını timeline'dan çek
                            for event in obs.timeline:
                                if event["type"] == "TRANSACTION_COMPLETED":
                                    sonuc_fiyat = event["payload"]["final_price"]
                                    break

                            if sonuc_fiyat is not None:
                                tasarruf = piyasa_fiyati - sonuc_fiyat
                                yuzde = int(tasarruf / piyasa_fiyati * 100) if piyasa_fiyati > 0 else 0
                                st.markdown(f"""
                                <div class="result-box">
                                    <div class="result-title">✅ UZLAŞILAN FİYAT</div>
                                    <div class="result-price">{sonuc_fiyat:,.0f} ₺</div>
                                    <div class="result-original">{piyasa_fiyati:,.0f} ₺ (piyasa fiyatı)</div>
                                    <div class="savings-row">
                                        <span class="savings-chip chip-green">🎉 {tasarruf:,.0f} ₺ tasarruf</span>
                                        <span class="savings-chip chip-purple">%{yuzde} indirim</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Debug paneli
                                with st.expander("🛠️ Jüri ve XAI Debug Paneli", expanded=False):
                                    st.subheader("📊 Ajan Karar Zinciri ve Policy Motoru")
                                    timeline_data = []
                                    for idx, event in enumerate(obs.timeline):
                                        event_type = event.get("type", "UNKNOWN")
                                        timestamp = event.get("timestamp", "N/A")
                                        if event_type == "AGENT_THOUGHT":
                                            agent = event["payload"].get("agent", "Unknown")
                                            thought = event["payload"].get("thought", "")[:80]
                                            timeline_data.append({"Sıra": idx+1, "Tip": "🧠 DÜŞÜNCE", "Ajan": agent, "İçerik": thought, "Zaman": timestamp[-8:]})
                                        elif event_type == "POLICY_BLOCK":
                                            reason = event["payload"].get("reason", "")
                                            agent = event["payload"].get("agent", "")
                                            timeline_data.append({"Sıra": idx+1, "Tip": "🚫 POLİCY BLOCK", "Ajan": agent, "İçerik": reason[:60], "Zaman": timestamp[-8:]})
                                        elif event_type == "TRANSACTION_COMPLETED":
                                            price = event["payload"].get("final_price", 0)
                                            timeline_data.append({"Sıra": idx+1, "Tip": "✅ BAŞARILI", "Ajan": "SİSTEM", "İçerik": f"Anlaşma: {price:,.0f} TL", "Zaman": timestamp[-8:]})
                                        elif event_type == "ACTION_TRIGGERED":
                                            payload = event["payload"]
                                            agent = payload.get("agent", "")
                                            fiyat = payload.get("fiyat", 0)
                                            timeline_data.append({"Sıra": idx+1, "Tip": "⚡ AKSİYON", "Ajan": agent, "İçerik": f"Teklif: {fiyat:,.0f} TL", "Zaman": timestamp[-8:]})
                                    if timeline_data:
                                        df_timeline = pd.DataFrame(timeline_data)
                                        st.dataframe(df_timeline, use_container_width=True, hide_index=True)
                                    st.divider()
                                    st.subheader("⚖️ Uygulanan Policy Kuralları")
                                    st.markdown(f"""
                                    - **Max Bütçe (Alıcı)**: {butce:,} TL
                                    - **Taban Fiyat (Satıcı)**: {int(piyasa_fiyati * 0.90):,} TL
                                    - **Piyasa Fiyatı**: {piyasa_fiyati:,.0f} TL
                                    - **Anomali Koruması**: {piyasa_fiyati * 0.5:,.0f} TL altında reddet
                                    """)
                                    st.subheader("💬 Tam Müzakere Geçmişi")
                                    for msg in session.history:
                                        agent = msg["rol"]
                                        text = msg["mesaj"]
                                        icon2 = "🤖" if agent == "Alıcı" else "🏪"
                                        with st.container(border=True):
                                            st.write(f"**{icon2} {agent}**: {text}")

                                # Başarılı anlaşma → session state güncelle
                                st.session_state.sonuc = sonuc_fiyat
                                st.session_state.piyasa = piyasa_fiyati
                                st.session_state.bitti = True
                            else:
                                # Gerçek ajanlar anlaşamadı → mock'a DÜŞME!
                                st.info("⚠️ Ajanlar kendi stratejik sınırları dışına çıkamadığı için anlaşma sağlanamadı.")

                except Exception as e:
                    st.error(f"❌ Backend hatası: {str(e)[:100]}")
                    is_mock_run = True          # backend çöktü → mock çalıştır
            else:
                # Backend hazır değil veya CSV yok → mock
                is_mock_run = True

        # ── MOCK FALLBACK (sadece is_mock_run True ise) ──
        if is_mock_run:
            konusma, sonuc, piyasa = mock_chat(butce, gun, urun)
            ph_chat = st.empty()
            displayed = []
            for tip, isim, mesaj in konusma:
                displayed.append((tip, isim, mesaj))
                html = '<div class="chat-wrap">'
                for t, n, m in displayed:
                    icon2 = "🤖" if t == "buyer" else "🏪"
                    html += f"""
                    <div class="msg {t}">
                        <div class="av {t}">{icon2}</div>
                        <div><div class="bbl {t}">{m}</div><div class="meta {'r' if t=='seller' else ''}">{n}</div></div>
                    </div>"""
                html += '</div>'
                ph_chat.markdown(html, unsafe_allow_html=True)
                time.sleep(0.6)

            tasarruf = piyasa - sonuc
            yuzde = int(tasarruf / piyasa * 100) if piyasa > 0 else 0
            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">✅ UZLAŞILAN FİYAT</div>
                <div class="result-price">{sonuc:,} ₺</div>
                <div class="result-original">{piyasa:,} ₺ (piyasa fiyatı)</div>
                <div class="savings-row">
                    <span class="savings-chip chip-green">🎉 {tasarruf:,} ₺ tasarruf</span>
                    <span class="savings-chip chip-purple">%{yuzde} indirim</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Mock başarılı → session state güncelle
            st.session_state.sonuc = sonuc
            st.session_state.piyasa = piyasa
            st.session_state.bitti = True

    # Daha önce anlaşma yapıldıysa sonucu göster
    elif st.session_state.bitti:
        s = st.session_state.sonuc
        p = st.session_state.piyasa
        if p > 0:                              # sıfıra bölünmeyi engelle
            t = p - s
            y = int(t / p * 100)
        else:
            t = 0
            y = 0
        st.markdown(f"""
        <div class="result-box">
            <div class="result-title">UZLAŞILAN FİYAT</div>
            <div class="result-price">{s:,.0f} ₺</div>
            <div class="result-original">{p:,.0f} ₺ (piyasa fiyatı)</div>
            <div class="savings-row">
                <span class="savings-chip chip-green">🎉 {t:,.0f} ₺ tasarruf</span>
                <span class="savings-chip chip-purple">%{y} indirim</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# PROFIL
# ═══════════════════════════════════════
elif st.session_state.page == "Profil":
    st.markdown("""
    <div style="text-align:center; padding:16px 0;">
        <div style="width:70px; height:70px; border-radius:50%; background:linear-gradient(135deg,#5b5ef4,#8b5cf6); display:flex; align-items:center; justify-content:center; margin:auto; font-size:30px;">👩‍💻</div>
        <div style="color:white; font-size:20px; font-weight:800; margin-top:12px;">Başak Hanım</div>
        <div style="color:#6b7280; font-size:11px; margin-top:4px;">@basak · Öğrenci Üye</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Toplam Tasarruf", "2.840 ₺")
    with col2:
        st.metric("Tamamlanan İşlem", "7")

    st.markdown("---")
    st.markdown("### ⚙️ Hesap")
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02); border:1px solid #1d1d35; border-radius:14px; padding:12px; margin-bottom:6px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="font-size:20px;">🤖</div>
            <div><div style="color:white; font-weight:600; font-size:13px;">Aktif Ajanlarım</div><div style="color:#6b7280; font-size:10px;">3 ajan şu an aktif</div></div>
            <div style="margin-left:auto; color:#6b7280; font-size:14px;">›</div>
        </div>
    </div>
    <div style="background:rgba(255,255,255,0.02); border:1px solid #1d1d35; border-radius:14px; padding:12px; margin-bottom:6px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="font-size:20px;">📊</div>
            <div><div style="color:white; font-weight:600; font-size:13px;">Tasarruf Raporu</div><div style="color:#6b7280; font-size:10px;">Aylık analiz</div></div>
            <div style="margin-left:auto; color:#6b7280; font-size:14px;">›</div>
        </div>
    </div>
    <div style="background:rgba(255,255,255,0.02); border:1px solid #1d1d35; border-radius:14px; padding:12px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="font-size:20px;">🔔</div>
            <div><div style="color:white; font-weight:600; font-size:13px;">Bildirimler</div><div style="color:#6b7280; font-size:10px;">Anlaşma ve fiyat uyarıları</div></div>
            <div style="margin-left:auto; color:#6b7280; font-size:14px;">›</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════
st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🏠\nAna Sayfa", use_container_width=True):
        st.session_state.page = "Ana"
        st.rerun()
with col2:
    if st.button("🚀\nAjan", use_container_width=True):
        st.session_state.page = "Ajan"
        st.session_state.bitti = False
        st.session_state.mesajlar = []
        st.rerun()
with col3:
    if st.button("👤\nProfil", use_container_width=True):
        st.session_state.page = "Profil"
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)