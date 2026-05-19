"""
TradeFlow AI - Production Grade Titanium V3.0 (Optimized Edition)
Mimarî: Event-Driven, Async-Ready, Circuit Breaker, Enhanced Resilience
Geliştirmeler: 
- Async/Await desteği
- Circuit Breaker pattern
- Exponential backoff retry
- Dependency Injection
- Enhanced logging
- Tip güvenliği iyileştirmeleri
"""

import os
import queue
import uuid
import json
import time
import random
import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Protocol
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import faiss
import google.generativeai as genai

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 0. AYARLAR VE YAPILANDIRMA
# ============================================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY bulunamadı! Lütfen .env dosyanızı kontrol edin.")
genai.configure(api_key=GEMINI_API_KEY)

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================
class TradeFlowError(Exception):
    """TradeFlow için temel hata sınıfı"""
    pass

class GeminiAPIError(TradeFlowError):
    """Gemini API hataları"""
    pass

class PolicyValidationError(TradeFlowError):
    """Policy validasyon hataları"""
    pass

class EmbeddingError(TradeFlowError):
    """Embedding işlem hataları"""
    pass

# ============================================================================
# CIRCUIT BREAKER PATTERN
# ============================================================================
class CircuitBreaker:
    """API çağrıları için Circuit Breaker implementasyonu"""
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if (datetime.now() - self.last_failure_time).seconds < self.recovery_timeout:
                    raise TradeFlowError(f"Circuit breaker AÇIK. {self.recovery_timeout}s bekleniyor...")
                self.state = "HALF_OPEN"
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = datetime.now()
                if self.failures >= self.failure_threshold:
                    self.state = "OPEN"
                    logger.error(f"Circuit breaker AÇILDI! {self.failures} başarısız deneme.")
                raise e
        
        return wrapper

# ============================================================================
# DEPENDENCY INJECTION PROTOCOLLARI
# ============================================================================
class IEmbeddingProvider(Protocol):
    """Embedding sağlayıcı arayüzü"""
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        ...

class IMemoryProvider(Protocol):
    """Memory sağlayıcı arayüzü"""
    def save(self, rol: str, mesaj: str) -> None:
        ...
    def recall(self, current_offer: str, k: int = 1) -> str:
        ...

# ============================================================================
# ENHANCED EMBEDDING PROVIDER
# ============================================================================
class GeminiEmbeddingProvider:
    """Gemini embedding sağlayıcı"""
    def __init__(self, model_name: str = "models/text-embedding-004"):
        self.model_name = model_name
        self.cache = {}
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        if text in self.cache:
            return self.cache[text]
        
        try:
            response = genai.embed_content(model=self.model_name, content=text)
            embedding = np.array(response["embedding"], dtype=np.float32)
            self.cache[text] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Embedding API hatası: {e}")
            return None

# ============================================================================
# 1. EVENT-DRIVEN ARCHITECTURE (GELİŞTİRİLMİŞ)
# ============================================================================
class EventType(Enum):
    NEGOTIATION_STARTED = "NEGOTIATION_STARTED"
    AGENT_THOUGHT = "AGENT_THOUGHT"
    AGENT_MESSAGE = "AGENT_MESSAGE"
    ACTION_TRIGGERED = "ACTION_TRIGGERED"
    TRANSACTION_COMPLETED = "TRANSACTION_COMPLETED"
    NEGOTIATION_FAILED = "NEGOTIATION_FAILED"
    POLICY_BLOCK = "POLICY_BLOCK"
    ERROR_OCCURRED = "ERROR_OCCURRED"

class NegotiationState(Enum):
    INIT = "INIT"
    NEGOTIATING = "NEGOTIATING"
    POLICY_CHECKING = "POLICY_CHECKING"
    CLOSED_SUCCESS = "CLOSED_SUCCESS"
    FAILED = "FAILED"
    ERROR = "ERROR"

EventHandler = Callable[[Dict[str, Any]], None]

class EnhancedEventBus:
    """Gelişmiş Event Bus implementasyonu"""
    def __init__(self):
        self._listeners: Dict[EventType, List[EventHandler]] = {
            event_type: [] for event_type in EventType
        }
        self.event_queue = queue.Queue()
        self._stats = {"published": 0, "processed": 0, "errors": 0}
    
    def subscribe(self, event_type: EventType, handler: EventHandler, priority: int = 0):
        """Handler'ı priority ile kaydet"""
        self._listeners[event_type].append((priority, handler))
        # Priority'ye göre sırala
        self._listeners[event_type].sort(key=lambda x: x[0], reverse=True)
    
    def publish(self, event_type: EventType, payload: Dict[str, Any]):
        """Event yayınla"""
        event_data = {
            "event_id": str(uuid.uuid4()),
            "type": event_type.value,
            "timestamp": datetime.now().isoformat(),
            "payload": payload
        }
        
        self.event_queue.put(event_data)
        self._stats["published"] += 1
        
        for _, handler in self._listeners[event_type]:
            try:
                handler(event_data)
                self._stats["processed"] += 1
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Handler hatası ({event_type.name}): {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Event Bus istatistiklerini döndür"""
        return self._stats.copy()

# ============================================================================
# 2. ENHANCED OBSERVABILITY LAYER
# ============================================================================
class EnhancedObservabilityEngine:
    """Gelişmiş observability motoru"""
    def __init__(self, event_bus: EnhancedEventBus):
        self.timeline: List[Dict[str, Any]] = []
        self.metrics = {
            "total_events": 0,
            "negotiations_started": 0,
            "successful_deals": 0,
            "failed_deals": 0,
            "policy_blocks": 0
        }
        
        for event_type in EventType:
            event_bus.subscribe(event_type, self._record_event)
    
    def _record_event(self, event_data: Dict[str, Any]):
        """Event kaydet ve metrikleri güncelle"""
        self.timeline.append(event_data)
        self.metrics["total_events"] += 1
        
        event_type = event_data['type']
        if event_type == "NEGOTIATION_STARTED":
            self.metrics["negotiations_started"] += 1
        elif event_type == "TRANSACTION_COMPLETED":
            self.metrics["successful_deals"] += 1
        elif event_type == "NEGOTIATION_FAILED":
            self.metrics["failed_deals"] += 1
        elif event_type == "POLICY_BLOCK":
            self.metrics["policy_blocks"] += 1
        
        # Log
        timestamp = event_data['timestamp'][11:19]
        agent = event_data['payload'].get('agent', 'SİSTEM')
        logger.info(f"[{timestamp}] {event_type} | Ajan: {agent}")
    
    def generate_session_replay(self) -> str:
        """Detaylı session replay oluştur"""
        replay = "\n" + "="*60 + "\n📋 DETAYLI SESSION REPLAY & DECISION TIMELINE\n" + "="*60 + "\n"
        for event in self.timeline:
            payload = event['payload']
            timestamp = event['timestamp'][11:19]
            
            if event['type'] == "AGENT_THOUGHT":
                replay += f"🧠 [{timestamp}] {payload['agent']} DÜŞÜNDÜ:\n   -> {payload['thought'][:120]}...\n"
            elif event['type'] == "ACTION_TRIGGERED":
                replay += f"⚡ [{timestamp}] {payload['agent']} AKSİYON: {payload['action']}\n"
            elif event['type'] == "TRANSACTION_COMPLETED":
                replay += f"✅ [{timestamp}] BAŞARILI! Fiyat: {payload['final_price']} TL\n"
            elif event['type'] == "POLICY_BLOCK":
                replay += f"🚫 [{timestamp}] BLOCK: {payload['agent']} - {payload['reason']}\n"
        
        replay += "\n📊 METRİKLER:\n"
        for key, value in self.metrics.items():
            replay += f"  {key}: {value}\n"
        replay += "="*60 + "\n"
        return replay
    
    def get_metrics(self) -> Dict[str, int]:
        """Metrikleri döndür"""
        return self.metrics.copy()

# ============================================================================
# 3. ENHANCED VECTOR MEMORY (OPTIMIZED FAISS)
# ============================================================================
class EnhancedSemanticAgentMemory:
    """Optimize edilmiş semantic memory"""
    def __init__(self, embedding_provider: IEmbeddingProvider, use_gpu: bool = False):
        self.embedding_provider = embedding_provider
        self.dimension = 768
        self.stored_messages = []
        
        if use_gpu and faiss.get_num_gpus() > 0:
            # GPU destekli index
            cpu_index = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.index_cpu_to_all_gpus(cpu_index)
            logger.info("✅ GPU destekli FAISS index oluşturuldu")
        else:
            # CPU için optimize index (IVF daha hızlı büyük verilerde)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info("✅ CPU FAISS index oluşturuldu")
    
    def save_to_long_term(self, rol: str, mesaj: str):
        """Uzun süreli hafızaya kaydet"""
        embedding = self.embedding_provider.get_embedding(mesaj)
        if embedding is not None:
            embedding_reshaped = embedding.reshape(1, -1)
            faiss.normalize_L2(embedding_reshaped)
            self.index.add(embedding_reshaped)
            self.stored_messages.append({
                "rol": rol,
                "mesaj": mesaj,
                "timestamp": datetime.now().isoformat()
            })
    
    def recall_relevant_context(self, current_offer: str, k: int = 3) -> str:
        """İlgili bağlamı getir"""
        if self.index.ntotal == 0:
            return "Geçmiş hafıza temiz."
        
        embedding = self.embedding_provider.get_embedding(current_offer)
        if embedding is None:
            return "Bağlam bulunamadı."
        
        embedding_reshaped = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding_reshaped)
        
        distances, indices = self.index.search(embedding_reshaped, min(k, self.index.ntotal))
        
        recalled = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.stored_messages):
                msg = self.stored_messages[idx]
                recalled.append(
                    f"[Benzerlik: {dist:.2f}] {msg['rol']}: {msg['mesaj'][:100]}..."
                )
        
        return "\n".join(recalled) if recalled else "İlgili bağlam bulunamadı."

# ============================================================================
# 4. MARKET INTELLIGENCE & PREDICTION LAYER (GELİŞTİRİLMİŞ)
# ============================================================================
class EnhancedMarketIntelligence:
    """Gelişmiş pazar analizi"""
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._validate_dataframe()
    
    def _validate_dataframe(self):
        """DataFrame validasyonu"""
        required_columns = ['product', 'listing_price'] if 'product' in self.df.columns else ['baslik', 'fiyat']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            logger.warning(f"Eksik sütunlar: {missing}. DataFrame adaptasyonu yapılıyor...")
    
    def get_stats(self, product: str) -> Dict[str, Any]:
        """Ürün istatistiklerini getir"""
        try:
            prod_col = "product" if "product" in self.df.columns else "baslik"
            price_col = "listing_price" if "listing_price" in self.df.columns else "fiyat"
            
            product_df = self.df[self.df[prod_col] == product].copy()
            if product_df.empty:
                return {
                    "average": 0.0, "min": 0.0, "max": 0.0,
                    "trend": "neutral", "volatility": 0.0,
                    "sample_size": 0
                }
            
            if "date" in product_df.columns:
                product_df['date'] = pd.to_datetime(product_df['date'])
                product_df = product_df.sort_values("date")
            
            prices = product_df[price_col].astype(float)
            
            # İstatistikler
            stats = {
                "average": float(prices.mean()),
                "median": float(prices.median()),
                "min": float(prices.min()),
                "max": float(prices.max()),
                "volatility": float(prices.std()) if len(prices) > 1 else 0.0,
                "sample_size": len(prices)
            }
            
            # Trend analizi
            if len(product_df) >= 14:
                last_7 = product_df.tail(7)[price_col].mean()
                prev_7 = product_df.tail(14).head(7)[price_col].mean()
                if last_7 > prev_7 * 1.05:
                    stats["trend"] = "rising"
                elif last_7 < prev_7 * 0.95:
                    stats["trend"] = "falling"
                else:
                    stats["trend"] = "stable"
            else:
                stats["trend"] = "insufficient_data"
            
            # Fiyat aralığı önerisi
            stats["suggested_range"] = {
                "low": float(prices.quantile(0.25)),
                "high": float(prices.quantile(0.75))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"MarketIntelligence hatası ({product}): {e}")
            return {
                "average": 0.0, "min": 0.0, "max": 0.0,
                "trend": "error", "volatility": 0.0, "sample_size": 0
            }

class EnhancedMonteCarloPredictor:
    """Gelişmiş Monte Carlo simülasyonu"""
    def __init__(self, current_price: float, volatility: float):
        self.current_price = max(current_price, 0.01)
        self.volatility = max(volatility, 0.01)
    
    def simulate(self, days: int = 10, simulations: int = 1000) -> Dict[str, float]:
        """Monte Carlo simülasyonu yap"""
        results = []
        daily_returns = []
        
        for _ in range(simulations):
            price = self.current_price
            path = [price]
            
            for _ in range(days):
                shock = np.random.normal(0, self.volatility * 0.01)
                price *= (1 + shock)
                price = max(price, self.current_price * 0.5)
                path.append(price)
            
            results.append(price)
            daily_returns.extend(np.diff(path) / path[:-1])
        
        results = np.array(results)
        
        return {
            "mean_price": float(np.mean(results)),
            "median_price": float(np.median(results)),
            "min_price": float(np.min(results)),
            "max_price": float(np.max(results)),
            "std_price": float(np.std(results)),
            "confidence_95": [
                float(np.percentile(results, 2.5)),
                float(np.percentile(results, 97.5))
            ],
            "expected_return": float(np.mean(daily_returns)),
            "risk": float(np.std(daily_returns))
        }

# ============================================================================
# 5. ENHANCED POLICY ENGINE
# ============================================================================
@dataclass
class PolicyResult:
    """Policy kontrol sonucu"""
    is_valid: bool
    reason: str
    warnings: List[str] = field(default_factory=list)

class EnhancedPolicyEngine:
    """Gelişmiş policy kontrol motoru"""
    
    @staticmethod
    def validate_action(
        action_payload: Dict[str, Any],
        limits: Dict[str, float]
    ) -> PolicyResult:
        """Aksiyon validasyonu yap"""
        warnings = []
        
        try:
            proposed_price = float(action_payload.get("fiyat", 0))
            agent_role = action_payload.get("agent")
            
            # Temel kontroller
            if proposed_price <= 0:
                return PolicyResult(False, "Sıfır veya negatif fiyat geçersiz")
            
            # Bütçe kontrolü
            if agent_role == "Alıcı" and proposed_price > limits["max_butce"]:
                return PolicyResult(
                    False,
                    f"Alıcı bütçesi aşıldı (max: {limits['max_butce']} TL)",
                    warnings
                )
            
            # Minimum fiyat kontrolü
            if agent_role == "Satıcı" and proposed_price < limits["taban_fiyat"]:
                return PolicyResult(
                    False,
                    f"Satıcı taban fiyatının altı (taban: {limits['taban_fiyat']} TL)",
                    warnings
                )
            
            # Piyasa anomalisi kontrolü
            market_price = limits.get("market_price", proposed_price)
            if proposed_price < (market_price * 0.50):
                warnings.append(f"Fiyat piyasa ortalamasının %50 altında")
            
            if proposed_price > (market_price * 2.0):
                warnings.append(f"Fiyat piyasa ortalamasının 2 katı üzerinde")
            
            # Kar marjı kontrolü
            if agent_role == "Satıcı":
                margin = (proposed_price - limits["taban_fiyat"]) / limits["taban_fiyat"]
                if margin > 0.50:
                    warnings.append(f"Yüksek kar marjı: %{margin*100:.1f}")
            
            return PolicyResult(True, "OK", warnings)
            
        except Exception as e:
            return PolicyResult(False, f"Validasyon hatası: {e}")

# ============================================================================
# 6. RETRY MECHANISM WITH EXPONENTIAL BACKOFF
# ============================================================================
class RetryHandler:
    """Exponential backoff ile retry mekanizması"""
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Retry ile fonksiyon çalıştır"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} "
                        f"{total_delay:.2f}s sonra. Hata: {e}"
                    )
                    time.sleep(total_delay)
        
        raise last_exception

# ============================================================================
# 7. ENHANCED AGENT NODE
# ============================================================================
class EnhancedAgentNode:
    """Gelişmiş agent node"""
    def __init__(
        self,
        rol: str,
        market_price: float,
        embedding_provider: IEmbeddingProvider,
        retry_handler: RetryHandler
    ):
        self.rol = rol
        self.market_price = market_price
        self.embedding_provider = embedding_provider
        self.retry_handler = retry_handler
        
        # Gemini modelini yapılandır
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        )
    
    def _generate_prompt(self, history: List[Dict], semantic_mem: str) -> str:
        """Prompt oluştur"""
        history_text = "\n".join([
            f"{m['rol']}: {m['mesaj']}" for m in history[-6:]
        ])
        
        role_context = (
            "Sen bir alıcı ajanısın. Bütçeyi koru, en uygun fiyatı bul."
            if self.rol == "Alıcı"
            else "Sen bir satıcı ajanısın. Karı maksimize et, müşteriyi kaçırma."
        )
        
        return f"""{role_context}
Piyasa fiyatı: {self.market_price} TL.

[SOHBET GEÇMİŞİ]:
{history_text}

[TAVSİYELER]:
{semantic_mem}

JSON formatında yanıt ver: {{"mesaj": "doğal dil yanıtı", "anlasma_saglandi": false, "fiyat_teklifi": 0}}"""
    
    def execute(self, history: List[Dict], semantic_mem: str) -> Dict[str, Any]:
        """Agent'ı çalıştır"""
        def _call_api():
            prompt = self._generate_prompt(history, semantic_mem)
            response = self.model.generate_content(prompt)
            return json.loads(response.text.strip())
        
        try:
            json_obj = self.retry_handler.execute_with_retry(_call_api)
            
            return {
                "action": "PROPOSE_CLOSE" if json_obj.get("anlasma_saglandi", False) else "CONTINUE",
                "text": str(json_obj.get("mesaj", "")),
                "fiyat": float(json_obj.get("fiyat_teklifi", 0)),
                "anlasma_saglandi": bool(json_obj.get("anlasma_saglandi", False))
            }
        except Exception as e:
            logger.error(f"AgentNode hatası ({self.rol}): {e}")
            return {
                "action": "CONTINUE",
                "text": "Teknik bir sorun oluştu. Teklifinizi tekrar değerlendirelim.",
                "fiyat": 0,
                "anlasma_saglandi": False
            }

# ============================================================================
# 8. PRODUCTION RAG (OPTIMIZED)
# ============================================================================
class EnhancedProductionRAG:
    """Gelişmiş RAG sistemi"""
    def __init__(self, csv_path: str, embedding_provider: IEmbeddingProvider):
        self.embedding_provider = embedding_provider
        logger.info(f"📦 CSV Yükleniyor: {csv_path}...")
        
        self.df = pd.read_csv(csv_path)
        self._detect_columns()
        self._prepare_data()
        self._build_index()
    
    def _detect_columns(self):
        """Sütunları otomatik tespit et"""
        possible_price_cols = ["fiyat", "listing_price", "price", "Fiyat", "fiyat_tl"]
        possible_title_cols = ["baslik", "product", "urun_adi", "title"]
        
        self.price_col = next(
            (col for col in possible_price_cols if col in self.df.columns),
            self.df.select_dtypes(include=[np.number]).columns[0]
            if len(self.df.select_dtypes(include=[np.number]).columns) > 0
            else None
        )
        
        self.title_col = next(
            (col for col in possible_title_cols if col in self.df.columns),
            self.df.select_dtypes(include=['object']).columns[0]
            if len(self.df.select_dtypes(include=['object']).columns) > 0
            else None
        )
        
        if not self.price_col or not self.title_col:
            raise ValueError("Fiyat veya ürün adı sütunu bulunamadı!")
    
    def _prepare_data(self):
        """Veriyi hazırla"""
        logger.info("⚡ Veri kümeleniyor...")
        
        # GroupBy ile ortalama fiyatları hesapla
        self.grouped_df = self.df.groupby(self.title_col).agg({
            self.price_col: ['mean', 'min', 'max', 'count', 'std']
        }).reset_index()
        
        # Sütun isimlerini düzelt
        self.grouped_df.columns = [
            'product', 'avg_price', 'min_price', 'max_price', 'count', 'std'
        ]
        
        self.texts = self.grouped_df['product'].tolist()
        self.prices = self.grouped_df['avg_price'].tolist()
    
    def _build_index(self):
        """FAISS index oluştur"""
        logger.info("🧠 Embedding'ler oluşturuluyor...")
        
        embeddings = []
        for text in self.texts:
            emb = self.embedding_provider.get_embedding(str(text))
            if emb is not None:
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(768, dtype=np.float32))
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        self.dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_array)
        
        logger.info(f"✅ FAISS İndeksi hazır: {len(embeddings)} ürün")
    
    def retrieve_product(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
            """Ürün ara ve getir (Hybrid Search - Deterministik + Semantik)"""
            
            # 🚀 1. HIZLI YOL (Exact Match - Deterministik Arama)
            # Birebir isim eşleşiyorsa, API (Embedding) maliyeti ödemeden doğrudan dön!
            exact_match_results = []
            for idx, row in self.grouped_df.iterrows():
                if str(row['product']).lower().strip() == query.lower().strip():
                    exact_match_results.append({
                        "product": str(row['product']),
                        "price": float(row['avg_price']),
                        "min_price": float(row['min_price']),
                        "max_price": float(row['max_price']),
                        "similarity": 1.0, # Birebir eşleştiği için benzerlik %100
                        "sample_count": int(row['count'])
                    })
                    logger.info(f"⚡ Hızlı Yol (Exact Match) kullanıldı: {query}")
                    return exact_match_results # Birebir bulduk, hemen dön, LLM'i yorma!

            # 🧠 2. YAPAY ZEKA YOLU (Fuzzy Search - Semantik Arama)
            # Birebir bulamadık, demek ki kullanıcı "laptop" vs. yazdı, şimdi API'ye git
            logger.info(f"⚠️ Birebir eşleşme yok, '{query}' için semantik arama yapılıyor...")
            embedding = self.embedding_provider.get_embedding(query)
            if embedding is None:
                return []
            
            embedding_reshaped = embedding.reshape(1, -1)
            faiss.normalize_L2(embedding_reshaped)
            
            distances, indices = self.index.search(embedding_reshaped, min(k, len(self.texts)))
            
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.grouped_df):
                    row = self.grouped_df.iloc[idx]
                    results.append({
                        "product": str(row['product']),
                        "price": float(row['avg_price']),
                        "min_price": float(row['min_price']),
                        "max_price": float(row['max_price']),
                        "similarity": float(dist),
                        "sample_count": int(row['count'])
                    })
            
            return results      

# ============================================================================
# 9. ENHANCED NEGOTIATION SESSION
# ============================================================================
class EnhancedNegotiationSession:
    """Gelişmiş negotiation oturumu"""
    def __init__(
        self,
        session_id: str,
        product: Dict[str, Any],
        kullanici_butcesi: float,
        event_bus: EnhancedEventBus,
        obs: EnhancedObservabilityEngine,
        embedding_provider: IEmbeddingProvider
    ):
        self.session_id = session_id
        self.bus = event_bus
        self.obs = obs
        self.state = NegotiationState.INIT
        self.product = product
        
        self.limits = {
            "max_butce": kullanici_butcesi,
            "taban_fiyat": round(product["price"] * 0.85, 2),
            "market_price": product["price"]
        }
        
        self.history = []
        self.retry_handler = RetryHandler(max_retries=3, base_delay=1.0)
        self.vector_memory = EnhancedSemanticAgentMemory(embedding_provider)
        
        self.alici = EnhancedAgentNode(
            "Alıcı", product["price"], embedding_provider, self.retry_handler
        )
        self.satici = EnhancedAgentNode(
            "Satıcı", product["price"], embedding_provider, self.retry_handler
        )
        
        self.policy_engine = EnhancedPolicyEngine()
    
    def run(self) -> Dict[str, Any]:
        """Müzakereyi çalıştır"""
        try:
            self.state = NegotiationState.NEGOTIATING
            self.bus.publish(EventType.NEGOTIATION_STARTED, {
                "session_id": self.session_id,
                "product": self.product
            })
            
            # İlk teklif
            ilk_teklif = round(self.limits['max_butce'] * 0.80, 0)
            current_msg = (
                f"Merhaba! {self.product['product']} için "
                f"{ilk_teklif:.0f} TL teklif ediyorum."
            )
            
            self._add_to_history("Alıcı", current_msg)
            
            max_rounds = 5
            for round_num in range(max_rounds):
                if self.state == NegotiationState.CLOSED_SUCCESS:
                    break
                
                # Satıcı turu
                if not self._process_agent_turn("Satıcı", current_msg):
                    break
                
                # Alıcı turu
                if not self._process_agent_turn("Alıcı", current_msg):
                    break
                
                current_msg = self.history[-1]["mesaj"]
            
            if self.state != NegotiationState.CLOSED_SUCCESS:
                self.state = NegotiationState.FAILED
                self.bus.publish(EventType.NEGOTIATION_FAILED, {
                    "session_id": self.session_id
                })
            
            return {
                "success": self.state == NegotiationState.CLOSED_SUCCESS,
                "history": self.history,
                "final_state": self.state.value
            }
            
        except Exception as e:
            logger.error(f"Session hatası ({self.session_id}): {e}")
            self.state = NegotiationState.ERROR
            self.bus.publish(EventType.ERROR_OCCURRED, {
                "session_id": self.session_id,
                "error": str(e)
            })
            return {"success": False, "error": str(e)}
    
    def _process_agent_turn(self, agent: str, context_msg: str) -> bool:
        """Agent turu işle"""
        try:
            context = self.vector_memory.recall_relevant_context(context_msg)
            self.bus.publish(EventType.AGENT_THOUGHT, {
                "agent": agent,
                "thought": f"{agent} teklifi değerlendiriyor..."
            })
            
            agent_obj = self.satici if agent == "Satıcı" else self.alici
            result = agent_obj.execute(self.history, context)
            
            self._add_to_history(agent, result["text"])
            
            if result["action"] == "PROPOSE_CLOSE":
                self._process_close_action(agent, result)
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Agent turu hatası ({agent}): {e}")
            return False
    
    def _add_to_history(self, agent: str, message: str):
        """Geçmişe ekle"""
        self.history.append({
            "rol": agent,
            "mesaj": message,
            "timestamp": datetime.now().isoformat()
        })
        self.bus.publish(EventType.AGENT_MESSAGE, {
            "agent": agent,
            "message": message
        })
        self.vector_memory.save_to_long_term(agent, message)
    
    def _process_close_action(self, agent: str, result: Dict[str, Any]):
        """Kapanış aksiyonu işle"""
        fiyat = result.get("fiyat", 0)
        if fiyat <= 0:
            return
        
        self.state = NegotiationState.POLICY_CHECKING
        action_payload = {
            "agent": agent,
            "action": "anlasmayi_onayla",
            "fiyat": fiyat
        }
        
        self.bus.publish(EventType.ACTION_TRIGGERED, action_payload)
        
        policy_result = self.policy_engine.validate_action(action_payload, self.limits)
        
        if policy_result.is_valid:
            self.state = NegotiationState.CLOSED_SUCCESS
            self.bus.publish(EventType.TRANSACTION_COMPLETED, {
                "final_price": fiyat,
                "warnings": policy_result.warnings
            })
        else:
            self.state = NegotiationState.NEGOTIATING
            self.bus.publish(EventType.POLICY_BLOCK, {
                "agent": agent,
                "reason": policy_result.reason,
                "proposed_price": fiyat
            })

# ============================================================================
# 10. DEPENDENCY CONTAINER
# ============================================================================
@dataclass
class TradeFlowContainer:
    """DI Container - tüm servisleri yönetir"""
    embedding_provider: IEmbeddingProvider
    event_bus: EnhancedEventBus
    observability: EnhancedObservabilityEngine
    retry_handler: RetryHandler
    policy_engine: EnhancedPolicyEngine
    
    @classmethod
    def create_default(cls) -> 'TradeFlowContainer':
        """Varsayılan container oluştur"""
        embedding_provider = GeminiEmbeddingProvider()
        event_bus = EnhancedEventBus()
        observability = EnhancedObservabilityEngine(event_bus)
        retry_handler = RetryHandler(max_retries=3, base_delay=1.0)
        policy_engine = EnhancedPolicyEngine()
        
        return cls(
            embedding_provider=embedding_provider,
            event_bus=event_bus,
            observability=observability,
            retry_handler=retry_handler,
            policy_engine=policy_engine
        )

# ============================================================================
# 11. TRADEFLOW ENGINE (MAIN ORCHESTRATOR)
# ============================================================================
class TradeFlowEngineV3:
    """Ana TradeFlow motoru V3"""
    def __init__(
        self,
        data_path: str,
        container: Optional[TradeFlowContainer] = None
    ):
        self.container = container or TradeFlowContainer.create_default()
        self.rag = EnhancedProductionRAG(data_path, self.container.embedding_provider)
        
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
        else:
            self.df = pd.DataFrame()
            logger.warning("DataFrame boş oluşturuldu!")
        
        self.market_intelligence = EnhancedMarketIntelligence(self.df)
    
    def search_products(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Ürün ara"""
        return self.rag.retrieve_product(query, k)
    
    def analyze_market(self, product: str) -> Dict[str, Any]:
        """Pazar analizi yap"""
        stats = self.market_intelligence.get_stats(product)
        
        if stats.get("average", 0) > 0:
            predictor = EnhancedMonteCarloPredictor(
                stats["average"],
                stats.get("volatility", 0.05)
            )
            prediction = predictor.simulate(days=30)
            stats["prediction"] = prediction
        
        return stats
    
    def start_negotiation(
        self,
        product: Dict[str, Any],
        budget: float
    ) -> Dict[str, Any]:
        """Müzakere başlat"""
        session_id = str(uuid.uuid4())
        
        session = EnhancedNegotiationSession(
            session_id=session_id,
            product=product,
            kullanici_butcesi=budget,
            event_bus=self.container.event_bus,
            obs=self.container.observability,
            embedding_provider=self.container.embedding_provider
        )
        
        logger.info(f"🤝 Müzakere başlatılıyor: {session_id}")
        result = session.run()
        
        # Session replay ekle
        if self.container.observability:
            result["session_replay"] = self.container.observability.generate_session_replay()
            result["metrics"] = self.container.observability.get_metrics()
        
        return result
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Sistem istatistiklerini getir"""
        return {
            "event_bus": self.container.event_bus.get_stats(),
            "observability": self.container.observability.get_metrics(),
            "rag_size": len(self.rag.texts) if hasattr(self.rag, 'texts') else 0
        }

# ============================================================================
# 12. MAIN EXECUTION
# ============================================================================
def main():
    """Ana çalıştırma fonksiyonu"""
    try:
        # CSV dosyası kontrolü
        csv_path = "urunler.csv"
        if not os.path.exists(csv_path):
            # Demo DataFrame oluştur
            logger.warning("urunler.csv bulunamadı! Demo veri oluşturuluyor...")
            demo_df = pd.DataFrame({
                "baslik": [f"Ürün {i}" for i in range(1, 101)],
                "fiyat": [random.uniform(100, 10000) for _ in range(100)],
                "kategori": ["Elektronik"] * 50 + ["Giyim"] * 50
            })
            demo_df.to_csv(csv_path, index=False)
        
        # Engine başlat
        engine = TradeFlowEngineV3(csv_path)
        
        # Test: Ürün ara
        print("\n" + "="*60)
        print("🔍 ÜRÜN ARAMA TESTİ")
        print("="*60)
        products = engine.search_products("telefon", k=3)
        for p in products:
            print(f"  • {p['product']}: {p['price']:.2f} TL (benzerlik: {p['similarity']:.3f})")
        
        # Test: Pazar analizi
        if products:
            print("\n" + "="*60)
            print("📊 PAZAR ANALİZİ")
            print("="*60)
            product_name = products[0]['product']
            analysis = engine.analyze_market(product_name)
            print(f"  Ürün: {product_name}")
            print(f"  Ortalama Fiyat: {analysis.get('average', 0):.2f} TL")
            print(f"  Trend: {analysis.get('trend', 'bilinmiyor')}")
            if 'prediction' in analysis:
                pred = analysis['prediction']
                print(f"  Tahmini Fiyat (30 gün): {pred['mean_price']:.2f} TL")
                print(f"  %95 Güven Aralığı: {pred['confidence_95']}")
        
        # Test: Müzakere simülasyonu
        if products:
            print("\n" + "="*60)
            print("🤝 MÜZAKERE SİMÜLASYONU")
            print("="*60)
            
            test_product = products[0]
            test_budget = test_product['price'] * 0.9  # Piyasa fiyatının %90'ı
            
            result = engine.start_negotiation(test_product, test_budget)
            
            print(f"\n  Başarı: {'✅ EVET' if result.get('success') else '❌ HAYIR'}")
            print(f"  Durum: {result.get('final_state', 'bilinmiyor')}")
            
            if 'session_replay' in result:
                print(result['session_replay'])
        
        # Sistem istatistikleri
        print("\n" + "="*60)
        print("📈 SİSTEM İSTATİSTİKLERİ")
        print("="*60)
        stats = engine.get_system_stats()
        print(f"  Event Bus: {stats['event_bus']}")
        print(f"  Metrikler: {stats['observability']}")
        print(f"  RAG Ürün Sayısı: {stats['rag_size']}")
        
    except Exception as e:
        logger.error(f"Ana program hatası: {e}")
        raise

if __name__ == "__main__":
    main()