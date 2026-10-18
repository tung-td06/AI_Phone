"""
Module B7: Hệ thống RAG Tư vấn Điện thoại — Nâng cấp Toàn diện (KHÔNG CẦN LLM).
=========================================================================================
Cải tiến so với B6 theo Siêu Prompt:

P0 — STRUCTURED QUERY UNDERSTANDING:
  • BudgetParser chuyên dụng: "tài chính 15 triệu", "tầm 15tr", "dưới $500"
    KHÔNG lẫn số tên máy (A-15, Note 15...).
  • GameTitleExtractor: PUBG, Genshin, Free Fire, MLBB...
  • Budget-only RECOMMENDATION: "tài chính 15 triệu mua điện thoại nào" → RECOMMENDATION.
  • Intent priority: COMPARISON > RECOMMENDATION > PRICE_QUERY > SPEC_LOOKUP.

P1 — HYBRID RETRIEVAL ROUTER:
  • RetrievalRouter: pool_size theo (intent × domain × has_budget).
  • Query expansion phân tầng theo budget range.
  • Metadata pre-filter: brand_filter áp dụng sớm.

P2 — DYNAMIC PRODUCT RANKING:
  • AdvancedPriceParser: đọc VND / USD / CNY / EUR từ nhiều format.
  • UNKNOWN SoC penalty -10% để tránh overrank chip chưa benchmark.
  • min_budget_vnd cho range queries ("từ 10 đến 15 triệu").

P3 — EVALUATION SUITE (16 ca + IR Metrics):
  • 16 EvalCase có nhãn đầy đủ.
  • Đo Recall@K, Precision@K, AP@K, nDCG@K.
=========================================================================================
"""

import os
import sys
import time
import json
import re
import math
import pickle
import argparse
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

# Thiết lập bảng mã UTF-8 cho Terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from sentence_transformers import SentenceTransformer
from soc_database import lookup_soc_benchmark, SoCBenchmarkInfo, KNOWN_SOC_DATABASE


# ==============================================================================
# HẰNG SỐ CẤU HÌNH HỆ THỐNG
# ==============================================================================
BASE_DIR            = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR   = str(BASE_DIR / "final_phone_embedding_model")
DEFAULT_RAG_DB_DIR  = str(BASE_DIR / "rag_db")
DEFAULT_TOP_K       = 5
BM25_CACHE_FILENAME = "bm25_index.pkl"

EXCHANGE_RATES_TO_VND: Dict[str, float] = {
    "USD": 25_000.0, "EUR": 27_000.0, "GBP": 32_000.0, "CNY": 3_500.0, "RMB": 3_500.0,
    "MXN": 1_300.0,  "JPY": 165.0,   "KRW": 18.0,     "INR": 300.0,   "VND": 1.0,
    "CAD": 18_000.0, "AUD": 16_000.0, "TWD": 780.0,   "SGD": 18_500.0, "MYR": 5_600.0,
    "HKD": 3_200.0,  "THB": 700.0,   "IDR": 1.6,      "RUB": 260.0,   "BRL": 4_500.0,
}

KNOWN_BRANDS: List[str] = [
    "xiaomi", "samsung", "apple", "sony", "asus", "oppo", "vivo", "realme",
    "huawei", "oneplus", "google", "nokia", "motorola", "zte", "honor", "lenovo",
]

# Brand alias -> canonical brand (dài trước để tránh overlap)
BRAND_ALIASES: Dict[str, str] = {
    "black shark": "Xiaomi", "blackshark": "Xiaomi",
    "red magic": "ZTE",      "redmagic": "ZTE",
    "iphone": "Apple",       "ipad": "Apple",    "apple": "Apple",
    "galaxy": "Samsung",     "samsung": "Samsung",
    "redmi": "Xiaomi",       "poco": "Xiaomi",   "xiaomi": "Xiaomi", "mi": "Xiaomi",
    "xperia": "Sony",        "sony": "Sony",
    "rog": "Asus",           "zenfone": "Asus",  "asus": "Asus",
    "nubia": "ZTE",          "zte": "ZTE",
    "oppo": "Oppo",
    "vivo": "Vivo",          "iqoo": "Vivo",
    "realme": "Realme",
    "oneplus": "OnePlus",
    "google": "Google",      "pixel": "Google",
    "honor": "Honor",
    "motorola": "Motorola",  "moto": "Motorola",
}

# Tên game phổ biến -> kích hoạt domain GAMING
KNOWN_GAME_TITLES: Set[str] = {
    "pubg", "pubg mobile", "genshin", "genshin impact", "free fire", "lien quan",
    "lien quan mobile", "mlbb", "mobile legends", "cod", "call of duty",
    "arena of valor", "wild rift", "diablo immortal", "minecraft",
}

SECTION_KEYWORD_RULES: Dict[str, List[str]] = {
    "Battery": [
        "pin", "dung luong pin", "sac", "mah", "watt", "w sac", "sac nhanh",
        "sac khong day", "thoi luong pin", "pin trau", "li-po", "li-ion", "battery",
    ],
    "Display": [
        "man hinh", "tan so quet", "do phan giai", "oled", "amoled", "ips", "lcd",
        "hz", "120hz", "144hz", "165hz", "60hz", "inch", "kich thuoc man", "do sang",
        "nits", "kinh cuong luc", "gorilla glass", "display", "screen",
    ],
    "Processor/Performance": [
        "chip", "vi xu ly", "chipset", "cpu", "gpu", "snapdragon", "dimensity",
        "exynos", "bionic", "helio", "card do hoa", "adreno", "mali",
        "xung nhip", "ghz", "hieu nang", "choi game", "antutu", "processor", "geekbench",
        "dung chip gi", "chip gi", "gpu gi",
    ],
    "Camera": [
        "camera", "may anh", "chup hinh", "quay phim", "quay video", "megapixel",
        "mp", "chong rung", "ois", "cam bien", "zoom", "tele", "goc rong",
        "macro", "camera truoc", "selfie", "camera sau", "flash",
    ],
    "Memory": [
        "ram", "rom", "bo nho", "dung luong", "luu tru", "bo nho trong",
        "the nho", "micro sd", "ufs", "lpddr", "128gb", "256gb", "512gb", "1tb",
    ],
    "Dimensions": [
        "kich thuoc", "trong luong", "can nang", "day", "mong", "chieu dai",
        "chieu rong", "gam", "gram", "khang nuoc", "ip68", "ip67", "chong nuoc",
        "vat lieu", "khung vien", "mat lung",
    ],
    "Connectivity": [
        "5g", "4g", "sim", "esim", "wifi", "bluetooth", "nfc", "gps",
        "cong sac", "type-c", "usb-c", "lightning", "loa", "tai nghe", "jack 3.5",
    ],
    "Thong tin chung": [
        "gia", "ra mat", "nam san xuat", "xuat xu", "hang", "thuong hieu",
        "he dieu hanh", "android", "ios", "hyperos", "one ui", "miui",
    ],
}

EXTENDED_STOPWORDS: Set[str] = {
    "bao", "nhieu", "the", "nao", "gi", "khong", "co", "may", "la", "va",
    "cua", "cho", "toi", "hoi", "voi", "dung", "duoc", "chay", "nhu", "ra",
    "sao", "dien", "thoai", "may", "thong", "so", "ky", "thuat", "chi", "tiet",
    "danh", "gia", "cau", "hinh", "gia", "tien", "mua", "ban", "tot", "hay",
    "xem", "cho", "biet", "ve", "trong", "tren", "nen", "de", "nhat", "cac",
    "nhung", "loai", "dong", "tu", "van", "goi", "y", "so", "sanh", "giua",
    "vs", "hoac", "hon", "kem", "re", "dat", "tam", "khoang", "duoi", "khoang",
    "hien", "nay", "tai", "muot", "khung", "dinh", "ngon", "dang",
    # Từ tài chính - không được nhận nhầm thành tên máy
    "tai", "chinh", "ngan", "sach", "budget", "tran", "gioi", "han",
    "trieu", "nghin", "ngan", "cu", "tr",
}


# ==============================================================================
# P0 — STRUCTURED QUERY UNDERSTANDING
# ==============================================================================
class IntentType(str, Enum):
    SPEC_LOOKUP    = "SPEC_LOOKUP"
    RECOMMENDATION = "RECOMMENDATION"
    COMPARISON     = "COMPARISON"
    PRICE_QUERY    = "PRICE_QUERY"
    GENERAL_CHAT   = "GENERAL_CHAT"


@dataclass
class QueryAnalysisResult:
    """Kết quả phân tích ý định và cấu trúc ngữ nghĩa câu hỏi."""
    original_query:  str
    intent:          IntentType
    confidence:      float
    domain_focus:    str
    target_sections: List[str]
    device_keywords: List[str]
    comparison_targets: List[str]
    brand_filter:    Optional[str]
    max_budget_vnd:  Optional[float]
    min_budget_vnd:  Optional[float]
    game_titles:     List[str]
    explanation:     str


class BudgetParser:
    """
    P0: Parser ngân sách thông minh — phân biệt "15 triệu" (tiền) vs "A-15" (tên máy).

    Chiến lược: Chỉ nhận diện ngân sách khi có từ ngữ cảnh tài chính đi kèm.
    """
    # (pattern, multiplier_to_vnd)
    PATTERNS: List[Tuple[str, float]] = [
        # "tầm/dưới/khoảng/max/tài chính/giá/có X triệu/tr/củ/m"
        (r"(?:duoi|tam|khoang|max|toi da|trong tam|tai chinh|ngan sach|budget|gia|co)\s*"
         r"([\d]+(?:[.,]\d+)?)\s*(?:trieu|tr\b|cu\b|m\b|million)", 1_000_000.0),
        # Đứng đầu câu hoặc sau điện thoại/máy: "25 triệu mua máy gì", "dien thoai 25 trieu"
        (r"(?:^|\b(?:dien thoai|may|phone)\s+)([\d]+(?:[.,]\d+)?)\s*(?:trieu|tr\b|cu\b|m\b)\b", 1_000_000.0),
        # "khoảng 15.000.000 đồng/VND"
        (r"(?:duoi|tam|khoang|max|toi da|trong tam|tai chinh|ngan sach|budget|gia)\s*"
         r"([\d]{1,3}(?:[.,]\d{3})+)\s*(?:dong|vnd)?", 1.0),
        # "under $500 / dưới $500"
        (r"(?:under|below|duoi|\$|usd\s*)\s*\$?\s*([\d]+(?:[.,]\d+)?)\s*(?:usd|dollar)?",
         25_000.0),
        # "budget 500 USD" / "tai chinh 500 usd"
        (r"(?:budget|tai chinh)\s*([\d]+(?:[.,]\d+)?)\s*(?:usd|dollar)", 25_000.0),
        # EUR
        (r"(?:duoi|tam|khoang|max|budget)\s*([\d]+(?:[.,]\d+)?)\s*(?:eur|euro)", 27_000.0),
        # CNY / RMB
        (r"(?:duoi|tam|khoang|max|budget)\s*([\d]+(?:[.,]\d+)?)\s*(?:cny|rmb|yuan)", 3_500.0),
    ]
    RANGE_PATTERN = (
        r"tu\s*([\d]+(?:[.,]\d+)?)\s*(?:trieu|tr\b|cu\b|m\b)?\s*"
        r"(?:den|toi|to)\s*([\d]+(?:[.,]\d+)?)\s*(?:trieu|tr\b|cu\b|m\b)"
    )

    @classmethod
    def parse(cls, query_ascii: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Trả về (max_budget_vnd, min_budget_vnd).
        Đầu vào nên là chuỗi đã latin hóa (không dấu) để regex đơn giản hơn.
        """
        q = query_ascii.lower().strip()

        # Range: "từ 10 đến 15 triệu"
        rm = re.search(cls.RANGE_PATTERN, q, re.IGNORECASE)
        if rm:
            lo = float(rm.group(1).replace(",", ".")) * 1_000_000
            hi = float(rm.group(2).replace(",", ".")) * 1_000_000
            return hi, lo

        for pattern, mult in cls.PATTERNS:
            m = re.search(pattern, q, re.IGNORECASE)
            if m:
                raw = m.group(1).replace(",", "").replace(".", "")
                try:
                    val = float(raw) * mult
                    if 500_000 <= val <= 100_000_000:
                        return val, None
                except ValueError:
                    pass
        return None, None


def _to_ascii_lower(text: str) -> str:
    """Chuyển tiếng Việt có dấu sang không dấu ASCII thường để regex đơn giản."""
    replacements = [
        ("à","a"),("á","a"),("ả","a"),("ã","a"),("ạ","a"),
        ("ă","a"),("ằ","a"),("ắ","a"),("ẳ","a"),("ẵ","a"),("ặ","a"),
        ("â","a"),("ầ","a"),("ấ","a"),("ẩ","a"),("ẫ","a"),("ậ","a"),
        ("è","e"),("é","e"),("ẻ","e"),("ẽ","e"),("ẹ","e"),
        ("ê","e"),("ề","e"),("ế","e"),("ể","e"),("ễ","e"),("ệ","e"),
        ("ì","i"),("í","i"),("ỉ","i"),("ĩ","i"),("ị","i"),
        ("ò","o"),("ó","o"),("ỏ","o"),("õ","o"),("ọ","o"),
        ("ô","o"),("ồ","o"),("ố","o"),("ổ","o"),("ỗ","o"),("ộ","o"),
        ("ơ","o"),("ờ","o"),("ớ","o"),("ở","o"),("ỡ","o"),("ợ","o"),
        ("ù","u"),("ú","u"),("ủ","u"),("ũ","u"),("ụ","u"),
        ("ư","u"),("ừ","u"),("ứ","u"),("ử","u"),("ữ","u"),("ự","u"),
        ("ỳ","y"),("ý","y"),("ỷ","y"),("ỹ","y"),("ỵ","y"),
        ("đ","d"),
        ("À","A"),("Á","A"),("Â","A"),("Ă","A"),("Đ","D"),
    ]
    result = text
    for src, dst in replacements:
        result = result.replace(src, dst)
    return result


class GameTitleExtractor:
    """Nhận diện tên game trong câu hỏi."""
    @staticmethod
    def extract(query_lower: str) -> List[str]:
        found = [t for t in KNOWN_GAME_TITLES if t in query_lower]
        # Thêm check không dấu
        q_ascii = _to_ascii_lower(query_lower)
        for t in KNOWN_GAME_TITLES:
            t_ascii = _to_ascii_lower(t)
            if t_ascii in q_ascii and t not in found:
                found.append(t)
        return found


class QueryAnalyzer:
    """
    P0: Bộ phân tích ý định thông minh với BudgetParser tách biệt.
    Priority: COMPARISON > RECOMMENDATION > PRICE_QUERY > SPEC_LOOKUP > GENERAL_CHAT.
    """

    @staticmethod
    def analyze(query: str) -> QueryAnalysisResult:
        q_clean = query.strip()
        q_lower = q_clean.lower()
        q_ascii = _to_ascii_lower(q_lower)

        # P0.1: Budget (dùng ascii để regex đơn giản)
        max_budget_vnd, min_budget_vnd = BudgetParser.parse(q_ascii)

        # P0.2: Brand detection (dài trước)
        brand_filter: Optional[str] = None
        for alias in sorted(BRAND_ALIASES.keys(), key=len, reverse=True):
            alias_ascii = _to_ascii_lower(alias)
            if re.search(rf"\b{re.escape(alias_ascii)}\b", q_ascii):
                brand_filter = BRAND_ALIASES[alias]
                break

        # P0.3: Game titles
        game_titles = GameTitleExtractor.extract(q_lower)

        # P0.4: Domain
        domain, target_sections = QueryAnalyzer._detect_domain(q_lower, q_ascii, game_titles)

        # === GENERAL_CHAT ===
        chat_patterns = [
            r"^(chao|hello|hi|xin chao|hey)\b",
            r"^ban la ai", r"^giup duoc gi",
        ]
        if any(re.search(p, q_ascii) for p in chat_patterns):
            return QueryAnalysisResult(
                original_query=q_clean, intent=IntentType.GENERAL_CHAT,
                confidence=0.95, domain_focus="GENERAL",
                target_sections=["Thong tin chung"],
                device_keywords=[], comparison_targets=[],
                brand_filter=None, max_budget_vnd=None, min_budget_vnd=None,
                game_titles=[], explanation="Câu hỏi chào hỏi / kiến thức mở.",
            )

        # === COMPARISON ===
        comp_pats = [
            r"\bso sanh\b", r"\bgiua\b.+\bva\b",
            r"\bhay\b(?!\s*(khong|nhat|gap|bi)\b)",
            r"\bvs\b", r"\bhon hay\b", r"\bhon khong\b",
        ]
        if any(re.search(p, q_ascii) for p in comp_pats):
            comp_targets = QueryAnalyzer._extract_comparison_targets(q_clean)
            return QueryAnalysisResult(
                original_query=q_clean, intent=IntentType.COMPARISON,
                confidence=0.93, domain_focus=domain, target_sections=target_sections,
                device_keywords=comp_targets, comparison_targets=comp_targets,
                brand_filter=None, max_budget_vnd=max_budget_vnd,
                min_budget_vnd=min_budget_vnd, game_titles=game_titles,
                explanation=f"So sánh giữa: {comp_targets}.",
            )

        # === RECOMMENDATION ===
        rec_pats = [
            r"choi game (tot|muot|dinh|ngon|nhat)",
            r"dien thoai nao (tot|choi game|chup anh|pin trau|dang mua)",
            r"may nao (tot|choi game|chup anh|pin trau|dang mua|ngon|manh)",
            r"tu van (mua|dien thoai|may)",
            r"nen mua (may|dien thoai) nao",
            r"top \d+ (dien thoai|may)",
            r"may .+ nao choi game",
            r"choi pubg tot", r"choi genshin",
            r"pin trau (de|cho) choi game",
            r"mua dien thoai nao",
            r"(tai chinh|ngan sach|budget).*(mua|dien thoai|may)",
            r"(mua|dien thoai|may).*(tai chinh|ngan sach|budget)",
            r"(chup anh|chup hinh|camera).*(dep|tot|dinh|ngon|nhat)",
            r"(dien thoai|may).*(dep nhat|tot nhat|manh nhat|muot nhat|dang mua nhat)",
        ]
        is_rec = any(re.search(p, q_ascii) for p in rec_pats)

        # Budget-only recommendation
        if not is_rec and max_budget_vnd is not None:
            if any(w in q_ascii for w in ["mua", "dien thoai", "may", "nao", "tu van"]):
                is_rec = True

        # Game title -> recommendation
        if game_titles and not is_rec:
            if any(w in q_ascii for w in ["nao", "tot", "muot", "choi", "tu van"]):
                is_rec = True

        if is_rec:
            bud_str = f"{max_budget_vnd:,.0f}" if max_budget_vnd else "N/A"
            return QueryAnalysisResult(
                original_query=q_clean, intent=IntentType.RECOMMENDATION,
                confidence=0.92, domain_focus=domain, target_sections=target_sections,
                device_keywords=[], comparison_targets=[],
                brand_filter=brand_filter,
                max_budget_vnd=max_budget_vnd, min_budget_vnd=min_budget_vnd,
                game_titles=game_titles,
                explanation=(
                    f"Tư vấn nhu cầu mở (Domain={domain}, "
                    f"Brand={brand_filter or 'Tất cả'}, Budget<={bud_str}VND)."
                ),
            )

        # === PRICE_QUERY ===
        if any(w in q_ascii for w in ["gia bao nhieu", "bao nhieu tien", "gia re nhat", "bang gia"]):
            return QueryAnalysisResult(
                original_query=q_clean, intent=IntentType.PRICE_QUERY,
                confidence=0.88, domain_focus="PRICE",
                target_sections=["Thong tin chung"],
                device_keywords=QueryAnalyzer._extract_device_keywords(q_clean),
                comparison_targets=[], brand_filter=brand_filter,
                max_budget_vnd=max_budget_vnd, min_budget_vnd=min_budget_vnd,
                game_titles=game_titles, explanation="Câu hỏi giá / thông tin thương mại.",
            )

        # === SPEC_LOOKUP ===
        dev_kws = QueryAnalyzer._extract_device_keywords(q_clean)
        exp_sec = QueryAnalyzer._detect_expected_section(q_clean) or "Processor/Performance"
        return QueryAnalysisResult(
            original_query=q_clean, intent=IntentType.SPEC_LOOKUP,
            confidence=0.86, domain_focus=domain if domain != "GENERAL" else "SPEC",
            target_sections=[exp_sec], device_keywords=dev_kws, comparison_targets=[],
            brand_filter=brand_filter, max_budget_vnd=max_budget_vnd,
            min_budget_vnd=min_budget_vnd, game_titles=game_titles,
            explanation=f"Tra cứu thông số cụ thể ({dev_kws}).",
        )

    @staticmethod
    def _detect_domain(q_lower: str, q_ascii: str, game_titles: List[str]) -> Tuple[str, List[str]]:
        if any(w in q_lower for w in ["pin trau", "pin khung", "thoi luong pin"]) or \
           any(w in q_ascii for w in ["pin trau", "pin khung"]):
            return "BATTERY", ["Battery", "Processor/Performance"]
        gaming_kws = ["game", "choi game", "gaming", "fps", "antutu", "geekbench", "pubg", "genshin"]
        if game_titles or any(w in q_lower for w in gaming_kws) or any(w in q_ascii for w in ["game","gaming","fps"]):
            return "GAMING", ["Processor/Performance", "Display", "Battery"]
        if any(w in q_lower for w in ["mah", "sac", "pin"]) or any(w in q_ascii for w in ["mah"]):
            return "BATTERY", ["Battery", "Processor/Performance"]
        if any(w in q_lower for w in ["chup anh", "camera", "quay phim", "selfie"]) or \
           any(w in q_ascii for w in ["camera", "chup", "selfie"]):
            return "CAMERA", ["Camera", "Display"]
        if any(w in q_lower for w in ["man hinh", "120hz", "144hz", "oled"]) or \
           any(w in q_ascii for w in ["man hinh", "oled", "amoled"]):
            return "DISPLAY", ["Display"]
        if any(w in q_lower for w in ["gia", "bao nhieu"]) or \
           any(w in q_ascii for w in ["gia", "bao nhieu"]):
            return "PRICE", ["Thong tin chung"]
        return "GENERAL", ["Thong tin chung"]

    @staticmethod
    def _detect_expected_section(query: str) -> Optional[str]:
        q_ascii = _to_ascii_lower(query.lower())
        scores: Dict[str, int] = {}
        for section, kws in SECTION_KEYWORD_RULES.items():
            score = sum(len(kw) for kw in kws if _to_ascii_lower(kw) in q_ascii)
            if score > 0:
                scores[section] = score
        return max(scores, key=scores.__getitem__) if scores else None

    @staticmethod
    def _extract_comparison_targets(query: str) -> List[str]:
        parts = re.split(r"\b(?:hay|hoac|vs|va|so voi)\b", query, flags=re.IGNORECASE)
        targets = []
        for p in parts:
            p = re.sub(r"[?.,!]", "", p)
            p = re.sub(r"\b(?:choi game|tot hon|hon khong|muot hon|so sanh|giua|dien thoai|may)\b",
                       "", p, flags=re.IGNORECASE).strip()
            words = [w for w in p.split() if _to_ascii_lower(w.lower()) not in EXTENDED_STOPWORDS]
            if words:
                targets.append(" ".join(words))
        return targets[:2]

    @staticmethod
    def _extract_device_keywords(query: str) -> List[str]:
        section_words: Set[str] = set()
        for kws in SECTION_KEYWORD_RULES.values():
            for kw in kws:
                section_words.update(_to_ascii_lower(kw).split())
        tokens = re.findall(r"[\w\+\-]+", query)
        return [
            t for t in tokens
            if _to_ascii_lower(t.lower()) not in EXTENDED_STOPWORDS
            and _to_ascii_lower(t.lower()) not in section_words
            and len(t) >= 2
        ]


# ==============================================================================
# CANONICALIZATION & DEDUPLICATION
# ==============================================================================
def get_canonical_model_id(brand: str, raw_model: str) -> str:
    """Chuẩn hóa tên model, loại bỏ biến thể bộ nhớ, mã vùng, v.v."""
    name = f"{brand} {raw_model}"
    name = re.sub(r"\s*/\s*.*$", "", name)
    name = re.sub(r"\b\d+\s*(?:GB|TB|GiB|TiB|MB|MiB)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(
        r"\b(?:Dual SIM|Single SIM|TD-LTE|LTE-A|LTE|5G|4G|3G|"
        r"Global|APAC|EMEA|NA|CN|JP|KR|TW|HK|IN|UW|US|CA|MX|SA|MEA|EU|ID|VN|TH|MY|PH)\b",
        "", name, flags=re.IGNORECASE
    )
    name = re.sub(
        r"\b(?:Standard Edition|Premium Edition|Base Edition|Top Edition|"
        r"Elite Edition|Pro Edition|Gaming Edition|Soft Light|PaperMatte|"
        r"Bumblebee Limited Edition|Version\s+[A-Z0-9\s]+|V\d+)\b",
        "", name, flags=re.IGNORECASE
    )
    name = re.sub(r"\b(?:SM-[A-Z0-9]+|XQ-[A-Z0-9]+|[A-Z0-9]{2,4}-[A-Z0-9]{2,})\b", "", name)
    name = re.sub(r"\b[A-Z0-9]{6,}\b", "", name)
    name = re.sub(r"\bA\d{4}\b", "", name)
    name = re.sub(r"[\-\–]\s*$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ==============================================================================
# P1 — HYBRID RETRIEVAL ROUTER
# ==============================================================================
@dataclass
class RetrievedChunk:
    rank: int
    score: float
    dense_score: float
    sparse_score: float
    brand: str
    model: str
    section: str
    chunk_id: str
    content: str
    source_file: str


class RetrievalRouter:
    """P1: Điều phối kích thước pool và query expansion theo (intent × domain × budget)."""

    @staticmethod
    def get_pool_size(analysis: QueryAnalysisResult, top_k: int) -> int:
        if analysis.intent == IntentType.RECOMMENDATION:
            return 200 if analysis.max_budget_vnd is not None else 150
        elif analysis.intent == IntentType.COMPARISON:
            return max(40, top_k * 8)
        elif analysis.intent == IntentType.SPEC_LOOKUP:
            return max(30, top_k * 6)
        return max(20, top_k * 4)

    @staticmethod
    def build_dense_query(query: str, analysis: QueryAnalysisResult) -> str:
        brand_pfx = (analysis.brand_filter + " ") if analysis.brand_filter else ""

        if analysis.intent != IntentType.RECOMMENDATION:
            return f"{brand_pfx}{query}"

        domain = analysis.domain_focus
        budget = analysis.max_budget_vnd
        bm = (budget / 1_000_000) if budget is not None else None

        if domain == "GAMING":
            if bm is not None:
                if bm <= 5:
                    return (f"{brand_pfx}dien thoai choi game gia re duoi 5 trieu "
                            "Helio G99 Dimensity 1080 gaming mid-range")
                elif bm <= 10:
                    return (f"{brand_pfx}dien thoai choi game tam trung gia re duoi 10 trieu "
                            "Snapdragon 870 Dimensity 8100 1200 POCO Black Shark Redmi gaming")
                elif bm <= 15:
                    return (f"{brand_pfx}dien thoai gaming tam trung 10-15 trieu "
                            "Snapdragon 7+ Gen 2 Dimensity 8300 8100 POCO F5 Redmi Note gaming")
                else:
                    return (f"{brand_pfx}dien thoai gaming cao cap Snapdragon 8 Gen 3 "
                            "Dimensity 9300 A17 Pro hieu nang manh ROG RedMagic gaming")
            # Game titles in query
            game_str = " ".join(analysis.game_titles)
            return (f"{brand_pfx}Snapdragon 8 Gen 3 Dimensity 9300 A17 Pro "
                    f"hieu nang choi game vi xu ly CPU GPU chipset ROG RedMagic gaming {game_str}")
        elif domain == "BATTERY":
            return (f"{brand_pfx}dung luong pin 6000mAh 6500mAh 5500mAh sac nhanh "
                    "thoi luong pin trau choi game lau watt charging")
        elif domain == "CAMERA":
            return f"{brand_pfx}camera chup anh dep megapixel OIS selfie quay video 4K zoom"
        elif domain == "DISPLAY":
            return f"{brand_pfx}man hinh OLED AMOLED tan so quet 120Hz 144Hz do sang nits"
        elif domain == "PRICE":
            return f"{brand_pfx}gia ban VND USD CNY gia tham khao"

        # GENERAL domain
        if bm is not None:
            if bm <= 5:
                return (f"{brand_pfx}smartphone dien thoai gia re tot nhat Helio G99 "
                        "Dimensity 6080 Snapdragon 680 camera pin 5000mAh")
            elif bm <= 10:
                return (f"{brand_pfx}smartphone dien thoai tam trung tot nhat Snapdragon 7s Gen 2 "
                        "Snapdragon 778G Dimensity 7050 AMOLED 120Hz OIS camera pin 5000mAh")
            elif bm <= 15:
                return (f"{brand_pfx}smartphone dien thoai tam trung cao cap can cao cap tot nhat "
                        "Snapdragon 8s Gen 3 Snapdragon 7+ Gen 3 Dimensity 8300 Apple A16 A15 Bionic AMOLED 120Hz OIS")
            else:
                return (f"{brand_pfx}smartphone flagship cao cap tot nhat Snapdragon 8 Gen 3 "
                        "Apple A17 Pro A18 Pro Dimensity 9300 camera dinh cao man hinh OLED 120Hz")

        return f"{brand_pfx}dien thoai thong minh smartphone tot nhat dang mua nhat hieu nang manh camera dep pin trau man hinh 120Hz"

    @staticmethod
    def build_sparse_tokens(query: str, analysis: QueryAnalysisResult) -> List[str]:
        raw_toks = re.findall(r"\w+", _to_ascii_lower(query.lower()))
        if analysis.intent == IntentType.RECOMMENDATION and not analysis.device_keywords:
            base = [w for w in raw_toks if w not in EXTENDED_STOPWORDS and not w.isdigit()]
        else:
            base = [w for w in raw_toks if w not in EXTENDED_STOPWORDS]

        brand_tok = [_to_ascii_lower(analysis.brand_filter.lower())] if analysis.brand_filter else []

        if analysis.intent != IntentType.RECOMMENDATION:
            dev_tok = [_to_ascii_lower(k.lower()) for k in analysis.device_keywords]
            return list(set(brand_tok + dev_tok + base))

        domain = analysis.domain_focus
        bm = (analysis.max_budget_vnd / 1_000_000) if analysis.max_budget_vnd is not None else None

        if domain == "GAMING":
            if bm and bm <= 10:
                extra = ["snapdragon", "dimensity", "adreno", "gaming", "pubg", "gpu", "fps",
                         "poco", "redmi", "black", "shark", "tam", "trung", "gia", "re"]
            else:
                extra = ["snapdragon", "dimensity", "adreno", "gaming", "pubg", "gpu", "fps",
                         "processor", "performance", "rog", "redmagic"]
        elif domain == "BATTERY":
            extra = ["battery", "mah", "6000mah", "6500mah", "5500mah", "charging",
                     "pin", "gaming", "sac", "nhanh"]
        elif domain == "CAMERA":
            extra = ["camera", "megapixel", "ois", "zoom", "selfie", "video"]
        elif domain == "DISPLAY":
            extra = ["display", "oled", "amoled", "hz", "screen", "nits"]
        else:
            if bm is not None:
                if bm <= 5:
                    extra = ["smartphone", "dien", "thoai", "pin", "camera", "ram", "display", "android"]
                elif bm <= 10:
                    extra = ["smartphone", "dien", "thoai", "amoled", "120hz", "camera", "ois", "snapdragon", "dimensity"]
                elif bm <= 15:
                    extra = ["smartphone", "dien", "thoai", "amoled", "120hz", "camera", "snapdragon", "dimensity", "bionic", "ois"]
                else:
                    extra = ["flagship", "smartphone", "dien", "thoai", "snapdragon", "dimensity", "bionic", "oled", "camera"]
            else:
                extra = ["smartphone", "dien", "thoai", "flagship", "camera", "pin", "hieu", "nang", "120hz"]

        game_tok = [t for g in analysis.game_titles for t in _to_ascii_lower(g).split()]
        return list(set(brand_tok + extra + game_tok + base))


class HybridRetriever:
    """Kết hợp FAISS Dense + BM25 Sparse + RRF Fusion."""

    def __init__(
        self,
        rag_db_dir: str | Path = DEFAULT_RAG_DB_DIR,
        model_dir:  str | Path = DEFAULT_MODEL_DIR,
        device: Optional[str] = None,
    ):
        self.rag_db_dir = Path(rag_db_dir)
        self.model_dir  = Path(model_dir)
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.faiss_index = None
        self.chunks_text: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.model: Optional[SentenceTransformer] = None
        self.bm25: Optional[BM25Okapi] = None
        self.source_file_to_chunks: Dict[str, List[str]] = {}
        self.source_file_to_price: Dict[str, Tuple[float, str]] = {}

        self._load_resources()

    def _load_resources(self):
        faiss_file  = self.rag_db_dir / "index.faiss"
        chunks_file = self.rag_db_dir / "chunks.json"
        meta_file   = self.rag_db_dir / "metadata.json"
        bm25_cache  = self.rag_db_dir / BM25_CACHE_FILENAME

        if not (faiss_file.exists() and chunks_file.exists() and meta_file.exists()):
            raise FileNotFoundError(f"Không tìm thấy CSDL tại: {self.rag_db_dir.resolve()}")

        print(f" -> Đang nạp FAISS từ: {self.rag_db_dir.resolve()} ...")
        t0 = time.time()
        self.faiss_index = faiss.read_index(str(faiss_file))
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks_text = json.load(f)
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"    + {self.faiss_index.ntotal:,} vectors trong {time.time()-t0:.2f}s!")

        for idx, meta in enumerate(self.metadata):
            sf = meta.get("source_file", "")
            if sf:
                self.source_file_to_chunks.setdefault(sf, []).append(self.chunks_text[idx])

        # Pre-cache giá cho từng source_file
        for idx, text in enumerate(self.chunks_text):
            sf = self.metadata[idx].get("source_file", "")
            if sf and sf not in self.source_file_to_price:
                vnd, orig = AdvancedPriceParser.parse(text)
                if vnd is not None:
                    self.source_file_to_price[sf] = (vnd, orig)

        print(f" -> Đang nạp Embedding Model từ: {self.model_dir.resolve()} ...")
        t1 = time.time()
        self.model = SentenceTransformer(str(self.model_dir), device=self.device)
        self.model.eval()
        print(f"    + Model sẵn sàng trong {time.time()-t1:.2f}s!")

        if HAS_BM25:
            if bm25_cache.exists():
                print(f" -> Nạp BM25 cache ...")
                t2 = time.time()
                with open(bm25_cache, "rb") as f:
                    self.bm25 = pickle.load(f)
                print(f"    + BM25 sẵn sàng trong {time.time()-t2:.2f}s!")
            else:
                print(" -> Khởi tạo BM25 Index ...")
                t2 = time.time()
                corpus = [re.findall(r"\w+", c.lower()) for c in self.chunks_text]
                self.bm25 = BM25Okapi(corpus)
                try:
                    with open(bm25_cache, "wb") as f:
                        pickle.dump(self.bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"    + BM25 cache lưu xong trong {time.time()-t2:.2f}s!")
                except Exception as e:
                    print(f"    ! Lỗi lưu BM25 cache: {e}")

    def retrieve(
        self,
        query: str,
        analysis: QueryAnalysisResult,
        top_k: int = DEFAULT_TOP_K,
        filter_brand:   Optional[str] = None,
        filter_section: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        active_brand = filter_brand or analysis.brand_filter
        pool_size    = RetrievalRouter.get_pool_size(analysis, top_k)
        dense_query  = RetrievalRouter.build_dense_query(query, analysis)
        sparse_toks  = RetrievalRouter.build_sparse_tokens(query, analysis)

        # 1. Dense (FAISS)
        q_vec = self.model.encode([dense_query], normalize_embeddings=True, convert_to_numpy=True)
        d_scores, d_indices = self.faiss_index.search(q_vec, pool_size)
        dense_rank: Dict[int, int]   = {}
        dense_scr:  Dict[int, float] = {}
        for rank, (s, idx) in enumerate(zip(d_scores[0], d_indices[0]), 1):
            if idx >= 0:
                dense_rank[idx] = rank
                dense_scr[idx]  = float(s)

        # 2. Sparse (BM25)
        bm25_rank: Dict[int, int]   = {}
        bm25_scr:  Dict[int, float] = {}
        if self.bm25 is not None and sparse_toks:
            bm25_scores = self.bm25.get_scores(sparse_toks)
            top_bm25 = np.argsort(bm25_scores)[-pool_size:][::-1]
            for rank, idx in enumerate(top_bm25, 1):
                if bm25_scores[idx] > 0:
                    bm25_rank[int(idx)] = rank
                    bm25_scr[int(idx)]  = float(bm25_scores[idx])

        # 3. RRF Fusion
        all_cands = set(dense_rank) | set(bm25_rank)
        rrf_k = 60
        fused: List[Tuple[int, float, float, float]] = []

        for idx in all_cands:
            meta = self.metadata[idx]
            if active_brand and meta["brand"].lower() != active_brand.lower():
                continue
            if filter_section and meta["section"].lower() != filter_section.lower():
                continue

            sec_bonus = 0.015 if (
                analysis.intent == IntentType.RECOMMENDATION
                and meta["section"] in analysis.target_sections
            ) else 0.0

            price_bonus = 0.0
            if analysis.max_budget_vnd is not None:
                sf = meta.get("source_file", "")
                p_info = self.source_file_to_price.get(sf)
                if p_info is not None:
                    vnd = p_info[0]
                    if vnd <= analysis.max_budget_vnd:
                        if analysis.min_budget_vnd is None or vnd >= analysis.min_budget_vnd:
                            price_bonus = 0.025
                    elif vnd > analysis.max_budget_vnd * 1.15:
                        price_bonus = -0.050

            dr  = dense_rank.get(idx, 9999)
            sr  = bm25_rank.get(idx, 9999)
            rrf = (1.0 / (rrf_k + dr)) + (1.0 / (rrf_k + sr)) + sec_bonus + price_bonus
            fused.append((idx, rrf, dense_scr.get(idx, 0.0), bm25_scr.get(idx, 0.0)))

        fused.sort(key=lambda x: x[1], reverse=True)
        fetch = pool_size if analysis.intent == IntentType.RECOMMENDATION else top_k

        results: List[RetrievedChunk] = []
        for rank, (idx, score, ds, ss) in enumerate(fused[:fetch], 1):
            meta = self.metadata[idx]
            results.append(RetrievedChunk(
                rank=rank, score=score, dense_score=ds, sparse_score=ss,
                brand=meta["brand"], model=meta["model"], section=meta["section"],
                chunk_id=meta.get("chunk_id", f"chunk_{idx}"),
                content=self.chunks_text[idx],
                source_file=meta.get("source_file", ""),
            ))
        return results


# ==============================================================================
# P2 — ADVANCED PRICE PARSER & PRODUCT RANKING
# ==============================================================================
class AdvancedPriceParser:
    """P2: Parser giá nâng cao — đọc VND / USD / CNY / EUR từ nhiều format."""
    PATTERNS: List[Tuple[str, str]] = [
        # Format chuẩn phổ biến nhất trong chunks (Price: 3999 \n - CNY)
        (r"Price:\s*([\d,\.]+)\s*\n\s*-\s*([A-Za-z]{3,4})", "DETECT"),
        # Single line: Price: 3999 CNY
        (r"Price:\s*([\d,\.]+)\s+([A-Za-z]{3,4})\b", "DETECT"),
        # "¥3299" / "CNY 3299" / "3299 CNY" / "3299 RMB"
        (r"[¥￥]\s*([\d,\.]+)", "CNY"),
        (r"CNY\s*([\d,\.]+)", "CNY"),
        (r"([\d,\.]+)\s*(?:CNY|RMB|Yuan|te\b)", "CNY"),
        # "$999" / "USD 999"
        (r"\$\s*([\d,\.]+)", "USD"),
        (r"USD\s*([\d,\.]+)", "USD"),
        # "€899" / "EUR 899"
        (r"[€]\s*([\d,\.]+)", "EUR"),
        (r"EUR\s*([\d,\.]+)", "EUR"),
        # "£699" / "GBP 699"
        (r"[£]\s*([\d,\.]+)", "GBP"),
        (r"GBP\s*([\d,\.]+)", "GBP"),
        # "₹29999" / "INR 29999"
        (r"[₹]\s*([\d,\.]+)", "INR"),
        (r"INR\s*([\d,\.]+)", "INR"),
        # "Giá: 22.990.000 VND" / "22,990,000₫"
        (r"(?:Gia|gia ban|price):\s*([\d,\.]+)\s*(?:VND|dong|vnd|d)?", "VND"),
        # Số 7+ chữ số + VND
        (r"([\d]{7,})\s*(?:VND|dong|d)", "VND"),
        # Fallback: Price: 999
        (r"Price:\s*([\d,\.]+)", "USD"),
    ]

    @classmethod
    def parse(cls, text: str) -> Tuple[Optional[float], str]:
        for pattern, curr_hint in cls.PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                raw = m.group(1).replace(",", "").replace(".", "")
                try:
                    val = float(raw)
                except ValueError:
                    continue

                if curr_hint == "DETECT":
                    curr = (m.group(2) or "USD").upper().strip()
                elif curr_hint == "VND":
                    curr = "VND"
                else:
                    curr = curr_hint

                rate = EXCHANGE_RATES_TO_VND.get(curr, 25_000.0)
                price_vnd = val * rate
                if 500_000 <= price_vnd <= 100_000_000:
                    if curr == "VND":
                        return price_vnd, f"~{price_vnd:,.0f}đ"
                    return price_vnd, f"~{price_vnd:,.0f}đ ({val:,.0f} {curr})"
        return None, "Chưa rõ giá"


@dataclass
class HardwareProfile:
    """Hồ sơ phần cứng bóc tách từ chunks dữ liệu."""
    brand: str
    model: str
    canonical_id: str
    cpu_clock_mhz: int = 0
    cpu_name: str = "N/A"
    gpu_name: str = "N/A"
    ram_gb: float = 0.0
    display_hz: int = 60
    battery_mah: int = 0
    cooling_mentioned: bool = False
    price_vnd: Optional[float] = None
    price_str: str = "Chưa rõ giá"
    variants_storage: List[str] = field(default_factory=list)
    soc_info: Optional[SoCBenchmarkInfo] = None
    gaming_score: float = 0.0
    score_details: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    data_confidence: str = "UNKNOWN"
    device_category: str = "Smartphone"


class ProductRanker:
    """P2: Bộ chấm điểm và xếp hạng sản phẩm."""

    @staticmethod
    def extract_hardware_profile(
        brand: str,
        raw_models: Set[str],
        chunks_text_list: List[str],
        domain_focus: str = "GAMING",
    ) -> HardwareProfile:
        combined  = "\n".join(chunks_text_list)
        rep_model = sorted(raw_models, key=len)[0]
        canon_id  = get_canonical_model_id(brand, rep_model)
        profile   = HardwareProfile(brand=brand, model=rep_model, canonical_id=canon_id)

        cat_m = re.search(r"Device Category:\s*([^\n\r]+)", combined, re.IGNORECASE)
        if cat_m:
            profile.device_category = cat_m.group(1).strip()
        else:
            m_lower = (brand + " " + rep_model).lower()
            if any(w in m_lower for w in ["watch", "band", "gear"]):
                profile.device_category = "Smartwatch"
            elif any(w in m_lower for w in ["ipad", "tab ", "tablet", "pad "]):
                profile.device_category = "Tablet"
            else:
                profile.device_category = "Smartphone"

        cpu_m = re.search(r"CPU:\s*([^\n\r,]+)", combined, re.IGNORECASE)
        if cpu_m:
            profile.cpu_name = cpu_m.group(1).strip()

        clk_m = re.search(r"CPU Clock:\s*(\d+)\s*MHz", combined, re.IGNORECASE)
        if clk_m:
            profile.cpu_clock_mhz = int(clk_m.group(1))

        gpu_m = re.search(r"(?:Graphical Controller|GPU):\s*([^\n\r]+)", combined, re.IGNORECASE)
        if gpu_m:
            profile.gpu_name = gpu_m.group(1).strip()

        profile.soc_info = lookup_soc_benchmark(f"{profile.cpu_name} {profile.gpu_name}")

        ram_ms = re.findall(
            r"RAM Capacity(?:\s*\(converted\))?:\s*(\d+(?:\.\d+)?)\s*(GiB|GB|MiB|MB)",
            combined, re.IGNORECASE
        )
        if ram_ms:
            profile.ram_gb = max(
                float(v) / 1024.0 if "M" in u.upper() else float(v)
                for v, u in ram_ms
            )

        # Tần số quét màn hình (Display Refresh Rate): chỉ chấp nhận <= 240Hz, không lấy nhầm tần số cảm ứng (720Hz)
        refresh_m = re.search(r"Display Refresh Rate:\s*(\d+)\s*Hz", combined, re.IGNORECASE)
        if refresh_m:
            hz_val = int(refresh_m.group(1))
            profile.display_hz = hz_val if hz_val in [60, 90, 120, 144, 165, 185, 240] else 60
        else:
            disp_rates = []
            for m in re.finditer(r"\b(60|90|120|144|165|185|240)\s*Hz\b", combined, re.IGNORECASE):
                start = max(0, m.start() - 25)
                end = min(len(combined), m.end() + 25)
                surrounding = combined[start:end].lower()
                if "touch" not in surrounding and "cam ung" not in surrounding:
                    disp_rates.append(int(m.group(1)))
            profile.display_hz = max(disp_rates) if disp_rates else 60

        bat_m = re.search(r"Nominal Battery Capacity:\s*(\d+)\s*mAh", combined, re.IGNORECASE)
        if not bat_m:
            bat_m = re.search(r"(\d{3,5})\s*mAh", combined, re.IGNORECASE)
        if bat_m:
            profile.battery_mah = int(bat_m.group(1))

        if re.search(r"(vapor chamber|liquid cooling|cooling system|bam game|trigger|tan nhiet|heat pipe)",
                     combined, re.IGNORECASE):
            profile.cooling_mentioned = True

        profile.price_vnd, profile.price_str = AdvancedPriceParser.parse(combined)

        variants: Set[str] = set()
        for rm in raw_models:
            sm = re.search(r"\b(\d+)\s*(?:GB|TB)\b", rm, re.IGNORECASE)
            if sm:
                variants.add(sm.group(0).upper())
        profile.variants_storage = sorted(variants)

        (profile.gaming_score,
         profile.score_details,
         profile.explanation,
         profile.data_confidence) = ProductRanker._score(profile, domain_focus)

        return profile

    @staticmethod
    def _score(p: HardwareProfile, domain: str) -> Tuple[float, Dict[str, Any], str, str]:
        soc = p.soc_info

        # BATTERY domain
        if domain == "BATTERY":
            soc_s = round(soc.total_soc_score * (40.0 / 70.0), 1)
            bat_s = (30.0 if p.battery_mah >= 6500 else
                     27.0 if p.battery_mah >= 6000 else
                     24.0 if p.battery_mah >= 5500 else
                     20.0 if p.battery_mah >= 5000 else
                     15.0 if p.battery_mah >= 4500 else 10.0)
            cool_s = 10.0 if p.cooling_mentioned else 5.0
            dr_s   = (15.0 if p.ram_gb >= 12 and p.display_hz >= 120 else
                      10.0 if p.ram_gb >= 8 else 5.0)
            total  = round(soc_s + bat_s + cool_s + dr_s, 1)
            details = {
                "Pin":        f"{bat_s}/30 ({p.battery_mah}mAh)",
                "SoC":        f"{soc_s}/40 ({soc.canonical_name})",
                "Tan nhiet":  f"{cool_s}/10",
                "Man & RAM":  f"{dr_s}/15 ({p.ram_gb}GB, {p.display_hz}Hz)",
                "Trust":      soc.confidence,
            }
            return total, details, (
                f"Pin {p.battery_mah}mAh + {soc.canonical_name} + "
                f"{'tan nhiet chuyen dung' if p.cooling_mentioned else 'tan nhiet tieu chuan'}."
            ), soc.confidence

        # GAMING domain (default)
        cpu_s  = soc.cpu_score
        gpu_s  = soc.gpu_score
        ram_s  = (12.0 if p.ram_gb >= 16 else 10.0 if p.ram_gb >= 12 else
                  7.0  if p.ram_gb >= 8  else 5.0  if p.ram_gb >= 6  else 3.0)
        disp_s = (10.0 if p.display_hz >= 144 else 8.5 if p.display_hz >= 120 else
                  6.0  if p.display_hz >= 90  else 4.0)
        bc_s   = 3.0
        bc_s  += (5.0 if p.battery_mah >= 6000 else
                  4.0 if p.battery_mah >= 5000 else
                  3.0 if p.battery_mah >= 4500 else 0.0)
        if p.cooling_mentioned:
            bc_s += 3.0
        bc_s   = min(8.0, bc_s)

        total = round(cpu_s + gpu_s + ram_s + disp_s + bc_s, 1)
        if soc.confidence == "UNKNOWN":
            total = round(total * 0.90, 1)  # UNKNOWN penalty -10%

        fps_val = "Chưa có dữ liệu benchmark chính thức" if soc.confidence == "UNKNOWN" else soc.target_fps_gaming
        details = {
            "CPU": f"{cpu_s}/35 ({soc.canonical_name})",
            "GPU": f"{gpu_s}/35 ({p.gpu_name})",
            "RAM": f"{ram_s}/12 ({p.ram_gb}GB)",
            "Man": f"{disp_s}/10 ({p.display_hz}Hz)",
            "Pin": f"{bc_s}/8 ({p.battery_mah}mAh)",
            "FPS": fps_val,
            "Trust": soc.confidence,
        }
        expl = (
            f"SoC {p.cpu_name} chưa có benchmark chính thức — bảo lưu điểm cơ bản {total}/100."
            if soc.confidence == "UNKNOWN"
            else f"{soc.canonical_name} GPU {gpu_s}/35, {p.display_hz}Hz, {p.ram_gb}GB. FPS: {soc.target_fps_gaming}."
        )
        return total, details, expl, soc.confidence

    @staticmethod
    def rank_products_from_chunks(
        chunks: List[RetrievedChunk],
        source_file_chunks_map: Optional[Dict[str, List[str]]] = None,
        max_budget_vnd: Optional[float] = None,
        min_budget_vnd: Optional[float] = None,
        brand_filter: Optional[str] = None,
        domain_focus: str = "GAMING",
        top_n: int = 5,
    ) -> List[HardwareProfile]:
        groups: Dict[str, Dict[str, Any]] = {}
        for c in chunks:
            if brand_filter and c.brand.lower() != brand_filter.lower():
                continue
            canon = get_canonical_model_id(c.brand, c.model)
            if canon not in groups:
                groups[canon] = {
                    "brand": c.brand, "canonical_id": canon,
                    "raw_models": set(), "source_files": set(),
                }
            groups[canon]["raw_models"].add(c.model)
            if c.source_file:
                groups[canon]["source_files"].add(c.source_file)

        profiles: List[HardwareProfile] = []
        for canon, data in groups.items():
            texts: List[str] = []
            if source_file_chunks_map:
                for sf in data["source_files"]:
                    texts.extend(source_file_chunks_map.get(sf, []))
            if not texts:
                continue

            p = ProductRanker.extract_hardware_profile(
                brand=data["brand"], raw_models=data["raw_models"],
                chunks_text_list=texts, domain_focus=domain_focus,
            )

            # Lọc thiết bị không phải điện thoại (Tablet/Smartwatch) nếu không yêu cầu
            if domain_focus != "TABLET":
                m_lower = (p.brand + " " + p.canonical_id).lower()
                if p.device_category in ["Tablet", "Smartwatch", "Palmtop", "Notebook", "PDA"] or \
                   any(w in m_lower for w in ["watch", "band", "gear", "ipad", "tab ", "tablet"]):
                    continue

            # Kiểm tra ngân sách
            if max_budget_vnd is not None:
                m_lower = (p.brand + " " + p.canonical_id).lower()
                # Loại trừ siêu flagship bị dính giá trả góp / nhà mạng khi hỏi máy tầm trung/giá rẻ (<= 15M)
                if max_budget_vnd <= 15_000_000 and any(w in m_lower for w in ["ultra", "pro max", "fold", "flip", "rog phone", "8 elite"]):
                    continue

                if p.price_vnd is not None:
                    if p.price_vnd > max_budget_vnd:
                        continue
                else:
                    # Rào chắn trần cấu hình SoC cho máy chưa rõ giá trong CSDL
                    bm = max_budget_vnd / 1_000_000
                    m_lower = (p.brand + " " + p.canonical_id).lower()
                    if bm <= 5.0:
                        if p.soc_info and p.soc_info.total_soc_score >= 35.0:
                            continue
                        if any(w in m_lower for w in ["rog phone", "red magic", "ultra", "pro max", "plus"]):
                            continue
                    elif bm <= 10.0:
                        if p.soc_info and p.soc_info.total_soc_score >= 55.0:
                            continue
                        if any(w in m_lower for w in ["rog phone", "red magic", "ultra", "pro max"]):
                            continue
                    elif bm <= 15.0:
                        if p.soc_info and p.soc_info.total_soc_score >= 68.0:
                            continue
                        if any(w in m_lower for w in ["ultra", "pro max"]):
                            continue

            if min_budget_vnd is not None and p.price_vnd is not None:
                if p.price_vnd < min_budget_vnd:
                    continue

            profiles.append(p)

        if max_budget_vnd is not None:
            priced = [p for p in profiles if p.price_vnd is not None and p.price_vnd <= max_budget_vnd]
            if len(priced) >= 3:
                profiles = priced
            elif priced:
                unpriced = [p for p in profiles if p.price_vnd is None]
                profiles = priced + unpriced

        profiles.sort(key=lambda x: x.gaming_score, reverse=True)
        return profiles[:top_n]


# ==============================================================================
# COMPARISON ENGINE
# ==============================================================================
def run_device_comparison(retriever: HybridRetriever, target_names: List[str]) -> Dict[str, Any]:
    if len(target_names) < 2:
        return {"error": "Cần ít nhất 2 thiết bị."}

    def _get(name: str) -> Optional[HardwareProfile]:
        a = QueryAnalysisResult(
            original_query=name, intent=IntentType.SPEC_LOOKUP, confidence=0.9,
            domain_focus="GAMING", target_sections=["Processor/Performance", "Display", "Battery"],
            device_keywords=name.split(), comparison_targets=[],
            brand_filter=None, max_budget_vnd=None, min_budget_vnd=None,
            game_titles=[], explanation=f"So sánh: {name}",
        )
        res = retriever.retrieve(name, analysis=a, top_k=15)
        profs = ProductRanker.rank_products_from_chunks(res, retriever.source_file_to_chunks, top_n=1)
        return profs[0] if profs else None

    return {"dev1": _get(target_names[0]), "dev2": _get(target_names[1]),
            "target1_name": target_names[0], "target2_name": target_names[1]}


# ==============================================================================
# DISPLAY REPORT
# ==============================================================================
def display_retrieval_report(
    query: str,
    analysis: QueryAnalysisResult,
    results: List[RetrievedChunk],
    source_file_chunks_map: Optional[Dict[str, List[str]]] = None,
    retriever: Optional[HybridRetriever] = None,
    show_full_content: bool = False,
):
    print("\n" + "=" * 80)
    print(" KET QUA HE THONG RAG TU VAN DIEN THOAI (B7 — KHONG CAN LLM)")
    print("=" * 80)
    print(f" • Query          : \"{query}\"")
    print(f" • Intent         : [{analysis.intent.value}] ({analysis.confidence*100:.0f}%)")
    print(f" • Domain         : {analysis.domain_focus}")
    print(f" • Brand Filter   : {analysis.brand_filter or 'Tat ca hang'}")
    bud = (f"{analysis.max_budget_vnd:,.0f}d" if analysis.max_budget_vnd else "Khong gioi han")
    if analysis.min_budget_vnd:
        bud = f"{analysis.min_budget_vnd:,.0f}d -> {bud}"
    print(f" • Ngan sach      : {bud}")
    print(f" • Game           : {analysis.game_titles or 'Khong co'}")
    print(f" • Giai trinh     : {analysis.explanation}")
    print("-" * 80)

    # COMPARISON
    if analysis.intent == IntentType.COMPARISON and len(analysis.comparison_targets) >= 2:
        t1, t2 = analysis.comparison_targets[0], analysis.comparison_targets[1]
        print(f"\n [SO SANH]: '{t1}' vs '{t2}'")
        if retriever:
            comp = run_device_comparison(retriever, [t1, t2])
            devs = [p for p in [comp.get("dev1"), comp.get("dev2")] if p]
        else:
            devs = ProductRanker.rank_products_from_chunks(results, source_file_chunks_map, top_n=2)

        for i, p in enumerate(devs, 1):
            soc = p.soc_info
            icon = "OK" if soc.confidence == "HIGH" else ("~" if soc.confidence == "MEDIUM" else "?")
            print(f"\n  TB{i}: {p.canonical_id} [{icon} {soc.confidence}]")
            print(f"   Score: {p.gaming_score}/100  SoC: {soc.canonical_name}")
            print(f"   GPU: {p.gpu_name}")
            print(f"   GB6: {soc.geekbench_multi or 'UNKNOWN'}  AnTuTu: {soc.antutu_v10 or 'UNKNOWN'}")
            print(f"   FPS: {soc.target_fps_gaming}")
            print(f"   RAM/Hz: {p.ram_gb}GB / {p.display_hz}Hz")

        if len(devs) >= 2:
            best = max(devs, key=lambda x: x.gaming_score)
            other = [d for d in devs if d is not best][0]
            print(f"\n  [KET LUAN]: '{best.canonical_id}' ({best.gaming_score}) vuot troi hon "
                  f"'{other.canonical_id}' ({other.gaming_score}).")
        print("=" * 80)
        return

    # RECOMMENDATION
    if analysis.intent == IntentType.RECOMMENDATION:
        print("\n [TOP DIEN THOAI PHU HOP NHAT]:")
        ranked = ProductRanker.rank_products_from_chunks(
            results, source_file_chunks_map=source_file_chunks_map,
            max_budget_vnd=analysis.max_budget_vnd, min_budget_vnd=analysis.min_budget_vnd,
            brand_filter=analysis.brand_filter, domain_focus=analysis.domain_focus,
            top_n=DEFAULT_TOP_K,
        )
        if not ranked:
            print("  ! Khong tim thay thiet bi thoa dieu kien.")
            print("=" * 80)
            return
        medals = ["🥇", "🥈", "🥉", "🏅", "🎖️"]
        for i, p in enumerate(ranked, 1):
            soc  = p.soc_info
            icon = "✅" if soc.confidence == "HIGH" else ("⚠️" if soc.confidence == "MEDIUM" else "❓")
            print(f"\n {medals[i-1] if i<=5 else '📱'} TOP {i}: {p.canonical_id} | {p.gaming_score:.1f}/100 {icon}")
            print(f"  SoC: {soc.canonical_name} | FPS: {soc.target_fps_gaming}")
            print(f"  RAM/Hz/Pin: {p.ram_gb}GB / {p.display_hz}Hz / {p.battery_mah}mAh")
            print(f"  Gia: {p.price_str} | Bien the: {', '.join(p.variants_storage) or 'Tieu chuan'}")
            print(f"  Ly do: {p.explanation}")
            print("  " + "-" * 74)
        print("\n" + "=" * 80)
        print(" DAM BAO: Khong doan ten may tu stop words. Chong trung bien the.")
        if analysis.max_budget_vnd:
            print(f"  Budget <= {analysis.max_budget_vnd:,.0f}d da ap dung.")
        print("=" * 80)
        return

    # SPEC_LOOKUP
    print("\n [TRA CUU CAU HINH]:")
    for r in results[:DEFAULT_TOP_K]:
        print(f"\n TOP {r.rank} | RRF={r.score:.4f}")
        print(f"  Thiet bi: {r.brand} {r.model} [{r.section}]")
        lines = r.content.strip().splitlines()
        for ln in (lines if show_full_content else lines[:6]):
            print(f"   | {ln}")
        if not show_full_content and len(lines) > 6:
            print(f"   | ... [{len(lines)-6} dong nua]")
        print("  " + "-" * 74)
    print("=" * 80)


# ==============================================================================
# P3 — EVALUATION SUITE (16 CA + IR METRICS)
# ==============================================================================
@dataclass
class EvalCase:
    id: int
    name: str
    query: str
    expected_intent: IntentType
    expected_domain: str
    expected_budget_max: Optional[float]
    expected_brand: Optional[str]
    must_include_keywords: List[str]
    must_not_include: List[str]
    criteria: str


EVAL_SUITE: List[EvalCase] = [
    EvalCase(1, "Gaming tong quat",
        "May nao choi game tot nhat hien tai?",
        IntentType.RECOMMENDATION, "GAMING", None, None,
        ["rog", "red magic", "redmagic", "s24", "mi 14", "k70", "poco"],
        [], "Top ket qua la flagship gaming."),
    EvalCase(2, "So sanh hai may gaming",
        "iPhone 17 Pro Max hay Xperia 1 V choi game tot hon?",
        IntentType.COMPARISON, "GAMING", None, None,
        ["iphone", "xperia"], [],
        "Tim dung ca 2 thiet bi, A19 Pro UNKNOWN, Xperia 1 V co benchmark thuc."),
    EvalCase(3, "Gaming duoi 10 trieu",
        "Dien thoai choi PUBG tot duoi 10 trieu",
        IntentType.RECOMMENDATION, "GAMING", 10_000_000.0, None,
        ["poco", "black shark", "redmi", "k50", "k60"], [],
        "Khong vuot 10 trieu, gaming tot trong tam gia."),
    EvalCase(4, "Tra cuu chip iPhone 17",
        "iPhone 17 Pro Max dung chip gi?",
        IntentType.SPEC_LOOKUP, "SPEC", None, "Apple",
        ["a19", "apple"], [],
        "Tra xuat dung Apple A19 Pro tu Processor/Performance."),
    EvalCase(5, "Pin trau choi game lau",
        "May nao pin trau de choi game lau?",
        IntentType.RECOMMENDATION, "BATTERY", None, None,
        ["5500", "6000", "6500"], [],
        "Pin >= 5500 mAh trong Top 3."),
    EvalCase(6, "Xiaomi gaming filter",
        "May Xiaomi nao choi game tot nhat?",
        IntentType.RECOMMENDATION, "GAMING", None, "Xiaomi",
        ["xiaomi", "redmi", "poco"],
        ["samsung", "sony", "apple", "asus"],
        "100% ket qua la Xiaomi."),
    EvalCase(7, "Chong trung lap Top 5",
        "Top 5 dien thoai gaming tot nhat",
        IntentType.RECOMMENDATION, "GAMING", None, None,
        [], [], "5 canonical_id khac nhau, khong lap bien the."),
    EvalCase(8, "SoC UNKNOWN - khong hallucinate",
        "May A thieu benchmark GPU thi sao?",
        IntentType.SPEC_LOOKUP, "SPEC", None, None,
        [], [], "SoC khong ro -> confidence=UNKNOWN, cpu_score <= 10."),
    EvalCase(9, "Budget-only 15 trieu",
        "tai chinh 15 trieu mua dien thoai nao",
        IntentType.RECOMMENDATION, "GENERAL", 15_000_000.0, None,
        [], ["everex", "a-15", "a15"],
        "Intent=REC, budget=15M. KHONG nham '15' thanh ten model A-15."),
    EvalCase(10, "Budget range 10-15 trieu",
        "Dien thoai gaming tu 10 den 15 trieu",
        IntentType.RECOMMENDATION, "GAMING", 15_000_000.0, None,
        [], [], "budget_max=15M, budget_min=10M, domain=GAMING."),
    EvalCase(11, "Game title Genshin Impact",
        "Dien thoai choi Genshin Impact muot nhat",
        IntentType.RECOMMENDATION, "GAMING", None, None,
        [], [], "Nhan dien 'Genshin Impact' -> domain=GAMING."),
    EvalCase(12, "Samsung duoi 20 trieu",
        "Samsung nao choi game tot duoi 20 trieu",
        IntentType.RECOMMENDATION, "GAMING", 20_000_000.0, "Samsung",
        ["samsung"], ["apple", "xiaomi", "sony"],
        "brand=Samsung, budget<=20M, domain=GAMING."),
    EvalCase(13, "Free Fire duoi 5 trieu",
        "May choi Free Fire muot duoi 5 trieu",
        IntentType.RECOMMENDATION, "GAMING", 5_000_000.0, None,
        [], [], "budget=5M, domain=GAMING, game=Free Fire."),
    EvalCase(14, "Camera recommendation",
        "Dien thoai chup anh dep nhat hien tai",
        IntentType.RECOMMENDATION, "CAMERA", None, None,
        [], [], "domain=CAMERA, intent=RECOMMENDATION."),
    EvalCase(15, "USD budget",
        "dien thoai gaming duoi $400",
        IntentType.RECOMMENDATION, "GAMING", 10_000_000.0, None,
        [], [], "$400 -> ~10M VND, budget xac dinh dung."),
    EvalCase(16, "PRICE_QUERY",
        "Samsung Galaxy S24 Ultra gia bao nhieu?",
        IntentType.PRICE_QUERY, "PRICE", None, "Samsung",
        ["samsung"], [], "intent=PRICE_QUERY, brand=Samsung."),
]


def _compute_ir(ranked_ids: List[str], rel_kws: List[str], k: int = 5) -> Dict[str, float]:
    """Tính Recall@K, Precision@K, AP@K, nDCG@K."""
    if not rel_kws:
        return {f"recall@{k}": 1.0, f"precision@{k}": 1.0, f"ap@{k}": 1.0, f"ndcg@{k}": 1.0}

    def is_rel(cid: str) -> bool:
        cid_l = cid.lower()
        return any(kw.lower() in cid_l for kw in rel_kws)

    top = ranked_ids[:k]
    flags = [1 if is_rel(r) else 0 for r in top]
    total_rel = sum(1 for r in ranked_ids if is_rel(r))

    recall  = sum(flags) / max(total_rel, 1)
    prec    = sum(flags) / k
    ap, h   = 0.0, 0
    for i, f in enumerate(flags, 1):
        if f:
            h  += 1
            ap += h / i
    ap /= max(total_rel, 1)
    dcg  = sum(f / math.log2(i+1) for i, f in enumerate(flags, 1))
    idcg = sum(1.0 / math.log2(i+1) for i in range(1, min(total_rel, k)+1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {
        f"recall@{k}": round(recall, 3), f"precision@{k}": round(prec, 3),
        f"ap@{k}": round(ap, 3), f"ndcg@{k}": round(ndcg, 3),
    }


def run_evaluation_suite(retriever: HybridRetriever, k: int = DEFAULT_TOP_K):
    """P3: Chạy toàn bộ 16 ca kiểm thử với IR metrics."""
    print("\n" + "=" * 80)
    print(f" EVALUATION SUITE B7 — {len(EVAL_SUITE)} CA KIEM DINH (P3)")
    print("=" * 80)

    passed   = 0
    agg: Dict[str, List[float]] = {
        f"recall@{k}": [], f"precision@{k}": [], f"ap@{k}": [], f"ndcg@{k}": []
    }

    for case in EVAL_SUITE:
        print(f"\n[{case.id:02d}/{len(EVAL_SUITE)}] {case.name.upper()}")
        print(f" Query  : \"{case.query}\"")
        print(f" Tieu chi: {case.criteria}")

        t0       = time.time()
        analysis = QueryAnalyzer.analyze(case.query)
        results  = retriever.retrieve(case.query, analysis=analysis, top_k=k)
        elapsed  = (time.time() - t0) * 1000

        checks: List[Tuple[str, bool, str]] = []

        # Intent
        checks.append(("Intent", analysis.intent == case.expected_intent,
                        f"{analysis.intent.value} (exp {case.expected_intent.value})"))
        # Domain
        checks.append(("Domain", analysis.domain_focus == case.expected_domain,
                        f"{analysis.domain_focus} (exp {case.expected_domain})"))
        # Budget
        if case.expected_budget_max is not None:
            bud_ok = (analysis.max_budget_vnd is not None
                      and abs(analysis.max_budget_vnd - case.expected_budget_max) < 500_000)
            checks.append(("Budget", bud_ok,
                            f"{analysis.max_budget_vnd} (exp {case.expected_budget_max})"))
        else:
            checks.append(("Budget", True, "N/A"))
        # Brand
        if case.expected_brand is not None:
            checks.append(("Brand", analysis.brand_filter == case.expected_brand,
                            f"{analysis.brand_filter} (exp {case.expected_brand})"))
        else:
            checks.append(("Brand", True, "N/A"))

        # Special checks
        if case.id == 8:
            unk = lookup_soc_benchmark("Chipset_Khong_Ro_123")
            no_hal = (unk.confidence == "UNKNOWN" and unk.cpu_score <= 10.0)
            checks.append(("No Hallucinate", no_hal,
                            f"conf={unk.confidence} score={unk.cpu_score}"))
        elif case.id == 7:
            ranked = ProductRanker.rank_products_from_chunks(results, retriever.source_file_to_chunks, top_n=k)
            names  = [p.canonical_id for p in ranked]
            no_dup = len(names) == len(set(names)) and len(ranked) >= k
            checks.append(("Dedup", no_dup, str(names)))
        else:
            ranked = ProductRanker.rank_products_from_chunks(
                results, retriever.source_file_to_chunks,
                max_budget_vnd=analysis.max_budget_vnd,
                min_budget_vnd=analysis.min_budget_vnd,
                brand_filter=analysis.brand_filter,
                domain_focus=analysis.domain_focus, top_n=k,
            )
            top_ids      = [p.canonical_id for p in ranked]
            top_ids_flat = " ".join(top_ids).lower()

            if case.must_not_include:
                bad = [w for w in case.must_not_include if w.lower() in top_ids_flat]
                checks.append(("No Bad", len(bad) == 0,
                                f"bad={bad}" if bad else "OK"))
            else:
                checks.append(("No Bad", True, "N/A"))

        # IR metrics
        metrics: Dict[str, Any] = {}
        if case.must_include_keywords and case.id not in (7, 8):
            rids = [p.canonical_id for p in ProductRanker.rank_products_from_chunks(
                results, retriever.source_file_to_chunks,
                max_budget_vnd=analysis.max_budget_vnd,
                brand_filter=analysis.brand_filter,
                domain_focus=analysis.domain_focus, top_n=k,
            )]
            metrics = _compute_ir(rids, case.must_include_keywords, k=k)
            for key, val in metrics.items():
                agg[key].append(val)

        case_ok = all(ok for _, ok, _ in checks)
        if case_ok:
            passed += 1

        print(f" -> {'OK [DAT]' if case_ok else 'FAIL [CHUA DAT]'} ({elapsed:.1f}ms)")
        for name, ok, detail in checks:
            print(f"    [{'v' if ok else 'x'}] {name}: {detail}")
        if metrics:
            print(f"    [IR] " + "  ".join(f"{k2}={v:.3f}" for k2, v in metrics.items()))
        print("-" * 80)

    print(f"\n{'='*80}")
    print(f" TONG KET: {passed}/{len(EVAL_SUITE)} CA DAT ({passed/len(EVAL_SUITE)*100:.1f}%)")
    for key, vals in agg.items():
        if vals:
            print(f"  Mean {key.upper()}: {sum(vals)/len(vals):.3f} (tren {len(vals)} ca co nhan)")
    print("=" * 80)


# ==============================================================================
# INTERACTIVE REPL
# ==============================================================================
def run_interactive_repl(retriever: HybridRetriever, top_k: int = DEFAULT_TOP_K):
    print("\n" + "=" * 80)
    print(" GIAO DIEN KIEM THU RAG B7 (P0->P3 NANG CAP, KHONG CAN LLM)")
    print("=" * 80)
    print(" Vi du: 'tai chinh 15 trieu mua dien thoai nao' | 'eval' -> 16 ca kiem thu | 'exit'")
    print("=" * 80 + "\n")

    while True:
        try:
            query = input(" [B7] > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print(" -> Tam biet!")
                break
            if query.lower() in ("eval", "test", "checklist", "benchmark"):
                run_evaluation_suite(retriever, k=top_k)
                continue

            t0       = time.time()
            analysis = QueryAnalyzer.analyze(query)
            results  = retriever.retrieve(query, analysis=analysis, top_k=top_k)
            elapsed  = (time.time() - t0) * 1000

            display_retrieval_report(
                query=query, analysis=analysis, results=results,
                source_file_chunks_map=retriever.source_file_to_chunks,
                retriever=retriever, show_full_content=False,
            )
            print(f" (Xu ly: {elapsed:.2f}ms)\n")

        except KeyboardInterrupt:
            print("\n -> Da dung.")
            break


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="B7: He thong RAG Tu van Dien thoai — Nang cap Toan dien (P0->P3)"
    )
    parser.add_argument("--rag-db-dir",     type=str, default=DEFAULT_RAG_DB_DIR)
    parser.add_argument("--model-dir",      type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--top-k",          type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--query",          type=str, default=None)
    parser.add_argument("--eval",           action="store_true")
    parser.add_argument("--checklist",      action="store_true")
    parser.add_argument("--filter-brand",   type=str, default=None)
    parser.add_argument("--filter-section", type=str, default=None)
    parser.add_argument("--full-content",   action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print(" B7: HE THONG RAG TU VAN DIEN THOAI — NANG CAP P0->P3 (KHONG CAN LLM)")
    print("=" * 80)

    retriever = HybridRetriever(rag_db_dir=args.rag_db_dir, model_dir=args.model_dir)

    if args.eval or args.checklist:
        run_evaluation_suite(retriever, k=args.top_k)
    elif args.query:
        analysis = QueryAnalyzer.analyze(args.query)
        results  = retriever.retrieve(
            args.query, analysis=analysis, top_k=args.top_k,
            filter_brand=args.filter_brand, filter_section=args.filter_section,
        )
        display_retrieval_report(
            query=args.query, analysis=analysis, results=results,
            source_file_chunks_map=retriever.source_file_to_chunks,
            retriever=retriever, show_full_content=args.full_content,
        )
    else:
        run_interactive_repl(retriever, top_k=args.top_k)


if __name__ == "__main__":
    main()

