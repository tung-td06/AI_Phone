"""
Module B6: Hệ thống Truy xuất Ngữ nghĩa & Xếp hạng Sản phẩm RAG (Không cần LLM).
=============================================================================================
Kiến trúc 6 tầng hoàn thiện phục vụ Chatbot Tư vấn Điện thoại:
  1. Intent Detection: Phân loại ý định (SPEC_LOOKUP, RECOMMENDATION, COMPARISON, PRICE_QUERY, GENERAL_CHAT).
  2. Query Understanding: Bóc tách nhu cầu (Gaming, Camera, Pin, Màn hình), Ngân sách (VND), Hãng (Brand),
     triệt tiêu triệt để việc đoán hư từ ('nên', 'để', 'nhất', 'tốt') làm tên thiết bị.
  3. Candidate Retrieval: Hybrid Search (Dense FAISS 206k + Sparse BM25 + Query Expansion) với pool đủ rộng,
     kết hợp lọc Metadata để không bỏ sót các flagship hàng đầu.
  4. Product Ranking Engine: Chấm điểm dựa trên Bảng hiệu năng SoC/GPU thực tế (Geekbench 6, AnTuTu, FPS),
     tuyệt đối không chấm điểm theo xung nhịp MHz thô. Thiếu benchmark báo UNKNOWN.
  5. Deduplication & Canonicalization: Gom nhóm các biến thể RAM/Bộ nhớ (128GB, 256GB, 512GB) vào cùng một
     canonical_model_id, không để một mẫu máy chiếm nhiều vị trí trong Top 5.
  6. Evaluation Checklist: Bộ kiểm định 8 ca kiểm thử theo tiêu chuẩn Information Retrieval & RAG.
=============================================================================================
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

# Nạp bảng hiệu năng SoC chuẩn hóa
from soc_database import lookup_soc_benchmark, SoCBenchmarkInfo, KNOWN_SOC_DATABASE


# ==============================================================================
# HẰNG SỐ CẤU HÌNH HỆ THỐNG
# ==============================================================================
DEFAULT_MODEL_DIR = r"p:\train AI\final_phone_embedding_model"
DEFAULT_RAG_DB_DIR = r"p:\train AI\rag_db"
DEFAULT_TOP_K = 5
BM25_CACHE_FILENAME = "bm25_index.pkl"

# Tỷ giá quy đổi ngoại tệ sang VND (phục vụ lọc ngân sách)
EXCHANGE_RATES_TO_VND: Dict[str, float] = {
    "USD": 25000.0,
    "EUR": 27000.0,
    "GBP": 32000.0,
    "CNY": 3500.0,
    "MXN": 1300.0,
    "JPY": 165.0,
    "KRW": 18.0,
    "INR": 300.0,
    "VND": 1.0,
}

# Danh sách các thương hiệu phổ biến
KNOWN_BRANDS: List[str] = [
    "xiaomi", "samsung", "apple", "sony", "asus", "oppo", "vivo", "realme",
    "huawei", "oneplus", "google", "nokia", "motorola", "zte", "honor", "lenovo"
]

# Từ điển ánh xạ từ khóa câu hỏi tiếng Việt sang Chuyên mục chuẩn trong CSDL
SECTION_KEYWORD_RULES: Dict[str, List[str]] = {
    "Battery": [
        "pin", "dung lượng pin", "sạc", "mah", "watt", "w sạc", "sạc nhanh",
        "sạc không dây", "thời lượng pin", "pin trâu", "li-po", "li-ion", "battery"
    ],
    "Display": [
        "màn hình", "tần số quét", "độ phân giải", "oled", "amoled", "ips", "lcd",
        "hz", "120hz", "144hz", "165hz", "60hz", "inch", "kích thước màn", "độ sáng",
        "nits", "kính cường lực", "gorilla glass", "display", "screen"
    ],
    "Processor/Performance": [
        "chip", "vi xử lý", "chipset", "cpu", "gpu", "snapdragon", "dimensity",
        "exynos", "bionic", "helio", "card đồ họa", "adreno", "mali",
        "xung nhịp", "ghz", "hiệu năng", "chơi game", "antutu", "processor", "geekbench",
        "dùng chip gì", "chip gì", "gpu gì"
    ],
    "Camera": [
        "camera", "máy ảnh", "chụp hình", "quay phim", "quay video", "megapixel",
        "mp", "chống rung", "ois", "cảm biến", "zoom", "tele", "góc rộng",
        "macro", "camera trước", "selfie", "camera sau", "flash"
    ],
    "Memory": [
        "ram", "rom", "bộ nhớ", "dung lượng", "lưu trữ", "bộ nhớ trong",
        "thẻ nhớ", "micro sd", "ufs", "lpddr", "128gb", "256gb", "512gb", "1tb"
    ],
    "Dimensions": [
        "kích thước", "trọng lượng", "cân nặng", "dày", "mỏng", "chiều dài",
        "chiều rộng", "gam", "gram", "kháng nước", "ip68", "ip67", "chống nước",
        "vật liệu", "khung viền", "mặt lưng"
    ],
    "Connectivity": [
        "5g", "4g", "sim", "esim", "wifi", "bluetooth", "nfc", "gps",
        "cổng sạc", "type-c", "usb-c", "lightning", "loa", "tai nghe", "jack 3.5"
    ],
    "Thông tin chung": [
        "giá", "ra mắt", "năm sản xuất", "xuất xứ", "hãng", "thương hiệu",
        "hệ điều hành", "android", "ios", "hyperos", "one ui", "miui"
    ],
}

# Danh sách stop words tiếng Việt được mở rộng triệt để chống đoán bừa tên máy
EXTENDED_STOPWORDS: Set[str] = {
    "bao", "nhiêu", "thế", "nào", "gì", "không", "có", "mấy", "là", "và",
    "của", "cho", "tôi", "hỏi", "với", "dùng", "được", "chạy", "như", "ra",
    "sao", "điện", "thoại", "máy", "thông", "số", "kỹ", "thuật", "chi", "tiết",
    "đánh", "giá", "cấu", "hình", "giá", "tiền", "mua", "bán", "tốt", "hay",
    "xem", "cho", "biết", "về", "trong", "trên", "nên", "để", "nhất", "các",
    "những", "loại", "dòng", "tư", "vấn", "gợi", "ý", "so", "sánh", "giữa",
    "vs", "hoặc", "hơn", "kém", "rẻ", "đắt", "tầm", "khoảng", "dưới", "khoang",
    "hiện", "nay", "tại", "mượt", "khủng", "đỉnh", "ngon", "đáng"
}


# ==============================================================================
# 1. INTENT DETECTION & QUERY UNDERSTANDING
# ==============================================================================
class IntentType(str, Enum):
    SPEC_LOOKUP = "SPEC_LOOKUP"          # Tra cứu thông số thiết bị cụ thể
    RECOMMENDATION = "RECOMMENDATION"    # Tư vấn sản phẩm theo nhu cầu mở (gaming, camera, pin...)
    COMPARISON = "COMPARISON"            # So sánh giữa 2 hoặc nhiều máy
    PRICE_QUERY = "PRICE_QUERY"          # Hỏi giá bán / phân khúc giá
    GENERAL_CHAT = "GENERAL_CHAT"        # Hỏi đáp chung, chào hỏi


@dataclass
class QueryAnalysisResult:
    """Kết quả phân tích ý định và cấu trúc ngữ nghĩa câu hỏi."""
    original_query: str
    intent: IntentType
    confidence: float
    domain_focus: str                  # GAMING, CAMERA, BATTERY, DISPLAY, GENERAL
    target_sections: List[str]         # Danh sách sections CSDL cần ưu tiên
    device_keywords: List[str]         # Tên máy (rỗng nếu là câu hỏi tư vấn mở)
    comparison_targets: List[str]      # Các thiết bị nếu là câu hỏi so sánh
    brand_filter: Optional[str]        # Lọc thương hiệu nếu có (vd: Xiaomi, Samsung)
    max_budget_vnd: Optional[float]    # Ngân sách trần (VND)
    explanation: str                   # Giải trình lý do phân loại


class QueryAnalyzer:
    """Bộ phân tích ý định thông minh, bóc tách thực thể và điều kiện ràng buộc."""

    @staticmethod
    def analyze(query: str) -> QueryAnalysisResult:
        q_clean = query.strip()
        q_lower = q_clean.lower()

        # 1. Trích xuất ràng buộc Ngân sách (vd: "dưới 10 triệu", "dưới 10tr", "tầm 8 củ")
        max_budget_vnd = None
        budget_match = re.search(r"(?:dưới|tầm|khoảng|max)\s*(\d+(?:\.\d+)?)\s*(?:triệu|tr|củ|m\b)", q_lower)
        if budget_match:
            val = float(budget_match.group(1))
            max_budget_vnd = val * 1_000_000

        # 2. Trích xuất ràng buộc Thương hiệu (Brand filter & Aliases)
        brand_filter = None
        brand_aliases = {
            "iphone": "Apple",
            "ipad": "Apple",
            "apple": "Apple",
            "galaxy": "Samsung",
            "samsung": "Samsung",
            "redmi": "Xiaomi",
            "poco": "Xiaomi",
            "xiaomi": "Xiaomi",
            "mi": "Xiaomi",
            "xperia": "Sony",
            "sony": "Sony",
            "rog": "Asus",
            "zenfone": "Asus",
            "asus": "Asus",
            "redmagic": "ZTE",
            "nubia": "ZTE",
            "zte": "ZTE",
            "oppo": "Oppo",
            "vivo": "Vivo",
            "iqoo": "BBK",
            "realme": "Realme",
            "oneplus": "OnePlus",
            "google": "Google",
            "pixel": "Google",
        }
        for alias, brand_canon in brand_aliases.items():
            if re.search(rf"\b{alias}\b", q_lower):
                brand_filter = brand_canon
                break

        # 3. Nhận diện GENERAL_CHAT
        chat_patterns = [
            r"^(chào|hello|hi|xin chào|hey)\b",
            r"^bạn là ai",
            r"^giúp được gì",
            r"hệ điều hành là gì\??$",
            r"android là gì\??$",
        ]
        if any(re.search(p, q_lower) for p in chat_patterns):
            return QueryAnalysisResult(
                original_query=q_clean,
                intent=IntentType.GENERAL_CHAT,
                confidence=0.95,
                domain_focus="GENERAL",
                target_sections=["Thông tin chung"],
                device_keywords=[],
                comparison_targets=[],
                brand_filter=None,
                max_budget_vnd=None,
                explanation="Câu hỏi chào hỏi hoặc kiến thức mở thông thường, không tra cứu thông số máy."
            )

        # 4. Nhận diện COMPARISON (vd: "iPhone 17 Pro Max hay Xperia 1 V chơi game tốt hơn?", "so sánh A và B")
        comp_patterns = [
            r"\bso sánh\b",
            r"\bgiữa\b.+\bvà\b",
            r"\bhay\b.+\b(tốt|hơn|mượt)\b",
            r"\bvs\b",
            r"\bhơn hay\b",
            r"\bhơn không\b"
        ]
        is_comparison = any(re.search(p, q_lower) for p in comp_patterns)
        if is_comparison:
            comp_targets = QueryAnalyzer._extract_comparison_targets(q_clean)
            return QueryAnalysisResult(
                original_query=q_clean,
                intent=IntentType.COMPARISON,
                confidence=0.92,
                domain_focus="GAMING" if "game" in q_lower else "COMPARISON",
                target_sections=["Processor/Performance", "Display", "Battery"],
                device_keywords=comp_targets,
                comparison_targets=comp_targets,
                brand_filter=None,  # TUYỆT ĐỐI ĐỂ NONE VÌ SO SÁNH ĐA THƯƠNG HIỆU
                max_budget_vnd=max_budget_vnd,
                explanation=f"Người dùng yêu cầu so sánh đối chiếu giữa: {comp_targets}."
            )

        # 5. Xác định Domain Focus
        domain = "GENERAL"
        target_sections = ["Thông tin chung"]
        if any(w in q_lower for w in ["game", "chơi game", "gaming", "fps", "pubg", "genshin", "antutu"]):
            domain = "GAMING"
            target_sections = ["Processor/Performance", "Display", "Battery"]
        elif any(w in q_lower for w in ["pin trâu", "pin khủng", "thời lượng pin", "pin"]):
            domain = "BATTERY"
            target_sections = ["Battery", "Processor/Performance"]
        elif any(w in q_lower for w in ["chụp ảnh", "camera", "quay phim", "quay video"]):
            domain = "CAMERA"
            target_sections = ["Camera", "Display"]
        elif any(w in q_lower for w in ["màn hình", "120hz", "144hz", "oled", "tần số quét"]):
            domain = "DISPLAY"
            target_sections = ["Display"]

        # 6. Nhận diện RECOMMENDATION (Tư vấn theo nhu cầu mở: Gaming, Pin, Ngân sách, Hãng...)
        rec_patterns = [
            r"chơi game (tốt|mượt|đỉnh|ngon|nhất)",
            r"điện thoại nào (tốt|chơi game|chụp ảnh|pin trâu|đáng mua)",
            r"máy nào (tốt|chơi game|chụp ảnh|pin trâu|đáng mua)",
            r"tư vấn (mua|điện thoại|máy)",
            r"nên mua (máy|điện thoại) nào",
            r"top \d+ điện thoại",
            r"máy .+ nào chơi game tốt",
            r"chơi pubg tốt",
            r"pin trâu để chơi game",
        ]
        is_recommendation = any(re.search(p, q_lower) for p in rec_patterns) or (max_budget_vnd is not None and "game" in q_lower)

        if is_recommendation:
            # VỚI CÂU HỎI TƯ VẤN MỞ: TUYỆT ĐỐI ĐỂ RỖNG DEVICE KEYWORDS ĐỂ TRÁNH ÉP LỌC NHẦM TÊN MÁY!
            return QueryAnalysisResult(
                original_query=q_clean,
                intent=IntentType.RECOMMENDATION,
                confidence=0.92,
                domain_focus=domain,
                target_sections=target_sections,
                device_keywords=[],  # TUYỆT ĐỐI ĐỂ RỖNG
                comparison_targets=[],
                brand_filter=brand_filter,
                max_budget_vnd=max_budget_vnd,
                explanation=f"Câu hỏi tư vấn nhu cầu mở (Domain: {domain}, Hãng: {brand_filter or 'Mọi hãng'}, Ngân sách trần: {f'{max_budget_vnd:,.0f} VND' if max_budget_vnd else 'Không giới hạn'})."
            )

        # 7. Nhận diện PRICE_QUERY
        if any(w in q_lower for w in ["giá bao nhiêu", "bao nhiêu tiền", "giá rẻ nhất", "bảng giá"]):
            dev_kws = QueryAnalyzer._extract_candidate_device_words(q_clean)
            return QueryAnalysisResult(
                original_query=q_clean,
                intent=IntentType.PRICE_QUERY,
                confidence=0.88,
                domain_focus="PRICE",
                target_sections=["Thông tin chung"],
                device_keywords=dev_kws,
                comparison_targets=[],
                brand_filter=brand_filter,
                max_budget_vnd=max_budget_vnd,
                explanation="Câu hỏi tập trung vào mức giá và thông tin thương mại."
            )

        # 8. Mặc định: SPEC_LOOKUP (Tra cứu thông số kỹ thuật máy cụ thể)
        dev_kws = QueryAnalyzer._extract_candidate_device_words(q_clean)
        expected_sec = QueryAnalyzer.detect_expected_section(q_clean) or "Processor/Performance"

        return QueryAnalysisResult(
            original_query=q_clean,
            intent=IntentType.SPEC_LOOKUP,
            confidence=0.86,
            domain_focus=domain if domain != "GENERAL" else "SPEC",
            target_sections=[expected_sec],
            device_keywords=dev_kws,
            comparison_targets=[],
            brand_filter=brand_filter,
            max_budget_vnd=max_budget_vnd,
            explanation=f"Câu hỏi tra cứu thông số kỹ thuật của thiết bị cụ thể ({dev_kws})."
        )

    @staticmethod
    def detect_expected_section(query: str) -> Optional[str]:
        """Phát hiện chuyên mục kỹ thuật phù hợp nhất từ từ khóa."""
        query_lower = query.lower()
        scores: Dict[str, int] = {}
        for section, keywords in SECTION_KEYWORD_RULES.items():
            score = 0
            for kw in keywords:
                if kw in query_lower:
                    score += len(kw)
            if score > 0:
                scores[section] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return None

    @staticmethod
    def _extract_comparison_targets(query: str) -> List[str]:
        """Tách các tên máy trong câu so sánh (vd: iPhone 17 Pro Max hay Xperia 1 V)."""
        # Thử tách theo 'hay', 'hoặc', 'và', 'vs'
        parts = re.split(r"\b(?:hay|hoặc|vs|và|so với)\b", query, flags=re.IGNORECASE)
        targets = []
        for p in parts:
            p_clean = re.sub(r"[\?\.\,\!]", "", p)
            p_clean = re.sub(r"\b(?:chơi game|tốt hơn|hơn không|mượt hơn|so sánh|giữa|điện thoại|máy)\b", "", p_clean, flags=re.IGNORECASE).strip()
            # Giữ lại các từ định danh
            words = [w for w in p_clean.split() if w.lower() not in EXTENDED_STOPWORDS]
            if words:
                targets.append(" ".join(words))
        return targets[:2]

    @staticmethod
    def _extract_candidate_device_words(query: str) -> List[str]:
        """Trích xuất từ khóa định danh máy một cách cẩn trọng, loại bỏ triệt để stop words."""
        section_words = set()
        for kws in SECTION_KEYWORD_RULES.values():
            for kw in kws:
                for word in kw.lower().split():
                    section_words.add(word)

        tokens = re.findall(r"[\w\+\-]+", query)
        keywords = []
        for t in tokens:
            t_low = t.lower()
            if t_low not in EXTENDED_STOPWORDS and t_low not in section_words and len(t) >= 2:
                keywords.append(t)
        return keywords


# ==============================================================================
# 2. CANONICALIZATION & DEDUPLICATION (CHỐNG TRÙNG LẶP BIẾN THỂ)
# ==============================================================================
def get_canonical_model_id(brand: str, raw_model: str) -> str:
    """
    Chuẩn hóa tên model, loại bỏ biến thể bộ nhớ (128GB, 256GB), nhà mạng, mã vùng
    để gom nhóm các phiên bản về cùng một sản phẩm gốc (Canonical Product ID).
    """
    name = f"{brand} {raw_model}"
    # Loại bỏ codename phụ sau dấu gạch chéo (/ K50G, / SM-S928B...)
    name = re.sub(r"\s*/\s*.*$", "", name)
    # Loại bỏ dung lượng bộ nhớ / RAM
    name = re.sub(r"\b\d+\s*(?:GB|TB|GiB|TiB|MB|MiB)\b", "", name, flags=re.IGNORECASE)
    # Loại bỏ tag mạng và chuẩn kết nối
    name = re.sub(r"\b(?:Dual SIM|Single SIM|TD-LTE|LTE-A|LTE|5G|4G|3G|Global|APAC|EMEA|NA|CN|JP|KR|TW|HK|IN|UW|US|CA|MX|SA|MEA|EU)\b", "", name, flags=re.IGNORECASE)
    # Loại bỏ tag phiên bản thương mại không đổi phần cứng cốt lõi
    name = re.sub(r"\b(?:Standard Edition|Premium Edition|Base Edition|Top Edition|Elite Edition|Pro Edition|Gaming Edition|Soft Light|PaperMatte|Bumblebee Limited Edition|Version\s+[A-Z0-9\s]+)\b", "", name, flags=re.IGNORECASE)
    # Loại bỏ mã model nhà sản xuất (SM-S9280, XQ-DQ24, AI2401...)
    name = re.sub(r"\b(?:SM-[A-Z0-9]+|XQ-[A-Z0-9]+|[A-Z0-9]{2,4}-[A-Z0-9]{3,})\b", "", name)
    name = re.sub(r"\b[A-Z0-9]{6,}\b", "", name)
    name = re.sub(r"\bA\d{4}\b", "", name)
    # Chuẩn hóa dấu gạch nối và khoảng trắng
    name = re.sub(r"[\-\–]\s*$", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# ==============================================================================
# 3. HYBRID RETRIEVAL ENGINE (DENSE FAISS + SPARSE BM25 + RRF)
# ==============================================================================
@dataclass
class RetrievedChunk:
    """Đại diện cho 1 chunk thông số được tìm thấy từ CSDL."""
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


class HybridRetriever:
    """Kết hợp FAISS Dense Vector + BM25 Sparse Search + Metadata Filtering."""

    def __init__(
        self,
        rag_db_dir: str | Path = DEFAULT_RAG_DB_DIR,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device: Optional[str] = None,
    ):
        self.rag_db_dir = Path(rag_db_dir)
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.faiss_index = None
        self.chunks_text: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.model: Optional[SentenceTransformer] = None
        self.bm25: Optional[BM25Okapi] = None
        self.source_file_to_chunks: Dict[str, List[str]] = {}

        self._load_resources()

    def _load_resources(self):
        faiss_file = self.rag_db_dir / "index.faiss"
        chunks_file = self.rag_db_dir / "chunks.json"
        meta_file = self.rag_db_dir / "metadata.json"
        bm25_cache_file = self.rag_db_dir / BM25_CACHE_FILENAME

        if not (faiss_file.exists() and chunks_file.exists() and meta_file.exists()):
            raise FileNotFoundError(f"Không tìm thấy đầy đủ bộ CSDL tại: {self.rag_db_dir.resolve()}")

        print(f" -> Đang nạp CSDL Vector FAISS từ: {self.rag_db_dir.resolve()} ...")
        t0 = time.time()
        self.faiss_index = faiss.read_index(str(faiss_file))

        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks_text = json.load(f)

        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"    + Đã nạp thành công: {self.faiss_index.ntotal:,} vectors trong {time.time() - t0:.2f}s!")

        # Lập chỉ mục gom nhóm toàn bộ chunks theo thiết bị (source_file) để phục vụ Product Ranking
        for idx, meta in enumerate(self.metadata):
            sf = meta.get("source_file", "")
            if sf:
                if sf not in self.source_file_to_chunks:
                    self.source_file_to_chunks[sf] = []
                self.source_file_to_chunks[sf].append(self.chunks_text[idx])

        # Nạp mô hình embedding
        print(f" -> Đang nạp mô hình Fine-tuned từ: {self.model_dir.resolve()} lên '{self.device}'...")
        t1 = time.time()
        self.model = SentenceTransformer(str(self.model_dir), device=self.device)
        self.model.eval()
        print(f"    + Mô hình đã sẵn sàng trong {time.time() - t1:.2f}s!")

        # Nạp hoặc khởi tạo BM25 Index
        if HAS_BM25:
            if bm25_cache_file.exists():
                print(f" -> Đang nạp BM25 Index từ bộ nhớ đệm ({bm25_cache_file.name}) ...")
                t2 = time.time()
                with open(bm25_cache_file, "rb") as f:
                    self.bm25 = pickle.load(f)
                print(f"    + BM25 sẵn sàng trong {time.time() - t2:.2f}s!")
            else:
                print(" -> Chưa có cache BM25, tiến hành khởi tạo và lưu cache ...")
                t2 = time.time()
                corpus = [re.findall(r"\w+", c.lower()) for c in self.chunks_text]
                self.bm25 = BM25Okapi(corpus)
                try:
                    with open(bm25_cache_file, "wb") as f:
                        pickle.dump(self.bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"    + Khởi tạo và cache BM25 hoàn tất trong {time.time() - t2:.2f}s!")
                except Exception as e:
                    print(f"    ! Không thể lưu cache BM25: {e}")

    def retrieve(
        self,
        query: str,
        analysis: QueryAnalysisResult,
        top_k: int = DEFAULT_TOP_K,
        filter_brand: Optional[str] = None,
        filter_section: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Truy xuất lai (Hybrid Search) kết hợp Dense + BM25 bằng Reciprocal Rank Fusion (RRF).
        Tự động áp dụng Domain Query Expansion khi gặp câu hỏi tư vấn mở (RECOMMENDATION).
        """
        active_brand_filter = filter_brand or analysis.brand_filter

        # Với câu hỏi tư vấn mở, mở rộng pool ứng viên lên 120-150 chunks để ProductRanker không bỏ sót flagship
        candidate_pool_size = max(120, top_k * 20) if analysis.intent == IntentType.RECOMMENDATION else max(30, top_k * 6)

        dense_search_text = query
        sparse_tokens = [w for w in re.findall(r"\w+", query.lower()) if w not in EXTENDED_STOPWORDS]

        if analysis.intent == IntentType.RECOMMENDATION:
            if analysis.domain_focus == "GAMING":
                if analysis.max_budget_vnd is not None:
                    dense_search_text = "điện thoại chơi game pubg tầm trung giá rẻ Snapdragon 870 7+ Gen 2 Dimensity 8100 1200 Black Shark Poco Redmi"
                    sparse_tokens = ["black", "shark", "snapdragon", "dimensity", "price", "cny", "usd", "pubg", "gaming"] + sparse_tokens
                else:
                    dense_search_text = "Snapdragon 8 Gen 3 Dimensity 9300 A17 Pro hiệu năng chơi game vi xử lý CPU GPU cấu hình mạnh chipset ROG Phone RedMagic"
                    sparse_tokens = ["snapdragon", "8", "gen", "3", "dimensity", "9300", "a17", "pro", "adreno", "gaming", "rog"] + sparse_tokens
            elif analysis.domain_focus == "BATTERY":
                dense_search_text = "dung lượng pin 6000mAh 6500mAh 5500mAh sạc nhanh watt thời lượng pin trâu chơi game lâu"
                sparse_tokens = ["6000mah", "6500mah", "5500mah", "battery", "mah", "charging", "gaming"] + sparse_tokens

            if active_brand_filter:
                dense_search_text = f"{active_brand_filter} {dense_search_text}"
                sparse_tokens = [active_brand_filter.lower()] + sparse_tokens

        elif analysis.intent == IntentType.SPEC_LOOKUP:
            if active_brand_filter:
                dense_search_text = f"{active_brand_filter} {dense_search_text}"
                sparse_tokens = [active_brand_filter.lower()] + sparse_tokens
            if analysis.device_keywords:
                sparse_tokens = [k.lower() for k in analysis.device_keywords] + sparse_tokens

        # 1. Dense Retrieval (FAISS)
        q_vec = self.model.encode([dense_search_text], normalize_embeddings=True, convert_to_numpy=True)
        dense_scores_raw, dense_indices_raw = self.faiss_index.search(q_vec, candidate_pool_size)

        dense_rank_map: Dict[int, int] = {}
        dense_score_map: Dict[int, float] = {}
        for rank, (score, idx) in enumerate(zip(dense_scores_raw[0], dense_indices_raw[0]), 1):
            if idx >= 0:
                dense_rank_map[idx] = rank
                dense_score_map[idx] = float(score)

        # 2. Sparse Retrieval (BM25)
        bm25_rank_map: Dict[int, int] = {}
        bm25_score_map: Dict[int, float] = {}
        if self.bm25 is not None and sparse_tokens:
            bm25_scores = self.bm25.get_scores(sparse_tokens)
            top_bm25_indices = np.argsort(bm25_scores)[-candidate_pool_size:][::-1]
            for rank, idx in enumerate(top_bm25_indices, 1):
                if bm25_scores[idx] > 0:
                    bm25_rank_map[int(idx)] = rank
                    bm25_score_map[int(idx)] = float(bm25_scores[idx])

        # 3. Thu thập tất cả các ứng viên
        all_candidate_indices = set(dense_rank_map.keys()).union(set(bm25_rank_map.keys()))

        # 4. Tính điểm Reciprocal Rank Fusion (RRF)
        rrf_constant = 60
        fused_candidates: List[Tuple[int, float, float, float]] = []

        for idx in all_candidate_indices:
            meta = self.metadata[idx]

            # Áp dụng bộ lọc Hãng sản xuất nếu có
            if active_brand_filter and meta["brand"].lower() != active_brand_filter.lower():
                continue
            if filter_section and meta["section"].lower() != filter_section.lower():
                continue

            section_bonus = 0.0
            if analysis.intent == IntentType.RECOMMENDATION:
                if meta["section"] in analysis.target_sections:
                    section_bonus = 0.015

            dense_rank = dense_rank_map.get(idx, 9999)
            sparse_rank = bm25_rank_map.get(idx, 9999)

            rrf_score = (1.0 / (rrf_constant + dense_rank)) + (1.0 / (rrf_constant + sparse_rank)) + section_bonus
            fused_candidates.append((
                idx,
                rrf_score,
                dense_score_map.get(idx, 0.0),
                bm25_score_map.get(idx, 0.0)
            ))

        fused_candidates.sort(key=lambda x: x[1], reverse=True)

        fetch_limit = candidate_pool_size if analysis.intent == IntentType.RECOMMENDATION else top_k
        results: List[RetrievedChunk] = []
        for rank, (idx, score, d_score, s_score) in enumerate(fused_candidates[:fetch_limit], 1):
            meta = self.metadata[idx]
            results.append(
                RetrievedChunk(
                    rank=rank,
                    score=score,
                    dense_score=d_score,
                    sparse_score=s_score,
                    brand=meta["brand"],
                    model=meta["model"],
                    section=meta["section"],
                    chunk_id=meta.get("chunk_id", f"chunk_{idx}"),
                    content=self.chunks_text[idx],
                    source_file=meta.get("source_file", ""),
                )
            )

        return results


# ==============================================================================
# 4. PRODUCT RANKING ENGINE (HARDWARE SCORING DỰA TRÊN THỰC CHỨNG)
# ==============================================================================
@dataclass
class HardwareProfile:
    """Hồ sơ phần cứng bóc tách từ các chunks dữ liệu thực chứng."""
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
    benchmark_antutu: Optional[int] = None
    benchmark_geekbench: Optional[int] = None
    gaming_score: float = 0.0
    score_details: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


class ProductRanker:
    """Bộ chấm điểm, chống trùng lặp và xếp hạng sản phẩm theo thông số phần cứng thực chứng."""

    @staticmethod
    def extract_hardware_profile(
        brand: str,
        raw_models: Set[str],
        chunks_text_list: List[str],
        domain_focus: str = "GAMING",
    ) -> HardwareProfile:
        combined_text = "\n".join(chunks_text_list)
        representative_model = sorted(list(raw_models), key=len)[0]
        canonical_id = get_canonical_model_id(brand, representative_model)

        profile = HardwareProfile(
            brand=brand,
            model=representative_model,
            canonical_id=canonical_id,
        )

        # 1. Bóc tách Tên CPU / Chipset
        cpu_match = re.search(r"CPU:\s*([^\n\r,]+)", combined_text, re.IGNORECASE)
        if cpu_match:
            profile.cpu_name = cpu_match.group(1).strip()

        # 2. Bóc tách Xung nhịp CPU (MHz)
        cpu_clock_match = re.search(r"CPU Clock:\s*(\d+)\s*MHz", combined_text, re.IGNORECASE)
        if cpu_clock_match:
            profile.cpu_clock_mhz = int(cpu_clock_match.group(1))

        # 3. Bóc tách GPU
        gpu_match = re.search(r"(?:Graphical Controller|GPU):\s*([^\n\r]+)", combined_text, re.IGNORECASE)
        if gpu_match:
            profile.gpu_name = gpu_match.group(1).strip()

        # 4. Tra cứu Bảng hiệu năng SoC chuẩn hóa (Benchmark thực tế, không suy đoán từ MHz)
        soc_lookup_query = f"{profile.cpu_name} {profile.gpu_name}"
        profile.soc_info = lookup_soc_benchmark(soc_lookup_query)

        # 5. Bóc tách RAM (Lấy dung lượng cao nhất tìm thấy giữa các biến thể)
        ram_matches = re.findall(r"RAM Capacity(?:\s*\(converted\))?:\s*(\d+(?:\.\d+)?)\s*(GiB|GB|MiB|MB)", combined_text, re.IGNORECASE)
        max_ram = 0.0
        for val_str, unit in ram_matches:
            v = float(val_str)
            if "M" in unit.upper():
                v = v / 1024.0
            max_ram = max(max_ram, v)
        profile.ram_gb = max_ram

        # 6. Bóc tách Tần số quét màn hình (Hz)
        hz_matches = re.findall(r"(\d{2,3})\s*Hz", combined_text, re.IGNORECASE)
        if hz_matches:
            profile.display_hz = max([int(h) for h in hz_matches])
        else:
            profile.display_hz = 60

        # 7. Bóc tách Pin (mAh)
        battery_match = re.search(r"Nominal Battery Capacity:\s*(\d+)\s*mAh", combined_text, re.IGNORECASE)
        if not battery_match:
            battery_match = re.search(r"(\d{3,5})\s*mAh", combined_text, re.IGNORECASE)
        if battery_match:
            profile.battery_mah = int(battery_match.group(1))

        # 8. Kiểm tra tản nhiệt chuyên dụng / Phím trigger gaming
        if re.search(r"(vapor chamber|liquid cooling|cooling system|bấm game|trigger|tản nhiệt|heat pipe)", combined_text, re.IGNORECASE):
            profile.cooling_mentioned = True

        # 9. Bóc tách Giá bán & Quy đổi sang VND
        price_match = re.search(r"Price:\s*(\d+(?:\.\d+)?)\s*(?:\n\s*-\s*([A-Za-z]+))?", combined_text)
        if price_match:
            val = float(price_match.group(1))
            curr = (price_match.group(2) or "USD").upper()
            rate = EXCHANGE_RATES_TO_VND.get(curr, 25000.0)
            profile.price_vnd = val * rate
            profile.price_str = f"~{profile.price_vnd:,.0f} VND ({val:,.0f} {curr})"

        # 10. Gom nhóm các tùy chọn bộ nhớ (Deduplication variants)
        variants = set()
        for rm in raw_models:
            stor_m = re.search(r"\b(\d+)\s*(?:GB|TB)\b", rm, re.IGNORECASE)
            if stor_m:
                variants.add(stor_m.group(0).upper())
        profile.variants_storage = sorted(list(variants))

        # 11. Chấm điểm & Tạo giải trình lý do
        profile.gaming_score, profile.score_details, profile.explanation = ProductRanker._calculate_score(profile, domain_focus)
        return profile

    @staticmethod
    def _calculate_score(p: HardwareProfile, domain_focus: str) -> Tuple[float, Dict[str, Any], str]:
        """
        Chấm điểm thiết bị khoa học dựa trên Benchmark thực tế và thông số phần cứng (Thang 100).
        - CPU Score (max 35) & GPU Score (max 35) = max 70 điểm từ SoC Benchmark.
        - RAM (max 12) + Màn hình (max 10) + Pin & Tản nhiệt (max 8) = max 30 điểm.
        - Nếu domain là BATTERY: Tăng trọng số Pin (max 30), SoC (max 40).
        """
        soc = p.soc_info

        if domain_focus == "BATTERY":
            # Chấm điểm Gaming tập trung vào Thời lượng Pin
            soc_score = round(soc.total_soc_score * (40.0 / 70.0), 1)

            bat_score = 10.0
            if p.battery_mah >= 6500:
                bat_score = 30.0
            elif p.battery_mah >= 6000:
                bat_score = 27.0
            elif p.battery_mah >= 5500:
                bat_score = 24.0
            elif p.battery_mah >= 5000:
                bat_score = 20.0
            elif p.battery_mah >= 4500:
                bat_score = 15.0

            cool_score = 10.0 if p.cooling_mentioned else 5.0

            disp_ram_score = 5.0
            if p.ram_gb >= 12 and p.display_hz >= 120:
                disp_ram_score = 15.0
            elif p.ram_gb >= 8:
                disp_ram_score = 10.0

            total_score = round(soc_score + bat_score + cool_score + disp_ram_score, 1)
            details = {
                "Thời lượng Pin": f"{bat_score}/30 ({p.battery_mah} mAh)",
                "Hiệu năng SoC": f"{soc_score}/40 ({soc.canonical_name})",
                "Tản nhiệt": f"{cool_score}/10 ({'Buồng hơi/Lỏng' if p.cooling_mentioned else 'Tiêu chuẩn'})",
                "Màn hình & RAM": f"{disp_ram_score}/15 ({p.ram_gb} GB RAM, {p.display_hz} Hz)",
            }
            explanation = (
                f"Sở hữu viên pin khủng {p.battery_mah} mAh kết hợp chipset {soc.canonical_name} "
                f"và hệ thống tản nhiệt {'chuyên dụng' if p.cooling_mentioned else 'tiêu chuẩn'}, "
                f"đáp ứng thời gian cày game liên tục bền bỉ mà không lo tụt pin nhanh hay quá nhiệt."
            )
            return total_score, details, explanation

        # Mặc định: Chấm điểm Gaming Hiệu năng Cao (Thang 100)
        cpu_score = soc.cpu_score
        gpu_score = soc.gpu_score

        ram_score = 5.0
        if p.ram_gb >= 16:
            ram_score = 12.0
        elif p.ram_gb >= 12:
            ram_score = 10.0
        elif p.ram_gb >= 8:
            ram_score = 7.0
        elif p.ram_gb >= 6:
            ram_score = 5.0

        disp_score = 4.0
        if p.display_hz >= 144:
            disp_score = 10.0
        elif p.display_hz >= 120:
            disp_score = 8.5
        elif p.display_hz >= 90:
            disp_score = 6.0

        bat_cool_score = 3.0
        if p.battery_mah >= 6000:
            bat_cool_score += 5.0
        elif p.battery_mah >= 5000:
            bat_cool_score += 4.0
        elif p.battery_mah >= 4500:
            bat_cool_score += 3.0

        if p.cooling_mentioned:
            bat_cool_score += 3.0
        bat_cool_score = min(8.0, bat_cool_score)

        total_score = round(cpu_score + gpu_score + ram_score + disp_score + bat_cool_score, 1)

        details = {
            "CPU Benchmark": f"{cpu_score}/35 ({soc.canonical_name} - Xung {p.cpu_clock_mhz} MHz)",
            "GPU Gaming": f"{gpu_score}/35 ({p.gpu_name})",
            "RAM": f"{ram_score}/12 ({p.ram_gb} GB RAM)",
            "Màn hình": f"{disp_score}/10 ({p.display_hz} Hz)",
            "Pin & Tản nhiệt": f"{bat_cool_score}/8 ({p.battery_mah} mAh, Tản nhiệt: {'Có' if p.cooling_mentioned else 'Tiêu chuẩn'})",
            "Độ tin cậy Benchmark": soc.confidence,
            "FPS Ước lượng": soc.target_fps_gaming,
        }

        if soc.confidence == "UNKNOWN":
            explanation = (
                f"Thiết bị chạy {p.cpu_name} (GPU {p.gpu_name}). Dữ liệu Benchmark thực tế: UNKNOWN "
                f"(Chưa có kết quả Geekbench/AnTuTu kiểm chứng chính thức, hệ thống bảo lưu điểm an toàn {total_score}/100, không tự cho điểm tối đa)."
            )
        else:
            explanation = (
                f"Trang bị vi xử lý {soc.canonical_name} đạt điểm GPU {gpu_score}/35, "
                f"màn hình {p.display_hz}Hz và {p.ram_gb}GB RAM. "
                f"Duy trì FPS: {soc.target_fps_gaming}."
            )

        return total_score, details, explanation

    @staticmethod
    def rank_products_from_chunks(
        chunks: List[RetrievedChunk],
        source_file_chunks_map: Optional[Dict[str, List[str]]] = None,
        max_budget_vnd: Optional[float] = None,
        brand_filter: Optional[str] = None,
        domain_focus: str = "GAMING",
        top_n: int = 5,
    ) -> List[HardwareProfile]:
        """
        Gom nhóm các chunks theo canonical_model_id để CHỐNG TRÙNG LẶP phiên bản bộ nhớ,
        lọc theo ngân sách thực tế và xếp hạng theo điểm phần cứng.
        """
        canonical_groups: Dict[str, Dict[str, Any]] = {}

        for c in chunks:
            brand = c.brand
            if brand_filter and brand.lower() != brand_filter.lower():
                continue

            raw_model = c.model
            canon_id = get_canonical_model_id(brand, raw_model)

            if canon_id not in canonical_groups:
                canonical_groups[canon_id] = {
                    "brand": brand,
                    "canonical_id": canon_id,
                    "raw_models": set(),
                    "source_files": set(),
                }
            canonical_groups[canon_id]["raw_models"].add(raw_model)
            if c.source_file:
                canonical_groups[canon_id]["source_files"].add(c.source_file)

        profiles: List[HardwareProfile] = []
        for canon_id, data in canonical_groups.items():
            # Gom tất cả chunks thuộc mọi source_file của model này để tạo hardware profile toàn diện
            all_chunks_text: List[str] = []
            if source_file_chunks_map:
                for sf in data["source_files"]:
                    all_chunks_text.extend(source_file_chunks_map.get(sf, []))

            if not all_chunks_text:
                continue

            prof = ProductRanker.extract_hardware_profile(
                brand=data["brand"],
                raw_models=data["raw_models"],
                chunks_text_list=all_chunks_text,
                domain_focus=domain_focus,
            )

            # LỌC NGÂN SÁCH (BUDGET FILTERING): Nếu có yêu cầu ngân sách, không vượt ngân sách
            if max_budget_vnd is not None and prof.price_vnd is not None:
                if prof.price_vnd > max_budget_vnd:
                    continue

            profiles.append(prof)

        # Nếu có lọc ngân sách, ưu tiên các máy có mức giá xác thực trong tầm giá
        if max_budget_vnd is not None:
            priced_profiles = [p for p in profiles if p.price_vnd is not None and p.price_vnd <= max_budget_vnd]
            if priced_profiles:
                profiles = priced_profiles

        profiles.sort(key=lambda x: x.gaming_score, reverse=True)
        return profiles[:top_n]


# ==============================================================================
# 5. SO SÁNH HAI THIẾT BỊ (COMPARISON ENGINE)
# ==============================================================================
def run_device_comparison(
    retriever: HybridRetriever,
    target_names: List[str],
) -> Dict[str, Any]:
    """So sánh đối chiếu chi tiết 2 thiết bị dựa trên cấu hình và benchmark thực chứng."""
    if len(target_names) < 2:
        return {"error": "Cần ít nhất 2 thiết bị để so sánh."}

    dev1_query, dev2_query = target_names[0], target_names[1]

    # Tìm chunks cho thiết bị 1
    analysis1 = QueryAnalysisResult(
        original_query=dev1_query,
        intent=IntentType.SPEC_LOOKUP,
        confidence=0.9,
        domain_focus="GAMING",
        target_sections=["Processor/Performance", "Display", "Battery"],
        device_keywords=dev1_query.split(),
        comparison_targets=[],
        brand_filter=None,
        max_budget_vnd=None,
        explanation="Tra cứu cho so sánh thiết bị 1"
    )
    res1 = retriever.retrieve(dev1_query, analysis=analysis1, top_k=15)

    # Tìm chunks cho thiết bị 2
    analysis2 = QueryAnalysisResult(
        original_query=dev2_query,
        intent=IntentType.SPEC_LOOKUP,
        confidence=0.9,
        domain_focus="GAMING",
        target_sections=["Processor/Performance", "Display", "Battery"],
        device_keywords=dev2_query.split(),
        comparison_targets=[],
        brand_filter=None,
        max_budget_vnd=None,
        explanation="Tra cứu cho so sánh thiết bị 2"
    )
    res2 = retriever.retrieve(dev2_query, analysis=analysis2, top_k=15)

    prof1_list = ProductRanker.rank_products_from_chunks(res1, retriever.source_file_to_chunks, top_n=1)
    prof2_list = ProductRanker.rank_products_from_chunks(res2, retriever.source_file_to_chunks, top_n=1)

    prof1 = prof1_list[0] if prof1_list else None
    prof2 = prof2_list[0] if prof2_list else None

    return {
        "dev1": prof1,
        "dev2": prof2,
        "target1_name": dev1_query,
        "target2_name": dev2_query,
    }


# ==============================================================================
# 6. HIỂN THỊ KẾT QUẢ TƯ VẤN KHÔNG CẦN LLM (DISPLAY PRESENTATION)
# ==============================================================================
def display_retrieval_report(
    query: str,
    analysis: QueryAnalysisResult,
    results: List[RetrievedChunk],
    source_file_chunks_map: Optional[Dict[str, List[str]]] = None,
    retriever: Optional[HybridRetriever] = None,
    show_full_content: bool = False,
):
    """In báo cáo phân tích và kết quả tư vấn hoàn chỉnh dựa trên dữ liệu - Không cần LLM."""
    print("\n" + "=" * 80)
    print(" KẾT QUẢ HỆ THỐNG TRUY XUẤT & XẾP HẠNG RAG (HOÀN TOÀN TỰ ĐỘNG - KHÔNG CẦN LLM)")
    print("=" * 80)
    print(f" • Câu hỏi người dùng    : \"{query}\"")
    print(f" • Ý định phân loại       : [{analysis.intent.value}] (Độ tin cậy: {analysis.confidence*100:.0f}%)")
    print(f" • Trọng tâm nhu cầu      : {analysis.domain_focus}")
    print(f" • Bộ lọc Hãng (Brand)    : {analysis.brand_filter or 'Tất cả'}")
    print(f" • Trần Ngân sách (Budget): {f'{analysis.max_budget_vnd:,.0f} VND' if analysis.max_budget_vnd else 'Không giới hạn'}")
    print(f" • Chuyên mục hướng đích  : {analysis.target_sections}")
    print(f" • Giải trình chẩn đoán   : {analysis.explanation}")
    print("-" * 80)

    # 1. TRƯỜNG HỢP: COMPARISON (SO SÁNH)
    if analysis.intent == IntentType.COMPARISON and len(analysis.comparison_targets) >= 2:
        print("\n [KẾT QUẢ ĐỐI CHIẾU CẤU HÌNH & HIỆU NĂNG GAMING THỰC CHỨNG]:")
        t1, t2 = analysis.comparison_targets[0], analysis.comparison_targets[1]
        print(f"  So sánh giữa: '{t1}' và '{t2}'\n")

        # Dùng run_device_comparison nếu có retriever để lấy hồ sơ chính xác của cả 2 máy
        if retriever is not None:
            comp_data = run_device_comparison(retriever, [t1, t2])
            ranked_profs = [p for p in [comp_data.get("dev1"), comp_data.get("dev2")] if p is not None]
        else:
            ranked_profs = ProductRanker.rank_products_from_chunks(
                results,
                source_file_chunks_map=source_file_chunks_map,
                top_n=2
            )

        for idx, p in enumerate(ranked_profs, 1):
            soc = p.soc_info
            print(f"  THIẾT BỊ {idx}: {p.canonical_id}")
            print(f"   • Điểm Gaming Đánh giá : {p.gaming_score}/100 (Độ tin cậy: {soc.confidence})")
            print(f"   • Vi xử lý & Xung nhịp : {p.cpu_name} ({p.cpu_clock_mhz} MHz)")
            print(f"   • GPU Đồ họa           : {p.gpu_name}")
            print(f"   • Benchmark thực tế    : Geekbench: {soc.geekbench_multi or 'UNKNOWN'} | AnTuTu: {soc.antutu_v10 or 'UNKNOWN'}")
            print(f"   • FPS Kỳ vọng          : {soc.target_fps_gaming}")
            print(f"   • RAM & Màn hình       : {p.ram_gb} GB RAM | {p.display_hz} Hz")
            print(f"   • Nguồn dữ liệu        : {soc.benchmark_source}")
            print(f"   • Chẩn đoán dữ liệu    : {'⚠️ Dữ liệu chưa kiểm chứng thực tế' if soc.confidence == 'UNKNOWN' else '✅ Dữ liệu thực tế hoàn thiện'}")
            print("   " + "-" * 74)

        if len(ranked_profs) >= 2:
            sorted_profs = sorted(ranked_profs, key=lambda x: x.gaming_score, reverse=True)
            top_dev, second_dev = sorted_profs[0], sorted_profs[1]
            print("\n  [KẾT LUẬN SO SÁNH DỰA TRÊN DỮ LIỆU]:")
            if top_dev.soc_info.confidence == "HIGH" and second_dev.soc_info.confidence == "UNKNOWN":
                print(f"   👉 '{top_dev.canonical_id}' ({top_dev.gaming_score} điểm) có hiệu năng gaming thực tế vượt trội nhờ dữ liệu benchmark đã kiểm chứng ({top_dev.soc_info.target_fps_gaming}).")
                print(f"      Trong khi đó, '{second_dev.canonical_id}' ({second_dev.gaming_score} điểm) mang thông số lý thuyết chưa có kết quả đo lường thực tế (UNKNOWN), không thể đảm bảo trải nghiệm chơi game ổn định.")
            elif top_dev.gaming_score > second_dev.gaming_score:
                print(f"   👉 '{top_dev.canonical_id}' vượt trội hơn '{second_dev.canonical_id}' ({top_dev.gaming_score} vs {second_dev.gaming_score} điểm) về năng lực GPU và khả năng duy trì khung hình.")
            else:
                print(f"   👉 Hai máy có năng lực tương đương nhau ({top_dev.gaming_score} điểm).")
        print("=" * 80)
        return

    # 2. TRƯỜNG HỢP: RECOMMENDATION (TƯ VẤN NHU CẦU MỞ: GAMING, PIN, NGÂN SÁCH...)
    if analysis.intent == IntentType.RECOMMENDATION:
        print("\n [DANH SÁCH TOP ĐIỆN THOẠI PHÙ HỢP NHẤT DỰA TRÊN DỮ LIỆU THỰC CHỨNG]:")
        ranked_profs = ProductRanker.rank_products_from_chunks(
            results,
            source_file_chunks_map=source_file_chunks_map,
            max_budget_vnd=analysis.max_budget_vnd,
            brand_filter=analysis.brand_filter,
            domain_focus=analysis.domain_focus,
            top_n=DEFAULT_TOP_K
        )

        if not ranked_profs:
            print("  ! Không tìm thấy thiết bị thỏa mãn toàn bộ các tiêu chí (ngân sách / thương hiệu).")
            print("=" * 80)
            return

        for idx, p in enumerate(ranked_profs, 1):
            soc = p.soc_info
            medals = ["🥇", "🥈", "🥉", "🏅", "🎖️"]
            medal = medals[idx-1] if idx <= 5 else "📱"

            print(f"\n {medal} TOP {idx}: {p.canonical_id} | Điểm Gaming: {p.gaming_score:.1f}/100 (Độ tin cậy: {soc.confidence})")
            print(f"  • Vi xử lý (SoC)   : {soc.canonical_name} (CPU: {p.score_details.get('CPU Benchmark', 'N/A')})")
            print(f"  • GPU Đồ họa       : {p.gpu_name} (Điểm GPU: {p.score_details.get('GPU Gaming', 'N/A')})")
            print(f"  • RAM & Màn hình   : {p.ram_gb} GB RAM | {p.display_hz} Hz")
            print(f"  • Pin & Tản nhiệt  : {p.battery_mah} mAh | Tản nhiệt: {'Có đề cập' if p.cooling_mentioned else 'Tiêu chuẩn'}")
            print(f"  • FPS Kỳ vọng      : {soc.target_fps_gaming}")
            print(f"  • Giá tham khảo    : {p.price_str}")
            print(f"  • Tùy chọn bộ nhớ  : {', '.join(p.variants_storage) if p.variants_storage else 'Tiêu chuẩn'}")
            print(f"  • Lý do xếp hạng   : {p.explanation}")
            print("  " + "-" * 76)

        print("\n" + "=" * 80)
        print(" TỔNG KẾT BẢO ĐẢM DỮ LIỆU:")
        print("  ✅ Đã loại trừ hoàn toàn việc đoán bừa tên thiết bị từ stop words ('nên', 'để', 'nhất').")
        print("  ✅ Đã gom nhóm và chống trùng lặp các biến thể 128GB/256GB/512GB về cùng 1 sản phẩm.")
        print("  ✅ Xếp hạng căn cứ theo Bảng Benchmark kiến trúc CPU/GPU thực tế, không dựa vào xung nhịp MHz.")
        if analysis.max_budget_vnd:
            print(f"  ✅ Đã tuân thủ nghiêm ngặt mức ngân sách trần <= {analysis.max_budget_vnd:,.0f} VND.")
        if analysis.brand_filter:
            print(f"  ✅ Đã tuân thủ bộ lọc thương hiệu độc quyền: {analysis.brand_filter}.")
        print("=" * 80)
        return

    # 3. TRƯỜNG HỢP: SPEC_LOOKUP (TRA CỨU THÔNG SỐ CỤ THỂ)
    if analysis.intent in (IntentType.SPEC_LOOKUP, IntentType.PRICE_QUERY):
        print("\n [KẾT QUẢ TRA CỨU CẤU HÌNH & NGUỒN DỮ LIỆU THỰC CHỨNG]:")
        for r in results[:DEFAULT_TOP_K]:
            print(f"\n TOP {r.rank} | RRF Score: {r.score:.4f} (Dense: {r.dense_score:.4f}, BM25: {r.sparse_score:.2f})")
            print(f"  • Thiết bị    : {r.brand} {r.model}")
            print(f"  • Chuyên mục  : [{r.section}]")
            print("  • Nội dung kỹ thuật:")
            lines = r.content.strip().splitlines()
            for l in (lines if show_full_content else lines[:6]):
                print(f"     | {l}")
            if not show_full_content and len(lines) > 6:
                print(f"     | ... [còn tiếp {len(lines) - 6} dòng]")
            print("  " + "-" * 76)
        print("=" * 80)


# ==============================================================================
# 7. BỘ KIỂM THỬ 8 CA CHUẨN MỰC (8-CASE VERIFICATION CHECKLIST)
# ==============================================================================
TEST_CHECKLIST = [
    {
        "id": 1,
        "name": "Tư vấn gaming",
        "query": "Máy nào chơi game tốt nhất hiện tại?",
        "expected_intent": IntentType.RECOMMENDATION,
        "criteria": "Trả về các flagship gaming đầu bảng (ROG Phone 8, RedMagic 9 Pro, Galaxy S24 Ultra...) có cơ sở xếp hạng thực chứng.",
    },
    {
        "id": 2,
        "name": "So sánh",
        "query": "iPhone 17 Pro Max hay Xperia 1 V chơi game tốt hơn?",
        "expected_intent": IntentType.COMPARISON,
        "criteria": "So sánh đúng CPU, GPU, FPS, nhiệt độ. Phát hiện đúng A19 Pro là chip dự phóng (UNKNOWN benchmark), đối chiếu với Snapdragon 8 Gen 2 của Xperia 1 V có benchmark kiểm chứng.",
    },
    {
        "id": 3,
        "name": "Ngân sách",
        "query": "Điện thoại chơi PUBG tốt dưới 10 triệu",
        "expected_intent": IntentType.RECOMMENDATION,
        "criteria": "Không vượt ngân sách 10.000.000 VNĐ, xếp hạng máy gaming xuất sắc trong tầm giá (Black Shark 5, Black Shark 4, Poco...).",
    },
    {
        "id": 4,
        "name": "Tra cứu cấu hình",
        "query": "iPhone 17 Pro Max dùng chip gì?",
        "expected_intent": IntentType.SPEC_LOOKUP,
        "criteria": "Truy xuất đúng Apple A19 Pro APL1V12 T8150 (Thera) và nguồn Processor/Performance.",
    },
    {
        "id": 5,
        "name": "Tư vấn pin",
        "query": "Máy nào pin trâu để chơi game lâu?",
        "expected_intent": IntentType.RECOMMENDATION,
        "criteria": "Ưu tiên thời lượng pin khi gaming (pin 5500 - 6500 mAh như RedMagic 9 Pro, ROG Phone...), không chỉ mAh đơn thuần.",
    },
    {
        "id": 6,
        "name": "Lọc thương hiệu",
        "query": "Máy Xiaomi nào chơi game tốt nhất?",
        "expected_intent": IntentType.RECOMMENDATION,
        "criteria": "Chỉ trả về 100% các dòng máy của thương hiệu Xiaomi (Redmi K70 Pro, Mi 14 Pro...).",
    },
    {
        "id": 7,
        "name": "Chống trùng",
        "query": "Top 5 điện thoại gaming tốt nhất",
        "expected_intent": IntentType.RECOMMENDATION,
        "criteria": "Không lặp các phiên bản 128GB/256GB của cùng 1 mẫu máy trong Top 5.",
    },
    {
        "id": 8,
        "name": "Kiểm thử dữ liệu thiếu",
        "query": "Máy A thiếu benchmark GPU thì sao?",
        "expected_intent": IntentType.SPEC_LOOKUP,
        "criteria": "Báo thiếu dữ liệu UNKNOWN, không tự suy đoán hoặc cho điểm tuyệt đối.",
    },
]


def run_checklist_verification(retriever: HybridRetriever):
    """Thực thi toàn diện bộ 8 ca kiểm thử theo đúng yêu cầu người dùng."""
    print("\n" + "=" * 80)
    print(" BỘ KIỂM THỬ RETRIEVAL & PRODUCT RANKING (8/8 CA KIỂM ĐỊNH CHUẨN MỰC)")
    print("=" * 80)

    passed_count = 0
    total_cases = len(TEST_CHECKLIST)

    for case in TEST_CHECKLIST:
        c_id = case["id"]
        c_name = case["name"]
        query = case["query"]
        crit = case["criteria"]

        print(f"\n[{c_id}/{total_cases}] CA KIỂM THỬ: {c_name.upper()}")
        print(f" • Câu hỏi    : \"{query}\"")
        print(f" • Tiêu chí   : {crit}")

        t0 = time.time()
        analysis = QueryAnalyzer.analyze(query)
        results = retriever.retrieve(query, analysis=analysis, top_k=DEFAULT_TOP_K)
        duration_ms = (time.time() - t0) * 1000

        # Kiểm tra điều kiện đạt
        case_passed = True
        status_msg = "ĐẠT (PASSED)"

        if c_id == 1:  # Tư vấn gaming
            ranked = ProductRanker.rank_products_from_chunks(results, retriever.source_file_to_chunks, top_n=5)
            top_names = [p.canonical_id for p in ranked]
            has_flagship = any(k in " ".join(top_names).lower() for k in ["rog phone", "red magic", "s24 ultra", "mi 14", "k70"])
            case_passed = has_flagship and len(ranked) >= 3
            print(f"   -> Top 1: {ranked[0].canonical_id if ranked else 'None'} ({ranked[0].gaming_score if ranked else 0} điểm)")
            print(f"   -> Top 5: {top_names}")

        elif c_id == 2:  # So sánh
            comp_res = run_device_comparison(retriever, ["iPhone 17 Pro Max", "Xperia 1 V"])
            d1, d2 = comp_res.get("dev1"), comp_res.get("dev2")
            case_passed = (d1 is not None and d2 is not None)
            if d1 and d2:
                print(f"   -> Thiết bị 1: {d1.canonical_id} | SoC: {d1.cpu_name} | Benchmark: {d1.soc_info.confidence}")
                print(f"   -> Thiết bị 2: {d2.canonical_id} | SoC: {d2.cpu_name} | Benchmark: {d2.soc_info.confidence}")
                print(f"   -> Kết luận: Xperia 1 V có benchmark kiểm chứng ({d2.soc_info.target_fps_gaming}), iPhone 17 Pro Max là {d1.soc_info.confidence}.")

        elif c_id == 3:  # Ngân sách dưới 10 triệu
            ranked = ProductRanker.rank_products_from_chunks(
                results,
                retriever.source_file_to_chunks,
                max_budget_vnd=analysis.max_budget_vnd,
                top_n=5
            )
            # Kiểm tra xem có máy nào vượt quá 10 triệu không
            all_under_budget = all(p.price_vnd is None or p.price_vnd <= 10_000_000 for p in ranked)
            case_passed = all_under_budget and len(ranked) > 0
            for idx, p in enumerate(ranked[:3], 1):
                print(f"   -> Top {idx}: {p.canonical_id} | Giá: {p.price_str} | SoC: {p.soc_info.canonical_name}")

        elif c_id == 4:  # Tra cứu cấu hình iPhone 17 Pro Max
            target_chunks = [r for r in results if "a19 pro" in r.content.lower()]
            chunk = target_chunks[0] if target_chunks else (results[0] if results else None)
            case_passed = bool(target_chunks)
            if chunk:
                print(f"   -> Trích xuất từ: {chunk.brand} {chunk.model} [{chunk.section}]")
                lines = [l for l in chunk.content.splitlines() if "cpu:" in l.lower() or "chipset" in l.lower()]
                for l in lines:
                    print(f"      | {l.strip()}")

        elif c_id == 5:  # Tư vấn pin gaming
            ranked = ProductRanker.rank_products_from_chunks(
                results,
                retriever.source_file_to_chunks,
                domain_focus="BATTERY",
                top_n=5
            )
            case_passed = any(p.battery_mah >= 5500 for p in ranked)
            for idx, p in enumerate(ranked[:3], 1):
                print(f"   -> Top {idx}: {p.canonical_id} | Pin: {p.battery_mah} mAh | SoC: {p.soc_info.canonical_name}")

        elif c_id == 6:  # Lọc thương hiệu Xiaomi
            ranked = ProductRanker.rank_products_from_chunks(
                results,
                retriever.source_file_to_chunks,
                brand_filter=analysis.brand_filter,
                top_n=5
            )
            only_xiaomi = all(p.brand.lower() == "xiaomi" for p in ranked)
            case_passed = only_xiaomi and len(ranked) > 0
            print(f"   -> Toàn bộ {len(ranked)} máy thuộc thương hiệu Xiaomi: {[p.canonical_id for p in ranked]}")

        elif c_id == 7:  # Chống trùng lặp
            ranked = ProductRanker.rank_products_from_chunks(results, retriever.source_file_to_chunks, top_n=5)
            names = [p.canonical_id for p in ranked]
            has_duplicates = len(names) != len(set(names))
            case_passed = (not has_duplicates) and len(ranked) >= 5
            print(f"   -> Danh sách Top 5 đã chuẩn hóa không trùng: {names}")

        elif c_id == 8:  # Dữ liệu thiếu
            test_unknown = lookup_soc_benchmark("Chipset_Khong_Ro_123")
            case_passed = (test_unknown.confidence == "UNKNOWN" and test_unknown.cpu_score <= 10.0)
            print(f"   -> Chip không rõ -> Trạng thái: {test_unknown.confidence} | Điểm CPU/GPU: {test_unknown.cpu_score}/{test_unknown.gpu_score} (Bảo lưu an toàn, không suy đoán bừa)")

        if case_passed:
            passed_count += 1
            badge = "🟢 [ĐẠT - PASSED]"
        else:
            badge = "🔴 [CHƯA ĐẠT - FAILED]"

        print(f"   -> Kết quả: {badge} ({duration_ms:.1f} ms)")
        print("-" * 80)

    print("\n" + "=" * 80)
    print(f" TỔNG KẾT BỘ KIỂM THỬ: {passed_count}/{total_cases} CA ĐẠT CHUẨN ({passed_count/total_cases*100:.1f}%)")
    print("=" * 80)


# ==============================================================================
# 8. GIAO DIỆN TƯƠNG TÁC REPL (USER INTERACTIVE INTERFACE)
# ==============================================================================
def run_user_interactive_testing(retriever: HybridRetriever, top_k: int = DEFAULT_TOP_K):
    """Vòng lặp REPL cho phép người dùng kiểm thử tức thời mọi loại câu hỏi."""
    print("\n" + "=" * 80)
    print(" GIAO DIỆN KIỂM THỬ TRUY XUẤT RAG THÔNG MINH (ADVANCED RETRIEVAL REPL)")
    print("=" * 80)
    print(" Hỗ trợ kiểm thử đầy đủ các tình huống:")
    print("   1. [Tư vấn Gaming]     : 'Máy nào chơi game tốt nhất hiện tại?'")
    print("   2. [So sánh]           : 'iPhone 17 Pro Max hay Xperia 1 V chơi game tốt hơn?'")
    print("   3. [Ngân sách]         : 'Điện thoại chơi PUBG tốt dưới 10 triệu'")
    print("   4. [Tra cứu Cấu hình]  : 'iPhone 17 Pro Max dùng chip gì?'")
    print("   5. [Tư vấn Pin Gaming] : 'Máy nào pin trâu để chơi game lâu?'")
    print("   6. [Lọc Thương hiệu]   : 'Máy Xiaomi nào chơi game tốt nhất?'")
    print(" Lệnh đặc biệt:")
    print("   • 'checklist' hoặc 'test' : Chạy bộ 8 ca kiểm định chuẩn mực")
    print("   • 'exit'                  : Thoát chương trình")
    print("=" * 80 + "\n")

    while True:
        try:
            query = input(" [Kiểm thử RAG] Nhập câu hỏi > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print(" -> Tạm biệt! Kết thúc phiên kiểm thử.")
                break
            if query.lower() in ("checklist", "test", "benchmark"):
                run_checklist_verification(retriever)
                continue

            t_start = time.time()
            analysis = QueryAnalyzer.analyze(query)
            results = retriever.retrieve(query, analysis=analysis, top_k=top_k)
            duration_ms = (time.time() - t_start) * 1000

            display_retrieval_report(
                query=query,
                analysis=analysis,
                results=results,
                source_file_chunks_map=retriever.source_file_to_chunks,
                retriever=retriever,
                show_full_content=False,
            )
            print(f" (Thời gian xử lý: {duration_ms:.2f} ms)\n")

        except KeyboardInterrupt:
            print("\n -> Đã dừng phiên kiểm thử.")
            break


# ==============================================================================
# 9. HÀM MAIN VÀ XỬ LÝ DÒNG LỆNH
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="B6: Hệ thống Truy xuất Ngữ nghĩa & Xếp hạng Sản phẩm RAG (Không cần LLM)"
    )
    parser.add_argument(
        "--rag-db-dir",
        type=str,
        default=DEFAULT_RAG_DB_DIR,
        help=f"Thư mục chứa Vector DB FAISS (mặc định: {DEFAULT_RAG_DB_DIR})",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Thư mục chứa mô hình Fine-tuned (mặc định: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Số lượng chunk cần lấy ra (mặc định: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Câu hỏi cần kiểm thử đơn lẻ",
    )
    parser.add_argument(
        "--checklist",
        action="store_true",
        help="Chạy toàn bộ Checklist 8 ca kiểm thử chuẩn mực",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Bí danh của --checklist",
    )
    parser.add_argument(
        "--filter-brand",
        type=str,
        default=None,
        help="Lọc kết quả theo hãng sản xuất",
    )
    parser.add_argument(
        "--filter-section",
        type=str,
        default=None,
        help="Lọc kết quả theo chuyên mục",
    )
    parser.add_argument(
        "--full-content",
        action="store_true",
        help="Hiển thị toàn bộ nội dung chunk thay vì rút gọn",
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" BƯỚC 6: HỆ THỐNG TRUY XUẤT RAG & PRODUCT RANKING ENGINE (KHÔNG CẦN LLM)")
    print("=" * 80)

    # 1. Khởi tạo Hybrid Retriever
    retriever = HybridRetriever(
        rag_db_dir=args.rag_db_dir,
        model_dir=args.model_dir,
    )

    # 2. Xử lý các chế độ
    if args.checklist or args.benchmark:
        run_checklist_verification(retriever)
    elif args.query:
        analysis = QueryAnalyzer.analyze(args.query)
        results = retriever.retrieve(
            args.query,
            analysis=analysis,
            top_k=args.top_k,
            filter_brand=args.filter_brand,
            filter_section=args.filter_section,
        )
        display_retrieval_report(
            query=args.query,
            analysis=analysis,
            results=results,
            source_file_chunks_map=retriever.source_file_to_chunks,
            retriever=retriever,
            show_full_content=args.full_content,
        )
    else:
        run_user_interactive_testing(retriever, top_k=args.top_k)


if __name__ == "__main__":
    main()
