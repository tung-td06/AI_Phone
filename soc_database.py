"""
Module soc_database: Bảng hiệu năng SoC (CPU/GPU) chuẩn hóa dựa trên Benchmark thực tế.
Không suy đoán bừa bãi điểm số từ MHz.
"""

import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class SoCBenchmarkInfo:
    canonical_name: str
    generation_year: int
    cpu_score: float                # 0 - 35 (chuẩn hóa theo Geekbench 6 Single & Multi-Core)
    gpu_score: float                # 0 - 35 (chuẩn hóa theo 3DMark Wild Life Extreme & AnTuTu GPU)
    geekbench_single: Optional[int]
    geekbench_multi: Optional[int]
    antutu_v10: Optional[int]
    target_fps_gaming: str          # FPS thực tế trong game nặng (Genshin, PUBG...)
    benchmark_source: str
    confidence: str                 # HIGH, MEDIUM, UNKNOWN
    
    @property
    def total_soc_score(self) -> float:
        return round(self.cpu_score + self.gpu_score, 1)


# Bảng xếp hạng SoC đầu bảng và cận cao cấp chuẩn hóa
KNOWN_SOC_DATABASE: Dict[str, SoCBenchmarkInfo] = {
    # Qualcomm Flagship Series
    "snapdragon_8_gen_3": SoCBenchmarkInfo(
        canonical_name="Qualcomm Snapdragon 8 Gen 3",
        generation_year=2024,
        cpu_score=35.0,
        gpu_score=35.0,
        geekbench_single=2250,
        geekbench_multi=7100,
        antutu_v10=2100000,
        target_fps_gaming="60 FPS Max Settings (Khung hình cực ổn định)",
        benchmark_source="Geekbench 6 & 3DMark Wild Life Extreme (Dữ liệu tổng hợp 2024)",
        confidence="HIGH"
    ),
    "snapdragon_8_gen_2": SoCBenchmarkInfo(
        canonical_name="Qualcomm Snapdragon 8 Gen 2",
        generation_year=2023,
        cpu_score=31.0,
        gpu_score=31.0,
        geekbench_single=1980,
        geekbench_multi=5400,
        antutu_v10=1550000,
        target_fps_gaming="60 FPS Max Settings (Rất mượt)",
        benchmark_source="Geekbench 6 & 3DMark Wild Life Extreme",
        confidence="HIGH"
    ),
    "snapdragon_8_plus_gen_1": SoCBenchmarkInfo(
        canonical_name="Qualcomm Snapdragon 8+ Gen 1",
        generation_year=2022,
        cpu_score=28.0,
        gpu_score=27.5,
        geekbench_single=1780,
        geekbench_multi=4600,
        antutu_v10=1300000,
        target_fps_gaming="58-60 FPS High Settings",
        benchmark_source="Geekbench 6 & AnTuTu v10",
        confidence="HIGH"
    ),
    "snapdragon_8_gen_1": SoCBenchmarkInfo(
        canonical_name="Qualcomm Snapdragon 8 Gen 1",
        generation_year=2022,
        cpu_score=25.0,
        gpu_score=25.0,
        geekbench_single=1650,
        geekbench_multi=4100,
        antutu_v10=1050000,
        target_fps_gaming="55-60 FPS High Settings",
        benchmark_source="Geekbench 6 & AnTuTu v10",
        confidence="HIGH"
    ),
    "snapdragon_888": SoCBenchmarkInfo(
        canonical_name="Qualcomm Snapdragon 888",
        generation_year=2021,
        cpu_score=22.0,
        gpu_score=21.0,
        geekbench_single=1500,
        geekbench_multi=3700,
        antutu_v10=820000,
        target_fps_gaming="50-55 FPS Medium/High Settings",
        benchmark_source="Geekbench 6 & AnTuTu v10",
        confidence="HIGH"
    ),
    "snapdragon_870": SoCBenchmarkInfo(
        canonical_name="Qualcomm Snapdragon 870",
        generation_year=2021,
        cpu_score=20.0,
        gpu_score=19.0,
        geekbench_single=1350,
        geekbench_multi=3400,
        antutu_v10=720000,
        target_fps_gaming="50-55 FPS Medium Settings",
        benchmark_source="Geekbench 6 & AnTuTu v10",
        confidence="HIGH"
    ),
    "snapdragon_865": SoCBenchmarkInfo(
        canonical_name="Qualcomm Snapdragon 865",
        generation_year=2020,
        cpu_score=18.0,
        gpu_score=17.5,
        geekbench_single=1250,
        geekbench_multi=3200,
        antutu_v10=650000,
        target_fps_gaming="45-50 FPS Medium Settings",
        benchmark_source="Geekbench 6",
        confidence="HIGH"
    ),
    "snapdragon_7_plus_gen_2": SoCBenchmarkInfo(
        canonical_name="Qualcomm Snapdragon 7+ Gen 2",
        generation_year=2023,
        cpu_score=26.0,
        gpu_score=25.0,
        geekbench_single=1700,
        geekbench_multi=4400,
        antutu_v10=1120000,
        target_fps_gaming="58-60 FPS High Settings",
        benchmark_source="Geekbench 6 & AnTuTu v10",
        confidence="HIGH"
    ),

    # Apple A-Series
    "apple_a19_pro": SoCBenchmarkInfo(
        canonical_name="Apple A19 Pro (Dự phóng Datasheet)",
        generation_year=2025,
        cpu_score=20.0,
        gpu_score=20.0,
        geekbench_single=None,
        geekbench_multi=None,
        antutu_v10=None,
        target_fps_gaming="Chưa đo lường thực tế (Dữ liệu dự phóng)",
        benchmark_source="Thông số lý thuyết Datasheet (Chưa có kết quả Geekbench/3DMark thực tế)",
        confidence="UNKNOWN"
    ),
    "apple_a17_pro": SoCBenchmarkInfo(
        canonical_name="Apple A17 Pro",
        generation_year=2023,
        cpu_score=35.0,
        gpu_score=33.5,
        geekbench_single=2950,
        geekbench_multi=7350,
        antutu_v10=1580000,
        target_fps_gaming="60 FPS Max Settings (Hỗ trợ Ray Tracing)",
        benchmark_source="Geekbench 6 & 3DMark Solar Bay",
        confidence="HIGH"
    ),
    "apple_a16_bionic": SoCBenchmarkInfo(
        canonical_name="Apple A16 Bionic",
        generation_year=2022,
        cpu_score=32.0,
        gpu_score=29.0,
        geekbench_single=2600,
        geekbench_multi=6600,
        antutu_v10=1450000,
        target_fps_gaming="60 FPS Max Settings",
        benchmark_source="Geekbench 6",
        confidence="HIGH"
    ),
    "apple_a15_bionic": SoCBenchmarkInfo(
        canonical_name="Apple A15 Bionic",
        generation_year=2021,
        cpu_score=28.5,
        gpu_score=26.0,
        geekbench_single=2350,
        geekbench_multi=5800,
        antutu_v10=1250000,
        target_fps_gaming="58-60 FPS High Settings",
        benchmark_source="Geekbench 6",
        confidence="HIGH"
    ),
    "apple_a14_bionic": SoCBenchmarkInfo(
        canonical_name="Apple A14 Bionic",
        generation_year=2020,
        cpu_score=24.0,
        gpu_score=21.0,
        geekbench_single=2100,
        geekbench_multi=4900,
        antutu_v10=980000,
        target_fps_gaming="50-55 FPS High Settings",
        benchmark_source="Geekbench 6",
        confidence="HIGH"
    ),

    # MediaTek Dimensity Series
    "dimensity_9300": SoCBenchmarkInfo(
        canonical_name="MediaTek Dimensity 9300",
        generation_year=2024,
        cpu_score=34.5,
        gpu_score=34.5,
        geekbench_single=2220,
        geekbench_multi=7400,
        antutu_v10=2150000,
        target_fps_gaming="60 FPS Max Settings (All-Big-Core Architecture)",
        benchmark_source="Geekbench 6 & GFXBench",
        confidence="HIGH"
    ),
    "dimensity_9200": SoCBenchmarkInfo(
        canonical_name="MediaTek Dimensity 9200",
        generation_year=2023,
        cpu_score=29.0,
        gpu_score=29.0,
        geekbench_single=1950,
        geekbench_multi=5200,
        antutu_v10=1450000,
        target_fps_gaming="58-60 FPS Max Settings",
        benchmark_source="Geekbench 6",
        confidence="HIGH"
    ),
    "dimensity_9000": SoCBenchmarkInfo(
        canonical_name="MediaTek Dimensity 9000",
        generation_year=2022,
        cpu_score=26.0,
        gpu_score=24.0,
        geekbench_single=1700,
        geekbench_multi=4400,
        antutu_v10=1050000,
        target_fps_gaming="55-60 FPS High Settings",
        benchmark_source="Geekbench 6",
        confidence="HIGH"
    ),
    "dimensity_8300": SoCBenchmarkInfo(
        canonical_name="MediaTek Dimensity 8300",
        generation_year=2024,
        cpu_score=27.0,
        gpu_score=27.0,
        geekbench_single=1520,
        geekbench_multi=4800,
        antutu_v10=1380000,
        target_fps_gaming="58-60 FPS High Settings",
        benchmark_source="Geekbench 6 & AnTuTu v10",
        confidence="HIGH"
    ),
    "dimensity_8100": SoCBenchmarkInfo(
        canonical_name="MediaTek Dimensity 8100",
        generation_year=2022,
        cpu_score=21.0,
        gpu_score=20.0,
        geekbench_single=1300,
        geekbench_multi=3900,
        antutu_v10=820000,
        target_fps_gaming="50-55 FPS Medium/High Settings",
        benchmark_source="Geekbench 6",
        confidence="HIGH"
    ),
    "dimensity_1200": SoCBenchmarkInfo(
        canonical_name="MediaTek Dimensity 1200",
        generation_year=2021,
        cpu_score=18.0,
        gpu_score=16.0,
        geekbench_single=1100,
        geekbench_multi=3300,
        antutu_v10=680000,
        target_fps_gaming="45-50 FPS Medium Settings",
        benchmark_source="Geekbench 6",
        confidence="HIGH"
    ),
}


def lookup_soc_benchmark(raw_chip_text: str) -> SoCBenchmarkInfo:
    """
    Tra cứu điểm hiệu năng thực tế của vi xử lý từ Bảng chuẩn hóa.
    Nếu không tìm thấy hoặc chip quá cũ/không rõ, trả về UNKNOWN với điểm bảo toàn.
    """
    text = raw_chip_text.lower()

    # Rule-based matching patterns
    patterns = [
        (r"8\s*gen\s*3|sm8650", "snapdragon_8_gen_3"),
        (r"8\s*gen\s*2|sm8550", "snapdragon_8_gen_2"),
        (r"8\+\s*gen\s*1|sm8475", "snapdragon_8_plus_gen_1"),
        (r"8\s*gen\s*1|sm8450", "snapdragon_8_gen_1"),
        (r"888\b|sm8350", "snapdragon_888"),
        (r"870\b|sm8250-ac", "snapdragon_870"),
        (r"865\b|sm8250", "snapdragon_865"),
        (r"7\+\s*gen\s*2|sm7475", "snapdragon_7_plus_gen_2"),
        (r"a19\s*pro|apl1v12", "apple_a19_pro"),
        (r"a17\s*pro|apl1v02", "apple_a17_pro"),
        (r"a16\b|apl1w10", "apple_a16_bionic"),
        (r"a15\b|apl1w07", "apple_a15_bionic"),
        (r"a14\b|apl1w01", "apple_a14_bionic"),
        (r"9300\b|mt6989", "dimensity_9300"),
        (r"9200\b|mt6985", "dimensity_9200"),
        (r"9000\b|mt6983", "dimensity_9000"),
        (r"8300\b|mt6897", "dimensity_8300"),
        (r"8100\b|8200\b|mt6895|mt6896", "dimensity_8100"),
        (r"1200\b|mt6893", "dimensity_1200"),
    ]

    for pat, key in patterns:
        if re.search(pat, text):
            return KNOWN_SOC_DATABASE[key]

    # Kiểm tra các chip tầm trung thông dụng khác (Snapdragon 700/600 series, Helio G...)
    if "snapdragon 7" in text or "adreno 642" in text or "adreno 643" in text:
        return SoCBenchmarkInfo(
            canonical_name=raw_chip_text.strip(),
            generation_year=2021,
            cpu_score=15.0,
            gpu_score=14.0,
            geekbench_single=950,
            geekbench_multi=2800,
            antutu_v10=550000,
            target_fps_gaming="40-45 FPS Medium Settings",
            benchmark_source="Ước lượng theo phân khúc Snapdragon 7-series",
            confidence="MEDIUM"
        )
    elif "snapdragon 6" in text or "helio g9" in text or "helio g8" in text:
        return SoCBenchmarkInfo(
            canonical_name=raw_chip_text.strip(),
            generation_year=2020,
            cpu_score=10.0,
            gpu_score=8.0,
            geekbench_single=600,
            geekbench_multi=1800,
            antutu_v10=350000,
            target_fps_gaming="30-40 FPS Low Settings",
            benchmark_source="Ước lượng phân khúc tầm trung / giá rẻ",
            confidence="MEDIUM"
        )

    # TRƯỜNG HỢP KHÔNG CÓ THÔNG TIN BENCHMARK HOẶC CHIP CŨ
    return SoCBenchmarkInfo(
        canonical_name=raw_chip_text.strip() if raw_chip_text else "UNKNOWN_SOC",
        generation_year=2018,
        cpu_score=5.0,
        gpu_score=5.0,
        geekbench_single=None,
        geekbench_multi=None,
        antutu_v10=None,
        target_fps_gaming="< 30 FPS (Không khuyến nghị chơi game)",
        benchmark_source="Không tìm thấy dữ liệu benchmark chính thức",
        confidence="UNKNOWN"
    )
