"""Filter latest regulation PDFs into a 'latest/' subfolder.

Groups PDFs by regulation topic, keeps:
- Latest original version per topic group
- Latest amendment (sua_doi/cap_nhat/bo_sung) per topic group — both kept
- CTDT curriculum files: all khoa versions copied as-is
- Unrecognized files (forms, guides, announcements): skipped

Usage:
    python src/data_processing/filter-latest-regulations.py [--dry-run] [--source PATH]
"""

import argparse
import re
import shutil
from datetime import date
from pathlib import Path
from urllib.parse import unquote

# ---------------------------------------------------------------------------
# Source / destination
# ---------------------------------------------------------------------------

DEFAULT_PDF_DIR = Path("data/raw/pdfs")

# ---------------------------------------------------------------------------
# Topic keyword → canonical topic name (more specific patterns first)
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: list[tuple[str, str]] = [
    # Quy chế đào tạo
    ("quy_che_dao_tao", "quy_che_dao_tao"),
    ("quychets", "quy_che_dao_tao"),
    # Ngoại ngữ / tiếng anh
    ("dao_tao_ngoai_ngu", "ngoai_ngu"),
    ("quy_dinh_ngoai_ngu", "ngoai_ngu"),
    ("day_va_hoc_nn", "ngoai_ngu"),
    ("day_hoc_nn", "ngoai_ngu"),
    ("tieng_anh_khoa", "ngoai_ngu"),
    ("ngoai_ngu", "ngoai_ngu"),
    # Văn bằng / chứng chỉ
    ("van_bang_chung_chi", "van_bang_chung_chi"),
    ("bang_tot_nghiep", "van_bang_chung_chi"),
    ("mau_plvb", "van_bang_chung_chi"),
    ("plvb", "van_bang_chung_chi"),
    # Tổ chức thi / kiểm tra
    ("to_chuc_thi", "to_chuc_thi"),
    ("danh_gia_ket_qua_hoc_tap", "to_chuc_thi"),
    ("thi_tren_may_tinh", "to_chuc_thi"),
    ("dgkqht", "to_chuc_thi"),
    # Mở ngành / xác định chỉ tiêu
    ("quy_dinh_mo_nganh", "mo_nganh"),
    ("mo_nganh", "mo_nganh"),
    ("chi_tieu_tuyen_sinh_dai_hoc", "mo_nganh"),
    # Chương trình tài năng
    ("chuong_trinh_tai_nang", "ctdt_tai_nang"),
    ("tai_nang", "ctdt_tai_nang"),
    # Chương trình tiên tiến
    ("chuong_trinh_tien_tien", "ctdt_tien_tien"),
    ("tien_tien", "ctdt_tien_tien"),
    # Chương trình chất lượng cao
    ("chat_luong_cao", "ctdt_clc"),
    # Song ngành
    ("song_nganh", "song_nganh"),
    # Liên thông
    ("lien_thong", "lien_thong"),
    # KLTN / Đồ án
    ("do_an_tot_nghiep_tai_doanh_nghiep", "kltn"),
    ("kltn", "kltn"),
    ("do_an_tot_nghiep", "kltn"),
    ("khoa_luan_tot_nghiep", "kltn"),
    ("nop_kltn", "kltn"),
    # Tuyển sinh
    ("quy_che_tuyen_sinh", "tuyen_sinh"),
    ("tuyen_sinh_trinh_do_dh", "tuyen_sinh"),
    # Luật giáo dục
    ("luat_giao_duc_dai_hoc", "luat_giao_duc"),
    ("luat_giao_duc", "luat_giao_duc"),
    # Công nhận tín chỉ
    ("cong_nhan_tin_chi", "tin_chi"),
    ("quy_trinh_cong_nhan_tin_chi", "tin_chi"),
    ("hoc_che_tin_chi", "tin_chi"),
    ("cong_nhan_mon_hoc_theo_hinh_thuc_mooc", "tin_chi_mooc"),
    # Giáo dục thể chất
    ("giao_duc_the_chat", "giao_duc_the_chat"),
    # Phân công cán bộ coi thi
    ("phan_cong_cbct", "phan_cong_cbct"),
    ("phan_cong_can_bo_coi_thi", "phan_cong_cbct"),
    # Dạy học trực tuyến
    ("dao_tao_truc_tuyen", "day_hoc_truc_tuyen"),
    ("day_hoc_theo_phuong_thuc_truc_tuyen", "day_hoc_truc_tuyen"),
    ("ung_dung_cntt_trong_dao_tao", "day_hoc_truc_tuyen"),
    ("quy_che_dttx", "day_hoc_truc_tuyen"),
    # Liên kết nước ngoài
    ("lien_ket_dao_tao_voi_nuoc_ngoai", "lien_ket_nuoc_ngoai"),
    ("lien_ket_dao_tao_giua", "lien_ket_nuoc_ngoai"),
    ("hop_tac_dau_tu_voi_nuoc_ngoai", "lien_ket_nuoc_ngoai"),
    # Chuẩn CTDT / chương trình đào tạo
    ("chuan_ctdt", "chuan_ctdt"),
    ("chuan_chuong_trinh_dao_tao", "chuan_ctdt"),
    ("ban_hanh_chuan_ctdt", "chuan_ctdt"),
    # Xử lý học vụ
    ("xlhv", "xu_ly_hoc_vu"),
    ("quy_trinh_xlhv", "xu_ly_hoc_vu"),
    # Danh mục ngành đào tạo
    ("danh_muc_giao_duc_dao_tao", "danh_muc_nganh"),
    ("danh_muc_thong_ke_nganh", "danh_muc_nganh"),
    # Đánh giá cập nhật CTDT
    ("quy_trinh_danh_gia_cap_nhat_ctdt", "danh_gia_ctdt"),
    # Báo nghỉ / dạy bù
    ("bao_nghi_day_day_bu", "bao_nghi_day_bu"),
    ("bao_nghi_bao_bu", "bao_nghi_day_bu"),
    # Học ngoài giờ hành chính
    ("ngoai_gio_hanh_chinh", "day_hoc_ngoai_gio"),
    # Giáo trình
    ("quy_che_cong_tac_giao_trinh", "giao_trinh"),
    # Khung năng lực ngoại ngữ
    ("khung_nang_luc_ngoai_ngu", "khung_nlnn"),
    ("khung_trinh_do_quoc_gia", "khung_tdqg"),
    # Chứng chỉ ngoại ngữ quốc tế được công nhận
    ("cong_nhan_chung_chi_ngoai_ngu", "cc_ngoai_ngu"),
    ("cong_nhan_chung_chi", "cc_ngoai_ngu"),
    ("su_dung_ket_qua_thi", "cc_ngoai_ngu"),
    ("ngung_su_dung", "cc_ngoai_ngu"),
    # Thi đua / hỗ trợ sinh viên
    ("ho_tro_cong_bo_khoa_hoc", "ho_tro_sv"),
    # Nội dung in trên văn bằng / biểu mẫu
    ("in_noi_dung_tren_vbcc", "in_vbcc"),
    ("tu_ngu_su_dung_in_tren", "in_vbcc"),
    # Lưu trữ hồ sơ
    ("thoi_han_luu_tru_ho_so", "luu_tru"),
    # Chế độ làm việc giảng viên
    ("che_do_lam_viec_cua_gv", "lam_viec_gv"),
    # Liên kết đào tạo trong nước (DHQG)
    ("lien_ket_dao_tao", "lien_ket_trong_nuoc"),
    # Đào tạo song ngành (DHQG level)
    ("phe_duyet_song_nganh", "phe_duyet_song_nganh"),
    ("phe_duyet_lien_ket_dao_tao", "phe_duyet_lien_ket"),
    ("phe_duyet_thi_diem", "phe_duyet_thi_diem"),
    # Chuẩn đầu ra tiếng Anh sinh viên
    ("chuan_dau_ra_tieng_anh", "chuan_dau_ra_nn"),
    # Đào tạo lien truong
    ("dao_tao_lien_truong", "dao_tao_lien_truong"),
    # Quản lý chất lượng
    ("quan_ly_chat_luong", "qlcl"),
    # Thực hiện MOOC
    ("mooc", "mooc"),
    # Cấp bằng tốt nghiệp / xét TN
    ("to_chuc_danh_gia_ket_qua", "to_chuc_thi"),
]

AMENDMENT_KEYWORDS = {"sua_doi", "cap_nhat", "bo_sung"}


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    # DD-MM-YYYY or DD_MM_YYYY
    (
        re.compile(r"(\d{1,2})[_-](\d{1,2})[_-](20\d{2})"),
        lambda m: date(int(m.group(3)), int(m.group(2)), int(m.group(1))),
    ),
    # DD-MM-YY (2-digit year, assume 2000s)
    (
        re.compile(r"(\d{1,2})[_-](\d{1,2})[_-](\d{2})\b"),
        lambda m: date(2000 + int(m.group(3)), int(m.group(2)), int(m.group(1))),
    ),
    # Standalone 4-digit year
    (
        re.compile(r"(20\d{2})"),
        lambda m: date(int(m.group(1)), 1, 1),
    ),
]


def extract_date(stem: str) -> date:
    """Extract best available date from filename stem."""
    for pattern, extractor in _DATE_PATTERNS:
        m = pattern.search(stem)
        if m:
            try:
                return extractor(m)
            except ValueError:
                continue
    return date(1900, 1, 1)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _normalize(stem: str) -> str:
    """Decode URL-encoding and lowercase."""
    return unquote(stem).lower().replace("%2c", "_").replace(" ", "_")


def is_amendment(stem: str) -> bool:
    return any(kw in stem for kw in AMENDMENT_KEYWORDS)


def extract_topic(stem: str) -> str | None:
    """Return canonical topic name or None if unrecognized."""
    for keyword, topic in TOPIC_KEYWORDS:
        if keyword in stem:
            return topic
    return None


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_pdfs(pdf_dir: Path) -> tuple[dict, list[Path], list[Path]]:
    """
    Returns:
        groups: topic -> {'originals': [Path], 'amendments': [Path]}
        ctdt_files: CTDT curriculum files (keep all khoa)
        skipped: unrecognized files
    """
    groups: dict[str, dict[str, list[Path]]] = {}
    ctdt_files: list[Path] = []
    skipped: list[Path] = []

    for pdf in sorted(pdf_dir.glob("*.pdf")):
        norm = _normalize(pdf.stem)

        # CTDT curriculum files — keep all khoa versions
        if norm.startswith("ctdt_"):
            ctdt_files.append(pdf)
            continue

        topic = extract_topic(norm)
        if topic is None:
            skipped.append(pdf)
            continue

        bucket = "amendments" if is_amendment(norm) else "originals"
        groups.setdefault(topic, {"originals": [], "amendments": []})
        groups[topic][bucket].append(pdf)

    return groups, ctdt_files, skipped


# ---------------------------------------------------------------------------
# System classification: chinh-quy / tu-xa / chung
# ---------------------------------------------------------------------------

# Keywords that clearly indicate hệ từ xa (distance learning)
_TU_XA_KEYWORDS = {"tu_xa", "dttx", "dao_tao_tu_xa", "hoc_tu_xa"}

# Keywords that clearly indicate hệ chính quy (regular full-time)
_CHINH_QUY_KEYWORDS = {
    "chinh_quy", "chinh_qui", "he_chinh_quy", "he_dai_hoc_chinh_quy",
    "he_cq", "dai_hoc_chinh_quy",
}


def classify_system(stem: str) -> str:
    """Return 'tu-xa', 'chinh-quy', or 'chung' based on filename keywords."""
    if any(kw in stem for kw in _TU_XA_KEYWORDS):
        return "tu-xa"
    if any(kw in stem for kw in _CHINH_QUY_KEYWORDS):
        return "chinh-quy"
    return "chung"


# ---------------------------------------------------------------------------
# Selection: latest original + latest amendment per topic
# ---------------------------------------------------------------------------

def select_files(groups: dict) -> list[tuple[Path, str]]:
    """Returns (path, reason) pairs for files to copy."""
    selected: list[tuple[Path, str]] = []

    for topic, buckets in sorted(groups.items()):
        originals = sorted(
            buckets["originals"],
            key=lambda p: extract_date(_normalize(p.stem)),
            reverse=True,
        )
        amendments = sorted(
            buckets["amendments"],
            key=lambda p: extract_date(_normalize(p.stem)),
            reverse=True,
        )

        if originals:
            selected.append((originals[0], f"{topic}/latest-original"))
        if amendments:
            selected.append((amendments[0], f"{topic}/latest-amendment"))

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy latest regulation PDFs to latest/ subfolder"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without copying files",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_PDF_DIR),
        help=f"Source PDF directory (default: {DEFAULT_PDF_DIR})",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    latest_dir = source_dir / "latest"

    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}")
        return

    groups, ctdt_files, skipped = group_pdfs(source_dir)
    selected = select_files(groups)

    # Classify each selected file into its system subdir
    # [(pdf, reason, subdir), ...]
    classified: list[tuple[Path, str, str]] = []
    for pdf, reason in selected:
        norm = _normalize(pdf.stem)
        subdir = classify_system(norm)
        classified.append((pdf, reason, subdir))
    # CTDT files are all chinh-quy programs
    for pdf in ctdt_files:
        classified.append((pdf, "ctdt/all-khoa", "chinh-quy"))

    total_to_copy = len(classified)
    counts = {"chinh-quy": 0, "tu-xa": 0, "chung": 0}
    for _, _, sd in classified:
        counts[sd] += 1

    # ── Summary header ──────────────────────────────────────────────────────
    prefix = "DRY RUN -- " if args.dry_run else ""
    print(f"\n{prefix}Filtering latest regulations")
    print("=" * 60)
    total_scanned = sum(
        len(b["originals"]) + len(b["amendments"]) for b in groups.values()
    ) + len(ctdt_files) + len(skipped)
    print(f"Scanned      : {total_scanned} PDFs")
    print(f"Topics       : {len(groups)} groups")
    print(f"Skipped      : {len(skipped)} unrecognized")
    print(f"To copy      : {total_to_copy} files")
    print(f"  chinh-quy  : {counts['chinh-quy']}")
    print(f"  tu-xa      : {counts['tu-xa']}")
    print(f"  chung      : {counts['chung']}")

    # ── Per-subdir breakdown ─────────────────────────────────────────────────
    for sd in ("chinh-quy", "tu-xa", "chung"):
        entries = [(p, r) for p, r, s in classified if s == sd]
        if not entries:
            continue
        print(f"\n" + "-" * 60)
        print(f"[{sd}] ({len(entries)} files):\n")
        for pdf, reason in entries:
            print(f"  [{reason}]")
            print(f"    {pdf.name}")

    # ── Skipped list (truncated) ──────────────────────────────────────────────
    if skipped:
        print("\n" + "-" * 60)
        print(f"Skipped ({len(skipped)} files):")
        for p in skipped[:15]:
            print(f"  {p.name}")
        if len(skipped) > 15:
            print(f"  ... and {len(skipped) - 15} more")

    # ── Copy ─────────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n[Dry run] Would copy {total_to_copy} files into {latest_dir}/{{chinh-quy,tu-xa,chung}}/")
        return

    # Remove old flat files in latest/ (if any) and recreate with subdirs
    for sd in ("chinh-quy", "tu-xa", "chung"):
        (latest_dir / sd).mkdir(parents=True, exist_ok=True)

    # Remove flat PDFs directly under latest/ (leftover from previous run)
    for old_pdf in latest_dir.glob("*.pdf"):
        old_pdf.unlink()

    copied = 0
    for pdf, _, sd in classified:
        shutil.copy2(pdf, latest_dir / sd / pdf.name)
        copied += 1

    print(f"\nDone. Copied {copied} files into {latest_dir}/{{chinh-quy,tu-xa,chung}}/")


if __name__ == "__main__":
    main()
