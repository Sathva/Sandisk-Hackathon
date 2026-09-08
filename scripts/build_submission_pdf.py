"""
Build Master Submission PDF for the SanDisk Hackathon.
Compiles a publication-grade, multi-page engineering document to:
reports/FINAL_SUBMISSION_REPORT.pdf
"""

import base64
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PLOTS_DIR = REPO_ROOT / "plots"
OUTPUT_PDF = REPORTS_DIR / "FINAL_SUBMISSION_REPORT.pdf"
TEMPLATE_PATH = REPORTS_DIR / "submission_template.html"
HTML_BUILD = REPORTS_DIR / "FINAL_SUBMISSION_REPORT_BUILD.html"


def img_to_b64(path: Path) -> str:
    if not path.exists():
        print(f"Warning: image {path} not found!")
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = path.suffix.lower().replace(".", "")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return f"data:{mime};base64,{data}"


def build_pdf():
    print(f"Loading template from: {TEMPLATE_PATH}")
    if not TEMPLATE_PATH.exists():
        print(f"Error: template {TEMPLATE_PATH} does not exist!")
        sys.exit(1)

    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    print("Encoding visual artifacts to base64...")
    replacements = {
        "__B64_OVERVIEW__": img_to_b64(FIGURES_DIR / "FINAL_INTERPRETABILITY_OVERVIEW.png"),
        "__B64_GLOBAL_SHAP__": img_to_b64(FIGURES_DIR / "27_global_shap_importance.png"),
        "__B64_DOMAIN_SHAP__": img_to_b64(FIGURES_DIR / "27b_domain_shap_contribution.png"),
        "__B64_SPATIAL_MAPS__": img_to_b64(FIGURES_DIR / "28_wafer_spatial_attribution_maps.png"),
        "__B64_BLOCK_PATTERNS__": img_to_b64(FIGURES_DIR / "29_block_pattern_analysis.png"),
        "__B64_A_TO_B__": img_to_b64(FIGURES_DIR / "30_a_to_b_block_gain.png"),
        "__B64_SAMPLE_WAFER__": img_to_b64(PLOTS_DIR / "6_sample_wafer_map.png"),
        "__B64_HYBRID_PR__": img_to_b64(FIGURES_DIR / "26_model_f_plus_cnn_comparison.png"),
        "__B64_PR_CURVES__": img_to_b64(PLOTS_DIR / "24_final_test_model_f_pr_curves.png"),
    }

    html = template
    for placeholder, b64_str in replacements.items():
        count = html.count(placeholder)
        print(f"  Replacing {placeholder}: {count} occurrences, {len(b64_str):,} chars")
        html = html.replace(placeholder, b64_str)

    with open(HTML_BUILD, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved populated HTML build to: {HTML_BUILD} ({len(html):,} bytes)")

    print("\nCompiling PDF via Headless Google Chrome...")
    t0 = time.time()
    cmd = [
        "google-chrome",
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUTPUT_PDF}",
        str(HTML_BUILD),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error generating PDF: {res.stderr}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"PDF compiled successfully in {elapsed:.2f}s: {OUTPUT_PDF}")

    # Inspect with pdfinfo
    info_res = subprocess.run(["pdfinfo", str(OUTPUT_PDF)], capture_output=True, text=True)
    print("\nPDFINFO METADATA:")
    for line in info_res.stdout.splitlines():
        if any(k in line for k in ["Title", "Pages", "Page size", "File size", "PDF version"]):
            print(f"  {line}")


if __name__ == "__main__":
    build_pdf()
