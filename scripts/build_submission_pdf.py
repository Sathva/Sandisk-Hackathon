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


def find_browser() -> str:
    """
    Locate a Chromium-family browser for headless PDF printing.

    The original hard-coded ``google-chrome`` is the Linux binary name and raises
    FileNotFoundError on Windows and macOS, where the executable is called
    ``chrome.exe`` / ``Google Chrome`` and lives outside PATH. Edge is accepted as a
    fallback because it is Chromium-based and supports the same
    ``--print-to-pdf`` flag, and it ships with Windows.
    """
    import shutil

    env = os.environ.get("CHROME_PATH")
    if env and Path(env).exists():
        return env

    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found

    candidates = [
        # Windows
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        # macOS
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        # Linux
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/snap/bin/chromium",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c

    print(
        "Error: no Chromium-family browser found for PDF printing.\n"
        "Install Google Chrome, or set the CHROME_PATH environment variable to the\n"
        "browser executable and re-run."
    )
    sys.exit(1)


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
        # Section 17A/17B: interpretability of the two OFFICIAL deliverables, produced
        # by src/interpretability/model_ab_attribution.py.
        "__B64_MODEL_A_SPATIAL__": img_to_b64(FIGURES_DIR / "40_model_a_spatial_contribution.png"),
        "__B64_MODEL_B_BLOCK__": img_to_b64(FIGURES_DIR / "41_model_b_block_attribution.png"),
    }

    html = template
    for placeholder, b64_str in replacements.items():
        count = html.count(placeholder)
        print(f"  Replacing {placeholder}: {count} occurrences, {len(b64_str):,} chars")
        html = html.replace(placeholder, b64_str)

    with open(HTML_BUILD, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved populated HTML build to: {HTML_BUILD} ({len(html):,} bytes)")

    browser = find_browser()
    print(f"\nCompiling PDF via headless browser: {browser}")
    t0 = time.time()
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUTPUT_PDF}",
        HTML_BUILD.as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error generating PDF: {res.stderr}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"PDF compiled successfully in {elapsed:.2f}s: {OUTPUT_PDF}")

    # Inspect with pdfinfo. It ships with poppler-utils and is normally absent on
    # Windows, so this is a best-effort report rather than a build step -- the PDF is
    # already written by this point and a missing pdfinfo must not fail the build.
    import shutil as _shutil

    size = OUTPUT_PDF.stat().st_size if OUTPUT_PDF.exists() else 0
    if _shutil.which("pdfinfo") is None:
        print(f"\nPDF size: {size:,} bytes")
        print("  (pdfinfo not on PATH; skipping metadata inspection)")
        return

    info_res = subprocess.run(["pdfinfo", str(OUTPUT_PDF)], capture_output=True, text=True)
    print("\nPDFINFO METADATA:")
    for line in info_res.stdout.splitlines():
        if any(k in line for k in ["Title", "Pages", "Page size", "File size", "PDF version"]):
            print(f"  {line}")


if __name__ == "__main__":
    build_pdf()
