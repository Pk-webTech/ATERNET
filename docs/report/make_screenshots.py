"""
Capture real screenshots of the Streamlit dashboard for the report (Chapter 6).

    streamlit run app.py --server.port 8605 &      # in another terminal
    python docs/report/make_screenshots.py [port]

Uses headless Chrome through Selenium. Each page is opened by clicking its
entry in the sidebar, exactly as a reviewer would.
"""

import sys
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

OUT = Path(__file__).resolve().parent / "figures"
PORT = sys.argv[1] if len(sys.argv) > 1 else "8605"

# (sidebar label, file name, window height). Captured at a narrow width so the text stays legible when printed.
PAGES = [
    ("Overview", "shot_6_1_overview.png", 3200),
    ("Is ATERNET better?", "shot_6_2_is_aternet_better.png", 4600),
    ("How sure is the model?", "shot_6_3_how_sure.png", 3000),
    ("Built-in safety checks", "shot_6_4_safety_checks.png", 2600),
    ("India demo (simulated)", "shot_6_5_india_demo.png", 3600),
    ("Explore forecasts", "shot_6_6_explore.png", 2000),
    ("How the experts work", "shot_6_7_experts.png", 2600),
]


def trim_bottom(path, margin=40, sidebar=270):
    """Cut the empty white area below the last piece of content (looking only at the main area)."""
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)[:, sidebar:, :]
    rows = np.where((a < 245).any(axis=(1, 2)))[0]
    if len(rows):
        im.crop((0, 0, im.width, min(im.height, int(rows[-1]) + margin))).save(path)


def click_sidebar(driver, label):
    labels = driver.find_elements(By.CSS_SELECTOR, '[data-testid="stSidebar"] label')
    for el in labels:
        if el.text.strip() == label:
            driver.execute_script("arguments[0].click();", el)
            return True
    return False


def main():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--hide-scrollbars")
    opts.add_argument("--force-device-scale-factor=1")
    opts.add_argument("--window-size=1250,1000")
    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(f"http://localhost:{PORT}")
        time.sleep(7)
        for label, fname, height in PAGES:
            ok = click_sidebar(driver, label)
            driver.set_window_size(1250, height)
            time.sleep(7 if "better" in label or "India" in label else 5)
            driver.save_screenshot(str(OUT / fname))
            trim_bottom(OUT / fname)
            print(("OK  " if ok else "MISS"), label, "->", fname)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
