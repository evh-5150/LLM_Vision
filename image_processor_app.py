import streamlit as st
import pydicom
import numpy as np
import cv2
from llama_cpp import Llama
import io
import inspect
import re

# --- 定数設定，LLM ロード，画像処理関数群，FUNCTION_MAP，LLM 連携関数 ---
# --- 定数設定 ---
MODEL_PATH = "./Llama-3-ELYZA-JP-8B-Q4_K_M.gguf"
WINDOW_PRESETS = {
    "カスタム": {}, "腹部": {"ww": 400, "wl": 50}, "骨": {"ww": 2000, "wl": 600},
    "肺": {"ww": 1500, "wl": -600}, "脳": {"ww": 80, "wl": 40},
}

# 形態素変換処理を1つに統合
PROCESSING_KNOWLEDGE = {
    # 平滑化・ノイズ除去系
    "noise_removal": {
        "title": "ノイズ除去 (ガウシアン)", "keywords": ["ノイズ", "滑らか", "スムーズ", "ザラザラ", "平滑化", "ガウシアン", "ぼかす", "ノイズ除去"]
    },
    "median_filter": {
        "title": "ノイズ除去 (メディアン)", "keywords": ["メディアン", "ごま塩ノイズ", "塩胡椒ノイズ", "中間値フィルタ"]
    },
    "bilateral_filter": {
        "title": "エッジ保持平滑化", "keywords": ["エッジ保持平滑化", "エッジ保持", "バイラテラル", "bilateral", "輪郭を残して平滑化", "輪郭を残してノイズ除去"]
    },
    "blur": {
        "title": "ぼかし (平均化)", "keywords": ["ぼかし", "ブラー", "平均化"]
    },
    # 鮮鋭化・エッジ強調系
    "edge_enhancement": {
        "title": "エッジ強調 (鮮鋭化)", "keywords": ["エッジ強調", "シャープ", "鮮明", "くっきり", "はっきり", "鮮鋭度", "鮮鋭化", "シャープネス", "先鋭化"]
    },
    # エッジ検出系
    "canny_edge": {
        "title": "エッジ検出 (Canny)", "keywords": ["canny", "キャニー", "エッジ検出", "輪郭抽出", "輪郭だけ", "線画", "エッジ抽出", "境界抽出"]
    },
    "sobel_filter": {
        "title": "エッジ検出 (Sobel)", "keywords": ["sobel", "ソーベル", "勾配", "一次微分"]
    },
    "laplacian_filter": {
        "title": "エッジ検出 (ラプラシアン)", "keywords": ["laplacian", "ラプラシアン", "二次微分"]
    },
    # 階調・コントラスト系
    "brightness_contrast": {
        "title": "明るさ・コントラスト", "keywords": ["コントラスト", "明るさ", "明るく", "暗く", "白く", "黒く", "見やすく", "ウィンドウ", "ww", "wl", "レベル"]
    },
    "clahe": {
        "title": "コントラスト強調 (CLAHE)", "keywords": ["clahe", "適応的ヒストグラム", "コントラスト強調", "明瞭化"]
    },
    "histogram_equalization": {
        "title": "ヒストグラム平坦化", "keywords": ["ヒストグラム平坦化", "ヒストグラム均一化"]
    },
    "gamma_correction": {
        "title": "ガンマ補正", "keywords": ["ガンマ補正", "ガンマ"]
    },
    "invert": {
        "title": "ネガポジ反転", "keywords": ["ネガポジ", "色を反転", "階調反転", "反転表示", "ネガポジ反転"]
    },
    # 閾値処理・二値化系
    "thresholding": {
        "title": "二値化 (単純閾値)", "keywords": ["二値化", "閾値", "白黒", "単純閾値"]
    },
    "adaptive_thresholding": {
        "title": "二値化 (適応的閾値)", "keywords": ["適応的閾値", "局所的二値化"]
    },
    # 形態学的処理 (統合)
    "morphological": {
        "title": "形態素変換",
        "keywords": ["形態素", "収縮", "細くする", "侵食", "erode", "膨張", "太くする", "dilate", "オープニング", "open", "クロージング", "close", "白点ノイズ除去", "黒点ノイズ除去"]
    },
    # 幾何学的変換
    "zoom_in": { "title": "拡大", "keywords": ["拡大", "ズーム", "大きく", "アップ", "zoom"] },
    "rotate": { "title": "回転", "keywords": ["回転", "回して", "向き", "rotate"] },
    "flip": { "title": "反転", "keywords": ["反転", "裏返し", "ミラー", "flip"] },
    # 特徴抽出・その他
    "draw_contours": { "title": "輪郭描画", "keywords": ["輪郭を描画", "輪郭の描画"] },
    "hough_lines": { "title": "直線検出 (ハフ変換)", "keywords": ["直線検出", "ハフ変換", "hough"] },
    "detect_circles": { "title": "円検出 (ハフ変換)", "keywords": ["円検出", "丸", "サークル"] },
    "fourier_transform": { "title": "フーリエ変換スペクトル", "keywords": ["フーリエ", "周波数", "スペクトル", "fft"] }
}

# --- LLM のロード ---
@st.cache_resource
def load_model():
    try:
        st.info("LLM をロード中です．初回起動時は時間がかかります...")
        llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096, verbose=False, chat_format="llama-3")
        st.success("LLM のロードが完了しました．")
        return llm
    except Exception as e:
        st.error(f"モデルのロードに失敗しました: {e}．モデルファイル '{MODEL_PATH}' が正しい場所にありますか？")
        return None
llm = load_model()

# --- 画像処理関数群 ---
def process_adaptive_thresholding(img_in, block_size=11, C=2):
    gray_8bit = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    bs = int(block_size); bs = bs if bs % 2 else bs + 1
    return cv2.adaptiveThreshold(gray_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, int(C))
def process_bilateral_filter(img_in, d=9, sigma_color=75.0, sigma_space=75.0):
    img_float32 = img_in.astype(np.float32) / 65535.0; sigma_color_normalized = sigma_color / 255.0
    filtered_float32 = cv2.bilateralFilter(img_float32, int(d), sigma_color_normalized, sigma_space); return np.clip(filtered_float32 * 65535.0, 0, 65535).astype(np.uint16)
def process_blur(img_in, kernel_size=9):
    k = int(kernel_size); return cv2.GaussianBlur(img_in, (k if k % 2 else k + 1, k if k % 2 else k + 1), 0)
def process_brightness_contrast(img_in, alpha=1.0, beta=0.0): return np.clip(img_in.astype(np.float64) * alpha + beta, 0, 65535).astype(np.uint16)
def process_canny_edge(img_in, threshold1=100, threshold2=200):
    gray_8bit = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U); return cv2.Canny(gray_8bit, int(threshold1), int(threshold2))
def process_clahe(img_in, clip_limit=2.0, tile_grid_size=8):
    gray_8bit = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    return clahe.apply(gray_8bit)
def process_detect_bounding_box(img_in):
    output_img = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U); output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
    gray_8bit = output_img[:,:,0]; _, thresh = cv2.threshold(gray_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours: largest_contour = max(contours, key=cv2.contourArea); x, y, w, h = cv2.boundingRect(largest_contour); cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return output_img
def process_detect_circles(img_in, dp=1.2, min_dist=100, param1=50, param2=30, min_radius=10, max_radius=100):
    output_img = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U); output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
    gray_8bit = output_img[:,:,0]; gray_8bit = cv2.medianBlur(gray_8bit, 5)
    circles = cv2.HoughCircles(gray_8bit, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles));
        for i in circles[0, :]: center = (i[0], i[1]); radius = i[2]; cv2.circle(output_img, center, radius, (255, 0, 255), 3); cv2.circle(output_img, center, 2, (255, 255, 0), 3)
    return output_img
def process_draw_contours(img_in, threshold=127.0, thickness=2):
    gray_8bit = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U); _, thresh_img = cv2.threshold(gray_8bit, int(threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE); output_img = cv2.cvtColor(gray_8bit, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_img, contours, -1, (0, 255, 0), int(thickness)); return output_img
def process_edge_enhancement(img_in, strength=1.5):
    img_float = img_in.astype(np.float64); blurred_float = cv2.GaussianBlur(img_float, (3, 3), 0)
    weight = float(strength) - 1.0; enhanced_float = np.clip(img_float + weight * (img_float - blurred_float), 0, 65535)
    return enhanced_float.astype(np.uint16)
def process_flip(img_in, direction="左右反転"):
    flip_map = {"左右反転": 1, "上下反転": 0, "両方": -1}; return cv2.flip(img_in, flip_map[direction])
def process_fourier_transform(img_in):
    img_float = img_in.astype(np.float32)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
def process_gamma_correction(img_in, gamma=1.0):
    g = float(gamma) if gamma > 0 else 0.1
    inv_gamma = 1.0 / g
    table = np.array([((i / 65535.0) ** inv_gamma) * 65535 for i in np.arange(0, 65536)]).astype("uint16")
    return cv2.LUT(img_in, table)
def process_histogram_equalization(img_in):
    gray_8bit = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.equalizeHist(gray_8bit)
def process_hough_lines(img_in, rho=1, theta_deg=1, threshold=100, min_line_length=50, max_line_gap=10):
    gray_8bit = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    edges = cv2.Canny(gray_8bit, 50, 150, apertureSize=3)
    output_img = cv2.cvtColor(gray_8bit, cv2.COLOR_GRAY2BGR)
    theta_rad = np.deg2rad(float(theta_deg))
    lines = cv2.HoughLinesP(edges, float(rho), theta_rad, int(threshold), minLineLength=int(min_line_length), maxLineGap=int(max_line_gap))
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]; cv2.line(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return output_img
def process_invert(img_in):
    if img_in.dtype == np.uint16: return 65535 - img_in
    else: return 255 - img_in
def process_laplacian_filter(img_in, ksize=3):
    gray_8bit = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    k = int(ksize); k = k if k % 2 else k + 1
    laplacian = cv2.Laplacian(gray_8bit, cv2.CV_64F, ksize=k)
    return cv2.convertScaleAbs(laplacian)
def process_median_filter(img_in, kernel_size=5):
    k = int(kernel_size); return cv2.medianBlur(img_in, k if k % 2 else k + 1)
def process_morphological(img_in, operation="オープニング", kernel_size=5):
    gray_8bit = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U); kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
    op_map = {"オープニング": cv2.MORPH_OPEN, "クロージング": cv2.MORPH_CLOSE, "膨張": cv2.MORPH_DILATE, "収縮": cv2.MORPH_ERODE}; return cv2.morphologyEx(gray_8bit, op_map[operation], kernel)
def process_noise_removal(img_in, kernel_size=5):
    k = int(kernel_size); return cv2.GaussianBlur(img_in, (k if k % 2 else k + 1, k if k % 2 else k + 1), 0)
def process_rotate(img_in, angle="90度 右"):
    angle_map = {"90度 右": cv2.ROTATE_90_CLOCKWISE, "90度 左": cv2.ROTATE_90_COUNTERCLOCKWISE, "180度": cv2.ROTATE_180}; return cv2.rotate(img_in, angle_map[angle])
def process_sobel_filter(img_in, ksize=3):
    gray_8bit = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    k = int(ksize); k = k if k % 2 else k + 1
    sobelx = cv2.Sobel(gray_8bit, cv2.CV_64F, 1, 0, ksize=k)
    sobely = cv2.Sobel(gray_8bit, cv2.CV_64F, 0, 1, ksize=k)
    sobel_abs = np.sqrt(sobelx**2 + sobely**2)
    return cv2.normalize(sobel_abs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
def process_thresholding(img_in, threshold_value=127.0):
    gray_8bit = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U); _, thresh_img = cv2.threshold(gray_8bit, int(threshold_value), 255, cv2.THRESH_BINARY); return thresh_img
def process_zoom_in(img_in, scale=2.0, offset_x=0.0, offset_y=0.0):
    h, w = img_in.shape[:2]; center_x = w // 2 + int(offset_x * w / 2); center_y = h // 2 + int(offset_y * h / 2)
    new_w, new_h = int(w / scale), int(h / scale); x1 = max(0, center_x - new_w // 2); y1 = max(0, center_y - new_h // 2)
    x2, y2 = min(w, x1 + new_w), min(h, y1 + new_h)
    if x2 - x1 < new_w: x1 = x2 - new_w
    if y2 - y1 < new_h: y1 = y2 - new_h
    cropped = img_in[y1:y2, x1:x2]; return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)


FUNCTION_MAP = {k: {"func": globals()[f"process_{k}"], **v} for k, v in PROCESSING_KNOWLEDGE.items()}

def get_tasks_from_llm(user_instruction):
    if llm is None:
        st.warning("LLM がロードされていません．キーワード検索で代用します．");
        tasks = [name for name, info in FUNCTION_MAP.items() if any(keyword in user_instruction for keyword in info['keywords'])];
        # キーワード検索でも重複する可能性があるため，順序を保持しつつ重複を削除
        return list(dict.fromkeys(tasks)) if tasks else []

    task_list = "\n".join([f"- `{name}`: {info['title']}" for name, info in FUNCTION_MAP.items()])
    
    # 改善されたシステムプロンプト
    system_prompt = f"""あなたはユーザーの指示を分析し，実行すべき処理を順番に特定する専門家です．以下のルールに従って，タスクリストから適切なタスク名を1つ以上選び，カンマ区切りのリストとして返答してください．

### ルール ###
1. ユーザーの指示に最も合致するタスクを，必要最小限の数だけ選んでください．
2. 似たような目的のタスクが複数ある場合，より基本的で一般的なタスクを選択してください．
   - 例えば，「平滑化」「ノイズ除去」という一般的な指示には，まず`noise_removal`を選択します．
   - 「エッジを保持しながら平滑化」「バイラテラルフィルタ」のように，ユーザーが明確に高度な処理を指定した場合にのみ，`bilateral_filter`を選択してください．
3. ユーザーが意図していない余計な処理は絶対に追加しないでください．

### タスクリスト ###
{task_list}

### 指示と回答の例 ###
- 指示: 「ノイズを消して」 -> 回答: `noise_removal`
- 指示: 「平滑化してエッジ検出」 -> 回答: `noise_removal,canny_edge`
- 指示: 「輪郭をはっきりさせてから，エッジを検出して」 -> 回答: `edge_enhancement,canny_edge`
- 指示: 「拡大して，滑らかにしてほしい」 -> 回答: `zoom_in,noise_removal`
- 指示: 「エッジは残して滑らかに」 -> 回答: `bilateral_filter`
- 指示: 「画像を平滑化して」 -> 回答: `noise_removal`
- 指示: 「収縮して」 -> 回答: `morphological`
"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_instruction}]
    try:
        output = llm.create_chat_completion(messages=messages, max_tokens=100, temperature=0.0)
        content = output['choices'][0]['message']['content'].strip()
        st.session_state.debug_llm_response = content
        all_task_names = list(FUNCTION_MAP.keys())
        pattern = r'\b(' + '|'.join(all_task_names) + r')\b'
        found_tasks = re.findall(pattern, content)
        # LLM が誤って重複したタスクを返す可能性に備え，ここで重複を削除する
        return list(dict.fromkeys(found_tasks)) if found_tasks else []
    except Exception as e:
        st.error(f"AI の応答処理中に予期せぬエラーが発生しました: {e}"); return []


def convert_to_16bit_gray(pixel_array):
    if len(pixel_array.shape) > 2: pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2GRAY)
    if pixel_array.dtype != np.uint16: pixel_array = cv2.normalize(pixel_array.astype(float), None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
    return pixel_array
def apply_ww_wl_and_convert_to_bgr(pixel_array, ww, wl, enhance=False):
    if pixel_array is None: return None
    if pixel_array.dtype == np.uint8: return cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR) if len(pixel_array.shape) == 2 else pixel_array
    min_val, max_val = wl - (ww / 2), wl + (ww / 2); img_float = pixel_array.astype(np.float64); img_float = np.clip(img_float, min_val, max_val)
    if ww == 0: ww = 1
    img_8bit_gray = ((img_float - min_val) / ww * 255.0).astype(np.uint8)
    if enhance: clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)); img_8bit_gray = clahe.apply(img_8bit_gray)
    return cv2.cvtColor(img_8bit_gray, cv2.COLOR_GRAY2BGR)


st.set_page_config(layout="wide", page_title="LLM 医用画像処理")
st.title("LLM 医用画像処理システム")

PARAM_CONFIG = {
    "noise_removal": {"kernel_size": {"label": "除去強度", "min": 3, "max": 41, "step": 2, "default": 15}},
    "median_filter": {"kernel_size": {"label": "カーネルサイズ", "min": 3, "max": 41, "step": 2, "default": 5}},
    "bilateral_filter": {"d": {"label": "近傍領域の直径", "min": 1, "max": 15, "step": 2, "default": 9}, "sigma_color": {"label": "輝度差の許容範囲", "min": 1.0, "max": 255.0, "default": 75.0}, "sigma_space": {"label": "空間距離の許容範囲", "min": 1.0, "max": 255.0, "default": 75.0}},
    "blur": {"kernel_size": {"label": "ぼかし強度", "min": 3, "max": 41, "step": 2, "default": 15}},
    "edge_enhancement": {"strength": {"label": "強調度", "min": 1.1, "max": 5.0, "step": 0.1, "default": 2.5}},
    "canny_edge": {"threshold1": {"label": "下位閾値", "min": 0, "max": 500, "default": 100}, "threshold2": {"label": "上位閾値", "min": 0, "max": 500, "default": 200}},
    "sobel_filter": {"ksize": {"label": "カーネルサイズ", "min": 3, "max": 31, "step": 2, "default": 3}},
    "laplacian_filter": {"ksize": {"label": "カーネルサイズ", "min": 3, "max": 31, "step": 2, "default": 3}},
    "brightness_contrast": {"alpha": {"label": "コントラスト", "min": 0.1, "max": 3.0, "step": 0.1, "default": 1.0}, "beta": {"label": "明るさ", "min": -10000.0, "max": 10000.0, "step": 100.0, "default": 0.0}},
    "clahe": {"clip_limit": {"label": "クリップ上限", "min": 1.0, "max": 10.0, "step": 0.5, "default": 2.0}, "tile_grid_size": {"label": "タイルサイズ", "min": 2, "max": 16, "default": 8}},
    "gamma_correction": {"gamma": {"label": "ガンマ値", "min": 0.1, "max": 3.0, "step": 0.1, "default": 1.0}},
    "thresholding": {"threshold_value": {"label": "閾値", "min": 0.0, "max": 255.0, "default": 127.0}},
    "adaptive_thresholding": {"block_size": {"label": "ブロックサイズ", "min": 3, "max": 51, "step": 2, "default": 11}, "C": {"label": "定数C", "min": -10, "max": 10, "default": 2}},
    "morphological": {"operation": {"type": "selectbox", "label": "操作タイプ", "options": ["オープニング", "クロージング", "膨張", "収縮"], "default": "オープニング"}, "kernel_size": {"label": "カーネルサイズ", "min": 3, "max": 21, "step": 2, "default": 5}},
    "zoom_in": {"scale": {"label": "拡大率", "min": 1.1, "max": 10.0, "step": 0.1, "default": 2.0}, "offset_x": {"label": "X座標オフセット", "min": -1.0, "max": 1.0, "step": 0.05, "default": 0.0}, "offset_y": {"label": "Y座標オフセット", "min": -1.0, "max": 1.0, "step": 0.05, "default": 0.0}},
    "rotate": {"angle": {"type": "selectbox", "label": "回転角度", "options": ["90度 右", "90度 左", "180度"], "default": "90度 右"}},
    "flip": {"direction": {"type": "selectbox", "label": "反転方向", "options": ["左右反転", "上下反転", "両方"], "default": "左右反転"}},
    "draw_contours": {"threshold": {"label": "二値化閾値", "min": 0.0, "max": 255.0, "default": 127.0}, "thickness": {"label": "輪郭の太さ", "min": 1, "max": 10, "default": 2}},
    "hough_lines": {"rho": {"label": "Rho (距離分解能)", "min": 1, "max": 10, "default": 1}, "theta_deg": {"label": "Theta (角度分解能)", "min": 1, "max": 10, "default": 1}, "threshold": {"label": "閾値", "min": 10, "max": 200, "default": 100}, "min_line_length": {"label": "最小の線長", "min": 10, "max": 200, "default": 50}, "max_line_gap": {"label": "最大線間ギャップ", "min": 1, "max": 50, "default": 10}},
    "detect_circles": {"dp": {"label": "dp", "min": 1.0, "max": 2.0, "step": 0.1, "default": 1.2}, "min_dist": {"label": "円同士の最小距離", "min": 10, "max": 200, "default": 100}, "param1": {"label": "Canny上位閾値", "min": 10, "max": 200, "default": 50}, "param2": {"label": "検出感度", "min": 10, "max": 100, "default": 30}, "min_radius": {"label": "最小半径", "min": 0, "max": 200, "default": 10}, "max_radius": {"label": "最大半径", "min": 0, "max": 500, "default": 100}},
}
def reset_all_params_to_default():
    for task, params in PARAM_CONFIG.items():
        for param_name, config in params.items(): st.session_state[f"p_{task}_{param_name}"] = config["default"]

def handle_instruction_submit():
    """指示が送信されたときに実行されるコールバック関数"""
    if 'debug_llm_response' in st.session_state:
        del st.session_state['debug_llm_response']

    instruction = st.session_state.get("instruction_input", "").strip()

    if not instruction:
        st.session_state.last_tasks = []
        reset_all_params_to_default()
        st.toast("画像処理をリセットしました．")
    else:
        st.toast("AIがタスクを解析中...")
        task_names = get_tasks_from_llm(instruction)
        
        # get_tasks_from_llm内で重複削除は行われているが，念のためここでもチェック
        if task_names:
            task_names = list(dict.fromkeys(task_names))

        if st.session_state.get("last_tasks", []) != task_names:
            reset_all_params_to_default()

        st.session_state.last_tasks = task_names

        if task_names:
            titles = " → ".join([FUNCTION_MAP[t]['title'] for t in task_names])
            st.toast(f"✅ パイプライン設定: {titles}")
        else:
            st.session_state.show_task_not_found_error = True

# --- 状態の初期化 ---
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.last_tasks = []
    st.session_state.last_file_id = None
    st.session_state.p_ww = 400.0
    st.session_state.p_wl = 50.0
    st.session_state.preset = "カスタム"
    reset_all_params_to_default()

# --- ファイルアップロードとDICOM読み込み ---
uploaded_file = st.file_uploader("16-bit DICOM画像を選択", type=["dcm", "dicom"])
original_16bit_gray = None
if uploaded_file:
    if st.session_state.last_file_id != uploaded_file.file_id:
        st.session_state.last_tasks = []
        st.session_state.dicom_ww_wl_initialized = False
        if 'instruction_input' in st.session_state: st.session_state.instruction_input = ""
        st.session_state.last_file_id = uploaded_file.file_id
    try:
        ds = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()))
        pixel_array = ds.pixel_array * ds.get("RescaleSlope", 1) + ds.get("RescaleIntercept", 0)
        original_16bit_gray = convert_to_16bit_gray(pixel_array)
        if not st.session_state.get('dicom_ww_wl_initialized', False):
            try:
                wl_val = ds.WindowCenter[0] if isinstance(ds.WindowCenter, pydicom.multival.MultiValue) else ds.WindowCenter
                ww_val = ds.WindowWidth[0] if isinstance(ds.WindowWidth, pydicom.multival.MultiValue) else ds.WindowWidth
                st.session_state.initial_wl, st.session_state.initial_ww = float(wl_val), float(ww_val)
            except (AttributeError, TypeError, IndexError):
                min_val, max_val = np.min(original_16bit_gray), np.max(original_16bit_gray)
                st.session_state.initial_ww = float(max_val - min_val if max_val > min_val else 1)
                st.session_state.initial_wl = float(min_val + st.session_state.initial_ww / 2)
            
            st.session_state.p_wl, st.session_state.p_ww = st.session_state.initial_wl, st.session_state.initial_ww
            st.session_state.p_wl_slider, st.session_state.p_ww_slider = st.session_state.initial_wl, st.session_state.initial_ww
            st.session_state.dicom_ww_wl_initialized = True
    except Exception as e:
        st.error(f"DICOM読み込みエラー: {e}")
        original_16bit_gray = None

# --- サイドバーUI ---
with st.sidebar:
    st.header("🖼️ 表示オプション")
    enhance_default = st.checkbox("画質を向上させる (CLAHE)", value=False)
    st.header("表示調整 (WW/WL)")
    is_disabled = (original_16bit_gray is None)
    
    def reset_ww_wl():
        if "initial_ww" in st.session_state and "initial_wl" in st.session_state:
            st.session_state.p_ww, st.session_state.p_wl = st.session_state.initial_ww, st.session_state.initial_wl
            st.session_state.p_ww_slider, st.session_state.p_wl_slider = st.session_state.initial_ww, st.session_state.initial_wl

    def on_preset_change():
        preset_key = st.session_state.preset
        if preset_key != "カスタム" and preset_key in WINDOW_PRESETS:
            st.session_state.p_ww, st.session_state.p_wl = float(WINDOW_PRESETS[preset_key]["ww"]), float(WINDOW_PRESETS[preset_key]["wl"])
            st.session_state.p_ww_slider, st.session_state.p_wl_slider = st.session_state.p_ww, st.session_state.p_wl

    st.selectbox("プリセット", list(WINDOW_PRESETS.keys()), key="preset", on_change=on_preset_change, disabled=is_disabled)

    WW_MIN, WW_MAX, WL_MIN, WL_MAX = 1.0, 65536.0, -32768.0, 65536.0
    
    def sync_ww_from_slider(): st.session_state.p_ww = st.session_state.p_ww_slider; st.session_state.preset = "カスタム"
    def sync_ww_from_number_input(): st.session_state.p_ww_slider = st.session_state.p_ww
    def sync_wl_from_slider(): st.session_state.p_wl = st.session_state.p_wl_slider; st.session_state.preset = "カスタム"
    def sync_wl_from_number_input(): st.session_state.p_wl_slider = st.session_state.p_wl

    if 'p_ww_slider' not in st.session_state: st.session_state.p_ww_slider = st.session_state.p_ww
    if 'p_wl_slider' not in st.session_state: st.session_state.p_wl_slider = st.session_state.p_wl

    col1, col2 = st.columns([0.4, 0.6])
    with col1: st.number_input("WW", WW_MIN, WW_MAX, key="p_ww", disabled=is_disabled, step=10.0, format="%.1f", on_change=sync_ww_from_number_input)
    with col2: st.slider(" ", WW_MIN, WW_MAX, key="p_ww_slider", label_visibility="collapsed", disabled=is_disabled, on_change=sync_ww_from_slider)
    col3, col4 = st.columns([0.4, 0.6])
    with col3: st.number_input("WL", WL_MIN, WL_MAX, key="p_wl", disabled=is_disabled, step=10.0, format="%.1f", on_change=sync_wl_from_number_input)
    with col4: st.slider(" ", WL_MIN, WL_MAX, key="p_wl_slider", label_visibility="collapsed", disabled=is_disabled, on_change=sync_wl_from_slider)
    st.button("WW/WLを初期値に戻す", on_click=reset_ww_wl, disabled=is_disabled, use_container_width=True)
    
    st.header("⚙️ 処理パラメータ調整")
    tasks = st.session_state.get("last_tasks", [])
    if original_16bit_gray is not None and tasks:
        for i, task in enumerate(tasks):
            task_config = PARAM_CONFIG.get(task)
            with st.expander(f"ステップ {i+1}: {FUNCTION_MAP[task]['title']}", expanded=True):
                if task_config:
                    for param_name, config in task_config.items():
                        session_key = f"p_{task}_{param_name}"
                        if config.get("type") == "selectbox": st.selectbox(config["label"], config["options"], key=session_key)
                        else: st.slider(config["label"], config["min"], config["max"], step=config.get("step", 0.1 if isinstance(config["min"], float) else 1), key=session_key)
                else: st.info("調整可能なパラメータはありません．")
        st.button("全てのパラメータを初期値に戻す", on_click=reset_all_params_to_default, use_container_width=True)
    else: st.info("処理を選択すると，調整項目が表示されます．")

# --- メインエリアのUIと処理ロジック ---
if original_16bit_gray is not None:
    st.text_input(
        "画像処理の指示",
        placeholder="例: 拡大して平滑化 / 空欄でリセット",
        key="instruction_input",
        label_visibility="collapsed",
        on_change=handle_instruction_submit
    )
    st.button(
        label='画像処理を実行',
        use_container_width=True,
        disabled=(llm is None),
        on_click=handle_instruction_submit
    )
    
    if st.session_state.get("show_task_not_found_error", False):
        st.error("指示内容に対応する処理が見つかりませんでした．")
        st.session_state.show_task_not_found_error = False
    
    if 'debug_llm_response' in st.session_state and st.session_state.debug_llm_response:
        with st.expander("デバッグ情報: AIからの生の応答"):
            st.text(st.session_state.debug_llm_response)

    if llm is None: st.warning("LLM がロードされていないため，指示による画像処理は無効です．")

    tasks_to_run = st.session_state.get("last_tasks", [])
    display_original_image = original_16bit_gray
    if tasks_to_run and tasks_to_run[0] == 'zoom_in':
        zoom_params = {p_name: st.session_state[f"p_zoom_in_{p_name}"]
                       for p_name in inspect.signature(process_zoom_in).parameters
                       if p_name != 'img_in' and f"p_zoom_in_{p_name}" in st.session_state}
        display_original_image = process_zoom_in(original_16bit_gray, **zoom_params)

    processed_image = original_16bit_gray
    header_text = "処理後"
    if tasks_to_run:
        header_text = f"処理後: " + " → ".join([FUNCTION_MAP[t]['title'] for t in tasks_to_run])
    temp_image = original_16bit_gray.copy()
    if tasks_to_run:
        try:
            for task in tasks_to_run:
                processing_function = FUNCTION_MAP[task]["func"]
                params_to_pass = {param_name: st.session_state[f"p_{task}_{param_name}"]
                                  for param_name in inspect.signature(processing_function).parameters
                                  if param_name != 'img_in' and f"p_{task}_{param_name}" in st.session_state}
                temp_image = processing_function(temp_image, **params_to_pass)
            processed_image = temp_image
        except Exception as e:
            st.error(f"画像処理中にエラーが発生しました: {e}"); st.exception(e)
            
    col1, col2 = st.columns(2)
    ww, wl = st.session_state.p_ww, st.session_state.p_wl
    
    # st.headerの代わりにst.markdownを使い，CSSでスタイルを調整
    header_style = "font-weight: 600; font-size: 1.7rem; margin-bottom: 1rem;"

    with col1:
        # 左側のヘッダー
        st.markdown(f'<p style="{header_style}">処理前</p>', unsafe_allow_html=True)
        st.image(apply_ww_wl_and_convert_to_bgr(display_original_image, ww, wl, enhance=enhance_default), use_container_width=True)
    
    with col2:
        # 右側のヘッダーに改行防止と省略(...)のスタイルを追加
        # title属性に全文を入れることで，マウスオーバーで全文を確認できる
        st.markdown(
            f'<p style="{header_style} white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{header_text}">{header_text}</p>',
            unsafe_allow_html=True
        )
        st.image(apply_ww_wl_and_convert_to_bgr(processed_image, ww, wl, enhance=enhance_default), use_container_width=True)
else:
    if not uploaded_file: st.info("DICOM ファイルをアップロードして，画像処理を開始してください．")