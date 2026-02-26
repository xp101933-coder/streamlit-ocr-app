import streamlit as st
import google.generativeai as genai
from PIL import Image
import os

# --- Constants ---
MAX_FILE_SIZE_MB = 5
MAX_IMAGE_DIMENSION = 1024

# --- Prompt Templates ---
PROMPT_TEMPLATES = {
    "標準（テキストのみ）": "この画像に含まれているすべてのテキストを正確に抽出して出力してください。テキスト以外の説明文、マークダウン装飾、挨拶などは一切不要です。",
    "Markdown（表や構造の保持）": "この画像に含まれている内容を読み取り、表データや見出し構造があれば可能な限りMarkdown形式（テーブルやヘッダー記法等）を維持して出力してください。",
    "翻訳（英語から日本語）": "画像内の英語テキストを読み取り、自然な日本語に翻訳して出力してください。翻訳結果のみを出力し、元の英語や余計な説明は不要です。",
    "要約": "画像内に書かれている文章の重要なポイントを抽出し、簡潔に要約して箇条書きで出力してください。"
}

# --- Page Config ---
st.set_page_config(page_title="画像OCRツール", layout="wide") # Changed layout to wide for 2-column viewing
st.title("画像文字起こし (OCR) ツール")

# --- Session State Initialization ---
if 'ocr_result' not in st.session_state:
    st.session_state.ocr_result = ""

# --- Helper Functions ---
def process_image(uploaded_file):
    """
    Check size and resize image if necessary.
    Returns: PIL Image object or None if size exceeded.
    """
    # Size check (uploaded_file is a BytesIO-like object)
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return None, f"ファイルサイズが{MAX_FILE_SIZE_MB}MBを超えています。"

    try:
        image = Image.open(uploaded_file)
        # Convert to RGB if necessary (e.g., RGBA or P)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
            
        width, height = image.size
        # Resize if the longest edge exceeds MAX_IMAGE_DIMENSION
        if max(width, height) > MAX_IMAGE_DIMENSION:
            if width > height:
                new_width = MAX_IMAGE_DIMENSION
                new_height = int(height * (MAX_IMAGE_DIMENSION / width))
            else:
                new_height = MAX_IMAGE_DIMENSION
                new_width = int(width * (MAX_IMAGE_DIMENSION / height))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image, None
    except Exception as e:
        return None, f"画像の処理に失敗しました: {str(e)}"

def extract_text_gemini(image, prompt):
    """Extract text using Gemini securely loaded API Key"""
    try:
        # Require API Key to be present in Streamlit Secrets or Environment Variables
        api_key = None
        
        # 1. Try checking standard OS environment variables first (often easier for cloud deployments)
        if "GEMINI_API_KEY" in os.environ:
             api_key = os.environ["GEMINI_API_KEY"]
             
        # 2. Fallback to Streamlit Secrets
        if not api_key:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            elif "general" in st.secrets and "GEMINI_API_KEY" in st.secrets["general"]:
                api_key = st.secrets["general"]["GEMINI_API_KEY"]
            
        if not api_key:
             return None, "サーバー側でAPIキーが設定されていません。管理者に連絡してください。"
             
        genai.configure(api_key=api_key)
        
        # Get all available models that support generateContent
        available_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority list for vision
        preferred_models = ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-1.5-pro', 'gemini-1.5-pro-latest', 'gemini-pro-vision']
        
        selected_model = None
        for pm in preferred_models:
            if pm in available_models:
                selected_model = pm
                break
                
        if not selected_model:
            # Fallback
            if available_models:
               for am in available_models:
                   if 'vision' in am or '1.5' in am:
                       selected_model = am
                       break
               if not selected_model:
                   selected_model = available_models[0] # ultimate fallback
            else:
               return None, "有効なGeminiモデルが見つかりません。APIキーの権限を確認してください。"
               
        model = genai.GenerativeModel(selected_model)
        response = model.generate_content([prompt, image])
        return response.text, None
    except Exception as e:
        return None, f"Gemini APIエラー: {str(e)}"

# --- Sidebar ---
st.sidebar.header("情報")
st.sidebar.success("システムは安全な環境下でGemini APIと連携しています。キーの入力は不要です。")

st.sidebar.header("出力設定")
output_mode = st.sidebar.selectbox("抽出形式を選択", list(PROMPT_TEMPLATES.keys()))
promp_to_use = PROMPT_TEMPLATES[output_mode]

# --- Main Area ---
# Changed to accept multiple files
uploaded_files = st.file_uploader("画像をアップロード (PNG, JPG, JPEG) 最大5MB", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if st.button(f"一括 文字起こし実行 ({len(uploaded_files)}枚)", type="primary"):
        st.session_state.ocr_result = {} # Use dict to store results per file
        
        # Process and extract all uploaded images
        with st.spinner(f"Gemini({output_mode})を用いて全{len(uploaded_files)}枚のテキストを抽出中..."):
            for file in uploaded_files:
                image, error_msg = process_image(file)
                if error_msg:
                    st.error(f"{file.name}: {error_msg}")
                    continue
                if image:
                    result_text, extract_error = extract_text_gemini(image, promp_to_use)
                    if extract_error:
                         st.error(f"{file.name}: {extract_error}")
                    elif result_text is not None:
                         st.session_state.ocr_result[file.name] = {
                             "text": result_text,
                             "image": image # Store image for display
                         }
            if st.session_state.ocr_result:
                st.success("すべての処理が完了しました。")

# --- Result Area ---
if 'ocr_result' in st.session_state and isinstance(st.session_state.ocr_result, dict) and st.session_state.ocr_result:
    
    # Render results using a 2-column layout for each file
    for filename, result_data in st.session_state.ocr_result.items():
        st.markdown(f"### {filename} の抽出結果")
        col1, col2 = st.columns([1, 1], gap="large") # 2-column layout
        
        with col1:
             st.image(result_data["image"], caption="元画像 (プレビュー)", use_container_width=True)
             
        with col2:
             # Text area to review and copy
             st.text_area(f"抽出テキスト ({output_mode})", value=result_data["text"], height=300, key=f"text_{filename}")
             
             # Download Button
             st.download_button(
                 label="テキストファイルとしてダウンロード",
                 data=result_data["text"],
                 file_name=f"ocr_result_{filename}.txt",
                 mime="text/plain",
                 key=f"dl_{filename}"
             )
        st.divider() # visually separate multiple files

    if st.button("すべての結果をクリア", type="secondary"):
        st.session_state.ocr_result = {}
        st.rerun()

