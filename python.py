# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
from google.genai import types # Import thêm để quản lý lịch sử chat

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Khởi tạo Gemini Client (Sử dụng chung cho cả phân tích và chat) ---
CHAT_MODEL = "gemini-2.5-flash"

try:
    # Lấy API Key từ st.secrets
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.warning("Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng các tính năng AI.")
        client = None
    else:
        client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Lỗi khởi tạo Gemini Client: {e}")
    client = None

# --- Khởi tạo Chat Session và Lịch sử tin nhắn ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_session" not in st.session_state and client:
    try:
        # Khởi tạo Chat Session để duy trì lịch sử
        st.session_state.chat_session = client.chats.create(model=CHAT_MODEL)
    except Exception as e:
        st.error(f"Không thể khởi tạo Chat Session: {e}")
        st.session_state.chat_session = None

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini (Chức năng Phân tích báo cáo) ---
def get_ai_analysis(data_for_ai):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    if not client:
        return "Lỗi: Gemini Client chưa được khởi tạo. Vui lòng kiểm tra API Key."
        
    try:
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Hàm xử lý gửi tin nhắn Chat (Chức năng Chat) ---
def handle_chat_submit(prompt):
    if not st.session_state.chat_session:
        st.warning("Chat Session chưa được khởi tạo. Vui lòng kiểm tra API Key.")
        return

    # 1. Thêm tin nhắn người dùng vào lịch sử
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Hiển thị tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Gửi tin nhắn đến Gemini và hiển thị phản hồi
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Gửi tin nhắn và nhận phản hồi
            response = st.session_state.chat_session.send_message(prompt)
            full_response = response.text
            
            # Cập nhật hiển thị cuối cùng
            message_placeholder.markdown(full_response)
            
            # Thêm tin nhắn của assistant vào lịch sử
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_message = f"Lỗi Gemini: {e}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})


# ====================================================================
# KHUNG CHAT ĐỘC LẬP VỚI GEMINI (ĐẶT Ở SIDEBAR)
# ====================================================================
with st.sidebar:
    st.title("🤖 Chat với Gemini")
    st.markdown("Hỏi Gemini bất cứ điều gì bạn muốn!")

    # Hiển thị lịch sử chat
    chat_container = st.container(height=400)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Input field cho người dùng
    if client:
        # Sử dụng on_submit để xử lý khi người dùng nhấn Enter/Gửi
        user_prompt = st.chat_input("Hỏi Gemini điều gì đó...", on_submit=lambda: handle_chat_submit(st.session_state.chat_input))
        
        # Nút reset lịch sử chat
        if st.button("Xóa Lịch sử Chat", help="Bắt đầu cuộc trò chuyện mới"):
            st.session_state["messages"] = []
            st.session_state.chat_session = client.chats.create(model=CHAT_MODEL)
            st.rerun() # Tải lại trang để cập nhật giao diện
    else:
        st.warning("Không thể chat. Vui lòng kiểm tra API Key.")

# ====================================================================
# BẮT ĐẦU PHẦN CODE CHÍNH CỦA ỨNG DỤNG PHÂN TÍCH TÀI CHÍNH
# (GIỮ NGUYÊN HOẶC TỐI ƯU HÓA)
# ====================================================================

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán (Xử lý lỗi chia cho 0)
                thanh_toan_hien_hanh_N = tsnh_n / (no_ngan_han_N if no_ngan_han_N != 0 else 1e-9)
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / (no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else 1e-9)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                 thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                 thanh_toan_hien_hanh_N_1 = "N/A"
            
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if "TÀI SẢN NGẮN HẠN" in df_processed['Chỉ tiêu'].str.upper().str.cat(sep=' ') else "N/A",
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                if client:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        # Truyền thẳng dữ liệu, không cần API Key nữa vì đã dùng 'client' toàn cục
                        ai_result = get_ai_analysis(data_for_ai)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
