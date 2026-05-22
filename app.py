import streamlit as st
import pandas as pd
import joblib

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Laptop Price Estimator",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# LOAD DATA & MODEL
@st.cache_data
def load_data():
    data = pd.read_csv('laptops.csv')
    data = data[data['Status'] == 'New']
    data['GPU'] = data['GPU'].fillna('None/Integrated')
    data['Storage type'] = data['Storage type'].fillna('Unknown')
    return data

@st.cache_resource
def load_model():
    return joblib.load('best_laptop_price_model.pkl')

df = load_data()
model = load_model()

# HEADER & DESKRIPSI
st.markdown("<h1 style='text-align: center;'>💻 Laptop Price Estimator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Pilih spesifikasi di bawah, dan sistem kami akan membantu Anda menemukan estimasi harga laptop</p>", unsafe_allow_html=True)
st.write("")
st.write("")

# AREA INPUT SPESIFIKASI 

# Baris 1
col1, col2 = st.columns(2)
with col1:
    brand_list = sorted(df['Brand'].dropna().unique().tolist())
    brand = st.selectbox("📌 Merek Laptop", brand_list)
    current_df = df[df['Brand'] == brand]

with col2:
    model_list = ["Unknown/Other"] + sorted(current_df['Model'].dropna().unique().tolist())
    model_name = st.selectbox("🏷️ Seri/Model", model_list)
    if model_name != "Unknown/Other" and model_name in current_df['Model'].values:
        current_df = current_df[current_df['Model'] == model_name]

# Baris 2
col3, col4 = st.columns(2)
with col3:
    cpu_list = sorted(current_df['CPU'].dropna().unique().tolist())
    if not cpu_list: cpu_list = df['CPU'].unique().tolist()
    cpu = st.selectbox("🚀 Prosesor (CPU)", cpu_list)
    if cpu in current_df['CPU'].values:
        current_df = current_df[current_df['CPU'] == cpu]

with col4:
    gpu_list = sorted(current_df['GPU'].dropna().unique().tolist())
    if not gpu_list: gpu_list = df['GPU'].unique().tolist()
    gpu = st.selectbox("🎮 Kartu Grafis (GPU)", gpu_list)
    if gpu in current_df['GPU'].values:
        current_df = current_df[current_df['GPU'] == gpu]

# Baris 3
col5, col6 = st.columns(2)
with col5:
    ram_list = sorted(current_df['RAM'].dropna().unique().tolist())
    if not ram_list: ram_list = [8]
    ram = st.selectbox("⚡ RAM (GB)", ram_list)

with col6:
    storage_list = sorted(current_df['Storage'].dropna().unique().tolist())
    if not storage_list: storage_list = [512]
    storage = st.selectbox("💾 Kapasitas Storage (GB)", storage_list)

# Baris 4
col7, col8 = st.columns(2)
with col7:
    storage_type_list = sorted(current_df['Storage type'].dropna().unique().tolist())
    if not storage_type_list: storage_type_list = df['Storage type'].unique().tolist()
    storage_type = st.selectbox("💽 Tipe Penyimpanan", storage_type_list)

with col8:
    screen_list = sorted(current_df['Screen'].dropna().unique().tolist())
    if not screen_list: screen_list = [15.6]
    screen = st.selectbox("🖥️ Ukuran Layar (Inci)", screen_list)

st.write("")
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    touch_list = sorted(current_df['Touch'].dropna().unique().tolist())
    if not touch_list: touch_list = df['Touch'].unique().tolist()
    touch = st.selectbox("👆 Layar Sentuh (Touchscreen)?", touch_list)
    
    st.write("")
    predict_btn = st.button("Hitung Estimasi Harga", use_container_width=True, type="primary")

st.divider()

# HASIL PREDIKSI
if predict_btn:
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Model': [model_name],
        'CPU': [cpu],
        'RAM': [ram],
        'Storage': [storage],
        'Storage type': [storage_type],
        'GPU': [gpu],
        'Screen': [screen],
        'Touch': [touch]
    })
    
    try:
        with st.spinner("Memproses estimasi harga..."):
            prediction = model.predict(input_data)[0]
        
        _, result_col, _ = st.columns([1, 3, 1])
        with result_col:
            st.markdown(f"""
                <div style="text-align: center; padding: 30px 0; border: 1px solid rgba(128,128,128,0.2); border-radius: 16px; background-color: rgba(128,128,128,0.05); box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <p style="font-size: 1.2rem; font-weight: 500; margin-bottom: 0; opacity: 0.8;">Estimasi Harga</p>
                    <h1 style="font-size: 4rem; margin-top: 5px; margin-bottom: 0; color: #ff4b4b;">${prediction:,.2f}</h1>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan pada sistem: {e}")
