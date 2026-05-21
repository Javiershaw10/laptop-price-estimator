import streamlit as st
import pandas as pd
import joblib

# 1. KONFIGURASI HALAMAN 
st.set_page_config(
    page_title="Laptop Price Estimator", 
    page_icon="💻", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. LOAD DATA & MODEL
@st.cache_data
def load_data():
    data = pd.read_csv('laptops.csv')
    data['GPU'] = data['GPU'].fillna('None/Integrated')
    data['Storage type'] = data['Storage type'].fillna('Unknown')
    return data

@st.cache_resource
def load_model():
    return joblib.load('best_laptop_price_model.pkl')

df = load_data()
model = load_model()

# 3. SIDEBAR (AREA INPUT SPESIFIKASI)
st.sidebar.title("⚙️ Spesifikasi Laptop")
st.sidebar.markdown("Pilih spesifikasi untuk melihat estimasi harga pasar.")

st.sidebar.subheader("📌 Merek & Tipe")
status_list = sorted(df['Status'].dropna().unique().tolist())
status = st.sidebar.selectbox("Kondisi", status_list)

brand_list = sorted(df['Brand'].dropna().unique().tolist())
brand = st.sidebar.selectbox("Merek", brand_list)

current_df = df[df['Brand'] == brand]

model_list = ["Unknown/Other"] + sorted(current_df['Model'].dropna().unique().tolist())
model_name = st.sidebar.selectbox("Seri/Model", model_list)

if model_name != "Unknown/Other" and model_name in current_df['Model'].values:
    current_df = current_df[current_df['Model'] == model_name]

st.sidebar.divider() 

st.sidebar.subheader("🚀 Spesifikasi")
cpu_list = sorted(current_df['CPU'].dropna().unique().tolist())
if not cpu_list: cpu_list = df['CPU'].unique().tolist()
cpu = st.sidebar.selectbox("Prosesor (CPU)", cpu_list)

if cpu in current_df['CPU'].values:
    current_df = current_df[current_df['CPU'] == cpu]

gpu_list = sorted(current_df['GPU'].dropna().unique().tolist())
if not gpu_list: gpu_list = df['GPU'].unique().tolist()
gpu = st.sidebar.selectbox("Kartu Grafis (GPU)", gpu_list)

if gpu in current_df['GPU'].values:
    current_df = current_df[current_df['GPU'] == gpu]

ram_list = sorted(current_df['RAM'].dropna().unique().tolist())
if not ram_list: ram_list = [8]
ram = st.sidebar.selectbox("RAM (GB)", ram_list)

st.sidebar.divider()

st.sidebar.subheader("💾 Penyimpanan & Layar")
storage_list = sorted(current_df['Storage'].dropna().unique().tolist())
if not storage_list: storage_list = [512]
storage = st.sidebar.selectbox("Kapasitas Storage (GB)", storage_list)

storage_type_list = sorted(current_df['Storage type'].dropna().unique().tolist())
if not storage_type_list: storage_type_list = df['Storage type'].unique().tolist()
storage_type = st.sidebar.radio("Tipe Penyimpanan", storage_type_list, horizontal=True)

screen_list = sorted(current_df['Screen'].dropna().unique().tolist())
if not screen_list: screen_list = [15.6]
screen = st.sidebar.selectbox("Ukuran Layar (Inci)", screen_list)

touch_list = sorted(current_df['Touch'].dropna().unique().tolist())
if not touch_list: touch_list = df['Touch'].unique().tolist()
touch = st.sidebar.radio("Layar Sentuh?", touch_list, horizontal=True)

# 4. AREA UTAMA
st.title("💻 Laptop Price Estimator")
st.markdown("""
Pilih spesifikasi di menu samping, dan sistem kami akan membantu Anda menemukan estimasi harga laptop.
""")

st.markdown("### 📋 Profil Laptop Saat Ini")
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Merek & Seri", value=f"{brand}", delta=model_name, delta_color="off")
    col2.metric(label="Prosesor", value=f"{cpu}")
    col3.metric(label="Kartu Grafis", value=f"{gpu}")
    col4.metric(label="Memori", value=f"{ram}GB", delta=f"{storage}GB {storage_type}", delta_color="off")

st.write("")

_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    predict_btn = st.button("Hitung Estimasi Harga Pasar", use_container_width=True, type="primary")

st.divider()

if predict_btn:
    input_data = pd.DataFrame({
        'Status': [status],
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
        
        st.markdown(f"""
            <div style="text-align: center; padding: 25px 0; margin-top: 15px; border: 1px solid rgba(128,128,128,0.2); border-radius: 12px; background-color: rgba(128,128,128,0.05);">
                <p style="font-size: 1.1rem; font-weight: 400; margin-bottom: 0; opacity: 0.8;">Estimasi Harga Pasar</p>
                <h1 style="font-size: 3.2rem; margin-top: 5px; margin-bottom: 0;">${prediction:,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan pada sistem: {e}")
