import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Jamii Health Cover | Estimator", 
    page_icon=":material/health_and_safety:", 
    layout="wide" 
)

# --- CLEAN CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; }
    .header-box {
        background-color: #005F56;
        color: white;
        padding: 30px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    .header-box h1 { color: white !important; margin: 0; padding-bottom: 8px; font-size: 2.6rem; }
    .header-box h3 { color: #E0F2F1 !important; margin: 0; font-weight: 400; font-size: 1.4rem; }
    .header-box p { color: #B2DFDB !important; margin-top: 20px; margin-bottom: 0; font-size: 1.1rem; line-height: 1.6;}
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- RESET CALLBACK ---
def reset_form():
    st.session_state["age_key"] = 30
    st.session_state["bmi_key"] = 25.0
    st.session_state["children_key"] = 0
    st.session_state["sex_key"] = "male"
    st.session_state["smoker_key"] = "no"
    st.session_state["region_key"] = "northeast"
    st.session_state["prediction_made"] = False

# Initialize state
if "age_key" not in st.session_state:
    reset_form()

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_model_and_calculate_rmse():
    # Local paths for testing (Change to '../' for GitHub)
    model_path = 'G02_model.joblib'
    data_path = 'clean_insurance.csv'
    
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['charges', 'log_charges', 'bmi_band'])
    y = df['log_charges']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, rmse

model, app_rmse = load_model_and_calculate_rmse()

# --- HEADER ---
st.markdown("""
<div class="header-box">
    <h1>Jamii Health Cover</h1>
    <h3>Medical Charges Estimator</h3>
    <p><b>Group G2:</b> Daisy Akinyi, Prince Emmanuel, William Otieno, Ronald Kipkirui, Javet Manyu<br>
    <b>Dataset:</b> Medical Cost Personal Datasets</p>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    with st.container(border=True):
        st.subheader("Patient Information")
        st.write("Enter the member details below.")
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, key="age_key")
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, key="bmi_key")
            children = st.number_input("Number of Children", min_value=0, max_value=10, key="children_key")
        with c2:
            sex = st.selectbox("Sex", ["male", "female"], key="sex_key")
            smoker = st.selectbox("Smoker", ["no", "yes"], key="smoker_key")
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], key="region_key")
            
        st.write("") 
        
        btn1, btn2 = st.columns(2)
        with btn1:
            # The Predict button, as required by the PDF
            if st.button("Predict", icon=":material/calculate:", use_container_width=True, type="primary"):
                st.session_state["prediction_made"] = True
        with btn2:
            st.button("Reset Form", icon=":material/refresh:", use_container_width=True, on_click=reset_form)

with right_col:
    with st.container(border=True):
        st.subheader("Estimated Premium")
        
        if st.session_state.get("prediction_made", False):
            # Encode inputs
            sex_encoded = 1 if sex == "male" else 0
            smoker_encoded = 1 if smoker == "yes" else 0
            bmi_smoker = bmi * smoker_encoded 
            
            input_data = pd.DataFrame({
                'age': [age], 'sex': [sex_encoded], 'bmi': [bmi], 
                'children': [children], 'smoker': [smoker_encoded],
                'region_northeast': [1 if region=="northeast" else 0],
                'region_northwest': [1 if region=="northwest" else 0],
                'region_southeast': [1 if region=="southeast" else 0],
                'region_southwest': [1 if region=="southwest" else 0],
                'bmi_smoker': [bmi_smoker]
            })
            
            log_pred = model.predict(input_data)[0]
            actual_pred = np.exp(log_pred)
            dollar_margin = actual_pred * app_rmse 
            
            st.divider()
            st.write("") 
            st.metric(label="Predicted Yearly Charges", value=f"${actual_pred:,.2f}")
            
            st.write("") 
            st.info(f"**Estimated Range:** ± ${dollar_margin:,.2f}")
            st.caption(f"Based on Test RMSE: {app_rmse:.4f}")
            
            st.divider()
            st.write("**Patient Summary:**")
            st.write(f"- {age} year old {sex}")
            st.write(f"- BMI: {bmi} | Children: {children}")
            st.write(f"- Smoker: {smoker.capitalize()} | Region: {region.capitalize()}")
            
        else:
            # Clean empty state
            st.divider()
            st.write("")
            st.write("")
            st.info("Adjust the patient details on the left and click **Predict** to view the estimated premium.")
            st.write("")
            st.write("")
