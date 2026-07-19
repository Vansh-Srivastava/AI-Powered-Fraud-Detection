import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import warnings

# Suppress XGBoost deprecation warning
warnings.filterwarnings('ignore', category=UserWarning)

st.set_page_config(
    page_title="Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CACHING DECORATOR - Load model only once
# ============================================================================
@st.cache_resource
def load_model():
    """Load model with caching to prevent reloading on every rerun"""
    try:
        with open("fraud_detection_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

try:
    model = load_model()
    if model is None:
        st.error("Model file not found")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.markdown('<h1 style="text-align: center; color: #2c3e50;">Transaction Risk Assessment</h1>', unsafe_allow_html=True)

MEANS = {
    'distance_from_home': 37.3137,
    'distance_from_last_transaction': 7.1226,
    'ratio_to_median_purchase_price': 2.958,
    'repeat_retailer': 0.8813,
    'used_chip': 0.3261,
    'used_pin_number': 0.07503,
    'online_order': 0.7317
}

tab1, tab2 = st.tabs(["Single Transaction", "Batch Processing"])

with tab1:
    with st.sidebar:
        st.header("Transaction Details")
        
        use_distance_home = st.checkbox("Distance from home address", value=True)
        if use_distance_home:
            distance_home = st.number_input("Distance (km)", 0.0, 500.0, 37.31, key="home")
        else:
            distance_home = MEANS['distance_from_home']
        
        use_distance_last = st.checkbox("Distance from last transaction", value=True)
        if use_distance_last:
            distance_last = st.number_input("Distance (km)", 0.0, 200.0, 7.12, key="last")
        else:
            distance_last = MEANS['distance_from_last_transaction']
        
        use_price_ratio = st.checkbox("Price vs typical spending", value=True)
        if use_price_ratio:
            price_ratio = st.number_input("Ratio", 0.1, 20.0, 2.96, key="price")
        else:
            price_ratio = MEANS['ratio_to_median_purchase_price']
        
        use_repeat = st.checkbox("Previously used retailer")
        repeat_val = 1 if use_repeat else MEANS['repeat_retailer']
        
        use_chip = st.checkbox("Chip card used")
        chip_val = 1 if use_chip else MEANS['used_chip']
        
        use_pin = st.checkbox("PIN entered")
        pin_val = 1 if use_pin else MEANS['used_pin_number']
        
        use_online = st.checkbox("Online transaction")
        online_val = 1 if use_online else MEANS['online_order']

    X = np.array([[distance_home, distance_last, price_ratio, repeat_val, chip_val, pin_val, online_val]])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Credit Card Fraud Detection"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#34495e"},
            'steps': [
                {'range': [0, 25], 'color': "#2ecc71"},
                {'range': [25, 50], 'color': "#f39c12"},
                {'range': [50, 75], 'color': "#e67e22"},
                {'range': [75, 100], 'color': "#e74c3c"}
            ]
        }
    ))
    fig.update_layout(
        height=380, 
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(size=14)
    )
    st.plotly_chart(fig, width='stretch')

    if probability > 0.7:
        recommendation = "Block transaction and contact cardholder immediately"
    elif probability > 0.4:
        recommendation = "Flag for manual review before processing"
    else:
        recommendation = "Process transaction normally"
    st.info(f"Recommendation: {recommendation}")

with tab2:
    st.header("Batch Fraud Detection")
    st.write("Upload an Excel file (.xlsx, .xls) or CSV file with transaction data to analyze multiple transactions at once.")
    st.write("**For Google Sheets:** Download your sheet as CSV or Excel format.")
    
    # Add info about file size limitations
    st.info("⚠️ **Performance Tip:** For files > 10,000 rows, consider processing in smaller batches for faster results.")
    
    st.subheader("Expected Excel Format")
    expected_format = pd.DataFrame({
        'distance_from_home': [37.31, 150.5, 5.2],
        'distance_from_last_transaction': [7.12, 25.3, 1.8],
        'ratio_to_median_purchase_price': [2.96, 8.5, 0.8],
        'repeat_retailer': [1, 0, 1],
        'used_chip': [1, 0, 1],
        'used_pin_number': [0, 1, 0],
        'online_order': [1, 0, 0]
    })
    st.dataframe(expected_format, width='stretch')
    
    st.write("**Column Descriptions:**")
    st.write("- `distance_from_home`: Distance in km from cardholder's home address")
    st.write("- `distance_from_last_transaction`: Distance in km from previous transaction location")
    st.write("- `ratio_to_median_purchase_price`: Ratio of current purchase to median purchase amount")
    st.write("- `repeat_retailer`: 1 if previously used retailer, 0 if new")
    st.write("- `used_chip`: 1 if chip card used, 0 if not")
    st.write("- `used_pin_number`: 1 if PIN entered, 0 if not")
    st.write("- `online_order`: 1 if online transaction, 0 if in-person")
    
    uploaded_file = st.file_uploader("Choose a file", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        try:
            # Show loading progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Load file
            status_text.text("📂 Loading file...")
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            progress_bar.progress(20)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head(10), width='stretch')
            st.caption(f"Showing first 10 of {len(df)} rows")
            
            # Validate columns
            required_columns = [
                'distance_from_home', 'distance_from_last_transaction', 
                'ratio_to_median_purchase_price', 'repeat_retailer', 
                'used_chip', 'used_pin_number', 'online_order'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.stop()
            
            progress_bar.progress(40)
            
            # Fill missing values
            for col in required_columns:
                if col in MEANS:
                    df[col] = df[col].fillna(MEANS[col])
                else:
                    df[col] = df[col].fillna(0)
            
            # ================================================================
            # PROCESS IN CHUNKS - Key fix for memory issues
            # ================================================================
            status_text.text("🔍 Analyzing transactions (processing in chunks)...")
            
            chunk_size = 50000  # Process 50k rows at a time
            predictions_list = []
            probabilities_list = []
            
            num_chunks = (len(df) + chunk_size - 1) // chunk_size
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(df))
                
                chunk_data = df.iloc[start_idx:end_idx][required_columns].values
                chunk_predictions = model.predict(chunk_data)
                chunk_probabilities = model.predict_proba(chunk_data)[:, 1]
                
                predictions_list.append(chunk_predictions)
                probabilities_list.append(chunk_probabilities)
                
                # Update progress
                progress = 40 + int((i + 1) / num_chunks * 40)
                progress_bar.progress(progress)
            
            progress_bar.progress(80)
            
            # Combine results
            all_predictions = np.concatenate(predictions_list)
            all_probabilities = np.concatenate(probabilities_list)
            
            # Add predictions to dataframe
            results_df = df.copy()
            results_df['fraud_probability'] = all_probabilities
            results_df['fraud_prediction'] = all_predictions
            results_df['fraud_status'] = results_df['fraud_prediction'].map({0: 'Not Fraud', 1: 'Fraud'})
            
            def get_recommendation(prob):
                if prob > 0.7:
                    return "Block transaction and contact cardholder immediately"
                elif prob > 0.4:
                    return "Flag for manual review before processing"
                else:
                    return "Process transaction normally"
            
            results_df['recommendation'] = results_df['fraud_probability'].apply(get_recommendation)
            
            progress_bar.progress(90)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            st.subheader("Fraud Detection Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", len(results_df))
            with col2:
                fraud_count = len(results_df[results_df['fraud_prediction'] == 1])
                st.metric("Fraudulent Transactions", fraud_count)
            with col3:
                avg_prob = results_df['fraud_probability'].mean()
                st.metric("Average Fraud Probability", f"{avg_prob:.1%}")
            with col4:
                high_risk = len(results_df[results_df['fraud_probability'] > 0.7])
                st.metric("High Risk Transactions", high_risk)
            
            # Display results table with pagination for large datasets
            if len(results_df) > 1000:
                st.warning(f"📊 Large dataset detected ({len(results_df):,} rows). Showing summary and top results below.")
                display_df = results_df.nlargest(1000, 'fraud_probability').copy()
                st.caption("Showing top 1,000 transactions by fraud probability")
            else:
                display_df = results_df.copy()
            
            display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df, width='stretch')
            
            # Download results
            progress_bar.progress(95)
            
            def convert_df_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Fraud_Detection_Results', index=False)
                return output.getvalue()
            
            excel_data = convert_df_to_excel(results_df)
            
            st.download_button(
                label="📥 Download Full Results as Excel",
                data=excel_data,
                file_name="fraud_detection_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Visualization
            st.subheader("Results Visualization")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=results_df['fraud_probability'],
                nbinsx=20,
                name="Fraud Probability Distribution",
                marker_color="#3498db"
            ))
            fig_hist.update_layout(
                title="Distribution of Fraud Probabilities",
                xaxis_title="Fraud Probability",
                yaxis_title="Number of Transactions",
                height=400
            )
            st.plotly_chart(fig_hist, width='stretch')
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Fraud Probability", f"{results_df['fraud_probability'].min():.2%}")
            with col2:
                st.metric("Max Fraud Probability", f"{results_df['fraud_probability'].max():.2%}")
            with col3:
                st.metric("Median Fraud Probability", f"{results_df['fraud_probability'].median():.2%}")
            
            progress_bar.progress(100)
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.write("Please make sure your file has the correct format and column names.")
            import traceback
            st.write(traceback.format_exc())
