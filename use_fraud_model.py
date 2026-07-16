"""
🔍 Fraud Detection Model - Usage Examples
This script shows different ways to use the fraud_detection_model.pkl
"""

import pickle
import pandas as pd
import numpy as np

# ============================================================================
# STEP 1: LOAD THE MODEL
# ============================================================================

def load_model():
    """Load the fraud detection model from pickle file"""
    try:
        with open('1784175900539_fraud_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully!")
        return model
    except ModuleNotFoundError:
        print("❌ XGBoost not installed!")
        print("   Run: pip install xgboost")
        return None
    except FileNotFoundError:
        print("❌ Model file not found!")
        print("   Make sure the .pkl file is in the same directory")
        return None

# ============================================================================
# STEP 2: MAKE SINGLE PREDICTIONS
# ============================================================================

def predict_single_transaction(model, transaction_data):
    """
    Predict fraud for a single transaction
    
    Args:
        model: Loaded fraud detection model
        transaction_data: List or array of feature values
                         (must match training data format)
    
    Returns:
        prediction: 0 (Legitimate) or 1 (Fraudulent)
        probability: Confidence score (0.0 to 1.0)
    """
    # Reshape data for model (must be 2D)
    X = np.array(transaction_data).reshape(1, -1)
    
    # Get prediction and probability
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # probabilities[0] = probability of class 0 (legitimate)
    # probabilities[1] = probability of class 1 (fraud)
    
    return prediction, probabilities

# ============================================================================
# STEP 3: BATCH PREDICTIONS (MULTIPLE TRANSACTIONS)
# ============================================================================

def predict_batch_transactions(model, transactions_df):
    """
    Predict fraud for multiple transactions at once
    
    Args:
        model: Loaded fraud detection model
        transactions_df: DataFrame with transaction data
    
    Returns:
        DataFrame with original data + predictions
    """
    # Make predictions
    predictions = model.predict(transactions_df)
    probabilities = model.predict_proba(transactions_df)
    
    # Add results to dataframe
    transactions_df['prediction'] = predictions
    transactions_df['fraud_probability'] = probabilities[:, 1]
    transactions_df['is_fraud'] = predictions == 1
    
    return transactions_df

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🔍 FRAUD DETECTION MODEL USAGE")
    print("=" * 80)
    
    # Load the model
    model = load_model()
    
    if model is None:
        print("\n❌ Could not load model. Please install XGBoost first:")
        print("   pip install xgboost")
        exit(1)
    
    # ========================================================================
    # EXAMPLE 1: SINGLE TRANSACTION PREDICTION
    # ========================================================================
    print("\n" + "─" * 80)
    print("EXAMPLE 1: Single Transaction Prediction")
    print("─" * 80)
    
    # IMPORTANT: This is just a template. You need to know:
    # - How many features the model expects
    # - What each feature represents
    # - In what order the features should be
    
    # Example: Assuming model expects 10 features
    sample_transaction = [100.50, 1, 0, 1, 23, 45.32, 1, 0, 1, 0]
    #                     amount, cat1, cat2, ..., (10 features total)
    
    print(f"\nTransaction data: {sample_transaction}")
    
    try:
        prediction, probabilities = predict_single_transaction(model, sample_transaction)
        
        print(f"\nResult:")
        print(f"  Prediction: {'🚨 FRAUDULENT' if prediction == 1 else '✓ LEGITIMATE'}")
        print(f"  Legitimate probability: {probabilities[0]:.2%}")
        print(f"  Fraudulent probability: {probabilities[1]:.2%}")
        print(f"  Confidence: {max(probabilities):.2%}")
        
    except ValueError as e:
        print(f"\n⚠️ Error: {e}")
        print("This likely means the feature count doesn't match.")
        print("You need to provide the exact number of features the model expects.")
    
    # ========================================================================
    # EXAMPLE 2: BATCH PREDICTIONS FROM CSV
    # ========================================================================
    print("\n" + "─" * 80)
    print("EXAMPLE 2: Batch Predictions from CSV File")
    print("─" * 80)
    
    print("\nTo use this with your own data:")
    print("""
    # Load your transaction data
    transactions = pd.read_csv('transactions.csv')
    
    # Make predictions
    results = predict_batch_transactions(model, transactions)
    
    # View results
    print(results.head())
    
    # Save results
    results.to_csv('transactions_with_fraud_predictions.csv', index=False)
    
    # Filter fraudulent transactions
    fraud_transactions = results[results['is_fraud']]
    print(f"Found {len(fraud_transactions)} fraudulent transactions")
    """)
    
    # ========================================================================
    # EXAMPLE 3: GET MODEL INFORMATION
    # ========================================================================
    print("\n" + "─" * 80)
    print("EXAMPLE 3: Model Information")
    print("─" * 80)
    
    if hasattr(model, 'get_params'):
        print("\nModel parameters (first 10):")
        params = model.get_params()
        for i, (key, value) in enumerate(params.items()):
            if i < 10:
                print(f"  {key}: {value}")
        if len(params) > 10:
            print(f"  ... and {len(params) - 10} more")
    
    if hasattr(model, 'n_estimators'):
        print(f"\nNumber of trees: {model.n_estimators}")
    
    if hasattr(model, 'max_depth'):
        print(f"Max tree depth: {model.max_depth}")
    
    # ========================================================================
    # STEP BY STEP GUIDE
    # ========================================================================
    print("\n" + "=" * 80)
    print("HOW TO USE THIS MODEL")
    print("=" * 80)
    print("""
    1. INSTALL XGBOOST:
       $ pip install xgboost
    
    2. LOAD THE MODEL:
       model = load_model()
    
    3. PREPARE YOUR DATA:
       - Must be in numerical format
       - Must have exactly the same number of features as training data
       - Features must be in the same order as training data
    
    4. MAKE A PREDICTION:
       prediction, probs = predict_single_transaction(model, your_data)
    
    5. USE THE RESULT:
       if prediction == 1:
           print("⚠️ Likely Fraudulent!")
       else:
           print("✓ Likely Legitimate")
    
    IMPORTANT NOTES:
    ────────────────
    - You need to know the exact feature names/order from training
    - Data must be preprocessed the same way as training data
    - Probability score indicates confidence in the prediction
    - Consider using probability threshold (e.g., >0.8) for real decisions
    """)

    print("\n" + "=" * 80)
    print("✓ Script completed")
    print("=" * 80)
