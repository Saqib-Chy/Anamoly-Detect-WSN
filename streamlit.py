import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import os

# Load and preprocess the data
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("WSN-DS.csv")  # Replace with your dataset path
    df = df.dropna()

    # Convert categorical target to numerical if needed
    df['Attack type'] = pd.Categorical(df['Attack type']).codes

    # Split features and target
    X = df.drop(['Attack type'], axis=1)
    y = df['Attack type']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X, y, scaler

# Load model comparison results
@st.cache_data
def load_model_comparison():
    comparison_file = "model_comparison.json"
    with open(comparison_file, "r") as f:
        model_results = json.load(f)
    return model_results

# Load a specific model
@st.cache_resource
def load_model(model_name):
    return joblib.load(f"models/{model_name}_Best.pkl")

# Prediction function
def predict_user_input(model, scaler, user_input_df):
    # Scale user input
    user_input_scaled = scaler.transform(user_input_df)
    # Predict
    prediction = model.predict(user_input_scaled)[0]
    return prediction

# Sidebar
st.sidebar.title("Navigation")
selected_option = st.sidebar.selectbox(
    "Choose a section:",
    ["Introduction", "Data Overview", "Model Comparison", "Visualizations", "Confusion Matrices", "Prediction"]
)

# Section 1: Introduction
if selected_option == "Introduction":
    st.title("DATA 245 Final Project")
    st.write("""
        This application demonstrates model training, evaluation, and visualization 
        for the Wireless Sensor Network Dataset (WSN-DS).
             
        Group 4:
        
        Darpankumar Jiyani - 017536623\n   
        Kush Bindal - 017441359\n
        Manjot Singh - 017557462\n
        Saqib Chowdhury - 017514978\n
        Sai Prasad Thalluri - 017512781\n
    """)

# Section 2: Data Overview
elif selected_option == "Data Overview":
    st.title("Data Overview")

    # Load data and preprocess
    df, X, y, scaler = load_and_process_data()

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Statistics")
    st.write(df.describe())

    st.write("### Class Distribution")
    class_distribution = df["Attack type"].value_counts()
    st.bar_chart(class_distribution)

# Section 3: Model Comparison
elif selected_option == "Model Comparison":
    st.title("Model Comparison")

    # Load comparison results
    model_results = load_model_comparison()
    comparison_df = pd.DataFrame.from_dict(
        model_results, orient='index', columns=['Accuracy']
    ).reset_index().rename(columns={"index": "Model"})

    st.write("### Model Accuracy Table")
    st.dataframe(comparison_df)

    st.write("### Model Accuracy Bar Chart")
    st.bar_chart(comparison_df.set_index("Model"))

# Section 4: Visualizations
elif selected_option == "Visualizations":
    st.title("Visualizations")

    # Display saved visualizations
    visualization_files = {
        "Attack Type Distribution": "attack_distribution.png",
        "Correlation Matrix": "correlation_matrix.png",
        "Node Distribution": "node distribution.png",
        "Average Message Patterns": "avg message patterns.png",
        "Energy Consumption": "energy consumption.png",
        "Time-Based Energy Analysis": "time based.png",
        "Data Transmission Patterns": "data transmission.png"
    }

    for title, filepath in visualization_files.items():
        if os.path.exists(filepath):
            st.write(f"### {title}")
            st.image(filepath)
        else:
            st.warning(f"Visualization {title} not found!")

# Section 5: Confusion Matrices
elif selected_option == "Confusion Matrices":
    st.title("Confusion Matrices")

    # Load data
    df, X, y, scaler = load_and_process_data()

    # Available models
    models = {
        "Logistic Regression": "Logistic_Regression",
        "Random Forest": "Random_Forest",
        "Gradient Boosting": "Gradient_Boosting",
        "Support Vector Classifier": "Support_Vector",
        "XGBoost": "XGBoost"
    }

    # Select a model
    model_choice = st.selectbox("Select a model:", list(models.keys()))
    model_name = models[model_choice]

    try:
        model = load_model(model_name)
        st.write(f"### Confusion Matrix for {model_choice}")

        # Split data into training and validation
        from sklearn.model_selection import train_test_split
        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Predict on validation data
        y_val_pred = model.predict(scaler.transform(X_val))

        # Generate confusion matrix
        cm = confusion_matrix(y_val, y_val_pred)

        # Create figure and plot
        fig, ax = plt.subplots()
        sns.heatmap(cm, cmap="Blues", annot=True, fmt="d", cbar=True, ax=ax, linewidths=0.5)
        ax.set_title(f"Confusion Matrix: {model_choice}")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")

        # Display plot
        st.pyplot(fig)

        st.write("### Classification Report")
        report = classification_report(y_val, y_val_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    except Exception as e:
        st.error(f"Error displaying confusion matrix for {model_choice}: {e}")

# Section 6: Prediction
elif selected_option == "Prediction":
    st.title("Make a Prediction")
    
    # Load data and model
    df, X, y, scaler = load_and_process_data()
    model = load_model("XGBoost")  # Replace "XGBoost" with your default model name

    # Create attack type mapping from the original dataset
    original_df = pd.read_csv("WSN-DS.csv")  # Load original dataset
    attack_type_mapping = {code: label for code, label in enumerate(pd.Categorical(original_df['Attack type']).categories)}

    # Input fields for user
    st.write("### Enter values for prediction:")
    feature_columns = [
        ' id', ' Time', ' Is_CH', ' who CH', ' Dist_To_CH', ' ADV_S', ' ADV_R', ' JOIN_S', ' JOIN_R',
        ' SCH_S', ' SCH_R', 'Rank', ' DATA_S', ' DATA_R', ' Data_Sent_To_BS',
        ' dist_CH_To_BS', ' send_code ', 'Expaned Energy'
    ]

    # Min-max values from df.describe()
    feature_ranges = {
        ' Time': (50, 3600),
        ' Is_CH': (0, 1),
        ' who CH': (101000, 3402100),
        ' Dist_To_CH': (0.0, 214.275),
        ' ADV_S': (0, 97),
        ' ADV_R': (0, 117),
        ' JOIN_S': (0, 1),
        ' JOIN_R': (0, 124),
        ' SCH_S': (0, 99),
        ' SCH_R': (0, 1),
        'Rank': (0, 99),
        ' DATA_S': (0, 241),
        ' DATA_R': (0, 1496),
        ' Data_Sent_To_BS': (0, 241),
        ' dist_CH_To_BS': (0.0, 201.935),
        ' send_code ': (0, 15),
        'Expaned Energy': (0.0, 45.094)
    }

    user_input = {}

    # Set default value for 'id' internally
    user_input[' id'] = 0
    
    for feature in feature_columns:
        if feature == ' id':
            # Skip the 'id' input as it's hidden
            continue
        elif feature in [' Is_CH', ' SCH_R', ' JOIN_S']:
            # Handle binary fields with dropdown
            user_input[feature] = st.selectbox(f"{feature} (0 or 1)", options=[0, 1])
        else:
            # Numeric input for other fields
            min_val, max_val = feature_ranges[feature]
            user_input[feature] = st.number_input(
                f"{feature} ({min_val} - {max_val})",
                min_value=float(min_val),
                max_value=float(max_val),
                step=1.0 if isinstance(min_val, int) else 0.1,
                format="%.5f"
            )

    # Convert user input to a DataFrame
    user_input_df = pd.DataFrame([user_input])

    # Reorder columns to match the training feature order
    user_input_df = user_input_df[feature_columns]

    # Make a prediction
    if st.button("Predict"):
        try:
            # Ensure the input DataFrame matches the scaler and model expectations
            numeric_prediction = predict_user_input(model, scaler, user_input_df)
            # Map numeric prediction to original label
            predicted_label = attack_type_mapping[numeric_prediction]
            st.success(f"Predicted Attack Type: {predicted_label}")
        except ValueError as e:
            st.error(f"Prediction failed: {e}")
