import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def main():
    # Page config
    st.set_page_config(
        page_title="F1nalyze - F1 Predictions",
        page_icon="🏎️",
        layout="wide"
    )

    # Header
    st.title("🏎️ F1nalyze: Formula 1 Position Predictions")
    st.markdown("### Predicting F1 Driver Standings using Machine Learning")

    # Sidebar for navigation
    page = st.sidebar.radio(
        "Navigate",
        ["Project Overview", "Data Insights", "Model Performance", "About"]
    )

    if page == "Project Overview":
        show_project_overview()
    elif page == "Data Insights":
        show_data_insights()
    elif page == "Model Performance":
        show_model_performance()
    else:
        show_about()

def show_project_overview():
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Goal
        - Predict F1 driver finishing positions
        - Part of F1nalyze Kaggle Datathon
        - Ranked 23rd out of 50 teams
        """)
        
        st.markdown("""
        #### 📊 Dataset
        - Training data: 2.8M rows
        - Test data: 352K rows
        - Key features: grid position, nationality, team, points
        """)
    
    with col2:
        st.markdown("""
        #### 🛠️ Technologies
        - Python
        - Scikit-learn
        - Pandas
        - Streamlit
        """)
        
        st.markdown("""
        #### 🏆 Results
        - Best RMSE: 3.46918
        - Multiple models tested
        - Logistic Regression performed best
        """)

def show_data_insights():
    st.header("Data Insights")
    
    # Sample data for visualization
    st.subheader("📈 Data Processing Steps")
    
    steps = {
        "1. Data Cleaning": ["Handled missing values", "Removed columns with >100 null values", "Standardized data formats"],
        "2. Feature Engineering": ["Encoded categorical variables", "Selected key features", "Standardized numerical data"],
        "3. Feature Selection": ["Grid position", "Points", "Laps", "Round", "Nationality", "Team", "Status"]
    }
    
    for step, details in steps.items():
        with st.expander(step):
            for detail in details:
                st.write(f"• {detail}")
    
    # Create sample visualization
    st.subheader("📊 Sample Feature Distribution")
    
    # Dummy data for visualization
    positions = list(range(1, 21))
    frequencies = [np.random.randint(100, 1000) for _ in range(20)]
    
    fig = px.bar(
        x=positions,
        y=frequencies,
        labels={'x': 'Grid Position', 'y': 'Frequency'},
        title='Distribution of Grid Positions'
    )
    st.plotly_chart(fig)

def show_model_performance():
    st.header("Model Performance")
    
    # Model comparison
    models = ['Decision Tree', 'Random Forest', 'Logistic Regression']
    rmse_scores = [5.72788, 4.51769, 3.46918]
    
    fig = px.bar(
        x=models,
        y=rmse_scores,
        labels={'x': 'Model', 'y': 'RMSE Score'},
        title='Model Performance Comparison'
    )
    st.plotly_chart(fig)
    
    # Interactive model details
    st.subheader("🔍 Model Details")
    
    selected_model = st.selectbox(
        "Select a model to learn more",
        models
    )
    
    model_details = {
        'Decision Tree': {
            'description': 'Initial baseline model with simple decision rules',
            'pros': ['Simple to understand', 'No data scaling needed', 'Handles non-linear relationships'],
            'cons': ['Highest RMSE', 'Prone to overfitting', 'Less stable predictions']
        },
        'Random Forest': {
            'description': 'Ensemble model combining multiple decision trees',
            'pros': ['Better than single decision tree', 'Handles non-linear relationships', 'Reduced overfitting'],
            'cons': ['Slower training time', 'More complex', 'Higher memory usage']
        },
        'Logistic Regression': {
            'description': 'Linear model with probability-based predictions',
            'pros': ['Best RMSE score', 'Fast predictions', 'Stable results'],
            'cons': ['Assumes linear relationships', 'Requires scaled features', 'May underfit complex patterns']
        }
    }
    
    details = model_details[selected_model]
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Description**: {details['description']}")
        st.markdown("**Pros:**")
        for pro in details['pros']:
            st.markdown(f"• {pro}")
    
    with col2:
        st.markdown("**Cons:**")
        for con in details['cons']:
            st.markdown(f"• {con}")

def show_about():
    st.header("About F1nalyze")
    
    st.markdown("""
    #### 👥 Team Frostbiters
    - Kaggle competition participants
    - Focused on F1 predictions
    - 23rd place finish
    
    #### 🔄 Future Improvements
    - Feature engineering optimization
    - Deep learning models
    - Real-time predictions
    
    #### 🔗 Links
    - [GitHub Repository](https://github.com/yourusername/f1nalyze)
    - [Kaggle Competition](https://kaggle.com)
    """)

if __name__ == "__main__":
    main()