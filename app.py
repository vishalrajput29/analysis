import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# App title
st.title("Comprehensive Water Quality Index (WQI) Analysis App")

# Upload dataset (accepting both CSV and JSON formats)
uploaded_file = st.file_uploader("Upload your dataset (CSV or JSON format)", type=["csv", "json"])

if uploaded_file:
    # Load the dataset based on file extension
    if uploaded_file.type == "application/json":
        data = pd.read_json(uploaded_file)
    elif uploaded_file.type == "text/csv":
        data = pd.read_csv(uploaded_file)
        
    st.write("### Dataset Preview")
    st.dataframe(data)

    # Check if WQI column exists
    if "WQI" in data.columns:
        st.write("### Water Quality Index (WQI) Awareness")

        # Alert for impure water (WQI > 75)
        impure_count = data[data["WQI"] > 75].shape[0]
        if impure_count > 0:
            st.error(f"\u26A0\ufe0f Alert: {impure_count} samples have WQI > 75. The water is impure.")
        else:
            st.success("\u2714\ufe0f All samples have WQI <= 75. The water is pure.")

        # Display top 5 and lowest 5 WQI values
        st.write("#### Top 5 and Lowest 5 WQI Values")
        st.write("**Top 5 WQI Values**")
        st.dataframe(data.nlargest(5, "WQI"))
        st.write("**Lowest 5 WQI Values**")
        st.dataframe(data.nsmallest(5, "WQI"))
    

        # Interactive WQI Distribution
        st.write("#### Interactive WQI Distribution")
        fig = px.histogram(data, x="WQI", nbins=20, title="Interactive WQI Distribution")
        fig.update_layout(xaxis_title="WQI", yaxis_title="Frequency")
        st.plotly_chart(fig)

        # Categorize WQI
        st.write("#### WQI Categorization")
        def categorize_wqi(wqi):
            if wqi <= 50:
                return "Excellent"
            elif wqi <= 75:
                return "Good"
            elif wqi <= 100:
                return "Poor"
            else:
                return "Very Poor"

        data["WQI_Category"] = data["WQI"].apply(categorize_wqi)
        st.bar_chart(data["WQI_Category"].value_counts())

        # Interactive WQI Categories Bar Chart
        st.write("#### Interactive WQI Categories Bar Chart")
        fig = px.bar(data_frame=data, x="WQI_Category", title="WQI Categories Distribution", color="WQI_Category")
        fig.update_layout(xaxis_title="WQI Category", yaxis_title="Count")
        st.plotly_chart(fig)

        # Time-Series Analysis (if timestamp available)
        if "timestamp" in data.columns:
            st.write("#### Interactive Time-Series Analysis of WQI")
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            time_series = data.groupby(data["timestamp"].dt.date)["WQI"].mean()
            
            # Create an interactive line plot
            fig = px.line(time_series, title="Time-Series Analysis of WQI")
            fig.update_layout(xaxis_title="Date", yaxis_title="Average WQI")
            st.plotly_chart(fig)

    else:
        st.warning("The dataset does not contain a 'WQI' column. Please upload a dataset with WQI.")

    # Sidebar for analysis options
    st.sidebar.title("Analysis Options")
    features = st.sidebar.multiselect("Select features for analysis:", data.columns)

    # Feature-Wise Analysis
    if features:
        st.write("### Feature-Wise Analysis")
        for feature in features:
            st.write(f"#### {feature}")

            # Top 5 and Lowest 5 Values
            st.write("**Top 5 Values**")
            st.dataframe(data.nlargest(5, feature))
            st.write("**Lowest 5 Values**")
            st.dataframe(data.nsmallest(5, feature))

            # Distribution
            st.write("**Distribution**")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data[feature], kde=True, bins=20, ax=ax, color="green")
            ax.set_title(f"Distribution of {feature}", fontsize=14)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            st.pyplot(fig)

        # Summary Statistics
        st.write("### Summary Statistics")
        st.write(data[features].describe())

        # Pairwise Comparison with Plotly
        st.write("### Interactive Pairwise Comparison")
        pairwise_features = st.multiselect("Select features for pairwise comparison:", data.columns)

        if len(pairwise_features) > 1:
            fig = px.scatter_matrix(data, dimensions=pairwise_features, title="Pairwise Feature Comparison")
            fig.update_layout(width=800, height=800)
            st.plotly_chart(fig)

        # Custom Visualization
        st.write("### Custom Visualization")
        x_axis = st.selectbox("Select X-axis:", options=features, key="x_axis")
        y_axis = st.selectbox("Select Y-axis:", options=features, key="y_axis")

        if x_axis and y_axis:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
            ax.set_title(f"Scatter Plot: {x_axis} vs {y_axis}", fontsize=14)
            st.pyplot(fig)

    else:
        st.warning("Please select at least one feature for analysis.")

    # Feature Importance Analysis with Plotly
    if "WQI" in data.columns:
        st.write("### Interactive Feature Importance in Predicting WQI")
        target = "WQI"
        feature_columns = [col for col in data.columns if col != target]

        X = data[feature_columns].select_dtypes(include=['number']).dropna()
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance = importance.sort_values(ascending=False)

        # Create an interactive bar chart
        fig = px.bar(importance, x=importance.index, y=importance.values, labels={'x': 'Features', 'y': 'Importance'}, title="Feature Importance")
        st.plotly_chart(fig)

    # Interactive Correlation Heatmap with Plotly
    st.write("### Interactive Correlation Heatmap")
    corr_matrix = data.corr()

    # Create a heatmap with Plotly
    fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap")
    fig.update_layout(width=800, height=800)
    st.plotly_chart(fig)

    # Interactive 3D Scatter Plot
    st.write("### Interactive 3D Scatter Plot")
    x_axis_3d = st.selectbox("Select X-axis for 3D Scatter Plot:", options=data.columns)
    y_axis_3d = st.selectbox("Select Y-axis for 3D Scatter Plot:", options=data.columns)
    z_axis_3d = st.selectbox("Select Z-axis for 3D Scatter Plot:", options=data.columns)

    if x_axis_3d and y_axis_3d and z_axis_3d:
        fig = px.scatter_3d(data, x=x_axis_3d, y=y_axis_3d, z=z_axis_3d, color="WQI", title="3D Scatter Plot")
        st.plotly_chart(fig)

    # Pairwise Feature Comparison with Plotly
    st.write("### Pairwise Feature Comparison")
    pairwise_features = st.multiselect("Select features for pairwise comparison:", data.columns, key="pairwise_features")

    if len(pairwise_features) > 1:
        fig = px.scatter_matrix(data, dimensions=pairwise_features, title="Pairwise Feature Comparison")
        fig.update_layout(width=800, height=800)
        st.plotly_chart(fig)

        # KDE Plot for Feature Distribution
    st.write("### KDE Plot for Feature Distribution")
    kde_feature = st.selectbox("Select feature for KDE plot:", options=data.columns)

    if kde_feature:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.kdeplot(data[kde_feature], ax=ax, color="blue", shade=True)
        ax.set_title(f"KDE Plot of {kde_feature}", fontsize=14)
        ax.set_xlabel(kde_feature, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        st.pyplot(fig)

    # Interactive Feature Correlation
    st.write("### Interactive Feature Correlation")
    feature1 = st.selectbox("Select the first feature:", options=data.columns)
    feature2 = st.selectbox("Select the second feature:", options=data.columns)

    if feature1 and feature2:
        # Convert non-numeric columns to numeric if possible
        if data[feature1].dtype == 'object':
            try:
                data[feature1] = pd.to_datetime(data[feature1])
                data[feature1] = data[feature1].apply(lambda x: x.timestamp())
            except Exception as e:
                st.warning(f"Could not convert {feature1} to numeric: {e}")
        
        # If feature2 is 'WQI_Category', map categorical values to numeric
        if feature2 == 'WQI_Category':
            category_mapping = {
                'Excellent': 1,
                'Good': 2,
                'Poor': 3,
                'Very Poor': 4
            }
            data[feature2] = data[feature2].map(category_mapping)
        
        # Convert feature2 to numeric if it's not already
        if data[feature2].dtype == 'object':
            try:
                data[feature2] = pd.to_datetime(data[feature2])
                data[feature2] = data[feature2].apply(lambda x: x.timestamp())
            except Exception as e:
                st.warning(f"Could not convert {feature2} to numeric: {e}")

        # Now calculate the correlation between the two selected features
        correlation = data[feature1].corr(data[feature2])
        st.write(f"The correlation between {feature1} and {feature2} is: {correlation:.2f}")


else:
    st.info("Please upload a CSV or JSON file to get started.")

# Footer
st.write("Developed with \u2764\ufe0f using Streamlit")
