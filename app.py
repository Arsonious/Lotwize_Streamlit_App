from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
from plotnine import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Load and preprocess data
def load_data():
    ama = pd.read_csv("lotwize_case.csv")
    ama['property_id'] = ama.index + 1

    # Define columns for imputation
    mean_columns = ['property_id', 'monthlyHoaFee', 'resoFacts/storiesTotal', 'nearbyHomes/2/price', 'priceHistory/0/price', 
                    'priceHistory/2/price', 'priceHistory/4/price', 'adTargets/price', 'livingArea', 'bathrooms', 
                    'bedrooms', 'isPremierBuilder', 'lastSoldPrice', 'lotAreaValue', 'lotSize', 'nearbyHomes/0/livingArea', 
                    'nearbyHomes/0/price', 'nearbyHomes/1/livingAreaValue', 'nearbyHomes/1/price', 'photoCount', 'price']
    mode_columns = ["city", "homeType", 'county', 'description']
    
    # Impute missing values
    imputer_mean = SimpleImputer(strategy='mean')
    imputer_mode = SimpleImputer(strategy='most_frequent')
    ama[mean_columns] = imputer_mean.fit_transform(ama[mean_columns])
    ama[mode_columns] = imputer_mode.fit_transform(ama[mode_columns])
    
    # Sentiment analysis for 'description'
    ama['sentiment_score'] = ama['description'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)
    return ama

df = load_data()

# Shiny app UI
app_ui = ui.page_fluid(
    ui.h2("Real Estate Analysis Dashboard"),
    
    ui.row(
        ui.column(6,
            ui.input_action_button("load_data", "Reload Data"),
            ui.output_text("data_summary"),
        ),
    ),
    
    ui.h3("Correlation Matrix"),
    ui.output_plot("correlation_matrix"),
    
    ui.h3("Actual vs Predicted Prices"),
    ui.output_plot("prediction_plot"),
    
    ui.h3("Word Clouds"),
    ui.input_action_button("generate_wordcloud", "Generate Word Clouds"),
    ui.output_image("positive_wordcloud"),
    ui.output_image("negative_wordcloud")
)

# Shiny app server logic
def server(input, output, session):
    # Load data
    @reactive.Effect
    def load():
        if input.load_data():
            global df
            df = load_data()
    
    @output
    @render.text
    def data_summary():
        return f"Data summary:\n{df.describe()}"
    
    # Correlation matrix plot
    @output
    @render.plot
    def correlation_matrix():
        numeric_data = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        return fig
    
    # Model training and prediction plot
    @output
    @render.plot
    def prediction_plot():
        numeric_cols = ["sentiment_score", "bathrooms", "bedrooms", "livingArea", "lotAreaValue"]
        categorical_cols = ["homeType", "city"]
        X = df[numeric_cols + categorical_cols]
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numeric_cols),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
            ]
        )

        model_pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
        model_pipeline.fit(X_train, y_train)
        y_pred_test = model_pipeline.predict(X_test)

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_test, alpha=0.5, label="Predicted vs Actual")
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal Line")
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")
        ax.set_title("Actual vs Predicted Prices")
        ax.legend()
        return fig

    # Generate word clouds
    @reactive.event(input.generate_wordcloud)
    @output
    @render.image
    def positive_wordcloud():
        positive_text = ' '.join(df[df['sentiment_score'] > 0]['description'].fillna(''))
        return create_wordcloud(positive_text, "white")

    @reactive.event(input.generate_wordcloud)
    @output
    @render.image
    def negative_wordcloud():
        negative_text = ' '.join(df[df['sentiment_score'] < 0]['description'].fillna(''))
        return create_wordcloud(negative_text, "black")

# Helper to create word clouds
def create_wordcloud(text, background_color):
    wordcloud = WordCloud(width=800, height=400, background_color=background_color).generate(text)
    img = BytesIO()
    wordcloud.to_image().save(img, format="PNG")
    return "data:image/png;base64,{}".format(base64.b64encode(img.getvalue()).decode())

# Run the app
app = App(app_ui, server)
