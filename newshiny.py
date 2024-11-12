from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Define columns used in models globally, excluding 'avg_sentiment'
numeric_cols = ["resoFacts/hasAssociation", "lotSize", 'priceHistory/0/price', 
                'priceHistory/2/price', 'priceHistory/4/price', "propertyTaxRate", 
                "bathrooms", "bedrooms", 'resoFacts/hasAttachedProperty', 'livingArea', 
                'resoFacts/hasView', 'lotAreaValue', 'monthlyHoaFee', 
                'nearbyHomes/0/livingArea', 'resoFacts/canRaiseHorses']
categorical_cols = ["homeType", "city"]

from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    # Load the dataset
    ama = pd.read_csv("lotwize_case.csv")
    ama['property_id'] = ama.index + 1
    
    # Columns for imputation
    mean_columns = ['monthlyHoaFee', 'resoFacts/storiesTotal', 'nearbyHomes/2/price', 'priceHistory/0/price', 
                    'priceHistory/2/price', 'priceHistory/4/price', 'adTargets/price', 'livingArea', 
                    'bathrooms', 'bedrooms', 'isPremierBuilder', 'lastSoldPrice', 'lotAreaValue', 
                    'lotSize', 'nearbyHomes/0/livingArea', 'nearbyHomes/0/price', 'nearbyHomes/1/livingAreaValue', 
                    'nearbyHomes/1/price', 'photoCount', 'price']
    mode_columns = ["city", "homeType", 'county', 'description']
    
    # Impute missing values
    imputer_mean = SimpleImputer(strategy='mean')
    imputer_mode = SimpleImputer(strategy='most_frequent')
    ama[mean_columns] = imputer_mean.fit_transform(ama[mean_columns])
    ama[mode_columns] = imputer_mode.fit_transform(ama[mode_columns])
    
    # Filter down to columns used in models
    columns_to_keep = numeric_cols + categorical_cols + ['price', 'description', 'property_id']
    df = ama[columns_to_keep]
    
    # Sentiment analysis with rounded values
    df['sentiment_score'] = df['description'].apply(lambda x: round(TextBlob(x).sentiment.polarity, 2) if isinstance(x, str) else 0)
    df['avg_sentiment'] = df.groupby('property_id')['sentiment_score'].transform('mean').round(2)

    return df

# Reduce the keyword extraction results
def extract_keywords(df):
    # Take a sample of descriptions to limit the output
    sample_texts = df['description'].fillna('').sample(50, random_state=42)
    tfidf = TfidfVectorizer(stop_words='english', max_features=5)
    tfidf_matrix = tfidf.fit_transform(sample_texts)
    keywords_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    return keywords_df

# Define helper functions for visualization
def plot_correlation_matrix(df):
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    return fig

def analyze_sentiment(df):
    df['sentiment_score'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)
    return df

def plot_sentiment_distribution(df):
    fig, ax = plt.subplots()
    sns.histplot(df['sentiment_score'], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution of Sentiment Scores")
    ax.set_xlabel("Sentiment Score")
    return fig

def create_wordcloud(text, background_color):
    if not text:
        return {"src": ""}
    wordcloud = WordCloud(width=800, height=400, background_color=background_color).generate(text)
    img = BytesIO()
    wordcloud.to_image().save(img, format="PNG")
    img_base64 = base64.b64encode(img.getvalue()).decode("utf-8")
    return {"src": f"data:image/png;base64,{img_base64}"}

# Shiny app UI with sidebar layout
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_radio_buttons("section", "Select Section", 
            choices=["Data Overview", "Sentiment Analysis", "Keyword Extraction", "Model Metrics", "Feature Importances", "Predicted vs Actual"]
        )
    ),
    ui.panel_title("Real Estate Analysis Dashboard"),
    ui.output_text("data_description"),
    ui.output_plot("correlation_matrix"),
    ui.output_plot("sentiment_distribution"),
    ui.output_image("positive_wordcloud"),
    ui.output_image("negative_wordcloud"),
    ui.output_table("keywords_table"),
    ui.output_table("model_metrics"),
    ui.output_plot("feature_importances"),
    ui.output_plot("predicted_vs_actual")
)

# Shiny app server logic
def server(input, output, session):
    df = load_data()
    
    @output
    @render.text
    def data_description():
        print("Rendering data description...")
        return f"Data Summary:\n{df.describe()}"

    @output
    @render.plot
    def correlation_matrix():
        print("Rendering correlation matrix...")
        numeric_columns = df.select_dtypes(include=['number']).columns
        return plot_correlation_matrix(df[numeric_columns])
    
    @output
    @render.plot
    def sentiment_distribution():
        print("Rendering sentiment distribution...")
        df_sentiment = analyze_sentiment(df)
        return plot_sentiment_distribution(df_sentiment)

    @output
    @render.image
    def positive_wordcloud():
        print("Rendering positive wordcloud...")
        # Sample positive descriptions for a concise word cloud
        positive_text = ' '.join(df[df['sentiment_score'] > 0]['description'].dropna().sample(100, random_state=42))
        return create_wordcloud(positive_text, "white")

    @output
    @render.image
    def negative_wordcloud():
        print("Rendering negative wordcloud...")
        # Sample negative descriptions for a concise word cloud
        negative_text = ' '.join(df[df['sentiment_score'] < 0]['description'].dropna().sample(100, random_state=42))
        return create_wordcloud(negative_text, "black")

    @output
    @render.table
    def keywords_table():
        print("Rendering keywords table...")
        keywords_df = extract_keywords(df)
        return keywords_df

    def train_models(df):
        X = df[numeric_cols + categorical_cols]
        y = df["price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        preprocessor = ColumnTransformer([
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numeric_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ])
        
        lasso_model = Lasso(alpha=100.0)
        rf_model = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=5, random_state=42)
        
        pipe_lasso = Pipeline([("preprocessor", preprocessor), ("lasso", lasso_model)])
        pipe_rf = Pipeline([("preprocessor", preprocessor), ("rf", rf_model)])
        
        pipe_lasso.fit(X_train, y_train)
        pipe_rf.fit(X_train, y_train)
        
        metrics = {
            "Lasso MAE": mean_absolute_error(y_test, pipe_lasso.predict(X_test)),
            "RF MAE": mean_absolute_error(y_test, pipe_rf.predict(X_test)),
            "Lasso R2": r2_score(y_test, pipe_lasso.predict(X_test)),
            "RF R2": r2_score(y_test, pipe_rf.predict(X_test))
        }
        return metrics, pipe_rf
    
    metrics, pipe_rf = train_models(df)
    
    @output
    @render.table
    def model_metrics():
        print("Rendering model metrics...")
        return pd.DataFrame(metrics, index=[0])

    @output
    @render.plot
    def feature_importances():
        print("Rendering feature importances...")
        importances = pipe_rf.named_steps['rf'].feature_importances_
        feature_names = numeric_cols + list(pipe_rf.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out())
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        ax.set_title("Feature Importances")
        return fig

    @output
    @render.plot
    def predicted_vs_actual():
        print("Rendering predicted vs actual plot...")
        y_test = df["price"]
        y_pred = pipe_rf.predict(df[numeric_cols + categorical_cols])
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_title("Predicted vs Actual Prices")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        return fig

app = App(app_ui, server)
