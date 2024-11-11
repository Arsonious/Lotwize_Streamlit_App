>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('averaged_perceptron_tagger')
>>> nltk.download('brown')
>>> nltk.download('wordnet')
>>> exit()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from wordcloud import WordCloud
from statsmodels.stats.outliers_influence import variance_inflation_factor
import nltk
from nltk import download

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('wordnet')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Streamlit App
def main():
    st.title("Lotwize Case Study Analysis")

    st.sidebar.title("Navigation")
    options = st.sidebar.selectbox("Select Section", 
                                   ["Data Upload", "Data Preprocessing", "Visualization", 
                                    "Modeling - Lasso", "Modeling - Random Forest", 
                                    "Sentiment Analysis", "Keyword Extraction"])

    if options == "Data Upload":
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df['property_id'] = df.index + 1  # Start IDs from 1
            st.write("### Raw Data", df.head())
            st.session_state['df'] = df
        else:
            st.warning("Please upload a CSV file.")

    elif options == "Data Preprocessing":
        st.header("Data Preprocessing")
        if 'df' in st.session_state:
            df = st.session_state['df']
            st.write("### Missing Values Before Imputation")

            # Define columns for imputation
            mean_columns = ['property_id','monthlyHoaFee', 'resoFacts/storiesTotal', 'nearbyHomes/2/price', 
                            'priceHistory/0/price', 'priceHistory/2/price', 'priceHistory/4/price',
                            'adTargets/price', 'livingArea', 'bathrooms', 'bedrooms', 'isPremierBuilder', 
                            'lastSoldPrice', 'lotAreaValue', 'lotSize', 'nearbyHomes/0/livingArea', 
                            'nearbyHomes/0/price', 'nearbyHomes/1/livingAreaValue', 'nearbyHomes/1/price', 
                            'photoCount', 'price', 'priceHistory/0/priceChangeRate',
                            'priceHistory/0/pricePerSquareFoot', 'priceHistory/1/price', 
                            'propertyTaxRate', 'rentZestimate', 'resoFacts/canRaiseHorses', 
                            'resoFacts/parkingCapacity', 'resoFacts/hasAssociation',
                            'resoFacts/hasAttachedProperty', 'resoFacts/hasHomeWarranty', 
                            'resoFacts/hasView', 'resoFacts/pricePerSquareFoot', 'resoFacts/yearBuilt', 
                            'zestimate']

            mode_columns = ["city", "homeType", 'county','schools/2/level', 'resoFacts/architecturalStyle',
                            'resoFacts/view/0', 'description']

            st.write("#### Numerical Columns with Missing Values")
            st.write(df[mean_columns].isnull().sum())

            st.write("#### Categorical Columns with Missing Values")
            st.write(df[mode_columns].isnull().sum())

            # Impute missing values
            imputer_mean = SimpleImputer(strategy='mean')
            imputer_mode = SimpleImputer(strategy='most_frequent')

            df[mean_columns] = imputer_mean.fit_transform(df[mean_columns])
            df[mode_columns] = imputer_mode.fit_transform(df[mode_columns])

            st.write("### Missing Values After Imputation")
            st.write("#### Numerical Columns")
            st.write(df[mean_columns].isnull().sum())
            st.write("#### Categorical Columns")
            st.write(df[mode_columns].isnull().sum())

            # Optionally, inspect distributions before and after imputation
            st.write("### DataFrame Description (Numerical Columns)")
            st.write(df.describe())
            st.write("### Categorical Columns Value Counts")
            for col in mode_columns:
                st.write(f"#### {col}")
                st.write(df[col].value_counts())

            # Drop rows with more than 200 missing values
            dropped = df.dropna(thresh=df.shape[1] - 200)
            st.write(f"Rows after dropping: {dropped.shape[0]}")
            st.session_state['df'] = dropped

        else:
            st.warning("Please upload and preprocess the data first.")

    elif options == "Visualization":
        st.header("Data Visualization")
        if 'df' in st.session_state:
            df = st.session_state['df']
            st.write("### Correlation Matrix")
            numeric_columns = df.select_dtypes(include=['number']).columns
            corr_matrix = df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                        vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, 
                        linecolor='black', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Please upload and preprocess the data first.")

    elif options == "Modeling - Lasso":
        st.header("Modeling - Lasso Regression")
        if 'df' in st.session_state:
            df = st.session_state['df']

            # Sentiment Analysis
            st.subheader("Sentiment Analysis")
            def analyze_sentiment(text):
                if isinstance(text, str):
                    return TextBlob(text).sentiment.polarity
                return 0

            df['sentiment_score'] = df['description'].apply(analyze_sentiment)
            sentiment_agg = df.groupby('property_id')['sentiment_score'].mean().reset_index()
            sentiment_agg.rename(columns={'sentiment_score': 'avg_sentiment'}, inplace=True)
            df = df.merge(sentiment_agg, on='property_id', how='left')

            # Keyword Extraction
            st.subheader("Keyword Extraction")
            def extract_keywords(texts):
                tfidf = TfidfVectorizer(stop_words='english', max_features=10)
                tfidf_matrix = tfidf.fit_transform(texts)
                return tfidf.get_feature_names_out(), tfidf_matrix.toarray()

            keywords, keyword_matrix = extract_keywords(df['description'].fillna(''))
            keywords_df = pd.DataFrame(keyword_matrix, columns=[f"{keyword}_keyword" for keyword in keywords])
            df = pd.concat([df, keywords_df], axis=1)

            st.write("### Data with Sentiment and Keywords")
            st.write(df.head())

            # Feature Selection
            st.subheader("Feature Selection")
            numeric_cols = ["avg_sentiment", "resoFacts/hasAssociation", "lotSize",
                           'priceHistory/0/price', 'priceHistory/2/price',
                           'priceHistory/4/price', "propertyTaxRate",
                           "bathrooms", "bedrooms", 'resoFacts/hasAttachedProperty',
                           'livingArea', 'resoFacts/hasView', 'lotAreaValue',
                           'monthlyHoaFee', 'nearbyHomes/0/livingArea', 
                           'resoFacts/canRaiseHorses'] + [f"{keyword}_keyword" for keyword in keywords]

            categorical_cols = ["homeType", "city"]

            X = df[numeric_cols + categorical_cols]
            y = df["price"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            # Column Transformer
            z = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())
                    ]), numeric_cols),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
                ],
                remainder='passthrough'
            )

            # Lasso Model
            lasso_best = Lasso(alpha=100.0)
            pipe_best = Pipeline([
                ("preprocessor", z),
                ("lassoreg", lasso_best)
            ])

            pipe_best.fit(X_train, y_train)
            y_pred_train_best = pipe_best.predict(X_train)
            y_pred_test_best = pipe_best.predict(X_test)

            # Model Performance
            st.subheader("Model Performance on Training Set")
            st.write("Train MSE : ", mean_squared_error(y_train, y_pred_train_best))
            st.write("Train MAE : ", mean_absolute_error(y_train, y_pred_train_best))
            st.write("Train MAPE: ", mean_absolute_percentage_error(y_train, y_pred_train_best))
            st.write("Train R²  : ", r2_score(y_train, y_pred_train_best))

            st.subheader("Model Performance on Testing Set")
            st.write("Test MSE  : ", mean_squared_error(y_test, y_pred_test_best))
            st.write("Test MAE  : ", mean_absolute_error(y_test, y_pred_test_best))
            st.write("Test MAPE : ", mean_absolute_percentage_error(y_test, y_pred_test_best))
            st.write("Test R²   : ", r2_score(y_test, y_pred_test_best))

            # Scatter Plot
            st.subheader("Scatter Plot of Actual vs Predicted Values (Test Set)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=y_pred_test_best, alpha=0.5, ax=ax)
            ax.set_xlabel('Actual values')
            ax.set_ylabel('Predicted values')
            ax.set_title('Scatter Plot of Actual vs Predicted Values')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', linestyle='--')
            st.pyplot(fig)

            # Display Lasso Coefficients
            st.subheader("Lasso Regression Coefficients")
            lasso_model = pipe_best.named_steps["lassoreg"]
            preprocessor = pipe_best.named_steps["preprocessor"]
            try:
                encoded_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features=categorical_cols)
            except:
                encoded_cols = preprocessor.named_transformers_['cat'].get_feature_names(categorical_cols)
            feature_names = list(encoded_cols) + numeric_cols
            coefficients = lasso_model.coef_
            coefficients_dict = dict(zip(feature_names, coefficients))
            coef_df = pd.DataFrame.from_dict(coefficients_dict, orient='index', columns=['Coefficient'])
            coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
            st.write(coef_df)

        else:
            st.warning("Please upload and preprocess the data first.")

    elif options == "Modeling - Random Forest":
        st.header("Modeling - Random Forest Regressor")
        if 'df' in st.session_state:
            df = st.session_state['df']

            # Assuming sentiment and keywords are already computed
            if 'avg_sentiment' not in df.columns:
                st.warning("Please perform Sentiment Analysis and Keyword Extraction first.")
                return

            # Feature Selection
            numeric_cols = ["avg_sentiment", "resoFacts/hasAssociation", "lotSize",
                           'priceHistory/0/price', 'priceHistory/2/price',
                           'priceHistory/4/price', "propertyTaxRate",
                           "bathrooms", "bedrooms", 'resoFacts/hasAttachedProperty',
                           'livingArea', 'resoFacts/hasView', 'lotAreaValue',
                           'monthlyHoaFee', 'nearbyHomes/0/livingArea', 
                           'resoFacts/canRaiseHorses'] + [f"{keyword}_keyword" for keyword in keywords]

            categorical_cols = ["homeType", "city"]

            X = df[numeric_cols + categorical_cols]
            y = df["price"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            # Column Transformer
            z = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())
                    ]), numeric_cols),
                    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
                ],
                remainder='passthrough'
            )

            # Random Forest Model
            rf_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=5,
                min_samples_leaf=5,
                min_samples_split=10,
                max_features='sqrt',
                random_state=42
            )

            pipe_rf = Pipeline([
                ("preprocessor", z),
                ("rfreg", rf_model)
            ])

            pipe_rf.fit(X_train, y_train)
            y_pred_train_rf = pipe_rf.predict(X_train)
            y_pred_test_rf = pipe_rf.predict(X_test)

            # Model Performance
            st.subheader("Model Performance on Training Set")
            st.write("Train MSE : ", mean_squared_error(y_train, y_pred_train_rf))
            st.write("Train MAE : ", mean_absolute_error(y_train, y_pred_train_rf))
            st.write("Train MAPE: ", mean_absolute_percentage_error(y_train, y_pred_train_rf))
            st.write("Train R²  : ", r2_score(y_train, y_pred_train_rf))

            st.subheader("Model Performance on Testing Set")
            st.write("Test MSE  : ", mean_squared_error(y_test, y_pred_test_rf))
            st.write("Test MAE  : ", mean_absolute_error(y_test, y_pred_test_rf))
            st.write("Test MAPE : ", mean_absolute_percentage_error(y_test, y_pred_test_rf))
            st.write("Test R²   : ", r2_score(y_test, y_pred_test_rf))

            # Scatter Plot
            st.subheader("Scatter Plot of Actual vs Predicted Values (Test Set)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=y_pred_test_rf, alpha=0.5, ax=ax)
            ax.set_xlabel('Actual values')
            ax.set_ylabel('Predicted values')
            ax.set_title('Scatter Plot of Actual vs Predicted Values')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='r', linestyle='--')
            st.pyplot(fig)

            # Feature Importances
            st.subheader("Feature Importances from Random Forest Model")
            forest_model = pipe_rf.named_steps["rfreg"]
            preprocessor = pipe_rf.named_transformers_['preprocessor']
            try:
                encoded_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features=categorical_cols)
            except:
                encoded_cols = preprocessor.named_transformers_['cat'].get_feature_names(categorical_cols)
            feature_names_rf = list(encoded_cols) + numeric_cols
            importances = forest_model.feature_importances_
            feature_importances = pd.DataFrame({'Feature': feature_names_rf, 'Importance': importances})
            feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
            st.write(feature_importances)

            # Plot Feature Importances
            st.subheader("Plot of Feature Importances")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax, palette='viridis')
            ax.set_title('Feature Importances from Random Forest Model')
            st.pyplot(fig)

        else:
            st.warning("Please upload and preprocess the data first.")

    elif options == "Sentiment Analysis":
        st.header("Sentiment Analysis")
        if 'df' in st.session_state:
            df = st.session_state['df']
            st.write("### Sentiment Score Distribution")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df['sentiment_score'], bins=30, kde=True, ax=ax)
            ax.set_title('Distribution of Sentiment Scores')
            ax.set_xlabel('Sentiment Score')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)

            st.write("### Example Sentiment Analysis on New Review")
            new_review = st.text_input("Enter a new property description for sentiment analysis:", 
                                       "This property is amazing! It has beautiful views and is very spacious.")
            if st.button("Analyze Sentiment"):
                sentiment_score = analyze_sentiment(new_review)
                if sentiment_score > 0.1:
                    sentiment_class = "Positive"
                elif sentiment_score < -0.1:
                    sentiment_class = "Negative"
                else:
                    sentiment_class = "Neutral"
                st.write(f"**Review:** {new_review}")
                st.write(f"**Sentiment Score:** {sentiment_score:.2f}")
                st.write(f"**Sentiment Classification:** {sentiment_class}")
        else:
            st.warning("Please upload and preprocess the data first.")

    elif options == "Keyword Extraction":
        st.header("Keyword Extraction")
        if 'df' in st.session_state:
            df = st.session_state['df']
            st.write("### Word Clouds for Positive and Negative Sentiments")

            # Filter positive and negative descriptions
            positive_descriptions = df[df['sentiment_score'] > 0]['description']
            negative_descriptions = df[df['sentiment_score'] < 0]['description']

            positive_text = ' '.join(positive_descriptions)
            negative_text = ' '.join(negative_descriptions)

            positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
            negative_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(negative_text)

            # Plot the word clouds
            fig, axes = plt.subplots(1, 2, figsize=(24, 12))
            axes[0].imshow(positive_wordcloud, interpolation='bilinear')
            axes[0].axis('off')
            axes[0].set_title('Positive Sentiment Word Cloud')

            axes[1].imshow(negative_wordcloud, interpolation='bilinear')
            axes[1].axis('off')
            axes[1].set_title('Negative Sentiment Word Cloud')

            st.pyplot(fig)

            # Keyword Frequencies
            st.write("### Keyword Frequencies")
            keyword_sums = df[[f"{keyword}_keyword" for keyword in keywords]].sum().reset_index()
            keyword_sums.columns = ['Keyword', 'Frequency']
            keyword_sums = keyword_sums.sort_values(by='Frequency', ascending=False)
            st.write(keyword_sums)

            # Plot Keyword Frequencies
            st.write("### Plot of Keyword Frequencies")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Frequency', y='Keyword', data=keyword_sums, ax=ax, palette='magma')
            ax.set_title('Keyword Frequencies in Descriptions')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Keyword')
            st.pyplot(fig)
        else:
            st.warning("Please upload and preprocess the data first.")

if __name__ == "__main__":
    main()
