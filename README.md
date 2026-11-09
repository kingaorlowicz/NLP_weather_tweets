# Partly Sunny with a Chance of Hashtags
## Overview
This repository contains a solution for the Kaggle competition "Partly Sunny with a Chance of Hashtags," hosted in partnership with CrowdFlower. The challenge revolves around analyzing a dataset of weather-related tweets to extract multiple layers of information from just a few characters. Technically, this is a Natural Language Processing  and Multi-Target Regression problem, as the objective is to predict continuous confidence scores for multiple classes simultaneously rather than assigning hard binary labels.

## Goal
The main objective is to analyze each tweet and predict the confidence score (ranging from 0 to 1) for 24 distinct labels. The evaluation metric for the competition is the Root Mean Squared Error (RMSE).

The 24 labels are divided into three categories:

- Sentiment (s1-s5): Measures the sentiment of the tweet (e.g., Positive, Negative, Neutral, I can't tell, Not related).
- When (w1-w4): Determines the time frame referenced (e.g., Current, Future, Past, I can't tell).
- Kind (k1-k15): Specifies the exact weather phenomenon mentioned (e.g., clouds, cold, rain, tornado, hurricane).

## Dataset Description
The dataset consists of a training set and a test set containing tweets and metadata:
- Train Set: 77,946 rows. Contains id, tweet, state, location, and 24 columns representing the confidence 
- Test Set: 42,157 rows. Contains id, tweet, state, and location.

Missing Data: A preliminary check revealed that the state column in the test set is missing ~66% of its values, and location also contains missing data. Because of this, these metadata columns were deemed unreliable for modeling.

## Data cleaning
To prepare the data for machine learning algorithms, the following cleaning steps were applied to the dataset:

1. Dropping Metadata: The state and location columns were completely removed from both the train and test sets to focus solely on the text data.
2. Text Preprocessing: A custom regex-based cleaning function was applied to the tweet column:

    - Converted all text to lowercase.
    - Removed all URLs (e.g., {link}, http://...).
    - Removed user mentions (e.g., @mention, @user).
    - Removed the # symbol from hashtags but kept the hashtag text, as it contains highly relevant weather information.
    - Removed special characters and punctuation.
    - Stripped multiple whitespaces.

## Exploratory Data Analysis (EDA)

A comprehensive Exploratory Data Analysis was conducted to understand the underlying patterns in the text and target variables:
- **Target Distribution**: The distribution of weather phenomena (k1-k15) is heavily imbalanced. "I can't tell" is the most frequent label, while extreme weather events like "hurricane" and "tornado" are extremely rare (long-tail distribution). Most tweets are labeled as "Neutral / sharing info" (Sentiment) and "current weather" (When).

<figure align="center">
  <img src="figures/target_dist.png" 
  alt="Kind Distribution">
</figure>

<figure align="center">
  <img src="figures/sentiment_time_dist.png" 
  alt="Time and sentiment Distribution">
</figure>

- **Correlation:** A correlation matrix of the weather phenomena (k1-k15) showed very weak linear relationships overall. There are minor, logical positive correlations (e.g., between "hot" and "humid"), but the overall independence of the targets supports using models that treat each target separately.

<figure align="center">
  <img src="figures/corr_matrix.png" 
  alt="Correlation Matrix">
</figure>

- **Tweet Length Impact:** An analysis of word counts revealed that the vast majority of tweets are between 5 and 20 words long. Interestingly, "very short" tweets show a noticeably higher average confidence score for the s1 ("I can't tell") category, making tweet length a potentially valuable feature.

<figure align="center">
  <img src="figures/word_count_dist.png" 
  alt="Word count distribution">
</figure>

<figure align="center">
  <img src="figures/tweet_length_vs_s1.png" 
  alt="Tweet Length Impact">
</figure>

- **Hashtag Analysis:** Extracting the top 20 most frequent hashtags showed #weather as the absolute leader. While some noisy tags were present (#xb0, #fb), many highly predictive tags were frequently used (e.g., #rain, #cold, #tornado), validating the decision to keep the hashtag text during the cleaning phase.

<figure align="center">
  <img src="figures/top20_hashtags.png" 
  alt="Top 20 most freq hashtags">
</figure>


## Baseline Model
To get things started, I built a quick and simple baseline model just to see where I stand.

First, I used a TfidfVectorizer:  `TfidfVectorizer(max_features=15000, ngram_range=(1, 2))`.

For the algorithm, I went with Ridge Regression. 

The Post-processing Trick:
Since the model doesn't know the actual rules of the Kaggle competition, it sometimes predicted confidence scores below 0 or above 1. It also didn't know that the scores for sentiment (s1-s5) and time (w1-w4) must sum up to 1. To fix this, I added a post-processing steps.

The baseline model achived RMSE 0.16173 on kaggle leaderboard.

## LightGBM with Feature Engineering

Based on the EDA, I knew that the length of the tweet influenced the "I can't tell" sentiment class. So, I engineered a new numerical feature: word_count.

Because LightGBM doesn't natively output 24 variables at once like Ridge does, I wrapped it in Scikit-learn's MultiOutputRegressor, which essentially trains 24 separate LightGBM models under the hood—one for each target.


## Deep Learning: RoBERTa
While LightGBM provided a better result than Ridge, it eventually reached a performance ceiling. The primary limitation of TF-IDF is that it treats text as a "bag of words," ignoring word order, context, and nuances. To overcome this and significantly improve the score, I transitioned to Transformer models.

I utilized Hugging Face's transformers library to fine-tune the cardiffnlp/twitter-roberta-base model. This specific architecture was chosen because it was pre-trained on millions of actual tweets, allowing it to effectively process Twitter-specific language.

A few key changes for this approach:

- Raw Data: I skiped the previous text cleaning steps.
- Regression Head: I initialized the model with problem_type="regression". This automatically adds a linear layer on top of the RoBERTa architecture to output 24 continuous values, optimizing for MSE.
- No Engineered Features: I removed the word_count feature.

After fine-tuning for a few epochs on a GPU and applying the previous post-processing rules (normalizing the sentiment and time probabilities), the model achieved a substantial improvement in overall performance.

## Results
| Model | RMSE (Kaggle public leaderboard)|  
| --- | --- |
| Baseline: TD-IDF + Ridge | 0.16173 |
| LightGBM | 0.15806 |
| RoBERTa | **0.12441** |

## Conclusions
Working through this NLP competition highlighted several key takeaways regarding modern machine learning pipelines:
1. Context is King: The massive leap in performance from LightGBM to RoBERTa proves that treating text as a mere "bag of words" (TF-IDF) is insufficient for complex tasks like sentiment analysis. Transformers' ability to understand sentence structure, negation, and context is invaluable.
2. Logic and Post-Processing win free points: Integrating domain knowledge-specifically the Kaggle competition rules regarding mutually exclusive targets summing to 1-into a simple post-processing step reliably improved the RMSE across all models. Ensuring model predictions align with real-world logic is just as important as hyperparameter tuning.


## Repo Structure
```bash
Repository
 ┣ figures
 ┣ analysis_cleaning.ipynb      # data cleaning, EDA
 ┣ baseline_model.ipynb         # TD-IDF + Ridge
 ┣ LGBM_model.ipynb 
 ┣ roberta_transformer.ipynb
 ┣ README.md
 ┗ variableNames.txt
 ```