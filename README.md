# Product Recommendation System

## Overview
This is a content-based product recommendation system that suggests similar products to users based on product features like category and product name. The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization and **Cosine Similarity** to find and rank similar products.

## Table of Contents
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Output](#output)
- [Key Components](#key-components)
- [Advantages](#advantages)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

## Features
✨ **Content-Based Filtering** - Recommends products based on product attributes
✨ **TF-IDF Vectorization** - Converts text features into numerical vectors
✨ **Cosine Similarity** - Calculates similarity between product vectors
✨ **Fuzzy String Matching** - Handles user input variations using `difflib`
✨ **Top-N Recommendations** - Returns the 5 most similar products

## Dependencies
```
numpy          - Numerical computing library
pandas         - Data manipulation and analysis
difflib        - Python standard library for string matching
scikit-learn   - Machine learning library
  - TfidfVectorizer
  - cosine_similarity
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
```bash
pip install numpy pandas scikit-learn
```

## How It Works

### Step 1: Data Loading
The system loads product data from a CSV file containing:
- `customer_id` - Unique customer identifier
- `order_date` - Date of purchase
- `product_id` - Unique product identifier
- `category_id` - Category identifier (numeric)
- `category_name` - Category name (text)
- `product_name` - Product name (text)
- Other fields: quantity, price, payment_method, city, review_score, gender, age

### Step 2: Feature Selection
Selected features for recommendation:
- `category_id` - Numeric category identifier
- `category_name` - Text-based category
- `product_name` - Product description

### Step 3: Feature Combination
Features are combined into a single string:
```
"10 Electronics Smartphone"
"50 Sports & Outdoors Soccer Ball"
"40 Books & Stationery Story Book"
```

### Step 4: TF-IDF Vectorization
Text data is converted into a numerical feature matrix using `TfidfVectorizer`:
- Creates sparse matrix with shape (1000, 41) for 1000 products and 41 unique terms
- Each product becomes a vector of TF-IDF scores

### Step 5: Similarity Calculation
Cosine similarity is computed between all products:
- Results in a 1000×1000 similarity matrix
- Values range from 0 to 1 (1 = identical products)

### Step 6: User Input & Matching
- User enters a product name
- `difflib.get_close_matches()` finds exact or close matches
- Retrieves the index of the best matching product

### Step 7: Recommendation
- Extracts similarity scores for the selected product
- Sorts products by similarity score (descending)
- Returns top 5 most similar products

## Project Structure
```
product_recommendation_system/
│
├── data/
│   └── synthetic_online_retail_data.csv
│
├── script.py (main recommendation script)
│
└── README.md (this file)
```

## Usage

### Basic Workflow
```python
# 1. Import dependencies
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 2. Load data
product_data = pd.read_csv('/content/synthetic_online_retail_data.csv')

# 3. Select and combine features
selected_features = ['category_id', 'category_name', 'product_name']
combined_features = (product_data['category_id'].astype(str) + ' ' + 
                     product_data['category_name'] + ' ' + 
                     product_data['product_name'])

# 4. Vectorize features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# 5. Calculate similarity
similarity = cosine_similarity(feature_vectors)

# 6. Get user input
product_name = input('Enter the product name: ')

# 7. Find close match
list_of_all_products = product_data['product_name'].tolist()
close_match = difflib.get_close_matches(product_name, list_of_all_products)
index_of_product = product_data[product_data.product_name == close_match[0]].index[0]

# 8. Get recommendations
similarity_score = list(enumerate(similarity[index_of_product]))
sorted_similar_products = sorted(similarity_score, key=lambda x: x[1], reverse=True)

# 9. Display recommendations
print('Products suggested for you:\n')
i = 1
for product in sorted_similar_products:
    index = product[0]
    product_from_dataframe = product_data[product_data.index == index]['product_name'].values[0]
    if i < 6:
        print(f"{i}. {product_from_dataframe}")
        i += 1
```

### Example Input & Output
```
Enter the product name: Story Book

Products suggested for you:

1. Story Book
2. Story Book
3. Story Book
4. Story Book
5. Story Book
```

## Key Components

### 1. TfidfVectorizer
Converts text into numerical features by:
- Computing term frequency (how often a word appears)
- Applying inverse document frequency (penalizing common words)
- Normalizing vectors to unit length

**Result:** Sparse matrix (1000, 41) with 3766 non-zero elements

### 2. Cosine Similarity
Measures angle between two vectors:
- Formula: `similarity = (A · B) / (||A|| × ||B||)`
- Range: 0 to 1 (1 = identical, 0 = completely different)
- Symmetric: similarity(A, B) = similarity(B, A)

### 3. difflib.get_close_matches()
Handles user typos and variations:
- Finds approximate matches when exact match not found
- Uses SequenceMatcher algorithm
- Returns list of best matches (up to 3 by default)

## Advantages
✅ **Simple & Interpretable** - Easy to understand how recommendations are generated
✅ **No Cold Start Problem** - Works for new products immediately
✅ **Fast Computation** - Efficient for datasets up to thousands of products
✅ **Scalable** - Can handle large feature sets
✅ **Typo Tolerant** - Handles user input variations
✅ **Domain Agnostic** - Works for any product domain

## Limitations
❌ **Limited to Attributes** - Only uses category and product name
❌ **Ignores User Behavior** - Doesn't consider purchase history or ratings
❌ **Sparse Data** - Might struggle with unique products
❌ **No Temporal Effects** - Doesn't account for trends or seasonality
❌ **Identical Results** - Multiple products with same name get same score
❌ **No Diversity** - All recommendations might be too similar

## Future Improvements

### 1. **Enhanced Features**
- Include product descriptions
- Add price range information
- Incorporate customer reviews/ratings
- Add product images for visual similarity

### 2. **Hybrid Approach**
- Combine content-based with collaborative filtering
- Consider user purchase history
- Weight recommendations by user preferences

### 3. **Better Similarity Metrics**
- Use Word2Vec or GloVe embeddings
- Implement deep learning models (neural networks)
- Use Elasticsearch for full-text search

### 4. **Personalization**
- Track user interactions
- Build user preference profiles
- A/B test different recommendation strategies

### 5. **Performance Optimization**
- Cache similarity matrix for faster queries
- Implement approximate nearest neighbors (ANN)
- Use dimensionality reduction (PCA, UMAP)

### 6. **Evaluation Metrics**
- Precision@K - % of recommended items user interacted with
- Recall@K - % of items user interacted with that were recommended
- NDCG (Normalized Discounted Cumulative Gain)
- Mean Average Precision (MAP)

## Dataset Information
- **Total Records:** 1000 products
- **Columns:** 13
- **Categories:** 5 (Electronics, Sports & Outdoors, Books & Stationery, Fashion, etc.)
- **Date Range:** 2024-2025
- **Missing Values:** Review scores have some NaN values (handled appropriately)

## Mathematical Background

### TF-IDF Formula
```
TF-IDF(t,d) = TF(t,d) × log(N/DF(t))

where:
- TF = Term Frequency (count of term in document)
- DF = Document Frequency (documents containing term)
- N = Total documents
```

### Cosine Similarity Formula
```
similarity(A,B) = (Σ(A_i × B_i)) / (√(Σ(A_i²)) × √(Σ(B_i²)))
```

## Performance Notes
- Vectorization Time: < 1 second
- Similarity Computation: < 1 second
- Recommendation Retrieval: < 100ms
- Memory Usage: ~5-10 MB for 1000 products

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No close match found | Check product name spelling, try different names |
| All recommendations identical | Add more diverse features to input data |
| Slow performance | Reduce dataset size or use approximate similarity |
| Missing data errors | Check for null values in features |

## Author & License
This is a sample educational project for learning content-based recommendation systems.

## References
- [TF-IDF Vectorizer - Scikit-learn](https://scikit-learn.org/stable/modules/generated.tfidfvectorizer.html)
- [Cosine Similarity - Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Information Retrieval Basics](https://nlp.stanford.edu/IR-book/)

---

**Last Updated:** March 2026
**Version:** 1.0
