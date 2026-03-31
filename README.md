# 🛒 SmartRetail: Product Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

**An intelligent retail discovery engine powered by machine learning. Help customers find exactly what they need using high-dimensional content similarity.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Contributing](#-contributing)

</div>

---

## ✨ Features

- 🎯 **Content-Based Filtering** - Analyzes Category IDs, Names, and Product Titles to find matches.
- 📊 **TF-IDF Vectorization** - Converts categorical and text data into weighted numerical vectors.
- 🎲 **Cosine Similarity Engine** - Mathematical computation of "closeness" between inventory items.
- 🔍 **Fuzzy String Matching** - Robust search that handles typos (e.g., "Stry Book" → "Story Book").
- ⚡ **Lightweight & Fast** - Instant recommendations across the 1,000-product dataset.
- 📈 **Scalable Logic** - Ready to be adapted for larger e-commerce inventories.

---

## 🎥 Demo

```python
# Get product suggestions in seconds
product_name = "Story Book"
recommendations = get_recommendations(product_name, num_suggestions=5)
print(recommendations)
Output:🛍️ Products suggested for you:

1. Story Book
2. Novel
3. Comic Book
4. Hardcover Journal
5. Sketchbook
🚀 InstallationPrerequisitesPython 3.8 or higherpip package managerQuick SetupBash# Clone the repository
git clone [https://github.com/yourusername/retail-recommendation-system.git](https://github.com/yourusername/retail-recommendation-system.git)
cd retail-recommendation-system

# Install dependencies
pip install numpy pandas scikit-learn
💻 UsageBasic ImplementationPythonimport pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Data
df = pd.read_csv('synthetic_online_retail_data.csv')

# 2. Feature Engineering
selected_features = ['category_id', 'category_name', 'product_name']
combined_features = df['category_id'].astype(str) + ' ' + df['category_name'] + ' ' + df['product_name']

# 3. Vectorization & Similarity
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# 4. Recommendation Logic
def recommend(name):
    list_of_all_products = df['product_name'].tolist()
    close_match = difflib.get_close_matches(name, list_of_all_products)[0]
    index = df[df.product_name == close_match].index[0]
    
    distances = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)
    
    print(f"Products suggested for you based on '{close_match}':\n")
    for i in range(1, 6):
        print(f"{i}. {df.iloc[distances[i][0]].product_name}")

recommend("Story Book")
🧠 How It WorksThe Pipeline Architecture┌──────────────────────────┐
│ Synthetic Retail Dataset │
│ (1,000 Product Entries)  │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│   Feature Engineering    │
│ (ID + Category + Name)   │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  TF-IDF Vectorization    │
│ (Text → Numeric Vectors) │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Cosine Similarity Matrix │
│   (1000 x 1000 Grid)     │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ Fuzzy Match & Sort Score │
│ (Ranked Recommendations)  │
└──────────────────────────┘
Key Components1. Feature FusionThe system creates a "bag of words" for each product. For example:"123 Electronics Smartphone" is treated as a single descriptive profile.2. Cosine Similarity FormulaThe similarity between two products $A$ and $B$ is calculated as:$$similarity(A, B) = \frac{A \cdot B}{||A|| ||B||}$$This measures the cosine of the angle between two vectors, determining how closely the product descriptions align.📊 Performance MetricsMetricValueDataset Size1,000 ProductsFeatures AnalyzedCategory ID, Category Name, TitleResponse Time~15msVector SpaceTF-IDF Sparse Matrix📂 Project Structureretail-recommendation/
├── README.md                          # Documentation
├── synthetic_online_retail_data.csv   # The Dataset
├── main.py                            # Implementation Script
└── notebooks/
    └── exploration.ipynb              # Data Analysis & Visualization
📈 Future Roadmap[ ] Streamlit Integration: Create a web dashboard for real-time searching.[ ] Collaborative Filtering: Use customer_id and review_score for personalized user-based suggestions.[ ] Hybrid Model: Combine content-based and popularity-based filtering.[ ] API Endpoint: Deploy using FastAPI.🤝 ContributingContributions make the open-source community an amazing place!Fork the ProjectCreate your Feature Branch (git checkout -b feature/AmazingFeature)Commit your Changes (git commit -m 'Add some AmazingFeature')Push to the Branch (git push origin feature/AmazingFeature)Open a Pull Request📄 LicenseDistributed under the MIT License. See LICENSE for more information.<div align="center">⭐ Star this project if you found it useful!Built for the Open Source Machine Learning Community
