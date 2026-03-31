# 🛒 SmartRetail: Product Recommendation System

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

An intelligent content-based recommendation engine that suggests retail products by analyzing metadata and calculating text-based similarities.

---

## 📌 Project Overview
This system uses **Machine Learning** to help users discover products similar to their interests. By processing features like `category_name` and `product_name`, the model builds a mathematical profile for every item in the inventory.

### 🛠️ The Tech Stack
* **Data Analysis:** `Pandas`, `NumPy`
* **NLP & ML:** `TfidfVectorizer`, `Cosine Similarity`
* **Search Logic:** `Difflib` (for fuzzy string matching)

---

## 🧬 How the Engine Works

The recommendation logic follows a structured pipeline to move from raw text to ranked suggestions:

1.  **Feature Fusion:** Combines product IDs, categories, and names into a single "content string."
2.  **TF-IDF Vectorization:** Converts text data into a matrix of numerical features, highlighting unique keywords.
3.  **Cosine Similarity:** Calculates the "distance" between product vectors. Products with a score closer to `1.0` are nearly identical in content.
4.  **Fuzzy Matching:** When a user types a product name, the system handles typos or partial names to find the closest database entry.
5.  **Ranking:** Sorts the entire inventory based on similarity to the selected item and returns the Top 5.

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install numpy pandas scikit-learn

2. Basic Usage
The script will prompt you for a product name:

Plaintext
Enter the product name: Story Book
Output:

Plaintext
Products suggested for you :
1 . Story Book
2 . Story Book
3 . Story Book
4 . Novel (Example)
5 . Hardcover Journal (Example)


📊 Logic VisualizationStepProcessGoal01Data CleaningHandle nulls and normalize text02VectorizationTurn words into numbers ($TF-IDF$)03SimilarityCompute Dot Product of vectors04OutputDisplay Top N similar items

📂 Repository Structure
product_recommendation.py : Main Python logic.

synthetic_online_retail_data.csv : Dataset containing 1000+ entries.

README.md : Project documentation.

💡 Future Roadmap
[ ] Build a web UI using Streamlit.

[ ] Add Collaborative Filtering to include user ratings.

[ ] Implement Image-based recommendations using Computer Vision.
