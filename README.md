# Movie Recommender System Based on Cosine Similarity

This repository contains a movie recommender system built using cosine similarity. The system recommends movies similar to a given movie based on various features such as genres, keywords, cast, and overview.

## Features

- **Cosine Similarity**: Calculates similarity between movies based on text features.
- **Data Processing**: Handles missing data and processes text fields to create meaningful tags.
- **Recommendations**: Provides a list of movies similar to a given movie.

## Dataset

The system uses the TMDB 5000 Movies and TMDB 5000 Credits datasets, which contain information about movies, their genres, keywords, cast, and crew.

## Installation

To run the system locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-recommender-system.git
   cd movie-recommender-system
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas scikit-learn
   ```

3. Place the `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` files in the project directory.

## Code Explanation

### Importing Libraries

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
```

### Loading and Merging Datasets

```python
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)
```

### Processing Data

#### Converting JSON Fields to Lists

```python
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
```

#### Extracting Top 3 Cast Members

```python
def convert3(input):
    L = []
    if isinstance(input, str):
        input = ast.literal_eval(input)
    count = 0
    for i in input:
        if count < 3:
            L.append(i['name'])
            count += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)
```

#### Extracting Director Name

```python
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
```

### Creating Tags

```python
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1

movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['keywords'] + movies['genres'] + movies['crew'] + movies['overview']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
movies = movies[['id', 'title', 'tags']]
```

### Vectorization and Similarity Calculation

```python
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vector)
```

### Recommendation Function

```python
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(movies.iloc[i[0]].title)
```

## How to Use

1. Ensure you have the datasets in the project directory.
2. Run the code in a Jupyter Notebook or any Python environment.
3. Call the `recommend` function with the title of a movie you like:
   ```python
   recommend('Gandhi')
   ```

## Contributing

If you would like to contribute to the development of this recommender system, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes.
4. Submit a pull request with a description of your changes.

## License

This project is licensed under the MIT License.

## Acknowledgments

Inspired by various movie recommender systems and the use of cosine similarity for text-based recommendations. Enjoy exploring and enhancing the system!
