#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import torch

from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


model_id = "Qwen/Qwen2.5-1.5B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


csv_path = "../data/games.csv"

df = pd.read_csv(csv_path, nrows=100000)

target = "Name"

# Usuwamy rekordy bez nazwy gry
df = df.dropna(subset=[target])

# Zamieniamy puste wartości na pusty tekst
df = df.fillna("")

print("Liczba wczytanych gier:", len(df))
print("Kolumny:", df.columns.tolist())


# In[ ]:





# In[ ]:


import kagglehub
kagglehub.dataset_download('fronkongames/steam-games-dataset', path='games.csv', output_dir='../data')


# In[ ]:


# Przygotowanie danych do wyszukiwania

# Kolumny, które opisują grę.
# Name to tytuł
feature_columns = [col for col in df.columns if col != target]

def row_to_text(row):
    parts = []
    for col in feature_columns:
        value = str(row[col])
        if value.strip() != "":
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

df["game_description"] = df.apply(row_to_text, axis=1)

# TF-IDF do wyszukiwania najbardziej podobnych gier
vectorizer = TfidfVectorizer(
    max_features=50000,
    stop_words="english"
)

game_vectors = vectorizer.fit_transform(df["game_description"])

print("Gotowe. Liczba opisów:", game_vectors.shape[0])


# In[ ]:


# wyszukuj gry

def find_best_games(user_question, top_k=5):
    question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(question_vector, game_vectors).flatten()

    best_indices = similarities.argsort()[-top_k:][::-1]

    results = df.iloc[best_indices].copy()
    results["similarity"] = similarities[best_indices]

    return results


# In[ ]:


# Funkcja odpowiedzi LLM

def recommend_game(user_question, top_k=5):
    best_games = find_best_games(user_question, top_k=top_k)

    context = ""

    for i, row in best_games.iterrows():
        context += f"Tytuł: {row[target]}\n"
        context += f"Opis danych: {row['game_description']}\n"
        context += f"Dopasowanie: {row['similarity']:.4f}\n\n"

    prompt = f"""
Jesteś systemem rekomendacji gier.

Użytkownik pyta:
{user_question}

Na podstawie poniższych danych z pliku CSV wybierz najbardziej odpowiedni tytuł gry.
Nie wymyślaj gry spoza danych.
Odpowiedz po polsku.
Podaj:
1. najlepszy tytuł gry,
2. krótkie uzasadnienie,
3. ewentualnie 2 alternatywy.

Dane z CSV:
{context}
"""

    messages = [
        {
            "role": "system",
            "content": "Jesteś pomocnym asystentem do rekomendowania gier na podstawie danych CSV."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    output = pipe(
        messages,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    return output[0]["generated_text"][-1]["content"], best_games[[target, "similarity"]]


# In[ ]:


# Pobieranie pytania z konsoli

while True:
    question = input("\nNapisz, jakiej gry szukasz albo wpisz 'exit': ")

    if question.lower() in ["exit", "quit", "koniec"]:
        break

    answer, matched_games = recommend_game(question, top_k=5)

    print("\nNajbardziej podobne gry z CSV:")
    print(matched_games)

    print("\nOdpowiedź modelu:")
    print(answer)


# In[ ]:




