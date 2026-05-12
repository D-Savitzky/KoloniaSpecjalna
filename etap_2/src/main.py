import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download latest version
kagglehub.dataset_download('fronkongames/steam-games-dataset', path='games.csv', output_dir='../data')

games = pd.read_csv("../data/games.csv", encoding='latin-1')


# ======================
# 2. Czyszczenie kolumn
# ======================
games.columns = games.columns.str.strip().str.lower()

print("Kolumny:", games.columns)


# ======================
# 3. Automatyczne znalezienie ceny
# ======================
price_col = None

for col in games.columns:
    if "price" in col:
        price_col = col
        break

print("Wykryta kolumna ceny:", price_col)


# ======================
# 4. Automatyczne znalezienie popularności
# ======================
owners_col = None

for col in games.columns:
    if "owners" in col or "player" in col:
        owners_col = col
        break

print("Wykryta kolumna popularności:", owners_col)


# ======================
# 5. Konwersja danych
# ======================
if price_col:
    games[price_col] = pd.to_numeric(games[price_col], errors='coerce')


# ======================
# 6. EDA
# ======================

# Tworzenie folderu outputs jesli nie istnieje
output_dir = '../outputs'
os.makedirs(output_dir, exist_ok=True)

# 📊 Rozkład cen
if price_col:
    plt.hist(games[price_col].dropna(), bins=50)
    plt.title("Rozkład cen gier")
    plt.xlabel("Cena")
    plt.ylabel("Liczba gier")
    # Najpierw zapis
    plt.savefig(os.path.join(output_dir, 'rozklad_cen.png'))
    plt.show()

# 📊 Popularność vs cena
if price_col and owners_col:
    plt.scatter(games[price_col], games[owners_col])
    plt.xlabel("Cena")
    plt.ylabel("Popularność")
    plt.title("Cena vs popularność")
    # Najpierw zapis
    plt.savefig(os.path.join(output_dir, 'popularnosc_vs_cena.png'))
    plt.show()

# ======================
# 7. Gatunki (jeśli istnieją)
# ======================
genre_col = None

for col in games.columns:
    if "genre" in col:
        genre_col = col
        break

if genre_col:
    genres = games[genre_col].dropna().str.split(';').explode()
    genres.value_counts().head(10).plot(kind='bar')
    plt.title("Najpopularniejsze gatunki")
    # Najpierw zapis
    plt.savefig(os.path.join(output_dir, 'gatunki.png'))
    plt.show()


# ======================
# 8. Korelacja
# ======================
corr = games.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size":8})
# Najpierw zapis
plt.savefig(os.path.join(output_dir, 'korelacje.png'))
plt.show()