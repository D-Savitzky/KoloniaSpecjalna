import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# 📊 Rozkład cen
if price_col:
    plt.hist(games[price_col].dropna(), bins=50)
    plt.title("Rozkład cen gier")
    plt.xlabel("Cena")
    plt.ylabel("Liczba gier")
    plt.show()
    plt.savefig('../outputs/rozklad_cen.png')

# 📊 Popularność vs cena
if price_col and owners_col:
    plt.scatter(games[price_col], games[owners_col])
    plt.xlabel("Cena")
    plt.ylabel("Popularność")
    plt.title("Cena vs popularność")
    plt.show()
    plt.savefig('../outputs/popularnosc_vs_cena.png')

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
    plt.show()
    plt.savefig('../outputs/gatunki.png')


# ======================
# 8. Korelacja
# ======================
corr = games.corr(numeric_only=True)

plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size":8})
plt.show()
plt.savefig('../outputs/korelacje.png')