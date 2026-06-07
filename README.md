# System Rekomendacji Gier Steam (RAG: LLM + FAISS)

Projekt Szpont to inteligentny system rekomendacji gier wideo, bazujący na architekturze Retrieval-Augmented Generation (RAG). Łączy on szybkie wyszukiwanie wektorowe za pomocą FAISS z mocą generatywną sfinetunowanego modelu językowego Qwen, aby dostarczać spersonalizowane, uargumentowane polecenia gier na podstawie naturalnych zapytań użytkownika.

## 📁 Zawartość repozytorium

* **`main.ipynb`** – Główny plik aplikacji. Odpowiada za wczytanie bazy danych, wektoryzację opisów gier, budowę indeksu FAISS (oraz re-ranking) i obsługę interaktywnego czatu z asystentem AI.
* **`training.py`** – Skrypt do douczania (fine-tuningu) bazowego modelu LLM przy użyciu techniki QLoRA, przystosowany do działania na konsumenckich kartach graficznych (np. 6GB VRAM).
* **`dane_treningowe.jsonl`** *(wymagany do treningu)* – Plik ze zbiorem danych instrukcyjnych używanych podczas fine-tuningu.
* **`qwen_gotowy`** - folder z wytrenowanym plastrem
* **`games_faiss.index`** wynik z wektoryzacji
* **`README.md`**
## 🧠 Wykorzystane modele oraz uzycie AI

W projekcie wykorzystano dwa główne modele:

1. **Model Generatywny (LLM): `Qwen/Qwen2.5-1.5B-Instruct`**
   * **Rola:** Odbiera kontekst (listę 5 najlepiej dopasowanych gier z bazy) i generuje ostateczną, spersonalizowaną odpowiedź dla użytkownika w języku polskim.
   * **Fine-tuning (QLoRA):** Model został dodatkowo douczony na małym zbiorze (30 przykładów) w celu lepszego formatowania odpowiedzi i nabrania odpowiedniego tonu. Użyto 4-bitowej kwantyzacji (`BitsAndBytes`), aby zmieścić proces w 6GB pamięci VRAM. Trenowane były tylko wybrane wagi (moduły uwagi `q_proj`, `v_proj` – LoRA adapter).
   
2. **Model Embeddingowy: `paraphrase-multilingual-MiniLM-L12-v2`**
   * **Rola:** Zamienia opisy, tagi i kategorie gier (a także zapytania użytkownika) na gęste wektory o 384 wymiarach.

3. **Baza Wektorowa: `FAISS`**
   * Wykorzystana do błyskawicznego odnajdywania najbliższych sąsiadów w przestrzeni wielowymiarowej (L2). Po wyszukaniu Top-500 wyników, system stosuje autorski **re-ranking**, biorąc pod uwagę odległość semantyczną oraz odsetek pozytywnych recenzji gry na Steamie, aby wyłonić Top-5.

4. **Oraz był użyty Gemini przy tworzeniu projektu:**

   * AI zostało użyte do debugowania problemów ze środowiskiem i pythonem, żeby na bieżąco sprawnie rozwiązywać problemy.
   * AI Zostalo zużyte do stworzenia struktury PDF (Został on pisany w Latexie)
   * AI Zostało użyte do wygenerowania przykładowych json potrzebnych do trenowania
   * AI zostało użyte do pomocy w wyborze modeli na Hugging Face oraz do pomocy przetronowaniu.
   * AI zostało użyte do optymalizacji kodu.

## 🚀 Jak uruchomić projekt i odtworzyć wyniki

1. Wymagania wstępne
Przed uruchomieniem któregokolwiek ze skryptów należy upewnić się, że w środowisku
zainstalowane są odpowiednie biblioteki oraz że dostępne jest środowisko wykonawcze z
obsługą CUDA (karta graficzna NVIDIA), ponieważ oba procesy ściśle wymagają akcele
racji GPU do poprawnego i wydajnego działania. Wymagane pakiety można zainstalować
w terminalu za pomocą menedżera pip:
pip install pandas numpy torch transformers peft trl datasets bitsandbytes
faiss-gpu sentence-transformers kagglehub
2. Krok 1: Przygotowanie i uruchomienie procesu uczenia (tra
ining.py)
Skrypt training.py odpowiada za przeprowadzenie procesu fine-tuningu (QLoRA) na
bazie modelu Qwen/Qwen2.5-1.5B-Instruct. Procedura uruchomienia wygląda następu
jąco:
3. Należy upewnić się, że plik z przygotowanymi danymi treningowymi
o nazwie dane_treningowe.jsonl znajduje się w tym samym katalogu co skrypt
training.py.
4. Uruchomić skrypt z poziomu terminala za pomocą polecenia:
python -X utf8 training.py
5. Skrypt automatycznie pobierze model bazowy, zastosuje 4-bitową kompresję w celu
zaoszczędzenia pamięci VRAM i rozpocznie proces treningu LoRA. 
6. Po zakończeniu domyślnych 100 kroków uczenia (max_steps), sfinetunowane wagi
(adapter) oraz zaktualizowany tokenizer zostaną automatycznie zapisane w nowo
utworzonym folderze ./qwen_gotowy. 
7. Krok2:Uruchomieniegłównej aplikacji rekomendującej (main.ipynb)
Główna aplikacja integruje bazę wektorową FAISS oraz douczony w poprzednim kroku
model do obsługi interaktywnych zapytań użytkownika. Instrukcja uruchomienia:
8. Należy upewnić się, że w katalogu roboczym aplikacji znajdują się:
• Plik z bazą gier games.csv zawierający dane do przeszukiwania.
• Folder ./qwen_gotowy (wygenerowany w kroku treningowym) zawierający na
uczone wagi adaptera. 
9. Otworzyć plik main.ipynb w środowisku obsługującym Jupyter Notebooks (tutaj
uzyto PyCharm) i uruchomić kolejno wszystkie komórki (lub alternatywnie wyeks
portować kod do zwykłego pliku .py i wywołać go w terminalu).
