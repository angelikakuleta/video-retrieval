# Analiza wpływu komponentów wizualno-językowych i klasycznych na skuteczność przeszukiwania nagrań wideo

Kod źródłowy, konfiguracje i wyniki eksperymentów. Repozytorium pracy dyplomowej, Wydział Elektryczny Politechniki Warszawskiej, 2026.

---

## O czym jest to badanie

Potok przeszukiwania nagrań wideo zbudowany jest z komponentów o różnym charakterze: bazowej reprezentacji kontrastowej, generatorów opisów scen opartych na modelach wizualno-językowych oraz klasycznych detektorów obiektów, twarzy i ruchu. Praca bada, w jakim stopniu każdy z tych komponentów wnosi wkład do skuteczności wyszukiwania, i jak ten wkład zależy od charakteru nagrań oraz typu zapytania.

Metodą jest badanie ablacyjne: ten sam potok uruchamiany jest wielokrotnie na tym samym materiale, przy zmieniającym się zestawie aktywnych komponentów. Każdy wariant opisany jest jednym plikiem konfiguracyjnym.

---

## Odtworzenie środowiska

Środowisko przygotowano dla systemu Windows 11 i karty NVIDIA RTX 5080 (architektura Blackwell, `sm_120`).

```bash
conda env create -f environment.yml
conda activate wideo
python scripts/check_gpu.py
```

**Wiążący jest `requirements.txt`, nie `environment.yml`.** Polecenie `conda env export` zapisuje pakiet `pytorchvideo` jako zwykłe `pytorchvideo==0.1.5`, gubiąc adres repozytorium i identyfikator rewizji. Odtworzenie środowiska wyłącznie z `environment.yml` zainstaluje wydanie z repozytorium pakietów, które odwołuje się do modułu usuniętego z nowszych wersji biblioteki `torchvision` i nie działa. Poprawny zapis znajduje się w `requirements.txt`.

**PyTorch instaluje się z osobnego indeksu.** Pakiet pobierany z domyślnego repozytorium PyPI jest na Windows zbudowany bez obsługi CUDA - uruchomi się i policzy, ale na procesorze, bez żadnego ostrzeżenia:

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

Wariant `cu126` nie zawiera skompilowanego kodu dla architektury `sm_120`.

**`onnxruntime-gpu` musi być w wersji co najmniej 1.27.** Wcześniejsze wydania nie zawierają kodu dla `sm_120` i po cichu przenoszą obliczenia na procesor. Pakiet `onnxruntime` w wariancie procesorowym, instalowany automatycznie jako zależność biblioteki `insightface`, trzeba usunąć - oba dostarczają ten sam moduł i wzajemnie się nadpisują.

Skrypt [`scripts/check_gpu.py`](scripts/check_gpu.py) weryfikuje te warunki i podaje, na jakim urządzeniu faktycznie wykonuje się każdy komponent.

**Modele twarzy nie przychodzą z pakietami.** Potok używa detektora RetinaFace-ResNet-50 (`data/models/retinaface_r50.onnx`, konwersja z repozytorium Pytorch_Retinaface) i embeddingów ArcFace R100 `glintr100` z pakietu `antelopev2` - kroki pobrania i konwersji opisuje [`docs/00_srodowisko.md`](docs/00_srodowisko.md). Pakiet `buffalo_l` nie jest używany.

---

## Układ repozytorium

```
configs/        konfiguracje eksperymentów w formacie YAML; jeden plik = jeden wariant
src/
  annotation/   produkcja adnotacji: format adnotatora, maski, zakresy, rejestr
  data/         rejestr ścieżek zbiorów, czytanie zakresów, zapytań i manifestów wejść
  segmentation/ strategie podziału nagrań (E1), korekta długości, dobór klatek
  features/     ekstraktory: reprezentacje kontrastowe, opisy scen, obiekty, twarze, ruch
  indexing/     budowa indeksów wektorowych i katalogu metadanych fragmentów
  retrieval/    sygnały cząstkowe i ich ważone łączenie
  evaluation/   reguła relewancji, miary rankingowe, agregacja wyników w przekrojach
  runners/      orkiestracja przebiegu: etapy cache-or-compute i zapis artefaktów runu
  utils/        wejście-wyjście, logowanie, walidacja i wczytywanie konfiguracji
scripts/        punkty wejścia uruchamiane z linii poleceń
  prepare_data/ pozyskanie i normalizacja materiału (VATEX, zgrywanie płyt)
notebooks/      prototypowanie i inspekcja wyników pośrednich
  prepare_data/ droga od materiału do gotowych adnotacji; jeden notatnik na krok
tools/
  interval-annotator/  przeglądarkowe narzędzie do adnotowania odcinków
  judge/               ocena par fraza-klasa: program konsolowy i reguły oceny
data/
  raw/          materiał źródłowy: nagrania master, jeden podkatalog na zbiór (poza repozytorium)
  interim/      pliki pośrednie produkujące adnotacje; jeden podkatalog na zbiór
    threshold/  wyjatek: prog nie nalezy do zadnego zbioru (proby do oceny, populacja)
  processed/    nagrania znormalizowane pod potok - CFR, stałe klatki kluczowe (poza repozytorium)
  annotations/  gotowe zapytania testowe wraz z etykietami poprawności; podkatalog na zbiór
  cache/        wyliczone reprezentacje i indeksy (poza repozytorium)
results/
  figures/      wykresy i rysunki do pracy
  measurements/ pomiary środowiska i kosztu komponentów
  reports/      raporty porównań i pojedynczych przebiegów (Markdown + JSON)
  runs/         katalog na przebieg: config.resolved.yaml, metadata.json,
                relevance.csv, per_query.jsonl, metrics.json (nigdy nienadpisywane)
tests/          testy funkcji czystych: korekta długości, reguła relewancji, kontrakt konfiguracji
```

Role warstwy `data/`, po jednym podkatalogu na zbiór:

- **`raw/`** - nietknięte nagrania źródłowe (klipy VATEX, bezstratne odcinki seriali).
- **`interim/`** - wszystko, co skrypty czytają i piszą w drodze do gotowej adnotacji: pobrania upstream ([`vatex_validation_v1.0.json`](data/interim/vatex/vatex_validation_v1.0.json), `tvr_*_release.jsonl`, listy Kinetics), raporty pozyskania, eksporty opisów, wybór odcinków, kotwiczenie znaczników czasu. W `interim/<zbiór>/work/` pliki pośrednie pojedynczej procedury - pomiary wideo, przesunięcia klipów, tytuły odcinków; wszystkie odtwarzalne obliczeniem.
- **`processed/`** - nagrania po normalizacji do stałego framerate i stałych klatek kluczowych (wejście potoku; [`scripts/prepare_data/normalize-video.ps1`](scripts/prepare_data/normalize-video.ps1)).
- **`annotations/`** - produkt końcowy: zapytania z etykietami poprawności, dla seriali osobno `<zbiór>_queries_dev.jsonl` i `<zbiór>_queries_test.jsonl`, dla VATEX `vatex_queries_test.jsonl` i `vatex_queries_dev.jsonl` oraz `vatex_check.jsonl` do kontroli poprawności implementacji. Część testowa VATEX służy jako kontrola przenoszenia wniosków na inną domenę i nie uczestniczy w decyzjach konfiguracyjnych; część deweloperska nie uczestniczy w nich także, ale ma własne dwa zadania - wyznaczenie progu dopasowania fraz do nazw klas i kontrolę sygnału ruchu. VATEX nie ma pliku zakresów: klip jest całym fragmentem, więc zakresy powstają w pamięci z plików zapytań (`src/data/ranges.py::vatex_ranges`). Dla seriali leżą tam też adnotacje odcinków w formacie adnotatora (`<zbiór>_<odcinek>_intervals.csv`) - jedyne pliki w repozytorium, których nie da się odtworzyć obliczeniem.
- **`cache/`** - wyliczalne reprezentacje: embeddingi, indeksy FAISS, warianty segmentacji.

Poza repozytorium (`.gitignore`) pozostają `raw/`, `processed/`, `cache/` i duże pobrania upstream w `interim/`. Małe pliki pośrednie i gotowe adnotacje są wersjonowane jako dowód reprodukowalności.

Każdy komponent potoku ma odpowiadający mu moduł w `src/features`, a wszystkie moduły udostępniają wspólny interfejs: przyjmują listę fragmentów i zwracają reprezentację zapisywaną do pamięci podręcznej. Dołożenie kolejnego komponentu nie wymaga zatem zmian w warstwie wyszukiwania ani ewaluacji, a włączenie lub wyłączenie istniejącego sprowadza się do jednej wartości w pliku konfiguracyjnym.

---

## Uruchomienie eksperymentu

```bash
python scripts/run_experiment.py configs/e1a_tbbt.yaml --split dev
```

Plik konfiguracyjny opisuje wyłącznie wariant potoku; oceniana część materiału (`dev`/`test`) jest parametrem uruchomienia. Dzięki temu fazę testową wykonuje się na tym samym, niezmienionym pliku, który służył fazie deweloperskiej - zamrożenie konfiguracji jest dosłowne. Runner waliduje konfigurację (Pydantic, schemat zgodny z tabelą w załączniku pracy), gwarantuje artefakty pośrednie w trybie cache-or-compute (przedziały czarnego obrazu -> segmentacja -> embeddingi klatek -> cechy komponentów -> kolekcja z indeksem FAISS; indeksy budowane są **osobno dla dev i test**, więc zapytanie testowe nie może trafić w fragment deweloperski), wykonuje ranking i zapisuje komplet artefaktów do `results/runs/<wariant>_<zbiór>_<split>_<RRRRMMDDGGMMSS>/`: scaloną konfigurację, metadane (wersje pakietów, sumy kontrolne wejść, ziarno), zbiory relewancji, wyniki per zapytanie, agregaty oraz standaryzowane sygnały (`signals.npz`), z których liczy się analiza wrażliwości na dobór wag.

Przebieg rusza wyłącznie odcinki ocenianej części materiału i dolicza wszystko, czego brakuje - poza wejściem, którego nie da się odtworzyć obliczeniem. Wykaz tego wejścia jest w [`docs/02_dane_i_znaczniki.md`](docs/02_dane_i_znaczniki.md), w skrócie także w [`notebooks/README.md`](notebooks/README.md).

Segmentację i cechy komponentów można zbudować zawczasu (wznawialne per odcinek; oba detektory ujęć dzielą jeden przebieg dekodowania, a modele ładowane są rozłącznie):

```bash
python scripts/run_segmentation.py configs/e1b_tbbt.yaml configs/e1c_tbbt.yaml
python scripts/run_features.py configs/e5b_tbbt.yaml configs/e5c_tbbt.yaml --split dev
```

Porównania między wariantami (np. rozstrzygnięcie E1) nie mieszkają w pojedynczym runie - wykonuje je osobny krok na gotowych katalogach runów:

```bash
python scripts/compare_runs.py results/runs/E1-*_dev_* \
    --reference E1-A --metric recall@10 --simplicity E1-A E1-B E1-C \
    --contrast requirements:wymaga_scenerii --out results/reports/e1_decision
```

Ekstrakcja cech dla całego korpusu wykonywana jest jednokrotnie i trwa godziny; pojedynczy przebieg ewaluacyjny, w którym zmienia się wyłącznie zestaw aktywnych komponentów i wagi ich łączenia, operuje na gotowych plikach i trwa minuty.

Pełną specyfikację potoku - architekturę, warstwę adnotacji, wykaz modułów oraz instrukcje krok po kroku dla obu faz, od adnotacji po liczby do rozdziału 6 - zbiera [`docs/README.md`](docs/README.md). Liczby do tabel rozdziału 6 składa [`notebooks/experiments/dev_results.ipynb`](notebooks/experiments/dev_results.ipynb).

---

## Powtarzalność wyników

Ziarno generatora liczb pseudolosowych ustalane jest jawnie i zapisywane razem z wynikami przebiegu. Generowanie opisów scen odbywa się w trybie deterministycznym - model wybiera zawsze najbardziej prawdopodobne słowo - dzięki czemu ta sama klatka daje przy każdym uruchomieniu identyczny opis.

---

## Dostępność danych

Nagrania wykorzystane w badaniu nie podlegają redystrybucji i nie znajdują się w repozytorium. Udostępnione są natomiast zapytania testowe wraz z etykietami poprawności (`data/annotations/`), pełne konfiguracje wszystkich raportowanych eksperymentów oraz surowe wyniki przebiegów, co pozwala zweryfikować obliczenia i miary bez dostępu do materiału źródłowego.
