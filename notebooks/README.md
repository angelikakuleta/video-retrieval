# Notatniki

Miejsce, w którym się liczy, przygotowuje adnotacje i podgląda wyniki. Kod, który się ustabilizuje, przenosi się do `src/`, a notatnik zostaje jako zapis tego, jak do wyniku doszło.

| plik | do czego |
|---|---|
| [`environment_check.ipynb`](environment_check.ipynb) | sprawdzenie każdego komponentu potoku: czy się ładuje, na czym liczy, ile pamięci zajmuje, jak długo trwa |
| [`vatex_check.ipynb`](vatex_check.ipynb) | kontrola poprawności implementacji potoku bazowego na zbiorze VATEX |
| `prepare_data/` | droga od materiału do gotowych plików zapytań; jeden notatnik na krok |
| `experiments/` | liczby z gotowych katalogów przebiegów; nic tam nie liczy na nagraniach (wyjątkiem jest [`examples.ipynb`](experiments/examples.ipynb), który dekoduje klatki do rysunków przykładów) |

---

## Uruchomienie po raz pierwszy

**1. Doinstalować obsługę notatników w środowisku `wideo`.** W Miniforge Prompt:

```cmd
conda activate wideo
pip install ipykernel ipywidgets nbstripout
```

`ipywidgets` odpowiada za paski postępu (`tqdm`, pobieranie modeli z Hugging Face); bez niego zamiast paska pojawia się ostrzeżenie i ściana tekstu.

**2. Zarejestrować kernel pod czytelną nazwą:**

```cmd
python -m ipykernel install --user --name wideo --display-name "Python (wideo)"
```

Bez tego kroku VS Code i tak zwykle znajduje środowisko, ale na liście widnieje ono jako surowa ścieżka do `python.exe`, a notatniki w tym repozytorium mają w metadanych wpisane `wideo`; po rejestracji kernel dobiera się sam.

**3. Włączyć czyszczenie wyników przed commitem:**

```cmd
nbstripout --install
```

Polecenie dopisuje definicję filtra do `.git/config` tego repozytorium; reguła w `.gitattributes` już czeka. Od tej chwili `git add` zdejmuje z notatnika wyniki komórek i numery wykonania.

**4. Otworzyć [`environment_check.ipynb`](environment_check.ipynb)** i w prawym górnym rogu wybrać kernel **Python (wideo)**.

---

## Ustalenia obowiązujące w notatnikach

**Katalogiem roboczym jest korzeń repozytorium**. Ustawia to `jupyter.notebookFileRoot` w [`.vscode/settings.json`](../.vscode/settings.json), dzięki czemu ścieżka `data/raw/vatex` znaczy to samo w notatniku i w skrypcie z `scripts/`.

**Pierwsza komórka kodu jest komórką konfiguracji.** Importy, ścieżki, stałe i wszystko, co da się wczytać z dysku; nic, co liczy. Uruchamia się ją raz po otwarciu notatnika, a potem dowolną inną komórkę osobno, bez wracania do poprzednich. Komórka, która potrzebuje wyniku innej, oznaczona jest w opisie.

**Każdy notatnik zaczyna się od `setup()`** z [`src/utils/notebook.py`](../src/utils/notebook.py). Funkcja dokłada korzeń repozytorium do ścieżki importów, włącza automatyczne przeładowanie modułów z `src/` i nakłada limit 95% pamięci karty. Bez tego ostatniego sterownik potrafi przy braku pamięci po cichu przenieść nadmiar do pamięci operacyjnej i zamiast błędu pojawia się kilkunastokrotne spowolnienie bez żadnej wskazówki.

**Jeden duży model ładowany naraz.** Po zakończeniu pracy z modelem: `release("model", "processor")`. Sama instrukcja `del` nie wystarcza, bo Jupyter trzyma referencję do wyniku ostatniej komórki w `Out` i `_`; `release` czyści jedno i drugie.

**Liczby do pracy pochodzą z `results/measurements/`**. `save_measurement(name, data)` dopisuje do każdego pomiaru datę, wersje bibliotek i nazwę karty. Zestawienie wszystkich zapisanych przebiegów: `summary()`.

**Pliki pośrednie są nadpisywane** przy każdym uruchomieniu. Wyjątki wymienia nagłówek notatnika i powtarza opis odpowiedniego bloku. Są trzy: pomiary z [`environment_check.ipynb`](environment_check.ipynb) (nazwa niesie datę i godzinę, więc kolejne przebiegi zostają obok siebie), pliki `data/annotations/<zbiór>/<zbiór>_<odcinek>_intervals.csv` (żaden notatnik do tego katalogu nie pisze; `tbbt_02` zostawia kandydatów w `candidates/`, a zweryfikowane pliki przenosi się ręcznie) oraz pamięć podręczna w `data/cache/` (budowana raz i potem wczytywana).

---

## Wejście, którego przebieg nie wytworzy

Poniższe pliki są wejściem całej warstwy eksperymentalnej. Powstają z pracy ręcznej albo z jednorazowej normalizacji materiału i żaden przebieg ich nie odtworzy:

| plik | co to jest | skąd |
|---|---|---|
| `data/processed/<zbiór>/*.mp4` | nagrania po normalizacji do stałego framerate i stałych klatek kluczowych | [`scripts/prepare_data/normalize-video.ps1`](../scripts/prepare_data/normalize-video.ps1) |
| `data/annotations/<zbiór>/<zbiór>_<odcinek>_intervals.csv` | adnotacje zdarzeń i maski, w formacie adnotatora | `tools/interval-annotator`, ręcznie |
| `data/annotations/<zbiór>/<zbiór>_query_tags.csv` | znaczniki wymagań, złożoność, postacie | ręcznie, szkielet z ostatniego notatnika zapytań danego zbioru |
| `data/annotations/<zbiór>/<zbiór>_profiles.csv` | wskaźniki wzorcowych twarzy każdej postaci (E6) | ręcznie, z kontaktówek |
| `data/annotations/threshold_judgments.csv` | oceny par fraza / nazwa klasy | `tools/judge/judge_pairs.py`, ręcznie |

Reszta jest wyliczalna w trybie cache-or-compute: przedziały czarnego obrazu, granice fragmentów, embeddingi klatek, cechy komponentów i indeks FAISS kolekcji. Pełny wykaz jest w [`docs/02_dane_i_znaczniki.md`](../docs/02_dane_i_znaczniki.md).

---

## Kolejność w `prepare_data/`

Notatniki uruchamia się po kolei, bez wracania.

| krok | co robi |
|---|---|
| [`vatex_01_dev.ipynb`](prepare_data/vatex_01_dev.ipynb) | zamrożone losowanie i pobranie części deweloperskiej, plik zapytań dev |
| [`vatex_02_test_acquisition.ipynb`](prepare_data/vatex_02_test_acquisition.ipynb) | pobranie i pomiar klipów części testowej |
| [`vatex_03_test_annotations.ipynb`](prepare_data/vatex_03_test_annotations.ipynb) | szkielet znaczników, plik zapytań, statystyki |
| *ręczny* | wypełnienie znaczników w `vatex_query_tags.csv`, potem `vatex_03` jeszcze raz |
| [`tbbt_01_episode_selection.ipynb`](prepare_data/tbbt_01_episode_selection.ipynb) | ranking wg zapytań `v`, dobór 24 odcinków, podział dev/test, opisy `v` |
| *ręczny* | oznaczenie masek w adnotatorze, do `data/annotations/tbbt/masks/` |
| [`tbbt_02_annotation_candidates.ipynb`](prepare_data/tbbt_02_annotation_candidates.ipynb) | czasy na skali odcinka, filtr długości, pliki dla adnotatora w `candidates/` |
| *ręczny* | weryfikacja ręczna, przeniesienie gotowych plików z `candidates/` do `data/annotations/tbbt/` |
| [`tbbt_03_annotations.ipynb`](prepare_data/tbbt_03_annotations.ipynb) | rejestr, zakresy korpusu, szkielet znaczników, pliki zapytań, statystyki |
| [`tbbt_04_profiles.ipynb`](prepare_data/tbbt_04_profiles.ipynb) sekcje 1-2 | bufor twarzy, wektory tożsamości, kontaktówki kandydatów (E6) |
| *ręczny* | wybór 8 przykładów na postać, wpisanie do `tbbt_profiles.csv` |
| [`tbbt_04_profiles.ipynb`](prepare_data/tbbt_04_profiles.ipynb) sekcje 4-6 | kontrola jakości, budowa i bufor profilu |
| [`office_01_episode_selection.ipynb`](prepare_data/office_01_episode_selection.ipynb) | podział dev/test ręcznie wybranych odcinków |
| *ręczny* | adnotowanie od zera w adnotatorze |
| [`office_02_annotations.ipynb`](prepare_data/office_02_annotations.ipynb) | to samo co `tbbt_03` |
| [`office_03_profiles.ipynb`](prepare_data/office_03_profiles.ipynb) | to samo co `tbbt_04` |

Notatniki profili dla tożsamości (E6) są osobną gałęzią. Dopiero po nich zaczyna się `experiments/`.
