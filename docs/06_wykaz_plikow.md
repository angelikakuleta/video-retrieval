# Wykaz plików

## Moduły

Wszystkie pliki `.py` repozytorium, po katalogach. Opis odpowiada pierwszemu zdaniu docstringa modułu - dłuższe uzasadnienia zostają w kodzie, przy rzeczy, której dotyczą.

### `src/utils/` - Wspólne - konfiguracja, parametry, słowniki, narzędzia

| Plik | Co robi |
|---|---|
| [`config.py`](../src/utils/config.py) | Wczytanie i walidacja konfiguracji potoku |
| [`frozen.py`](../src/utils/frozen.py) | Schemat [`configs/frozen.yaml`](../configs/frozen.yaml): rozstrzygnięcia fazy deweloperskiej w dwu etapach (`STAGE_ONE` - blok `matching`; `STAGE_TWO` - sześć werdyktów), wypełniany ręcznie; `require()` jest bramą fazy testowej |
| [`notebook.py`](../src/utils/notebook.py) | Wspólny start notatników i szybkich sprawdzeń |
| [`queries.py`](../src/utils/queries.py) | Rekord zapytania testowego i jego zapis w formacie JSONL |
| [`settings.py`](../src/utils/settings.py) | Parametry wspólne dla więcej niż jednego notatnika, skryptu lub modułu |
| [`progress.py`](../src/utils/progress.py) | Logowanie pętli, która może przejść przez tysiące pozycji |
| [`tools.py`](../src/utils/tools.py) | Lokalizacja zewnętrznych narzędzi wiersza poleceń |
| [`vatex.py`](../src/utils/vatex.py) | Adapter VATEX: raport pobrania + opisy -> rekordy zapytań i ścieżki klipów |
| [`video_probe.py`](../src/utils/video_probe.py) | Pomiar właściwości plików wideo (ffprobe) |
| [`vocabulary.py`](../src/utils/vocabulary.py) | Zamknięte słowniki schematu adnotacji |

### `src/data/` - Dostęp do danych - ścieżki, zakresy, zapytania, metadane wejścia

| Plik | Co robi |
|---|---|
| [`datasets.py`](../src/data/datasets.py) | Jedyny rejestr nazw zbiorów i ich położenia na dysku |
| [`manifest.py`](../src/data/manifest.py) | Sumy kontrolne plików wejściowych przebiegu - część jego metadanych |
| [`queries.py`](../src/data/queries.py) | Odczyt zapytań testowych jednej części zbioru |
| [`ranges.py`](../src/data/ranges.py) | Odczyt zakresów korpusu zbioru i oś czasu zbudowana z nich |

### `src/annotation/` - Warstwa adnotacji - od opisów źródłowych do plików zapytań

| Plik | Co robi |
|---|---|
| [`adjust.py`](../src/annotation/adjust.py) | Dopasowanie zaimportowanej adnotacji do masek jej odcinka |
| [`build.py`](../src/annotation/build.py) | Rejestr serialu -> pliki zapytań czytane przez potok |
| [`intervals.py`](../src/annotation/intervals.py) | Odczyt i zapis plików przedziałów adnotatora `tools/interval-annotator` |
| [`ranges.py`](../src/annotation/ranges.py) | Zakresy korpusu odcinka - wszystko, czego nie zakrywa maska |
| [`registry.py`](../src/annotation/registry.py) | Rejestr zbiorczy adnotacji serialu - jeden wiersz na adnotację |
| [`tags.py`](../src/annotation/tags.py) | Znaczniki wymagań, etykieta złożoności i nazwane postacie zapytania |
| [`tbbt_time_mapping.py`](../src/annotation/tbbt_time_mapping.py) | Kotwiczenie czasu klipów TVQA/TVR na skali całego odcinka |

### `src/segmentation/` - Segmentacja i oś czasu treści

| Plik | Co robi |
|---|---|
| [`black.py`](../src/segmentation/black.py) | Przedziały (niemal) czarnego obrazu, liczone raz na odcinek i cachowane |
| [`build.py`](../src/segmentation/build.py) | Budowa bufora segmentów - jedna implementacja za skryptem i za przebiegiem |
| [`decode.py`](../src/segmentation/decode.py) | Sekwencyjne dekodowanie klatek dla detektorów granic ujęć |
| [`frames.py`](../src/segmentation/frames.py) | Próbkowanie klatek nagrania na stałej siatce czasu |
| [`histogram.py`](../src/segmentation/histogram.py) | Nieuczona detekcja granic ujęć: różnica histogramów sąsiednich klatek |
| [`segments.py`](../src/segmentation/segments.py) | Strategie segmentacji dla eksperymentu E1 |
| [`timeline.py`](../src/segmentation/timeline.py) | Oś czasu treści jednego odcinka |
| [`transnet.py`](../src/segmentation/transnet.py) | Uczona detekcja granic ujęć: TransNetV2 (wariant E1-C) |

### `src/features/` - Ekstraktory - jeden moduł na komponent, wspólny kontrakt

| Plik | Co robi |
|---|---|
| [`captions.py`](../src/features/captions.py) | Opisy scen dla klatek siatki: sygnał opisowy (E3) |
| [`encoders.py`](../src/features/encoders.py) | Enkoder konfiguracji: jedno miejsce, które rozstrzyga, który to jest |
| [`episode_cache.py`](../src/features/episode_cache.py) | Wspólna pętla każdej pamięci podręcznej liczonej per odcinek |
| [`expressions.py`](../src/features/expressions.py) | Rozpoznawanie mimiki: sygnał mimiki eksperymentu E5 |
| [`faces.py`](../src/features/faces.py) | Detekcja i wyrównanie twarzy: bufor, na którym stoją trzy sygnały |
| [`frame_cache.py`](../src/features/frame_cache.py) | Embeddingi klatek na wspólnej siatce czasu, cachowane per odcinek |
| [`identity.py`](../src/features/identity.py) | Wektory tożsamości wykrytych twarzy i profile postaci (E6) |
| [`motion.py`](../src/features/motion.py) | Rozpoznawanie czynności: klasyczny komponent ruchu eksperymentu E2 |
| [`objects.py`](../src/features/objects.py) | Detekcja obiektów na siatce klatek: sygnał obiektowy (E4) |
| [`openclip.py`](../src/features/openclip.py) | Enkoder OpenCLIP, wspólny dla sygnału scenicznego i kodowania zapytań |
| [`regions.py`](../src/features/regions.py) | Embeddingi wyrównanych wycinków twarzy: sygnał regionów eksperymentu E5 |
| [`scene.py`](../src/features/scene.py) | Sygnał sceniczny - embedding fragmentu złożony z embeddingów jego klatek |
| [`text_vocab.py`](../src/features/text_vocab.py) | Embeddingi zamkniętych słowników, z którymi sygnał porównuje zapytanie |
| [`xclip.py`](../src/features/xclip.py) | Enkoder X-CLIP: wideo-językowa reprezentacja bazowa eksperymentu E2 |

### `src/indexing/` - Indeksowanie

| Plik | Co robi |
|---|---|
| [`faiss_index.py`](../src/indexing/faiss_index.py) | Indeks wektorowy FAISS i katalog metadanych fragmentów |

### `src/retrieval/` - Obsługa zapytania - sygnały, standaryzacja, ranking

| Plik | Co robi |
|---|---|
| [`fusion.py`](../src/retrieval/fusion.py) | Standaryzacja i ważone łączenie sygnałów komponentów (rozdział 4) |
| [`phrases.py`](../src/retrieval/phrases.py) | Frazy zapytania: strona tekstowa sygnałów zamkniętego słownika (rozdział 4) |
| [`pipeline.py`](../src/retrieval/pipeline.py) | Łączy sygnały komponentów w jeden ranking |
| [`query_detection.py`](../src/retrieval/query_detection.py) | E4-D: detekcja sterowana zapytaniem - przeranżowanie 50 najlepszych fragmentów BAZY (6.5.4) |
| [`signals.py`](../src/retrieval/signals.py) | Sygnały komponentów: każdy ocenia zapytanie wobec każdego fragmentu |
| [`vocab_match.py`](../src/retrieval/vocab_match.py) | Dopasowanie frazy do nazwy klasy zamkniętego słownika (rozdział 4) |

### `src/evaluation/` - Ewaluacja - relewancja, miary, porównania, tabele

| Plik | Co robi |
|---|---|
| [`compare.py`](../src/evaluation/compare.py) | Porównywanie przebiegów: różnice sparowane z przedziałami na właściwej jednostce |
| [`metrics.py`](../src/evaluation/metrics.py) | Miary rankingowe, liczone na poziomie zdarzenia (rozdział 4) |
| [`pool.py`](../src/evaluation/pool.py) | Pula ocen: kontrola kompletności adnotacji na 10% zapytań testowych (6.7.6) |
| [`relevance.py`](../src/evaluation/relevance.py) | Reguła relewancji z rozdziału 4: adnotacja -> zbiór poprawnych fragmentów |
| [`sensitivity.py`](../src/evaluation/sensitivity.py) | Wrażliwość Recall@10 na wagi sygnałów - przeważanie zapisanych macierzy (6.7.5) |
| [`tables.py`](../src/evaluation/tables.py) | Formatowanie tabel wynikowych rozdziału 6 do czytania w notatniku |
| [`figures.py`](../src/evaluation/figures.py) | Skład rysunków pracy: paleta, znaczniki, `usetex`, zapis do `results/figures/` - wspólny dla notatników, które rysują |
| [`examples.py`](../src/evaluation/examples.py) | Przykłady jakościowe: zamrożony wykaz sześciu przypadków, złożenie rankingów z katalogów przebiegów, populacja kandydatów i plik wyboru w `results/reports/examples/` |
| [`thumbnails.py`](../src/evaluation/thumbnails.py) | Klatki fragmentu próbkowane na osi treści, w pamięci podręcznej `data/cache/thumbnails/` - jedyny krok przykładów, który rusza nagranie |
| [`sheets.py`](../src/evaluation/sheets.py) | Rysunki przykładów: siatka kadrów rysowana w milimetrach, jeden plik na konfigurację, głębokość rankingu z `selection.csv` |

### `src/runners/` - Orkiestracja przebiegu

| Plik | Co robi |
|---|---|
| [`components.py`](../src/runners/components.py) | Pamięci podręczne komponentów i budowane z nich sygnały |
| [`run.py`](../src/runners/run.py) | Jeden przebieg: konfiguracja + część materiału -> niezmienny katalog przebiegu |
| [`stages.py`](../src/runners/stages.py) | Etapy przebiegu w trybie cache-or-compute: czerń -> fragmenty -> klatki |

### `src/measurement/` - Pomiar kosztu obliczeniowego

| Plik | Co robi |
|---|---|
| [`cost.py`](../src/measurement/cost.py) | Koszt obliczeniowy komponentów potoku (rozdział 4) |
| [`text_bridge.py`](../src/measurement/text_bridge.py) | Strona tekstowa pomiaru progu: próby do oceny ręcznej, próg z ocen, sprawdzian na listach kontrolnych (rozdział 6.4.3) |

### `src/` - Poziom pakietu

| Plik | Co robi |
|---|---|
| [`vatex_pipeline.py`](../src/vatex_pipeline.py) | Złożenie potoku scenicznego dla VATEX - sterowane konfiguracją |

### `scripts/` - Punkty wejścia z linii poleceń

| Plik | Co robi |
|---|---|
| [`check_gpu.py`](../scripts/check_gpu.py) | Kontrola środowiska: czy potok faktycznie liczy na karcie? |
| [`compare_runs.py`](../scripts/compare_runs.py) | Porównuje warianty eksperymentu na gotowych katalogach przebiegów |
| [`face_size_survey.py`](../scripts/face_size_survey.py) | Rozkład rozmiarów wykrytych twarzy przy obniżonym progu detekcji |
| [`make_threshold_samples.py`](../scripts/make_threshold_samples.py) | Wypisuje cztery próby par fraza-klasa do oceny ręcznej (`data/interim/threshold/`) |
| [`measure_cost.py`](../scripts/measure_cost.py) | Mierzy koszt obliczeniowy komponentów potoku |
| [`measure_e4d_cost.py`](../scripts/measure_e4d_cost.py) | Koszt etapu E4-D, naturalną pętlą, ze składowymi osobno (`cost.e4d`) |
| [`measure_phrase_heads.py`](../scripts/measure_phrase_heads.py) | Diagnostyka reguł fraz: najczęstsze głowy fraz mimicznych i przedmiotowych |
| [`measure_yoloe_prompted.py`](../scripts/measure_yoloe_prompted.py) | Szczyt pamięci i czas na klatkę promptowalnego YOLOE, czyli wiersz E4-D tabeli kosztu |
| [`pool_sensitivity.py`](../scripts/pool_sensitivity.py) | Jak niekompletna musiałaby być adnotacja, żeby zmienić uporządkowanie wkładów |
| [`measure_threshold.py`](../scripts/measure_threshold.py) | Wyznacza próg dopasowania τ i raportuje sprawdzian na listach kontrolnych |
| [`run_blackdetect.py`](../scripts/run_blackdetect.py) | Znajduje przedziały (niemal) czarnego obrazu w każdym odcinku |
| [`run_experiment.py`](../scripts/run_experiment.py) | Uruchamia jeden wariant eksperymentu na jednej części materiału |
| [`run_features.py`](../scripts/run_features.py) | Wylicza cechy komponentów dla podanych konfiguracji |
| [`run_segmentation.py`](../scripts/run_segmentation.py) | Buduje pamięć podręczną fragmentów dla eksperymentu E1 |
| [`vatex_check.py`](../scripts/vatex_check.py) | Kontrola poprawności implementacji na zbiorze VATEX - potok bazowy |
| [`make_configs.py`](../scripts/make_configs.py) | Generuje 57 konfiguracji z [`configs/frozen.yaml`](../configs/frozen.yaml), rodzinami - każdy plik czeka tylko na te decyzje, które faktycznie czyta, więc wkłady indywidualne powstają przed werdyktami E4 i E5. Do plików ręcznych nie pisze nic: sześć konfiguracji E1, dwie `e5bp_*` i [`vatex_base.yaml`](../configs/vatex_base.yaml) zostają nietknięte |
| [`weight_sensitivity.py`](../scripts/weight_sensitivity.py) | Liczy siatkę wag dla trzech zbiorów i zapisuje pomiar `weight_sensitivity` |

### `scripts/prepare_data/` - Pozyskanie i normalizacja materiału

| Plik | Co robi |
|---|---|
| [`check_frame_offset.py`](../scripts/prepare_data/check_frame_offset.py) | Czy czas w adnotatorze i numer klatki w potoku wskazują tę samą klatkę? |
| [`query_tags.py`](../scripts/prepare_data/query_tags.py) | Tworzy i sprawdza plik znaczników wymagań serialu |
| [`vatex_candidates.py`](../scripts/prepare_data/vatex_candidates.py) | Zamrożone losowanie klipów części deweloperskiej ze splitu treningowego |
| [`vatex_download.py`](../scripts/prepare_data/vatex_download.py) | Pobranie i weryfikacja klipów VATEX z YouTube; `--split` wybiera część testową albo deweloperską |
| [`vatex_exclude_corrupt.py`](../scripts/prepare_data/vatex_exclude_corrupt.py) | Przenosi do raportu ręczne decyzje o odrzuceniu klipów |
| [`vatex_leak_filter.py`](../scripts/prepare_data/vatex_leak_filter.py) | Odfiltrowuje przeciek treningowy i dzieli klasy VATEX wg słownika K400; `--split` jak wyżej |
| [`vatex_retry_errors.py`](../scripts/prepare_data/vatex_retry_errors.py) | Przygotowuje raport do ponowienia klipów zakończonych błędem |

### `tools/` - Narzędzia obsługiwane ręcznie

| Plik | Co robi |
|---|---|
| `interval-annotator/` | Przeglądarkowy adnotator przedziałów; własny `README.md` w katalogu |
| `judge/judge_pairs.py` | Ocena ręczna par fraza-klasa w konsoli, jeden klawisz na parę - uruchamia **oceniająca**; `Y` tak, `N` nie, `S` pomiń, `P` cofnij, `Q` zapisz i wyjdź. Wymaga `--samples` i `--output`, bez wartości domyślnych. Stoi na samej bibliotece standardowej, nie importuje `src/` |
| `judge/README.md` | Do czego służy `judge_pairs.py`, jak go uruchomić i reguła rozstrzygania wątpliwych par - do przeczytania przed pierwszym uruchomieniem |

## Konfiguracje, notatniki i testy

| Plik | Rola |
|---|---|
| konfiguracje - **8 ręcznych** (sześć deweloperskich E1 razy dwa seriale oraz `e5bp_{tbbt,office}`) i **57 generowanych** przez [`make_configs.py`](../scripts/make_configs.py) z [`configs/frozen.yaml`](../configs/frozen.yaml), w tym trzy `full_*`. Poza tym [`vatex_base.yaml`](../configs/vatex_base.yaml) (poza schematem `ExperimentConfig` - kontrola poprawności implementacji) i sam [`frozen.yaml`](../configs/frozen.yaml) |
| e1a / e1b / e1c | stałe okna · histogram · TransNetV2 |
| e2a / e2ap / e2b | OC ViT-H/14 · OC ViT-B/32 openai (kontrola) · X-CLIP |
| e3b / e3c | BAZA + BLIP · BAZA + LLaVA-1.5 |
| e4b / e4c | BAZA + YOLO11 · BAZA + YOLOE-11 |
| e5b / e5c | BAZA + regiony twarzy · BAZA + HSEmotion |
| e2a_vatex / e2c_vatex | BAZA · BAZA + SlowFast na VATEX (obie części) |
| full_tbbt / full_office / full_vatex | potok pełny trzech zbiorów; na VATEX-ie bez sygnału twarzy i bez tożsamości |
| skrypty |
| run_features.py | wsadowa ekstrakcja cech; grupuje po komponencie, żeby model ładował się raz |
| prepare_data/query_tags.py | szkielet pliku znaczników i raport pokrycia |
| run_blackdetect.py | cienkie CLI nad `segmentation/black.py` |
| compare_runs.py | kontrast jako różnica różnic, selektor podzbioru, ostrzeżenie o jednym serialu |
| notatniki |
| experiments/dev_results.ipynb | liczby do wszystkich dziesięciu tabel fazy deweloperskiej |
| experiments/face_sizes.ipynb | rozkład rozmiarów twarzy na obu częściach i uzasadnienie progu `MIN_FACE_PX`; rysunek |
| experiments/test_results.ipynb | tabele rozdziału 6 dla każdego eksperymentu na części testowej |
| experiments/test_summary.ipynb | wkłady zbiorczo, aktywacja, PB3, PB4, wrażliwość na wagi, koszt, kontrola kompletności; dwa rysunki |
| experiments/test_appendix.ipynb | wszystkie miary wszystkich konfiguracji, podzbiory, K400, tabela przejść, wykaz materiału |
| experiments/examples.ipynb | rysunki z przykładami działania systemu; czyta gotowe przebiegi, dekoduje tylko klatki |
| experiments/threshold.ipynb | progi dopasowania, sprawdzian list kontrolnych, aktywacja na dev; nagłówek niesie instrukcję oceny ręcznej |
| prepare_data/office_02_annotations.ipynb | rejestr, zakresy korpusu, szkielet znaczników i pliki zapytań; *The Office* nie ma kroku kandydatów, więc jego ścieżka jest o jeden notatnik krótsza niż TBBT |
| prepare_data/tbbt_01 / tbbt_03 | dobór odcinków i rejestr; statystyki adnotacji liczy `tbbt_03`, tam, gdzie powstają pliki, które je niosą |
| testy |
| test_metrics · test_fusion | binarność Recall@K, próg ε, wartość neutralna, wagi `1/|A(q)|` |
| test_signals · test_components | maksimum po przedmiotach, wzór (5), koniunkcja tożsamości, wiązanie z fragmentem |
| test_tags · test_vocabulary | słownik, scalanie bez utraty pracy ręcznej, odrzucanie literówek |
| test_compare · test_relevance · test_pipeline | różnica różnic, bootstrap, suma po `event_id`, sygnały nieaktywne |

Dodatkowo `pytest.ini`: katalogi tymczasowe testów zostają w repozytorium, bo we współdzielonym temp Windows nie pozwala usunąć dowiązania `pytest-current` i sprzątanie wywalało sesję *po* przejściu wszystkich testów.
