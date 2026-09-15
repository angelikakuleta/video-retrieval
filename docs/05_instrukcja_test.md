# Instrukcja krok po kroku - faza testowa

Od zamrożonej konfiguracji do liczb w rozdziale 6. Zakłada, że faza deweloperska jest zamknięta w całości: [04_instrukcja_dev.md](04_instrukcja_dev.md).

```
conda activate wideo
cd C:\video-retrieval
python scripts/check_gpu.py
```

## Bramka: czego nie wolno przed zamrożeniem

Zbiór testowy jest nieosiągalny, dopóki [`configs/frozen.yaml`](../configs/frozen.yaml) nie ma kompletu etapu drugiego - sześciu werdyktów wymienionych w [`src/utils/frozen.py`](../src/utils/frozen.py) jako `STAGE_TWO`: `segmentation`, `scene_embedding`, `caption`, `objects`, `face_regions`, `motion`. Egzekwuje to [`scripts/run_experiment.py`](../scripts/run_experiment.py): przy `--split test` woła `frozen.require(STAGE_TWO, ...)` i odmawia, wymieniając brakujące klucze, zanim dotknie czegokolwiek testowego. Powód jest metodyczny, nie ostrożnościowy: przebieg testowy, którego skład może się jeszcze zmienić, nie jest przebiegiem testowym.

Dwa wyjścia sięgają części testowej wcześniej i tylko one:

- **`scripts/run_features.py --split test`** - ekstrakcja cech. Nie liczy żadnej metryki, więc nie ma czego nagiąć pod wynik, a trwa godzinami i nie ma powodu czekać na werdykty.
- **przegląd rozmiaru twarzy** (`scripts/face_size_survey.py --split test`) - rysunek opisowy, po zamrożeniu `MIN_FACE_PX`, bez Recall.

Poza tymi dwoma nic z części testowej nie jest czytane przed zamrożeniem.

**Od tej chwili pliki konfiguracyjne nie zmieniają się już wcale.** Fazę testową uruchamia się na tych samych, nietkniętych plikach; zmienia się wyłącznie flaga `--split`. Jeśli wynik testowy okaże się rozczarowujący, jest to wynik, a nie powód do strojenia.

## Stan wyjściowy

Adnotacje części testowej muszą być kompletne dla obu seriali - zakresy korpusu, dziury osi treści, pliki zapytań i znaczniki. Drogę do nich opisuje [03_warstwa_adnotacji.md](03_warstwa_adnotacji.md), a kontrolę kompletności znaczników wykonuje krok "Kontrola kompletności znaczników" w [`tbbt_03_annotations.ipynb`](../notebooks/prepare_data/tbbt_03_annotations.ipynb) i [`office_02_annotations.ipynb`](../notebooks/prepare_data/office_02_annotations.ipynb): wypisuje odcinki, którym brakuje któregoś znacznika, i niczego nie blokuje.

## 1. Ekstrakcja cech

```
python scripts/run_features.py configs/full_office.yaml configs/full_tbbt.yaml --split test
python scripts/run_features.py configs/e4c_office.yaml configs/e4c_tbbt.yaml --split test --only objects
python scripts/run_features.py configs/e3b_office.yaml configs/e3b_tbbt.yaml --split test --only captions
python scripts/run_features.py configs/e5c_office.yaml configs/e5c_tbbt.yaml --split test --only expressions
python scripts/run_features.py configs/e3c_vatex.yaml configs/e3b_vatex.yaml configs/e4b_vatex.yaml configs/e4c_vatex.yaml configs/e2c_vatex.yaml --split test
```

Pierwsza linia liczy scenę, YOLO11, wycinki twarzy, regiony, tożsamość i ruch; trzy następne dokładają komponenty wariantów, których potok PEŁNY nie obejmuje. Bufory są kluczowane **modelem**, nie konfiguracją, więc te pięć poleceń zapełnia komplet dla wszystkich wariantów naraz.

`--only` przyjmuje wyłącznie nazwy z listy `objects`, `faces`, `regions`, `expressions`, `identity`, `captions`, `motion` - sceny wśród nich nie ma. **Embeddingi klatek i indeks FAISS powstają dopiero w przebiegu**, w etapach czwartym i piątym, więc pierwszy przebieg każdej kolekcji jest o nie dłuższy: dekodowanie około 2,3 min na godzinę materiału plus 0,93 min/h na enkoder, czyli rzędu pół godziny na zbiór, jednorazowo.

LLaVA-1.5 jest najdroższa (około 62,6 min na godzinę materiału) i wymaga około 14 GB pamięci karty - **zamknij wcześniej wszystko, co ją zajmuje**. Skrypt jest wznawialny odcinek po odcinku.

## 2. BAZA i potok PEŁNY

Idą pierwsze, bo są odniesieniem każdej pozostałej tabeli: bez nich żaden wkład nie ma się do czego odnieść.

```
python scripts/run_experiment.py configs/e2a_office.yaml --split test
python scripts/run_experiment.py configs/e2a_tbbt.yaml  --split test
python scripts/run_experiment.py configs/e2a_vatex.yaml --split test

python scripts/run_experiment.py configs/full_office.yaml --split test
python scripts/run_experiment.py configs/full_tbbt.yaml  --split test
python scripts/run_experiment.py configs/full_vatex.yaml --split test
```

Skład potoku PEŁNEGO dla VATEX jest węższy i wynika z `SIGNAL_DATASETS` w [`src/utils/experiments.py`](../src/utils/experiments.py): odpada tożsamość, bo VATEX nie ma powracających postaci z profilami, i odpada sygnał twarzy, bo plik znaczników tego zbioru nie ma kolumny `wymaga_mimiki`, wobec której dałoby się odczytać wynik.

## 3. Wkłady indywidualne

BAZA plus jeden sygnał. Każdy z nich to jedna kolumna w tab. `wklady-zbiorczo`.

```
python scripts/run_experiment.py configs/e2c_office.yaml --split test   # ruch
python scripts/run_experiment.py configs/e2c_tbbt.yaml   --split test
python scripts/run_experiment.py configs/e2c_vatex.yaml  --split test

python scripts/run_experiment.py configs/e3c_office.yaml --split test   # opisy (werdykt E3)
python scripts/run_experiment.py configs/e3c_tbbt.yaml   --split test
python scripts/run_experiment.py configs/e3c_vatex.yaml  --split test

python scripts/run_experiment.py configs/e4b_office.yaml --split test   # obiekty (werdykt E4)
python scripts/run_experiment.py configs/e4b_tbbt.yaml   --split test
python scripts/run_experiment.py configs/e4b_vatex.yaml  --split test

python scripts/run_experiment.py configs/e5b_office.yaml --split test   # twarze (werdykt E5)
python scripts/run_experiment.py configs/e5b_tbbt.yaml   --split test

python scripts/run_experiment.py configs/e6b_office.yaml --split test   # tozsamosc
python scripts/run_experiment.py configs/e6b_tbbt.yaml   --split test
```

Wariantów przegranych (`e3b_*`, `e4c_*`, `e5c_*`) na teście **nie liczy się jako wkładu** - ich rolę pełni efekt podmiany w kroku 5. Ekstrakcja ich komponentów jest jednak potrzebna, bo z nich korzystają konfiguracje `full_swap_*`.

## 4. Wkład krańcowy

Potok PEŁNY bez jednego sygnału. Trzynaście konfiguracji; VATEX nie ma wariantów bez tożsamości i bez twarzy, bo tych sygnałów w jego potoku nie ma.

```
python scripts/run_experiment.py configs/full_no_caption_office.yaml --split test
python scripts/run_experiment.py configs/full_no_objects_office.yaml --split test
python scripts/run_experiment.py configs/full_no_motion_office.yaml --split test
python scripts/run_experiment.py configs/full_no_face_regions_office.yaml --split test
python scripts/run_experiment.py configs/full_no_identity_office.yaml --split test
```

i analogicznie `_tbbt` (pięć) oraz `_vatex` (trzy: `caption`, `objects`, `motion`).

## 5. Efekt podmiany

Potok PEŁNY z jednym mechanizmem zamienionym na ten, który przegrał. Osiem konfiguracji: opisy i obiekty na trzech zbiorach, twarze tylko na serialach.

```
python scripts/run_experiment.py configs/full_swap_caption_office.yaml --split test
python scripts/run_experiment.py configs/full_swap_objects_office.yaml --split test
python scripts/run_experiment.py configs/full_swap_face_regions_office.yaml --split test
```

i analogicznie `_tbbt` oraz `_vatex` (bez `face_regions`).

## 6. E4-D - detekcja sterowana zapytaniem

Na końcu, bo jest najdroższy w obsłudze zapytania i nie blokuje niczego wcześniejszego.

```
python scripts/run_experiment.py configs/e4d_office.yaml --split test
python scripts/run_experiment.py configs/e4d_tbbt.yaml   --split test
python scripts/run_experiment.py configs/e4d_vatex.yaml  --split test
```

Wariant działa na 50 pierwszych fragmentach rankingu bazowego i dekoduje ich klatki przy każdym zapytaniu, więc jego wiersz w tab. `koszt-wyniki` jest innego rzędu niż pozostałe.

## 7. Pomiary pochodne

**Wrażliwość na wagi** - przelicza zapisane macierze sygnałów, nie dotyka nagrań, więc trwa sekundy:

```
python scripts/weight_sensitivity.py --split test
```

**Koszt komponentów** - cała tab. `koszt-wyniki` poza wierszem E4-D. Trwa około pół godziny i **musi biec sam**, bo LLaVA chce 14,4 GB karty. Procedurę opisuje [krok 10 instrukcji deweloperskiej](04_instrukcja_dev.md); mierzy się na części **testowej**, bo czas zapytania i rozmiar indeksu zależą od wielkości kolekcji.

```
python scripts/measure_cost.py --dataset office --split test
python scripts/measure_cost.py --dataset tbbt  --split test
```

Wiersze X-CLIP i OpenCLIP ViT-B/32 w kolumnie "Indeks" pozostaną puste: indeksy tych wariantów istnieją tylko dla części deweloperskiej, bo na teście nie ma ich przebiegów. Bierze się je osobno, i to bez karty:

```
python scripts/measure_cost.py --dataset office --split dev --skip-extraction
```

**Koszt obliczeniowy wariantu E4-D** - osobny skrypt, **musi biec sam**: to pomiar zegara, więc każde inne obciążenie karty albo dysku go zafałszuje.

```
python scripts/measure_e4d_cost.py --config configs/e2a_office.yaml --split test --limit 5
```

Skrypt przechodzi zapytanie po zapytaniu **pętlą naturalną**, otwierając nagrania osobno dla każdego zapytania. Prawdziwy etap odwraca tę kolejność i płaci za dekodowanie raz na odcinek, więc jest kilkadziesiąt razy szybszy - różnica jest zamierzona i opisana w nagłówku skryptu: wiersz tabeli ma podawać koszt zapytania **zadanego samotnie**.

Dlatego **zacznij od sondy**, jak wyżej. Pięć zapytań powie, ile naprawdę kosztuje jedno, i dopiero z tego wylicz `--limit` na swój budżet czasu. Flaga losuje bez zwracania, z ziarna konfiguracji, i zapisuje pomiar razem z blokiem `sample` (ile zapytań, z ilu, ziarno, na ilu obserwacjach stoi p95).

**Ile zapytań potrzeba.** Mediana trzyma się już przy kilkudziesięciu. P95 nie: powyżej niego leży około 5% obserwacji, więc przy 45 zapytaniach opiera się on na dwóch, a rzetelny wymaga **około 200**. Jeśli budżet na tyle nie starcza, raportuj medianę z podaną liczebnością próby i nie podawaj p95.

Uwaga: flaga `--dataset` jest w tym skrypcie **ignorowana** - zbiór bierze się z `--config`.

Koszty pozostałych komponentów są już zmierzone (`results/measurements/cost_*.json`) i nie trzeba ich powtarzać. Jeśli powtarzasz cały pomiar, obowiązuje ta sama zasada: **zamknij wszystko, co zajmuje kartę**.

**Pula ocen** - kontrola kompletności adnotacji. Buduje się ją z pierwszych dziesięciu wyników głównych konfiguracji testowych dla 10% losowo wybranych zapytań każdego serialu, po czym **następuje przerwa na ocenę ręczną**, a po niej przeliczenie. Blok budowy, jawna komórka "Przerwa: ocena ręczna" i blok przeliczenia stoją w [`notebooks/experiments/test_summary.ipynb`](../notebooks/experiments/test_summary.ipynb). Pulę warto zbudować wcześnie, żeby ocena mogła iść równolegle do reszty przebiegów.

## 8. Liczby do rozdziału 6

```
notebooks/experiments/test_results.ipynb     wyniki E2-E6, kontrasty, wkłady
notebooks/experiments/test_summary.ipynb     wkłady zbiorczo, wrażliwość wag, pula ocen
notebooks/experiments/test_appendix.ipynb    wszystkie miary, podzbiory, przejścia, wykaz materiału
notebooks/experiments/face_sizes.ipynb       rozkład rozmiarów twarzy
notebooks/experiments/cost.ipynb             tabela kosztu
notebooks/experiments/examples.ipynb         rysunki z przykładami działania systemu
```

Wszystkie wykonują się na niekompletnym `results/runs/`: wypisują, czego brakuje, zamiast rzucać wyjątkiem. Można je więc uruchamiać w trakcie napływania przebiegów i patrzeć, jak się wypełniają.

## 9. Przykłady jakościowe

Sześć rysunków z przykładami działania systemu - ranking BAZY obok rankingu wariantu, po trzy klatki na każdy pokazany fragment - składa [`notebooks/experiments/examples.ipynb`](../notebooks/experiments/examples.ipynb).

**Nic się tu nie liczy od nowa.** Rankingi pochodzą z `per_query.jsonl`, zbiory poprawnych fragmentów z `relevance.csv` tego samego przebiegu, przedziały czasowe z pamięci podręcznej segmentacji. Żaden model się nie ładuje, więc rysunek pokazuje dokładnie ten ranking, z którego policzono tabele.

**Wybór zapytań jest ręczny i zapisany.** W `results/reports/examples/selection.csv` stoi jeden wiersz na przykład: `desc_id` wybranego zapytania i kolumna `top`, która ustala głębokość rankingu na rysunku. Klasa przykładu - znacznik, złożoność, wariant, kierunek - zostaje w `CASES` w kodzie. `candidates_<przykład>.csv` niesie całą populację, z której wybrano rankingi.

**Rysunki nie wchodzą pod kontrolę wersji.** `results/figures/examples/` jest w `.gitignore`: to klatki nagrań, a repozytorium jest publiczne. Same klatki dekodują się raz do `data/cache/thumbnails/`, kolejne przebiegi czytają już pamięć podręczną.
