# Warstwa adnotacji

Droga od surowych opisów do plików zapytań. Każdy krok ma jednego właściciela: kod pisze wyłącznie do katalogów, które sam wytwarza, a do plików przeglądanych ręcznie nie pisze nigdy.

Wersje generowane leżą obok tych, które wypełniasz, pod inną nazwą - `candidates/` dla plików adnotatora, przyrostek `_new` dla pliku znaczników. Przenosisz je ręcznie, gdy uznasz za gotowe. Dzięki temu żaden ponowny przebieg nie może skasować pracy ręcznej i nie ma potrzeby pilnowania przełączników: zniknęły `OVERWRITE`, `CARRY_OVER`, `SHIFT_PROPAGATION` i `REJECT_OUT_OF_RANGE`.

## Dziewięć kroków - *The Big Bang Theory*

Kolejność wierszy jest kolejnością uruchamiania: notatniki idą od góry do dołu i nie wraca się do wcześniejszej komórki.

| # | Kto | Notatnik · krok | Co powstaje | Z czego |
|---|---|---|---|---|
| 1 | kod | `tbbt_02` · 2. Przesunięcia klipów | `work/tbbt_clip_mapping.csv` | opisy TVR + napisy klipów i odcinka |
| 2 | kod | `tbbt_02` · 3. Rejestr zbiorczy adnotacji | [`tbbt_annotations.csv`](../data/interim/tbbt/tbbt_annotations.csv) | opisy `v` + przesunięcia z kroku 1 |
| 3 | kod | `tbbt_02` · 4. Pliki dla adnotatora | `candidates/*_intervals_off.csv`<br>`candidates/*_intervals_forward.csv`<br>`work/tbbt_fitting_<wariant>.csv` | rejestr + maski z `masks/` |
| 4 | **ręcznie** | adnotator | `tbbt_<odcinek>_intervals.csv` | wybrany wariant, przejrzany w adnotatorze |
| 5 | kod | `tbbt_03` · 2. Aktualizacja rejestru | [`tbbt_annotations.csv`](../data/interim/tbbt/tbbt_annotations.csv) z naniesionymi decyzjami | zweryfikowane `_intervals.csv` |
| 6 | kod | `tbbt_03` · 3. Zakresy korpusu i dziury | [`tbbt_ranges.csv`](../data/interim/tbbt/tbbt_ranges.csv), [`tbbt_holes.csv`](../data/interim/tbbt/tbbt_holes.csv) | maski ze zweryfikowanych `_intervals.csv` |
| 7 | kod | `tbbt_03` · 4. Znaczniki zapytań | [`tbbt_query_tags_new.csv`](../data/annotations/tbbt/tbbt_query_tags_new.csv) | rejestr + istniejący plik znaczników |
| 8 | **ręcznie** | arkusz | [`tbbt_query_tags.csv`](../data/annotations/tbbt/tbbt_query_tags.csv) | uzupełnienie znaczników, złożoności, postaci i literówek |
| 9 | kod | `tbbt_03` · 6. Pliki zapytań | `tbbt_queries_<split>.jsonl` | czasy z `_intervals.csv`, treść i znaczniki z pliku znaczników; **tylko odcinki zweryfikowane** |

Kroki 5-7 to jeden przebieg `tbbt_03` od góry. Po wypełnieniu znaczników (krok 8) uruchamiasz notatnik **jeszcze raz**: krok 7 przeniesie Twoje komórki do świeżego szkieletu, a krok 9 zapisze pliki zapytań już z nimi. Pierwszy przebieg zostawia `.jsonl` bez znaczników, co niczego nie blokuje - nie policzą się tylko tabele kontrastów.

## Siedem kroków - *The Office*

Odpada kotwiczenie i wybór wariantu dopasowania: adnotacje powstają od zera w adnotatorze, razem z maskami, więc wszystko mieści się w jednym notatniku.

| # | Kto | Notatnik · krok | Co powstaje | Z czego |
|---|---|---|---|---|
| 1 | **ręcznie** | adnotator | `office_<odcinek>_intervals.csv` | adnotowanie od zera, razem z maskami |
| 2 | kod | `office_02` · 1. Pomiar plików wideo | `work/office_video.csv` | pliki `.mp4` |
| 3 | kod | `office_02` · 3. Aktualizacja rejestru | [`office_annotations.csv`](../data/interim/office/office_annotations.csv) | zweryfikowane `_intervals.csv` |
| 4 | kod | `office_02` · 4. Zakresy korpusu i dziury | [`office_ranges.csv`](../data/interim/office/office_ranges.csv), [`office_holes.csv`](../data/interim/office/office_holes.csv) | maski ze zweryfikowanych `_intervals.csv` |
| 5 | kod | `office_02` · 5. Znaczniki zapytań | [`office_query_tags_new.csv`](../data/annotations/office/office_query_tags_new.csv) | rejestr + istniejący plik znaczników |
| 6 | **ręcznie** | arkusz | [`office_query_tags.csv`](../data/annotations/office/office_query_tags.csv) | uzupełnienie znaczników, złożoności, postaci i literówek |
| 7 | kod | `office_02` · 7. Pliki zapytań | `office_queries_<split>.jsonl` | czasy z `_intervals.csv`, treść i znaczniki z pliku znaczników; **tylko odcinki zweryfikowane** |

Tak samo jak w TBBT: kroki 2-5 to jeden przebieg notatnika, a po wypełnieniu znaczników uruchamiasz go ponownie, żeby krok 7 wziął je do plików zapytań.

Kroki, których w tabelach nie ma, bo niczego nie zapisują: `tbbt_03` · 1 i `office_02` · 2 wczytują pliki adnotatora i sprawdzają je, wypisując odcinki jeszcze niezweryfikowane; `tbbt_03` · 5 i `office_02` · 6 raportują kompletność znaczników, a dalsze kroki obu notatników liczą statystyki.

## Trzy kroki - VATEX

VATEX nie ma odcinków, masek ani rejestru: klip jest gotowym zdarzeniem, a opis przychodzi z pliku upstream. Zostaje sama warstwa znaczników.

| # | Kto | Notatnik · krok | Co powstaje | Z czego |
|---|---|---|---|---|
| 1 | kod | `vatex_03_test_annotations` · 1. Znaczniki zapytań | [`vatex_query_tags_new.csv`](../data/annotations/vatex/vatex_query_tags_new.csv) | opisy klipów + istniejący plik znaczników |
| 2 | **ręcznie** | arkusz | [`vatex_query_tags.csv`](../data/annotations/vatex/vatex_query_tags.csv) | uzupełnienie trzech znaczników, złożoności i literówek |
| 3 | kod | `vatex_03_test_annotations` · 2. Plik zapytań | `vatex_queries_test.jsonl` | opisy klipów + znaczniki z pliku znaczników |

Kolumn jest mniej niż w serialach - `wymaga_obiektu`, `wymaga_scenerii`, `wymaga_ruchu` i `complexity` - bo tylko tyle rozstrzyga jednozdaniowy opis klipu; szerzej w [`docs/02_dane_i_znaczniki.md`](02_dane_i_znaczniki.md). Plik kontrolny `vatex_check.jsonl` do tej warstwy nie należy: powstaje w [`notebooks/vatex_check.ipynb`](../notebooks/vatex_check.ipynb), razem z testem, któremu służy.

## Rejestr trzyma czasy oryginalne

Krok 2 zapisuje czasy wprost z kotwiczenia, bez dopasowania do masek: dopasowanie należy do kroku 3 i różni się między wariantami, więc rejestr nie miałby dla niego jednej wartości. Kolumny `source_start` i `source_end` zapamiętują je na stałe, więc po kroku 5 różnica względem `start` i `end` pokazuje wszystko, co stało się z adnotacją po drodze. Flagi dopasowania (`shifted`, `trimmed`, `shift_failed`, `spans_mask`) opisują krok 3, więc lądują w plikach `work/`, nie w rejestrze.

## Słownik kolumn werdyktu

| Kolumna | Wartości | Znaczenie |
|---|---|---|
| `status` | `accepted`, `rejected` | czy adnotacja wchodzi do korpusu |
| `reason` | puste, `too_short`, `too_long`, `manual` | tylko przy `rejected`. Progi z `MIN_DURATION` i `MAX_DURATION`; `manual` = usunięte w adnotatorze |
| `origin` | `tvr`, `manual` | dla *The Office* zawsze `manual` |
| `edited_duration` | `yes` / `no` | zmieniona długość zdarzenia. Werdykt liczony wtedy od nowa w obie strony: odrzucona wraca po naprawieniu, zaakceptowana wypada po rozciągnięciu |
| `edited_desc` | `yes` / `no` | poprawiona treść, liczona osobno od długości |
| `flags` | puste, `uncertain_mapping` | ostrzeżenie o miejscu, w którym nagranie różni się od źródła TVQA |
| `tags` | etykiety z adnotatora | plik `_intervals.csv` jest ich źródłem prawdy |

Samo przesunięcie granic bez zmiany długości nie stawia żadnej flagi, bo nie może zmienić werdyktu. `in_mask` nie jest już werdyktem: adnotację leżącą w całości w masce krok 5 wypisuje jako ostrzeżenie do usunięcia w adnotatorze (w dzisiejszym rejestrze dev jest jedna na 462).

## Dwa pliki opisujące korpus

Krok 6 tabeli TBBT (4 dla *The Office*) zapisuje oba z tych samych zweryfikowanych plików, rozdzielając maski na dwie klasy. `<zbiór>_ranges.csv` to zakresy korpusu, czyli dopełnienie masek **strukturalnych** (logo, czołówka, napisy). `<zbiór>_holes.csv` to maski **przejścia**, które niczego nie tną - stają się dziurami osi treści, więc zakres biegnie przez nie, a ich sekundy przestają się liczyć. Oba czyta `src/data/ranges.py::load_timelines` i nic poniżej nie odtwarza ich z masek na własną rękę.

## Wspólne parametry

[`src/utils/settings.py`](../src/utils/settings.py) zbiera wartości, które powtarzały się w kilku notatnikach albo są ze sobą związane liczbowo: `MIN_DURATION` i `MAX_DURATION`, `PROFILED_CHARACTERS` (po pięć postaci na serial, po imieniu), `EPISODE_COUNT`, `DEV_COUNT`, `SPLITS` oraz czwórkę `FRAME_STEP` = `MIN_DURATION`, `MIN_LEN` = `MIN_RANGE`, `MAX_LEN` = `MAX_DURATION` z zapisanym uzasadnieniem, dlaczego są sobie równe. Progi detektorów, parametry modeli i ziarno zostają tam, gdzie były - mają jednego właściciela i nikt ich nie powiela.
