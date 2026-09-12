# Dane i znaczniki

## 2.1 Wejście, którego przebieg nie wytworzy

Skrócona wersja tego wykazu, do zajrzenia w trakcie pracy, jest w [`notebooks/README.md`](../notebooks/README.md):

| Plik | Skąd |
|---|---|
| data/processed/<zbiór>/*.mp4 | normalizacja materiału ([`normalize-video.ps1`](../scripts/prepare_data/normalize-video.ps1)) |
| .../<zbiór>_<odcinek>_intervals.csv | adnotator, ręcznie |
| .../<zbiór>_query_tags.csv | ręcznie, szkielet z `tbbt_03` / [`office_02_annotations.ipynb`](../notebooks/prepare_data/office_02_annotations.ipynb) / [`vatex_03_test_annotations.ipynb`](../notebooks/prepare_data/vatex_03_test_annotations.ipynb) (VATEX bez `wymaga_osoby`, `wymaga_mimiki` i postaci - opis klipu ich nie rozstrzyga) |
| .../<zbiór>_queries_<split>.jsonl | `tbbt_03` / [`office_02_annotations.ipynb`](../notebooks/prepare_data/office_02_annotations.ipynb) / [`vatex_03_test_annotations.ipynb`](../notebooks/prepare_data/vatex_03_test_annotations.ipynb) |
| data/interim/<zbiór>/<zbiór>_ranges.csv | krok "Zakresy korpusu i dziury osi treści" w [`tbbt_03_annotations.ipynb`](../notebooks/prepare_data/tbbt_03_annotations.ipynb) / [`office_02_annotations.ipynb`](../notebooks/prepare_data/office_02_annotations.ipynb) (VATEX nie ma tego pliku - klip jest całym fragmentem, więc `load_ranges("vatex")` składa zakresy w pamięci z plików zapytań obu części) |
| data/interim/<zbiór>/<zbiór>_holes.csv | ten sam krok tych samych notatników - dziury osi treści, czyli maski przejścia i przedziały czarnego obrazu wycięte z zakresów; VATEX nie ma tego pliku |
| data/annotations/threshold_judgments.csv | ręcznie, [`tools/judge/judge_pairs.py`](../tools/judge/judge_pairs.py) na próbach z `data/interim/threshold/` |

### Dlaczego pliki progu leżą poza katalogiem zbioru

`data/interim/threshold/` i [`data/annotations/threshold_judgments.csv`](../data/annotations/threshold_judgments.csv) łamią konwencję `data/interim/<zbiór>/` i `data/annotations/<zbiór>/`, i jest to świadome. Konwencja zakłada, że artefakt należy do zbioru - a **próg nie należy do żadnego**: jest własnością mostu między zapytaniem a nazwami klas i obowiązuje we wszystkich trzech kolekcjach naraz. Widać to w samym materiale: próba `expressions_hsemotion` ma 50 par, z czego 21 pochodzi z opisów VATEX-dev, 9 z zapytań deweloperskich TBBT i 20 z The Office. Kolumna `source` niesie to pochodzenie wprost, więc ścieżka `data/interim/vatex/` kłamałaby o dwóch piątych pliku.

## 2.2 Znaczniki wymagań

Nazwy znaczników i etykiet złożoności definiuje [`src/utils/vocabulary.py`](../src/utils/vocabulary.py) - jedno miejsce dla adnotatora, rejestru, pliku znaczników i rekordów zapytań. Literówka w którymkolwiek z nich nie opróżni po cichu podzbioru, tylko zostanie wypisana jako etykieta nieznana.

```
class RequirementTag(StrEnum):
    PERSON     = "wymaga_osoby"      # E6
    OBJECT     = "wymaga_obiektu"    # E4
    SCENERY    = "wymaga_scenerii"   # E1
    MOTION     = "wymaga_ruchu"      # E2
    EXPRESSION = "wymaga_mimiki"     # E2, E5

class Complexity(StrEnum):
    SIMPLE  = "P"
    COMPLEX = "Z"
```

### Dwa pliki, jedna droga

Znaczniki żyją w dwóch miejscach i przechodzą z jednego do drugiego w ustalonym kierunku:

- `<zbiór>_<odcinek>_intervals.csv` - kolumna `tags` na wierszu zdarzenia. Cokolwiek wpiszesz tam w adnotatorze, wędruje przez rejestr do pliku znaczników. Potrzebuje tego przede wszystkim `wymaga_osoby`, którego kryterium porównuje zapytanie z innymi zdarzeniami tego samego odcinka, ale mechanizm jest ogólny i przyjmie każdy znacznik ze słownika.
- `<zbiór>_query_tags.csv` - jeden wiersz na zaakceptowaną adnotację, **wstępnie wypełniony** tym, co przyszło z `_intervals`. Tu uzupełniasz resztę ręcznie i tu poprawiasz literówki w treści zapytania. **To ten plik trafia do `.jsonl`** - zarówno znaczniki, złożoność i lista postaci, jak i treść.

Plik znaczników jest osobnym wejściem ręcznym, nie polem w gotowym JSONL: pliki zapytań powstają od nowa przy każdym przebiegu notatnika, więc cokolwiek wpisanego wprost w nie zostałoby skasowane. Notatnik zostawia obok `<zbiór>_query_tags_new.csv` z przeniesionymi komórkami, które już masz; kopiujesz go pod nazwę bez przyrostka, kiedy uznasz za gotowy. Do `<zbiór>_query_tags.csv` kod nie pisze nigdy.

Reguła wciągania z adnotatora jest per znacznik i per odcinek: odcinek, w którym znacznik pada choć raz, liczy się jako przejrzany pod jego kątem i pozostałe jego zapytania dostają `0`; odcinek, w którym nie pada ani razu, zostaje pusty - brak przejrzenia nie może wyglądać jak przejrzenie bez trafień.

Pusty plik znaczników niczego nie blokuje. Wszystkie rozstrzygnięcia E1-E5 policzą się bez nich; nie policzą się tylko tab. `kontrast-e1` i `e2-ruch-dev`.

### VATEX ma węższy zestaw kolumn

[`vatex_query_tags.csv`](../data/annotations/vatex/vatex_query_tags.csv) niesie tylko `wymaga_obiektu`, `wymaga_scenerii`, `wymaga_ruchu` i `complexity` - tyle, ile da się rozstrzygnąć z samego jednozdaniowego opisu klipu. `wymaga_osoby` trzeba by czytać na tle pozostałych zdarzeń tego samego nagrania, a klipy VATEX są od siebie niezależne; `wymaga_mimiki` wymaga informacji o wyrazie twarzy, której opisy nie zawierają; `identities` nie ma do czego się odnieść, bo VATEX nie ma powracających postaci z profilami. **Brak kolumny znaczy "tego się tutaj nie przypisuje", a nie `0`** - podzbiory `wymaga_osoby` i `wymaga_mimiki` na VATEX są puste i kontrast na nich się nie liczy. Oba zestawy kolumn definiuje [`src/annotation/tags.py`](../src/annotation/tags.py) (`COLUMNS`, `VATEX_COLUMNS`, `columns_for`).

Plik jest kluczowany po `desc_id`, więc identyfikator zapytania VATEX pochodzi z nazwy klipu (`src/utils/vatex.py::desc_id_for`), a nie z pozycji na liście: przy numerowaniu po pozycji wypadnięcie jednego klipu przesunęłoby całą numerację i ręcznie wpisane znaczniki trafiłyby po cichu do innych zapytań. Notatnik dodatkowo porównuje kolumnę `episode` z klipem, na który dany `desc_id` wskazuje teraz, i przy rozjeździe przerywa.

## 2.3 Profile postaci

Profil nie jest zbiorem obrazków w repozytorium, tylko listą **wskaźników** do twarzy, które korpus już zawiera: postać, odcinek, czas i który to z wykrytych w tej klatce. Wynikają z tego dwie rzeczy: przykład przechodzi dokładnie tę samą detekcję i wyrównanie co reszta materiału, czego wymaga protokół, a plik jest kilkoma kilobajtami tekstu, który da się wersjonować - sam materiał zostaje poza repozytorium. Wiersz wskazujący odcinek testowy jest odrzucany, zamiast liczyć na to, że nikt się nie pomyli.
