# VATEX (split walidacyjny) - pozyskanie danych

> **Procedura krok po kroku wraz z kontrolami** znajduje się w [`notebooks/prepare_data/vatex_02_test_acquisition.ipynb`](../../notebooks/prepare_data/vatex_02_test_acquisition.ipynb) (sekcje "Skrypty, które trzeba uruchomić wcześniej" i "Procedura krok po kroku"). Ten plik opisuje **same skrypty**: co robią, jakie mają flagi i co zapisują.

**Cel:** cały oficjalny split walidacyjny - 3000 klipów po 10 s, ~5 klipów na każdą z 600 klas akcji. Bez ręcznego wybierania, co jest łatwiejsze do obrony metodologicznej.

## Co skąd

1. **Opisy (adnotacje):** [`vatex_validation_v1.0.json`](../../data/interim/vatex/vatex_validation_v1.0.json) (6,6 MB), pobierany automatycznie ze strony https://eric-xw.github.io/vatex-website/download.html. Każdy rekord zawiera `videoID` (identyfikator YouTube + sekundy startu i końca) oraz po 10 opisów `enCap` i `chCap`; chińskie nie są używane.
2. **"Napisy":** VATEX nie ma napisów dialogowych - rolę opisów pełnią właśnie zdania `enCap`.
3. **Klipy wideo:** pobierane z YouTube po identyfikatorze, tylko wycinek 10 s, do 1080p.
4. **Opcjonalnie cechy I3D** od autorów (3,0 GB) - jako kontrola własnej ekstrakcji; w potoku nieużywane.

## Gdzie to leży

Układ Wariant B rozdziela materiał źródłowy (`data/raw`) od plików pośrednich produkujących adnotację (`data/interim`). Gotowa adnotacja idzie do `data/annotations`.

```
C:\video-retrieval\data\
    raw\vatex\                pobrane segmenty .mp4 (materiał źródłowy, poza repo)
        rejected/          odrzucone: ręcznie po obejrzeniu albo wadliwe technicznie
    interim\vatex\
        vatex_validation_v1.0.json   pobrany plik adnotacji upstream
        vatex_training_v1.0.json     plik adnotacji upstream części deweloperskiej
        vatex_report_test.csv        jeden wiersz na klip (część testowa)
        vatex_report_dev.csv         jeden wiersz na klip (część deweloperska)
        vatex_split_test.csv         przeciek K400 + podział klas (część testowa)
        vatex_split_dev.csv          przeciek K400 + podział klas (część deweloperska)
        work\                        pliki pośrednie pojedynczej procedury
            vatex_descriptions_test.csv  opisy klipów, eksport z jsona
            vatex_descriptions_dev.csv   to samo dla części deweloperskiej
            clips_retried_test.txt            lista ID objętych ponowieniem (kontrola B1)
            clips_too_short_test.csv          videoID;duration, rosnąco (komórka C4)
    annotations\vatex\
        vatex_query_tags.csv         znaczniki wymagań i złożoność - praca ręczna
        vatex_queries_test.jsonl     gotowa adnotacja, zbiór testowy (1 opis/klip)
        vatex_check.jsonl            zapytania kontrolne (10 opisów/klip)
```

Skrypty liczą ścieżki **względem korzenia repozytorium** (`Path(__file__)`), więc działają niezależnie od katalogu roboczego - `cd` nie jest potrzebne. Katalogi `raw/vatex` i `interim/vatex` są tworzone automatycznie przy pierwszym pobieraniu.

---

## Instalacja (Windows, jednorazowo)

```powershell
pip install -U yt-dlp
conda install -c conda-forge "ffmpeg=*=gpl*"        # ffmpeg + ffprobe (wariant GPL, z libx264)
winget install DenoLand.Deno -e --source winget    # środowisko JavaScript dla yt-dlp
```

Po instalacji **nowy PowerShell**, potem kontrola - muszą odpowiedzieć wszystkie cztery:

```powershell
yt-dlp --version ; ffmpeg -version ; ffprobe -version ; deno --version
```

[`vatex_download.py`](vatex_download.py) wypisuje te wersje przy każdym starcie i głośno zgłasza brak Deno. Wersje razem z datą pobrania są częścią opisu pozyskania - bez nich wyniku nie da się odtworzyć.

### Deno to za mało - potrzebny jest jeszcze skrypt solvera

YouTube szyfruje parametry adresów strumieni. Żeby je odszyfrować, yt-dlp musi uruchomić javascript YouTube'a, a do tego potrzebuje **dwóch** rzeczy: środowiska JS (Deno) jako silnika oraz skryptu `yt-dlp-ejs` jako kodu, który ten silnik wykonuje. Oficjalne pliki `.exe` mają solver w środku, **instalacja przez `pip` nie ma** - pobiera go flaga `--remote-components ejs:github`, wpisana na sztywno w [`vatex_download.py`](vatex_download.py) (stała `EJS_ARGS`).

**Bez solvera yt-dlp nie zgłasza błędu.** Po cichu wybiera klienta niewymagającego rozwiązywania wyzwań i zwraca okrojoną listę formatów, czyli materiał o zaniżonej rozdzielczości. Sprawdzone 2026-08-12 na `FNcBtd4lGlM`: z solverem lista sięga 1920x1080@60, bez niego formaty HD znikają.

Nie usuwać tej flagi bez powtórzenia porównania `yt-dlp -F` z nią i bez niej. Szczegóły: https://github.com/yt-dlp/yt-dlp/wiki/EJS

### Ciasteczka - sprawdzone, niepotrzebne do jakości

Porównanie 2026-08-12 na `FNcBtd4lGlM`: warianty z ciasteczkami i bez dają **identyczny sufit 1920x1080@60** i te same formaty HD (299/303/399/301). Ciasteczka dokładają jedynie 720p30 i warianty DRC audio.

Zysk z logowania jest więc wyłącznie na tempie (ok. 2000 filmów/godzinę zamiast ok. 300 dla gościa), a koszt to ryzyko bana konta - wiki yt-dlp: *"you run the risk of it being banned"*. **Domyślnie pobieranie odbywa się bez logowania.**

Flaga `--cookies-from-browser firefox` przydaje się przy statusach `login_required` i `age_restricted`. Wtedy: konto zapasowe, nigdy główne, i zamknięty Firefox na czas przebiegu, bo YouTube rotuje ciasteczka w otwartych kartach.

---

## Skrypty

| Skrypt | Rola |
|---|---|
| [`vatex_download.py`](vatex_download.py) | pobiera klipy i prowadzi raport |
| [`vatex_retry_errors.py`](vatex_retry_errors.py) | przygotowuje raport do ponowienia nieudanych klipów |
| [`vatex_exclude_corrupt.py`](vatex_exclude_corrupt.py) | przepisuje do raportu ręczne decyzje o odrzuceniu klipów |
| [`vatex_leak_filter.py`](vatex_leak_filter.py) | oznacza przeciek klas K400 i zapisuje podział klas do `vatex_split_<część>.csv` |

### [`vatex_download.py`](vatex_download.py)

Trzy tryby: `--mode descriptions` (eksport zapytań do CSV), `--mode download` (właściwa praca), `--mode check` (inwentaryzacja dostępności bez pobierania). Flaga `--split` wybiera część zbioru: `validation` (testowa, domyślna) albo `dev` (deweloperska, ze splitu treningowego). Każda część ma własny plik jsona, własny katalog klipów, własny raport i własny eksport opisów.

Eksport opisów nie jest osobnym krokiem procedury: `--mode download` dorabia go na starcie, gdy pliku danej części nie ma. `--mode descriptions` zostaje do przegenerowania eksportu po zmianie zakresu przebiegu - czyta wyłącznie plik jsona, więc działa lokalnie i w sekundy.

Wykonuje **jedno zapytanie do YouTube na klip** i mierzy pobrany plik jednym wywołaniem ffprobe - rozdzielczość, fps i czas trwania naraz. Wznawia pracę pomijając każdy `videoID` obecny w raporcie, więc można przerywać `Ctrl+C`. Po **10 blokadach pod rząd** (`bot`, `rate_limited`) zatrzymuje się sam, żeby nie przemielić zbioru na pusto.

| Flaga | Znaczenie |
|---|---|
| `--split validation` / `--split dev` | która część zbioru; wybiera plik jsona, katalog klipów, raport i eksport opisów |
| (bez flagi) | zakres przebiegu bierze się z zamrożonej listy części ([`vatex_candidates.py`](vatex_candidates.py)); brak listy zatrzymuje przebieg |
| `--limit N` | przetwórz N **nowych** klipów w tym przebiegu |
| `--pause S` | przerwa między klipami; domyślnie 12 s (limit gościa), z ciasteczkami wystarczy 3 |
| `--cookies-from-browser firefox` | przekazuje `--cookies-from-browser` do yt-dlp |
| `--no-force-keyframes` | pomija `--force-keyframes-at-cuts` (patrz niżej) |
| `--check-source` | dodatkowe zapytanie wypełniające `source_height`/`source_fps`; **podwaja** liczbę zapytań, normalnie zbędne |

### [`vatex_retry_errors.py`](vatex_retry_errors.py)

Nie pobiera niczego. Skoro skrypt pobierający pomija wszystko, co jest już w raporcie, ponowienie polega na **usunięciu odpowiednich wierszy**. Domyślnie tylko pokazuje, co zrobi; przy `--execute` zakłada kopię zapasową z datą i dopiero wtedy zapisuje.

Ponawia statusy przejściowe: `error`, `download_error`, `bot`, `rate_limited`, `no_format`, `login_required`. Pomija trwałe: `missing`, `private`, `geo_blocked`, `members_only`, `corrupt`.

- `--statuses A B C` - zamiast domyślnej listy weź dokładnie te statusy,
- `--include-age-restricted` - dorzuć `age_restricted` (sens tylko z ciasteczkami),
- `--check-files` - ponów też wiersze `ok` bez pliku lub z plikiem poniżej 10 kB; uszkodzone przenosi do `clips/rejected/`. Bez tego przeniesienia ponowienie nic by nie dało, bo skrypt pobierający pomija klip, gdy plik docelowy istnieje.

### [`vatex_exclude_corrupt.py`](vatex_exclude_corrupt.py)

**Katalog jest źródłem prawdy.** Automat nie rozstrzygnie, czy klip nadaje się do zbioru: trzysekundowy wycinek bywa w porządku, jeśli pokazuje dokładnie to, co opisuje adnotacja, a dziewięciosekundowy bywa bezużyteczny, jeśli zdarzenie wypadło poza kadr. Dlatego decyzję podejmuje człowiek, przenosząc plik do `clips/rejected/`, a skrypt tylko przepisuje ten stan do raportu - nadaje status `corrupt`, niezależnie od powodu odrzucenia.

Działa **w obie strony**: plik w kwarantannie dostaje status `corrupt`, a plik, który wrócił do `clips\`, wraca na `ok`. Można go więc uruchamiać wielokrotnie, po każdej partii przeglądu - zawsze dosuwa raport do stanu katalogów. `--no-restore` wyłącza cofanie; `--dir NAZWA` zmienia nazwę podkatalogu kwarantanny (domyślnie `rejected`), `--report` i `--clips` - ścieżki raportu i klipów.

Zgłasza pliki w kwarantannie, które nie mają wiersza w raporcie (zwykle literówka w nazwie), i pomija pliki inne niż `.mp4`, żeby dało się trzymać tam własne notatki.

### [`vatex_leak_filter.py`](vatex_leak_filter.py)

Nie zmienia raportu i nie przenosi plików. Dla klipów `ok` sprawdza, czy nagranie należy do zbioru treningowego Kinetics-400 (przeciek) i czy jego klasa K600 mieści się w słowniku K400. Zapisuje plik podziału z kolumnami `videoID;class_k600;leak_k400;class_in_k400_vocab`; wartości `leak_k400` i `class_in_k400_vocab` to `yes`/`no`. Split testowy to wiersze z `leak_k400 = no`.

Flaga `--split` wybiera część tak samo jak w skrypcie pobierania: `validation` czyta [`vatex_report_test.csv`](../../data/interim/vatex/vatex_report_test.csv) i zapisuje [`vatex_split_test.csv`](../../data/interim/vatex/vatex_split_test.csv), `dev` czyta [`vatex_report_dev.csv`](../../data/interim/vatex/vatex_report_dev.csv) i zapisuje [`vatex_split_dev.csv`](../../data/interim/vatex/vatex_split_dev.csv).

Pozostałe flagi: `--report`, `--k400`, `--k600`, `--output` (ścieżki wejść i wyjścia; nadpisują wybór `--split`, domyślnie wszystko w `data/interim/vatex/`), `--dry-run` (tylko liczby, bez pliku).

---

## Kolejność uruchomień

```powershell
cd C:\video-retrieval                 # korzeń repo; ścieżki danych skrypt liczy sam
$PY = "C:\Users\PC\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$S  = "C:\video-retrieval\scripts\prepare_data\vatex_download.py"
$R  = "C:\video-retrieval\scripts\prepare_data\vatex_retry_errors.py"

& $PY $S --split validation --mode descriptions         # -> data\interim\vatex\work\vatex_descriptions_test.csv
& $PY $S --split validation --mode download --limit 20  # próba kontrolna
& $PY $S --split validation --mode download             # pełny przebieg, ok. 13 h
```

Podgląd postępu w drugim oknie (skrypt robi `flush()` po każdym wierszu):

```powershell
Get-Content C:\video-retrieval\data\interim\vatex\vatex_report_test.csv -Wait -Tail 20
```

Ponawianie, wykluczanie i kontrole wokół nich - patrz procedura w [`notebooks/prepare_data/vatex_02_test_acquisition.ipynb`](../../notebooks/prepare_data/vatex_02_test_acquisition.ipynb). Kolejności nie warto skracać: kontrole **B1** i **B2** są warunkiem, pod którym wolno w ogóle użyć `--no-force-keyframes`.

Uśpienie komputera zabije wielogodzinny przebieg. Przed startem ustawić plan zasilania i sprawdzić, czy Windows Update nie ma zaplanowanego restartu.

---

## [`vatex_report_test.csv`](../../data/interim/vatex/vatex_report_test.csv)

Kolumny: `videoID`, `status`, `source_height`, `source_fps`, `file_width`, `file_height`, `file_fps`, `duration`, `message`. Czasy w sekundach (bez przyrostka w nazwie kolumny).

Liczby zapisywane są z **kropką dziesiętną** - to plik danych, nie tekst pracy. Przecinek dziesiętny obowiązuje dopiero w treści dyplomu.

`source_height` i `source_fps` wypełnia tylko tryb `check` albo flaga `--check-source`; normalnie zostają puste, bo faktyczne parametry mierzy ffprobe na pobranym pliku.

| Status | Znaczenie | Ponawiać? |
|---|---|---|
| `ok` | klip pobrany | - |
| `missing` | usunięty lub niedostępny na YouTube | nie |
| `private` | ustawiony jako prywatny | nie |
| `geo_blocked` | zablokowany w tym kraju | nie z tego łącza |
| `members_only` | tylko dla członków kanału | nie |
| `corrupt` | odrzucony ręcznie po obejrzeniu (dowolny powód) | nie |
| `no_format` | film istnieje, ale yt-dlp nie zobaczył formatów | **tak** |
| `login_required` | "Please sign in" | tak, z ciasteczkami |
| `age_restricted` | wymaga potwierdzenia wieku | tak, z ciasteczkami |
| `bot` | "Sign in to confirm you're not a bot" | tak |
| `rate_limited` | HTTP 429 | tak, po przerwie |
| `download_error` | dostępny, ale pobranie segmentu padło | tak |
| `error` | nierozpoznany błąd przy sprawdzaniu | tak |

> **`no_format` ≠ `missing`.** Komunikat "Requested format is not available" zawiera frazę "not available", ale **nie** znaczy, że film zniknął - to najczęściej brak solvera EJS. Masowe `no_format` oznacza problem z instalacją, nie ze zbiorem.

---

## Awarie ffmpeg i `--no-force-keyframes`

W pierwszym pełnym przebiegu 183 klipy padły z komunikatem `ffmpeg exited with code 3436169992`. Awarie kumulowały się na wycinkach zaczynających się głębiej w filmie (mediana startu 76 s wobec 22 s dla klipów pobranych; udział wycinków od zerowej sekundy: 2,7% wobec 13,4%), czyli tam, gdzie `--force-keyframes-at-cuts` przekodowuje materiał, żeby wstawić klatkę kluczową na granicy.

Flaga `--no-force-keyframes` je odzyskuje, ale rozluźnia precyzję cięcia: ffmpeg tnie wtedy do najbliższej istniejącej klatki kluczowej, więc okno może przesunąć się względem adnotacji. Byłby to błąd **systematyczny**, uderzający w eksperyment z lokalizacją czasową akcji - a taki błąd nie wygląda jak szum, tylko jak wynik.

Dlatego kompromis wolno przyjąć **tylko wtedy, gdy da się go zmierzyć**: komórka **B1** notatnika zapisuje listę ponawianych identyfikatorów przed przebiegiem, **B2** porównuje po nim długości odzyskanych klipów z resztą zbioru.

Pomiar z 2026-08-13: mediana odzyskanych 10,07 s wobec 10,00 s w reszcie, odsetek odstających 3,2% wobec 8,5%. Kompromis nic nie kosztował, klipy zostały w zbiorze.

---

## Wynik pozyskania (2026-08-13)

| | Klipów | Udział |
|---|---:|---:|
| split walidacyjny (nominalnie) | 3000 | 100,0% |
| pozyskane | 2575 | 85,8% |
| niepozyskane | 425 | 14,2% |

Rozbicie niepozyskanych na przyczyny (trwałe wobec technicznych) daje komórka A2 notatnika - tam też widać, czy coś jeszcze nadaje się do ponowienia.

Spośród pozyskanych 270 klipów (10,5%) jest krótszych niż 9,5 s - nagranie źródłowe kończy się przed zamknięciem dziesięciosekundowego okna z adnotacji. **Żaden klip nie jest dłuższy niż 10,5 s**, co dowodzi, że samo cięcie jest precyzyjne, a całe odchylenie to jednostronne obcięcie po stronie materiału.

### Odrzucanie klipów - decyzja należy do człowieka

Krótszy nie znaczy gorszy. Trzysekundowy wycinek bywa poprawny, jeśli pokazuje opisane zdarzenie; dziewięciosekundowy bywa bezużyteczny, jeśli zdarzenie z niego wypadło. Próg automatyczny nie jest więc stosowany - jest przegląd:

1. komórka **C4** notatnika zapisuje [`clips_too_short_test.csv`](../../data/interim/vatex/work/clips_too_short_test.csv) (`videoID;duration`, rosnąco),
2. przejrzeć te klipy i przenieść odrzucone do `clips/rejected/`,
3. `vatex_exclude_corrupt.py --execute` przepisuje decyzje do raportu.

Kolejność listy - od najkrótszych - jest po to, żeby zacząć od przypadków najbardziej podejrzanych; nie jest kryterium sama w sobie.

Liczba odrzuconych i kryterium przeglądu muszą zostać odnotowane: wykluczenie opisane to metodologia, wykluczenie przemilczane to dziura, którą ktoś znajdzie.

---

## Uwagi

1. Ubytki są normalne, bo klipy znikają z YouTube. Liczba pobranych odnotowywana jest **z datą pobrania**; to standardowa praktyka przy zbiorach opartych na tym serwisie (Kinetics ma ten sam problem).
2. **VATEX zbudowano na filmach z Kinetics**, czyli na starych nagraniach - spora część zbioru ma sufit 480p niezależnie od konfiguracji pobierania. Warto to zmierzyć (kolumna `file_height`) i opisać jako ograniczenie materiału: przy 480p komponent rozpoznawania twarzy dostaje regiony rzędu kilkudziesięciu pikseli.
3. Identyfikator YouTube może zawierać `_` i `-`, dlatego `videoID` parsowany jest od prawej (`rsplit`) - nie zmieniać.
4. Klasyfikacja statusów zależy od **kolejności warunków** w `classify_error()`. Trzy pułapki: "Sign in to confirm your **age**" i "Sign in to confirm you're **not a bot**" zaczynają się identycznie; "Requested format is **not available**" dzieli frazę z komunikatami o usuniętych filmach; "Please sign in" trzeba dopasowywać dokładnie, bo komunikat o filmie prywatnym też zawiera "Sign in". Przy modyfikacji zachować kolejność od szczegółu do ogółu.
5. Zapytanie testowe do wyszukiwania to **pierwszy opis** `enCap` klipu (`desc_no = 1` w `work/vatex_descriptions_test.csv`) - to konwencja przyjęta w metodologii. Pozostałych dziewięć służy analizie wrażliwości na sformułowanie zapytania, a w kontroli poprawności implementacji wykorzystywane jest wszystkie dziesięć, zgodnie z protokołem prac referencyjnych.
6. Pobrany materiał przetwarzany jest lokalnie, bez redystrybucji.
