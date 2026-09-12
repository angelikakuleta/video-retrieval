# Zgrywanie płyt (TBBT, The Office) pod eksperymenty video-retrieval

Skrypt [`rip-disc.ps1`](rip-disc.ps1) jest wspólny dla wszystkich serii - folder docelowy podaje się parametrem `-Output` (np. `data\raw\tbbt`, `data\raw\office`).

Wydania:
- *The Big Bang Theory: The Complete Series*, EAN 5051892226585 (25 płyt: 24x BD-50 + 1 DVD; 1080p, 1.78:1, ~279 odcinków).
- *The Office* - płyty z wydania serii.

## Czy da się zgrać pojedynczy odcinek?

Tak. Na Blu-rayu każdy odcinek to osobny **title** (playlista). MakeMKV wyświetla listę wszystkich tytułów z czasem trwania - zaznacza się tylko te potrzebne. Nie ma potrzeby zgrywania całej płyty (jeden odcinek ≈ 3-5 GB bezstratnie, cała seria to ~1 TB - więc zgrywanie wybiórcze jest zdecydowanie sensowniejsze).

## Co doinstalować (Windows)

1. **MakeMKV** - dekoduje zabezpieczenie AACS i pozwala wybrać pojedyncze tytuły.
   - Pobranie: https://www.makemkv.com/download/ -> instalator Windows.
   - Zawiera `makemkvcon64.exe` (wersja CLI używana przez skrypt), zwykle w `C:\Program Files (x86)\MakeMKV\`.
   - Program w fazie beta jest darmowy z kluczem odświeżanym co miesiąc (forum MakeMKV -> aktualny klucz beta -> wkleić w *Help -> Register*).
   - **Napęd Blu-ray**: potrzebny zewnętrzny/wewnętrzny czytnik BD (zwykły napęd DVD nie odczyta płyt).

2. **FFmpeg** - do normalizacji materiału pod eksperymenty.
   - FFmpeg w wariancie GPL, ze środowiska: `conda install -c conda-forge "ffmpeg=*=gpl*"`. Jest wtedy w `PATH` tylko przy aktywnym środowisku, więc skrypty normalizujące uruchamiaj po `conda activate wideo`.

3. **Subtitle Edit** - OCR obrazkowych napisów PGS z płyty do `.srt`.
   - Pobranie: https://www.nikse.dk/subtitleedit (repozytorium: https://github.com/SubtitleEdit/subtitleedit).
   - Napisy TBBT są potrzebne, a nie opcjonalne: z nich liczone są przesunięcia klipów TVR w [`tbbt_02_annotation_candidates.ipynb`](../../notebooks/prepare_data/tbbt_02_annotation_candidates.ipynb). Dla "The Office" są nieużywane.

## Cała ścieżka

```powershell
cd C:\video-retrieval\scripts\prepare_data

.\rip-disc.ps1 -Action list

# jeden odcinek na wywołanie, osobno dla każdej serii
.\rip-disc.ps1 -Output C:\video-retrieval\data\raw\office -Title 23 -Season 5 -Episode 10 -Prefix office
.\rip-disc.ps1 -Output C:\video-retrieval\data\raw\office -Title 23 -Season 5 -Episode 17 -Prefix office
.\rip-disc.ps1 -Output C:\video-retrieval\data\raw\tbbt   -Title 4  -Season 1 -Episode 15 -Prefix tbbt

# normalizacja przyrostowa - gotowe pliki są pomijane
.\normalize-video.ps1 -InputPath C:\video-retrieval\data\raw\office -OutputPath C:\video-retrieval\data\processed\office
.\normalize-video.ps1 -InputPath C:\video-retrieval\data\raw\tbbt   -OutputPath C:\video-retrieval\data\processed\tbbt

# na końcu OCR napisów TBBT w Subtitle Edit: .mkv -> .srt (krok 5)
```

Skrypty nazywały się wcześniej `rip-tbbt.ps1` i `normalize-for-experiments.ps1` i leżały w `scripts/`; ten drugi miał domyślne ścieżki i uruchamiało się go bez parametrów. Teraz `-InputPath` i `-OutputPath` są obowiązkowe.

## Krok po kroku

1. Włożyć płytę, zainstalować MakeMKV (+ klucz beta).
2. Sprawdzić, co jest na płycie (nic nie zgrywa):
   ```powershell
   .\rip-disc.ps1 -Action list
   ```
   Kolumna `Nr` to numer playlisty `.mpls` (ten sam, który Leawo pokazuje jako "Title N") - to podawane jest w `-Title`. `Nr` mają tylko czyste playlisty (odcinki); strumienie `.m2ts` i warianty `.mpls(1)` nie mają `Nr` i skrypt ich nie zgrywa. Kolumna `Id` to wewnętrzny indeks MakeMKV, tylko informacyjnie. Zapisać numery odpowiadające odcinkom (czas ~18-22 min).
3. Zgrać odcinek (akcja `rip` jest domyślna, skrypt zawsze zgrywa po jednym tytule):
   ```powershell
   .\rip-disc.ps1 -Output C:\video-retrieval\data\raw\office -Title 4 -Season 2 -Episode 3 -Prefix office
   ```
   -> `office_s02e03.mkv` w `C:\video-retrieval\data\raw\office`. Bez `-Prefix` plik nazywa się `s02e03.mkv`; obowiązuje nazewnictwo z przedrostkiem serii i małymi literami (`tbbt_s01e01`, `office_s02e03`), bo tę samą nazwę dziedziczy znormalizowany plik w `data\processed\` (starsze ripy TBBT noszą jeszcze nazwy `S01E01.mp4` - kod rozpoznaje oba warianty).

   Bez `-Season`/`-Episode` plik dostaje nazwę z numeru tytułu:
   ```powershell
   .\rip-disc.ps1 -Output C:\video-retrieval\data\raw\office -Title 7
   ```
   -> `title_7.mkv` (z `-Prefix office`: `office_title_7.mkv`).

   Bez `-Title` zgrywa po kolei wszystkie playlisty z `Nr` dłuższe niż 15 min (pomija menu/dodatki), każdą jako `title_<Nr>.mkv`; próg zmienia `-MinLengthMin`:
   ```powershell
   .\rip-disc.ps1 -Output C:\video-retrieval\data\raw\office -MinLengthMin 18
   ```
   Istniejące pliki o docelowej nazwie są pomijane (można bezpiecznie wznowić przerwane zgrywanie).
4. Znormalizować pod potok - osobno dla każdej serii, wejście i wyjście podaje się zawsze jawnie:
   ```powershell
   .\normalize-video.ps1 -InputPath C:\video-retrieval\data\raw\office -OutputPath C:\video-retrieval\data\processed\office
   .\normalize-video.ps1 -InputPath C:\video-retrieval\data\raw\tbbt   -OutputPath C:\video-retrieval\data\processed\tbbt
   ```
   `-InputPath` to folder (przetwarzane są wszystkie `.mkv`/`.mp4` leżące bezpośrednio w nim, bez podfolderów) albo pojedynczy plik. `-OutputPath` to folder (plik wyjściowy dostaje nazwę wejściowego z rozszerzeniem `.mp4`) albo - tylko dla pojedynczego pliku - pełna ścieżka z nazwą, np. `...\processed\office\office_s01e02.mp4`. Gotowe pliki są pomijane (`-Force` nadpisuje), więc krok jest przyrostowy: po dograniu kolejnych odcinków wystarczy powtórzyć to samo polecenie.
5. Wyciągnąć napisy TBBT - OCR w Subtitle Edit, opisany niżej.

> Jeśli PowerShell blokuje skrypty: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## Napisy TBBT - OCR w Subtitle Edit

Ścieżki napisów na płycie są obrazkowe (PGS), więc trzeba je przepuścić przez OCR. Subtitle Edit czyta ścieżkę PGS wprost z kontenera `.mkv`, bez wcześniejszego wyciągania jej do `.sup`.

1. *File -> Open* i wskazać plik z `data\raw\tbbt` (np. `tbbt_s01e15.mkv`); przy kilku ścieżkach wybrać angielską.
2. W oknie OCR ustawić:
   - **OCR Engine:** Tesseract,
   - **Language:** English.
3. *Start OCR*, potem przejrzeć wynik - Tesseract myli `l` z `I` i `0` z `O`, a te pomyłki wchodzą wprost do dopasowywania słów.
4. *File -> Save as* -> format **SubRip (.srt)**, katalog `data\subs\tbbt`.

**Nazwa pliku musi być taka sama jak nazwa nagrania:** `tbbt_s01e15.srt` obok `tbbt_s01e15.mkv` i `tbbt_s01e15.mp4`. Tak i tylko tak szuka ich [`src/annotation/tbbt_time_mapping.py`](../../src/annotation/tbbt_time_mapping.py) (`subtitle_path`) - starsza konwencja `S01E15.srt` nie jest obsługiwana, a odcinek z takim plikiem zostanie zgłoszony jako "TO RIP - no subtitles".

Napisy są potrzebne tylko dla TBBT: z nich powstaje przesunięcie każdego klipu TVR względem pełnego odcinka. "The Office" nie ma adnotacji TVR, więc napisów nie potrzebuje.

## W jakim formacie - pod eksperymenty

Rozdzielenie na dwie warstwy:

**`data/raw/<seria>` (np. `tbbt`, `office`) - master (bezstratny).** Wyjście MakeMKV: `.mkv` z oryginalnym strumieniem H.264 z płyty, bez ponownego kodowania. To archiwum źródłowe - nie jest ruszane w eksperymentach.

**`data/processed/<seria>` - wejście do potoku (znormalizowane).** `.mp4` / H.264 / **CFR** / **stałe klatki kluczowe (GOP=48, ~2 s)**. Powód: OpenCLIP, BLIP/LLaVA, YOLO, SlowFast i zwłaszcza ActionFormer wielokrotnie dekodują i przewijają wideo. Zmienny fps albo długie GOP-y z płyty dają niedeterministyczne indeksowanie klatek - a przy lokalizacji czasowej akcji i ewaluacji segmentacji klatka nr *N* musi być zawsze tą samą klatką. Dlatego:

- kontener **MP4** - najlepiej wspierany przez decord / PyAV / torchvision / OpenCV;
- **H.264 yuv420p**, CRF 18 - praktycznie bezstratne wizualnie, uniwersalne;
- **CFR** + `sc_threshold=0`, `-g 48` - deterministyczny, szybki losowy seek;
- **1080p** zachowane (CLIP i tak przeskaluje do 224/336, ale YOLO i detekcja twarzy zyskują na detalu);
- **audio i napisy** usuwane (`-an -sn`) - potok jest wizualny, a napisy i tak trafiają osobno do `data/subs/`.

Skrypt nie ma przełączników: rozdzielczość, CRF, GOP i brak dźwięku są zaszyte, jedyne parametry to `-InputPath`, `-OutputPath` i `-Force`. Sam remux `.mkv` -> `.mp4` bez re-enkodowania byłby szybszy i bezstratny, ale zostawiałby długi GOP z płyty, a więc wolny i niedeterministyczny seek - przy gęstym próbkowaniu klatek re-enkodowanie z krótkim GOP wychodzi taniej.
