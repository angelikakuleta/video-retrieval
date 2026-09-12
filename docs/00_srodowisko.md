# Środowisko i modele pobierane ręcznie

Odtworzenie środowiska opisuje `README.md` w korzeniu repozytorium. Tutaj zostaje to, czego `conda env create` nie załatwia: dwa modele twarzy, które trzeba pobrać albo skonwertować raz, poza repozytorium.

Potok używa detektora **RetinaFace-ResNet-50** i embeddingów **ArcFace R100 (`glintr100`)** - dokładnie tych wariantów, które opisuje tabela pamięci w rozdziale 5. Pakiet `buffalo_l` (SCRFD + `w600k_r50`) nie jest używany.

Kod nie odsyła tu komunikatami błędu: brakujący plik zgłasza się sam, razem z krokami odtworzenia ([`src/features/faces.py`](../src/features/faces.py), [`src/features/identity.py`](../src/features/identity.py)). Ta strona jest po to, żeby mieć je w jednym miejscu przed pierwszym uruchomieniem.

## WordNet - korpus dla reguł fraz

Reguły wydobywania fraz z zapytania ([`src/retrieval/phrases.py`](../src/retrieval/phrases.py)) stoją na dwóch zasobach: modelu spaCy `en_core_web_sm` i **korpusie WordNet przez NLTK**. Pierwszy instaluje się razem ze środowiskiem, drugi trzeba pobrać osobno - NLTK nie wozi korpusów w pakiecie:

```powershell
conda activate wideo
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet')"
```

Korpus ląduje w `%APPDATA%\nltk_data\corpora\wordnet` i waży kilkanaście megabajtów. Bez niego `phrases._resources()` zgłasza błąd przy pierwszym imporcie, nie w połowie przebiegu.

**Wersje są częścią wyniku, nie szczegółem instalacji.** [`phrases.py`](../src/retrieval/phrases.py) sprawdza je przy imporcie i odmawia pracy na innych: spaCy `en_core_web_sm` 3.8.0, NLTK 3.10.3, WordNet 3.0. Reguły były uruchamiane na tych wersjach, a wyjścia w tabeli sekcji 04 planu i w [`tests/test_phrases.py`](../tests/test_phrases.py) są zapisem tego, co z nich wyszło. Jeśli któryś test fraz przestanie przechodzić po zmianie wersji, jest to sygnał do **ponownego pomiaru progu**, a nie do poprawienia testu.

## ArcFace `glintr100` - jednorazowe pobranie pakietu `antelopev2`

```powershell
conda activate wideo
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='antelopev2')"
```

Pliki lądują w `%USERPROFILE%\.insightface\models\antelopev2\`; potok korzysta z `glintr100.onnx` (dołączony detektor `scrfd_10g_bnkps.onnx` ignorujemy).

## RetinaFace-R50 - konwersja do ONNX

```powershell
cd %TEMP%
git clone https://github.com/biubug6/Pytorch_Retinaface
cd Pytorch_Retinaface
# pobierz wagi Resnet50_Final.pth wg sekcji "Training / Model zoo" w README repozytorium
# (~109 MB), umiesc w .\weights\
python convert_to_onnx.py --trained_model weights/Resnet50_Final.pth --network resnet50
copy FaceDetector.onnx C:\video-retrieval\data\models\retinaface_r50.onnx
```

`data/models/` jest poza kontrolą wersji (duże binaria), więc w repozytorium zostaje sama instrukcja odtworzenia. Wejście modelu jest stałe (640x640): klatka jest skalowana z zachowaniem proporcji i dopełniana kolorem średnim modelu, a ramki i punkty charakterystyczne wracają do współrzędnych oryginału.

## YOLOE promptowalny - wagi dla wariantu E4-D

Wariant E4-D (detekcja sterowana zapytaniem, rozdział 6.5.4) potrzebuje **innego checkpointu** niż tryb indeksowany: `yoloe-11l-seg-pf.pt` jest wariantem *prompt-free* i nie przyjmuje tekstu w ogóle. Ultralytics dociąga oba pliki sam przy pierwszym użyciu, z wydania `ultralytics/assets` **v8.4.0**:

| plik | rozmiar | co to jest |
|---|---|---|
| `yoloe-11l-seg.pt` | 67,7 MiB | checkpoint promptowalny, przyjmuje `set_classes(names, embeddings)` |
| `mobileclip_blt.ts` | 572 MiB | enkoder tekstu podpowiedzi; schodzi przy pierwszym `set_classes` |

```powershell
conda activate wideo
python -c "from ultralytics import YOLOE; m = YOLOE('yoloe-11l-seg.pt'); m.model.get_text_pe(['a coffee mug'])"
```

Pierwsze wywołanie `get_text_pe` doinstalowuje też pakiet `clip` (ultralytics robi to samo). Oba pliki lądują w korzeniu repozytorium, obok `yolo11l.pt` i `yoloe-11l-seg-pf.pt`, i są poza kontrolą wersji.

Embeddingi podpowiedzi liczy się **raz na unikalną krotkę napisów** ([`src/retrieval/query_detection.py`](../src/retrieval/query_detection.py)): enkoder ładuje się raz, a większość zapytań odcinka pyta o tę samą garść codziennych rzeczy.

## Kontrola

```powershell
python scripts\check_gpu.py
```

Sprawdza, czy oba modele są na miejscu i czy liczą na karcie, a nie na procesorze, oraz czy korpus WordNet jest pobrany. Pomiar wag do tabeli pamięci w rozdziale 5 wypisuje sekcja J notatnika [`notebooks/environment_check.ipynb`](../notebooks/environment_check.ipynb).
