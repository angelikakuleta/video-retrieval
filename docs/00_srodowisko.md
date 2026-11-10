# Środowisko: serwer Linux z GPU i modele pobierane ręcznie

Ta gałąź (`server-setup`) opisuje środowisko na serwerze obliczeniowym z Linuksem: Ubuntu 24.04, karty NVIDIA, systemowe CUDA i cuDNN, bez interfejsu graficznego, dostęp wyłącznie przez SSH. Wersja dla Windows z kartą RTX 5080 jest w `README.md` na gałęzi `main`. Dokument prowadzi od pustego konta na serwerze do działającego przykładu i zbiera to, czego `pip install -r requirements.txt` nie załatwia. Nazwy maszyn, ścieżki montowań i aliasy SSH są celowo pominięte: wszędzie poniżej katalog danych to `$DATA_ROOT`, a klon repozytorium to `~/video-retrieval`.

## 1. Układ katalogów

Dwa dyski o różnym charakterze: lokalny katalog domowy serwera i zewnętrzny katalog danych (u nas macierz montowana przez sieć). Kod i środowisko muszą leżeć lokalnie, bo składają się z tysięcy małych plików, na których opóźnienia sieciowego systemu plików są bolesne. Wszystko duże i czytane sekwencyjnie idzie na `$DATA_ROOT`.

```
~/miniforge3/envs/wideo          środowisko conda (lokalnie)
~/video-retrieval                klon repozytorium (lokalnie)
  data/processed/tbbt    -> $DATA_ROOT/datasets/tbbt/processed
  data/processed/office  -> $DATA_ROOT/datasets/office/processed
  data/cache             -> $DATA_ROOT/cache
$DATA_ROOT/datasets/<zbiór>/     jeden katalog na zbiór; nagrania w jego processed/
$DATA_ROOT/models/huggingface    HF_HOME: OpenCLIP, BLIP, LLaVA, X-CLIP, SlowFast
```

Dowiązania wchodzą poziom poniżej katalogów śledzonych przez git (`data/processed/tbbt`, nie `data/processed`), dzięki czemu pliki `.gitkeep` i reguły `.gitignore` zostają nietknięte, a `git status` jest czysty. `data/raw/vatex` i `data/subs/tbbt` nie mają dowiązań: klipów VATEX na serwerze nie ma, bo warianty VATEX liczą się z cache cech (sekcja 5), a napisy odcinków służą tylko notatnikowi `tbbt_02`, którego na serwerze się nie uruchamia. Wyjątkiem jest `data/cache`, który na tej gałęzi jest dowiązaniem w całości; reguła w `.gitignore` nie ma ukośnika na końcu właśnie po to, żeby ignorować także dowiązanie. Modele `antelopev2` zostają w `~/.insightface`, bo tak ich szuka `insightface`; wagi YOLO w korzeniu klonu, bo tak je zapisuje Ultralytics.

## 2. Dostęp i sesja

Logowanie kluczem zamiast hasłem jest potrzebne do przesyłania danych w tle, dla VS Code Remote-SSH i dla poleceń wykonywanych zdalnie. Na stacji roboczej: `ssh-keygen -t ed25519`, wpis `Host <alias>` z `HostName`, `User`, `IdentityFile` i `ServerAliveInterval 60` w `~/.ssh/config`, klucz publiczny dopisany do `~/.ssh/authorized_keys` na serwerze. Każde dłuższe polecenie uruchamia się w tmuxie, bo sesja przeżywa zerwanie połączenia:

```bash
tmux new -s wideo          # nowa sesja; odlaczenie: Ctrl-b, potem d
tmux attach -t wideo       # powrot
printf 'set -g mouse on\nset -g history-limit 50000\n' >> ~/.tmux.conf   # kolko myszy przewija
```

## 3. Miniforge i środowisko `wideo`

Conda daje tylko Pythona i ffmpeg w wariancie GPL; reszta idzie z `requirements.txt`, który jest wiążący (`README.md` tłumaczy, dlaczego nie `environment.yml`).

```bash
cd ~ && curl -fsSL -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && bash Miniforge3.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash && source ~/.bashrc && rm Miniforge3.sh
conda create -y -n wideo -c conda-forge python=3.12.13 "ffmpeg=9.0.1=gpl*"
conda activate wideo && cd ~/video-retrieval
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
grep -vE '^nvidia-' requirements.txt > /tmp/requirements_linux.txt
pip install -r /tmp/requirements_linux.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -m nltk.downloader wordnet
nbstripout --install
```

**Piny `nvidia-*` z `requirements.txt` są pomijane.** Zostały wygenerowane na Windows, a linuksowe koło `torch` deklaruje własny, pełniejszy zestaw bibliotek CUDA (m.in. `cusparse`, `cusolver`, `nccl`, których na Windows nie ma) w innych wersjach; pozostawione piny powodowałyby konflikt zależności. Wersje bibliotek CUDA dobiera więc `torch`.

**Karta nie musi być tą z Windows.** Koło `cu130` nie zawiera osobnych jąder dla każdej architektury; jądra `sm_86` obsługują karty Ada (`sm_89`), a `scripts/check_gpu.py` sprawdza właśnie to, czy w kole jest architektura obsługująca obecną kartę, i wypisuje, która to.

### Dwie poprawki po instalacji

**Podwójny `onnxruntime`.** Pakiet `hsemotion-onnx` dociąga procesorowy `onnxruntime` obok `onnxruntime-gpu` (na Windows to samo robi `insightface`). Oba dostarczają ten sam moduł i nadpisują się nawzajem; ten zainstalowany jako drugi wygrywa, więc wariant GPU trzeba postawić na nowo:

```bash
pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu==1.28.0
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"   # CUDAExecutionProvider na liscie
```

**cuDNN z pip nie może być starszy od systemowego.** Objaw: pierwsza konwolucja na karcie kończy się `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, choć `check_gpu.py` przechodzi (mnożenie macierzy nie używa cuDNN). Przyczyna: `libcudnn.so.9` doczytuje swoje biblioteki składowe po nazwie; składnik, którego pakiet pip nie zawiera (np. `libcudnn_engines_tensor_ir`, dodany po wersji 9.20, którą pinuje `torch`), system podsuwa z własnej, nowszej instalacji cuDNN i wersje się rozjeżdżają. Ani `LD_LIBRARY_PATH`, ani `LD_PRELOAD` tego nie naprawią, bo nie da się podstawić pliku, którego nie ma. Rozwiązanie: pakiet pip w wersji nie niższej niż systemowa, nadal w gałęzi 9, zgodnej binarnie:

```bash
ldconfig -p | grep 'libcudnn.so.9'                      # wersja systemowa, np. 9.27
pip install -U "nvidia-cudnn-cu13>=9.27,<10"            # pip ostrzeze o pinie torch; to tylko metadane
python -c "import torch; x=torch.randn(2,3,64,64,device='cuda'); print(torch.nn.Conv2d(3,8,3).cuda()(x).shape)"
```

Diagnostyka, gdy objaw wróci po aktualizacji systemu: wypisać pliki `cudnn` zmapowane w procesie (`/proc/self/maps`) i sprawdzić, czy któryś pochodzi spoza katalogu środowiska.

## 4. Klon repozytorium, dowiązania, zmienne

```bash
git clone -b server-setup <adres repozytorium> ~/video-retrieval && cd ~/video-retrieval
git config user.name "<imie i nazwisko>" && git config user.email "<e-mail konta, do ktorego nalezy repozytorium>"
N=$DATA_ROOT
mkdir -p $N/datasets/tbbt/processed $N/datasets/office/processed $N/cache $N/models/huggingface data/models
ln -s $N/cache data/cache
ln -s $N/datasets/tbbt/processed data/processed/tbbt && ln -s $N/datasets/office/processed data/processed/office
echo "export HF_HOME=$N/models/huggingface" >> ~/.bashrc && source ~/.bashrc
git status --short      # ma byc pusty
```

Tożsamość gita ustawia się lokalnie w klonie, żeby globalna konfiguracja konta na serwerze nie podpisała commitów innym adresem. `CLAUDE.md` jest poza gitem i trzeba go skopiować ręcznie, jeśli ma być na serwerze.

## 5. Dane, których nie da się odtworzyć obliczeniem

Z repozytorium przychodzi kod, konfiguracje, adnotacje, zakresy i wyniki przebiegów. Resztę trzeba przesłać ze stacji roboczej na `$DATA_ROOT`:

| co | dokąd | uwagi |
|---|---|---|
| nagrania `data/processed/{tbbt,office}` | `datasets/<zbiór>/processed/` | ok. 100 GB, nieredystrybuowalne |
| klipy VATEX `data/raw/vatex` | nie są przesyłane | kompletny cache pokrywa wszystkie warianty poza E4-D i miniaturami przykładów, które dekodują klatki; ponowne pobieranie z YouTube skryptami z `scripts/prepare_data/` dałoby inny skład zamrożonego zbioru |
| napisy odcinków `data/subs/tbbt` | nie są przesyłane | potrzebne tylko notatnikowi `tbbt_02`, którego na serwerze się nie uruchamia |
| `data/models/retinaface_r50.onnx` | `~/video-retrieval/data/models/` | albo konwersja wg sekcji 6 |
| cache cech `data/cache` | `cache/` | ok. 4 GB w 29 tys. małych plików, więc jako jedno archiwum tar; odtwarzalny z nagrań, ale oszczędza godziny karty, a bez nagrań jest jedynym źródłem cech |

Narzędzia: `scp` dla małych plików, `sftp` lub `rsync` dla dużych. W sftp `put -a` dociąga przerwany plik, ale wymaga, by plik zdalny już istniał; nowy plik wysyła się zwykłym `put`, kompletny pomija się po porównaniu rozmiarów. Przez VPN o przepustowości ok. 130 Mbit/s 100 GB zajmuje ok. 2 godzin; stacja robocza nie może w tym czasie usypiać. Po przesłaniu porównać rozmiar każdego pliku po obu stronach. Każdy moduł czytający cache sprawdza najpierw własny plik i dopiero przy jego braku otwiera nagranie, dlatego kompletny cache pozwala powtarzać przebiegi także dla zbioru, którego nagrań na serwerze nie ma.

## 6. Modele pobierane ręcznie

Potok używa detektora **RetinaFace-ResNet-50** i embeddingów **ArcFace R100 (`glintr100`)** z pakietu `antelopev2`; `buffalo_l` nie jest używany. Brakujący plik zgłasza się sam, razem z krokami odtworzenia (`src/features/faces.py`, `src/features/identity.py`); tu są zebrane w jednym miejscu.

### WordNet i spaCy

Model `en_core_web_sm` 3.8.0 instaluje się z `requirements.txt`; korpus WordNet pobiera `python -m nltk.downloader wordnet` z sekcji 3 (do `~/nltk_data`). **Wersje są częścią wyniku**: `src/retrieval/phrases.py` sprawdza przy imporcie spaCy `en_core_web_sm` 3.8.0, NLTK 3.10.3 i WordNet 3.0 i odmawia pracy na innych; jeśli po zmianie wersji przestanie przechodzić któryś test fraz, jest to sygnał do ponownego pomiaru progu, nie do poprawienia testu.

### ArcFace `glintr100`: pakiet `antelopev2`

```bash
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='antelopev2')"
```

Pobranie kończy się `AssertionError: 'detection' in self.models` i to jest spodziewane: archiwum z wydania v0.7 zawiera w środku katalog `antelopev2/`, więc modele lądują w `~/.insightface/models/antelopev2/antelopev2/`, poziom za głęboko. Trzeba je przenieść wyżej:

```bash
D=~/.insightface/models/antelopev2; mv $D/antelopev2/*.onnx $D/ && rmdir $D/antelopev2 && ls $D
```

Potok korzysta z `glintr100.onnx` (ok. 261 MB); dołączony detektor `scrfd_10g_bnkps.onnx` ignorujemy.

### RetinaFace-R50: konwersja do ONNX

Najprościej skopiować gotowy `data/models/retinaface_r50.onnx` (ok. 105 MB) z maszyny, na której już był. Konwersja od zera:

```bash
cd /tmp && git clone https://github.com/biubug6/Pytorch_Retinaface && cd Pytorch_Retinaface
# pobierz wagi Resnet50_Final.pth wg sekcji "Training / Model zoo" w README repozytorium (~109 MB) do ./weights/
python convert_to_onnx.py --trained_model weights/Resnet50_Final.pth --network resnet50
cp FaceDetector.onnx ~/video-retrieval/data/models/retinaface_r50.onnx
```

`data/models/` jest poza kontrolą wersji. Wejście modelu jest stałe (640x640): klatka jest skalowana z zachowaniem proporcji i dopełniana kolorem średnim modelu, a ramki i punkty charakterystyczne wracają do współrzędnych oryginału.

### YOLO i YOLOE promptowalny

Wariant E4-D (detekcja sterowana zapytaniem, rozdział 6.5.4) potrzebuje innego checkpointu niż tryb indeksowany: `yoloe-11l-seg-pf.pt` jest wariantem *prompt-free* i nie przyjmuje tekstu. Ultralytics dociąga wszystkie pliki sam, z wydania `ultralytics/assets` v8.4.0, do bieżącego katalogu, dlatego polecenie uruchamia się w korzeniu klonu:

```bash
cd ~/video-retrieval && python -c "from ultralytics import YOLO, YOLOE; YOLO('yolo11l.pt'); YOLOE('yoloe-11l-seg-pf.pt'); m = YOLOE('yoloe-11l-seg.pt'); m.model.get_text_pe(['a coffee mug'])"
```

| plik | rozmiar | co to jest |
|---|---|---|
| `yolo11l.pt` | 49 MiB | detektor E4-B |
| `yoloe-11l-seg.pt` | 68 MiB | checkpoint promptowalny, przyjmuje `set_classes(names, embeddings)` |
| `yoloe-11l-seg-pf.pt` | 71 MiB | wariant prompt-free (E4-C) |
| `mobileclip_blt.ts` | 572 MiB | enkoder tekstu podpowiedzi; schodzi przy pierwszym `get_text_pe` |

Embeddingi podpowiedzi liczy się raz na unikalną krotkę napisów (`src/retrieval/query_detection.py`). Ultralytics zapisuje ustawienia w `~/.config/Ultralytics`; jeśli katalog nie istnieje, przy pierwszym uruchomieniu ostrzega i używa `/tmp`, co jest nieszkodliwe.

### Modele z Hugging Face

OpenCLIP ViT-H/14, BLIP, LLaVA-1.5, X-CLIP i SlowFast pobierają się same przy pierwszym użyciu do `HF_HOME` (sekcja 4), łącznie ok. 30 GB. Ostrzeżenie o nieuwierzytelnionych żądaniach do HF Hub można zignorować albo ustawić `HF_TOKEN`.

## 7. Wybór karty i uruchamianie

Na współdzielonym serwerze karty nie są rezerwowane. Przed przebiegiem sprawdza się obciążenie i w powłoce tmuxa ogranicza widoczność do jednej, wolnej karty; potok używa wtedy urządzenia `cuda:0`, które wskazuje na nią:

```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
export CUDA_VISIBLE_DEVICES=<numer wolnej karty>
python scripts/run_experiment.py configs/e1a_tbbt.yaml --split dev 2>&1 | tee ~/e1a_tbbt_dev.log
```

Jedna karta na długie obliczenia; więcej tylko po uzgodnieniu z administratorem. Dekodowanie wideo jest zadaniem procesora i zwykle ono ogranicza tempo, więc częściowo bezczynna karta przy zajętym procesorze nie jest objawem błędu.

## 8. Kontrola

Trzy sprawdziany, w tej kolejności:

```bash
python scripts/check_gpu.py      # wszystkie wiersze [OK]; wiersz o jadrach wypisuje, ktora architektura obsluguje karte
pytest -q                        # testy funkcji czystych, bez danych i bez karty
python scripts/run_experiment.py configs/e1a_tbbt.yaml --split dev 2>&1 | tee ~/e1a_tbbt_dev.log
```

Pomiar TFLOPS w `check_gpu.py` wychodzi zaniżony na karcie dzielonej z cudzym procesem; liczy się, czy sprawdzenia przechodzą. Przebieg E1-A na części deweloperskiej TBBT (sześć odcinków) jest minimalnym przykładem: czerń, segmentacja stałymi oknami, embeddingi OpenCLIP, indeks, miary. Wynik porównuje się z przebiegiem referencyjnym z Windows (`results/runs/E1-A_tbbt_dev_20260908_012526`):

```bash
cd ~/video-retrieval && python - <<'EOF'
import json, glob, os
ref = json.load(open("results/runs/E1-A_tbbt_dev_20260908_012526/metrics.json"))
new_dir = sorted(glob.glob("results/runs/E1-A_tbbt_dev_*"))[-1]
new = json.load(open(os.path.join(new_dir, "metrics.json")))
print("new run:", new_dir, "| collection", ref["collection_size"], "->", new["collection_size"])
for k, a in ref["overall"].items():
    b = new["overall"].get(k)
    print(f"{k:16} reference {a!s:>20}  server {b!s:>20}{'' if a == b else '  <-- differs'}")
EOF
```

Liczba zapytań (208) i fragmentów (711) musi się zgadzać co do sztuki. Miary powinny być identyczne albo różnić się na trzecim-czwartym miejscu po przecinku, bo embeddingi liczone na innej karcie różnią się o ułamki promila i potrafią przestawić bliskie sobie pozycje rankingu; różnica rzędu punktów procentowych oznacza inne wejście. Przy pierwszym odtworzeniu na serwerze wszystkie miary wyszły identyczne.

## 9. Praca z VS Code

Rozszerzenie Remote - SSH łączy się z aliasem z `~/.ssh/config` i otwiera `~/video-retrieval` jak lokalny folder; przy pierwszym połączeniu wgrywa do `~/.vscode-server` swój komponent serwerowy. Rozszerzenia Python i Jupyter instaluje się po stronie serwera, interpreter to `~/miniforge3/envs/wideo/bin/python` (tak ustawia go `.vscode/settings.json` na tej gałęzi), a kernel notatników rejestruje się raz: `python -m ipykernel install --user --name wideo --display-name "Python (wideo)"`. Terminal VS Code ginie razem z oknem i połączeniem, więc długie przebiegi nadal idą w tmuxie.
