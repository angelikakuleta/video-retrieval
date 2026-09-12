# Instrukcja krok po kroku - faza deweloperska

Od adnotacji do zamrożonej konfiguracji, dla obu seriali naraz. Co dalej, gdy wszystko jest zamrożone: [05_instrukcja_test.md](05_instrukcja_test.md). Wszystko jest wznawialne - każde polecenie można przerwać i powtórzyć, doliczy tylko braki.

```
conda activate wideo
cd C:\video-retrieval
python scripts/check_gpu.py
```

Modele spoza pakietów: `data/models/retinaface_r50.onnx` (konwersja z Pytorch_Retinaface) i `%USERPROFILE%\.insightface\models\antelopev2\glintr100.onnx` - kroki w [00_srodowisko.md](00_srodowisko.md). Wagi YOLO leżą w korzeniu repozytorium, reszta pobiera się z Hugging Face przy pierwszym użyciu. FFmpeg pochodzi ze środowiska conda; kod szuka go obok interpretera, więc działa też bez aktywacji.

## Stan wyjściowy

Krok pierwszy wymaga kompletnego zbioru deweloperskiego obu seriali: rejestr, plik znaczników i plik zapytań mają się zgadzać co do identyfikatora, czyli tyle samo pozycji, żadnej w zapytaniach spoza przyjętych, żadnej przyjętej bez zapytania. Bieżące liczebności wypisuje [`e1_segmentation.ipynb`](../notebooks/experiments/e1_segmentation.ipynb).

**Oba seriale wchodzą do każdego kroku na równi.** Reguła decyzji z rozdziału 4 wymaga zgodnego kierunku różnicy w obu serialach i przedziału ufności na ich połączonych odcinkach, więc porównanie ma sens dopiero wtedy, gdy dany wariant przeliczył się po obu stronach. [`compare_runs.py`](../scripts/compare_runs.py) przyjmie też jeden serial, ale oznaczy werdykt jako prowizoryczny.

## 1. Adnotacje

Ścieżkę w całości opisuje sekcja 6. W skrócie: dla TBBT `tbbt_02` buduje rejestr i po dwa warianty kandydatów na odcinek, Ty wybierasz wariant i weryfikujesz go w adnotatorze, a `tbbt_03` nanosi decyzje i zostawia zakresy korpusu, plik dziur, szkielet znaczników i pliki zapytań. Dla *The Office* odpada krok kandydatów - adnotacje powstają od zera, więc wystarczy [`office_02_annotations.ipynb`](../notebooks/prepare_data/office_02_annotations.ipynb). Odcinki jeszcze nieadnotowane są pomijane i wypisywane, więc można startować na tym, co jest.

## 2. Znaczniki zapytań

```
python scripts/prepare_data/query_tags.py tbbt   --init --check --split dev
python scripts/prepare_data/query_tags.py office --init --check --split dev
```

Notatnik i skrypt zostawiają `<zbiór>_query_tags_new.csv` obok pliku, który wypełniasz - z przeniesionymi komórkami, które już masz. Skopiuj go pod nazwę bez przyrostka, uzupełnij (modelem, w adnotatorze, albo jedno i drugie) i uruchom notatnik z kroku 1 ponownie, żeby znaczniki weszły do plików zapytań. W tym pliku poprawia się też literówki w treści zapytań: **to on, a nie rejestr, jest źródłem treści dla plików `.jsonl`**.

Dla VATEX skryptu nie ma - szkielet [`vatex_query_tags_new.csv`](../data/annotations/vatex/vatex_query_tags_new.csv) zostawia [`notebooks/prepare_data/vatex_03_test_annotations.ipynb`](../notebooks/prepare_data/vatex_03_test_annotations.ipynb) i ten sam notatnik wciąga wypełniony plik do `vatex_queries_test.jsonl`. Zestaw kolumn jest tam węższy: `wymaga_obiektu`, `wymaga_scenerii`, `wymaga_ruchu` i `complexity` - tylko tyle rozstrzyga jednozdaniowy opis klipu.

Pokrycie znaczników decyduje o tym, które kontrasty potwierdzające da się policzyć. Obecny stan na zbiorze deweloperskim:

| znacznik | TBBT | *The Office* |
|---|---|---|
| `wymaga_osoby` | 414 | 204 |
| `wymaga_obiektu` | 216 | 138 |
| `wymaga_scenerii` | 111 | 47 |
| `wymaga_ruchu` | 381 | 170 |
| `wymaga_mimiki` | 35 | 36 |
| `complexity` P / Z | 97 / 325 | 101 / 117 |

Kontrast E5 opiera się na `wymaga_mimiki`, a tych zapytań jest po około 35 na serial - podpróba jest wąska i przedział wyjdzie szeroki. To ograniczenie materiału, nie błąd; warto je odnotować przy omówieniu wyniku E5.

## 3. Kontrola poprawności implementacji

```
python scripts/vatex_check.py
```

Buduje kolekcję scen VATEX i liczy miary na zbiorze kontrolnym (10 opisów na klip). Wynik do porównania z wartościami publikowanymi, sekcja "Weryfikacja poprawności implementacji" rozdziału 6.

## 4. E1 - wybór strategii segmentacji

```
python scripts/run_experiment.py configs/e1a_office.yaml --split dev
python scripts/run_experiment.py configs/e1b_office.yaml --split dev
python scripts/run_experiment.py configs/e1c_office.yaml --split dev

python scripts/run_experiment.py configs/e1a_tbbt.yaml --split dev
python scripts/run_experiment.py configs/e1b_tbbt.yaml --split dev
python scripts/run_experiment.py configs/e1c_tbbt.yaml --split dev

python scripts/compare_runs.py results/runs/E1-*_dev_* ^
    --reference E1-A --simplicity E1-A E1-B E1-C ^
    --contrast requirements:wymaga_scenerii ^
    --out results/reports/e1_dev
```

Pierwszy przebieg każdego serialu robi dwie rzeczy jednorazowo: wyznacza granice fragmentów i liczy embeddingi klatek ViT-H/14 (zmierzone: 75-130 s na odcinek). Warianty B i C dokładają jeden przebieg dekodowania na odcinek, dzielony przez oba detektory ujęć. Czarne klatki są już zmierzone dla wszystkich dwunastu odcinków deweloperskich i leżą w `data/cache/black/`.

Wzorzec `E1-*_dev_*` łapie oba seriale. Skrypt sam rozwija gwiazdki, więc to samo polecenie działa w PowerShellu i w bashu; PowerShell wzorców programom natywnym nie rozwija i przekazałby je dosłownie.

## 5. E2 - wybór reprezentacji bazowej

Najpierw wpisz zwycięzcę E1 w `segmentation.strategy` we wszystkich `configs/e2*`, `e3*`, `e4*`, `e5*` - w wersjach `_office` **i** `_tbbt`.

```
python scripts/run_experiment.py configs/e2a_office.yaml  --split dev
python scripts/run_experiment.py configs/e2ap_office.yaml --split dev
python scripts/run_experiment.py configs/e2b_office.yaml  --split dev
python scripts/run_experiment.py configs/e2bp_office.yaml --split dev

python scripts/run_experiment.py configs/e2a_tbbt.yaml  --split dev
python scripts/run_experiment.py configs/e2ap_tbbt.yaml --split dev
python scripts/run_experiment.py configs/e2b_tbbt.yaml  --split dev
python scripts/run_experiment.py configs/e2bp_tbbt.yaml --split dev

python scripts/compare_runs.py results/runs/E2-*_dev_* ^
    --reference E2-A --simplicity E2-A E2-B --control E2-Ap E2-Bp ^
    --contrast requirements:wymaga_ruchu ^
    --out results/reports/e2_dev
```

E2-A jest natychmiastowy - indeks powstał w kroku 4. E2-Ap liczy embeddingi mniejszym enkoderem. E2-B buduje kolekcję X-CLIP wraz z cechami klatek fragmentu (około 3-5 min na odcinek, czyli rzędu godziny na dwanaście odcinków) i dokłada mechanizm podpowiedzi; E2-Bp czyta ten sam indeks bez podpowiedzi, więc trwa minuty. Warianty kontrolne E2-Ap i E2-Bp (w pracy A′ i B′) **nie uczestniczą w decyzji** - stąd `--control`. Rozkładają różnicę B−A na trzy człony po jednej zmianie: rozmiar enkodera (Ap−A), typ reprezentacji i dane douczania (Bp−Ap), mechanizm podpowiedzi (B−Bp).

## 6. Zamrożenie BAZY

Zwycięzcę E2 wpisz w [`configs/frozen.yaml`](../configs/frozen.yaml), w klucz `scene_embedding` - zwycięzcę E1 w blok `segmentation`, i to cały blok, nie samą nazwę strategii. Ten plik jest jedynym miejscem, w którym decyzja zapada: konfiguracje, które z niej wynikają, powstają z niego generatorem, a nie z ręki.

[`frozen.yaml`](../configs/frozen.yaml) uzupełnia się **ręcznie i w dwóch etapach**. Etap pierwszy to blok `matching` - miara i próg, po raporcie z pomiaru progu. Etap drugi to sześć werdyktów: `segmentation`, `scene_embedding`, `caption`, `objects`, `face_regions`, `motion`. Plik niesie wyłącznie wartości, bez komentarzy - pochodzenie każdego werdyktu (eksperyment, przedział, przebiegi) zostaje w `results/reports/e<n>_dev.json` i w tabelach rozdziału 6.

Po każdym uzupełnieniu uruchom generator - najpierw na sucho, potem z zapisem:

```
python scripts/make_configs.py
python scripts/make_configs.py --execute
```

Bez `--execute` nic nie jest zapisywane: skrypt wypisuje rodzina po rodzinie, co powstanie, co się zmieni, co jest już zgodne i co czeka na którą decyzję. **Zwalnianie idzie rodzinami, nie wszystko naraz** - pliki, których decyzje już zapadły, powstają od razu, a reszta po powtórnym uruchomieniu tego samego polecenia, gdy dojdą werdykty E4 i E5.

Generator pisze 57 konfiguracji: warianty deweloperskie E2-E5, wkłady indywidualne, potok PEŁNY oraz konfiguracje wkładu krańcowego i efektu podmiany. **Nie pisze sześciu konfiguracji E1 ani dwóch `e5bp_*`** - te zostają ręczne. E1 bada segmentację przy jakiejś bazie, a bazę rozstrzyga dopiero E2, które biegnie po nim, więc generator musiałby nieść wartość, której nie ma jeszcze w żadnym werdykcie.

Blok `matching` z progami trafia do plików generowanych sam, prosto z [`frozen.yaml`](../configs/frozen.yaml). Konfiguracje ręczne go nie potrzebują: E1 punktuje samą bazą, a E5-Bp regionami twarzy w słowniku otwartym - żadna z nich nie sięga po zamknięty słownik.

Od tej chwili BAZA jest ustalona.

## 7. Ekstrakcja cech dla E3-E5

```
python scripts/run_features.py configs/e4b_office.yaml --split dev   # YOLO11
python scripts/run_features.py configs/e4c_office.yaml --split dev   # YOLOE-11
python scripts/run_features.py configs/e5b_office.yaml configs/e5c_office.yaml --split dev
python scripts/run_features.py configs/e3b_office.yaml --split dev   # BLIP

python scripts/run_features.py configs/e4b_tbbt.yaml --split dev
python scripts/run_features.py configs/e4c_tbbt.yaml --split dev
python scripts/run_features.py configs/e5b_tbbt.yaml configs/e5c_tbbt.yaml --split dev
python scripts/run_features.py configs/e3b_tbbt.yaml --split dev
```

Trzecia linia każdego bloku liczy detekcję twarzy raz i zasila z jednego bufora wycinków zarówno embeddingi regionów (E5-B), jak i HSEmotion (E5-C).

Rząd wielkości na serial (6 odcinków, około 6300 klatek siatki): detektory obiektów po kilka minut, twarze kilkanaście, BLIP kilkanaście. Krok jest opcjonalny - [`run_experiment.py`](../scripts/run_experiment.py) policzy to samo, gdy napotka brak - ale wygodniej mieć to z głowy przed serią przebiegów.

**LLaVA-1.5 osobno, na noc.** To około 2-3 h na serial, więc oba naraz nie zmieszczą się w jeden wieczór:

```
python scripts/run_features.py configs/e3c_office.yaml configs/e3c_tbbt.yaml --split dev --only captions
```

Skrypt bierze oba seriale po kolei, jest wznawialny odcinek po odcinku i loguje do `results/measurements/`. Potrzebuje wyłącznie zakresów korpusu i plików `.mp4` - podpisy liczą się na siatce 1,25 s całego odcinka i nie zależą ani od segmentacji, ani od zapytań. **Zamknij wcześniej wszystko, co zajmuje kartę**: LLaVA chce około 14 GB.

Bufor podpisów nie zależy od zakresów korpusu, więc przebudowa adnotacji go nie unieważnia. Osadzenia podpisów są osobnym, tanim krokiem i muszą się wykonać przed E3-C.

## 8. Próg dopasowania frazy do nazwy klasy

Sygnały o zamkniętym słowniku - obiektowy, mimiki i ruchu - porównują frazę wydobytą z zapytania z nazwami klas swojego słownika. Dopasowanie liczy się tylko wtedy, gdy osiąga próg τ; poniżej niego sygnał nie ma o tym zapytaniu nic do powiedzenia i pozostaje nieaktywny. **Próg wyznacza się osobno dla każdego słownika**, na parach ocenionych ręcznie.

Krok jest obowiązkowy przed E4 i E5: konfiguracja punktująca wobec zamkniętego słownika bez bloku `matching` zostaje odrzucona przez [`run_experiment.py`](../scripts/run_experiment.py), zanim cokolwiek policzy. E1, E2 i E3 progu nie potrzebują - E1 i E2 punktują samą bazą, a sygnał opisowy porównuje embeddingi tekstu, nie nazwy klas.

**Bramka: żaden przebieg z nową punktacją przed zamrożeniem progu.** Nie ma "przebiegów roboczych na roboczym progu": osoba wybierająca wartość nie może znać wyniku Recall dla wartości sąsiedniej.

### 8.1 Wylosowanie prób

```
python scripts/make_threshold_samples.py
```

Zapisuje `data/interim/threshold/threshold_sample_<słownik>.csv` - po jednej próbie na słownik.

Losowanie jest **warstwowe**: dziesięć przedziałów pewności równej szerokości (`b01`-`b10`) plus osobna warstwa `exact` na pary, w których fraza jest dosłownie nazwą klasy. Powód jest arytmetyczny. Dla słownika COCO populacja liczy 842 pary, z czego 612 leży w dwóch środkowych przedziałach; przy losowaniu prostym stu par w trzech najwyższych przedziałach wypadłyby **dwie**. A próg to najniższa pewność, przy której precyzja sięga 0,9 - czyli odczytuje się go właśnie z górnej części zakresu. Warstwowanie kładzie tam jedenaście par zamiast dwóch, więc krzywa precyzji daje się oszacować w miejscu, w którym się ją czyta.

Proporcje populacji wracają przez **wagi**: para z przedziału o 363 elementach, z którego oceniono osiem, waży 45,4, a para z przedziału o dwóch elementach waży 1,0. Precyzja przy progu jest tymi wagami ważona, więc nie jest zawyżona przez to, że rzadkie przedziały są w próbie nadreprezentowane. Wagi i liczebności pokazuje tabela "Rozkład po dziesięciu przedziałach" w [`notebooks/experiments/threshold.ipynb`](../notebooks/experiments/threshold.ipynb).

Słownik, którego populacja jest mała na tyle, by ocenić ją w całości, warstw nie dostaje i w tej tabeli go nie ma - tak jest z HSEmotion, gdzie ocenionych zostało wszystkie 50 par.

### 8.2 Ocena ręczna

To jedyny krok tej procedury, którego nie da się powtórzyć z pamięci, i jedyny wykonywany poza kodem. Wykonuje go **ta sama osoba, która sporządzała adnotacje**, nie autorka pracy; instrukcja dla niej leży w [`tools/judge/README.md`](../tools/judge/README.md) i jest samodzielna - nie wymaga wiedzy o systemie.

```
python tools/judge/judge_pairs.py --samples data/interim/threshold/threshold_sample_*.csv --output data/interim/threshold/threshold_judgments.csv
```

Obie ścieżki są **wymagane** i narzędzie nie zgaduje żadnej z nich. Odpowiedzi zapisują się po każdym naciśnięciu klawisza, więc ocenę można rozłożyć na kilka posiedzeń; ponowne uruchomienie wraca do pierwszej nieocenionej pary. Oceniająca nie widzi pewności dopasowania ani klasy źródłowej klipu, a pary idą w kolejności losowej.

Gotowy plik przenosi się ręcznie tam, gdzie czyta go pomiar:

```
Copy-Item data\interim\threshold\threshold_judgments.csv data\annotations\threshold_judgments.csv
```

### 8.3 Pomiar

```
python scripts/measure_threshold.py --split dev
```

Dla każdego słownika bierze **najniższą pewność, przy której wśród przyjętych dopasowań co najmniej 90% jest poprawnych**. Poziom 0,9 jest ustalony przed pomiarem i nie podlega strojeniu. Obok progu raportowany jest zasięg - odsetek fraz i odsetek zapytań, dla których dopasowanie osiąga próg - bo próg wysoki przy zasięgu bliskim zeru oznaczałby sygnał poprawny i bezużyteczny. Tabele pokazuje [`notebooks/experiments/threshold.ipynb`](../notebooks/experiments/threshold.ipynb).

### 8.4 Zamrożenie i generator

Cztery progi wpisz ręcznie do [`configs/frozen.yaml`](../configs/frozen.yaml), w blok `matching` (etap pierwszy, krok 6), a potem uruchom generator:

```
python scripts/make_configs.py
python scripts/make_configs.py --execute
```

Blok `matching` trafia stąd do każdej konfiguracji, która punktuje wobec zamkniętego słownika. **Od tej chwili wolno uruchamiać E4 i E5.**

## 9. E3, E4, E5 - wybory wariantowe

```
python scripts/run_experiment.py configs/e3b_office.yaml --split dev
python scripts/run_experiment.py configs/e3c_office.yaml --split dev
python scripts/run_experiment.py configs/e4b_office.yaml --split dev
python scripts/run_experiment.py configs/e4c_office.yaml --split dev
python scripts/run_experiment.py configs/e5b_office.yaml --split dev
python scripts/run_experiment.py configs/e5c_office.yaml --split dev

python scripts/run_experiment.py configs/e3b_tbbt.yaml --split dev
python scripts/run_experiment.py configs/e3c_tbbt.yaml --split dev
python scripts/run_experiment.py configs/e4b_tbbt.yaml --split dev
python scripts/run_experiment.py configs/e4c_tbbt.yaml --split dev
python scripts/run_experiment.py configs/e5b_tbbt.yaml --split dev
python scripts/run_experiment.py configs/e5c_tbbt.yaml --split dev

python scripts/compare_runs.py results/runs/E3-*_dev_* ^
    --reference E3-B --simplicity E3-B E3-C ^
    --contrast complexity:Z --out results/reports/e3_dev
python scripts/compare_runs.py results/runs/E4-*_dev_* ^
    --reference E4-B --simplicity E4-B E4-C ^
    --contrast requirements:wymaga_obiektu --out results/reports/e4_dev
python scripts/compare_runs.py results/runs/E5-*_dev_* ^
    --reference E5-B --simplicity E5-B E5-C ^
    --contrast requirements:wymaga_mimiki --out results/reports/e5_dev
```

Wariantem odniesienia każdej pary jest wariant prostszy według hierarchii złożoności z rozdziału 4 (BLIP przed LLaVA, YOLO11 przed YOLOE, regiony przed HSEmotion), więc gdy przedział obejmuje zero, wygrywa on.

## 10. Koszt obliczeniowy (około pół godziny)

```
python scripts/measure_cost.py --dataset office --split test
```

**Krok wykonuje się po zamrożeniu konfiguracji, na materiale testowym**, mimo że stoi w instrukcji fazy deweloperskiej. Powód: tabela kosztu stoi w rozdziale 6 obok wyników testowych i wchodzi do zestawienia wkład-koszt, z którego bierze się rekomendacje. Czas obsługi zapytania i rozmiar indeksu **zależą od wielkości kolekcji** (Office: 879 fragmentów na dev wobec 2893 na teście), więc muszą pochodzić z tej części, którą opisują wyniki. Stawki ekstrakcji są od podziału niezależne - mierzy się je na stałej próbce klatek - więc przebieg na teście trwa dokładnie tyle samo co na dev i nie ma powodu go dzielić.

**Zamknij wcześniej wszystko, co zajmuje kartę.** Skrypt ładuje modele po kolei, a sama LLaVA chce 14 GB - przeglądarka trzymająca pół giga zmienia mierzony szczyt pamięci i potrafi nie wpuścić największego modelu.

Mierzy cztery wielkości z tab. `koszt-wyniki`: czas ekstrakcji w minutach na godzinę materiału, szczyt pamięci karty, rozmiar reprezentacji na dysku w MB na godzinę oraz czas obsługi zapytania jako mediana i 95. percentyl. Procedura jest ta z rozdziału 4: ten sam sprzęt, modele rozgrzane (rozgrzewka odrzucana), ta sama liczba powtórzeń wszędzie, próbka z prawdziwego materiału - koszt detektora zależy od tego, co widzi.

Ekstrakcja mierzy się na próbce (320 klatek i 512 wycinków rozłożonych po odcinkach), a wynik przelicza na godzinę materiału. Każdy komponent przechodzi swoją próbkę **wsadami wielkości produkcyjnej**, więc szczyt pamięci jest tym, który potok osiąga naprawdę; dokładność bierze się z liczby przetworzonych jednostek i z 15 powtórzeń, nie z rozdmuchanego wsadu. Całość zajmuje około 29 minut, z czego 23 to LLaVA - `--caption-frames` jest pokrętłem decydującym o długości przebiegu.

Faza zapytania mierzy się domyślnie na potoku PEŁNYM (`configs/full_<zbiór>.yaml`), bo tylko konfiguracja z włączonymi komponentami ma dla nich czas - inaczej kolumna "Zapytanie" wypełniłaby się wyłącznie dla sceny. Pierwszy przebieg doliczy przy okazji to, czego PEŁNY jeszcze nie ma na dysku (bufor ruchu SlowFast); to ekstrakcja, nie pomiar, i zdarza się raz.

Tabela kosztu opisuje komponenty, nie seriale, więc wystarczy jeden pomiar - ale **podaj w podpisie, na którym materiale powstał**. Koszt detekcji twarzy zależy od liczby twarzy na godzinę, a ta różni się między serialami; skrypt tę liczbę wypisuje. Wynik ląduje w `results/measurements/cost_<data>_<godzina>.json`; kolejne przebiegi stają obok siebie, nic się nie nadpisuje. Podgląd: [`notebooks/experiments/cost.ipynb`](../notebooks/experiments/cost.ipynb).

### Czego ten pomiar nie obejmuje

Puste komórki w tabeli kosztu mają trzy różne przyczyny i żadna nie jest usterką:

- **Kolumna "Zapytanie" dla wariantów przegranych** (ViT-B/32, X-CLIP, BLIP, YOLOE-11, HSEmotion) jest pusta w tym pomiarze, bo faza zapytania mierzy się na potoku PEŁNYM, a w nim siedzą wyłącznie zwycięzcy. **Nie znaczy to jednak, że tych liczb nie ma** - czas obsługi zapytania każdego wariantu zapisuje `metrics.json` jego własnego przebiegu, na tej samej kolekcji, więc wystarczy je stamtąd wziąć. Warto, bo różnice są realne: na części deweloperskiej Office YOLOE-11 (E4-C) ma medianę 38,3 ms wobec 20,7 ms dla YOLO11 (E4-B), a p95 91,5 wobec 27,3 - dopasowanie frazy przechodzi przez kilka tysięcy nazw katalogu zamiast 80 klas COCO. Czasy X-CLIP stoją już w pracy, w tab. `e2-rozklad`. Osobno można też zmierzyć samą fazę zapytania dowolnej konfiguracji: `--config <plik> --skip-extraction` nie ładuje ani jednego modelu i wykonuje się w sekundy.
- **RetinaFace nie ma czasu zapytania** z innego powodu: detekcja twarzy jest wyłącznie ekstrakcją, a przy zapytaniu pracuje sygnał regionów.
- **"Łączenie sygnałów" nie ma ekstrakcji ani pamięci**, bo istnieje tylko przy zapytaniu.
- **Wiersz E4-D pochodzi z osobnego skryptu**, [`scripts/measure_e4d_cost.py`](../scripts/measure_e4d_cost.py), i stoi w [instrukcji fazy testowej](05_instrukcja_test.md). Musi biec sam.


## 11. Liczby do rozdziału 6

```
notebooks/experiments/dev_results.ipynb
```

## 12. Zamrożenie potoku PEŁNEGO

Skład potoku PEŁNEGO powstaje z [`configs/frozen.yaml`](../configs/frozen.yaml): po wpisaniu werdyktów E3, E4 i E5 generator zapisuje [`configs/full_office.yaml`](../configs/full_office.yaml), [`full_tbbt.yaml`](../configs/full_tbbt.yaml) i [`full_vatex.yaml`](../configs/full_vatex.yaml) razem z konfiguracjami wkładu krańcowego i efektu podmiany (do tego czasu wypisuje je jako czekające na `caption`, `objects` i `face_regions`). Skład: segmentacja i baza z werdyktów E1 i E2, opisy i obiekty z werdyktów E3 i E4, mechanizm twarzy z werdyktu E5, tożsamość włączona. Dla zbioru VATEX skład jest węższy: odpada tożsamość, bo nie ma powracających postaci z profilami, i odpada sygnał twarzy, bo plik znaczników tego zbioru nie ma kolumny `wymaga_mimiki`, wobec której dałoby się odczytać wynik. Ruch (SlowFast) też wchodzi do składu - X-CLIP przegrał E2, więc baza jest obrazowa i nie widzi ruchu, a bez tego sygnału potok nie miałby go wcale. Sprawdź wygenerowane pliki i zatwierdź stan repozytorium. Od tej chwili pliki konfiguracyjne nie zmieniają się już wcale: fazę testową uruchamia się na tych samych, nietkniętych plikach, zmienia się wyłącznie flaga `--split`.

