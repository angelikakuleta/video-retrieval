# Architektura

Zasada organizująca całość: **nic drogiego nie zależy od strategii segmentacji**. Ekstrakcja idzie po wspólnej siatce klatek 1,25 s i jest buforowana per odcinek; kolekcja fragmentów powstaje przez agregację. Rozstrzygnięcie E1 nie unieważnia więc opisów, detekcji ani twarzy.

## 1.1 Dwie fazy i warstwa artefaktów

Faza przetwarzania wstępnego wykonuje się raz dla kolekcji; faza obsługi zapytania czyta gotowe artefakty. Podział na indeksy i katalogi odwzorowuje rozdział 4: **indeksy FAISS** dla tego, co przeszukuje wektor zapytania (fragmenty, opisy scen, wycinki twarzy), **katalogi** dla danych adresowanych identyfikatorem fragmentu, **bufor** dla wyrównanych wycinków twarzy.

```
data/cache/
  black/<zbiór>_black.csv                        przedziały czarnego obrazu
  segmentation/<strategia>/<zbiór>_segments.csv  granice fragmentów
  embeddings/<model>/<zbiór>/<odcinek>.npz       embeddingi klatek siatki
  captions/<generator>/<zbiór>/<odcinek>.jsonl   opisy klatek (tekst)
  captions/<generator>/<enkoder>/...npz            embeddingi opisów
  objects/<detektor>/<zbiór>/<odcinek>.npz       (klatka, klasa, pewność)
  objects/<detektor>/vocab.json                  słownik nazw klas
  faces/crops/<zbiór>/<odcinek>.npz              bufor wycinków 112x112
  faces/regions/<enkoder>/...npz                   embeddingi wycinków
  faces/hsemotion/<zbiór>/<odcinek>.npz          rozkłady 8 klas emocji
  faces/arcface/<zbiór>/<odcinek>.npz            wektory tożsamości
  faces/profiles/<zbiór>.npz                     profile postaci
  motion/<strategia>/<zbiór>/<odcinek>.npz       rozkłady 400 klas czynności
  vocab/<enkoder>/<nazwa>.npz                    embeddingi nazw klas
  indexes/<zbiór>/<strategia>/<model>/<split>/   FAISS: sceny
      scene.faiss, catalog.json
      prompt_features.npy                        cechy klatek fragmentu (X-CLIP)
```

Dwie własności tego układu zmieniają sposób pracy. **Przebieg rusza tylko odcinki ocenianego splitu** - `--split dev` nie zdekoduje ani jednego nagrania testowego, więc fazę deweloperską da się przejść, zanim adnotacje testowe będą gotowe. I **przebieg dolicza braki sam**: czarne klatki, segmenty, embeddingi, opisy, detekcje, twarze, indeksy - wszystko poza wejściem z [02_dane_i_znaczniki.md](02_dane_i_znaczniki.md).

## 1.2 Ekstraktory

Jeden moduł na komponent, wspólny kontrakt *odcinek -> plik cache*, wznawialny per odcinek. Modele ładowane rozłącznie, bo 16 GB pamięci karty nie mieści dwóch.

*src/features - komponenty i ich stan weryfikacji*

| Moduł | Model | Wynik | Sprawdzone |
|---|---|---|---|
| openclip.py | ViT-H/14 `laion2b_s32b_b79k`; ViT-B/32 `openai` | embeddingi obrazów i tekstu | (przebieg) |
| xclip.py | `microsoft/xclip-base-patch32` | embedding wideo, cechy klatek, podpowiedzi | (1,3e-7) |
| captions.py | BLIP large, LLaVA-1.5-7B | jeden opis na klatkę siatki | (BLIP) |
| objects.py | `yolo11l.pt`, `yoloe-11l-seg-pf.pt` | klasy + pewności, próg 0,25 | (oba) |
| faces.py | RetinaFace-R50 ONNX + `norm_crop` | ramki, punkty, wycinki 112x112 | (wyrównanie) |
| expressions.py | HSEmotion EfficientNet-B0 | rozkład 8 klas na wycinek | (rozkład) |
| regions.py | enkoder wizualny konfiguracji | embeddingi wycinków twarzy | (testy jedn.) |
| identity.py | ArcFace R100 `glintr100` | wektory tożsamości, profile postaci | (testy jedn.) |
| motion.py | SlowFast R50 8x8 | rozkład 400 klas na fragment | (testy jedn.) |
| text_vocab.py | enkoder tekstu konfiguracji | embeddingi nazw klas, buforowane | (testy jedn.) |
| encoders.py | - | wybór enkodera konfiguracji, leniwy | (przebieg) |

Wszystkie ekstraktory klatkowe czytają **tę samą siatkę** i strumieniują ją partiami - odcinek to około tysiąca klatek pełnej rozdzielczości, których nie da się trzymać naraz. Który z policzonych wyników trafia do sygnału, rozstrzyga dopiero oś treści przy agregacji: klatka w masce przejścia albo w ciemnym odcinku należy do żadnego fragmentu. Dzięki temu przerysowanie maski nic nie kosztuje.

Okna modeli sekwencyjnych - X-CLIP i SlowFast - odmierzane są na tej samej osi: klatki okna rozkładają się równomiernie po czasie treści, więc przeskakują nad przerywnikiem zamiast go próbkować. Okno obejmujące animację dostaje materiał z obu jej stron, sklejony. **Sklejenie jest przyjęte świadomie**, żeby pokryć cały materiał: przerywnik trwa ok. 1,3 s przy oknie 2,7 s, więc odrzucanie takich okien kosztowałoby całą ocenę ruchu za ułamek długości okna. Oba modele robią to identycznie - to samo wywołanie `Timeline.sample_content` - więc porównanie w E2 dotyczy sposobu reprezentowania ruchu, a nie tego, co model zobaczył.

## 1.3 Sygnały

Wspólny kontrakt niesie oba przypadki z sekcji `nieaktywnosc` rozdziału 4: `NaN` w komórce oznacza brak danych dla fragmentu, a `active()` - nieaktywność wobec całego zapytania.

```
class Signal(Protocol):
    name: str
    def raw_values(self, queries: list[Query],
                   phrases: dict[str, list[Phrase]] | None = None) -> np.ndarray:
    def active(self, query: Query) -> bool:                     # nieaktywność per zapytanie
```

### Most między zapytaniem a nazwami klas

Sygnał o zamkniętym słowniku - obiektowy, mimiki i ruchu - **nie porównuje całego zapytania** z nazwami klas. Porównywana jest fraza, a droga od zapytania do fraz ma trzy niezależne reguły ([`src/retrieval/phrases.py`](../src/retrieval/phrases.py)): przedmiotową, mimiczną i czynnościową. Rodzaje nie wykluczają się wzajemnie - `laugh` jest zarazem frazą czynnościową (Kinetics ma klasę `laughing`) i mimiczną.

Dopasowanie frazy do nazwy klasy ([`src/retrieval/vocab_match.py`](../src/retrieval/vocab_match.py)) liczy się tylko wtedy, gdy **osiąga próg** τ, wyznaczany osobno dla każdego słownika na parach ocenionych ręcznie ([04_instrukcja_dev.md](04_instrukcja_dev.md), krok 8). Poniżej progu sygnał nie ma o tym zapytaniu nic do powiedzenia i pozostaje nieaktywny - a nieaktywność znaczy "ten sygnał nie bierze udziału", nie "ten sygnał mówi zero".

Dlaczego to nie jest szczegół implementacyjny: przy porównywaniu embeddingu **całego zdania** z listą nazw klas jakaś nazwa zawsze wygrywa, także wtedy, gdy słownik nie zawiera ani jednego wyrazu z zapytania - dla Kinetics-400 fraza `walk` lądowała na klasie `writing` 72 razy na 72. Sygnał, który zawsze coś mówi, przy wagach jednostajnych `1/|A(q)|` zawsze zabiera połowę wagi bazie, także wtedy, gdy nie ma nic prawdziwego do powiedzenia. Próg przywraca sygnałowi prawo do milczenia.

Frazy wydobywa się **raz na zapytanie**, w `Pipeline.signal_matrices`, i przekazuje sygnałom gotowe listy. Oszczędza to trzy parsowania spaCy na zapytanie i gwarantuje konstrukcyjnie, że E5-B i E5-C widzą tę samą listę fraz mimicznych - czego wymaga porównanie w E5.

*Trzy kształty sygnału*

| Sygnał | Agregacja do fragmentu | NaN, gdy |
|---|---|---|
| porównanie z jednym wektorem fragmentu |
| sceniczny (OC) | podobieństwo cosinusowe do wektora fragmentu | nigdy |
| sceniczny (XC) | dopasowanie z podpowiedziami, warunkowane cechami fragmentu | nigdy |
| najlepsze dopasowanie wśród przedmiotów fragmentu |
| opisowy | **maksimum** po opisach klatek | brak klatek z opisem |
| obiektowy | **maksimum** po nazwach wykrytych klas | brak detekcji ≥ 0,25 |
| regionów | **maksimum** po wycinkach twarzy | brak twarzy |
| wartość oczekiwana względem rozkładu klasyfikatora |
| mimiki | wzór (5) na 8 klasach dla twarzy, potem **maksimum** po twarzach | brak twarzy |
| ruchu | wzór (5) na pełnym rozkładzie 400 klas | nigdy |
| koniunkcja |
| tożsamości (test) | maksimum po twarzach dla każdej postaci, potem **minimum** po postaciach | brak twarzy |

Kolejność działań we wzorze (5) jest istotna i pokryta testem: podobieństwo liczone jest najpierw do nazwy każdej klasy z osobna, a dopiero potem uśredniane wagami prawdopodobieństw znormalizowanych w obrębie zbioru `T(f)`. Uśrednienie wektorów klas przed porównaniem dałoby co innego - test pokazuje obie liczby obok siebie.

## 1.4 Standaryzacja, wartość neutralna, wagi

[`src/retrieval/fusion.py`](../src/retrieval/fusion.py) to cała sekcja `standaryzacja` rozdziału 4 w jednym miejscu, i nic powyżej nie ma prawa jej powtarzać:

- dla każdego zapytania i sygnału `μ`, `σ` po fragmentach **z danymi**,
- `σ < 10⁻⁶` -> sygnał nierozróżniający, `z = 0` dla całego zapytania,
- fragment bez danych -> `z = 0`, czyli standaryzowana średnia kolekcji - wartość neutralna, nie minimalna,
- zbiór aktywnych `A(q)` ustalany per zapytanie, wagi `1/|A(q)|`,
- `S(q,f) = Σ w_j · z_j(q,f)`.

Macierz `z` przed sumowaniem trafia do katalogu przebiegu jako `signals.npz` (float16). To realizuje zapowiedź z sekcji `wrazliwosc-wag`: przeliczenie punktu siatki wag jest ponownym ważeniem zapisanych sygnałów, a nie kolejnym przejściem po nagraniach.

## 1.5 Ewaluacja i reguła decyzji

Jednostką oceny jest **zdarzenie**, więc Recall@K jest binarny, a rekordy o wspólnym `event_id` dzielą sumę swoich fragmentów poprawnych. Obok raportowane są odczyty fragmentowe (`recall_full@K`, `mAP_full`) - różnią się dokładnie wtedy, gdy zdarzenie zajmuje więcej niż jeden fragment.

Jednostką wnioskowania jest **odcinek** dla seriali (przedział t po odcinkach-klastrach) i **klip** dla VATEX (bootstrap, 10 000 replikacji) - VATEX nigdy nie wchodzi do puli. Wariant wygrywa, gdy kierunek różnicy zgadza się w obu serialach *i* przedział na ich połączonych odcinkach nie obejmuje zera; w przeciwnym razie wygrywa wariant prostszy. Werdykt policzony na jednym serialu jest oznaczany jako **prowizoryczny**.

Kontrast konfirmacyjny to **różnica różnic** z własnym przedziałem: różnica na zapytaniach ze znacznikiem minus różnica na pozostałych, liczona po odcinkach; odcinek bez zapytań po którejkolwiek stronie odpada.
