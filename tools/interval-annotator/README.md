# Interval Annotator

Lokalne narzędzie przeglądarkowe do adnotowania odcinka. Bez budowania, bez zależności, działa z `file://`. Otwiera się `index.html`, wskazuje plik MP4 i zapisuje wynik jako `<odcinek>_intervals.csv`.

Zapisywane są dwa rodzaje przedziałów:

- **adnotacje** (`type = event`) - początek i koniec z dokładnością 0,01 s plus opis. Mogą na siebie nachodzić. Opis ma mieć co najmniej 8 słów, a długość musi mieścić się w granicach z nagłówka;
- **maski** (`type = mask`) - odcinki wycinane z przetwarzania. Opis jest opcjonalny, długość niesprawdzana, ale **maski nie mogą na siebie nachodzić**: potok czyta je jako podział odcinka, więc dwie maski na tych samych sekundach zostawiłyby nierozstrzygnięte, czy ten fragment wycina się raz, czy dwa razy. Takiej edycji narzędzie odmawia i odmawia też eksportu pliku.

Maski są domyślnie zablokowane; odznaczenie **Lock masks** odsłania przycisk **Add mask**, wybór klasy i edycję wierszy masek. Klasa maski to znacznik na wierszu: pusty dla strukturalnej (logo, czołówka, napisy - tnie odcinek na zakresy korpusu) albo `przejscie` dla animacji między scenami, którą wchłania oś treści i która żadnego zakresu nie tnie.

Adnotacja **na masce** jest czym innym i celowo nie jest odrzucana: znacznik zaimportowany z TVR często zaczyna się chwilę przed tym, jak akcja staje się widoczna, więc bywa, że zaczyna się w przejściu. Taki wiersz ma dać się otworzyć i poprawić. Dodanie *nowej* adnotacji na masce jest już odrzucane, a wiersze, których to dotyczy, są oznaczone i policzone w liczniku **on a mask**.

## Sterowanie

| klawisz | działanie |
| --- | --- |
| `Space` | odtwarzanie / pauza |
| `←` `→` | ± 1 s |
| `Shift` + `←` `→` | ± 0,1 s |
| `Ctrl` + `←` `→` | ± 1 klatka (wg pola FPS) |
| `S` / `E` | ustaw początek / koniec na bieżącym czasie |
| `Ctrl` + `Enter` | dodaj adnotację |

Przyciski wiersza: `▶` odtwórz przedział, `[` `]` ustaw początek / koniec z bieżącego czasu, `✕` usuń.

## Przedziały półotwarte

Przedziały są **półotwarte: `[start, end)`**. `start` to czas pierwszej klatki, na której zdarzenie widać, `end` - pierwszej, na której już go nie ma. Dzięki temu czas trwania to po prostu `end - start`, tak samo jak w reszcie potoku (stałe okna `[0, 10)`, `[10, 20)`).

Stąd zdarzenie nie nachodzi na maskę, gdy `event.end <= mask.start` albo `event.start >= mask.end`: **stykające się granice są dozwolone**, bo przedziały półotwarte nie dzielą żadnej klatki. Tak samo dwie sąsiadujące maski.

## Granice długości

Para pól **Duration [s]** w nagłówku mówi, jak długo może trwać adnotacja. Domyślne wartości idą za potokiem: **1,25 s** to krok próbkowania klatek (nic krótszego nie da się trafić żadną próbkowaną klatką), **15 s** to granica, powyżej której opisane zdarzenie rozkłada się na kilka fragmentów i "poprawna odpowiedź" przestaje być jednym fragmentem.

Dodania adnotacji poza zakresem narzędzie odmawia. Wiersz, który wyjdzie poza zakres przy edycji albo przyjdzie taki z pliku, nie jest usuwany - zostaje oznaczony i policzony w liczniku **outside duration**, a decyzję podejmuje człowiek. Maski nie są sprawdzane.

Te same dwa progi żyją w notatnikach jako `MIN_DURATION` i `MAX_DURATION` ([`tbbt_02_annotation_candidates.ipynb`](../../notebooks/prepare_data/tbbt_02_annotation_candidates.ipynb) na nich odrzuca, [`office_02_annotations.ipynb`](../../notebooks/prepare_data/office_02_annotations.ipynb) ostrzega). Zmiana musi wejść w obu miejscach naraz, inaczej notatnik wyda werdykt, którego narzędzie nie wydało.

## Plik CSV

Eksport CSV jest właściwym zapisem; autozapis w `localStorage` to tylko zabezpieczenie na wypadek awarii.

Wiersze zapisywane są **zawsze w kolejności czasu** (po `start`, przy remisie krótszy przedział pierwszy), niezależnie od tego, czy kliknięto kiedykolwiek **Sort by start**. Strona pythonowa pisze tę samą kolejność, więc plik, który przeszedł przez adnotator, różni się od wejściowego wyłącznie edycjami, nigdy kolejnością wierszy.

Format: `id;type;start;end;desc;tags`, separator `;`, kropka dziesiętna, UTF-8 z BOM. Nazwa pliku: `<odcinek>_intervals.csv`.

```
id;type;start;end;desc;tags
office_s02e03_001;event;506.21;512.15;description one two three;wymaga_osoby
office_s02e03_m001;mask;0.00;31.40;;
office_s02e03_m002;mask;1287.55;1312.90;credits;
```

Pliki bez kolumny `type` importują się jako adnotacje, bez kolumny `tags` - bez znaczników.

**Export CSV pyta, gdzie zapisać**, i otwiera katalog użyty ostatnio, co przy jednym pliku na odcinek jest całym sensem. Wymaga to File System Access API: mają je Chrome i Edge, nie mają Firefox i Safari, a Chrome udostępnia je tylko stronie serwowanej po http. Wszędzie indziej eksport spada do zwykłego pobrania do katalogu pobierania i pasek stanu to mówi. Żeby dostać okno dialogowe przy pracy z dysku, wystaw katalog zamiast otwierać plik:

```powershell
.\serve.ps1
```

## Znaczniki

Każdy wiersz może nieść dowolne znaczniki; narzędzie nie przypisuje żadnemu z nich znaczenia, więc `wymaga_osoby` nie jest dla niego niczym szczególnym. Znacznik jest jednym tokenem: spacje zamieniają się na `_`, a `;` `,` `"` są usuwane, bo to składnia CSV.

Słownik znaczników mieszka w `localStorage` pod kluczem `interval-annotator:tags` i jest wspólny dla wszystkich nagrań - dzięki temu podpowiedzi działają między odcinkami. Przycisk **Tags** w nagłówku otwiera panel, w którym można dodać znacznik z góry, zobaczyć, ile wierszy go niesie, zmienić jego nazwę wszędzie naraz albo usunąć go ze słownika i ze wszystkich wierszy.

## Identyfikatory

Ostatnia część identyfikatora mówi, skąd wiersz pochodzi; tabela pokazuje ją pod numerem wiersza, a pełny identyfikator jest w podpowiedzi komórki:

| postać | znaczenie | przykład |
| --- | --- | --- |
| `<odcinek>_d<numer>` | zaimportowany, `<numer>` to identyfikator w zbiorze źródłowym | `tbbt_s01e01_d97650` |
| `<odcinek>_NNN` | adnotacja utworzona w tym narzędziu | `office_s02e03_001` |
| `<odcinek>_mNNN` | maska | `office_s02e03_m001` |

**Identyfikator wczytany z pliku nigdy się nie zmienia.** Własny identyfikator dostaje tylko wiersz, który przyszedł bez niego. To na tym stoi cała droga tam i z powrotem: adnotacje serialu powstają gdzie indziej, weryfikuje się je tutaj i czyta z powrotem, a skoro identyfikatory przeżywają, to czego brakuje w pliku wracającym, zostało usunięte celowo. Wiersz o identyfikatorze już obecnym na liście jest więc pomijany i zgłaszany, a nigdy nie dostaje przyrostka; eksport ostrzega przed zapisaniem pliku z powtórzonymi albo brakującymi identyfikatorami.

Import na niepustą listę proponuje jej **zastąpienie**, bo zaimportowanie tego samego pliku dwa razy jest częstym przypadkiem, a dopisanie podwoiłoby wszystkie wiersze. Anulowanie pytania dopisuje wiersze zamiast zastępować.

## Pliki

```
index.html        struktura strony
css/styles.css    style (zmienne na gorze pliku)
serve.ps1         wystawia katalog po http (potrzebne do okna zapisu)
js/time.js        zaokraglanie czasu i format m:ss.cc
js/csv.js         zapis i odczyt CSV (okno zapisu, pobranie awaryjne)
js/store.js       model zdarzen, identyfikatory, reguly nachodzenia i masek, autozapis
js/player.js      krokowanie wideo i odtwarzanie przedzialu w petli
js/app.js         spiecie z DOM: formularz, tabela, klawiatura, granice, import/eksport
```
