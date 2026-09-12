# Specyfikacja potoku

Kompletny opis potoku wyszukiwania nagrań wideo: jak jest zbudowany, skąd biorą się jego dane i jak przejść całą drogę od pustego repozytorium do liczb w rozdziale 6.

Dokument opisuje **stan faktyczny kodu**, nie plan i nie historię zmian. Jeśli coś tu nie zgadza się z repozytorium, to jest błąd - jednego albo drugiego.

## Rozstrzygnięcia fazy deweloperskiej

Pięć porównań, każde zamrażające jedną decyzję konfiguracyjną. Wynik zapisany jest w [`configs/frozen.yaml`](../configs/frozen.yaml) i stamtąd generator rozpisuje go na 57 konfiguracji.

| Eksp. | Decyzja | Warianty | Zwycięzca |
|---|---|---|---|
| E1 | strategia segmentacji | stałe okna · histogram · TransNetV2 | **histogram** (E1-B) |
| E2 | reprezentacja bazowa | OC ViT-H/14 · OC ViT-B/32 (kontrola) · X-CLIP | **OC ViT-H/14** (E2-A) |
| E3 | generator opisów | BLIP wobec LLaVA-1.5 | **LLaVA-1.5** (E3-C) |
| E4 | detektor obiektów | YOLO11 wobec YOLOE-11 | **YOLO11** (E4-B) |
| E5 | mechanizm twarzy | regiony twarzy wobec HSEmotion | **regiony** (E5-B) |

Poza werdyktami zamrożony jest też **próg dopasowania frazy do nazwy klasy**, osobny dla każdego z czterech słowników.

## Spis treści

0. **[Środowisko i modele pobierane ręcznie](00_srodowisko.md)** - co trzeba dociągnąć poza `conda env create`. Pierwszy krok, zanim cokolwiek ruszy.
1. **[Architektura](01_architektura.md)** - jak zbudowany jest potok: fazy, artefakty, sygnały, most między zapytaniem a nazwami klas, fuzja, reguła decyzji.
2. **[Dane i znaczniki](02_dane_i_znaczniki.md)** - wejście, którego przebieg nie wytworzy, znaczniki wymagań i profile postaci.
3. **[Warstwa adnotacji](03_warstwa_adnotacji.md)** - droga od opisów źródłowych do plików zapytań, krok po kroku, dla obu seriali i dla VATEX.
4. **[Instrukcja: faza deweloperska](04_instrukcja_dev.md)** - dwanaście kroków od adnotacji przez pomiar progu do zamrożenia potoku PEŁNEGO.
5. **[Instrukcja: faza testowa](05_instrukcja_test.md)** - od zamrożonej konfiguracji przez przebiegi do liczb w rozdziale 6.
6. **[Wykaz plików](06_wykaz_plikow.md)** - wszystkie moduły `.py` z opisem, plus konfiguracje, notatniki i testy.
