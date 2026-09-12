# judge_pairs - ocena par fraza / nazwa klasy

Konsolowy program do ręcznej oceny. Pokazuje po kolei pary: **frazę** wyjętą z jednego zdania i **nazwę klasy** ze słownika komponentu, i na każdą zadaje to samo pytanie:

> Czy ta nazwa klasy jest poprawnym odpowiednikiem tej frazy w tym zdaniu?

Z odpowiedzi liczy się próg pewności dopasowania (`scripts/measure_threshold.py`), czyli najniższą pewność, przy której co najmniej 90% przyjętych par jest poprawnych.

## Uruchomienie

```
python tools/judge/judge_pairs.py --samples data/interim/threshold/threshold_sample_*.csv --output data/interim/threshold/threshold_judgments.csv
```

Obie ścieżki są wymagane i nie mają wartości domyślnych: co się ocenia i gdzie lądują odpowiedzi to dwie rzeczy, których program nie ma prawa zgadywać. Próby tworzy `scripts/make_threshold_samples.py`.

| klawisz | działanie |
|---|---|
| `Y` | tak |
| `N` | nie |
| `S` | pomiń, para wróci na koniec kolejki |
| `P` | cofnij do poprzedniej pary |
| `Q` | zapisz i wyjdź |

Plik odpowiedzi jest przepisywany po każdym naciśnięciu, więc zamknięcie okna niczego nie kasuje, a ponowne uruchomienie wraca do pierwszej nieocenionej pary. Pominięcie nie jest odpowiedzią: para pominięta do końca nie wchodzi do wyniku, a jej udział raportowany jest osobno.

## Reguła oceny

Przypadki, które wracają najczęściej:

| para | odpowiedź | dlaczego |
|---|---|---|
| `coffee mug` / `cup` | `Y` | kategoria ogólniejsza obejmuje frazę |
| `guitar` / `playing bass guitar` | `Y` | kategoria szczegółowsza, fraza jej nie wyklucza |
| `acoustic guitar` / `playing bass guitar` | `N` | fraza wyklucza bas |
| `desk` / `dining table` | `N` | rzecz pokrewna, ale inna |
| `hand` / `person` | `N` | rzecz występująca obok, nie odpowiednik |
| `guitar` / `playing guitar` | `N` | przedmiot wobec czynności |
| `shot` / `cup` | zależy od zdania | w zdaniu o barze `Y`, w zdaniu o kamerze `N` |

Brak jakiegokolwiek odpowiednika w słowniku to `N`. Ocenia się sam tekst, nie nagranie, i nie wraca się do ocenionych par, żeby je ujednolicić; `P` służy do poprawienia pomyłki, nie do rewizji.
