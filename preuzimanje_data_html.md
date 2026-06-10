Otvori terminal u korijenu projekta i pokreni:

# Kreiraj data mapu ako ne postoji
mkdir -p data

# Preuzmi Wikipedia HTML datoteke
wget -O data/f1_car.html "https://en.wikipedia.org/wiki/Formula_One_car"
wget -O data/f1_2026_season.html "https://en.wikipedia.org/wiki/2026_Formula_One_World_Championship"
wget -O data/power_unit.html "https://en.wikipedia.org/wiki/Formula_One_engines"
wget -O data/safety_car.html "https://en.wikipedia.org/wiki/Safety_car"

(Napomena: FIA PDF-ove u mapu data/2026/ moraš ručno preuzeti sa službene FIA stranice jer zahtijevaju specifične URL-ove ili registraciju, ali ovi HTML-ovi su ti sada spremni za ingest.py)