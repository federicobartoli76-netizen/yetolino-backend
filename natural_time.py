import re
from datetime import date, timedelta
from typing import Optional

# Mappa dei giorni della settimana in italiano -> indice weekday (0 = lunedì, ... 6 = domenica)
WEEKDAY_MAP = {
    "lunedi": 0,
    "lunedì": 0,
    "martedi": 1,
    "martedì": 1,
    "mercoledi": 2,
    "mercoledì": 2,
    "giovedi": 3,
    "giovedì": 3,
    "venerdi": 4,
    "venerdì": 4,
    "sabato": 5,
    "domenica": 6,
}

def _normalize(text: str) -> str:
    return text.lower().strip()

def _next_weekday(from_day: date, target_weekday: int, include_today: bool = False) -> date:
    """
    Restituisce la prossima data che ha il weekday richiesto.
    target_weekday: 0 = lunedì, ..., 6 = domenica
    """
    current_weekday = from_day.weekday()
    delta = (target_weekday - current_weekday) % 7
    if delta == 0 and not include_today:
        delta = 7
    return from_day + timedelta(days=delta)

def parse_natural_date_it(text: str, today: Optional[date] = None) -> Optional[date]:
    """
    Cerca di estrarre una data "naturale" in italiano dal testo.
    Esempi che riconosce:
    - oggi, domani, dopodomani
    - ieri, l'altro ieri
    - tra 3 giorni / fra 2 giorni
    - lunedì, martedì prossimo, sabato prossimo
    - questo weekend, settimana prossima

    Ritorna:
        date -> se trova qualcosa
        None -> se non riconosce nessuna data
    """
    if today is None:
        today = date.today()

    t = _normalize(text)

    # --- 1. RIFERIMENTI SEMPLICI ---
    if "oggi" in t:
        return today

    if "dopodomani" in t:
        return today + timedelta(days=2)

    # importante: controlliamo "ieri l'altro" PRIMA di "ieri"
    if "ieri l'altro" in t or "l'altro ieri" in t:
        return today - timedelta(days=2)

    if "ieri" in t:
        return today - timedelta(days=1)

    if "domani" in t:
        return today + timedelta(days=1)

    # --- 2. TRA / FRA X GIORNI ---
    # es: "tra 3 giorni", "fra 2 giorni"
    match_days = re.search(r"(tra|fra)\s+(\d+)\s+giorni?", t)
    if match_days:
        n_days = int(match_days.group(2))
        return today + timedelta(days=n_days)

    # --- 3. NOMI DEI GIORNI DELLA SETTIMANA ---
    # es: "lunedì", "martedì prossimo", "giovedì", "sabato prossimo"
    for name, weekday_index in WEEKDAY_MAP.items():
        if name in t:
            # se c'è "prossimo" o "prossima"
            if "prossimo" in t or "prossima" in t:
                # saltiamo alla settimana prossima
                base = today + timedelta(days=7)
                return _next_weekday(base, weekday_index, include_today=True)
            else:
                # giorno di questa settimana o prossima se già passato
                candidate = _next_weekday(today, weekday_index, include_today=True)
                if candidate < today:
                    candidate = candidate + timedelta(days=7)
                return candidate

    # --- 4. QUESTO WEEKEND / SETTIMANA PROSSIMA ---
    if "weekend" in t or "fine settimana" in t:
        # prendiamo il sabato utile
        saturday = _next_weekday(today, 5, include_today=True)  # 5 = sabato
        if saturday < today:
            saturday = saturday + timedelta(days=7)
        return saturday

    if "settimana prossima" in t or "prossima settimana" in t:
        # lunedì della prossima settimana
        monday_this_week = today - timedelta(days=today.weekday())  # lunedì corrente
        monday_next_week = monday_this_week + timedelta(days=7)
        return monday_next_week

    # Se non riconosce niente:
    return None
