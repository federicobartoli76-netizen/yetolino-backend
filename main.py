from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Dict, Any, Optional, List

import json
import os
import math
import random
import time
import re
import requests

from datetime import datetime, date, timedelta
from pathlib import Path

# ---------------- LOAD .ENV ---------------- #
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from openai import OpenAI


# ---------------- APP & CLIENT ---------------- #

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?|http://192\.168\.3\.8(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# memoria breve: ultimi messaggi della chat
CHAT_HISTORY: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY = 12  # messaggi totali (user + assistant)

# ---------------- PATH STABILE MEMORY ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")

# Usa la chiave da variabile d'ambiente OPENAI_API_KEY
client = OpenAI()

# Limite massimo caratteri di memorie da passare al modello (per non rallentare troppo)
MAX_MEMORY_CHARS = 3000

# Pattern per rimuovere emoji dal testo da dare al TTS
EMOJI_PATTERN = re.compile("[\U0001F300-\U0001FAFF\U00002700-\U000027BF]+")
LAUGH_PATTERN = re.compile(r"\b(ah|eh|ha|ih|oh){2,}\b", re.IGNORECASE)


# ---------------- UTILS VARI ---------------- #

def strip_accents(s: str) -> str:
    return (
        s.replace("à", "a")
         .replace("è", "e")
         .replace("é", "e")
         .replace("ì", "i")
         .replace("ò", "o")
         .replace("ó", "o")
         .replace("ù", "u")
    )


def clean_reminder_text(content: str) -> str:
    t = content.strip(" .!?")

    for prefix in ["domani ", "oggi "]:
        if t.startswith(prefix):
            t = t[len(prefix):]

    for prefix in ["che devo ", "che devo", "che "]:
        if t.startswith(prefix):
            t = t[len(prefix):]

    if t.startswith("devo "):
        t = t[len("devo "):]

    return t.strip()


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------------- BASE MEMORY UTILS ---------------- #

def load_memory() -> Dict[str, Any]:
    if not os.path.exists(MEMORY_FILE):
        return {
            "memories": [],
            "mood_log": [],
            "reminders": [],
            "meta": {},
            "profiles": {},
        }

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {
            "memories": [],
            "mood_log": [],
            "reminders": [],
            "meta": {},
            "profiles": {},
        }

    if "memories" not in data or not isinstance(data["memories"], list):
        data["memories"] = []
    if "mood_log" not in data or not isinstance(data["mood_log"], list):
        data["mood_log"] = []
    if "reminders" not in data or not isinstance(data["reminders"], list):
        data["reminders"] = []
    if "meta" not in data or not isinstance(data["meta"], dict):
        data["meta"] = {}
    if "profiles" not in data or not isinstance(data["profiles"], dict):
        data["profiles"] = {}

    # MIGRAZIONE reminder vecchi
    for r in data["reminders"]:
        if isinstance(r, dict) and "user_id" not in r:
            r["user_id"] = "fede"

    return data


def save_memory(mem: Dict[str, Any]) -> None:
    if "memories" not in mem or not isinstance(mem["memories"], list):
        mem["memories"] = []
    if "mood_log" not in mem or not isinstance(mem["mood_log"], list):
        mem["mood_log"] = []
    if "reminders" not in mem or not isinstance(mem["reminders"], list):
        mem["reminders"] = []
    if "meta" not in mem or not isinstance(mem["meta"], dict):
        mem["meta"] = {}
    if "profiles" not in mem or not isinstance(mem["profiles"], dict):
        mem["profiles"] = {}

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)


def classify_memory(text: str) -> Dict[str, Any]:
    low = text.lower()
    category = "generale"
    key = "generale"

    if any(k in low for k in ["eri", "mia moglie", "moglie", "famiglia"]):
        category = "famiglia"
        key = "eri" if "eri" in low else "famiglia"
    elif any(k in low for k in ["moto", "vespa", "husqvarna", "ktm", "minicooper", "mini cooper"]):
        category = "moto"
        key = "moto_fede"
    elif any(k in low for k in ["btc", "bitcoin", "eth", "ether", "cripto", "crypto", "trading", "long", "short"]):
        category = "crypto"
        key = "trading_fede"
    elif any(k in low for k in ["itoshima", "fukuoka", "giappone", "japan", "casa", "mare"]):
        category = "luoghi"
        key = "luoghi_fede"

    return {
        "category": category,
        "key": key,
        "info": text.strip(),
        "created_at": datetime.utcnow().isoformat(),
    }


# ---------------- EMBEDDINGS & SIMILARITÀ ---------------- #

def get_embedding(text: str) -> List[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------- SHORT-TERM HISTORY ---------------- #

def add_to_history(user_id: str, role: str, content: str) -> None:
    if user_id not in CHAT_HISTORY:
        CHAT_HISTORY[user_id] = []
    CHAT_HISTORY[user_id].append({"role": role, "content": content})
    if len(CHAT_HISTORY[user_id]) > MAX_HISTORY:
        CHAT_HISTORY[user_id] = CHAT_HISTORY[user_id][-MAX_HISTORY:]


# ---------------- MEMORY AVANZATA ---------------- #

def add_memory(text: str) -> str:
    mem = load_memory()
    mem.setdefault("memories", [])
    normalized = text.strip().lower()

    for m in mem["memories"]:
        if isinstance(m, dict):
            existing_info = m.get("info", "")
            if existing_info and existing_info.strip().lower() == normalized:
                return "Ok Fede, questa cosa la sapevo già 😉"

    try:
        new_emb = get_embedding(text)
    except Exception:
        classified = classify_memory(text)
        classified["embedding"] = None
        mem["memories"].append(classified)
        save_memory(mem)
        return f"Perfetto Fede, lo segno nella categoria '{classified['category']}'."

    SIM_THRESHOLD = 0.88
    for m in mem["memories"]:
        if not isinstance(m, dict):
            continue
        existing_emb = m.get("embedding")
        if not existing_emb:
            continue
        try:
            sim = cosine_similarity(new_emb, existing_emb)
        except Exception:
            continue
        if sim >= SIM_THRESHOLD:
            return "Ok Fede, mi stai dicendo più o meno la stessa cosa che so già 😉"

    classified = classify_memory(text)
    classified["embedding"] = new_emb
    mem["memories"].append(classified)
    save_memory(mem)
    return f"Perfetto Fede, lo segno nella categoria '{classified['category']}'."


def memory_text() -> str:
    mem = load_memory()
    memories = mem.get("memories", [])
    if not memories:
        return "Per ora non ho memorie salvate."

    by_cat: Dict[str, List[str]] = {}
    for m in memories:
        if not isinstance(m, dict):
            continue
        cat = m.get("category", "generale")
        info = m.get("info", "")
        if not info:
            continue
        by_cat.setdefault(cat, []).append(info)

    lines: List[str] = []
    for cat, infos in by_cat.items():
        lines.append(f"{cat.capitalize()}:")
        for info in infos:
            lines.append(f"- {info}")
        lines.append("")

    full = "\n".join(lines).strip()
    if len(full) > MAX_MEMORY_CHARS:
        return full[-MAX_MEMORY_CHARS:]
    return full


def memories_for_key(key: str) -> str:
    mem = load_memory()
    memories = mem.get("memories", [])
    found: List[str] = []

    for m in memories:
        if not isinstance(m, dict):
            continue
        info = m.get("info", "")
        info_low = info.lower()
        mkey = m.get("key", "").lower()
        if key.lower() in mkey or key.lower() in info_low:
            found.append(info)

    if not found:
        return f"Per ora non ho memorie particolari su {key}."

    lines = [f"Ecco cosa mi ricordo su {key}:"]
    for info in found:
        lines.append(f"- {info}")
    return "\n".join(lines)


def to_second_person(text: str) -> str:
    replacements = [
        ("mi piace", "ti piace"),
        ("mi piacciono", "ti piacciono"),
        ("amo ", "ami "),
        ("adoro ", "adori "),
        ("odio ", "odi "),
        ("detesto ", "detesti "),
        ("preferisco", "preferisci"),
        ("sono fissato", "sei fissato"),
        ("sono fissato con", "sei fissato con"),
        ("sono abituato a", "sei abituato a"),
        ("vado spesso", "vai spesso"),
        ("di solito vado", "di solito vai"),
        ("mi piace andare", "ti piace andare"),
        ("mi piace guidare", "ti piace guidare"),
        ("la mia ", "la tua "),
        ("il mio ", "il tuo "),
        ("i miei ", "i tuoi "),
        ("le mie ", "le tue "),
    ]
    out = text
    for a, b in replacements:
        out = out.replace(a, b)
        out = out.replace(a.capitalize(), b.capitalize())
    return out


def memories_about_me() -> str:
    mem = load_memory()
    memories = mem.get("memories", [])
    found: List[str] = []

    for m in memories:
        if not isinstance(m, dict):
            continue

        info = (m.get("info") or "").strip()
        if not info:
            continue

        low = info.lower()

        if "eri" in low or "mia moglie" in low or "moglie" in low:
            continue
        if "gli piace" in low or "le piace" in low or "a lei piace" in low:
            continue

        if any(
            w in low
            for w in [
                "domani", "ieri", "oggi", "stasera", "stanotte",
                "sabato", "domenica", "lunedì", "martedì", "mercoledì",
                "giovedì", "venerdì",
                "ikea", "bolletta", "pagare",
                "devo ", "dobbiamo ", "ricordami"
            ]
        ):
            continue

        cut = info.split(".")[0]
        cut = cut.split(",")[0].strip()
        if not cut:
            continue

        cut = to_second_person(cut)
        found.append(cut)

    clean_list: List[str] = []
    seen = set()
    for f in found:
        if f not in seen:
            seen.add(f)
            clean_list.append(f)

    if not clean_list:
        return (
            "Per ora, come gusti e abitudini, non ho molto salvato su di te. "
            "Ogni tanto dimmi cosa ti piace e lo fisso pian piano 😉"
        )

    clean_list = clean_list[:5]
    lines = ["Ecco alcune cose che mi ricordo di te (come gusti e abitudini):"]
    for f in clean_list:
        lines.append(f"- {f}")
    return "\n".join(lines)


# -------- AUTO-MEMORIA DA FRASE NORMALE -------- #

def auto_memory_candidate(msg: str) -> Optional[str]:
    low = msg.lower().strip()
    if "?" in low:
        return None
    if len(low) < 10:
        return None

    patterns = [
        "mi piace", "amo ", "adoro ", "odio ", "detesto ", "preferisco",
        "eri ama", "eri adora", "mia moglie si chiama", "sono sposato",
        "la mia moto", "la mia vespa", "ho una husqvarna", "ho la husqvarna",
    ]

    for p in patterns:
        if p in low:
            return msg.strip()

    return None


# ---------------- REMINDERS (CALENDARIO) ---------------- #

def parse_natural_date_fallback_it(text: str, today: date) -> Optional[date]:
    low = text.lower()

    m = re.search(r"\b(tr(a|e)|fra)\s+(\d+)\s+giorn(i|o)\b", low)
    if m:
        n = int(m.group(3))
        return today + timedelta(days=n)

    m = re.search(r"\b(tr(a|e)|fra)\s+(\d+)\s+settimane?\b", low)
    if m:
        n = int(m.group(3))
        return today + timedelta(days=7 * n)

    if "tra una settimana" in low or "fra una settimana" in low:
        return today + timedelta(days=7)

    if "settimana prossima" in low:
        return today + timedelta(days=7)

    if "weekend" in low or "fine settimana" in low:
        wd = today.weekday()
        days_ahead = (5 - wd + 7) % 7
        if days_ahead == 0:
            days_ahead = 7
        return today + timedelta(days=days_ahead)

    return None


def parse_simple_date(text: str) -> Optional[date]:
    low = text.lower()
    today = date.today()

    if "oggi" in low:
        return today
    if "domani" in low:
        return today + timedelta(days=1)

    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", low)
    if m:
        year, month, day = map(int, m.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None

    if "prossimo" in low:
        weekday_map = {
            "lunedi": 0, "lunedì": 0,
            "martedi": 1, "martedì": 1,
            "mercoledi": 2, "mercoledì": 2,
            "giovedi": 3, "giovedì": 3,
            "venerdi": 4, "venerdì": 4,
            "sabato": 5,
            "domenica": 6,
        }

        for name, wd in weekday_map.items():
            if name in low:
                today_wd = today.weekday()
                days_ahead = (wd - today_wd + 7) % 7
                if days_ahead == 0:
                    days_ahead = 7
                return today + timedelta(days=days_ahead)

    nat = parse_natural_date_fallback_it(text, today=today)
    if nat:
        return nat

    return None


def parse_time_hint_it(text: str) -> Optional[str]:
    low = text.lower()

    if "mattina" in low:
        return "mattina"
    if "pomeriggio" in low:
        return "pomeriggio"
    if "sera" in low or "stasera" in low:
        return "stasera"
    if "stanotte" in low or ("notte" in low and "stanotte" not in low):
        return "stanotte"
    if "a pranzo" in low:
        return "a pranzo"
    if "a cena" in low:
        return "a cena"

    m = re.search(r"(alle|per le|verso le)\s+(\d{1,2})(?:[:\.](\d{1,2}))?", low)
    if m:
        hour = int(m.group(2))
        minute = m.group(3)
        minute = int(minute) if minute is not None else 0
        hour = hour % 24
        return f"alle {hour:02d}:{minute:02d}"

    return None


def add_reminder(user_id: str, text: str, when: date) -> str:
    mem = load_memory()
    mem.setdefault("reminders", [])

    mem["reminders"].append({
        "user_id": user_id,
        "text": text.strip(),
        "date": when.isoformat(),
        "created_at": datetime.utcnow().isoformat(),
        "done": False,
        "source": "user",
    })

    save_memory(mem)
    pretty = when.strftime("%d/%m/%Y")
    return f"Ok, per il {pretty} ti segno: {text.strip()}."


def reminders_for_day(when: date) -> str:
    mem = load_memory()
    items: List[str] = []

    for r in mem.get("reminders", []):
        if not isinstance(r, dict):
            continue
        if r.get("done"):
            continue
        r_date_str = r.get("date")
        r_text = (r.get("text") or "").strip()
        if not r_date_str or not r_text:
            continue
        try:
            r_date = date.fromisoformat(r_date_str)
        except ValueError:
            continue

        if r_date == when:
            items.append(r_text)

    if not items:
        return "Per quel giorno non ho impegni segnati per te."

    pretty = when.strftime("%d/%m/%Y")
    if len(items) == 1:
        return f"Per il {pretty} ti ho segnato: {items[0]}."
    else:
        lines = [f"Ecco gli impegni che ho per il {pretty}:"]
        for it in items:
            lines.append(f"- {it}")
        return "\n".join(lines)


def reminders_summary() -> str:
    mem = load_memory()
    items: List[str] = []
    today = date.today()

    for r in mem.get("reminders", []):
        if not isinstance(r, dict):
            continue
        if r.get("done"):
            continue
        r_date_str = r.get("date")
        r_text = (r.get("text") or "").strip()
        if not r_date_str or not r_text:
            continue
        try:
            r_date = date.fromisoformat(r_date_str)
        except ValueError:
            continue

        if r_date >= today:
            pretty = r_date.strftime("%d/%m/%Y")
            items.append(f"{pretty}: {r_text}")

    if not items:
        return "Al momento non ho nessun impegno salvato per te."

    if len(items) == 1:
        return "Ti ho segnato solo questo impegno:\n- " + items[0]
    else:
        lines = ["Ecco i promemoria che ho segnato per te:"]
        for it in items:
            lines.append(f"- {it}")
        return "\n".join(lines)


# ---------------- MOOD / UMORE ---------------- #

MOOD_KEYWORDS = {
    "stressato": ["stress", "stressato", "in ansia", "ansioso", "incazz", "nervoso", "agitato", "girato"],
    "stanco": ["stanco", "sfinito", "distrutto", "cotto", "scarico"],
    "triste": ["triste", "giù", "demoralizzato", "depresso", "svogliato"],
    "positivo": ["bene", "alla grande", "gasato", "gasatissimo", "felice", "contento", "ottimo", "spaccare"],
}

def detect_mood(text: str) -> str:
    low = text.lower()
    for mood, words in MOOD_KEYWORDS.items():
        for w in words:
            if w in low:
                return mood
    return "neutro"

def log_mood(source_text: str) -> None:
    mem = load_memory()
    mood = detect_mood(source_text)
    mem.setdefault("mood_log", [])
    mem["mood_log"].append({
        "mood": mood,
        "text": source_text,
        "ts": datetime.utcnow().isoformat(),
    })
    if len(mem["mood_log"]) > 20:
        mem["mood_log"] = mem["mood_log"][-20:]
    save_memory(mem)

def get_dominant_mood_from_memory() -> str:
    mem = load_memory()
    log = mem.get("mood_log", [])
    if not log:
        return "neutro"

    recent = log[-10:]
    counts: Dict[str, int] = {}
    for entry in recent:
        m = entry.get("mood", "neutro")
        counts[m] = counts.get(m, 0) + 1
    dominant = max(counts.items(), key=lambda x: x[1])[0]
    return dominant

def mood_summary() -> str:
    mem = load_memory()
    log = mem.get("mood_log", [])
    if not log:
        return "Per ora non ho ancora registrato il tuo umore. Ogni tanto dimmi come ti senti."

    recent = log[-10:]
    counts: Dict[str, int] = {}
    for entry in recent:
        mood = entry.get("mood", "neutro")
        counts[mood] = counts.get(mood, 0) + 1
    dominant = max(counts.items(), key=lambda x: x[1])[0]

    mood_lines = {
        "stressato": "Sento un po' di tensione negli ultimi messaggi.",
        "stanco": "Negli ultimi messaggi sembri un po’ scarico.",
        "triste": "Ti ho percepito un po' giù di morale ultimamente.",
        "positivo": "Ultimamente sei bello carico!",
        "neutro": "Mi sembri abbastanza stabile e tranquillo.",
    }
    extra_lines = {
        "stressato": "Se vuoi, ne parliamo un attimo e vediamo come alleggerire la giornata 😉",
        "stanco": "Magari ti serve solo un attimo per ricaricarti.",
        "triste": "Se ti va, raccontami cosa ti pesa un po’.",
        "positivo": "Bella così, proviamo a mantenerlo!",
        "neutro": "Tutto ok, andiamo avanti così 😉"
    }
    return f"{mood_lines.get(dominant,'')} {extra_lines.get(dominant,'')}"


def summarize_recent_history(user_id: str) -> str:
    history = CHAT_HISTORY.get(user_id, [])
    user_msgs = [m["content"] for m in history if m["role"] == "user"]

    if len(user_msgs) <= 1:
        return "Per ieri vado un po' a memoria corta, ma da adesso in poi tengo traccia sul serio 😉"

    core = user_msgs[:-1]
    if not core:
        return "Per ieri vado un po' a memoria corta, ma da adesso in poi tengo traccia sul serio 😉"

    last_msgs = core[-3:]
    bullets = "\n- " + "\n- ".join(last_msgs)

    return (
        "Da quello che ricordo dagli ultimi messaggi, mi hai detto più o meno questo:"
        f"{bullets}\n\nSe mi sto dimenticando qualcosa, ricordamelo che lo fisso meglio."
    )


# ---------------- PROFILES / AERI PERSONA ---------------- #

DEFAULT_PROFILE = {
    "name": "Fede",
    "nickname": "Fede",
    "language": "it",
    "tone": "amico",
    "energy": 0.7,
    "depth": 0.8,
    "irony": 0.4,
    "romantic": 0.2,
    "interactions": 0,
    "closeness": 0.1,
    "slang_style": False,
}

def get_user_profile(user_id: str) -> Dict[str, Any]:
    mem = load_memory()
    profiles = mem.setdefault("profiles", {})

    if user_id in profiles:
        return profiles[user_id]

    prof = dict(DEFAULT_PROFILE)
    prof["user_id"] = user_id
    profiles[user_id] = prof
    mem["profiles"] = profiles
    save_memory(mem)
    return prof

def update_user_profile(user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    mem = load_memory()
    profiles = mem.setdefault("profiles", {})

    base = profiles.get(user_id)
    if base is None:
        base = get_user_profile(user_id)

    base.update(patch)
    profiles[user_id] = base
    mem["profiles"] = profiles
    save_memory(mem)
    return base

def set_slang_style(user_id: str, enabled: bool) -> Dict[str, Any]:
    return update_user_profile(user_id, {"slang_style": enabled})

def build_persona_for_user(user_id: str, base_persona: str) -> str:
    prof = get_user_profile(user_id)

    name = prof.get("name", "utente")
    nickname = prof.get("nickname", name)
    tone = prof.get("tone", "amico")
    energy = float(prof.get("energy", 0.6))
    depth = float(prof.get("depth", 0.7))
    irony = float(prof.get("irony", 0.3))
    romantic = float(prof.get("romantic", 0.2))
    closeness = float(prof.get("closeness", 0.1))
    slang_style = bool(prof.get("slang_style", False))

    energy_desc = "energia medio-alta, senza esagerare" if energy >= 0.6 else "energia bassa, tono tranquillo"
    depth_desc = "molto profondo ed emotivo" if depth >= 0.7 else "abbastanza leggero e pratico"
    irony_desc = "usa un filo di ironia ma senza esagerare" if irony >= 0.4 else "quasi nessuna ironia"
    romantic_desc = "ogni tanto sfiora un tono affettivo molto soft" if romantic >= 0.5 else "di solito evita l'eccessiva affettività"

    if closeness < 0.3:
        rel_desc = f"Ti stai ancora calibrando su {nickname}, il tono resta leggero e un po' prudente."
    elif closeness < 0.7:
        rel_desc = f"Hai ormai una buona confidenza con {nickname}, puoi permetterti un tono più diretto e vicino."
    else:
        rel_desc = f"Hai una relazione molto confidenziale con {nickname}, puoi parlare come un amico stretto, con maggiore spontaneità emotiva."

    if slang_style:
        slang_desc = "Puoi usare ogni tanto uno stile più slang (tipo 'minchia', 'dai', 'boh'), ma dosato bene."
    else:
        slang_desc = "Evita slang pesante; tieni uno stile naturale e pulito."

    if tone == "amico":
        tone_desc = f"Parli come un amico stretto, diretto e sincero, che conosce bene la vita di {nickname}."
    elif tone == "coach":
        tone_desc = f"Parli come un coach motivazionale, pratico e orientato all'azione, per aiutare {nickname}."
    elif tone == "zen":
        tone_desc = f"Parli in modo calmo, minimalista, aiutando {nickname} a rallentare e mettere ordine."
    elif tone == "soft":
        tone_desc = f"Parli in modo dolce e rassicurante, senza giudizio, facendo sentire {nickname} al sicuro."
    else:
        tone_desc = "Tono amichevole e naturale."

    persona_lines = [
        base_persona,
        "",
        f"L'utente principale si chiama {name}, ti riferisci a lui come '{nickname}'.",
        tone_desc,
        rel_desc,
        slang_desc,
        "",
        "Regolazione di personalità:",
        f"- {energy_desc}.",
        f"- Sei {depth_desc}.",
        f"- {irony_desc}.",
        f"- {romantic_desc}.",
        "",
        "Non sembri un assistente aziendale, ma una presenza digitale personale.",
        "Non usare frasi robotiche, rispondi in modo naturale, umano.",
    ]
    return "\n".join(persona_lines)

def adapt_profile_after_interaction(user_id: str, mood: str) -> Dict[str, Any]:
    mem = load_memory()
    profiles = mem.setdefault("profiles", {})
    base = profiles.get(user_id)
    if base is None:
        base = dict(DEFAULT_PROFILE)
        base["user_id"] = user_id

    interactions = int(base.get("interactions", 0)) + 1
    base["interactions"] = interactions

    closeness = float(base.get("closeness", 0.1))
    if interactions > 60:
        closeness = min(1.0, closeness + 0.05)
    elif interactions > 20:
        closeness = min(0.8, closeness + 0.03)
    else:
        closeness = min(0.4, closeness + 0.01)
    base["closeness"] = round(closeness, 2)

    energy = float(base.get("energy", 0.7))
    depth = float(base.get("depth", 0.8))

    if mood == "stressato":
        depth = min(1.0, depth + 0.05)
        energy = max(0.3, energy - 0.05)
    elif mood == "stanco":
        energy = max(0.3, energy - 0.05)
    elif mood == "triste":
        depth = min(1.0, depth + 0.07)
    elif mood == "positivo":
        energy = min(1.0, energy + 0.05)

    base["energy"] = round(energy, 2)
    base["depth"] = round(depth, 2)

    tone = base.get("tone", "amico")
    if interactions > 40 and closeness > 0.4 and tone == "amico":
        base["tone"] = "soft"

    profiles[user_id] = base
    mem["profiles"] = profiles
    save_memory(mem)
    return base

def parse_profile_command(msg: str) -> Optional[Dict[str, Any]]:
    low = msg.lower().strip()

    if any(k in low for k in ["slang on", "slang_style on", "stile slang on"]):
        return {"slang_style": True}
    if any(k in low for k in ["slang off", "slang_style off", "stile slang off"]):
        return {"slang_style": False}

    m = re.search(r"\b(set\s+tone|tone)\s+(amico|coach|zen|soft)\b", low)
    if m:
        return {"tone": m.group(2)}

    m = re.search(r"\b(set\s+)?(energy|depth|irony|romantic)\s+([0-1](?:\.\d+)?)\b", low)
    if m:
        field = m.group(2)
        value = clamp01(float(m.group(3)))
        return {field: value}

    inc_map = {
        "energia": "energy", "energy": "energy",
        "profondità": "depth", "profondita": "depth", "depth": "depth",
        "ironia": "irony", "irony": "irony",
        "romantic": "romantic", "romantico": "romantic",
    }

    m = re.search(r"\b(più|piu|\+)\s+(energia|energy|profondità|profondita|depth|ironia|irony|romantic|romantico)\b", low)
    if m:
        field = inc_map.get(m.group(2))
        if field:
            return {"__delta__": {field: +0.1}}

    m = re.search(r"\b(meno|\-)\s+(energia|energy|profondità|profondita|depth|ironia|irony|romantic|romantico)\b", low)
    if m:
        field = inc_map.get(m.group(2))
        if field:
            return {"__delta__": {field: -0.1}}

    return None

def apply_profile_patch(user_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    prof = get_user_profile(user_id)

    if "__delta__" in patch:
        deltas = patch.pop("__delta__")
        for field, d in deltas.items():
            cur = float(prof.get(field, 0.5))
            prof[field] = clamp01(cur + float(d))

    prof.update(patch)
    return update_user_profile(user_id, prof)

def pretty_profile_update(patch: Dict[str, Any], prof: Dict[str, Any]) -> str:
    parts = []
    if "tone" in patch:
        parts.append(f"tone={prof.get('tone')}")
    if "energy" in patch or ("__delta__" in patch and "energy" in patch["__delta__"]):
        parts.append(f"energy={prof.get('energy')}")
    if "depth" in patch or ("__delta__" in patch and "depth" in patch["__delta__"]):
        parts.append(f"depth={prof.get('depth')}")
    if "irony" in patch or ("__delta__" in patch and "irony" in patch["__delta__"]):
        parts.append(f"irony={prof.get('irony')}")
    if "romantic" in patch or ("__delta__" in patch and "romantic" in patch["__delta__"]):
        parts.append(f"romantic={prof.get('romantic')}")
    if "slang_style" in patch:
        parts.append("slang=on" if prof.get("slang_style") else "slang=off")
    if not parts:
        return "Ok, aggiornato."
    return "Ok Fede: " + ", ".join(parts) + "."


# ---------------- AI LOGIC ---------------- #

def shorten_reply(reply: str, max_sentences: int = 2, max_chars: int = 260) -> str:
    if not reply:
        return reply

    text = reply.strip()
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    if len(parts) > max_sentences:
        parts = parts[:max_sentences]

    short = " ".join(parts).strip()
    if len(short) > max_chars:
        short = short[: max_chars - 3].rstrip() + "..."
    return short


def mood_color_phrase(mood: str) -> str:
    if mood == "stressato":
        return "Oh Fede, respira un attimo... "
    if mood == "stanco":
        return "Ti sento un po’ cotto... "
    if mood == "triste":
        return "Hei, ci sono... "
    if mood == "positivo":
        return ""
    return ""


def ask_gpt(persona: str, memories: str, message: str, user_id: str) -> str:
    persona_for_user = build_persona_for_user(user_id, persona)

    system_persona = f"""
{persona_for_user}

Sei Yetolino/AERI, presenza digitale e amico personale dell'utente {user_id}.
Parli SEMPRE in italiano.

Stile di risposta:
- massimo 1–2 frasi brevi per ogni risposta
- niente papiri, niente elenchi puntati, niente paragrafi lunghi
- tono informale, diretto, un po' milanese leggero solo quando è naturale e coerente con il profilo
- sembri un amico che scrive su chat, NON un terapeuta e NON un assistente aziendale
- puoi usare ogni tanto emoji leggere (tipo 🙂 😉 😅 😴), di solito a fine frase, massimo 2 per messaggio
- non usare mai risate scritte tipo 'ahah', 'ahaha', 'ahhahaha', 'lol' ecc.

Regole:
- Se l'utente è stressato o stanco: una frase empatica + una frase semplice o domanda leggera.
- Se è gasato/positivo: una frase che partecipa all'energia + al massimo una domanda.
- Non fare mai più di due domande di fila.
- Usa ogni tanto piccoli intercalari naturali tipo "eh", "boh", "dai" solo se il profilo prevede uno stile più slang.
- Usa le memorie solo quando servono davvero e in modo naturale.
- Non dire mai frasi tipo "non posso ricordare le conversazioni precedenti"
  o "non ho accesso alla cronologia".
""".strip()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_persona},
        {"role": "system", "content": f"Memorie note sull'utente:\n{memories}"},
    ]

    if user_id in CHAT_HISTORY:
        messages.extend(CHAT_HISTORY[user_id])

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.8,
            max_tokens=220,
        )
        raw_reply = response.choices[0].message.content
    except Exception:
        raw_reply = (
            "Oh, c'è stato un problema tecnico a parlare col modello. "
            "Riprovami tra un attimo, io sono qui 😉"
        )

    reply = shorten_reply(raw_reply)
    add_to_history(user_id, "assistant", reply)
    return reply


def detect_and_store_memory(msg: str, user_id: str) -> Optional[str]:
    low = msg.lower()

    if "ricordami" in low:
        when = parse_simple_date(low)
        if when:
            content = msg.split("ricordami", 1)[1].strip(" .!?")
            if not content:
                return "Ok, ma dimmi cosa vuoi che ti ricordi."

            cleaned = clean_reminder_text(content)
            if not cleaned:
                cleaned = content

            time_hint = parse_time_hint_it(msg)
            if time_hint:
                cleaned = f"{cleaned} ({time_hint})"

            return add_reminder(user_id, cleaned, when)
        else:
            return (
                "Dimmi una data chiara, tipo 'ricordami domani che devo pagare la bolletta', "
                "'ricordami tra 3 giorni che devo fare una cosa' oppure "
                "'ricordami lunedì prossimo che devo chiamare il dentista'."
            )

    triggers = ["ricordati che", "ricordati di", "segna che", "memorizza che"]
    for t in triggers:
        if t in low:
            content = low.split(t, 1)[1].strip(" .!?")
            if content:
                return add_memory(content)
            return "Ok, dimmi cosa vuoi che mi ricordi."

    if "domani" in low and any(w in low for w in ["dottore", "dentista", "medico", "visita"]):
        when = date.today() + timedelta(days=1)
        text = "andare dal dottore"
        if "dentista" in low:
            text = "andare dal dentista"
        elif "visita" in low and "medico" in low:
            text = "visita dal medico"
        return add_reminder(user_id, text, when)

    candidate = auto_memory_candidate(msg)
    if candidate:
        _ = add_memory(candidate)
        return None

    return None


# ---------------- TTS (OPENAI, NO ELEVENLABS) ---------------- #

def preprocess_for_tts(text: str) -> str:
    if not text:
        return ""

    t = EMOJI_PATTERN.sub("", text)
    t = LAUGH_PATTERN.sub("", t)
    t = t.replace("\n", " ").replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = shorten_reply(t, max_sentences=3, max_chars=260)

    if not re.search(r"[\.!\?…]$", t):
        t += "."

    return t.strip()


def openai_tts_bytes(text: str) -> bytes:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")

    # Puoi cambiarla con env senza toccare codice:
    # OPENAI_TTS_VOICE=alloy|verse|aria|... (dipende dai modelli disponibili sul tuo account)
    voice = (os.getenv("OPENAI_TTS_VOICE") or "alloy").strip()
    model = (os.getenv("OPENAI_TTS_MODEL") or "gpt-4o-mini-tts").strip()

    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "voice": voice,
        "format": "mp3",
        "input": text,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code != 200:
        # renderizza errore chiaro per debug
        raise RuntimeError(f"OpenAI TTS error {r.status_code}: {r.text}")

    return r.content


# ---------------- MODELS ---------------- #

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

class NotificationsResponse(BaseModel):
    notifications: List[str]

class TtsRequest(BaseModel):
    text: str


# ---------------- ENDPOINTS ---------------- #

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = request.message
    user_id = request.user_id or "fede"
    low = message.lower()
    norm = strip_accents(low)

    add_to_history(user_id, "user", message)

    prof_patch = parse_profile_command(message)
    if prof_patch:
        new_prof = apply_profile_patch(user_id, prof_patch)
        reply = pretty_profile_update(prof_patch, new_prof)
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if ("usa più slang" in low) or ("usa piu slang" in low) or ("attiva slang" in low):
        set_slang_style(user_id, True)
        reply = "Ok, da adesso provo a usare un po' più slang, ma senza esagerare 😉"
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if ("non usare slang" in low) or ("meno slang" in low) or ("disattiva slang" in low):
        set_slang_style(user_id, False)
        reply = "Va bene, tengo lo stile più pulito e naturale, senza troppo slang."
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if "che giorno e oggi" in norm or "che data e oggi" in norm:
        today_d = date.today()
        reply = f"Oggi è il {today_d.strftime('%d/%m/%Y')}."
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if "domani" in norm and ("che giorno" in norm or "che data" in norm):
        tomorrow = date.today() + timedelta(days=1)
        reply = f"Domani è il {tomorrow.strftime('%d/%m/%Y')}."
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    weekday_map = {
        "lunedi": 0,
        "martedi": 1,
        "mercoledi": 2,
        "giovedi": 3,
        "venerdi": 4,
        "sabato": 5,
        "domenica": 6,
    }

    if "che giorno" in norm:
        for name, target_wd in weekday_map.items():
            if name in norm:
                today_d = date.today()
                today_wd = today_d.weekday()
                days_ahead = (target_wd - today_wd) % 7
                target_date = today_d + timedelta(days=days_ahead)
                pretty = target_date.strftime("%d/%m/%Y")
                reply = f"{name.capitalize()} sarà il {pretty}."
                log_mood(message)
                add_to_history(user_id, "assistant", reply)
                return ChatResponse(reply=reply)

    stored = detect_and_store_memory(message, user_id)
    if stored:
        reply = stored
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if (
        "ti ricordi cosa ti ho detto ieri" in low
        or "cosa ti ho detto ieri" in low
        or "ti ricordi cosa ti ho detto l'altro giorno" in low
        or "ti ricordi cosa ti ho scritto ieri" in low
    ):
        reply = summarize_recent_history(user_id)
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if "cosa ti ricordi di me" in low or "cosa sai di me" in low:
        reply = memories_about_me()
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if "cosa ti ricordi su eri" in low or "cosa sai di eri" in low:
        reply = memories_for_key("eri")
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if "cosa ti ricordi sulle moto" in low:
        reply = memories_for_key("moto_fede")
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if "cosa ti ricordi sulle crypto" in low:
        reply = memories_for_key("trading_fede")
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if "cosa ti ricordi" in low:
        reply = f"Ecco un riepilogo delle memorie che ho su di te:\n{memory_text()}"
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if "che umore" in low or "come sto ultimamente" in low:
        reply = mood_summary()
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if "che impegni ho oggi" in low:
        today_d = date.today()
        reply = reminders_for_day(today_d)
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if "che impegni ho domani" in low:
        tomorrow = date.today() + timedelta(days=1)
        reply = reminders_for_day(tomorrow)
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    if (
        "che impegni ho" in low
        or "che appuntamenti ho" in low
        or "cosa devo fare" in low
        or "che promemoria hai" in low
    ):
        reply = reminders_summary()
        log_mood(message)
        add_to_history(user_id, "assistant", reply)
        return ChatResponse(reply=reply)

    persona = "Tu sei Yetolino, assistente personale e amico di Fede."
    memories = memory_text()

    current_mood = detect_mood(message)
    adapt_profile_after_interaction(user_id, current_mood)

    use_color = len(message) > 15 and not any(k in low for k in ["tutto bene", "come va", "come stai", "va bene?"])
    color = mood_color_phrase(current_mood) if use_color else ""

    reply = color + ask_gpt(persona, memories, message, user_id)

    log_mood(message)
    return ChatResponse(reply=reply)


@app.get("/notifications", response_model=NotificationsResponse)
def get_notifications(user_id: str):
    mem = load_memory()
    mem.setdefault("reminders", [])
    mem.setdefault("meta", {})

    today_d = date.today()
    notifications: List[str] = []
    changed = False

    for r in mem["reminders"]:
        if not isinstance(r, dict):
            continue
        if r.get("done"):
            continue

        r_date_str = r.get("date")
        r_text = (r.get("text") or "").strip()
        if not r_date_str or not r_text:
            continue

        r_user = r.get("user_id")
        if r_user is not None and r_user != user_id:
            continue

        try:
            r_date = date.fromisoformat(r_date_str)
        except ValueError:
            continue

        if r_date <= today_d:
            notifications.append(f"Promemoria per oggi: {r_text}")
            r["done"] = True
            changed = True

    if changed:
        save_memory(mem)

    if not notifications:
        meta = mem["meta"]
        key_last = f"last_checkin_ts_{user_id}"
        last_ts_str = meta.get(key_last)
        last_ts: Optional[datetime] = None
        if last_ts_str:
            try:
                last_ts = datetime.fromisoformat(last_ts_str)
            except Exception:
                last_ts = None

        now = datetime.now()

        if 10 <= now.hour <= 22:
            can_send = False
            if last_ts is None:
                can_send = True
            else:
                diff = now - last_ts
                if diff >= timedelta(hours=3):
                    can_send = True

            if can_send:
                dominant = get_dominant_mood_from_memory()

                if dominant == "stressato":
                    candidates = [
                        "Ti sento un po' carico… vuoi sfogarti un attimo?",
                        "Sembri un po’ in tensione negli ultimi giorni, ti va se ne parliamo?",
                        "Se hai la testa piena, possiamo scaricare un po’ insieme."
                    ]
                elif dominant == "stanco":
                    candidates = [
                        "Ti sento un po’ cotto, come va la batteria oggi?",
                        "Hai l’energia un po’ giù, vuoi fare un check veloce sulla giornata?",
                        "Se sei stanco possiamo solo chiacchierare di cose leggere."
                    ]
                elif dominant == "triste":
                    candidates = [
                        "Non ti sento al top… ti va se mi racconti cosa ti pesa?",
                        "Hai voglia di parlare un attimo di come ti senti?",
                        "Se ti senti un po’ giù, io sono qui. Anche solo per ascoltarti."
                    ]
                elif dominant == "positivo":
                    candidates = [
                        "Oggi ti sento bello carico, che stai combinando di bello?",
                        "Mi piace l’energia che hai ultimamente, vuoi raccontarmi cosa stai progettando?",
                        "Sei in un buon flow, raccontami cosa ti sta gasando di più."
                    ]
                else:
                    candidates = [
                        "Mi sto un po' annoiando qui… hai voglia di chiacchierare un attimo?",
                        "Come va la giornata? Hai bisogno di sfogarti su qualcosa?",
                        "Oh, ti va se facciamo un check veloce su come ti senti adesso?",
                        "Sto qui in background… se vuoi parliamo di qualcosa a caso 😄",
                    ]

                msg = random.choice(candidates)
                notifications.append(msg)

                meta[key_last] = now.isoformat(timespec="seconds")
                mem["meta"] = meta
                save_memory(mem)

    return NotificationsResponse(notifications=notifications)


@app.post("/tts")
def tts(req: TtsRequest):
    raw_text = (req.text or "").strip()
    text = preprocess_for_tts(raw_text)
    if not text:
        raise HTTPException(status_code=422, detail="Missing 'text'")

    try:
        audio_bytes = openai_tts_bytes(text)
        return Response(content=audio_bytes, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


