# src/facts.py
"""
Verified answers for questions whose answer is a lookup, not a judgement: a
service's price for a car, engine size, duration or sale price, and whether a
service is offered in a city or for a model year.

The fine-tuned model got these wrong on unseen phrasings (a new car model, a
Roman Urdu city question). Here code reads the question, looks the answer up
in the tables below (taken from data/pakwheels/knowledge_base.md; a smoke test
checks they still match it) and puts it at the top of the context as
"## Verified answer". The model only has to phrase it.

guard() is the second safety net: a reply quoting a price or percentage that
appears nowhere in its context is replaced with a pointer to the official page.

No heavy dependencies: the dataset builder uses the same functions, so the
training contexts contain exactly what the app adds at inference.
"""

import re
from dataclasses import dataclass

from src.utils import _history_to_messages

HELPLINE = "042-111-943-357 (042-111-WHEELS)"
FACT_HEADER = "## Verified answer"

# --- Tables (from the knowledge base) ----------------------------------------

TIER_LABEL = {
    "small": "up to 1000cc", "mid": "1001cc to 2000cc",
    "suv": "an SUV", "4x4": "a 4x4", "german": "a German car",
}
INSPECTION_PRICE = {"small": "4,950", "mid": "6,950", "suv": "9,950", "4x4": "9,950", "german": "9,950"}
SIFM_FEE = {"small": "2,000", "mid": "5,000", "suv": "7,000", "4x4": "7,000", "german": "7,000"}
PDI_PRICE = "6,900"
FEATURED_PRICE = {"7 days": "2,950", "14 days": "4,450", "28 days": "5,950"}
BUNDLE_PRICE = {5: "13,450", 10: "25,450"}
COMMISSION_MIN = 5_000

INSPECTION_CITIES = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar", "Faisalabad",
                     "Gujranwala", "Gujrat", "Hyderabad", "Multan", "Sargodha", "Sialkot"]
SIFM_CITIES = ["Karachi", "Lahore", "Faisalabad", "Gujranwala", "Islamabad", "Rawalpindi", "Peshawar"]
TRANSFER_CITIES = ["Lahore", "Karachi", "Islamabad"]
# Pakistani cities in none of the lists above: "No" for every city-based service
OTHER_CITIES = ["Quetta", "Bahawalpur", "Abbottabad", "Sukkur", "Mardan", "Sahiwal",
                "Rahim Yar Khan", "Jhelum", "Gilgit", "Mirpur", "Dera Ismail Khan",
                "Dera Ghazi Khan", "Larkana", "Nawabshah", "Okara", "Kasur", "Sheikhupura",
                "Murree", "Chitral", "Swat", "Mingora", "Muzaffarabad", "Gwadar", "Chakwal",
                "Attock", "Wah Cantt", "Taxila", "Khanewal", "Jhang", "Mandi Bahauddin",
                "Hafizabad", "Kohat", "Bannu", "Turbat", "Skardu", "Kamoke", "Wazirabad",
                "Vehari", "Muzaffargarh", "Mianwali", "Khushab", "Nowshera", "Charsadda"]
CITY_ALIASES = {"pindi": "Rawalpindi", "isb": "Islamabad", "isloo": "Islamabad", "khi": "Karachi",
                "lhr": "Lahore", "fsd": "Faisalabad", "di khan": "Dera Ismail Khan",
                "d.i. khan": "Dera Ismail Khan", "d i khan": "Dera Ismail Khan",
                "dg khan": "Dera Ghazi Khan", "d.g. khan": "Dera Ghazi Khan",
                "ryk": "Rahim Yar Khan", "wah": "Wah Cantt", "mirpur ajk": "Mirpur"}

# Common Pakistani-market models -> price band. "varies": sold with engines in
# two bands; "large": a sedan over 2000cc, which no listed band covers.
# (regex, name as answers show it, band)
CARS = [
    (r"suzuki alto|\balto\b", "Suzuki Alto", "small"),
    (r"mehran", "Suzuki Mehran", "small"),
    (r"cultus", "Suzuki Cultus", "small"),
    (r"wagon ?r\b", "Suzuki Wagon R", "small"),
    (r"bolan", "Suzuki Bolan", "small"),
    (r"suzuki every\b", "Suzuki Every", "small"),
    (r"suzuki ravi\b", "Suzuki Ravi", "small"),
    (r"suzuki khyber\b", "Suzuki Khyber", "small"),
    (r"\bswift\b", "Suzuki Swift", "mid"),
    (r"\bciaz\b", "Suzuki Ciaz", "mid"),
    (r"\bliana\b", "Suzuki Liana", "mid"),
    (r"\bbaleno\b", "Suzuki Baleno", "mid"),
    (r"suzuki margalla\b", "Suzuki Margalla", "mid"),
    (r"\bapv\b", "Suzuki APV", "mid"),
    (r"\bjimny\b", "Suzuki Jimny", "4x4"),
    (r"\bvitara\b", "Suzuki Vitara", "suv"),
    (r"\bmira\b", "Daihatsu Mira", "small"),
    (r"\bcuore\b", "Daihatsu Cuore", "small"),
    (r"daihatsu move\b", "Daihatsu Move", "small"),
    (r"\btanto\b", "Daihatsu Tanto", "small"),
    (r"\bhijet\b", "Daihatsu Hijet", "small"),
    (r"\bterios\b", "Daihatsu Terios", "suv"),
    (r"\bdayz\b", "Nissan Dayz", "small"),
    (r"\bmoco\b", "Nissan Moco", "small"),
    (r"nissan note\b", "Nissan Note", "mid"),
    (r"nissan sunny\b", "Nissan Sunny", "mid"),
    (r"x-?trail", "Nissan X-Trail", "suv"),
    (r"\bjuke\b", "Nissan Juke", "suv"),
    (r"\bpasso\b", "Toyota Passo", "small"),
    (r"\bpixis\b", "Toyota Pixis", "small"),
    (r"\bvitz\b", "Toyota Vitz", "varies"),
    (r"corolla cross", "Toyota Corolla Cross", "suv"),
    (r"\bcorolla\b", "Toyota Corolla", "mid"),
    (r"\byaris\b", "Toyota Yaris", "mid"),
    (r"\baqua\b", "Toyota Aqua", "mid"),
    (r"\bprius\b", "Toyota Prius", "mid"),
    (r"\bpremio\b", "Toyota Premio", "mid"),
    (r"\ballion\b", "Toyota Allion", "mid"),
    (r"\bprobox\b", "Toyota Probox", "mid"),
    (r"\bcamry\b", "Toyota Camry", "large"),
    (r"mark ?x\b", "Toyota Mark X", "large"),
    (r"land ?cruiser", "Toyota Land Cruiser", "suv"),
    (r"\bprado\b", "Toyota Prado", "suv"),
    (r"\bfortuner\b", "Toyota Fortuner", "suv"),
    (r"toyota rush\b", "Toyota Rush", "suv"),
    (r"\braize\b", "Toyota Raize", "suv"),
    (r"\bc-?hr\b", "Toyota C-HR", "suv"),
    (r"\bcivic\b", "Honda Civic", "mid"),
    (r"honda city\b", "Honda City", "mid"),
    (r"honda fit\b", "Honda Fit", "mid"),
    (r"honda grace\b", "Honda Grace", "mid"),
    (r"n-? ?wgn\b", "Honda N-WGN", "small"),
    (r"n-? ?box\b", "Honda N-Box", "small"),
    (r"\bbr-?v\b", "Honda BR-V", "suv"),
    (r"\bhr-?v\b", "Honda HR-V", "suv"),
    (r"\bcr-?v\b", "Honda CR-V", "suv"),
    (r"\bvezel\b", "Honda Vezel", "suv"),
    (r"ek ?wagon", "Mitsubishi eK Wagon", "small"),
    (r"\blancer\b", "Mitsubishi Lancer", "mid"),
    (r"\bpajero\b", "Mitsubishi Pajero", "suv"),
    (r"\boutlander\b", "Mitsubishi Outlander", "suv"),
    (r"\bsantro\b", "Hyundai Santro", "small"),
    (r"\belantra\b", "Hyundai Elantra", "mid"),
    (r"\btucson\b", "Hyundai Tucson", "suv"),
    (r"santa ?fe\b", "Hyundai Santa Fe", "suv"),
    (r"\bpicanto\b", "Kia Picanto", "small"),
    (r"\bsportage\b", "Kia Sportage", "suv"),
    (r"\bstonic\b", "Kia Stonic", "suv"),
    (r"\bsorento\b", "Kia Sorento", "suv"),
    (r"\balsvin\b", "Changan Alsvin", "mid"),
    (r"\bkarvaan\b", "Changan Karvaan", "small"),
    (r"oshan", "Changan Oshan X7", "suv"),
    (r"\bmg ?hs\b", "MG HS", "suv"),
    (r"\bmg ?zs\b", "MG ZS", "suv"),
    (r"\bmg ?5\b", "MG 5", "mid"),
    (r"\bsaga\b", "Proton Saga", "mid"),
    (r"\bx70\b", "Proton X70", "suv"),
    (r"haval ?h6|\bh6\b", "Haval H6", "suv"),
    (r"\bjolion\b", "Haval Jolion", "suv"),
    (r"\btiggo\b", "Chery Tiggo", "suv"),
    (r"dfsk glory|\bglory 580\b", "DFSK Glory", "suv"),
    (r"\bbravo\b", "United Bravo", "small"),
    (r"united alpha\b", "United Alpha", "small"),
    (r"prince pearl\b", "Prince Pearl", "small"),
    (r"faw v2\b", "FAW V2", "mid"),
    (r"peugeot 2008\b", "Peugeot 2008", "suv"),
]
_CAR_RES = [(re.compile(p, re.I), name, band) for p, name, band in CARS]
GERMAN_RE = re.compile(
    r"\b(mercedes(?:[- ]benz)?|benz|bmw|audi|volkswagen|vw|porsche)\b"
    r"(?:\s+([a-z]?\d+[a-z0-9-]*(?:\s+series)?|[a-z]{1,3}-class))?", re.I)
GERMAN_BRANDS = {"mercedes": "Mercedes", "mercedes-benz": "Mercedes-Benz", "mercedes benz": "Mercedes-Benz",
                 "benz": "Mercedes-Benz", "bmw": "BMW", "audi": "Audi", "volkswagen": "Volkswagen",
                 "vw": "Volkswagen", "porsche": "Porsche"}
BODY_RE = re.compile(r"\b(suv|4x4|jeep|german car)\b", re.I)
BODY = {"suv": ("an SUV", "suv"), "4x4": ("a 4x4", "4x4"), "jeep": ("a jeep", "4x4"),
        "german car": ("a German car", "german")}


# --- Answers (the training rows use these exact sentences) --------------------

def city_list(cities):
    return ", ".join(cities[:-1]) + " and " + cities[-1]


def car_phrase(car, tier):
    if tier in ("small", "mid"):
        return f"For a {car} ({TIER_LABEL[tier]}),"
    return f"The {car} is {TIER_LABEL[tier]}, so"


def cc_phrase(cc, tier):
    # "An 800cc", "an 1100cc" (eleven hundred), "a 1300cc"
    article = "An" if str(cc).startswith(("8", "11", "18")) else "A"
    return f"{article} {cc}cc car falls in the {TIER_LABEL[tier]} band, so"


def inspection_answer(lead, tier):
    verb = "its inspection costs" if lead.startswith("The ") else "a used car inspection costs"
    return f"{lead} {verb} PKR {INSPECTION_PRICE[tier]}. Inspection charges are non-refundable."


SIFM_TAIL = ("plus a 1% commission on the selling price after the sale, with a minimum of "
             "PKR 5,000 if it sells for 5 lakh or less.")


def sifm_fee_answer(lead, tier):
    verb = ("its Sell It For Me onboarding fee is" if lead.startswith("The ")
            else "the Sell It For Me onboarding fee is")
    return f"{lead} {verb} PKR {SIFM_FEE[tier]} (non-refundable), {SIFM_TAIL}"


LARGE_INSPECTION_ANSWER = (
    "The listed prices cover cars up to 2000cc and, separately, SUVs, 4x4s, jeeps and German "
    "cars, so a larger sedan isn't clearly listed. The booking form on the Car Inspection page "
    "shows the price for your exact car.")
LARGE_SIFM_ANSWER = (
    "The listed fees cover cars up to 2000cc and, separately, SUVs, 4x4s, jeeps and German cars, "
    "so a larger sedan isn't clearly listed. The Sell It For Me page shows the fee for your exact "
    "car when you sign up.")


def varies_answer(car, service):
    if service == "inspection":
        return (f"The {car} is sold with different engines: a used car inspection costs PKR "
                f"{INSPECTION_PRICE['small']} up to 1000cc and PKR {INSPECTION_PRICE['mid']} for "
                "1001cc to 2000cc. The engine size is on your registration papers. Inspection "
                "charges are non-refundable.")
    return (f"The {car} is sold with different engines: the Sell It For Me onboarding fee is PKR "
            f"{SIFM_FEE['small']} up to 1000cc and PKR {SIFM_FEE['mid']} for 1001cc to 2000cc "
            f"(non-refundable), {SIFM_TAIL}")


PDI_ANSWER = (f"A new car pre-delivery inspection (PDI) at the showroom costs PKR {PDI_PRICE}. "
              "Inspection charges are non-refundable.")


def inspection_city_answer(c):
    if c in INSPECTION_CITIES:
        where = ("at the PakWheels Inspection Center or a location of your choice" if c == "Lahore"
                 else "at your preferred location (including the seller's), subject to availability")
        return f"Yes, PakWheels car inspection is available in {c}, {where}."
    return (f"No, {c} isn't one of the current inspection cities. Inspection is available in "
            f"{city_list(INSPECTION_CITIES)}. If the car can be brought to one of these cities, "
            "you can book it there.")


def sifm_city_answer(c):
    if c in SIFM_CITIES:
        return (f"Yes, Sell It For Me is available in {c}. The car must be model year 2000 or newer, "
                "and open-letter cars aren't eligible.")
    extra = (f" Car inspection is available in {c}, though, if you want an inspection report "
             "for your ad." if c in INSPECTION_CITIES else "")
    return (f"No, Sell It For Me isn't offered in {c} right now. It's available in "
            f"{city_list(SIFM_CITIES)}. You can still post a free ad yourself and feature it "
            f"for more visibility.{extra}")


def transfer_city_answer(c):
    if c in TRANSFER_CITIES:
        return (f"Yes, PakWheels offers car registration and ownership transfer in {c}. Apply on "
                "the Car Registration or Car Transfer page; a service advisor calls to confirm the "
                "details and documents needed, and you can track progress online.")
    return (f"No, PakWheels' registration and transfer service isn't offered in {c} right now; "
            f"it's available in {city_list(TRANSFER_CITIES)}. In {c}, the transfer is handled by "
            "the provincial excise office.")


# label -> (days, price)
FEATURED = {"7 days": ("7 days", "2,950"), "a week": ("7 days", "2,950"),
            "14 days": ("14 days", "4,450"), "two weeks": ("14 days", "4,450"),
            "28 days": ("28 days", "5,950"), "a month": ("28 days", "5,950")}


def featured_answer(label):
    days, price = FEATURED[label]
    if label == "a month":
        answer = f"The longest single option is 28 days, which costs PKR {price}."
    else:
        answer = f"Featuring one ad for {days} costs PKR {price}."
    return answer + " Prices can change, so confirm on the Feature Your Ad page before paying."


FEATURED_OPTIONS_ANSWER = (
    "Single-ad feature options are 7 days (PKR 2,950), 14 days (PKR 4,450) and 28 days "
    "(PKR 5,950). Prices can change, so confirm on the Feature Your Ad page before paying.")


def bundle_answer(n):
    if n in BUNDLE_PRICE:
        return (f"A bundle of {n} feature credits is listed at PKR {BUNDLE_PRICE[n]}, which works "
                "out cheaper per ad than featuring ads one at a time.")
    return ("The listed bundles are 5 feature credits for PKR 13,450 and 10 for PKR 25,450, which "
            "work out cheaper per ad than featuring ads one at a time.")


def pkr(n):
    return f"PKR {n:,}"


def commission_for(lakh):
    """Sell It For Me commission: 1% of the price, minimum PKR 5,000 at 5 lakh or less."""
    price = round(lakh * 100_000)
    one_percent = price // 100
    return price, (max(one_percent, COMMISSION_MIN) if lakh <= 5 else one_percent), one_percent


def lakh_label(lakh):
    return f"{lakh / 100:g} crore" if lakh >= 100 else f"{lakh:g} lakh"


def commission_answer(lakh):
    # The deciding comparison comes first, then the amount
    label = lakh_label(lakh)
    price, commission, one_percent = commission_for(lakh)
    if lakh < 5:
        return (f"{label} is 5 lakh or less, so the minimum commission applies: {pkr(commission)}. "
                f"(1% of {pkr(price)} would only be {pkr(one_percent)}.) That's in addition to the "
                "non-refundable onboarding fee paid at sign-up.")
    if lakh == 5:
        return (f"The commission is {pkr(commission)}: 1% of 5 lakh and the PKR 5,000 minimum are "
                "the same amount here.")
    return (f"{label} is more than 5 lakh, so the commission is 1% of the selling price "
            f"({pkr(price)}): {pkr(commission)}. That's in addition to the non-refundable "
            "onboarding fee paid at sign-up.")


def model_year_answer(year, car=None):
    subject = f"{year} {car}" if car else f"{year} model car"
    if year < 2000:
        return (f"No, Sell It For Me only accepts cars from model year 2000 onward, so a {subject} "
                "isn't eligible. You can still post a free ad yourself.")
    return (f"Yes, a {subject} meets the model-year rule (2000 or newer), as long as it isn't an "
            f"open-letter car and you're in a Sell It For Me city ({city_list(SIFM_CITIES)}).")


OPEN_LETTER_ANSWER = (
    "No, open-letter cars aren't eligible for Sell It For Me. The car needs to be transferred "
    "properly first; PakWheels' registration and transfer service can help in Lahore, Karachi "
    "and Islamabad.")
AUCTION_FEE_ANSWER = (
    "The fee isn't listed in my information; it's shown on the Auction Sheet Verification page "
    "when you enter the chassis number and check out. The original sheet is then sent to you by "
    "SMS or email.")


# --- Reading the question -----------------------------------------------------

SERVICE_RES = {
    "auction": re.compile(r"auction[- ]?sheet", re.I),
    # "featur", typos like "fetured", and "keep my ad/listing on top"
    "featured": re.compile(r"\bfea?tu?r|\bboost|\b(?:ad|listing)\b.{0,30}\b(?:on top|top (?:par|pe|of))",
                           re.I),
    "commission": re.compile(r"\bcommission\b|pakwheels'? cut\b|\btheir cut\b", re.I),
    "sifm": re.compile(r"sell it for me|\bsifm\b|sell (?:my|the|our|his|her) (?:car|gari|gaari|gadi) "
                       r"for (?:me|us|him|her)", re.I),
    "transfer": re.compile(r"\btransfer|\bregist", re.I),
    "inspection": re.compile(r"inspect|\bpdi\b|pre-? ?delivery|check karwa", re.I),
}
PDI_RE = re.compile(r"\bpdi\b|pre-? ?delivery|new car inspection|showroom|brand[- ]new|\bzero ?meter\b"
                    r"|\bnayi gari\b|\bnew (?:car|gari) (?:ki|ka)? ?inspection", re.I)
OPEN_LETTER_RE = re.compile(r"open[- ]?(?:transfer[- ])?letter", re.I)
FEE_WORD_RE = re.compile(r"fee|cost|price|charge|paise|paisay|kitn|kharcha|how much|rate|qeemat", re.I)
SALE_RE = re.compile(r"commission|\bsold\b|sells? for|\bsell for|\bbik(?:i|a|ti|egi|ne|)\b|after the sale"
                     r"|\btake\b|\bcut\b|\bkaat", re.I)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
CC_RE = re.compile(r"\b(\d{3,4})\s?cc\b", re.I)
LITRE_RE = re.compile(r"\b(\d(?:\.\d)?)\s?(?:l|ltr|litres?|liters?)\b", re.I)
LAKH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?|laakh)\b", re.I)
CRORE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*crores?\b", re.I)
RUPEES_RE = re.compile(r"(?:pkr|rs\.?)\s*(\d[\d,]{5,})", re.I)
DAYS_RE = re.compile(r"\b(\d+)\s*(?:days?|din)\b", re.I)
DURATIONS = [
    (re.compile(r"\b(?:7|seven)\s*(?:days?|din)\b|\b(?:a|one|1)\s*week\b|\bhaft[ae]\b", re.I), "7 days"),
    (re.compile(r"\b(?:14|fourteen)\s*(?:days?|din)\b|\b(?:two|2)\s*weeks?\b", re.I), "14 days"),
    (re.compile(r"\b(?:28|twenty[- ]eight)\s*(?:days?|din)\b|\b(?:four|4)\s*weeks?\b"
                r"|\b(?:a|one|1|full)\s*month\b|\bmahin[ae]\b", re.I), "28 days"),
]
CREDITS_RE = re.compile(r"\b(\d+)\s*(?:feature\s*)?credits?\b", re.I)
_ALL_CITIES = sorted(set(INSPECTION_CITIES + SIFM_CITIES + TRANSFER_CITIES + OTHER_CITIES)
                     | set(CITY_ALIASES), key=len, reverse=True)
CITY_RE = re.compile(r"(?<![\w.])(" + "|".join(re.escape(c) for c in _ALL_CITIES) + r")(?![\w])", re.I)
_CANON = {c.lower(): c for c in INSPECTION_CITIES + SIFM_CITIES + TRANSFER_CITIES + OTHER_CITIES}


def _services(text):
    text = OPEN_LETTER_RE.sub(" ", text)
    return [name for name, regex in SERVICE_RES.items() if regex.search(text)]


def _cities(text):
    found = []
    for match in CITY_RE.finditer(text):
        key = match.group(1).lower()
        city = CITY_ALIASES.get(key) or _CANON.get(key)
        if city and city not in found:
            found.append(city)
    return found


def _german_name(brand, model):
    """"mercedes e-class" -> "Mercedes E-Class", "bmw 3 series" -> "BMW 3 Series", "audi a4" -> "Audi A4"."""
    if not model:
        return brand
    lower = model.lower()
    if lower.endswith("-class"):
        model = lower.split("-")[0].upper() + "-Class"
    elif lower.endswith("series"):
        model = lower.split()[0].upper() + " Series"
    else:
        model = model[0].upper() + model[1:]
    return f"{brand} {model}"


def _vehicle(text):
    """(name for answers, band) of the car, engine size or body type named, else None."""
    car = None
    match = GERMAN_RE.search(text)
    if match:
        car = (_german_name(GERMAN_BRANDS[match.group(1).lower()], match.group(2)), "german")
    else:
        for regex, name, band in _CAR_RES:
            if regex.search(text):
                car = (name, band)
                break
    # Body-based bands don't depend on engine size
    if car and car[1] in ("suv", "4x4", "german"):
        return ("car",) + car
    cc = None
    m = CC_RE.search(text)
    if m:
        cc = int(m.group(1))
    else:
        m = LITRE_RE.search(text)
        if m:
            cc = round(float(m.group(1)) * 1000)
    if cc:
        return ("cc", cc, "small" if cc <= 1000 else "mid" if cc <= 2000 else "large")
    if car:
        return ("car",) + car
    m = BODY_RE.search(text)
    if m:
        label, band = BODY[m.group(1).lower()]
        return ("body", label, band)
    return None


def _amount_in_lakh(text):
    m = CRORE_RE.search(text)
    if m:
        return float(m.group(1)) * 100
    m = LAKH_RE.search(text)
    if m:
        return float(m.group(1))
    m = RUPEES_RE.search(text)
    if m:
        value = int(m.group(1).replace(",", ""))
        if value >= 100_000:
            return value / 100_000
    return None


def _price_answer(vehicle, service):
    kind, name, band = vehicle
    if band == "large":
        return LARGE_INSPECTION_ANSWER if service == "inspection" else LARGE_SIFM_ANSWER
    if band == "varies":
        return varies_answer(name, service)
    if kind == "cc":
        lead = cc_phrase(name, band)
    elif kind == "body":
        lead = f"For {name},"
    else:
        lead = car_phrase(name, band)
    return inspection_answer(lead, band) if service == "inspection" else sifm_fee_answer(lead, band)


@dataclass
class Fact:
    service: str
    text: str


def _previous_services(history):
    for message in reversed([m["content"] for m in _history_to_messages(history or [])
                             if m["role"] == "user"]):
        services = _services(message)
        if services:
            return services
    return []


def detect(message: str, history: list = None) -> list:
    """
    Verified facts for the question, verdicts (Yes/No) first. A short follow-up
    with no service named ("aur Ciaz ke liye?") takes the service from the
    previous question; its car or city always comes from the new message.
    """
    services = _services(message)
    if not services and history and len(message.split()) <= 10:
        services = _previous_services(history)
    if not services:
        return []

    facts = []
    cities = _cities(message)
    vehicle = _vehicle(message)
    year = YEAR_RE.search(message)
    open_letter = OPEN_LETTER_RE.search(message)
    sifm = "sifm" in services or "commission" in services

    # Verdicts
    if sifm and open_letter:
        facts.append(Fact("sifm", OPEN_LETTER_ANSWER))
    elif sifm and year:
        car = vehicle[1] if vehicle and vehicle[0] == "car" else None
        facts.append(Fact("sifm", model_year_answer(int(year.group(1)), car)))
    for city in cities:
        if "inspection" in services:
            facts.append(Fact("inspection", inspection_city_answer(city)))
        if "sifm" in services:
            facts.append(Fact("sifm", sifm_city_answer(city)))
        if "transfer" in services:
            facts.append(Fact("transfer", transfer_city_answer(city)))

    # Figures
    if "auction" in services:
        if FEE_WORD_RE.search(message):
            facts.append(Fact("auction", AUCTION_FEE_ANSWER))
        return facts  # a car named here is the one being checked, not priced
    if "featured" in services:
        credits = CREDITS_RE.search(message)
        duration = next((label for regex, label in DURATIONS if regex.search(message)), None)
        if credits:
            facts.append(Fact("featured", bundle_answer(int(credits.group(1)))))
        elif duration:
            facts.append(Fact("featured", featured_answer(duration)))
        elif DAYS_RE.search(message):
            facts.append(Fact("featured", FEATURED_OPTIONS_ANSWER))
    lakh = _amount_in_lakh(message)
    if sifm and lakh and SALE_RE.search(message):
        facts.append(Fact("sifm", commission_answer(lakh)))
    elif "sifm" in services and vehicle and not (open_letter or year):
        facts.append(Fact("sifm", _price_answer(vehicle, "sifm")))
    if "inspection" in services:
        # A new car's inspection is the pre-delivery one, whatever the model
        if PDI_RE.search(message):
            facts.append(Fact("inspection", PDI_ANSWER))
        elif vehicle and "sifm" not in services:
            facts.append(Fact("inspection", _price_answer(vehicle, "inspection")))
    return facts


def fact_block(message: str, history: list = None) -> str:
    facts = detect(message, history)
    return f"{FACT_HEADER}\n" + " ".join(f.text for f in facts) if facts else ""


def build_context(message: str, history: list = None, retrieved: str = "") -> str:
    """Verified answer (if any) first, so context truncation never cuts it, then the KB sections."""
    return "\n---\n".join(part for part in (fact_block(message, history), retrieved) if part)


def strip_fact_block(context: str) -> str:
    if context.startswith(FACT_HEADER):
        return context.split("\n---\n", 1)[1] if "\n---\n" in context else ""
    return context


# --- Guard: no figure the context doesn't contain --------------------------------

FIGURE_RE = re.compile(r"(?:PKR|Rs\.?)\s?\d[\d,]*(?:\.\d+)?|\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?\s?%", re.I)
PAGES = {"inspection": "the Car Inspection page", "sifm": "the Sell It For Me page",
         "commission": "the Sell It For Me page", "featured": "the Feature Your Ad page",
         "auction": "the Auction Sheet Verification page", "transfer": "the Car Registration page"}


def _figure_key(figure):
    return re.sub(r"[^\d.%]", "", figure).rstrip(".")


def unsupported_figures(answer: str, sources: str) -> list:
    allowed = {_figure_key(f) for f in FIGURE_RE.findall(sources)}
    return [f for f in FIGURE_RE.findall(answer) if _figure_key(f) not in allowed]


def guard(answer: str, context: str, message: str = "", history: list = None) -> str:
    """
    Returns the answer, or a safe reply when it quotes a price or percentage that
    is in neither the context, the question nor the earlier (already checked) turns.
    """
    earlier = " ".join(m["content"] for m in _history_to_messages(history or []))
    if not unsupported_figures(answer, " ".join([context or "", message, earlier])):
        return answer
    services = _services(message) or _previous_services(history)
    page = next((PAGES[s] for s in services if s in PAGES), "the relevant page on pakwheels.com")
    return ("I don't have a confirmed figure for that, so I won't guess. Please check "
            f"{page} or call the PakWheels helpline at {HELPLINE}, 9 am to 9 pm every day.")
