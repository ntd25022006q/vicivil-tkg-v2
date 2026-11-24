"""textops: surface normalization, abbreviations, ASCII-folded citation patterns,
typing rules, predicate map. Deterministic only — no LLM here.

IMPORTANT (lesson learned 2026-11-25): U+0111 (đ) has NO NFD decomposition, so naive
diacritic stripping keeps 'đ'. All Vietnamese patterns here are ASCII and MUST be
applied to fold_text(text) output, never raw text.
"""
import re
import unicodedata

# ---------------------------------------------------------------- normalization
def strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")

def fold_text(s: str) -> str:
    """ASCII-folded, casefolded text for regex matching. Length may differ from raw."""
    return strip_diacritics(unicodedata.normalize("NFC", s or "")).casefold()

# Seed abbreviation dictionary (diacritics-stripped lowercase).
ABBREV = {
    "tand": "toa an nhan dan",
    "tandtc": "toa an nhan dan toi cao",
    "vksnd": "vien kiem sat nhan dan",
    "vksndtc": "vien kiem sat nhan dan toi cao",
    "ubnd": "uy ban nhan dan",
    "blhngd": "bo luat hon nhan va gia dinh",
    "blds": "bo luat dan su",
    "blttds": "bo luat to tung dan su",
    "thads": "thi hanh an dan su",
    "hngd": "hon nhan va gia dinh",
    "hotich": "ho tich",
    "ncn": "nuoi con nuoi",
    "vbhn": "van ban hop nhat",
    "ttlt": "thong tu lien tich",
    "ndcp": "nghi dinh chinh phu",
    "nd": "nghi dinh",
    "tt": "thong tu",
    "qh": "quoc hoi",
    "qsdd": "quyen su dung dat",
    "cp": "chinh phu",
    "tp": "thanh pho",
    "hn": "ha noi",
    "hcm": "ho chi minh",
    "cty": "cong ty",
    "tnhh": "trach nhiem huu han",
    "anle": "an le",
    "qsd": "quyen su dung",
}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]|_")

def norm_surface(s: str) -> str:
    """Canonical norm_key: NFC -> casefold -> ASCII-fold (incl. đ) -> expand abbrevs -> strip punct."""
    if not s:
        return ""
    s = strip_diacritics(unicodedata.normalize("NFC", s)).casefold()
    s = _PUNCT.sub(" ", s.replace("/", " "))
    s = _WS.sub(" ", s).strip()
    toks = s.split()
    out, i = [], 0
    while i < len(toks):
        matched = False
        for n in (4, 3, 2, 1):
            if i + n <= len(toks):
                cand = " ".join(toks[i:i + n])
                if cand in ABBREV:
                    out.extend(ABBREV[cand].split())
                    i += n
                    matched = True
                    break
        if not matched:
            out.append(toks[i])
            i += 1
    return " ".join(out)

# ---------------------------------------------------------------- literal detection
_RE_DATE = re.compile(r"^\d{1,2}[/.-]\d{1,2}([/.-]\d{2,4})?$|^\d{4}-\d{2}-\d{2}$")
_RE_DURATION = re.compile(r"^\d+[\s.]*(ngay|thang|nam|gio|phut|tuan|ky|ki)\b")
_RE_MONEY = re.compile(r"^\d[\d.,]*\s*(dong|vnd|usd|trieu|ty|nghin)\b|^\d+[%]")
_RE_NUMBER = re.compile(r"^\d+([.,]\d+)*$")
_RE_DATE_PHRASE = re.compile(r"^(ngay|hieu luc( ke tu)?( ngay)?|biet tich|ke tu)\s*\d")
_RE_FREQ_PHRASE = re.compile(r"^\d+\s*lan\b|^\w+\s*lan trong \d+")

def is_literal(name: str) -> bool:
    n = fold_text(name).strip()
    if not n:
        return True
    if _RE_DATE.match(n) or _RE_DURATION.match(n) or _RE_MONEY.match(n) or _RE_NUMBER.match(n):
        return True
    # date/duration embedded phrases: "Ngày 01 tháng 7 năm 2015", "hiệu lực kể từ ngày...",
    # "biệt tích 02 năm liền trở lên", "05 lần trong 02 ngày"
    return bool(_RE_DATE_PHRASE.match(n) or _RE_FREQ_PHRASE.match(n))

# ---------------------------------------------------------------- typing rules (on RAW name)
RULES = [
    (re.compile(r"^\s*Điều\s+\d+", re.I), "vclo:Article", 0.95, "rule-article"),
    (re.compile(r"\b(tòa án|toa án|TAND)\b", re.I), "vclo:JudicialOrgan", 0.9, "rule-court"),
    (re.compile(r"\bviện kiểm sát\b|\bVK SND\b", re.I), "vclo:JudicialOrgan", 0.9, "rule-proc"),
    (re.compile(r"\b(nghị định|NĐ-CP|nghị quyết)\b", re.I), "vclo:SubLawDecree", 0.85, "rule-decree"),
    (re.compile(r"\bthông tư\b|\bTTLT\b", re.I), "vclo:JointCircular", 0.85, "rule-circular"),
    (re.compile(r"\b(bộ luật|luật)\s+\S+", re.I), "vclo:NormativeDocument", 0.8, "rule-law"),
    (re.compile(r"\b(ủy ban nhân dân|UBND|chính phủ|quốc hội|bộ )", re.I), "vclo:StateOrgan", 0.8, "rule-state"),
    (re.compile(r"\b(hợp đồng|giao dịch|đặt cọc|cọc|thế chấp|cho thuê|gia công|ủy quyền|tặng cho|mua bán)\b", re.I),
     "vclo:LegalConcept", 0.7, "rule-concept"),
    (re.compile(r"\b(quyền|nghĩa vụ)\b", re.I), "vclo:LegalConcept", 0.6, "rule-right"),
]

def classify_entity(name: str):
    if is_literal(name):
        return "vclo:Literal", True, 1.0, "rule-literal"
    for rx, cls, conf, method in RULES:
        if rx.search(name or ""):
            return cls, False, conf, method
    toks = (name or "").strip().split()
    if 2 <= len(toks) <= 4 and all(t.isalpha() for t in toks):
        return "vclo:NaturalPerson", False, 0.5, "rule-person-weak"
    return "vclo:LegalConcept", False, 0.4, "fallback-generic"

# ---------------------------------------------------------------- predicate map
PRED_MAP = {
    "dieu kien": "vclo:hasCondition",
    "dieukien": "vclo:hasCondition",
    "co so": "vclo:hasBasis",
    "can cu": "vclo:hasBasis",
    "quy dinh": "vclo:regulates",
    "qui dinh": "vclo:regulates",
    "quyen": "vclo:grantsRight",
    "nghia vu": "vclo:imposesObligation",
    "cam": "vclo:prohibits",
    "thoi hieu": "vclo:hasStatuteOfLimitations",
    "hieu luc": "vclo:hasValidity",
    "sua doi": "vclo:amends",
    "bo sung": "vclo:amends",
    "bai bo": "vclo:abrogates",
    "ap dung": "vclo:appliesTo",
    "thuoc": "vclo:belongsTo",
    "bao gom": "vclo:includes",
    "gom": "vclo:includes",
    "chuyen nhuong": "vclo:transfersRight",
    "dang ky": "vclo:requiresRegistration",
    "cong chung": "vclo:requiresNotarization",
    "thanh toan": "vclo:requiresPayment",
    "tra": "vclo:requiresPayment",
    "tra lai": "vclo:requiresPayment",
    "tra tien": "vclo:requiresPayment",
    "hoan tra": "vclo:requiresReturn",
    "boi thuong": "vclo:requiresCompensation",
    "phat": "vclo:imposesPenalty",
    "phat coc": "vclo:imposesPenalty",
    "hoan coc": "vclo:requiresReturn",
    "thoa thuan": "vclo:agreedBy",
    "ky ket": "vclo:signedBy",
    "giao ket": "vclo:signedBy",
    "giam ho": "vclo:hasGuardian",
    "quan ly": "vclo:oversees",
    "giam sat": "vclo:supervises",
    "lien quan": "vclo:relatedTo",
    "xu ly": "vclo:handledBy",
    "giai quyet": "vclo:handledBy",
    "thu ly": "vclo:acceptsFiling",
    "khoi kien": "vclo:filesLawsuit",
    "uy quyen": "vclo:delegatesAuthority",
    "vo hieu": "vclo:voidsTransaction",
    "huy bo": "vclo:voidsTransaction",
    "chia": "vclo:divides",
    "quyet dinh": "vclo:decides",
    "ban hanh": "vclo:issues",
    "huong dan": "vclo:guides",
    "thay the": "vclo:replaces",
    "cap": "vclo:issues",
    "dan chieu": "vclo:cites",
    "phien ban khac": "vclo:hasOtherVersion",
    "thuc hien": "vclo:performs",
    "tai san": "vclo:concernsAsset",
    "yeu cau": "vclo:requests",
    "yeucau": "vclo:requests",
    "thoi han": "vclo:hasTimeLimit",
    "thi hanh an": "vclo:enforcesJudgment",
    "thi hanh": "vclo:enforcesJudgment",
    "trach nhiem": "vclo:hasLiability",
    "thong bao": "vclo:notifies",
    "xac dinh": "vclo:determines",
    "so huu": "vclo:concernsOwnership",
    "dai dien": "vclo:represents",
    "thiet hai": "vclo:concernsDamages",
    "hop dong": "vclo:concernsContract",
    "ho tich": "vclo:concernsCivilStatus",
    "hoan thien": "vclo:relatedTo",
    "khuyen mai": "vclo:relatedTo",
    "che hang": "vclo:relatedTo",
}

def map_relation(keyword: str):
    k = norm_surface(keyword)
    if not k:
        return "vclo:relatedTo", False
    if k in PRED_MAP:
        return PRED_MAP[k], True
    toks = k.split()
    for span in (2, 1):
        if len(toks) >= span:
            for i in range(0, len(toks) - span + 1):
                key = " ".join(toks[i:i + span])
                if key in PRED_MAP:
                    return PRED_MAP[key], True
    return "vclo:relatedTo", False

# ---------------------------------------------------------------- citation patterns (ASCII on fold_text)
_RE_DIEU = re.compile(r"dieu\s+(\d+)")
_RE_ARTICLE_AMEND = re.compile(
    r"(?:(?:khoan|dieu)\s+\d+\s+)?dieu\s+(\d+)\s+cua\s+((?:luat|nghi dinh|nghi quyet)\s+so:?\s*[\d/]+[\w-]*)")
_AMEND_WORDS = re.compile(r"sua doi|bo sung|bai bo|thay the")
_RE_AMENDS_TARGET = re.compile(r"mot so dieu cua (bo luat [^,.]*|luat [^,.]*)")

# ASCII law aliases applied on fold_text; (pattern, law_id, needs_year)
LAW_ALIAS_TO_ID = [
    (re.compile(r"bo luat dan su.{0,14}\b2015\b|bo luat dan su.{0,14}91/2015|91/2015/qh13"), "BLDS2015"),
    (re.compile(r"bo luat dan su.{0,14}\b2005\b|bo luat dan su.{0,14}33/2005|33/2005/qh11"), "BLDS2005"),
    (re.compile(r"bo luat dan su.{0,14}\b1995\b|44-l/ctn"), "BLDS1995"),
    (re.compile(r"to tung dan su.{0,14}\b2004\b|24/2004/qh11"), "BLTTDS2004"),
    (re.compile(r"to tung dan su.{0,14}\b2015\b|\b2015\b.{0,14}to tung"), "BLTTDS2015"),
    (re.compile(r"hon nhan va gia dinh.{0,14}\b2014\b|52/2014/qh13"), "HNGD2014"),
    (re.compile(r"hon nhan va gia dinh.{0,14}\b2000\b|22/2000/qh10"), "HNGD2000"),
    (re.compile(r"ho tich.{0,14}\b2026\b|03/2026/qh16"), "HOTICH2026"),
    (re.compile(r"ho tich.{0,14}\b2014\b|60/2014/qh13"), "HOTICH2014"),
    (re.compile(r"nuoi con nuoi.{0,14}\b2010\b|52/2010/qh12"), "NCN2010"),
    (re.compile(r"thi hanh an dan su.{0,14}\b2008\b|26/2008/qh12|58/2008/qh12"), "THADS2008"),
    (re.compile(r"thi hanh an dan su.{0,14}\b2025\b"), "THADS2025"),
    (re.compile(r"hoa giai o co so.{0,14}\b2026\b|16/2026/qh16"), "HOAGIAI2026"),
    (re.compile(r"nghi dinh so:?\s*68/2000"), "HNGD2000ND"),
    (re.compile(r"nghi dinh so:?\s*126/2014"), "ND126-2014"),
    (re.compile(r"nghi dinh so:?\s*158/2005"), "ND158-2005"),
    (re.compile(r"nghi dinh so:?\s*152/2026"), "ND152-2026"),
    (re.compile(r"luat so:?\s*(\d+)/(\d+)/qh15"), "LAW-QH15"),
]

def law_id_from_folded(folded: str):
    """Returns law_id or None. Requires explicit year/so_hieu — year-less mentions are ambiguous by design."""
    for rx, lid in LAW_ALIAS_TO_ID:
        m = rx.search(folded)
        if m:
            if lid == "LAW-QH15":
                return f"Luật số {m.group(1)}/{m.group(2)}/QH15"
            return lid
    return None
