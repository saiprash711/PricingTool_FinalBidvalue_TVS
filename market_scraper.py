"""
market_scraper.py
Live Market Intelligence & Comparable Listings Extractor
Fetches real-time retail listings, images, and prices from Cars24, Spinny, and OLX.
Provides side-by-side benchmark comparison against TVS Certified B2B Auction Valuation.
"""

import os
import re
import ssl
import json
import urllib.request
import urllib.parse
from typing import Dict, List, Any, Optional

# SSL context for resilient HTTPS requests
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Model slug aliases for Cars24 URL matching
MODEL_SLUG_MAP = {
    "SWIFT": "swift",
    "SWIFT DZIRE": "dzire",
    "DZIRE": "dzire",
    "BALENO": "baleno",
    "WAGON R": "wagon-r",
    "WAGONR": "wagon-r",
    "ALTO": "alto",
    "ALTO 800": "alto-800",
    "ALTO K10": "alto-k10",
    "CELERIO": "celerio",
    "ERTIGA": "ertiga",
    "BREZZA": "vitara-brezza",
    "VITARA BREZZA": "vitara-brezza",
    "I20": "i20",
    "ELITE I20": "elite-i20",
    "NEW I20": "i20",
    "I10": "i10",
    "GRAND I10": "grand-i10",
    "GRAND I10 NIOS": "grand-i10-nios",
    "CRETA": "creta",
    "VERNA": "verna",
    "VENUE": "venue",
    "SANTRO": "santro",
    "CITY": "city",
    "AMAZE": "amaze",
    "CIVIC": "civic",
    "JAZZ": "jazz",
    "WR-V": "wr-v",
    "WRV": "wr-v",
    "NEXON": "nexon",
    "TIAGO": "tiago",
    "ALTROZ": "altroz",
    "HARRIER": "harrier",
    "SAFARI": "safari",
    "PUNCH": "punch",
    "INNOVA": "innova",
    "INNOVA CRYSTA": "innova-crysta",
    "FORTUNER": "fortuner",
    "ETIOS": "etios",
    "GLANZA": "glanza",
    "URBAN CRUISER": "urban-cruiser",
    "POLO": "polo",
    "VENTO": "vento",
    "TAIGUN": "taigun",
    "RAPID": "rapid",
    "KUSHAQ": "kushaq",
    "SLAVIA": "slavia",
    "KWID": "kwid",
    "TRIBER": "triber",
    "DUSTER": "duster",
    "ECOSPORT": "ecosport",
    "FIGO": "figo",
    "ENDEAVOUR": "endeavour",
    "SELTOS": "seltos",
    "SONET": "sonet",
    "CARENS": "carens",
    "THAR": "thar",
    "SCORPIO": "scorpio",
    "SCORPIO-N": "scorpio-n",
    "XUV700": "xuv700",
    "XUV500": "xuv500",
    "XUV300": "xuv300",
    "BOLERO": "bolero",
}

CITY_SLUG_MAP = {
    "Chennai": "chennai",
    "Coimbatore": "coimbatore",
    "Madurai": "chennai",    # Fallback to nearest major Cars24 hub
    "Trichy": "chennai",
    "Tiruchirappalli": "chennai",
    "Salem": "coimbatore",
    "Tirunelveli": "chennai",
    "Vellore": "chennai",
    "All Tamil Nadu": "chennai",
    "Kochi": "kochi",
    "Hyderabad": "hyderabad",
    "Bangalore": "bengaluru",
}


def get_model_slug(model_name: str) -> str:
    """Normalize model string to URL-friendly slug."""
    clean = model_name.strip().upper()
    if clean in MODEL_SLUG_MAP:
        return MODEL_SLUG_MAP[clean]
    # Fallback to kebab-case
    return re.sub(r'[^a-zA-Z0-9]+', '-', clean).strip('-').lower()


def get_make_slug(make_name: str) -> str:
    """Normalize make string to URL-friendly slug."""
    clean = make_name.strip().lower()
    if "maruti" in clean:
        return "maruti"
    return re.sub(r'[^a-zA-Z0-9]+', '-', clean).strip('-')


def generate_external_portal_links(make: str, model: str, branch: str = "Chennai") -> Dict[str, str]:
    """Generate pre-filtered direct search URLs for Cars24, Spinny, and OLX."""
    make_clean = make.strip()
    model_clean = model.strip()
    make_slug = get_make_slug(make_clean)
    model_slug = get_model_slug(model_clean)
    city_slug = CITY_SLUG_MAP.get(branch, "chennai")
    query_str = urllib.parse.quote_plus(f"{make_clean} {model_clean}")

    return {
        "cars24": f"https://www.cars24.com/buy-used-{make_slug}-{model_slug}-cars-{city_slug}/",
        "spinny": f"https://www.spinny.com/used-{make_slug}-{model_slug}-cars-in-{city_slug}/s/",
        "olx": f"https://www.olx.in/tamil-nadu_g4059014/cars_c84?query={query_str}"
    }


def fetch_cars24_live_comps(
    make: str, 
    model: str, 
    branch: str = "Chennai",
    target_year: Optional[int] = None,
    target_variant: Optional[str] = None,
    max_results: int = 12,
    timeout_sec: int = 7
) -> List[Dict[str, Any]]:
    """
    Scrape live retail listings from Cars24 for the specified make and model in Tamil Nadu.
    Extracts real photos, verified retail prices, specifications, and listing URLs.
    """
    make_slug = get_make_slug(make)
    model_slug = get_model_slug(model)
    city_slug = CITY_SLUG_MAP.get(branch, "chennai")
    
    primary_url = f"https://www.cars24.com/buy-used-{make_slug}-{model_slug}-cars-{city_slug}/"
    fallback_url = f"https://www.cars24.com/buy-used-{make_slug}-cars-{city_slug}/"
    
    html = ""
    used_url = primary_url
    
    try:
        req = urllib.request.Request(primary_url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout_sec, context=_SSL_CTX) as res:
            html = res.read().decode('utf-8', errors='ignore')
    except Exception:
        # Fallback to make-level listing
        try:
            used_url = fallback_url
            req = urllib.request.Request(fallback_url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout_sec, context=_SSL_CTX) as res:
                html = res.read().decode('utf-8', errors='ignore')
        except Exception:
            return []

    if not html:
        return []

    # 1. Map appointment ID to listing price from meta blocks
    price_by_id = {}
    matches_price = re.findall(r'-(\d{9,13})/\\*\",\\*\"meta.+?cumulativeListingPrice[^\d]*(\d+)', html)
    for apt_id, price in matches_price:
        try:
            p_val = int(price)
            if 50000 <= p_val <= 20000000:
                price_by_id[apt_id] = p_val
        except ValueError:
            pass

    # 2. Map appointment ID to odometer reading
    odo_by_id = {}
    matches_odo = re.findall(r'appointmentId\\*\":\\*\"(\d{9,13})\\*\".+?odometerReading\\*\":\s*(\d+)', html)
    for apt_id, odo in matches_odo:
        try:
            o_val = int(odo)
            if 100 <= o_val <= 500000:
                odo_by_id[apt_id] = o_val
        except ValueError:
            pass

    # 3. Extract items from JSON-LD Schema.org graph
    ld_scripts = re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.DOTALL)
    cars = []
    seen_ids = set()

    for s in ld_scripts:
        try:
            d = json.loads(s)
            graph = d.get('@graph', []) if isinstance(d, dict) else []
            for item in graph:
                if item.get('@type') == 'ListItem':
                    u = item.get('url', '')
                    m_id = re.search(r'-(\d+)/?$', u)
                    apt_id = m_id.group(1) if m_id else ""
                    if not apt_id or apt_id in seen_ids:
                        continue
                    seen_ids.add(apt_id)

                    title = item.get('name', '').strip()
                    
                    # Filter: if we fell back to make URL, ensure model matches
                    if model.lower() not in title.lower() and used_url == fallback_url:
                        continue

                    # Extract year from title
                    year = None
                    y_m = re.match(r'^(\d{4})', title)
                    if y_m:
                        year = int(y_m.group(1))

                    # Format image URL
                    img = item.get('image', '')
                    if img.startswith("https://media.cars24.com/https://"):
                        img = img.replace("https://media.cars24.com/https://", "https://")

                    price = price_by_id.get(apt_id)
                    odo = odo_by_id.get(apt_id)
                    desc = item.get('description', '')

                    # Extract fuel & transmission from description or title
                    fuel = "Petrol" if "petrol" in (desc + title).lower() else ("Diesel" if "diesel" in (desc + title).lower() else ("CNG" if "cng" in (desc + title).lower() else "Petrol"))
                    transmission = "Automatic" if ("automatic" in (desc + title).lower() or "at" in title.lower().split() or "cvt" in title.lower().split() or "amt" in title.lower().split()) else "Manual"

                    # Variant name (remove year and make/model from title)
                    variant_guess = title
                    if year:
                        variant_guess = variant_guess.replace(str(year), "").strip()
                    variant_guess = re.sub(r'(?i)' + re.escape(make), '', variant_guess).strip()
                    variant_guess = re.sub(r'(?i)' + re.escape(model), '', variant_guess).strip()

                    cars.append({
                        "title": title,
                        "variant": variant_guess or "Standard",
                        "year": year,
                        "price": price,
                        "odometer": odo,
                        "fuel": fuel,
                        "transmission": transmission,
                        "url": u,
                        "image": img,
                        "description": desc,
                        "apt_id": apt_id,
                        "source": "Cars24"
                    })
        except Exception:
            pass

    # Filter out entries with invalid price
    valid_cars = [c for c in cars if c.get('price') and c['price'] > 50000]

    # Sort comps by proximity to target year and variant if provided
    if target_year:
        def sort_key(c):
            yr = c.get('year') or target_year
            yr_diff = abs(yr - target_year)
            var_match = 0
            if target_variant and target_variant.lower() in c.get('title', '').lower():
                var_match = -10  # prioritize exact variant
            return (var_match, yr_diff)
        valid_cars.sort(key=sort_key)

    return valid_cars[:max_results]


def compute_market_comparison_metrics(
    model_expected_price: float,
    live_comps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate retail benchmark summary metrics and wholesale-to-retail spread.
    
    TVS Model: Wholesale / B2B Auction Hammer Price
    Cars24 Comps: Retail / B2C Dealer Asking Price
    """
    if not live_comps:
        return {
            "has_data": False,
            "comps_count": 0,
            "median_retail": 0,
            "mean_retail": 0,
            "min_retail": 0,
            "max_retail": 0,
            "spread_amount": 0,
            "spread_pct": 0.0,
            "dealer_margin_pct": 0.0,
            "arbitrage_verdict": "No live comps available for comparison",
            "comps": []
        }

    prices = [c['price'] for c in live_comps if c.get('price')]
    if not prices:
        return {
            "has_data": False,
            "comps_count": 0,
            "comps": []
        }

    prices.sort()
    n = len(prices)
    median_retail = prices[n // 2] if n % 2 == 1 else int((prices[n // 2 - 1] + prices[n // 2]) / 2)
    mean_retail = int(sum(prices) / n)
    min_retail = prices[0]
    max_retail = prices[-1]

    spread_amount = median_retail - int(round(model_expected_price))
    spread_pct = (spread_amount / model_expected_price * 100) if model_expected_price > 0 else 0.0

    # Business classification of the auction vs retail spread
    # Typical healthy dealer spread is +12% to +25%
    if spread_pct >= 18.0:
        verdict = "High Arbitrage Margin (Wide Retail Spread)"
        verdict_color = "#22c55e"
    elif spread_pct >= 10.0:
        verdict = "Normal Dealer Margin (Balanced Liquidity)"
        verdict_color = "#38bdf8"
    elif spread_pct >= 0.0:
        verdict = "Tight Margin (High Competition / Inelastic)"
        verdict_color = "#f59e0b"
    else:
        verdict = "Auction Premium (Retail Discounted Comps)"
        verdict_color = "#ef4444"

    return {
        "has_data": True,
        "comps_count": len(live_comps),
        "median_retail": median_retail,
        "mean_retail": mean_retail,
        "min_retail": min_retail,
        "max_retail": max_retail,
        "spread_amount": spread_amount,
        "spread_pct": spread_pct,
        "arbitrage_verdict": verdict,
        "verdict_color": verdict_color,
        "comps": live_comps
    }
