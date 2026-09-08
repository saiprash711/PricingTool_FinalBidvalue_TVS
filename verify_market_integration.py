import market_scraper
from model_engine import VehiclePricePredictor

predictor = VehiclePricePredictor(data_path="cleaned_tn_vehicle_data.csv")

test_cases = [
    {"make": "Maruti", "model": "Swift", "year": 2021, "variant": "ZXI", "odo": 65000, "branch": "All Tamil Nadu"},
    {"make": "Hyundai", "model": "i20", "year": 2018, "variant": "ASTA", "odo": 55000, "branch": "Chennai Metro"},
    {"make": "Honda", "model": "City", "year": 2019, "variant": "ZX", "odo": 45000, "branch": "All Tamil Nadu"},
]

print("=== VERIFYING INTEGRATED VALUATION & MARKET COMPS ===")
for tc in test_cases:
    pred = predictor.predict(
        tc["make"],
        tc["model"],
        tc["year"],
        tc["odo"],
        transmission="Manual",
        fuel="Petrol",
        owner="1st Owner",
        branch=tc["branch"],
        variant=tc["variant"]
    )
    
    comps = market_scraper.fetch_cars24_live_comps(
        make=tc["make"],
        model=tc["model"],
        branch=tc["branch"],
        target_year=tc["year"],
        target_variant=tc["variant"],
        max_results=4
    )
    
    metrics = market_scraper.compute_market_comparison_metrics(
        pred["expected_price"],
        comps
    )
    
    links = market_scraper.generate_external_portal_links(tc["make"], tc["model"], tc["branch"])
    
    print(f"\n--- {tc['year']} {tc['make']} {tc['model']} ({tc['variant']}) ---")
    print(f"TVS Model Winning Bid: Rs {pred['expected_price']:,}")
    print(f"Cars24 Live Comps Count: {metrics['comps_count']}")
    if metrics['has_data']:
        print(f"Cars24 Median Retail: Rs {metrics['median_retail']:,}")
        print(f"Retail Spread: Rs {metrics['spread_amount']:,} ({metrics['spread_pct']:+.1f}%)")
        print(f"Arbitrage Verdict: {metrics['arbitrage_verdict']}")
        for c in metrics['comps'][:2]:
            print(f"  • {c['title']} | Rs {c['price']:,} | Img: {c['image'][:60]}...")
    else:
        print("No live comps found on Cars24")
    print(f"External links: Cars24: {links['cars24']}")

print("\nVerification successfully finished!")
