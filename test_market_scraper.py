import market_scraper

comps = market_scraper.fetch_cars24_live_comps('Maruti', 'Swift', 'Chennai', target_year=2021, target_variant='ZXI')
print(f"Retrieved {len(comps)} comps")
for c in comps[:4]:
    p = c.get('price')
    p_str = f"Rs {p:,}" if p else "N/A"
    o = c.get('odometer')
    o_str = f"{o:,} km" if o else "N/A km"
    print(f"- {c['title']} | {p_str} | {o_str}")
    print(f"  Img: {c['image'][:75]}...")
    print(f"  URL: {c['url']}")

metrics = market_scraper.compute_market_comparison_metrics(516295, comps)
print("\nMarket Comparison Metrics:")
print(f"Median Retail: Rs {metrics['median_retail']:,}")
print(f"Spread Amount: Rs {metrics['spread_amount']:,}")
print(f"Spread Pct: {metrics['spread_pct']:.1f}%")
print(f"Verdict: {metrics['arbitrage_verdict']}")

links = market_scraper.generate_external_portal_links('Maruti', 'Swift', 'Chennai')
print("\nExternal Links:")
for portal, link in links.items():
    print(f"  {portal}: {link}")
