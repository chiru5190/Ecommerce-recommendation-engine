# Synthetic catalog pricing audit

All 72 products remain fictional. Prices are deterministic illustrative INR retail
amounts selected for the stated product type and features. They are not scraped,
live quotes, exchange-rate conversions, verified offers, or a labeled evaluation dataset.
The former values were documented as USD, but were misleading when interpreted as
rupees. All 72 price cells have been repriced; no other catalog cells were changed.

## Findings and assumptions

- Category review found no clear mismatches. A laptop hub, charger, or stand is an
  accessory, not a laptop; generic word matching must not assign laptop price bands.
- No smartwatches, smartphones, laptops, or shoes are present. No products were added
  to fill those types. The user's illustrative bands apply only to types present.
- Existing specification units are consistent: 65W is charger power, 10000mAh is
  battery capacity, 20L/25L is backpack volume, and A5 is paper size. These are
  preserved verbatim so the retrieval vocabulary is unchanged.
- Counts and sizes for some sets, skin-care volumes, pans, and notice boards are
  unspecified. Prices assume ordinary entry/mid-range items or modest sets as
  described; there is insufficient information for unit-price comparisons.
  No counts, volumes, materials, brands, or performance claims were invented.
- Larger or feature-specific products can have different bands: vertical mice vs
  ordinary mice, mechanical vs compact Bluetooth keyboards, large vs smaller
  boards, and foldable travel vs standard cushioned yoga mats.
- CSV `price` remains an ungrouped numeric rupee amount with two decimal places.
  UI cards use ? and Indian grouping without redundant .00; nonzero paise are retained.
  Custom catalogs are also interpreted as INR, with no implicit currency conversion.
- The general loader still accepts zero-priced products; this demo's stronger
  positive-price policy is tested separately to preserve existing loader behavior.

## Reference context

Broad plausibility was cross-checked against Indian retail product listings:
[boAt wireless earbuds](https://www.boat-lifestyle.com/collections/true-wireless-earbuds),
[Decathlon yoga mats](https://www.decathlon.in/c/yoga-mats-and-accessories-26229), and
[IKEA India published price list](https://www.ikea.com/in/en/files/pdf/8a/8a/8a8a1c23/gst-rate-revision.pdf).
These references provide context, not provenance for the fictional products or
validation of every band endpoint. The bands below are engineering assumptions,
including the user's suggested earbuds, headphones, T-shirt, and backpack ranges.
They are not a comprehensive current market survey.

## Verification after repricing

- **72 price cells corrected**, with all IDs, names, categories, descriptions,
  image fields, row ordering, and CSV columns preserved.
- Before/after checks for all 72 queries confirmed identical unfiltered rankings,
  similarity scores, and explanations when price columns were excluded.
- `python -m pytest -q`: **50 passed in 3.69s** on Python 3.12.7.
- `python -m src.evaluation`: 72 queries, 360 recommendations, 100% catalog coverage
  at K=5, and all six sanity checks passed. Category agreement proxy remained
  0.8667; this is not a relevance-quality measure.
- `python scripts/smoke_test.py`: Streamlit health and root endpoints returned
  HTTP 200, and the test subprocess was terminated afterward.
- Price-filter tests verify inclusive INR bounds and category filtering. Existing
  invalid-range UI tests derive a limit above the catalog maximum instead of
  assuming the obsolete low prices.

## Complete price policy

The executable policy is `PRICE_BANDS` in `scripts/generate_demo.py`; fixed amounts
live in its product tuples. Tests check coverage of all IDs, plausible bands,
positive numeric values, formatter output, and exact CSV regeneration.

| Product type | Product IDs | Illustrative INR range |
| --- | --- | --- |
| Wireless over-ear headphones | DEMO001, DEMO002 | 999?7,999 |
| Sport wireless earbuds | DEMO003 | 799?4,999 |
| Portable waterproof Bluetooth speaker | DEMO004 | 699?3,999 |
| USB-C hub with HDMI | DEMO005 | 999?3,999 |
| 65W USB-C charger | DEMO006 | 999?3,499 |
| Wireless office mouse | DEMO007 | 299?1,499 |
| Ergonomic vertical wireless mouse | DEMO008 | 799?2,999 |
| Mechanical keyboard | DEMO009 | 1,499?5,999 |
| Compact Bluetooth keyboard | DEMO010 | 699?2,499 |
| Adjustable aluminum laptop stand | DEMO011 | 499?2,499 |
| 10000mAh USB-C power bank | DEMO012 | 799?2,499 |
| Insulated steel mug/tumbler | DEMO013, DEMO014 | 349?1,499 |
| Ceramic mug | DEMO015 | 149?599 |
| Glass tea pot with steel infuser | DEMO016 | 399?1,499 |
| Bamboo cutting board | DEMO017 | 299?999 |
| Large bamboo chopping board | DEMO018 | 499?1,499 |
| Nonstick frying pan | DEMO019 | 599?1,999 |
| Steel saucepan with glass lid | DEMO020 | 799?2,499 |
| Silicone spatula set | DEMO021 | 199?799 |
| Cotton kitchen towels | DEMO022 | 149?599 |
| Glass food container set | DEMO023 | 599?1,999 |
| Compartment lunch box | DEMO024 | 249?999 |
| Cotton T-shirt / running tee | DEMO025, DEMO026, DEMO028 | 299?1,499 |
| Long sleeve cotton shirt | DEMO027 | 499?1,999 |
| Running shorts | DEMO029 | 399?1,499 |
| Training leggings | DEMO030 | 499?1,999 |
| Fleece hoodie | DEMO031, DEMO032 | 799?2,999 |
| Water-resistant rain jacket | DEMO033 | 799?3,499 |
| Cotton everyday socks | DEMO034 | 99?399 |
| Wool-blend hiking socks | DEMO035 | 249?799 |
| Cotton canvas cap | DEMO036 | 199?799 |
| Cushioned yoga mat | DEMO037 | 499?1,999 |
| Foldable travel yoga mat | DEMO038 | 699?2,499 |
| Foam yoga blocks | DEMO039 | 249?799 |
| Resistance band set | DEMO040 | 299?1,499 |
| Adjustable skipping rope | DEMO041 | 149?699 |
| Insulated steel water bottle | DEMO042, DEMO043 | 399?1,499 |
| 20L/25L hiking backpack | DEMO044, DEMO045 | 499?2,999 |
| Adjustable aluminum trekking poles | DEMO046 | 799?2,999 |
| Rechargeable camping lantern | DEMO047 | 499?1,999 |
| Compact sleeping bag | DEMO048 | 999?3,999 |
| A5 hardcover notebook/journal | DEMO049, DEMO050 | 199?699 |
| Pocket notebook | DEMO051 | 49?249 |
| Gel pen set | DEMO052, DEMO053 | 99?499 |
| Mechanical pencil set | DEMO054 | 99?399 |
| Bamboo desk organizer | DEMO055 | 399?1,499 |
| Mesh desk organizer | DEMO056 | 249?799 |
| Adjustable dimmable LED desk lamp | DEMO057 | 699?2,499 |
| Clip LED reading lamp | DEMO058 | 299?999 |
| Weekly paper planner | DEMO059 | 149?599 |
| Cork notice board | DEMO060 | 349?1,499 |
| Daily facial cleanser | DEMO061, DEMO062 | 199?799 |
| Daily facial moisturizer | DEMO063 | 249?999 |
| Rich night moisturizer | DEMO064 | 349?1,499 |
| Hand cream | DEMO065 | 99?499 |
| Shea body lotion | DEMO066 | 199?799 |
| Bamboo cushioned hair brush | DEMO067 | 199?699 |
| Compact travel hair brush | DEMO068 | 99?399 |
| Reusable cotton makeup pads | DEMO069 | 149?599 |
| Cotton facial towels | DEMO070 | 199?699 |
| Travel toiletry organizer | DEMO071 | 299?999 |
| Hanging toiletry bag | DEMO072 | 399?1,499 |
