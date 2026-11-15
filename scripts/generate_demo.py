"""Generate fictional demo data with fixed illustrative INR prices, not live quotes."""

import csv
from pathlib import Path

# Product-type plausibility bands in INR, not market observations or hard retail limits.
# IDs make the price policy auditable without changing catalog metadata or its schema.
# Sets are priced as the described item/set; unspecified volumes/counts are not inferred.
PRICE_BANDS = [
    ("Wireless over-ear headphones", (1, 2), 999, 7999),
    ("Sport wireless earbuds", (3,), 799, 4999),
    ("Portable waterproof Bluetooth speaker", (4,), 699, 3999),
    ("USB-C hub with HDMI", (5,), 999, 3999),
    ("65W USB-C charger", (6,), 999, 3499),
    ("Wireless office mouse", (7,), 299, 1499),
    ("Ergonomic vertical wireless mouse", (8,), 799, 2999),
    ("Mechanical keyboard", (9,), 1499, 5999),
    ("Compact Bluetooth keyboard", (10,), 699, 2499),
    ("Adjustable aluminum laptop stand", (11,), 499, 2499),
    ("10000mAh USB-C power bank", (12,), 799, 2499),
    ("Insulated steel mug/tumbler", (13, 14), 349, 1499),
    ("Ceramic mug", (15,), 149, 599),
    ("Glass tea pot with steel infuser", (16,), 399, 1499),
    ("Bamboo cutting board", (17,), 299, 999),
    ("Large bamboo chopping board", (18,), 499, 1499),
    ("Nonstick frying pan", (19,), 599, 1999),
    ("Steel saucepan with glass lid", (20,), 799, 2499),
    ("Silicone spatula set", (21,), 199, 799),
    ("Cotton kitchen towels", (22,), 149, 599),
    ("Glass food container set", (23,), 599, 1999),
    ("Compartment lunch box", (24,), 249, 999),
    ("Cotton T-shirt / running tee", (25, 26, 28), 299, 1499),
    ("Long sleeve cotton shirt", (27,), 499, 1999),
    ("Running shorts", (29,), 399, 1499),
    ("Training leggings", (30,), 499, 1999),
    ("Fleece hoodie", (31, 32), 799, 2999),
    ("Water-resistant rain jacket", (33,), 799, 3499),
    ("Cotton everyday socks", (34,), 99, 399),
    ("Wool-blend hiking socks", (35,), 249, 799),
    ("Cotton canvas cap", (36,), 199, 799),
    ("Cushioned yoga mat", (37,), 499, 1999),
    ("Foldable travel yoga mat", (38,), 699, 2499),
    ("Foam yoga blocks", (39,), 249, 799),
    ("Resistance band set", (40,), 299, 1499),
    ("Adjustable skipping rope", (41,), 149, 699),
    ("Insulated steel water bottle", (42, 43), 399, 1499),
    ("20L/25L hiking backpack", (44, 45), 499, 2999),
    ("Adjustable aluminum trekking poles", (46,), 799, 2999),
    ("Rechargeable camping lantern", (47,), 499, 1999),
    ("Compact sleeping bag", (48,), 999, 3999),
    ("A5 hardcover notebook/journal", (49, 50), 199, 699),
    ("Pocket notebook", (51,), 49, 249),
    ("Gel pen set", (52, 53), 99, 499),
    ("Mechanical pencil set", (54,), 99, 399),
    ("Bamboo desk organizer", (55,), 399, 1499),
    ("Mesh desk organizer", (56,), 249, 799),
    ("Adjustable dimmable LED desk lamp", (57,), 699, 2499),
    ("Clip LED reading lamp", (58,), 299, 999),
    ("Weekly paper planner", (59,), 149, 599),
    ("Cork notice board", (60,), 349, 1499),
    ("Daily facial cleanser", (61, 62), 199, 799),
    ("Daily facial moisturizer", (63,), 249, 999),
    ("Rich night moisturizer", (64,), 349, 1499),
    ("Hand cream", (65,), 99, 499),
    ("Shea body lotion", (66,), 199, 799),
    ("Bamboo cushioned hair brush", (67,), 199, 699),
    ("Compact travel hair brush", (68,), 99, 399),
    ("Reusable cotton makeup pads", (69,), 149, 599),
    ("Cotton facial towels", (70,), 199, 699),
    ("Travel toiletry organizer", (71,), 299, 999),
    ("Hanging toiletry bag", (72,), 399, 1499),
]

GROUPS = {
    "Electronics": [
        ("Wireless Studio Headphones", "Over-ear bluetooth headphones with noise isolation and padded ear cups", 2499),
        ("Wireless Travel Headphones", "Foldable over-ear bluetooth headphones with noise isolation for travel", 1999),
        ("Sport Wireless Earbuds", "Sweat resistant bluetooth earbuds with secure fit for running", 1299),
        ("Compact Bluetooth Speaker", "Portable waterproof bluetooth speaker with rechargeable battery", 1499),
        ("USB-C Laptop Hub", "USB-C hub with HDMI display output and USB ports for laptop", 1799),
        ("USB-C Travel Charger", "Compact 65W USB-C charger for laptop phone and tablet", 1999),
        ("Wireless Office Mouse", "Quiet ergonomic wireless mouse with adjustable tracking", 599),
        ("Ergonomic Vertical Mouse", "Vertical wireless mouse with wrist support for office work", 1499),
        ("Mechanical Desk Keyboard", "Mechanical keyboard with tactile switches and USB-C cable", 2499),
        ("Compact Travel Keyboard", "Slim bluetooth keyboard for tablet and laptop", 1299),
        ("Adjustable Laptop Stand", "Aluminum ventilated laptop stand with adjustable height", 999),
        ("Portable Power Bank", "10000mAh rechargeable battery with USB-C fast charging", 1299),
    ],
    "Home & Kitchen": [
        ("Stainless Steel Travel Mug", "Insulated stainless steel coffee mug with leak resistant lid", 599),
        ("Insulated Coffee Tumbler", "Double wall stainless steel coffee tumbler with lid", 699),
        ("Ceramic Breakfast Mug", "Large ceramic coffee tea mug with comfortable handle", 249),
        ("Glass Tea Infuser", "Glass tea pot with removable stainless steel infuser", 799),
        ("Bamboo Cutting Board", "Bamboo kitchen cutting board for vegetables with juice groove", 499),
        ("Large Bamboo Chopping Board", "Large bamboo chopping board for kitchen food preparation", 799),
        ("Nonstick Frying Pan", "Nonstick aluminum frying pan with heat resistant handle", 999),
        ("Stainless Steel Saucepan", "Stainless steel saucepan with glass lid for stovetop cooking", 1299),
        ("Silicone Spatula Set", "Heat resistant silicone spatula utensils for nonstick cookware", 399),
        ("Cotton Kitchen Towels", "Absorbent cotton dish towels for kitchen cleaning", 249),
        ("Glass Food Containers", "Stackable glass food storage containers with sealed lids", 999),
        ("Lunch Storage Box", "Reusable sealed food storage lunch container with compartments", 449),
    ],
    "Clothing": [
        ("Classic Cotton T-Shirt", "Soft breathable cotton crew neck short sleeve casual shirt", 399),
        ("Relaxed Cotton T-Shirt", "Relaxed fit breathable cotton crew neck casual shirt", 499),
        ("Long Sleeve Cotton Shirt", "Soft cotton long sleeve casual shirt for layering", 799),
        ("Lightweight Running Tee", "Quick drying breathable polyester running sports shirt", 599),
        ("Performance Running Shorts", "Quick drying lightweight sports shorts with zip pocket", 699),
        ("Stretch Training Leggings", "Stretch breathable sports leggings with side pocket", 999),
        ("Fleece Zip Hoodie", "Warm fleece hooded sweatshirt with full zip and pockets", 1499),
        ("Fleece Pullover Hoodie", "Warm fleece pullover hooded sweatshirt with front pocket", 1299),
        ("Water Resistant Rain Jacket", "Lightweight hooded rain jacket for outdoor travel", 1799),
        ("Cotton Everyday Socks", "Breathable cotton cushioned everyday ankle socks", 199),
        ("Thermal Hiking Socks", "Warm cushioned wool blend hiking socks for outdoor boots", 399),
        ("Canvas Casual Cap", "Adjustable cotton canvas baseball cap for everyday wear", 299),
    ],
    "Sports & Outdoors": [
        ("Cushioned Yoga Mat", "Non-slip cushioned exercise yoga mat for floor stretching", 799),
        ("Travel Yoga Mat", "Foldable non-slip lightweight yoga mat for stretching", 999),
        ("Foam Yoga Blocks", "Supportive foam yoga blocks for balance and stretching", 399),
        ("Resistance Band Set", "Elastic resistance exercise bands for strength training", 599),
        ("Adjustable Jump Rope", "Lightweight adjustable skipping jump rope for cardio exercise", 249),
        ("Steel Water Bottle", "Insulated stainless steel water bottle for hiking and gym", 699),
        ("Trail Hydration Bottle", "Leak resistant insulated water bottle for outdoor hiking", 799),
        ("Compact Hiking Backpack", "Lightweight 20L hiking backpack with water bottle pockets", 1299),
        ("Day Trail Backpack", "Durable 25L hiking daypack with padded straps and pockets", 1799),
        ("Adjustable Trekking Poles", "Aluminum adjustable trekking poles with grip handles for hiking", 1499),
        ("Rechargeable Camping Lantern", "Portable rechargeable LED camping lantern for outdoor tents", 999),
        ("Compact Sleeping Bag", "Warm lightweight sleeping bag for camping and outdoor travel", 1999),
    ],
    "Office & Stationery": [
        ("Dotted Journal", "A5 dotted paper notebook journal with hardcover for planning", 349),
        ("Ruled Hardcover Notebook", "A5 ruled paper notebook with hardcover for office notes", 299),
        ("Pocket Travel Notebook", "Small ruled paper notebook for travel notes and lists", 99),
        ("Gel Pen Set", "Smooth writing black ink gel pens for notebooks and office", 149),
        ("Fine Point Gel Pens", "Fine point black ink gel pens for precise writing", 199),
        ("Mechanical Pencil Set", "Refillable mechanical pencils for writing drawing and study", 149),
        ("Bamboo Desk Organizer", "Bamboo desktop organizer for stationery pens and office supplies", 699),
        ("Mesh Desk Organizer", "Metal mesh desktop stationery organizer with pen compartments", 399),
        ("Adjustable LED Desk Lamp", "Dimmable LED reading desk lamp with adjustable arm", 1299),
        ("Clip Reading Lamp", "Compact dimmable LED reading lamp with flexible arm", 599),
        ("Weekly Desk Planner", "Undated paper weekly planner for tasks and office scheduling", 249),
        ("Cork Notice Board", "Wall mounted cork board for office notes and planning", 599),
    ],
    "Beauty & Personal Care": [
        ("Gentle Face Cleanser", "Fragrance free gentle facial cleanser for daily skincare", 299),
        ("Hydrating Face Cleanser", "Gentle hydrating facial cleanser for daily dry skin care", 349),
        ("Daily Face Moisturizer", "Lightweight fragrance free hydrating facial moisturizer", 399),
        ("Rich Night Moisturizer", "Rich hydrating night facial moisturizer for dry skin", 549),
        ("Unscented Hand Cream", "Fragrance free moisturizing hand cream for dry skin", 199),
        ("Shea Body Lotion", "Shea butter moisturizing body lotion for dry skin", 299),
        ("Bamboo Hair Brush", "Bamboo handle cushioned hair brush for daily grooming", 299),
        ("Travel Hair Brush", "Compact cushioned hair brush for travel grooming", 149),
        ("Cotton Makeup Pads", "Reusable washable cotton facial makeup removal pads", 249),
        ("Cotton Face Towels", "Soft reusable cotton facial towels for daily cleansing", 299),
        ("Travel Toiletry Bag", "Water resistant travel toiletry organizer with zip compartments", 499),
        ("Hanging Wash Bag", "Hanging travel toiletry wash bag with organizer pockets", 699),
    ],
}


def main(path: Path | None = None) -> None:
    """Write the bundled CSV in exclusive-create mode to preserve existing data."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "sample_products.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["product_id", "name", "category", "description", "price", "image_url"])
        number = 0
        for category, products in GROUPS.items():
            for name, description, price in products:
                number += 1
                writer.writerow([f"DEMO{number:03d}", name, category, description, f"{price:.2f}", ""])
    print(f"Created {number} fictional demo products with illustrative INR prices: {path}")


if __name__ == "__main__":
    main()
