import pandas as pd

from pipeline import clean_books


def test_clean_books():
    raw = pd.DataFrame([{
        "title": "Example", "price": "£10.00", "star_rating": "Four",
        "availability": "In stock (3 available)", "category": "Test"
    }])
    row = clean_books(raw).iloc[0]
    assert row["price_gbp"] == 10.0
    assert row["price_inr"] == 1055.0
    assert row["rating"] == 4
    assert bool(row["in_stock"]) is True
