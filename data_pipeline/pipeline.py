from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
OUT_DIR = Path(__file__).resolve().parent


# STEP 1: Download one webpage and convert its HTML into BeautifulSoup.
def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


# STEP 2: Collect category names and links from the home page.
def category_links(limit: int = 3) -> list[tuple[str, str]]:
    soup = get_soup(BASE_URL)
    links = []
    for anchor in soup.select(".side_categories ul li ul li a")[:limit]:
        links.append((anchor.get_text(strip=True), urljoin(BASE_URL, anchor["href"])))
    return links


# STEP 3: Scrape every page belonging to one category.
def scrape_category(category: str, first_url: str) -> list[dict]:
    rows = []
    url = first_url
    while url:
        soup = get_soup(url)
        for book in soup.select("article.product_pod"):
            rating_classes = book.select_one("p.star-rating").get("class", [])
            rating_text = next((x for x in rating_classes if x != "star-rating"), None)
            rows.append({
                "title": book.select_one("h3 a").get("title", "").strip(),
                "price": book.select_one("p.price_color").get_text(strip=True),
                "star_rating": rating_text,
                "availability": book.select_one("p.instock.availability").get_text(" ", strip=True),
                "category": category,
            })
        nxt = soup.select_one("li.next a")
        url = urljoin(url, nxt["href"]) if nxt else None
    return rows


# STEP 4: Continue through categories until we have at least 60 books
# from at least 3 categories.
def scrape_books(minimum: int = 60) -> pd.DataFrame:
    rows = []
    for category, url in category_links(limit=8):
        rows.extend(scrape_category(category, url))
        if len(rows) >= minimum and len({r["category"] for r in rows}) >= 3:
            break
    if len(rows) < minimum or len({r["category"] for r in rows}) < 3:
        raise RuntimeError("Scraping did not produce 60 books from 3 categories")
    return pd.DataFrame(rows)


# STEP 5: Clean price, rating and availability, then calculate INR price.
def clean_books(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["price_gbp"] = pd.to_numeric(
        df["price"].str.replace(r"[^0-9.]", "", regex=True), errors="coerce"
    )
    if df["price_gbp"].notna().any():
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())
    ratings = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    df["rating"] = df["star_rating"].map(ratings).astype("Int64")
    df["in_stock"] = df["availability"].str.contains(r"\bIn stock\b", case=False, na=False)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)
    df = df.dropna(subset=["title", "category", "price_gbp", "rating"]).copy()
    df["rating"] = df["rating"].astype(int)
    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


# STEP 6: Create the normalized categories and books tables.
def create_database(df: pd.DataFrame, db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;
            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL UNIQUE
            );
            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL,
                price_inr REAL NOT NULL,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
                category_id INTEGER NOT NULL REFERENCES categories(category_id)
            );
        """)
        categories = sorted(df["category"].unique())
        conn.executemany("INSERT INTO categories(category_name) VALUES (?)", [(x,) for x in categories])
        ids = dict(conn.execute("SELECT category_name, category_id FROM categories"))
        rows = []
        for book in df.itertuples(index=False):
            row = (
                book.title,
                book.price_gbp,
                book.price_inr,
                book.rating,
                int(book.in_stock),
                ids[book.category],
            )
            rows.append(row)
        conn.executemany("""
            INSERT INTO books(title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)


QUERIES = {
    "in_stock": "SELECT title, price_inr FROM books WHERE in_stock = 1 LIMIT 10",
    "most_expensive": "SELECT title, price_gbp FROM books ORDER BY price_gbp DESC LIMIT 10",
    "distinct_ratings": "SELECT DISTINCT rating FROM books ORDER BY rating",
    "price_range": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20 AND 40 ORDER BY price_gbp",
    "top_ratings": "SELECT title, rating FROM books WHERE rating IN (4, 5) ORDER BY rating DESC LIMIT 20",
    "join": """SELECT b.title, b.rating, c.category_name
               FROM books b JOIN categories c ON b.category_id = c.category_id
               ORDER BY c.category_name, b.title""",
}


# STEP 7: Run all required SQL queries and verify JOIN == pandas.merge.
def run_queries(db_path: Path, clean_df: pd.DataFrame) -> None:
    output = []
    with sqlite3.connect(db_path) as conn:
        results = {}
        for name, query in QUERIES.items():
            results[name] = pd.read_sql_query(query, conn)
        for name, frame in results.items():
            output.extend([f"## {name}\n{QUERIES[name]}", frame.to_string(index=False), ""])
        sql_join = results["join"].reset_index(drop=True)
        categories_df = pd.read_sql_query("SELECT * FROM categories", conn)
        books_df = pd.read_sql_query("SELECT * FROM books", conn)
        pandas_join = (
            books_df.merge(categories_df, on="category_id")
            [["title", "rating", "category_name"]]
            .sort_values(["category_name", "title"])
            .reset_index(drop=True)
        )
        if not sql_join.equals(pandas_join):
            raise AssertionError("SQL JOIN and pandas merge outputs do not match")
        output.extend([
            "## JOIN equivalence evidence",
            "### SQL JOIN output",
            sql_join.to_string(index=False),
            "",
            "### pandas.merge output",
            pandas_join.to_string(index=False),
            "",
            "SQL JOIN and pandas.merge are equivalent: True",
        ])
    (OUT_DIR / "query_outputs.txt").write_text("\n".join(output), encoding="utf-8")


def main() -> None:
    print("1. Scraping books...")
    raw = scrape_books()

    print("2. Cleaning and converting prices...")
    clean = clean_books(raw)
    clean.to_csv(OUT_DIR / "books.csv", index=False)

    print("3. Creating the SQLite database...")
    db_path = OUT_DIR / "books.db"
    create_database(clean, db_path)

    print("4. Executing and validating queries...")
    run_queries(db_path, clean)
    print(f"Completed: {len(clean)} books across {clean['category'].nunique()} categories")


if __name__ == "__main__":
    main()
