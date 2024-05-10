import argparse
from tardis_dev import datasets

def main(from_date, to_date):
    datasets.download(
        exchange="coinbase",
        data_types=["quotes"],
        from_date=from_date,
        to_date=to_date,
        symbols=["BTC-USDT", "ETH-USDT"],
        api_key="TD.DV-W-SYHUhTufUsm.EN4btEiFDy9wZ9Y.2osoo9EgGWo6zqr.6-uNQiQoOf6FbaV.7cnPAozw0wnmqwY.O40t"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from Tardis datasets.")
    parser.add_argument("--from_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--to_date", type=str, help="End date in YYYY-MM-DD format")
    args = parser.parse_args()

    # Use command-line arguments if provided, else use default values
    from_date = args.from_date
    to_date = args.to_date

    main(from_date, to_date)
