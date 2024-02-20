import argparse
from tardis_dev import datasets

def main(from_date, to_date):
    datasets.download(
        exchange="coinbase",
        data_types=["quotes"],
        from_date=from_date,
        to_date=to_date,
        symbols=["BTC-USDT", "ETH-USDT"],
        api_key="TD.EGbmz1ZjK48X8Umz.xA3stUptkohr9Bs.kkKNJo2yL78bZVP.YJRJl3xrqDmyvJE.ymLsIiFZJ-tTtLx.s2Xh",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from Tardis datasets.")
    parser.add_argument("--from_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--to_date", type=str, help="End date in YYYY-MM-DD format")
    args = parser.parse_args()

    # Use command-line arguments if provided, else use default values
    from_date = args.from_date if args.from_date else "2023-11-01"
    to_date = args.to_date if args.to_date else "2023-11-02"

    main(from_date, to_date)