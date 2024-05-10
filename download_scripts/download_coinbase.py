import argparse
from tardis_dev import datasets
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def main(from_date, to_date):
    # Retrieve the API key from environment variables
    api_key = os.getenv("TARDIS_API_KEY")

    # Ensure the API key is available
    if not api_key:
        raise Exception("API key not found. Please check your .env file.")

    datasets.download(
        exchange="coinbase",
        data_types=["quotes"],
        from_date=from_date,
        to_date=to_date,
        symbols=["BTC-USDT", "ETH-USDT"],
        api_key=api_key
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from Tardis datasets.")
    parser.add_argument("--from_date", type=str, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--to_date", type=str, help="End date in YYYY-MM-DD format")
    args = parser.parse_args()

    main(args.from_date, args.to_date)
