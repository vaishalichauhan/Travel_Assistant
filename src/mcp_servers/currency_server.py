"""
MCP server exposing one tool: convert_currency.

Uses the free exchange rate API at https://open.er-api.com/ (no key required).

Test standalone first:
    python src/mcp_servers/currency_server.py --test

Run as an actual MCP server (stdio transport):
    python src/mcp_servers/currency_server.py
"""

import sys
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("currency-server")


@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount from one currency to another using current exchange rates.

    Args:
        amount: the amount to convert.
        from_currency: 3-letter currency code, e.g. "INR", "USD".
        to_currency: 3-letter currency code, e.g. "SGD".
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    url = f"https://open.er-api.com/v6/latest/{from_currency}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") != "success":
            return {"error": f"Could not fetch rates for {from_currency}"}
        rate = data["rates"].get(to_currency)
        if rate is None:
            return {"error": f"No rate found for {to_currency}"}
        converted = round(amount * rate, 2)
        return {
            "amount": amount,
            "from": from_currency,
            "to": to_currency,
            "rate": rate,
            "converted_amount": converted,
            "source": "open.er-api.com",
            "last_updated": data.get("time_last_update_utc"),
        }
    except requests.RequestException as e:
        return {"error": f"Currency service unavailable: {e}"}


if __name__ == "__main__":
    if "--test" in sys.argv:
        import json
        print(json.dumps(convert_currency(50000, "INR", "SGD"), indent=2))
    else:
        mcp.run(transport="stdio")
