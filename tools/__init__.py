"""
Tool layer: data loading, the MarketAPI façade and shared market helpers.

Kept import-free on purpose (no eager submodule imports) so that
`tools.data_loader`, `tools.market_tools` and `tools.market_api` can import each
other without any circular-import risk. Import what you need explicitly:

    from tools.market_api import MarketAPI
"""

__all__ = ["data_loader", "market_tools", "market_api"]
