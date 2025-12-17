"""
Lightweight helper for Facebook Catalog integration via the Graph API.

The client wraps a few common actions (list, create, update, delete) and can be
used from Streamlit or any other script. Provide credentials directly or via
environment variables:
  - FACEBOOK_ACCESS_TOKEN or META_ACCESS_TOKEN
  - FACEBOOK_CATALOG_ID
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


class FacebookCatalogError(RuntimeError):
    """Raised when the Graph API responds with an error payload or bad status."""


@dataclass
class CatalogItem:
    retailer_id: str
    name: str
    description: str
    url: str
    image_url: str
    currency: str
    price: str
    availability: str = "in stock"
    condition: str = "new"
    brand: Optional[str] = None
    additional_image_urls: Optional[List[str]] = None
    inventory: Optional[int] = None
    category: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "retailer_id": self.retailer_id,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "image_url": self.image_url,
            "currency": self.currency,
            "price": self.price,
            "availability": self.availability,
            "condition": self.condition,
        }
        if self.brand:
            payload["brand"] = self.brand
        if self.additional_image_urls:
            payload["additional_image_urls"] = self.additional_image_urls
        if self.inventory is not None:
            payload["inventory"] = self.inventory
        if self.category:
            payload["category"] = self.category
        return payload


class FacebookCatalogClient:
    def __init__(
        self,
        access_token: Optional[str] = None,
        catalog_id: Optional[str] = None,
        api_version: str = "v19.0",
        timeout: int = 10,
        session: Optional[requests.Session] = None,
    ) -> None:
        token = access_token or os.getenv("FACEBOOK_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN")
        if not token:
            raise ValueError("Facebook access token is required. Set FACEBOOK_ACCESS_TOKEN.")

        catalog = catalog_id or os.getenv("FACEBOOK_CATALOG_ID")
        if not catalog:
            raise ValueError("Catalog ID is required. Set FACEBOOK_CATALOG_ID.")

        self.access_token = token
        self.catalog_id = catalog
        self.base_url = f"https://graph.facebook.com/{api_version}"
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        params = kwargs.pop("params", {}) or {}
        params["access_token"] = self.access_token

        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.request(method, url, params=params, timeout=self.timeout, **kwargs)

        try:
            payload = response.json()
        except ValueError as exc:  # non-JSON response
            response.raise_for_status()
            raise FacebookCatalogError("Facebook API returned a non-JSON response") from exc

        if not response.ok or isinstance(payload, dict) and payload.get("error"):
            message = payload.get("error", {}).get("message") if isinstance(payload, dict) else None
            raise FacebookCatalogError(message or f"Facebook API error: {response.status_code}")

        if not isinstance(payload, dict):
            raise FacebookCatalogError("Facebook API returned an unexpected payload shape")

        return payload

    # ---------- Public API ----------
    def list_products(
        self,
        limit: int = 25,
        after: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": limit}
        if after:
            params["after"] = after
        if fields:
            params["fields"] = ",".join(fields)
        return self._request("GET", f"{self.catalog_id}/products", params=params)

    def create_product(self, item: CatalogItem, skip_validation: bool = False) -> Dict[str, Any]:
        payload = item.to_payload()
        if skip_validation:
            payload["skip_validation"] = True
        return self._request("POST", f"{self.catalog_id}/products", json=payload)

    def get_product(self, product_item_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if fields:
            params["fields"] = ",".join(fields)
        return self._request("GET", product_item_id, params=params)

    def update_price_inventory(
        self,
        product_item_id: str,
        price: str,
        currency: str,
        inventory: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"price": price, "currency": currency}
        if inventory is not None:
            payload["inventory"] = inventory
        return self._request("POST", product_item_id, json=payload)

    def delete_product(self, product_item_id: str) -> Dict[str, Any]:
        return self._request("DELETE", product_item_id)

