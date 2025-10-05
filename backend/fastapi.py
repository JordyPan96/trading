from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from typing import Optional

app = FastAPI(title="Pepperstone API", version="1.0.0")


class PepperstoneAuth(BaseModel):
    client_id: str
    client_secret: str
    account_id: str


class TradeRequest(BaseModel):
    symbol: str
    volume: float
    order_type: str  # "BUY", "SELL"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


# Store authentication (in production, use secure storage)
pepperstone_auth = None


@app.post("/auth/pepperstone")
async def authenticate_pepperstone(auth: PepperstoneAuth):
    """Authenticate with Pepperstone cTrader"""
    global pepperstone_auth

    try:
        # This is a simplified version - you'll need Pepperstone's actual API endpoints
        auth_url = "https://api.ctrader.com/connect/token"

        data = {
            'grant_type': 'client_credentials',
            'client_id': auth.client_id,
            'client_secret': auth.client_secret,
            'scope': 'trade'
        }

        response = requests.post(auth_url, data=data)

        if response.status_code == 200:
            pepperstone_auth = {
                'access_token': response.json()['access_token'],
                'account_id': auth.account_id
            }
            return {"status": "success", "message": "Authenticated with Pepperstone"}
        else:
            raise HTTPException(status_code=400, detail="Authentication failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auth error: {str(e)}")


@app.get("/pepperstone/accounts")
async def get_pepperstone_accounts():
    """Get Pepperstone account information"""
    if not pepperstone_auth:
        raise HTTPException(status_code=400, detail="Not authenticated")

    try:
        headers = {'Authorization': f'Bearer {pepperstone_auth["access_token"]}'}
        response = requests.get(
            "https://api.ctrader.com/accounts",
            headers=headers
        )

        if response.status_code == 200:
            return {"status": "success", "accounts": response.json()}
        else:
            raise HTTPException(status_code=400, detail="Failed to fetch accounts")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/pepperstone/trade")
async def place_pepperstone_trade(trade: TradeRequest):
    """Place a trade with Pepperstone"""
    if not pepperstone_auth:
        raise HTTPException(status_code=400, detail="Not authenticated")

    try:
        headers = {
            'Authorization': f'Bearer {pepperstone_auth["access_token"]}',
            'Content-Type': 'application/json'
        }

        trade_data = {
            "symbol": trade.symbol,
            "volume": trade.volume,
            "orderType": trade.order_type,
            "accountId": pepperstone_auth["account_id"]
        }

        if trade.stop_loss:
            trade_data["stopLoss"] = trade.stop_loss
        if trade.take_profit:
            trade_data["takeProfit"] = trade.take_profit

        response = requests.post(
            "https://api.ctrader.com/orders",
            json=trade_data,
            headers=headers
        )

        if response.status_code == 200:
            return {"status": "success", "order": response.json()}
        else:
            raise HTTPException(status_code=400, detail=f"Trade failed: {response.text}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trade error: {str(e)}")


@app.get("/pepperstone/symbols")
async def get_pepperstone_symbols():
    """Get available symbols from Pepperstone"""
    # Pepperstone typically offers these major pairs
    symbols = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD",
        "AUDUSD", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY",
        "XAUUSD", "XAGUSD", "AUDJPY", "CADJPY", "CHFJPY"
    ]
    return {"status": "success", "symbols": symbols}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pepperstone_connected": pepperstone_auth is not None,
        "broker": "Pepperstone",
        "api_type": "cTrader WebAPI"
    }
