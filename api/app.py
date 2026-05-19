from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import HousePricePredictor

app = FastAPI(title="Home Value Prediction API")
predictor = HousePricePredictor()

class HouseDetails(BaseModel):
    sqft: int
    bedrooms: int
    bathrooms: int
    floors: int
    zipcode: str
    year_built: int
    condition: int

@app.on_event("startup")
def load_model():
    predictor.load_model()

@app.post("/predict")
async def predict_price(details: HouseDetails):
    try:
        input_data = details.dict()
        prediction = predictor.predict(input_data)
        return {"predicted_price": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
