"""
BloomWatch Pro Backend - Advanced Global Phenology Platform
A comprehensive backend system for monitoring and predicting global flowering patterns
using satellite data, citizen science observations, and AI-powered analytics.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from enum import Enum
import json
import asyncio
from dataclasses import dataclass
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
from redis import Redis
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import google.generativeai as genai
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Configuration
class Config:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/bloomwatch")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your-google-api-key")
    NASA_EARTHDATA_TOKEN = os.getenv("NASA_EARTHDATA_TOKEN", "your-nasa-token")
    INATURALIST_TOKEN = os.getenv("INATURALIST_TOKEN", "your-inaturalist-token")
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bloomwatch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = create_engine(Config.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis setup
redis_client = Redis.from_url(Config.REDIS_URL)

# Google Gemini setup
genai.configure(api_key=Config.GOOGLE_API_KEY)

# Enums
class Phenophase(str, Enum):
    FIRST_BLOOM = "first_bloom"
    PEAK_BLOOM = "peak_bloom"
    END_BLOOM = "end_bloom"
    LEAF_OUT = "leaf_out"
    SEED_SET = "seed_set"

class DataSource(str, Enum):
    NASA_MODIS = "nasa_modis"
    NASA_VIIRS = "nasa_viirs"
    NASA_LANDSAT = "nasa_landsat"
    USA_NPN = "usa_npn"
    GLOBE_OBSERVER = "globe_observer"
    GBIF = "gbif"
    INATURALIST = "inaturalist"
    PHENOCAM = "phenocam"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Database Models
class Species(Base):
    __tablename__ = "species"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scientific_name = Column(String(255), nullable=False, unique=True)
    common_name = Column(String(255))
    family = Column(String(100))
    genus = Column(String(100))
    native_range = Column(Text)
    conservation_status = Column(String(50))
    thermal_threshold = Column(Float)  # Temperature threshold for flowering
    created_at = Column(DateTime, default=datetime.utcnow)

class Observation(Base):
    __tablename__ = "observations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    species_id = Column(UUID(as_uuid=True))
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    phenophase = Column(String(50), nullable=False)
    observation_date = Column(DateTime, nullable=False)
    source = Column(String(50), nullable=False)
    observer_id = Column(String(100))
    confidence_score = Column(Float)
    temperature = Column(Float)
    precipitation = Column(Float)
    elevation = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    species_id = Column(UUID(as_uuid=True))
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    predicted_date = Column(DateTime, nullable=False)
    confidence = Column(Float)
    prediction_type = Column(String(50))  # short_term, seasonal, long_term
    model_version = Column(String(50))
    input_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class ClimateData(Base):
    __tablename__ = "climate_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    date = Column(DateTime, nullable=False)
    temperature = Column(Float)
    precipitation = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    solar_radiation = Column(Float)
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class ExtinctionRisk(Base):
    __tablename__ = "extinction_risks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    species_id = Column(UUID(as_uuid=True))
    risk_level = Column(String(20), nullable=False)
    risk_type = Column(String(50))  # thermal_stress, hydraulic_failure, photosynthetic_disruption
    probability = Column(Float)
    timeframe = Column(String(50))  # near_term, medium_term, long_term
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic Models for API
class SpeciesCreate(BaseModel):
    scientific_name: str
    common_name: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    native_range: Optional[str] = None
    conservation_status: Optional[str] = None
    thermal_threshold: Optional[float] = None

class ObservationCreate(BaseModel):
    species_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    phenophase: Phenophase
    observation_date: datetime
    source: DataSource
    observer_id: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    temperature: Optional[float] = None
    precipitation: Optional[float] = None
    elevation: Optional[float] = None

class PredictionRequest(BaseModel):
    species_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    prediction_type: str = Field(..., regex="^(short_term|seasonal|long_term)$")
    start_date: Optional[datetime] = None

class ExtinctionRiskRequest(BaseModel):
    species_id: str
    risk_type: str = Field(..., regex="^(thermal_stress|hydraulic_failure|photosynthetic_disruption)$")
    timeframe: str = Field(..., regex="^(near_term|medium_term|long_term)$")

class AnalyticsResponse(BaseModel):
    total_observations: int
    species_count: int
    regions_covered: int
    prediction_accuracy: float
    climate_correlation: Dict[str, float]
    risk_summary: Dict[str, Any]

# Machine Learning Models
class PhenologyPredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lstm': None
        }
        self.scalers = {}
        self.feature_columns = [
            'temperature', 'precipitation', 'humidity', 'wind_speed', 
            'solar_radiation', 'elevation', 'day_of_year', 'latitude', 'longitude'
        ]
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for machine learning models"""
        data['day_of_year'] = data['observation_date'].dt.dayofyear
        features = data[self.feature_columns].copy()
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        return features.values
    
    def train_models(self, training_data: pd.DataFrame):
        """Train all ML models with historical data"""
        logger.info("Training machine learning models...")
        
        X = self.prepare_features(training_data)
        y = training_data['days_to_bloom'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Train Random Forest
        self.models['random_forest'].fit(X_train_scaled, y_train)
        rf_pred = self.models['random_forest'].predict(X_test_scaled)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        logger.info(f"Random Forest MAE: {rf_mae:.2f}")
        
        # Train Gradient Boosting
        self.models['gradient_boosting'].fit(X_train_scaled, y_train)
        gb_pred = self.models['gradient_boosting'].predict(X_test_scaled)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        logger.info(f"Gradient Boosting MAE: {gb_mae:.2f}")
        
        # Train LSTM
        self._train_lstm(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Save models
        self.save_models()
    
    def _train_lstm(self, X_train, y_train, X_test, y_test):
        """Train LSTM model for time series prediction"""
        # Reshape data for LSTM
        X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train_reshaped, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.models['lstm'] = model
        lstm_pred = model.predict(X_test_reshaped).flatten()
        lstm_mae = mean_absolute_error(y_test, lstm_pred)
        logger.info(f"LSTM MAE: {lstm_mae:.2f}")
    
    def predict_flowering(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict flowering time using ensemble of models"""
        feature_df = pd.DataFrame([features])
        X = self.prepare_features(feature_df)
        X_scaled = self.scalers['standard'].transform(X)
        
        predictions = {}
        
        # Random Forest prediction
        rf_pred = self.models['random_forest'].predict(X_scaled)[0]
        predictions['random_forest'] = rf_pred
        
        # Gradient Boosting prediction
        gb_pred = self.models['gradient_boosting'].predict(X_scaled)[0]
        predictions['gradient_boosting'] = gb_pred
        
        # LSTM prediction
        if self.models['lstm']:
            X_reshaped = X_scaled.reshape((1, 1, X_scaled.shape[1]))
            lstm_pred = self.models['lstm'].predict(X_reshaped)[0][0]
            predictions['lstm'] = lstm_pred
        
        # Ensemble prediction (weighted average)
        ensemble_pred = np.mean(list(predictions.values()))
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def save_models(self):
        """Save trained models to disk"""
        os.makedirs('models', exist_ok=True)
        
        # Save scikit-learn models
        joblib.dump(self.models['random_forest'], 'models/random_forest.pkl')
        joblib.dump(self.models['gradient_boosting'], 'models/gradient_boosting.pkl')
        joblib.dump(self.scalers['standard'], 'models/scaler.pkl')
        
        # Save LSTM model
        if self.models['lstm']:
            self.models['lstm'].save('models/lstm_model.h5')
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            self.models['random_forest'] = joblib.load('models/random_forest.pkl')
            self.models['gradient_boosting'] = joblib.load('models/gradient_boosting.pkl')
            self.scalers['standard'] = joblib.load('models/scaler.pkl')
            
            if os.path.exists('models/lstm_model.h5'):
                self.models['lstm'] = load_model('models/lstm_model.h5')
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")

# Data Integration Services
class DataIntegrator:
    def __init__(self):
        self.nasa_base_url = "https://cmr.earthdata.nasa.gov/search"
        self.usa_npn_url = "https://www.usanpn.org/npn/observations"
        self.gbif_url = "https://api.gbif.org/v1"
        self.inaturalist_url = "https://api.inaturalist.org/v1"
        
    def fetch_nasa_satellite_data(self, bbox: List[float], start_date: str, end_date: str) -> List[Dict]:
        """Fetch satellite data from NASA Earthdata"""
        try:
            params = {
                'short_name': 'MOD13Q1',  # MODIS vegetation indices
                'bounding_box': ','.join(map(str, bbox)),
                'temporal': f'{start_date},{end_date}',
                'page_size': 100
            }
            
            response = requests.get(f"{self.nasa_base_url}/granules.json", params=params)
            response.raise_for_status()
            
            data = response.json()
            return self._process_nasa_data(data)
        except Exception as e:
            logger.error(f"Error fetching NASA data: {e}")
            return []
    
    def fetch_usa_npn_data(self, species_id: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch phenology data from USA-NPN"""
        try:
            params = {
                'species_id': species_id,
                'start_date': start_date,
                'end_date': end_date,
                'format': 'json'
            }
            
            response = requests.get(self.usa_npn_url, params=params)
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching USA-NPN data: {e}")
            return []
    
    def fetch_gbif_data(self, species_name: str, bbox: List[float]) -> List[Dict]:
        """Fetch species occurrence data from GBIF"""
        try:
            params = {
                'scientificName': species_name,
                'geometry': f'POLYGON(({bbox[0]} {bbox[1]}, {bbox[2]} {bbox[1]}, {bbox[2]} {bbox[3]}, {bbox[0]} {bbox[3]}, {bbox[0]} {bbox[1]}))',
                'limit': 1000
            }
            
            response = requests.get(f"{self.gbif_url}/occurrence/search", params=params)
            response.raise_for_status()
            
            return response.json().get('results', [])
        except Exception as e:
            logger.error(f"Error fetching GBIF data: {e}")
            return []
    
    def _process_nasa_data(self, raw_data: Dict) -> List[Dict]:
        """Process raw NASA satellite data"""
        processed_data = []
        
        for item in raw_data.get('feed', {}).get('entry', []):
            try:
                processed_item = {
                    'id': item.get('id'),
                    'title': item.get('title'),
                    'time_start': item.get('time_start'),
                    'time_end': item.get('time_end'),
                    'coordinates': self._extract_coordinates(item),
                    'vegetation_index': self._extract_vegetation_index(item)
                }
                processed_data.append(processed_item)
            except Exception as e:
                logger.warning(f"Error processing NASA data item: {e}")
                continue
        
        return processed_data
    
    def _extract_coordinates(self, item: Dict) -> List[float]:
        """Extract coordinates from NASA data item"""
        # Implementation depends on NASA data format
        return [0.0, 0.0]  # Placeholder
    
    def _extract_vegetation_index(self, item: Dict) -> float:
        """Extract vegetation index from NASA data item"""
        # Implementation depends on NASA data format
        return 0.0  # Placeholder

# AI Analysis Service
class AIAnalysisService:
    def __init__(self):
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
    def analyze_flowering_patterns(self, observations: List[Dict]) -> Dict[str, Any]:
        """Analyze flowering patterns using Google Gemini AI"""
        try:
            # Prepare data for AI analysis
            analysis_prompt = self._prepare_analysis_prompt(observations)
            
            response = self.gemini_model.generate_content(analysis_prompt)
            
            # Parse AI response
            analysis_result = self._parse_ai_response(response.text)
            
            return analysis_result
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {}
    
    def predict_extinction_risk(self, species_data: Dict, climate_projections: Dict) -> Dict[str, Any]:
        """Predict extinction risk using AI analysis"""
        try:
            prompt = f"""
            Analyze extinction risk for species {species_data.get('scientific_name')} based on:
            
            Species Data:
            - Thermal threshold: {species_data.get('thermal_threshold')}°C
            - Native range: {species_data.get('native_range')}
            - Conservation status: {species_data.get('conservation_status')}
            
            Climate Projections:
            - Temperature increase: {climate_projections.get('temp_increase')}°C
            - Precipitation change: {climate_projections.get('precip_change')}%
            - Extreme events frequency: {climate_projections.get('extreme_events')}
            
            Provide risk assessment for:
            1. Thermal stress vulnerability
            2. Hydraulic failure risk
            3. Photosynthetic disruption
            4. Overall extinction probability
            
            Format response as JSON with risk levels (LOW, MEDIUM, HIGH, CRITICAL) and probabilities.
            """
            
            response = self.gemini_model.generate_content(prompt)
            return self._parse_ai_response(response.text)
        except Exception as e:
            logger.error(f"Error in extinction risk prediction: {e}")
            return {}
    
    def _prepare_analysis_prompt(self, observations: List[Dict]) -> str:
        """Prepare prompt for AI analysis"""
        obs_summary = f"Total observations: {len(observations)}"
        
        # Group by phenophase
        phenophase_counts = {}
        for obs in observations:
            phase = obs.get('phenophase', 'unknown')
            phenophase_counts[phase] = phenophase_counts.get(phase, 0) + 1
        
        prompt = f"""
        Analyze flowering patterns based on the following data:
        
        {obs_summary}
        Phenophase distribution: {phenophase_counts}
        
        Provide insights on:
        1. Dominant flowering patterns
        2. Anomalies or unusual patterns
        3. Climate correlation indicators
        4. Predictions for next flowering season
        
        Format response as JSON with analysis sections and confidence scores.
        """
        
        return prompt
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response into structured format"""
        try:
            # Try to parse as JSON first
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If not JSON, create structured response from text
            return {
                'analysis': response_text,
                'confidence': 0.7,
                'timestamp': datetime.utcnow().isoformat()
            }

# Analytics Service
class AnalyticsService:
    def __init__(self, db: Session):
        self.db = db
        
    def generate_global_analytics(self) -> AnalyticsResponse:
        """Generate comprehensive global analytics"""
        try:
            # Basic statistics
            total_observations = self.db.query(Observation).count()
            species_count = self.db.query(Species).count()
            
            # Regions covered (approximate)
            regions_query = self.db.query(Observation.latitude, Observation.longitude).distinct()
            regions_covered = len(regions_query.all())
            
            # Prediction accuracy
            recent_predictions = self.db.query(Prediction).filter(
                Prediction.created_at >= datetime.utcnow() - timedelta(days=30)
            ).all()
            prediction_accuracy = self._calculate_prediction_accuracy(recent_predictions)
            
            # Climate correlation
            climate_correlation = self._calculate_climate_correlation()
            
            # Risk summary
            risk_summary = self._generate_risk_summary()
            
            return AnalyticsResponse(
                total_observations=total_observations,
                species_count=species_count,
                regions_covered=regions_covered,
                prediction_accuracy=prediction_accuracy,
                climate_correlation=climate_correlation,
                risk_summary=risk_summary
            )
        except Exception as e:
            logger.error(f"Error generating analytics: {e}")
            raise HTTPException(status_code=500, detail="Analytics generation failed")
    
    def _calculate_prediction_accuracy(self, predictions: List[Prediction]) -> float:
        """Calculate prediction accuracy"""
        if not predictions:
            return 0.0
        
        # Simple accuracy based on confidence scores
        avg_confidence = np.mean([p.confidence for p in predictions])
        return min(avg_confidence, 1.0)
    
    def _calculate_climate_correlation(self) -> Dict[str, float]:
        """Calculate climate correlation with flowering patterns"""
        # This would involve complex statistical analysis
        # For now, return placeholder values
        return {
            'temperature_correlation': 0.85,
            'precipitation_correlation': 0.62,
            'humidity_correlation': 0.45,
            'solar_radiation_correlation': 0.78
        }
    
    def _generate_risk_summary(self) -> Dict[str, Any]:
        """Generate extinction risk summary"""
        risk_counts = self.db.query(