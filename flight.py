"""
Flight Arrival Delay Prediction - ML Design Patterns Implementation
Complete code for Visual Studio Code (No TensorFlow required!)
"""

import numpy as np
import pandas as pd
import hashlib
from typing import List, Dict
import json
from datetime import datetime

print("=" * 80)
print("FLIGHT ARRIVAL DELAY PREDICTION - ML DESIGN PATTERNS")
print("=" * 80)

# ============================================================================
# 1A. HASHED FEATURE IMPLEMENTATION
# ============================================================================

print("\n" + "=" * 80)
print("1A. HASHED FEATURE PATTERN")
print("=" * 80)

def hash_airport_code(iata_code: str, num_buckets: int = 100) -> int:
    """
    Maps an IATA airport code to a bucket index using hashing.
    
    Args:
        iata_code: String IATA code (e.g., 'JFK', 'LAX')
        num_buckets: Number of hash buckets (default 100)
    
    Returns:
        Bucket index (0 to num_buckets-1)
    """
    # Using Python's built-in hash function
    hash_value = hash(iata_code)
    bucket_index = abs(hash_value) % num_buckets
    return bucket_index


def hash_airport_code_stable(iata_code: str, num_buckets: int = 100) -> int:
    """
    Alternative: Cross-platform stable hashing using MD5
    (Python's hash() can vary between sessions)
    """
    hash_object = hashlib.md5(iata_code.encode())
    hash_int = int(hash_object.hexdigest(), 16)
    return hash_int % num_buckets


# Demonstration
print("\n1. Basic Hashing Example:")
print("-" * 40)
airports = ['JFK', 'LAX', 'ORD', 'ATL', 'DFW', 'SFO', 'PHX', 'DEN', 'BOS', 'LGA']

print(f"{'Airport':<10} {'Bucket (hash)':<15} {'Bucket (MD5)':<15}")
print("-" * 40)
for airport in airports:
    bucket1 = hash_airport_code(airport, num_buckets=100)
    bucket2 = hash_airport_code_stable(airport, num_buckets=100)
    print(f"{airport:<10} {bucket1:<15} {bucket2:<15}")

# Demonstrate collision
print("\n2. Collision Example (with 10 buckets):")
print("-" * 40)
collision_buckets = {}
for airport in airports:
    bucket = hash_airport_code_stable(airport, num_buckets=10)
    if bucket not in collision_buckets:
        collision_buckets[bucket] = []
    collision_buckets[bucket].append(airport)

for bucket, airports_in_bucket in sorted(collision_buckets.items()):
    print(f"Bucket {bucket}: {', '.join(airports_in_bucket)}")
    if len(airports_in_bucket) > 1:
        print(f"  ⚠️  COLLISION: {len(airports_in_bucket)} airports in same bucket")

print("\n3. Analysis:")
print("-" * 40)
print("✅ Advantages:")
print("  - Handles 347 airports with just 100 buckets (3.5x compression)")
print("  - No vocabulary needed - any new airport auto-maps to a bucket")
print("  - Solves cold-start: unseen airports work immediately")
print("\n❌ Trade-off:")
print("  - Hash collisions lose information")
print("  - Model cannot distinguish between collided airports")
print("  - Unpredictable accuracy loss")

# Create a sample dataset with hashed features
print("\n4. Sample DataFrame with Hashed Features:")
print("-" * 40)

# Sample flight data
flight_data = {
    'departure_airport': ['JFK', 'LAX', 'ORD', 'ATL', 'DFW', 'JFK', 'LAX', 'SFO'],
    'scheduled_time': ['14:30', '09:15', '16:45', '11:20', '13:00', '15:30', '10:15', '12:45'],
    'actual_delay_min': [10, 5, 25, 0, 30, 15, -5, 20]
}

df = pd.DataFrame(flight_data)

# Add hashed feature column
df['airport_bucket'] = df['departure_airport'].apply(
    lambda x: hash_airport_code_stable(x, num_buckets=100)
)

print(df.to_string(index=False))


# ============================================================================
# 1B. EMBEDDING IMPLEMENTATION
# ============================================================================

print("\n\n" + "=" * 80)
print("1B. EMBEDDING PATTERN")
print("=" * 80)

# Sample airport vocabulary (in reality, all 347 airports)
airport_vocab = ['JFK', 'LAX', 'ORD', 'ATL', 'DFW', 'SFO', 'PHX', 'DEN', 
                 'BOS', 'LGA', 'EWR', 'MIA', 'SEA', 'MSP', 'DTW']

print(f"\n1. Vocabulary: {len(airport_vocab)} airports (sample)")
print(f"   {', '.join(airport_vocab)}")


class AirportEmbedding:
    """
    Simple embedding implementation without TensorFlow
    Simulates how embeddings work in neural networks
    """
    
    def __init__(self, vocabulary: List[str], embedding_dim: int = 8):
        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.vocab_size = len(vocabulary)
        
        # Create airport to ID mapping
        self.airport_to_id = {airport: idx for idx, airport in enumerate(vocabulary)}
        self.airport_to_id['<UNK>'] = len(vocabulary)  # Unknown/OOV token
        
        # Initialize random embeddings (in real training, these would be learned)
        np.random.seed(42)
        self.embeddings = np.random.randn(self.vocab_size + 1, embedding_dim) * 0.1
        
    def get_id(self, airport: str) -> int:
        """Convert airport code to integer ID"""
        return self.airport_to_id.get(airport, self.airport_to_id['<UNK>'])
    
    def get_embedding(self, airport: str) -> np.ndarray:
        """Get embedding vector for an airport"""
        airport_id = self.get_id(airport)
        return self.embeddings[airport_id]
    
    def __repr__(self):
        return f"AirportEmbedding(vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim})"


# Create embedding layer
print("\n2. Creating Embedding Layer:")
print("-" * 40)
embedding_dim = 8
embedder = AirportEmbedding(airport_vocab, embedding_dim)
print(f"✅ Created: {embedder}")

# Demonstrate integer ID mapping
print("\n3. Airport → Integer ID Mapping:")
print("-" * 40)
print(f"{'Airport':<10} {'Integer ID':<15}")
print("-" * 40)
for airport in ['JFK', 'LAX', 'ORD', 'XYZ']:  # XYZ is unknown
    airport_id = embedder.get_id(airport)
    status = "(OOV)" if airport == 'XYZ' else ""
    print(f"{airport:<10} {airport_id:<15} {status}")

# Show embedding vectors
print("\n4. Example Embedding Vectors:")
print("-" * 40)
sample_airports = ['JFK', 'LAX', 'ORD']
for airport in sample_airports:
    embedding_vector = embedder.get_embedding(airport)
    vector_str = np.array2string(embedding_vector, precision=2, suppress_small=True)
    print(f"{airport}: {vector_str}")

# Simulate learned embeddings (after training)
print("\n5. Simulated Learned Embeddings:")
print("-" * 40)
print("After training, similar airports would have similar vectors:\n")

# Manually create similar embeddings for NYC airports
nyc_base = np.array([0.42, -0.15, 0.88, -0.23, 0.34, -0.67, 0.12, 0.45])
print("NYC Airports (geographically close):")
print(f"  JFK: {np.array2string(nyc_base + np.random.randn(8) * 0.05, precision=2)}")
print(f"  LGA: {np.array2string(nyc_base + np.random.randn(8) * 0.05, precision=2)}")
print(f"  EWR: {np.array2string(nyc_base + np.random.randn(8) * 0.05, precision=2)}")

west_coast_base = np.array([-0.62, 0.73, -0.18, 0.44, -0.56, 0.21, -0.33, 0.67])
print("\nWest Coast Airports:")
print(f"  LAX: {np.array2string(west_coast_base + np.random.randn(8) * 0.05, precision=2)}")
print(f"  SFO: {np.array2string(west_coast_base + np.random.randn(8) * 0.05, precision=2)}")

print("\n6. Embedding in DataFrame:")
print("-" * 40)
# Add embeddings to our flight data
df_embed = df.copy()
df_embed['embedding'] = df_embed['departure_airport'].apply(
    lambda x: embedder.get_embedding(x) if x in airport_vocab else embedder.get_embedding('<UNK>')
)
print(df_embed[['departure_airport', 'embedding']].head())

print("\n7. Analysis:")
print("-" * 40)
print("✅ Advantages over One-Hot Encoding:")
print(f"  - Dense: {embedding_dim} dimensions vs {len(airport_vocab)} sparse dimensions")
print("  - Memory: ~95% reduction in space")
print("  - Learns relationships: similar airports → similar vectors")
print("  - Captures semantic meaning (geography, hub size, etc.)")
print("\n⚖️  Trade-off: Embedding Dimension")
print("  - Too small (2-4): Cannot capture distinctions → underfitting")
print("  - Too large (100+): Overfitting risk, more parameters to train")
print(f"  - Rule of thumb: ~⁴√(vocab_size) = ⁴√{len(airport_vocab)} ≈ {int(len(airport_vocab)**0.25)}-{int(len(airport_vocab)**0.25)*2}")


# ============================================================================
# 2. REFRAMING: REGRESSION → CLASSIFICATION
# ============================================================================

print("\n\n" + "=" * 80)
print("2. REFRAMING PATTERN: REGRESSION → CLASSIFICATION")
print("=" * 80)

def bucketize_delay(delay_minutes: float) -> int:
    """
    Convert continuous delay into categorical buckets
    
    Args:
        delay_minutes: Actual delay in minutes (can be negative for early)
    
    Returns:
        Bucket label (0, 1, or 2)
    """
    if delay_minutes < 0:
        return 0  # "Early/On-Time"
    elif delay_minutes <= 15:
        return 1  # "Minor Delay"
    else:
        return 2  # "Major Delay"


def get_bucket_name(bucket: int) -> str:
    """Get human-readable bucket name"""
    names = {0: "Early/On-Time", 1: "Minor Delay", 2: "Major Delay"}
    return names.get(bucket, "Unknown")


# Demonstration
print("\n1. Bucketing Strategy:")
print("-" * 40)
print("Bucket 0: Early/On-Time    → delay < 0 minutes")
print("Bucket 1: Minor Delay      → 0 ≤ delay ≤ 15 minutes")
print("Bucket 2: Major Delay      → delay > 15 minutes")

print("\n2. Example Transformations:")
print("-" * 40)
sample_delays = [-5, 0, 5, 10, 15, 20, 30, 45, 60]
print(f"{'Delay (min)':<15} {'Bucket':<10} {'Label':<20}")
print("-" * 40)
for delay in sample_delays:
    bucket = bucketize_delay(delay)
    label = get_bucket_name(bucket)
    print(f"{delay:<15} {bucket:<10} {label:<20}")

# Apply to our dataset
print("\n3. Reframing Applied to Dataset:")
print("-" * 40)
df_reframed = df.copy()
df_reframed['delay_bucket'] = df_reframed['actual_delay_min'].apply(bucketize_delay)
df_reframed['delay_label'] = df_reframed['delay_bucket'].apply(get_bucket_name)
print(df_reframed[['departure_airport', 'actual_delay_min', 'delay_bucket', 'delay_label']].to_string(index=False))

# Simulate probabilistic predictions
print("\n4. Regression vs Classification Output:")
print("-" * 40)
print("\n❌ Regression Output (single point estimate):")
print("   Flight AA123:")
print("   Prediction: 12.5 minutes delay")
print("   Problem: Doesn't capture uncertainty!")
print("   Reality: Same features might give 5 min OR 20 min delays")

print("\n✅ Classification Output (probability distribution):")
print("   Flight AA123:")
print("   P(Early/On-Time) = 0.25  (25%)")
print("   P(Minor Delay)   = 0.55  (55%)")
print("   P(Major Delay)   = 0.20  (20%)")
print("\n   → More actionable: 'There's a 20% risk of major delay'")
print("   → User can decide: 'I'll leave earlier if P(Major) > 30%'")

# Simulate model predictions
print("\n5. Example Model Predictions:")
print("-" * 40)
np.random.seed(42)
df_predictions = df_reframed[['departure_airport', 'actual_delay_min', 'delay_label']].copy()

# Simulate probability predictions
df_predictions['P(Early/On-Time)'] = np.random.dirichlet([2, 1, 1], len(df_predictions))[:, 0]
df_predictions['P(Minor Delay)'] = np.random.dirichlet([1, 3, 1], len(df_predictions))[:, 1]
df_predictions['P(Major Delay)'] = 1 - df_predictions['P(Early/On-Time)'] - df_predictions['P(Minor Delay)']

print(df_predictions.to_string(index=False))

print("\n6. Benefits of Reframing:")
print("-" * 40)
print("✅ Captures uncertainty in predictions")
print("✅ Enables risk-based decision making")
print("✅ Aligns with probabilistic nature of delays")
print("✅ Users can set thresholds (e.g., alert if P(Major) > 30%)")
print("✅ Better for operations: 'Prepare for delays' vs '12.5 min delay'")


# ============================================================================
# 3. STATELESS SERVING & MONITORING
# ============================================================================

print("\n\n" + "=" * 80)
print("3. STATELESS SERVING & CONTINUED EVALUATION")
print("=" * 80)

print("\n3.1 STATELESS SERVING FUNCTION")
print("-" * 40)
print("\nStep 1: Export to Language-Agnostic Format")
print("  → Save model as TensorFlow SavedModel / ONNX / PMML")
print("  → Example code:")
print("     import tensorflow as tf")
print("     model.save('flight_delay_model/')")
print("  → Creates portable format usable by any framework")

print("\nStep 2: Deploy as Stateless REST API")
print("  → Wrap in web server (TensorFlow Serving, FastAPI, Flask)")
print("  → Each request is independent, no state between calls")
print("  → Enable horizontal scaling with load balancer")
print("  → Example deployment:")
print("     docker run -p 8501:8501 tensorflow/serving \\")
print("       --model_base_path=/models/flight_delay")

print("\nStep 3: Handle Requests Independently")
print("  → Receive features → Load model → Predict → Return → Discard")
print("  → No memory of previous predictions")
print("  → Each instance can handle any request")

# Example API implementation
print("\n📡 Example API Request:")
api_request = {
    "departure_airport": "JFK",
    "arrival_airport": "LAX",
    "scheduled_time": "2025-11-29T14:30:00",
    "weather_condition": "clear",
    "day_of_week": "Saturday",
    "airline": "AA"
}
print(json.dumps(api_request, indent=2))

print("\n📡 Example API Response:")
api_response = {
    "prediction": {
        "early_ontime_prob": 0.25,
        "minor_delay_prob": 0.55,
        "major_delay_prob": 0.20
    },
    "predicted_class": "Minor Delay",
    "predicted_class_id": 1,
    "confidence": 0.55,
    "timestamp": "2025-11-29T10:15:30Z",
    "model_version": "v2.1.3"
}
print(json.dumps(api_response, indent=2))

print("\n✅ Benefits of Stateless Serving:")
print("  - Auto-scaling: Add/remove instances based on traffic")
print("  - Fault tolerance: Failed instance doesn't affect others")
print("  - Easy rollback: Switch between model versions instantly")
print("  - Load balancing: Distribute requests evenly")


print("\n\n3.2 CONTINUED MODEL EVALUATION")
print("-" * 40)

print("\nPurpose:")
print("  ✅ Detect concept drift (feature-label relationships change)")
print("  ✅ Identify data drift (input distributions shift)")
print("  ✅ Trigger retraining when performance degrades")
print("  ✅ Maintain service quality over time")

print("\n📊 Three Types of Data to Collect:")
print("-" * 40)

print("\n1️⃣  INPUT FEATURES (Prediction Requests)")
print("  What: All features sent to model")
print("    - departure_airport, arrival_airport")
print("    - scheduled_time, day_of_week")
print("    - weather_condition, airline")
print("  Why: Monitor for DATA DRIFT")
print("  Examples of drift:")
print("    • New airport suddenly appears frequently")
print("    • Weather patterns shift due to climate change")
print("    • Traffic shifts to different times of day")

print("\n2️⃣  MODEL PREDICTIONS (Outputs)")
print("  What: All predictions and probability distributions")
print("    - Predicted class (Early/Minor/Major)")
print("    - Probability for each class")
print("  Why: Track prediction distributions over time")
print("  Red flags:")
print("    • Model suddenly predicts 'Major Delay' 80% of time")
print("    • Confidence scores drop significantly")
print("    • Prediction distribution changes drastically")

print("\n3️⃣  GROUND TRUTH LABELS (Actual Outcomes)")
print("  What: Real arrival delays after flights land")
print("    - Actual delay in minutes")
print("    - Actual delay bucket")
print("  Why: Calculate REAL performance metrics")
print("  Critical for detecting CONCEPT DRIFT")
print("  Examples:")
print("    • Airport construction changes delay patterns")
print("    • New airline policies affect operations")
print("    • Weather patterns affect delays differently")

print("\n🔄 Monitoring Workflow:")
print("-" * 40)
monitoring_steps = [
    "1. Collect: Input features + Predictions + Ground truth",
    "2. Store: Save all three in time-series database",
    "3. Compare: Predictions vs Actual outcomes (daily/weekly)",
    "4. Calculate: Accuracy, Precision, Recall, F1, MAE",
    "5. Visualize: Dashboards showing performance trends",
    "6. Alert: If performance < threshold (e.g., accuracy < 0.75)",
    "7. Investigate: Analyze which features or airports degraded",
    "8. Trigger: Automated model retraining pipeline"
]
for step in monitoring_steps:
    print(f"   {step}")

# Example monitoring data
print("\n📈 Example Monitoring Metrics (Week of Nov 25-29, 2025):")
print("-" * 40)
monitoring_data = {
    "date_range": "2025-11-25 to 2025-11-29",
    "predictions_made": 12470,
    "ground_truth_collected": 11890,
    "overall_accuracy": 0.78,
    "precision_by_class": {
        "early_ontime": 0.82,
        "minor_delay": 0.75,
        "major_delay": 0.71
    },
    "recall_by_class": {
        "early_ontime": 0.79,
        "minor_delay": 0.78,
        "major_delay": 0.74
    },
    "f1_score": 0.76,
    "mean_absolute_error_minutes": 8.3,
    "data_drift_detected": False,
    "concept_drift_detected": False,
    "retraining_needed": False,
    "alerts": []
}
print(json.dumps(monitoring_data, indent=2))

print("\n⚠️  Example Alert Scenario:")
print("-" * 40)
alert_scenario = {
    "date": "2025-12-05",
    "alert_type": "PERFORMANCE_DEGRADATION",
    "severity": "HIGH",
    "details": {
        "current_accuracy": 0.68,
        "threshold": 0.75,
        "decline_percentage": -9.0,
        "affected_airports": ["JFK", "EWR", "LGA"],
        "suspected_cause": "NYC airport construction project started",
        "recommendation": "Retrain model with recent data including construction feature"
    }
}
print(json.dumps(alert_scenario, indent=2))

print("\n📊 Alert Conditions:")
print("  → Accuracy drops below 0.75 for 3 consecutive days")
print("  → Data distribution shifts (KL divergence > 0.5)")
print("  → New airports represent > 10% of traffic")
print("  → MAE increases by > 15% from baseline")
print("  → Precision for 'Major Delay' < 0.65 (safety critical)")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY - KEY TAKEAWAYS")
print("=" * 80)

summary = """
1️⃣  HASHED FEATURES:
   ✅ Compress 347 airports → 100 buckets
   ✅ Handle new airports without retraining (cold-start solution)
   ✅ No vocabulary maintenance needed
   ❌ Hash collisions lose information
   ❌ Model cannot distinguish between collided airports

2️⃣  EMBEDDINGS:
   ✅ Dense representation (8-16 dims vs 347 sparse dimensions)
   ✅ Learn airport relationships automatically (similar → similar vectors)
   ✅ Capture semantic meaning (geography, hub size, traffic patterns)
   ✅ ~95% memory reduction vs one-hot encoding
   ⚖️  Dimension trade-off:
       - Too small (2-4) → underfitting, can't capture distinctions
       - Too large (100+) → overfitting, excess parameters
       - Rule of thumb: ⁴√(vocab_size)

3️⃣  REFRAMING (Regression → Classification):
   ✅ Output probability distributions, not point estimates
   ✅ Capture uncertainty: P(Major Delay) = 20%
   ✅ Enable risk-based decisions (alert if P(Major) > 30%)
   ✅ Aligns with probabilistic nature of flight delays
   ✅ More actionable for users and operations teams

4️⃣  STATELESS SERVING:
   Step 1: Export → Language-agnostic format (SavedModel/ONNX)
   Step 2: Deploy → REST API (TensorFlow Serving, FastAPI)
   Step 3: Handle → Independent requests, no state
   ✅ Scalable, fault-tolerant, easy rollback
   ✅ Horizontal scaling with load balancer

5️⃣  CONTINUED EVALUATION:
   Data to collect:
     1. Input features (detect data drift)
     2. Model predictions (monitor distribution changes)
     3. Ground truth labels (measure real performance)
   ✅ Detect concept drift & data drift
   ✅ Calculate actual metrics (accuracy, precision, recall)
   ✅ Trigger automated retraining when needed
   ✅ Maintain production quality over time
"""

print(summary)

print("\n" + "=" * 80)
print("✅ Implementation Complete! No TensorFlow required.")
print("   All patterns demonstrated with NumPy and Pandas only.")
print("=" * 80)
