from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
import json
import pickle
import joblib
from datetime import datetime, timedelta
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-this-to-something-secure'  # Change this to a secure secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store model and data
model = None
scaler = None
potential_comps = None
knn_model = None
model_metrics = None  # Store model performance metrics

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def bath_to_numeric(bath_str):
    """Convert various bathroom notations to numeric values"""
    if pd.isna(bath_str):
        return 0.0

    # Handle string formats
    if isinstance(bath_str, str):
        # Handle colon format like "2:1" (2 full, 1 half)
        if ':' in bath_str:
            try:
                full, half = map(float, bath_str.split(':'))
                return full + (half * 0.5)
            except (ValueError, TypeError):
                pass

        # Try direct conversion if it's just a number as string
        try:
            return float(bath_str)
        except (ValueError, TypeError):
            return 1.0  # Default fallback

    # Handle numeric types
    try:
        return float(bath_str)
    except (ValueError, TypeError):
        return 0.0

def load_models_and_data():
    """Load the trained models and preprocessed data"""
    global model, scaler, potential_comps, knn_model, model_metrics
    
    try:
        # Load the trained model
        if os.path.exists('property_comp_model.pkl'):
            with open('property_comp_model.pkl', 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully")
        else:
            print("Model file not found")
            
        # Load the scaler
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
            print("Scaler loaded successfully")
        else:
            print("Scaler file not found")
            
        # Load model metrics
        if os.path.exists('model_metrics.json'):
            with open('model_metrics.json', 'r') as f:
                model_metrics = json.load(f)
            print("Model metrics loaded successfully")
        else:
            print("Model metrics file not found")
            
        # Load the potential comparables data
        if os.path.exists('potential_comps.csv'):
            potential_comps = pd.read_csv('potential_comps.csv')
            print(f"Potential comps data loaded successfully - {len(potential_comps)} properties")
            
            # Add num_baths_numeric column if not present
            if 'num_baths_numeric' not in potential_comps.columns and 'num_baths' in potential_comps.columns:
                potential_comps['num_baths_numeric'] = potential_comps['num_baths'].apply(bath_to_numeric)
            
            # Initialize KNN model if we have enough data
            comp_features = ['gla', 'year_built', 'num_beds']
            available_features = [f for f in comp_features if f in potential_comps.columns]
            
            if available_features and len(potential_comps) > 5:
                knn_model = NearestNeighbors(
                    n_neighbors=min(20, len(potential_comps)),
                    algorithm='ball_tree',
                    metric='manhattan'
                )
                # Fit on available numeric features only
                numeric_data = potential_comps[available_features].fillna(0)
                knn_model.fit(numeric_data)
                print("KNN model initialized successfully")
        else:
            print("Potential comps file not found")
            
    except Exception as e:
        print(f"Error loading models: {e}")

def process_appraisals_data(json_file_path):
    """Process the appraisals JSON data similar to your original script"""
    try:
        with open(json_file_path) as f:
            data = json.load(f)
        
        appraisals = data['appraisals']
        records = []
        
        def safe_get(d, key, default=None):
            val = d.get(key, default)
            if val is None:
                return default
            return str(val).strip() if isinstance(val, str) else val
        
        for appraisal in appraisals:
            subject = appraisal['subject']
            comps = appraisal['comps']
            properties = appraisal['properties']
            
            # Process subject property
            subject_record = {
                'type': 'subject',
                'orderID': appraisal['orderID'],
                'address': safe_get(subject, 'address'),
                'city_province_zip': safe_get(subject, 'subject_city_province_zip'),
                'effective_date': safe_get(subject, 'effective_date'),
                'municipality_district': safe_get(subject, 'municipality_district'),
                'structure_type': safe_get(subject, 'structure_type'),
                'year_built': safe_get(subject, 'year_built'),
                'gla': safe_get(subject, 'gla'),
                'num_beds': safe_get(subject, 'num_beds'),
                'num_baths': safe_get(subject, 'num_baths'),
                'condition': safe_get(subject, 'condition'),
                'sale_price': None,
                'sale_date': None,
                'distance_to_subject': None
            }
            records.append(subject_record)
            
            # Process comparable properties
            for comp in comps:
                gla = safe_get(comp, 'gla', '').split()[0] if comp.get('gla') else None
                
                comp_record = {
                    'type': 'comp',
                    'orderID': appraisal['orderID'],
                    'address': safe_get(comp, 'address'),
                    'city_province_zip': safe_get(comp, 'city_province'),
                    'effective_date': None,
                    'municipality_district': None,
                    'structure_type': safe_get(comp, 'prop_type'),
                    'year_built': safe_get(comp, 'age'),
                    'gla': gla,
                    'num_beds': safe_get(comp, 'bed_count'),
                    'num_baths': safe_get(comp, 'bath_count'),
                    'condition': safe_get(comp, 'condition'),
                    'sale_price': safe_get(comp, 'sale_price', '').replace(',', '') if comp.get('sale_price') else None,
                    'sale_date': safe_get(comp, 'sale_date'),
                    'distance_to_subject': safe_get(comp, 'distance_to_subject')
                }
                records.append(comp_record)
            
            # Process properties
            for prop in properties:
                city = safe_get(prop, 'city', '')
                province = safe_get(prop, 'province', '')
                postal_code = safe_get(prop, 'postal_code', '')
                
                prop_record = {
                    'type': 'property',
                    'orderID': appraisal['orderID'],
                    'address': safe_get(prop, 'address'),
                    'city_province_zip': f"{city}, {province}, {postal_code}",
                    'effective_date': None,
                    'municipality_district': None,
                    'structure_type': safe_get(prop, 'structure_type'),
                    'year_built': safe_get(prop, 'year_built'),
                    'gla': safe_get(prop, 'gla'),
                    'num_beds': safe_get(prop, 'bedrooms'),
                    'num_baths': f"{safe_get(prop, 'full_baths', 0)}:{safe_get(prop, 'half_baths', 0)}",
                    'condition': None,
                    'sale_price': safe_get(prop, 'close_price'),
                    'sale_date': safe_get(prop, 'close_date'),
                    'distance_to_subject': None
                }
                records.append(prop_record)
        
        df = pd.DataFrame(records)
        
        # Data preprocessing
        numeric_cols = ['gla', 'sale_price', 'year_built', 'num_beds']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['distance_km'] = df['distance_to_subject'].str.extract(r'(\d+\.\d+)', expand=False).astype(float)
        
        date_cols = ['effective_date', 'sale_date']
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Fill missing values
        df["num_baths"].fillna(value='2:0', inplace=True)
        df.num_beds.fillna(value=df.num_beds.median(), inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

def train_model_from_data(data):
    """Train the ML model exactly matching the Colab script"""
    try:
        print("Training model using exact Colab methodology...")
        
        # EXACT PREPROCESSING FROM YOUR COLAB SCRIPT
        
        # Step 1: Convert object types to categorical (CRITICAL MISSING STEP!)
        tmp = data.copy()
        for label, content in tmp.items():
            if pd.api.types.is_object_dtype(content):
                tmp[label] = content.astype("category").cat.as_ordered()
        
        # Step 2: Handle non-numeric missing values (CRITICAL MISSING STEP!)
        for label, content in tmp.items():
            if not pd.api.types.is_numeric_dtype(content):
                tmp[label + "_is_missing"] = content.isnull()
                if content.isna().any():
                    tmp[label] = pd.Categorical(content).codes + 1
        
        # Step 3: Handle numeric missing values (CRITICAL MISSING STEP!)
        for label, content in tmp.items():
            if pd.api.types.is_numeric_dtype(content):
                if content.isnull().any():  # Check if there are any missing values
                    tmp[label + "_is_missing"] = content.isnull()
                    tmp[label] = content.fillna(content.median())
        
        # Step 4: Scale numerical columns (EXACT FROM YOUR SCRIPT)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numerical_cols = ['year_built', 'gla', 'num_beds', 'sale_price', 'distance_km']
        x_scaled = tmp.copy()
        
        # Only scale columns that exist
        cols_to_scale = [col for col in numerical_cols if col in x_scaled.columns]
        if cols_to_scale:
            x_scaled[cols_to_scale] = scaler.fit_transform(tmp[cols_to_scale])
        
        # Step 5: Drop is_missing columns (EXACT FROM YOUR SCRIPT)
        X_clean = x_scaled.drop([col for col in x_scaled.columns if 'is_missing' in col], axis=1)
        
        # Step 6: Convert bathrooms to numeric (EXACT FROM YOUR SCRIPT)
        if 'num_baths' in X_clean.columns and not pd.api.types.is_numeric_dtype(X_clean['num_baths']):
            X_clean['num_baths_numeric'] = X_clean['num_baths'].apply(bath_to_numeric)
        
        # Step 7: Get potential_comps (EXACT FROM YOUR SCRIPT)
        potential_comps_processed = X_clean[(X_clean['type'] == 'comp') | (X_clean['type'] == 'property')]
        
        # Step 8: EXACT model training from your script
        # Create feature weights (EXACT FROM YOUR SCRIPT)
        feature_weights = {
            'structure_type': 3.0,
            'gla': 2.5,
            'year_built': 1.5,
            'num_beds': 1.0,
            'distance_km': 2.0
        }
        
        # Prepare the dataset - select only numerical columns (EXACT FROM YOUR SCRIPT)
        X = potential_comps_processed.copy()
        
        # List of features to use (we'll only use numeric ones) (EXACT FROM YOUR SCRIPT)
        numeric_features = []
        for feature in feature_weights.keys():
            if feature in X.columns:
                if pd.api.types.is_numeric_dtype(X[feature]):
                    numeric_features.append(feature)
                else:
                    print(f"Skipping non-numeric feature: {feature}")
        
        if len(numeric_features) < 2:
            print("Not enough numeric features for training")
            return False
        
        # Get a subset with only numeric features (EXACT FROM YOUR SCRIPT)
        X_numeric = X[numeric_features].copy()
        
        if len(X_numeric) < 10:
            print("Not enough data for training")
            return False
        
        # Calculate a similarity target based on weighted features (EXACT FROM YOUR SCRIPT)
        y = np.zeros(len(X_numeric))
        for feature, weight in feature_weights.items():
            if feature in numeric_features:
                # Normalize features (EXACT FROM YOUR SCRIPT)
                if X_numeric[feature].std() > 0:
                    X_numeric[feature] = (X_numeric[feature] - X_numeric[feature].mean()) / X_numeric[feature].std()
                
                # Add weighted contribution to similarity (EXACT FROM YOUR SCRIPT)
                y += np.abs(X_numeric[feature].values) * weight
        
        # Invert so higher values mean more similar (EXACT FROM YOUR SCRIPT)
        y = max(y) - y
        
        # Split data into train and test sets (EXACT FROM YOUR SCRIPT)
        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)
        
        # Create and train the model (EXACT FROM YOUR SCRIPT)
        global model_metrics
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
        ])
        
        # Train the model (EXACT FROM YOUR SCRIPT)
        model.fit(X_train, y_train)
        
        # Make predictions (EXACT FROM YOUR SCRIPT)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate accuracy metrics (EXACT FROM YOUR SCRIPT)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Get feature importance (EXACT FROM YOUR SCRIPT)
        feature_importance = dict(zip(numeric_features, model.named_steps['model'].feature_importances_))
        
        # Store metrics globally
        model_metrics = {
            'training_mse': round(train_mse, 4),
            'test_mse': round(test_mse, 4),
            'training_mae': round(train_mae, 4),
            'test_mae': round(test_mae, 4),
            'training_r2': round(train_r2, 4),
            'test_r2': round(test_r2, 4),
            'features_used': numeric_features,
            'feature_importance': feature_importance,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'total_properties': len(X_numeric),
            'target_type': 'similarity_score'
        }
        
        # Save model
        with open('property_comp_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save metrics
        with open('model_metrics.json', 'w') as f:
            json.dump(model_metrics, f, indent=2)
        
        # Save the exact scaler used in training
        joblib.dump(scaler, 'scaler.pkl')
        
        # Update global variables with processed data
        global potential_comps, knn_model
        potential_comps = X_clean  # Use the fully processed data
        
        # Initialize KNN model (EXACT FROM YOUR SCRIPT)
        comp_features = ['structure_type', 'gla', 'num_beds', 'num_baths_numeric', 'year_built', 'condition', 'distance_km']
        available_features = [f for f in comp_features if f in potential_comps.columns]
        
        if available_features and len(potential_comps) > 5:
            knn_model = NearestNeighbors(
                n_neighbors=10,        # EXACT FROM YOUR SCRIPT
                algorithm='ball_tree', # EXACT FROM YOUR SCRIPT
                metric='manhattan',    # EXACT FROM YOUR SCRIPT
                p=1                    # EXACT FROM YOUR SCRIPT
            )
            
            # Prepare features with appropriate handling for sale_date (EXACT FROM YOUR SCRIPT)
            knn_features = available_features.copy()
            if 'sale_date' in knn_features:
                knn_features.remove('sale_date')  # Handle separately
            
            # Fit model on non-date features (EXACT FROM YOUR SCRIPT)
            if knn_features:
                knn_model.fit(potential_comps[knn_features].fillna(0))
                print("KNN model initialized successfully")
        
        # Print metrics (EXACT FORMAT FROM YOUR SCRIPT)
        print(f"\nModel Performance Metrics:")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_random_sample_properties(n_samples=8):
    """Get random sample properties from the loaded dataset plus specific test cases"""
    global potential_comps
    
    sample_properties = []
    
    # Add specific test case first
    specific_property = {
        'name': '142-950 Oakview Ave Kingston ON K7M 6W8',
        'address': '142-950 Oakview Ave Kingston ON K7M 6W8',
        'structure_type': 'Condo',
        'gla': 1250,
        'year_built': 2015,
        'num_beds': 2,
        'num_baths': '2:0'
    }
    sample_properties.append(specific_property)
    
    if potential_comps is None or len(potential_comps) == 0:
        # If no dataset loaded, return just the specific property
        return sample_properties
    
    try:
        # Check if the specific property exists in the dataset
        if 'address' in potential_comps.columns:
            kingston_match = potential_comps[
                potential_comps['address'].str.contains('142-950 Oakview Ave', case=False, na=False) |
                potential_comps['address'].str.contains('Oakview Ave', case=False, na=False)
            ]
            
            if len(kingston_match) > 0:
                # Use the actual data from the dataset
                row = kingston_match.iloc[0]
                
                # Get structure type (handle both numeric and string)
                type_mapping = {
                    1: 'Detached', 2: 'Townhouse', 3: 'Condo', 
                    4: 'Semi-Detached', 5: 'Duplex', 6: 'Apartment'
                }
                
                structure_type = row.get('structure_type_name', row.get('structure_type'))
                if pd.api.types.is_numeric_dtype(type(structure_type)):
                    structure_type = type_mapping.get(structure_type, 'Condo')
                
                # Format bathrooms
                num_baths = row.get('num_baths', '2:0')
                if pd.isna(num_baths):
                    num_baths = '2:0'
                
                # Update the specific property with real data
                sample_properties[0] = {
                    'name': '142-950 Oakview Ave Kingston ON K7M 6W8',
                    'address': str(row.get('address', '142-950 Oakview Ave Kingston ON K7M 6W8')),
                    'structure_type': str(structure_type),
                    'gla': int(row['gla']) if pd.notna(row['gla']) else 1250,
                    'year_built': int(row['year_built']) if pd.notna(row['year_built']) else 2015,
                    'num_beds': int(row['num_beds']) if pd.notna(row['num_beds']) else 2,
                    'num_baths': str(num_baths)
                }
        
        # Filter for properties with complete data for random selection
        complete_data = potential_comps.dropna(subset=['structure_type', 'gla', 'year_built', 'num_beds'])
        
        if len(complete_data) == 0:
            return sample_properties  # Return just the specific property
        
        # Get random sample (n_samples - 1 since we already have the specific property)
        remaining_samples = n_samples - 1
        sample_size = min(remaining_samples, len(complete_data))
        
        if sample_size > 0:
            random_sample = complete_data.sample(n=sample_size, random_state=42)  # Fixed seed for consistency
            
            # Create mapping for encoded structure types if needed
            type_mapping = {
                1: 'Detached', 2: 'Townhouse', 3: 'Condo', 
                4: 'Semi-Detached', 5: 'Duplex', 6: 'Apartment'
            }
            
            for idx, row in random_sample.iterrows():
                # Skip if this is the Kingston property (to avoid duplicates)
                if 'address' in row and pd.notna(row['address']):
                    if 'Oakview Ave' in str(row['address']):
                        continue
                
                # Get structure type (handle both numeric and string)
                structure_type = row.get('structure_type_name', row.get('structure_type'))
                if pd.api.types.is_numeric_dtype(type(structure_type)):
                    structure_type = type_mapping.get(structure_type, 'Unknown')
                
                # Get address or create a display name
                address = row.get('address', f"Property #{idx}")
                if pd.isna(address) or address == '' or address == 'nan':
                    address = f"Property #{idx}"
                
                # Format bathrooms
                num_baths = row.get('num_baths', '2:0')
                if pd.isna(num_baths):
                    num_baths = '2:0'
                
                # Create property object
                property_data = {
                    'name': f"{address[:30]}..." if len(str(address)) > 30 else str(address),
                    'address': str(address),
                    'structure_type': str(structure_type),
                    'gla': int(row['gla']) if pd.notna(row['gla']) else 2000,
                    'year_built': int(row['year_built']) if pd.notna(row['year_built']) else 2000,
                    'num_beds': int(row['num_beds']) if pd.notna(row['num_beds']) else 3,
                    'num_baths': str(num_baths)
                }
                
                sample_properties.append(property_data)
        
        print(f"Generated {len(sample_properties)} sample properties (including Kingston property) from dataset")
        return sample_properties
        
    except Exception as e:
        print(f"Error generating sample properties: {e}")
        return sample_properties  # Return at least the specific property

def get_top_comps(subject_property, candidate_properties, knn_model, k=3, model_features=None):
    """EXACT get_top_comps function from your Colab script"""
    
    if model_features is None:
        model_features = ['structure_type', 'gla', 'num_beds', 'num_baths_numeric', 'year_built', 'condition', 'distance_km']
    
    # Remove sale_date from features for KNN (EXACT FROM YOUR SCRIPT)
    comp_features = model_features.copy()
    if 'sale_date' in comp_features:
        comp_features.remove('sale_date')  # Handle separately
    
    # Filter features that exist in both datasets
    available_features = [f for f in comp_features if f in candidate_properties.columns and f in subject_property.columns]
    
    if not available_features:
        print("No matching features available for comparison")
        return pd.DataFrame()
    
    try:
        # Ensure we use a DataFrame with proper column names (EXACT FROM YOUR SCRIPT)
        subject_features = subject_property[available_features].copy()

        # Find nearest neighbors (EXACT FROM YOUR SCRIPT)  
        distances, indices = knn_model.kneighbors(subject_features, n_neighbors=min(30, len(candidate_properties)))

        # Get the actual properties and their distances (EXACT FROM YOUR SCRIPT)
        neighbor_indices = indices[0]
        potential_matches = candidate_properties.iloc[neighbor_indices].copy()
        potential_matches['distance_score'] = distances[0]

        # Apply domain-specific rules: (EXACT FROM YOUR SCRIPT)

        # 1. Filter for same structure type first (critical) (EXACT FROM YOUR SCRIPT)
        subject_structure = subject_property['structure_type'].values[0]
        structure_matches = potential_matches[potential_matches['structure_type'] == subject_structure].copy()

        # If no structure matches, use all potential matches (EXACT FROM YOUR SCRIPT)
        if len(structure_matches) < 3:
            structure_matches = potential_matches.copy()

        # 2. Handle sale_date prioritization (EXACT FROM YOUR SCRIPT)
        if 'sale_date' in structure_matches.columns:
            structure_matches['days_since_sale'] = pd.NA  # Initialize

            mask = ~pd.isna(structure_matches['sale_date'])
            if mask.any():
                current_date = pd.Timestamp.now()
                structure_matches.loc[mask, 'days_since_sale'] = (
                    current_date - pd.to_datetime(structure_matches.loc[mask, 'sale_date'])
                ).dt.days

                # Prioritize recent sales (EXACT FROM YOUR SCRIPT)
                recent_sales = structure_matches[structure_matches['days_since_sale'] <= 90].copy()

                # If enough recent sales, use only those (EXACT FROM YOUR SCRIPT)
                if len(recent_sales) >= k:
                    structure_matches = recent_sales

        # 3. Remove duplicates by address (EXACT FROM YOUR SCRIPT)
        if 'address' in structure_matches.columns:
            structure_matches = structure_matches.drop_duplicates(subset=['address'])

        # 4. Sort by distance score and return top k (EXACT FROM YOUR SCRIPT)
        final_comps = structure_matches.sort_values('distance_score').head(k)

        return final_comps
        
    except Exception as e:
        print(f"Error in get_top_comps: {e}")
        return pd.DataFrame()

def calculate_manual_similarity(subject_data, comp_property):
    """Calculate similarity score using the weighted approach from original script"""
    similarity_score = 0
    total_weight = 0
    
    # Feature weights from original script
    feature_weights = {
        'structure_type': 3.0,
        'gla': 2.5,
        'year_built': 1.5,
        'num_beds': 1.0,
        'distance_km': 2.0
    }
    
    # Structure type match (exact match gets high score)
    comp_structure = comp_property.get('structure_type_name', comp_property.get('structure_type'))
    if comp_structure == subject_data.get('structure_type'):
        similarity_score += feature_weights['structure_type']
    total_weight += feature_weights['structure_type']
    
    # GLA similarity (normalized difference)
    if pd.notna(comp_property.get('gla')) and subject_data.get('gla'):
        try:
            comp_gla = float(comp_property['gla'])
            subject_gla = float(subject_data['gla'])
            if comp_gla > 0 and subject_gla > 0:
                gla_diff = abs(comp_gla - subject_gla) / max(comp_gla, subject_gla)
                similarity_score += (1 - min(gla_diff, 1)) * feature_weights['gla']
        except (ValueError, TypeError):
            pass
    total_weight += feature_weights['gla']
    
    # Year built similarity
    if pd.notna(comp_property.get('year_built')) and subject_data.get('year_built'):
        try:
            year_diff = abs(float(comp_property['year_built']) - float(subject_data['year_built']))
            similarity_score += max(0, 1 - year_diff/50) * feature_weights['year_built']
        except (ValueError, TypeError):
            pass
    total_weight += feature_weights['year_built']
    
    # Bedroom similarity
    if pd.notna(comp_property.get('num_beds')) and subject_data.get('num_beds'):
        try:
            bed_diff = abs(float(comp_property['num_beds']) - float(subject_data['num_beds']))
            similarity_score += max(0, 1 - bed_diff/5) * feature_weights['num_beds']
        except (ValueError, TypeError):
            pass
    total_weight += feature_weights['num_beds']
    
    # Distance similarity (if available)
    if pd.notna(comp_property.get('distance_km')):
        try:
            distance_score = max(0, 1 - float(comp_property['distance_km'])/10)
            similarity_score += distance_score * feature_weights['distance_km']
        except (ValueError, TypeError):
            pass
    total_weight += feature_weights['distance_km']
    
    # Normalize similarity score
    return similarity_score / total_weight if total_weight > 0 else 0

def format_comp_for_display(comp_dict, type_mapping):
    """Format comparable property for display"""
    reverse_mapping = {v: k for k, v in type_mapping.items()}
    
    # Format structure type for display
    if 'structure_type_name' in comp_dict:
        comp_dict['structure_type'] = comp_dict['structure_type_name']
    elif pd.api.types.is_numeric_dtype(type(comp_dict.get('structure_type'))):
        comp_dict['structure_type'] = reverse_mapping.get(comp_dict['structure_type'], 'Unknown')
    
    # Format sale price if available
    if pd.notna(comp_dict.get('sale_price')):
        try:
            price = float(str(comp_dict['sale_price']).replace(',', '').replace('$', ''))
            comp_dict['sale_price_formatted'] = f"${price:,.0f}"
        except (ValueError, TypeError):
            comp_dict['sale_price_formatted'] = str(comp_dict['sale_price'])
    
    # Format address
    if pd.isna(comp_dict.get('address')) or comp_dict.get('address') == '':
        comp_dict['address'] = f"Property (No Address)"
    
    # Format date
    if pd.notna(comp_dict.get('sale_date')):
        try:
            if isinstance(comp_dict['sale_date'], str):
                comp_dict['sale_date'] = comp_dict['sale_date']
            else:
                comp_dict['sale_date'] = str(comp_dict['sale_date'])
        except:
            comp_dict['sale_date'] = 'Unknown'
    
    return comp_dict

def find_comparable_properties(subject_data, n_comps=3):
    """Find comparable properties using the original script approach with KNN + manual similarity"""
    global potential_comps, knn_model
    
    if potential_comps is None or len(potential_comps) == 0:
        return []
    
    try:
        # Convert subject data to DataFrame format for KNN
        subject_df = pd.DataFrame([subject_data])
        
        # Convert bathroom format
        if 'num_baths' in subject_df.columns:
            subject_df['num_baths_numeric'] = subject_df['num_baths'].apply(bath_to_numeric)
        
        # Handle structure type encoding
        structure_type = subject_data.get('structure_type')
        type_mapping = {'Detached': 1, 'Townhouse': 2, 'Condo': 3, 'Semi-Detached': 4, 'Duplex': 5, 'Apartment': 6}
        
        if structure_type in type_mapping:
            subject_df['structure_type_encoded'] = type_mapping[structure_type]
        
        # Prepare comparables data
        comps_data = potential_comps.copy()
        
        # Handle structure type in comparables
        if 'structure_type_name' not in comps_data.columns and pd.api.types.is_numeric_dtype(comps_data['structure_type']):
            reverse_mapping = {v: k for k, v in type_mapping.items()}
            comps_data['structure_type_name'] = comps_data['structure_type'].map(reverse_mapping)
        
        # Use KNN to find similar properties (following original script approach)
        if knn_model is not None:
            # Features available for KNN comparison
            knn_features = ['gla', 'year_built', 'num_beds']
            available_knn_features = [f for f in knn_features if f in comps_data.columns and f in subject_df.columns]
            
            if available_knn_features:
                try:
                    # Use the get_top_comps function from original script
                    recommended_comps = get_top_comps(
                        subject_df, 
                        comps_data, 
                        knn_model, 
                        k=n_comps, 
                        model_features=available_knn_features
                    )
                    
                    if len(recommended_comps) > 0:
                        # Calculate manual similarity scores (weighted approach from original script)
                        results = []
                        for idx, comp in recommended_comps.iterrows():
                            similarity_score = calculate_manual_similarity(subject_data, comp)
                            comp_dict = comp.to_dict()
                            comp_dict['similarity_score'] = similarity_score
                            
                            # Format for display
                            comp_dict = format_comp_for_display(comp_dict, type_mapping)
                            results.append(comp_dict)
                        
                        # Sort by similarity score
                        results.sort(key=lambda x: x['similarity_score'], reverse=True)
                        return results
                except Exception as e:
                    print(f"KNN comparison failed: {e}")
        
        # Fallback to manual similarity calculation if KNN fails
        return find_comparable_properties_manual(subject_data, n_comps)
        
    except Exception as e:
        print(f"Error in find_comparable_properties: {e}")
        import traceback
        traceback.print_exc()
        return []

def find_comparable_properties_manual(subject_data, n_comps=3):
    """Fallback manual similarity calculation"""
    global potential_comps
    
    similarities = []
    
    for idx, comp in potential_comps.iterrows():
        similarity_score = calculate_manual_similarity(subject_data, comp)
        similarities.append((idx, similarity_score))
    
    # Sort by similarity and get top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in similarities[:n_comps]]
    
    # Format results
    results = []
    type_mapping = {'Detached': 1, 'Townhouse': 2, 'Condo': 3, 'Semi-Detached': 4, 'Duplex': 5, 'Apartment': 6}
    
    for i, idx in enumerate(top_indices):
        comp = potential_comps.loc[idx].copy()
        comp_dict = comp.to_dict()
        comp_dict['similarity_score'] = similarities[i][1]
        comp_dict = format_comp_for_display(comp_dict, type_mapping)
        results.append(comp_dict)
    
    return results

# Routes
@app.route('/')
def home():
    # Check if we have data loaded
    data_status = {
        'has_model': model is not None,
        'has_data': potential_comps is not None and len(potential_comps) > 0,
        'data_count': len(potential_comps) if potential_comps is not None else 0,
        'has_metrics': model_metrics is not None
    }
    
    # Get random sample properties from the actual dataset
    sample_properties = get_random_sample_properties(4)  # Just 4 for home page preview
    
    return render_template('index.html', data_status=data_status, sample_properties=sample_properties, metrics=model_metrics)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded file
            if filename.endswith('.json'):
                try:
                    df = process_appraisals_data(filepath)
                    if df is not None:
                        # Save processed data
                        global potential_comps
                        potential_comps = df[(df['type'] == 'comp') | (df['type'] == 'property')]
                        
                        # Handle numeric conversion for encoded data
                        if 'structure_type' in potential_comps.columns:
                            # If structure_type is numeric (encoded), create a mapping
                            if pd.api.types.is_numeric_dtype(potential_comps['structure_type']):
                                # Create reverse mapping for display
                                type_mapping = {
                                    1: 'Detached', 2: 'Townhouse', 3: 'Condo', 
                                    4: 'Semi-Detached', 5: 'Duplex', 6: 'Apartment'
                                }
                                potential_comps['structure_type_name'] = potential_comps['structure_type'].map(type_mapping)
                        
                        # Convert num_baths if needed
                        if 'num_baths' in potential_comps.columns:
                            potential_comps['num_baths_numeric'] = potential_comps['num_baths'].apply(bath_to_numeric)
                        
                        potential_comps.to_csv('potential_comps.csv', index=False)
                        
                        # Train/update the model
                        success = train_model_from_data(potential_comps)
                        
                        if success:
                            flash(f'🎉 File processed successfully! Found {len(df)} total records, {len(potential_comps)} comparable properties. Similarity model trained and ready for predictions!')
                        else:
                            flash(f'✅ File processed successfully! Found {len(df)} total records, {len(potential_comps)} comparable properties. Note: Model training skipped due to insufficient data.')
                        
                        # Reload models and data
                        load_models_and_data()
                    else:
                        flash('❌ Error processing the uploaded file. Please check the JSON format.')
                except Exception as e:
                    flash(f'❌ Error processing file: {str(e)}')
            else:
                flash('✅ File uploaded successfully!')
            
            return redirect(url_for('upload_file'))
        else:
            flash('❌ Invalid file type. Please upload JSON or CSV files.')
    
    return render_template('upload.html')

@app.route('/find-comps', methods=['GET', 'POST'])
def find_comps():
    # Get random sample properties from the actual dataset
    sample_properties = get_random_sample_properties(8)
    
    if request.method == 'POST':
        # Check if it's a sample property selection
        sample_property = request.form.get('sample_property')
        
        if sample_property and sample_property != 'custom':
            # Find the selected sample property by name
            selected_sample = None
            for sample in sample_properties:
                if sample['name'] == sample_property:
                    selected_sample = sample
                    break
            
            if selected_sample:
                subject_data = selected_sample.copy()
                print(f"Finding comps for sample property: {selected_sample['address']}")
            else:
                flash('Invalid sample property selected.')
                return render_template('find_comps.html', sample_properties=sample_properties)
        else:
            # Get custom form data
            subject_data = {
                'structure_type': request.form.get('structure_type'),
                'gla': request.form.get('gla'),
                'year_built': request.form.get('year_built'),
                'num_beds': request.form.get('num_beds'),
                'num_baths': request.form.get('num_baths'),
                'address': request.form.get('address', 'Subject Property'),
                'latitude': request.form.get('latitude'),
                'longitude': request.form.get('longitude')
            }
            print(f"Finding comps for custom property: {subject_data['address']}")
        
        # Validate required fields
        required_fields = ['structure_type', 'gla', 'year_built', 'num_beds']
        missing_fields = [field for field in required_fields if not subject_data.get(field)]
        
        if missing_fields:
            flash(f'Please fill in all required fields: {", ".join(missing_fields)}')
            return render_template('find_comps.html', sample_properties=sample_properties)
        
        # Check if we have data to compare against
        if potential_comps is None or len(potential_comps) == 0:
            flash('⚠️ No comparable properties data available. Please upload a dataset first.')
            return render_template('find_comps.html', sample_properties=sample_properties)
        
        # Find comparable properties
        comps = find_comparable_properties(subject_data, n_comps=3)
        
        if comps:
            print(f"Found {len(comps)} comparable properties")
            return render_template('comp_results.html', subject=subject_data, comps=comps)
        else:
            flash('❌ No comparable properties found. Please try different criteria or upload more data.')
    
    return render_template('find_comps.html', sample_properties=sample_properties)

@app.route('/model-metrics')
def model_metrics_page():
    """Display detailed model performance metrics"""
    global model_metrics
    return render_template('model_metrics.html', metrics=model_metrics)

@app.route('/about')
def about():
    return render_template('about.html')

# API endpoint for programmatic access
@app.route('/api/find-comps', methods=['POST'])
def api_find_comps():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Find comparable properties
        comps = find_comparable_properties(data, n_comps=3)
        
        # Convert to JSON-serializable format
        results = []
        for comp in comps:
            comp_dict = {}
            for key, value in comp.items():
                if pd.isna(value):
                    comp_dict[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    comp_dict[key] = float(value)
                else:
                    comp_dict[key] = str(value)
            results.append(comp_dict)
        
        return jsonify({
            'subject': data,
            'comparables': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load models and data on startup
    load_models_and_data()
    app.run(debug=True)