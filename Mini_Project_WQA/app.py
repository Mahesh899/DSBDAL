# water_quality_analyzer.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class WaterQualityAnalyzer:
    def __init__(self):
        # Define acceptable ranges for different water uses
        self.ph_range = {"drinking": (6.5, 8.5), "agriculture": (6.0, 8.5), "industrial": (6.0, 9.0)}
        self.turbidity_range = {"drinking": (0, 1), "agriculture": (0, 10), "industrial": (0, 50)}
        self.tds_range = {"drinking": (0, 500), "agriculture": (0, 2000), "industrial": (0, 3000)}
        self.do_range = {"drinking": (5, 10), "agriculture": (3, 10), "industrial": (2, 10)}
        self.conductivity_range = {"drinking": (200, 800), "agriculture": (200, 2000), "industrial": (200, 3000)}
        self.temperature_range = {"drinking": (5, 30), "agriculture": (5, 35), "industrial": (5, 40)}
        
        self.models = {
            "drinking": None,
            "agriculture": None,
            "industrial": None
        }
        
        self.scaler = MinMaxScaler()
        
        # Load and preprocess data
        self._load_and_preprocess_data()
        self._train_models()
    
    def _load_and_preprocess_data(self):
        # Load the CSV data
        data = pd.read_csv('Water Quality Testing.csv')
        
        # Convert to DataFrame
        self.df = pd.DataFrame(data)
        
        # Calculate TDS (Total Dissolved Solids) from conductivity
        # Using approximate conversion factor of 0.67 (varies by water composition)
        self.df['TDS'] = self.df['Conductivity (µS/cm)'] * 0.67
        
        # Prepare features (X) and labels (y) for each use case
        self.X = self.df[['pH', 'Turbidity (NTU)', 'TDS', 'Dissolved Oxygen (mg/L)', 
                         'Conductivity (µS/cm)', 'Temperature (°C)']].values
        
        # Scale the features
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # Create labels for drinking water safety
        y_drinking = np.zeros(len(self.df))
        for i, row in self.df.iterrows():
            if (6.5 <= row['pH'] <= 8.5 and
                row['Turbidity (NTU)'] < 1 and
                row['TDS'] < 500 and
                row['Dissolved Oxygen (mg/L)'] >= 5 and
                200 <= row['Conductivity (µS/cm)'] <= 800 and
                5 <= row['Temperature (°C)'] <= 30):
                y_drinking[i] = 1
        self.y_drinking = y_drinking
        
        # Create labels for agricultural water safety
        y_agriculture = np.zeros(len(self.df))
        for i, row in self.df.iterrows():
            if (6.0 <= row['pH'] <= 8.5 and
                row['TDS'] < 2000 and
                row['Dissolved Oxygen (mg/L)'] >= 3):
                y_agriculture[i] = 1
        self.y_agriculture = y_agriculture
        
        # Create labels for industrial water safety
        y_industrial = np.zeros(len(self.df))
        for i, row in self.df.iterrows():
            if (6.0 <= row['pH'] <= 9.0 and
                row['TDS'] < 3000):
                y_industrial[i] = 1
        self.y_industrial = y_industrial
    
    def _train_models(self):
        # Split data into training and testing sets
        X_train, X_test, y_drinking_train, y_drinking_test = train_test_split(
            self.X_scaled, self.y_drinking, test_size=0.2, random_state=42
        )
        
        # Train drinking water model
        self.models["drinking"] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models["drinking"].fit(X_train, y_drinking_train)
        
        # Train agriculture water model
        X_train, X_test, y_agri_train, y_agri_test = train_test_split(
            self.X_scaled, self.y_agriculture, test_size=0.2, random_state=42
        )
        self.models["agriculture"] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models["agriculture"].fit(X_train, y_agri_train)
        
        # Train industrial water model
        X_train, X_test, y_ind_train, y_ind_test = train_test_split(
            self.X_scaled, self.y_industrial, test_size=0.2, random_state=42
        )
        self.models["industrial"] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models["industrial"].fit(X_train, y_ind_train)
    
    def analyze_water(self, ph, turbidity, tds, dissolved_oxygen, conductivity, temperature):
        # Validate inputs
        input_params = [ph, turbidity, tds, dissolved_oxygen, conductivity, temperature]
        if any(param < 0 for param in input_params):
            raise ValueError("All parameters must be non-negative")
        
        if not 0 <= ph <= 14:
            raise ValueError("pH must be between 0 and 14")
        
        # Calculate individual parameter scores (0-100)
        ph_score = self._calculate_ph_score(ph)
        turbidity_score = self._calculate_turbidity_score(turbidity)
        tds_score = self._calculate_tds_score(tds)
        do_score = self._calculate_do_score(dissolved_oxygen)
        conductivity_score = self._calculate_conductivity_score(conductivity)
        temperature_score = self._calculate_temperature_score(temperature)
        
        # Calculate overall quality score (weighted average)
        overall_score = (
            ph_score * 0.20 +
            turbidity_score * 0.20 +
            tds_score * 0.20 +
            do_score * 0.15 +
            conductivity_score * 0.15 +
            temperature_score * 0.10
        )
        
        # Determine quality category
        quality = self._determine_quality_category(overall_score)
        
        # Use ML model to predict safety for different uses
        features = np.array([[ph, turbidity, tds, dissolved_oxygen, conductivity, temperature]])
        features_scaled = self.scaler.transform(features)
        
        safe_for_drinking = bool(self.models["drinking"].predict(features_scaled)[0])
        safe_for_agriculture = bool(self.models["agriculture"].predict(features_scaled)[0])
        safe_for_industrial = bool(self.models["industrial"].predict(features_scaled)[0])
        
        # Generate explanations for each usage
        drinking_explanation = self._generate_drinking_explanation(
            ph, turbidity, tds, dissolved_oxygen, conductivity, temperature, safe_for_drinking
        )
        
        agriculture_explanation = self._generate_agriculture_explanation(
            ph, turbidity, tds, dissolved_oxygen, conductivity, temperature, safe_for_agriculture
        )
        
        industrial_explanation = self._generate_industrial_explanation(
            ph, turbidity, tds, dissolved_oxygen, conductivity, temperature, safe_for_industrial
        )
        
        # Parameter analysis
        parameter_analysis = {
            "ph": {
                "value": ph,
                "score": ph_score,
                "status": self._get_parameter_status(ph_score),
                "description": self._get_ph_description(ph)
            },
            "turbidity": {
                "value": turbidity,
                "score": turbidity_score,
                "status": self._get_parameter_status(turbidity_score),
                "description": self._get_turbidity_description(turbidity)
            },
            "tds": {
                "value": tds,
                "score": tds_score,
                "status": self._get_parameter_status(tds_score),
                "description": self._get_tds_description(tds)
            },
            "dissolved_oxygen": {
                "value": dissolved_oxygen,
                "score": do_score,
                "status": self._get_parameter_status(do_score),
                "description": self._get_do_description(dissolved_oxygen)
            },
            "conductivity": {
                "value": conductivity,
                "score": conductivity_score,
                "status": self._get_parameter_status(conductivity_score),
                "description": self._get_conductivity_description(conductivity)
            },
            "temperature": {
                "value": temperature,
                "score": temperature_score,
                "status": self._get_parameter_status(temperature_score),
                "description": self._get_temperature_description(temperature)
            }
        }
        
        # Return complete analysis
        return {
            "score": overall_score,
            "quality": quality,
            "safe_for_drinking": safe_for_drinking,
            "safe_for_agriculture": safe_for_agriculture,
            "safe_for_industrial": safe_for_industrial,
            "drinking_explanation": drinking_explanation,
            "agriculture_explanation": agriculture_explanation,
            "industrial_explanation": industrial_explanation,
            "parameter_analysis": parameter_analysis
        }
    
    # All the helper methods (_calculate_ph_score, _get_ph_description, etc.) remain the same
    # as in the original code since they don't need to change
    
    def _calculate_ph_score(self, ph):
        if 6.5 <= ph <= 8.5:
            return 100
        elif 6.0 <= ph < 6.5 or 8.5 < ph <= 9.0:
            return 80
        elif 5.5 <= ph < 6.0 or 9.0 < ph <= 9.5:
            return 50
        else:
            return 20
    
    def _calculate_turbidity_score(self, turbidity):
        if turbidity < 1:
            return 100
        elif 1 <= turbidity < 5:
            return 80
        elif 5 <= turbidity < 10:
            return 60
        else:
            return 30
    
    def _calculate_tds_score(self, tds):
        if tds < 300:
            return 100
        elif 300 <= tds < 500:
            return 90
        elif 500 <= tds < 1000:
            return 70
        elif 1000 <= tds < 2000:
            return 50
        else:
            return 30
    
    def _calculate_do_score(self, do):
        if 6 <= do <= 8:
            return 100
        elif 4 <= do < 6:
            return 70
        elif 8 < do <= 10:
            return 80
        elif 2 <= do < 4:
            return 40
        else:
            return 20
    
    def _calculate_conductivity_score(self, conductivity):
        if 200 <= conductivity <= 800:
            return 100
        elif 100 <= conductivity < 200 or 800 < conductivity <= 1500:
            return 80
        elif 1500 < conductivity <= 3000:
            return 50
        else:
            return 30
    
    def _calculate_temperature_score(self, temperature):
        if 10 <= temperature <= 25:
            return 100
        elif 5 <= temperature < 10 or 25 < temperature <= 30:
            return 80
        else:
            return 60
    
    def _determine_quality_category(self, score):
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 50:
            return "Fair"
        elif score >= 25:
            return "Poor"
        else:
            return "Dangerous"
    
    def _get_parameter_status(self, score):
        if score >= 80:
            return "normal"
        elif score >= 50:
            return "concern"
        else:
            return "danger"
    
    def _get_ph_description(self, ph):
        if 6.5 <= ph <= 8.5:
            return "Normal range"
        elif ph < 6.5:
            return "Too acidic"
        else:
            return "Too alkaline"
    
    def _get_turbidity_description(self, turbidity):
        if turbidity < 1:
            return "Clear water"
        elif 1 <= turbidity < 5:
            return "Slightly cloudy"
        else:
            return "Very cloudy"
    
    def _get_tds_description(self, tds):
        if tds < 300:
            return "Excellent"
        elif 300 <= tds < 500:
            return "Good"
        elif 500 <= tds < 1000:
            return "Elevated minerals"
        else:
            return "High mineral content"
    
    def _get_do_description(self, do):
        if 6 <= do <= 8:
            return "Good oxygen levels"
        elif do < 6:
            return "Low oxygen level"
        else:
            return "High oxygen"
    
    def _get_conductivity_description(self, conductivity):
        if 200 <= conductivity <= 800:
            return "Normal range"
        elif conductivity < 200:
            return "Low conductivity"
        else:
            return "High conductivity"
    
    def _get_temperature_description(self, temperature):
        if 10 <= temperature <= 25:
            return "Optimal temperature"
        elif temperature < 10:
            return "Very cold"
        else:
            return "Very warm"
    
    def _generate_drinking_explanation(self, ph, turbidity, tds, do, conductivity, temperature, is_safe):
        if is_safe:
            return "All key parameters are within recommended ranges for drinking water standards."
        
        issues = []
        if ph < 6.5 or ph > 8.5:
            issues.append("pH level")
        if turbidity >= 1:
            issues.append("turbidity")
        if tds >= 500:
            issues.append("TDS")
        if do < 5:
            issues.append("dissolved oxygen")
        if conductivity < 200 or conductivity > 800:
            issues.append("conductivity")
        
        if issues:
            return f"Water does not meet drinking standards due to {', '.join(issues)}."
        else:
            return "Water may not be suitable for drinking based on our predictive model."
    
    def _generate_agriculture_explanation(self, ph, turbidity, tds, do, conductivity, temperature, is_safe):
        if is_safe:
            return "Water quality is suitable for irrigation and agricultural purposes."
        
        issues = []
        if ph < 6.0 or ph > 8.5:
            issues.append("pH level")
        if tds >= 2000:
            issues.append("high mineral content")
        if do < 3:
            issues.append("low dissolved oxygen")
        
        if issues:
            return f"Water may not be ideal for agriculture due to {', '.join(issues)}."
        else:
            return "Water may not be suitable for agriculture based on our predictive model."
    
    def _generate_industrial_explanation(self, ph, turbidity, tds, do, conductivity, temperature, is_safe):
        if is_safe:
            return "Water is suitable for general industrial use, but specific processes may have additional requirements."
        
        issues = []
        if ph < 6.0 or ph > 9.0:
            issues.append("pH level")
        if tds >= 3000:
            issues.append("high dissolved solids")
        
        if issues:
            return f"Water may not be suitable for industrial use due to {', '.join(issues)}."
        else:
            return "Water may not be ideal for industrial applications based on our predictive model."


# Flask API to serve the water quality analyzer
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import render_template

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the water quality analyzer
analyzer = WaterQualityAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_water():
    """API endpoint to analyze water quality"""
    try:
        # Get data from request
        data = request.json
        
        # Extract parameters
        ph = float(data.get('ph', 0))
        turbidity = float(data.get('turbidity', 0))
        tds = float(data.get('tds', 0))
        dissolved_oxygen = float(data.get('dissolved_oxygen', 0))
        conductivity = float(data.get('conductivity', 0))
        temperature = float(data.get('temperature', 0))
        
        # Analyze water quality
        result = analyzer.analyze_water(
            ph, turbidity, tds, dissolved_oxygen, conductivity, temperature
        )
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)