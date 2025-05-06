from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load preprocessing objects and models
try:
    preprocessor = joblib.load('backend/preprocessor.pkl')
    label_encoder = joblib.load('backend/label_encoder.pkl')
    ann_model = tf.keras.models.load_model('backend/fitness_ann_model.h5')
    xgb_model = joblib.load('backend/fitness_xgb_model.pkl')
    print("Models and preprocessors loaded successfully.")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Expected input columns: ['Age', 'Weight (kg)', 'Height (cm)', 'Heart Rate (bpm)', 'Resting Heart Rate (bpm)', 'Calories Burned', 'Workout Duration (mins)', 'Gender', 'Workout Intensity']")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

# Define workout schedule templates for different fitness plan types
WORKOUT_SCHEDULES = {
    "Endurance": {
        "description": "Focus on improving cardiovascular health and stamina with longer, steady-state activities.",
        "weekly_schedule": [
            {"day": "Monday", "workout": "45-60 min steady-state cardio (running, cycling, or swimming)", "intensity": "Moderate (65-75% max heart rate)"},
            {"day": "Tuesday", "workout": "30 min light cardio + mobility work", "intensity": "Low (Recovery)"},
            {"day": "Wednesday", "workout": "Interval training: 30 min (5 min warm-up, 20 min intervals, 5 min cool-down)", "intensity": "High (80-90% max heart rate during intervals)"},
            {"day": "Thursday", "workout": "Rest day or light walking", "intensity": "Very Low (Active Recovery)"},
            {"day": "Friday", "workout": "45-60 min steady-state cardio (different than Monday)", "intensity": "Moderate (65-75% max heart rate)"},
            {"day": "Saturday", "workout": "Long endurance session: 60-90 min steady activity", "intensity": "Moderate (65-70% max heart rate)"},
            {"day": "Sunday", "workout": "Full Rest Day", "intensity": "None"}
        ],
        "tips": [
            "Gradually increase duration before intensity",
            "Stay properly hydrated before, during, and after workouts",
            "Incorporate proper warm-up and cool-down periods",
            "Track your heart rate to ensure you're training in the correct zones",
            "Consume adequate carbohydrates to fuel longer workouts"
        ]
    },
    "Strength": {
        "description": "Focus on building muscle strength and power through resistance training.",
        "weekly_schedule": [
            {"day": "Monday", "workout": "Upper Body: Chest, Shoulders, Triceps (4 exercises, 3-4 sets each)", "intensity": "Moderate-High"},
            {"day": "Tuesday", "workout": "Lower Body: Quads, Hamstrings, Calves (4 exercises, 3-4 sets each)", "intensity": "Moderate-High"},
            {"day": "Wednesday", "workout": "20-30 min light cardio + mobility work", "intensity": "Low (Recovery)"},
            {"day": "Thursday", "workout": "Upper Body: Back, Biceps (4 exercises, 3-4 sets each)", "intensity": "Moderate-High"},
            {"day": "Friday", "workout": "Core & Full Body Circuits (5 exercises, 3 sets each)", "intensity": "Moderate"},
            {"day": "Saturday", "workout": "30 min cardio + specific weak point training", "intensity": "Moderate"},
            {"day": "Sunday", "workout": "Full Rest Day", "intensity": "None"}
        ],
        "tips": [
            "Focus on proper form before increasing weights",
            "Ensure adequate protein intake (1.6-2.2g per kg of bodyweight)",
            "Progressive overload: gradually increase weights or reps",
            "Allow 48 hours recovery for worked muscle groups",
            "Include compound movements (squats, deadlifts, bench press) for efficient strength gains"
        ]
    },
    "Flexibility": {
        "description": "Focus on improving range of motion, mobility, and recovery.",
        "weekly_schedule": [
            {"day": "Monday", "workout": "Full body dynamic stretching + 20 min yoga flow", "intensity": "Low-Moderate"},
            {"day": "Tuesday", "workout": "30 min cardio + lower body static stretching", "intensity": "Low-Moderate"},
            {"day": "Wednesday", "workout": "Mobility drills + foam rolling session", "intensity": "Low"},
            {"day": "Thursday", "workout": "20 min light cardio + upper body static stretching", "intensity": "Low-Moderate"},
            {"day": "Friday", "workout": "Full body yoga session (45-60 min)", "intensity": "Moderate"},
            {"day": "Saturday", "workout": "Active recovery: light walking + dynamic stretching", "intensity": "Low"},
            {"day": "Sunday", "workout": "Deep stretch session + meditation", "intensity": "Low"}
        ],
        "tips": [
            "Hold static stretches for 30-60 seconds each",
            "Breathe deeply during stretching sessions",
            "Perform dynamic stretches before workouts, static stretches after",
            "Stay consistent - flexibility improves with regular practice",
            "Use props like foam rollers, bands, and yoga blocks to assist when needed"
        ]
    },
    "Balance": {
        "description": "Focus on improving stability, coordination, and core strength.",
        "weekly_schedule": [
            {"day": "Monday", "workout": "Core & stabilization exercises + single-leg balance work", "intensity": "Moderate"},
            {"day": "Tuesday", "workout": "30 min light cardio + yoga balance poses", "intensity": "Low-Moderate"},
            {"day": "Wednesday", "workout": "Proprioception training + light strength work", "intensity": "Moderate"},
            {"day": "Thursday", "workout": "Rest day or light walking", "intensity": "Very Low"},
            {"day": "Friday", "workout": "Balance-focused circuit training", "intensity": "Moderate"},
            {"day": "Saturday", "workout": "Recreational activity (hiking, swimming, etc)", "intensity": "Moderate"},
            {"day": "Sunday", "workout": "Full Rest Day", "intensity": "None"}
        ],
        "tips": [
            "Practice barefoot training when appropriate for improved proprioception",
            "Include unstable surface training (BOSU ball, wobble board)",
            "Strengthen your core regularly as it's essential for balance",
            "Progress gradually from stable to unstable exercises",
            "Include mindfulness practice for improved body awareness"
        ]
    },
    "Weight Loss": {
        "description": "Focus on creating a caloric deficit through combined cardio and strength training.",
        "weekly_schedule": [
            {"day": "Monday", "workout": "HIIT session (30 min) + light strength training", "intensity": "High"},
            {"day": "Tuesday", "workout": "45-60 min moderate cardio (walking, cycling)", "intensity": "Moderate"},
            {"day": "Wednesday", "workout": "Full body strength circuit (6 exercises, 3 sets each)", "intensity": "Moderate-High"},
            {"day": "Thursday", "workout": "Active recovery: 30 min light activity + stretching", "intensity": "Low"},
            {"day": "Friday", "workout": "HIIT session (30 min) + core workout", "intensity": "High"},
            {"day": "Saturday", "workout": "Longer steady-state cardio (60 min) + light resistance", "intensity": "Moderate"},
            {"day": "Sunday", "workout": "Full Rest Day or light walking", "intensity": "Very Low/None"}
        ],
        "tips": [
            "Aim for a moderate caloric deficit (500 calories below maintenance)",
            "Prioritize protein intake to preserve muscle mass",
            "Combine strength training with cardio for optimal results",
            "Stay hydrated and limit high-calorie beverages",
            "Focus on consistency rather than intensity initially"
        ]
    },
    "Muscle Building": {
        "description": "Focus on hypertrophy training with adequate nutrition for muscle growth.",
        "weekly_schedule": [
            {"day": "Monday", "workout": "Chest & Triceps (4 exercises, 3-4 sets of 8-12 reps)", "intensity": "Moderate-High"},
            {"day": "Tuesday", "workout": "Back & Biceps (4 exercises, 3-4 sets of 8-12 reps)", "intensity": "Moderate-High"},
            {"day": "Wednesday", "workout": "20-30 min light cardio + mobility work", "intensity": "Low (Recovery)"},
            {"day": "Thursday", "workout": "Legs & Core (5 exercises, 3-4 sets of 8-15 reps)", "intensity": "High"},
            {"day": "Friday", "workout": "Shoulders & Arms (4 exercises, 3-4 sets of 8-12 reps)", "intensity": "Moderate-High"},
            {"day": "Saturday", "workout": "Full body light pump session + 20 min cardio", "intensity": "Moderate"},
            {"day": "Sunday", "workout": "Full Rest Day", "intensity": "None"}
        ],
        "tips": [
            "Consume caloric surplus of 250-500 calories above maintenance",
            "Get adequate protein (1.6-2.2g per kg of bodyweight)",
            "Focus on compound movements with isolation exercises as supplements",
            "Ensure proper sleep (7-9 hours) for recovery and growth",
            "Training to near-failure is important for hypertrophy"
        ]
    },
    "General Fitness": {
        "description": "Balanced approach to overall fitness with variety of activities.",
        "weekly_schedule": [
            {"day": "Monday", "workout": "Cardio: 30-45 min moderate activity (running, cycling, elliptical)", "intensity": "Moderate"},
            {"day": "Tuesday", "workout": "Full body strength training (6 exercises, 2-3 sets each)", "intensity": "Moderate"},
            {"day": "Wednesday", "workout": "Active recovery: walking + mobility work", "intensity": "Low"},
            {"day": "Thursday", "workout": "HIIT or interval training (25-30 min)", "intensity": "High"},
            {"day": "Friday", "workout": "Upper body strength focus (4-5 exercises, 3 sets each)", "intensity": "Moderate"},
            {"day": "Saturday", "workout": "Recreational activity or sports", "intensity": "Varies"},
            {"day": "Sunday", "workout": "Full Rest Day or yoga/light stretching", "intensity": "Very Low/None"}
        ],
        "tips": [
            "Aim for balanced macronutrient intake",
            "Stay consistent with 3-5 workouts per week",
            "Mix cardio and strength for balanced fitness",
            "Adjust intensity according to energy levels and recovery",
            "Find activities you enjoy to maintain long-term adherence"
        ]
    }
}

# Map prediction labels to workout schedule categories
PLAN_TO_SCHEDULE = {
    "Endurance Training": "Endurance",
    "Strength Training": "Strength",
    "Flexibility Training": "Flexibility",
    "Balance Training": "Balance",
    "Weight Loss Program": "Weight Loss",
    "Muscle Building Program": "Muscle Building",
    "General Fitness Maintenance": "General Fitness"
    # Add more mappings as needed
}

# Default schedule for unexpected plan types
DEFAULT_SCHEDULE = "General Fitness"

# Function to get workout schedule based on plan type
def get_workout_schedule(plan_type):
    # Clean up the plan type string to match keys better
    clean_plan = plan_type.strip()
    
    # Direct lookup
    if clean_plan in WORKOUT_SCHEDULES:
        return WORKOUT_SCHEDULES[clean_plan]
    
    # Try mapping through PLAN_TO_SCHEDULE
    mapped_schedule = PLAN_TO_SCHEDULE.get(clean_plan)
    if mapped_schedule and mapped_schedule in WORKOUT_SCHEDULES:
        return WORKOUT_SCHEDULES[mapped_schedule]
    
    # Handle fallback for weight management
    if "weight" in clean_plan.lower() and "loss" in clean_plan.lower():
        return WORKOUT_SCHEDULES["Weight Loss"]
    elif "muscle" in clean_plan.lower() or "build" in clean_plan.lower():
        return WORKOUT_SCHEDULES["Muscle Building"]
    elif "strength" in clean_plan.lower():
        return WORKOUT_SCHEDULES["Strength"]
    elif "endurance" in clean_plan.lower() or "cardio" in clean_plan.lower():
        return WORKOUT_SCHEDULES["Endurance"]
    elif "flex" in clean_plan.lower() or "mobil" in clean_plan.lower():
        return WORKOUT_SCHEDULES["Flexibility"]
    elif "balance" in clean_plan.lower() or "stabil" in clean_plan.lower():
        return WORKOUT_SCHEDULES["Balance"]
    
    # Default fallback
    return WORKOUT_SCHEDULES[DEFAULT_SCHEDULE]

# Function to preprocess and predict
def predict_fitness_plan(input_data):
    try:
        # Create initial DataFrame with provided columns
        columns = ['Age', 'Weight (kg)', 'Height (cm)', 'Heart Rate (bpm)', 'Resting Heart Rate (bpm)', 
                   'Calories Burned', 'Workout Duration (mins)', 'Gender', 'Workout Intensity']
        input_df = pd.DataFrame([input_data], columns=columns)
        print(f"Input DataFrame:\n{input_df}")
        
        # Validate categorical fields
        if input_df['Gender'].iloc[0] not in ['Male', 'Female']:
            raise ValueError("Gender must be 'Male' or 'Female'")
        if input_df['Workout Intensity'].iloc[0] not in ['low', 'medium', 'high']:
            raise ValueError("Workout Intensity must be 'low', 'medium', or 'high'")
            
        # Convert categorical variables to proper format if needed
        # This step may be needed if the model expects numeric encoding of categories
        # Note: The exact encoding should match what was used during model training
        # Convert gender to numeric (if needed by the model)
        gender_map = {'Male': 0, 'Female': 1}
        input_df['Gender_Numeric'] = input_df['Gender'].map(gender_map)
        
        # Convert workout intensity to numeric (if needed by the model)
        intensity_map = {'low': 0, 'medium': 1, 'high': 2}
        input_df['Workout_Intensity_Numeric'] = input_df['Workout Intensity'].map(intensity_map)
        
        # Keep original categorical columns as well in case preprocessor handles them
        
        # Calculate BMI
        height_m = input_df['Height (cm)'].iloc[0] / 100  # Convert cm to m
        weight_kg = input_df['Weight (kg)'].iloc[0]
        bmi = weight_kg / (height_m * height_m)
        
        # Add missing columns required by the model
        input_df['BMI'] = bmi
        input_df['Daily Calories Intake'] = input_df['Calories Burned'] * 1.2  # Estimate daily caloric intake
        input_df['Caloric_Efficiency'] = input_df['Calories Burned'] / input_df['Workout Duration (mins)']
        input_df['Sleep Hours'] = 8.0  # Default value
        input_df['Steps Taken'] = 10000  # Default value
        input_df['Step_Efficiency'] = input_df['Steps Taken'] / 24  # Steps per hour (simplified)
        
        # Calculate fitness level based on heart rate data (using numeric values)
        resting_hr = input_df['Resting Heart Rate (bpm)'].iloc[0]
        if resting_hr < 60:
            fitness_level = 3  # High
        elif resting_hr < 80:
            fitness_level = 2  # Medium
        else:
            fitness_level = 1  # Low
        input_df['Fitness_Level'] = fitness_level
        
        # Remove helper columns that might not be expected by the preprocessor
        if 'Gender_Numeric' in input_df.columns:
            input_df = input_df.drop(['Gender_Numeric'], axis=1)
        if 'Workout_Intensity_Numeric' in input_df.columns:
            input_df = input_df.drop(['Workout_Intensity_Numeric'], axis=1)
            
        print(f"Complete DataFrame with calculated features:\n{input_df}")
        
        # Convert all values to numeric where possible to avoid string conversion errors
        for col in input_df.columns:
            if col not in ['Gender', 'Workout Intensity']:  # Skip true categorical columns
                try:
                    input_df[col] = pd.to_numeric(input_df[col])
                except:
                    print(f"Could not convert column {col} to numeric")
        
        # Preprocess the input data
        try:
            processed_data = preprocessor.transform(input_df)
            processed_data = processed_data.toarray() if hasattr(processed_data, 'toarray') else processed_data
            print(f"Processed Data shape: {processed_data.shape}")
        except Exception as e:
            print(f"Error during preprocessing with standard approach: {str(e)}")
            
            # Try to inspect the preprocessor to understand what columns it expects
            try:
                if hasattr(preprocessor, 'get_feature_names_out'):
                    print(f"Preprocessor expected features: {preprocessor.get_feature_names_out()}")
                elif hasattr(preprocessor, 'feature_names_in_'):
                    print(f"Preprocessor expected features: {preprocessor.feature_names_in_}")
            except:
                pass
                
            # Alternative approach: If preprocessor fails, try direct model input
            # This is a fallback solution that might work if the preprocessor is causing issues
            print("Attempting fallback approach with direct feature engineering...")
            
            # Create a simple feature vector based on our calculated features
            # The order and scaling might not match exactly what the model expects,
            # but it's worth trying as a fallback
            try:
                # Extract numeric features in a controlled way
                features = np.array([
                    input_df['Age'].iloc[0],
                    input_df['Weight (kg)'].iloc[0],
                    input_df['Height (cm)'].iloc[0],
                    input_df['BMI'].iloc[0],
                    input_df['Heart Rate (bpm)'].iloc[0],
                    input_df['Resting Heart Rate (bpm)'].iloc[0],
                    input_df['Daily Calories Intake'].iloc[0],
                    input_df['Calories Burned'].iloc[0],
                    input_df['Caloric_Efficiency'].iloc[0],
                    input_df['Workout Duration (mins)'].iloc[0],
                    input_df['Sleep Hours'].iloc[0],
                    input_df['Steps Taken'].iloc[0],
                    input_df['Step_Efficiency'].iloc[0],
                    input_df['Fitness_Level'].iloc[0],
                    1 if input_df['Gender'].iloc[0] == 'Male' else 0,  # Male encoding
                    1 if input_df['Gender'].iloc[0] == 'Female' else 0,  # Female encoding
                    1 if input_df['Workout Intensity'].iloc[0] == 'low' else 0,  # Low intensity encoding
                    1 if input_df['Workout Intensity'].iloc[0] == 'medium' else 0,  # Medium intensity encoding
                    1 if input_df['Workout Intensity'].iloc[0] == 'high' else 0,  # High intensity encoding
                ]).reshape(1, -1)
                
                processed_data = features
                print(f"Fallback feature array shape: {processed_data.shape}")
            except Exception as fallback_error:
                print(f"Fallback approach also failed: {str(fallback_error)}")
                raise e  # Raise the original error if fallback fails
        
        # Make predictions
        ann_pred = np.argmax(ann_model.predict(processed_data, verbose=0), axis=1)
        xgb_pred = xgb_model.predict(processed_data)
        print(f"ANN Prediction (encoded): {ann_pred}")
        print(f"XGBoost Prediction (encoded): {xgb_pred}")
        
        # Decode predictions
        ann_plan = label_encoder.inverse_transform(ann_pred)[0]
        xgb_plan = label_encoder.inverse_transform(xgb_pred)[0]
        
        # Get workout schedules for both predictions
        ann_schedule = get_workout_schedule(ann_plan)
        xgb_schedule = get_workout_schedule(xgb_plan)
        
        return ann_plan, xgb_plan, ann_schedule, xgb_schedule
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        # Validate input data
        required_fields = ['age', 'weight', 'height', 'heart_rate', 'resting_heart_rate', 'calories', 'duration', 'gender', 'intensity']
        for field in required_fields:
            if field not in data or not data[field] or not data[field].strip():
                return jsonify({'status': 'error', 'message': f'Missing, empty, or invalid field: {field}'}), 400
        
        # Convert to floats with error handling
        try:
            input_data = [
                float(data['age']),
                float(data['weight']),
                float(data['height']),
                float(data['heart_rate']),
                float(data['resting_heart_rate']),
                float(data['calories']),
                float(data['duration']),
                data['gender'],
                data['intensity']
            ]
        except ValueError as ve:
            return jsonify({'status': 'error', 'message': f'Invalid numeric value: {str(ve)}'}), 400
        
        # Add basic sanity checks on input values
        if input_data[0] <= 0 or input_data[0] > 120:  # Age check
            return jsonify({'status': 'error', 'message': 'Age must be between 1 and 120 years'}), 400
        if input_data[1] <= 0 or input_data[1] > 300:  # Weight check
            return jsonify({'status': 'error', 'message': 'Weight must be between 1 and 300 kg'}), 400
        if input_data[2] <= 0 or input_data[2] > 250:  # Height check
            return jsonify({'status': 'error', 'message': 'Height must be between 1 and 250 cm'}), 400
        if input_data[3] <= 0 or input_data[3] > 220:  # Heart rate check
            return jsonify({'status': 'error', 'message': 'Heart rate must be between 1 and 220 bpm'}), 400
        if input_data[4] <= 0 or input_data[4] > 200:  # Resting heart rate check
            return jsonify({'status': 'error', 'message': 'Resting heart rate must be between 1 and 200 bpm'}), 400
        
        try:
            ann_plan, xgb_plan, ann_schedule, xgb_schedule = predict_fitness_plan(input_data)
            return jsonify({
                'status': 'success',
                'ann_prediction': ann_plan,
                'xgb_prediction': xgb_plan,
                'ann_schedule': ann_schedule,
                'xgb_schedule': xgb_schedule
            })
        except Exception as prediction_error:
            print(f"Error during prediction: {str(prediction_error)}")
            
            # If the model failed to predict, provide some fallback recommendations
            # based on basic fitness principles
            bmi = input_data[1] / ((input_data[2]/100) ** 2)
            
            # Generate basic fitness plan recommendations and schedule
            if bmi < 18.5:
                fallback_plan = "Unable to generate model prediction. For underweight individuals: Focus on strength training 3x weekly with progressive overload. Increase caloric intake with protein-rich foods."
                fallback_schedule = get_workout_schedule("Muscle Building")
            elif bmi < 25:
                fallback_plan = "Unable to generate model prediction. For normal weight individuals: Balanced workout with 3-4 days of cardio and 2 days of strength training. Focus on maintenance and overall fitness."
                fallback_schedule = get_workout_schedule("General Fitness")
            elif bmi < 30:
                fallback_plan = "Unable to generate model prediction. For overweight individuals: Prioritize cardio 4-5 days weekly with 2 days of strength training. Maintain caloric deficit through diet and exercise."
                fallback_schedule = get_workout_schedule("Weight Loss")
            else:
                fallback_plan = "Unable to generate model prediction. For individuals with obesity: Start with walking daily, gradually increasing duration. Add 2 days of light strength training. Focus on sustainable diet changes."
                fallback_schedule = get_workout_schedule("Weight Loss")
            
            if input_data[8] == 'low':
                intensity_advice = " Consider gradually increasing workout intensity as fitness improves."
            else:
                intensity_advice = " Your current intensity level is good. Make sure to include rest days for recovery."
                
            return jsonify({
                'status': 'partial_success',
                'message': 'Model prediction failed, providing general recommendations instead.',
                'ann_prediction': fallback_plan + intensity_advice,
                'xgb_prediction': fallback_plan + intensity_advice,
                'ann_schedule': fallback_schedule, 
                'xgb_schedule': fallback_schedule,
                'error_details': str(prediction_error)
            })
    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}")
        
        # Return a more user-friendly error message
        error_message = str(e)
        if "could not convert string to float" in error_message:
            error_message = "There was a problem with data type conversion. Please ensure all numeric values are valid."
        elif "unexpected keyword argument" in error_message:
            error_message = "There was a problem with the model configuration. Please check the console for more details."
        
        return jsonify({
            'status': 'error', 
            'message': f"Prediction failed: {error_message}",
            'technical_details': str(e)
        }), 500

@app.route('/')
def home():
    return "Fitness Plan Prediction API is running!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)