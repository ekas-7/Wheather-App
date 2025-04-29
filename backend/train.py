import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt # Not needed for saving
# import seaborn as sns # Not needed for saving
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Optional for evaluation during training
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
# from datetime import datetime, timedelta # Not needed directly for training saving
import joblib # To save the scaler

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class WeatherTrainingSystem: # Renamed for clarity
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler() # Scaler for all features + target
        self.target_scaler = MinMaxScaler() # Separate scaler JUST for the target column if needed for simpler inverse transform (optional, see notes)
        self.feature_cols = None
        self.target_col = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.history = None
        self.forecast_days = 7  # Default forecast for 7 days
        self.sequence_length = 30  # Default lookback period
        self.last_sequence_for_prediction = None # To save for API use

    def load_data(self):
        """Load weather data from file"""
        if self.data_path:
            print(f"Loading data from {self.data_path}")
            try:
                self.data = pd.read_csv(self.data_path)
                # Handle date column
                if 'date' in self.data.columns or 'Date' in self.data.columns:
                    date_col = 'date' if 'date' in self.data.columns else 'Date'
                    self.data[date_col] = pd.to_datetime(self.data[date_col])
                    self.data.set_index(date_col, inplace=True)
                print(f"Data loaded successfully with shape: {self.data.shape}")
                return True
            except FileNotFoundError:
                 print(f"Error: Data file not found at {self.data_path}")
                 return False
            except Exception as e:
                 print(f"Error loading data: {e}")
                 return False
        else:
            print("No data path provided.")
            return False

    # --- explore_data and _visualize_data can remain as they are for analysis ---
    # --- but are not strictly necessary for the training script's main goal ---
    # --- (Removed for brevity in this example, but you can keep them) ---

    def preprocess_data(self, target_col='temp', test_size=0.2, dropna=True):
        """Preprocess the data for model training"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False

        print(f"\n===== Preprocessing Data with target: {target_col} =====")
        self.target_col = target_col
        df = self.data.copy()

        # Handle missing values (simplified, adjust as needed)
        if dropna:
            df.dropna(inplace=True)
            print(f"Dropped rows with missing values. New shape: {df.shape}")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                df[col].fillna(df[col].mode()[0], inplace=True)
            print("Filled missing values.")

        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")
            return False

        self.feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in self.feature_cols:
            self.feature_cols.remove(target_col)
        else:
             print(f"Target column '{target_col}' must be numeric.")
             return False

        print(f"Using {len(self.feature_cols)} features: {self.feature_cols}")
        cols_to_scale = self.feature_cols + [self.target_col]

        # Scale the data
        scaled_data = self.scaler.fit_transform(df[cols_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=cols_to_scale, index=df.index)

        # --- Optional: Fit separate scaler for target ---
        # self.target_scaler.fit(df[[self.target_col]])
        # --- End Optional ---

        # Create sequences
        X, y = self._create_sequences(scaled_df, self.feature_cols, self.target_col)

        if len(X) == 0:
            print("Not enough data to create sequences with current settings.")
            return False

        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False # Keep shuffle=False for time series
        )

        # Store the last sequence from the test set for future predictions
        if len(self.X_test) > 0:
             self.last_sequence_for_prediction = self.X_test[-1]
        elif len(self.X_train) > 0:
             print("Warning: No test data generated, using last training sequence for prediction.")
             self.last_sequence_for_prediction = self.X_train[-1]
        else:
             print("Error: No sequences available to save for prediction.")
             return False


        print(f"Training set shape: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Testing set shape: X={self.X_test.shape}, y={self.y_test.shape}")
        print(f"Last sequence shape for prediction: {self.last_sequence_for_prediction.shape}")

        return True

    def _create_sequences(self, data, feature_cols, target_col):
        """Create sequences for time series forecasting"""
        X, y = [], []
        if len(data) < self.sequence_length + self.forecast_days:
            print(f"Warning: Data length ({len(data)}) is less than sequence_length ({self.sequence_length}) + forecast_days ({self.forecast_days}). Cannot create sequences.")
            return np.array(X), np.array(y)

        for i in range(self.sequence_length, len(data) - self.forecast_days + 1):
            # Input sequence (lookback period) for FEATURES ONLY
            X.append(data[feature_cols].iloc[i-self.sequence_length:i].values)
            # Output sequence (forecast period) for TARGET ONLY
            y.append(data[target_col].iloc[i:i+self.forecast_days].values)

        return np.array(X), np.array(y)

    def build_model(self, model_type='lstm'):
        """Build the deep learning model architecture"""
        print(f"\n===== Building {model_type.upper()} Model =====")
        if self.X_train is None:
            print("No training data available. Please preprocess data first.")
            return False

        n_features = self.X_train.shape[2] # Number of input features
        seq_length = self.X_train.shape[1]
        forecast_length = self.y_train.shape[1] # Output dimension (number of days to predict)

        self.model = Sequential()
        input_shape = (seq_length, n_features)

        if model_type.lower() == 'lstm':
            self.model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=input_shape))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(80, activation='relu', return_sequences=False))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(forecast_length)) # Output layer predicts 'forecast_length' steps

        elif model_type.lower() == 'gru':
            self.model.add(GRU(100, activation='relu', return_sequences=True, input_shape=input_shape))
            self.model.add(Dropout(0.2))
            self.model.add(GRU(80, activation='relu', return_sequences=False))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(forecast_length))

        elif model_type.lower() == 'bilstm':
            self.model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True), input_shape=input_shape))
            self.model.add(Dropout(0.2))
            self.model.add(Bidirectional(LSTM(80, activation='relu', return_sequences=False)))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(forecast_length))
        else:
            print(f"Error: Unknown model type '{model_type}'")
            return False

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        print(self.model.summary())
        return True

    def train_model(self, epochs=100, batch_size=32, patience=20, model_save_path='trained_model.h5'):
        """Train the deep learning model"""
        if self.model is None or self.X_train is None or self.y_train is None:
            print("Model not built or data not preprocessed.")
            return False

        print(f"\n===== Training Model for up to {epochs} epochs =====")

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience // 2, min_lr=0.00001, verbose=1),
            ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1) # Save the best model directly
        ]

        # Use validation split if no separate validation set is created
        validation_data = None
        if self.X_test is not None and len(self.X_test) > 0:
             validation_data=(self.X_test, self.y_test)
             validation_split = 0 # Use explicit test set if available
        else:
             validation_split = 0.2 # Use part of training data if no test set
             print("Warning: No test set available for validation, using validation_split=0.2")


        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        # Load the best model saved by ModelCheckpoint
        if os.path.exists(model_save_path):
             print(f"Loading best model from {model_save_path}")
             self.model = load_model(model_save_path)
        else:
             print("Warning: Best model file not found. Using the model state at the end of training.")

        # --- Plotting training history can be done here if needed ---
        # self._plot_training_history()

        return True

    def save_components(self, model_path='weather_model.h5', scaler_path='weather_scaler.pkl', sequence_path='last_sequence.npy'):
        """Save the necessary components for the prediction API"""
        print("\n===== Saving Components =====")
        saved = True
        # Save Model
        if self.model:
            try:
                self.model.save(model_path)
                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
                saved = False
        else:
            print("No model to save.")
            saved = False

        # Save Scaler
        if self.scaler:
            try:
                joblib.dump(self.scaler, scaler_path)
                print(f"Scaler saved to {scaler_path}")
            except Exception as e:
                print(f"Error saving scaler: {e}")
                saved = False
        else:
             print("No scaler to save.")
             saved = False

        # Save Last Sequence
        if self.last_sequence_for_prediction is not None:
             try:
                np.save(sequence_path, self.last_sequence_for_prediction)
                print(f"Last sequence saved to {sequence_path}")
             except Exception as e:
                print(f"Error saving last sequence: {e}")
                saved = False
        else:
             print("No last sequence to save.")
             saved = False

        # --- Optional: Save target scaler ---
        # if self.target_scaler:
        #     joblib.dump(self.target_scaler, 'target_scaler.pkl')
        #     print("Target scaler saved.")
        # --- End Optional ---

        # Save feature and target column names (important for API consistency)
        try:
            config = {
                'feature_cols': self.feature_cols,
                'target_col': self.target_col,
                'sequence_length': self.sequence_length,
                'forecast_days': self.forecast_days,
                'scaled_columns_order': self.feature_cols + [self.target_col] # Store the order used for scaling
            }
            joblib.dump(config, 'training_config.pkl')
            print("Training configuration saved to training_config.pkl")
        except Exception as e:
             print(f"Error saving training configuration: {e}")
             saved = False


        return saved


# --- Main execution block for training ---
def main_training():
    DATA_FILE = 'weather_data.csv' # <--- Make sure this file exists
    MODEL_TYPE = 'lstm'            # 'lstm', 'gru', or 'bilstm'
    TARGET_COLUMN = 'temp'         # Column to predict
    SEQUENCE_LENGTH = 30
    FORECAST_DAYS = 7
    EPOCHS = 50 # Adjust as needed
    BATCH_SIZE = 32
    PATIENCE = 10

    # --- Define save paths ---
    MODEL_SAVE_PATH = f'weather_model_{MODEL_TYPE}.h5'
    SCALER_SAVE_PATH = f'weather_scaler_{TARGET_COLUMN}.pkl'
    SEQUENCE_SAVE_PATH = 'last_sequence.npy'
    CONFIG_SAVE_PATH = 'training_config.pkl'

    trainer = WeatherTrainingSystem(DATA_FILE)
    trainer.sequence_length = SEQUENCE_LENGTH
    trainer.forecast_days = FORECAST_DAYS

    if not trainer.load_data():
        return

    # Optional: Explore data
    # trainer.explore_data()

    if not trainer.preprocess_data(target_col=TARGET_COLUMN):
         return

    if not trainer.build_model(model_type=MODEL_TYPE):
         return

    if not trainer.train_model(epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE, model_save_path=MODEL_SAVE_PATH):
         return

    # Optional: Evaluate model here if needed
    # trainer.evaluate_model()

    if not trainer.save_components(model_path=MODEL_SAVE_PATH, scaler_path=SCALER_SAVE_PATH, sequence_path=SEQUENCE_SAVE_PATH):
         print("Component saving failed.")
    else:
         print("Training and saving complete.")


if __name__ == "__main__":
    # Make sure you have a 'weather_data.csv' file in the same directory
    # Example: Create a dummy CSV if you don't have one
    if not os.path.exists('weather_data.csv'):
        print("Creating dummy weather_data.csv for demonstration.")
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        dummy_data = pd.DataFrame({
            'date': dates,
            'temp': np.random.rand(500) * 30 + 5, # Random temp between 5 and 35
            'humidity': np.random.rand(500) * 50 + 50, # Random humidity between 50 and 100
            'wind_speed': np.random.rand(500) * 15 # Random wind speed between 0 and 15
        })
        dummy_data.to_csv('weather_data.csv', index=False)

    main_training()