import os
import logging
from flask import Flask, request, render_template
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import StringIndexerModel
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set environment variables for PySpark
os.environ["PYSPARK_PYTHON"] = r".venv\Scripts\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r".venv\Scripts\python.exe"

# Define paths dynamically
MODEL_PATH = os.path.join("models", "trained_model_pipeline")
LABEL_INDEXER_MODEL_PATH = os.path.join("models", "label_indexer_model")

# Initialize Spark session
print("Initializing Spark session...")
try:
    spark = SparkSession.builder.appName("MusicGenreClassifier").getOrCreate()
    print("Spark session initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Spark session: {e}")
    raise RuntimeError("Spark session initialization failed. Check logs for details.")

# Load the trained model pipeline
try:
    model = PipelineModel.load(MODEL_PATH)
    print("Trained model pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load trained model pipeline: {e}")
    raise RuntimeError("Failed to load trained model pipeline. Check logs for details.")

# Load the label indexer model (used to map indices back to genres)
try:
    label_indexer_model = StringIndexerModel.load(LABEL_INDEXER_MODEL_PATH)
    print("Label indexer model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load label indexer model: {e}")
    raise RuntimeError("Failed to load label indexer model. Check logs for details.")

# Function to apply softmax for normalization
def softmax(logits):
    """
    Applies softmax to convert logits into a valid probability distribution.
    """
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return list(exp_logits / np.sum(exp_logits))

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get lyrics from the form
            lyrics = request.form.get("lyrics")
            if not lyrics:
                return render_template("error.html", error="Lyrics cannot be empty.")

            # Create a DataFrame with the input lyrics
            input_df = spark.createDataFrame([(lyrics,)], ["lyrics"])

            # Make predictions using the trained model
            prediction = model.transform(input_df)

            # Extract predicted genre
            result = prediction.select("predicted_genre").collect()[0][0]

            # Extract raw prediction (logits) and apply softmax
            raw_prediction_col = prediction.select("rawPrediction").collect()[0][0]
            probabilities = softmax(raw_prediction_col)

            # Get the list of genres
            genres = label_indexer_model.labels

            # Validate probabilities
            if not all(0 <= p <= 1 for p in probabilities):
                raise ValueError("Probabilities are not in the range [0, 1].")
            if not np.isclose(sum(probabilities), 1.0):
                raise ValueError("Probabilities do not sum to 1.")

            # Render the result page with the prediction and visualization data
            return render_template(
                "result.html",
                result=result,
                genres=genres,
                probabilities=probabilities
            )
        except Exception as e:
            # Log the error and render an error page
            logging.error(f"Error during prediction: {e}")
            return render_template("error.html", error=f"An unexpected error occurred: {str(e)}")
    # Render the input form for GET requests
    return render_template("index.html")

if __name__ == "__main__":
    # Run the Flask app without the reloader to avoid creating multiple Spark sessions
    app.run(debug=True, use_reloader=False)