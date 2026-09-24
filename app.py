from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

app = Flask(__name__)

# Load model
model = MobileNetV2(weights="imagenet")

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Calorie Database (basic)
calorie_db = {
    "pizza": {"calories": 266, "protein": 11, "fat": 10, "carbs": 33},
    "burger": {"calories": 295, "protein": 17, "fat": 12, "carbs": 30},
    "noodles": {"calories": 138, "protein": 5, "fat": 2, "carbs": 25},
    "pasta": {"calories": 131, "protein": 5, "fat": 1, "carbs": 25},
    "rice": {"calories": 130, "protein": 2, "fat": 0.3, "carbs": 28},
    "salad": {"calories": 33, "protein": 2, "fat": 0.2, "carbs": 6},
    "cake": {"calories": 257, "protein": 3, "fat": 9, "carbs": 38},
    "ice cream": {"calories": 207, "protein": 3.5, "fat": 11, "carbs": 24},
    "banana": {"calories": 89, "protein": 1, "fat": 0.3, "carbs": 23},
    "apple": {"calories": 52, "protein": 0.3, "fat": 0.2, "carbs": 14}
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Load image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # 🔥 Prediction Logic
            preds = model.predict(img_array)
            decoded = decode_predictions(preds, top=5)[0]

            food = None
            confidence = 0

            # Food keywords
            food_keywords = [
                "pizza","burger","sandwich","hotdog","pasta","spaghetti","carbonara",
                "noodle","ramen","rice","fried_rice","biryani",
                "salad","omelette","egg","soup",
                "cake","ice_cream","chocolate","donut",
                "apple","banana","orange","grape","watermelon",
                "coffee","tea","juice"
            ]

            # Mapping
            food_map = {
                "carbonara": "pasta",
                "spaghetti": "pasta",
                "ramen": "noodles",
                "chow_mein": "noodles",
                "fried_rice": "rice",
                "ice_cream": "ice cream",
                "hotdog": "hot dog"
            }

            # Step 1: Find food label
            for d in decoded:
                label = d[1].lower()

                if any(k in label for k in food_keywords):
                    food = label
                    confidence = round(d[2] * 100, 2)
                    break

            # Step 2: fallback
            if not food:
                food = decoded[0][1].lower()
                confidence = round(decoded[0][2] * 100, 2)

            # Step 3: clean name
            food = food.replace("_", " ")

            # Step 4: mapping fix
            for key in food_map:
                if key in food:
                    food = food_map[key]
                    break

            # Step 5: get nutrition
            nutrition = calorie_db.get(food, {
                "calories": "Not Available",
                "protein": "Not Available",
                "fat": "Not Available",
                "carbs": "Not Available"
            })

            return render_template("result.html",
                                   image=filepath,
                                   food=food,
                                   confidence=confidence,
                                   nutrition=nutrition)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)