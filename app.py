from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ── Load model and encoders ───────────────────────────────────────
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)

model     = data['model']
le_region = data['le_region']
le_soil   = data['le_soil']
le_crop   = data['le_crop']

# ── Price per ton (₹) for each crop ──────────────────────────────
price_per_ton = {
    'Rice':   20000,
    'Cotton': 60000,
    'Wheat':  18000,
    'Maize':  15000
}

# ── Home page ─────────────────────────────────────────────────────
@app.route('/')
def index():
    crops   = list(le_crop.classes_)
    soils   = list(le_soil.classes_)
    regions = list(le_region.classes_)
    return render_template('index.html', crops=crops, soils=soils, regions=regions)

# ── Prediction ────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        region      = request.form['region']
        soil        = request.form['soil']
        crop        = request.form['crop']
        rainfall    = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        fertilizer  = float(request.form['fertilizer'])
        area        = float(request.form['area'])

        region_enc = le_region.transform([region])[0]
        soil_enc   = le_soil.transform([soil])[0]
        crop_enc   = le_crop.transform([crop])[0]

        features = np.array([[region_enc, soil_enc, crop_enc,
                               rainfall, temperature, fertilizer]])

        yield_per_hectare = model.predict(features)[0]
        total_yield = round(yield_per_hectare * area, 2)
        price       = price_per_ton.get(crop, 20000)
        income      = round(total_yield * price, 2)

        # ── Crop Comparison: predict yield for ALL crops ──────────
        all_crops = list(le_crop.classes_)
        comparison = []
        best_crop = crop
        best_income = income

        for c in all_crops:
            c_enc = le_crop.transform([c])[0]
            f = np.array([[region_enc, soil_enc, c_enc, rainfall, temperature, fertilizer]])
            y = model.predict(f)[0]
            t = round(y * area, 2)
            p = price_per_ton.get(c, 20000)
            inc = round(t * p, 2)
            comparison.append({
                'crop': c,
                'yield': round(y, 2),
                'income': inc,
                'income_fmt': f"{inc:,.0f}"
            })
            if inc > best_income:
                best_income = inc
                best_crop = c

        # ── Crop Recommendation ───────────────────────────────────
        recommended = best_crop
        if recommended == crop:
            rec_msg = f"✅ Great choice! {crop} is the most profitable crop for your conditions."
        else:
            rec_msg = f"💡 Consider growing {recommended} instead — it could earn you ₹{best_income:,.0f}, more than {crop}!"

        crops   = list(le_crop.classes_)
        soils   = list(le_soil.classes_)
        regions = list(le_region.classes_)

        return render_template('index.html',
                               crops=crops, soils=soils, regions=regions,
                               prediction=True,
                               crop=crop, area=area,
                               yield_per_hectare=round(yield_per_hectare, 2),
                               total_yield=total_yield,
                               income=f"{income:,.0f}",
                               comparison=comparison,
                               recommended=recommended,
                               rec_msg=rec_msg)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)