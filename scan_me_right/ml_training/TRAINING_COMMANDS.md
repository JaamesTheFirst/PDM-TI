# 🎓 Step-by-Step Training Commands

Complete guide to train your formatting recognition model. Just copy and paste these commands!

---

## 📋 Prerequisites

Make sure you're in the right directory:
```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right/ml_training
```

---

## Step 1️⃣: Set Up Python Environment

### Create Virtual Environment (Optional but Recommended)
```bash
python3 -m venv venv
```

### Activate Virtual Environment
```bash
source venv/bin/activate
```
**Note:** You'll see `(venv)` in your terminal prompt

### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**What gets installed:**
- Pillow (image processing)
- NumPy (arrays)
- TensorFlow (ML framework)
- OpenCV (computer vision)
- Scikit-learn (data splitting)

**Expected time:** 2-5 minutes

---

## Step 2️⃣: Generate Training Data

### Run Data Generator
```bash
python generate_training_data.py
```

**What this creates:**
- `training_data/images/` - 5,000 synthetic images
- `training_data/labels/` - 5,000 JSON labels
- `training_data/metadata.json` - Dataset info

**Expected output:**
```
Starting training data generation...
Generating normal text samples... 1000/1000
Generating bold text samples... 1000/1000
Generating italic text samples... 1000/1000
Generating underlined text samples... 1000/1000
Generating title samples... 1000/1000
✅ Dataset generation complete!
Total samples: 5000
```

**Expected time:** 5-10 minutes

### Verify Data Was Created
```bash
ls -la training_data/
ls training_data/images/ | wc -l    # Should show ~5000
ls training_data/labels/ | wc -l   # Should show ~5000
```

---

## Step 3️⃣: Train the Model

### Run Training Script
```bash
python train_model.py
```

**What happens:**
1. Loads 5,000 images
2. Splits into train (80%), validation (10%), test (10%)
3. Creates MobileNetV3-based model
4. Trains for 20 epochs (with early stopping)
5. Evaluates accuracy
6. Saves Keras model (`.h5`)
7. Converts to TensorFlow Lite (`.tflite`)

**Expected output:**
```
📂 Loading dataset...
  Loaded 5000 samples
✅ Loaded 5000 images

🏗️  Building model...
✅ Model created

🚂 Training model...
Epoch 1/20
125/125 [==============================] - 45s 350ms/step
...
Epoch 15/20
125/125 [==============================] - 42s 335ms/step

📊 Evaluating model...
  Test Accuracy: 0.9240 (92.40%)

✅ Training Complete!
📦 Models saved:
   - Keras: formatting_classifier.h5
   - TFLite: formatting_classifier.tflite
```

**Expected time:** 
- **Mac M3 CPU**: 1-2 hours
- **With GPU**: 15-30 minutes

**Target Accuracy:** 
- Good: >85%
- Great: >90%
- Excellent: >95%

---

## Step 4️⃣: Prepare Model for Flutter

### Create Assets Directory
```bash
mkdir -p ../assets/models
```

### Copy TFLite Model
```bash
cp formatting_classifier.tflite ../assets/models/
```

### Verify Model Exists
```bash
ls -lh ../assets/models/formatting_classifier.tflite
```

You should see something like:
```
-rw-r--r--  1 user  staff   4.2M Oct 12 15:30 formatting_classifier.tflite
```

---

## Step 5️⃣: Update Flutter App

### 1. Update `pubspec.yaml`

Open `scan_me_right/pubspec.yaml` and uncomment:
```yaml
dependencies:
  # Uncomment this line:
  tflite_flutter: ^0.10.4
  
flutter:
  assets:
    # Uncomment this line:
    - assets/models/formatting_classifier.tflite
```

### 2. Install New Dependency
```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right
flutter pub get
```

### 3. Update `ocr_service.dart`

You'll modify `lib/services/ocr_service.dart` to use the ML model instead of heuristics.
(I'll help you with the code changes in the next step)

---

## 🎯 Quick Reference Commands

### Generate Data (First Time):
```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right/ml_training
source venv/bin/activate  # If using venv
python generate_training_data.py
```

### Train Model:
```bash
cd /Users/tiagomarques/PDM-TI/scan_me_right/ml_training
source venv/bin/activate  # If using venv
python train_model.py
```

### Deploy to App:
```bash
cp formatting_classifier.tflite ../assets/models/
cd ..
flutter pub get
flutter run
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"
```bash
pip install tensorflow
```

### "No such file or directory: training_data"
```bash
python generate_training_data.py
```

### Training is slow / CPU at 100%
**This is normal!** ML training is CPU-intensive. On Mac M3:
- Uses all CPU cores
- Fan will spin up
- Takes 1-2 hours
- Don't close laptop lid (may slow down)

### Want faster training?
Use Google Colab (free GPU):
1. Upload scripts to Google Drive
2. Open in Colab
3. Train on GPU (15-30 min instead of 2 hours)

---

## 📊 What to Expect

### Data Generation:
- **Time**: 5-10 minutes
- **Output**: 5,000 images + labels
- **Disk Space**: ~200 MB

### Model Training:
- **Time**: 1-2 hours (Mac M3 CPU)
- **Memory**: ~2-4 GB RAM
- **Output**: 2 model files
- **Final Model Size**: ~4-8 MB

### Expected Accuracy:
- **Synthetic Data Only**: 85-92%
- **With Real Data**: 93-97%
- **Production Goal**: >90%

---

## ✅ Validation Checklist

After training, verify:
- [ ] `formatting_classifier.h5` exists (~15-20 MB)
- [ ] `formatting_classifier.tflite` exists (~4-8 MB)
- [ ] Test accuracy > 85%
- [ ] Model copied to `assets/models/`
- [ ] Flutter app updated with tflite_flutter dependency

---

## 🚀 Next Steps After Training

Once your model is trained and deployed:

1. **Test on real documents** - Scan various books, papers
2. **Measure accuracy** - Does it detect bold/titles correctly?
3. **Iterate if needed** - Retrain with more diverse data
4. **Polish UI** - Show formatting visually in the app

---

**Ready? Start with Step 1!** 🎉

---

## 💡 Pro Tips

1. **Start small**: Generate 1,000 samples first to test (change `SAMPLES_PER_STYLE = 200`)
2. **Monitor progress**: Training shows progress bar
3. **Early stopping**: If validation accuracy stops improving, training ends early
4. **Save your work**: Models are automatically saved
5. **GPU acceleration**: Use Google Colab for faster training

---

**Questions?** Check `ml_training/README.md` for more details!

