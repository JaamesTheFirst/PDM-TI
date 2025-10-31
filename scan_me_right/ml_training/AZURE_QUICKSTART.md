# Train on Azure ML Studio (Quick Start)

## Prerequisites
- Azure ML Studio account (pay-as-you-go is fine)
- 2 files: `generate_realistic_documents.py` and `train_model_pages.py`

## Setup (5 minutes)

### 1. Create Compute Instance
1. Go to Azure ML Studio: https://ml.azure.com
2. Click **"Compute"** in the left sidebar
3. Click **"Compute instances"** tab
4. Click **"+ New"**
5. Configure:
   - **Compute name**: `gpu-trainer` (or any name)
   - **Virtual machine type**: **GPU**
   - **Virtual machine size**: **Standard_NC6s_v3** (most cost-effective GPU)
   - Click **"Create"**
   
   💰 Cost: ~$2/hour, but we only need ~30 minutes = **$1 total**

### 2. Launch Jupyter
1. Wait for compute instance to start (status will change to "Running")
2. Click **"Launch application"** → **"Jupyter"**
3. Jupyter notebook will open in a new tab

### 3. Upload Scripts
1. In Jupyter, click **"Upload"** button
2. Upload these 2 files:
   - `generate_realistic_documents.py`
   - `train_model_pages.py`
3. Wait for upload to complete

## Training (30 minutes)

### 4. Create New Notebook
1. In Jupyter, click **"New"** → **"Python 3"**
2. You'll see an empty notebook

### 5. Run Training (2 cells)

**Cell 1: Generate Data**
```python
# Install dependencies
!pip install tensorflow pillow scikit-learn numpy --quiet

# Generate training dataset
exec(open('generate_realistic_documents.py').read())
```
Click "Run" button (or Shift+Enter)

**Cell 2: Train Model**
```python
# Train the model
exec(open('train_model_pages.py').read())
```
Click "Run"

### 6. Wait & Monitor
You'll see progress:
- Data generation: ~5-10 minutes (progress bars)
- Training: ~20-30 minutes (epoch progress)

## Download Models

### 7. Download Files
1. In Jupyter file browser, you'll see these new files:
   - `formatting_classifier.h5`
   - `formatting_classifier.tflite`
   - `training_data_comprehensive/` folder
2. Right-click on each model file → **"Download"**
3. Save them to your local `scan_me_right/assets/` folder

## Costs Summary
- Compute instance: ~$2/hour
- Training time: ~30 minutes
- **Total cost: ~$1** 🎉

## Shut Down (Important!)
**After downloading models:**
1. Go back to Azure ML Studio
2. Click **"Compute"** → **"Compute instances"**
3. Click **"Stop"** next to your compute instance
4. This stops the billing! 💰

---

## Troubleshooting

### "GPU not available"
- Wait a few minutes for the instance to fully initialize
- Refresh the page and try again

### "ModuleNotFoundError"
```python
!pip install tensorflow pillow scikit-learn numpy --upgrade
```

### Out of disk space
- The `training_data_comprehensive/` folder is large (~1-2GB)
- You can delete it after training if needed
- The models themselves are only ~10-20MB

### Want to start/stop later?
- **Start**: Go to Compute instances → Click "Start"
- **Stop**: Go to Compute instances → Click "Stop" (saves money!)

---

**That's it! You'll have a trained model in ~30 minutes! 🚀**

