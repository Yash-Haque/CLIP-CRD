
# CLIP-CRD  

**CLIP-CRD** is a tool designed to automate report diagnosis for chest radiographs. Here the CRD stands for "**Chest Radiography Diagnosis**". This project was developed during my internship at Systech Datasoft Limited with the goal of building a multi-modal model that would leverage CLIP model's semantic segmentation capability to generate detailed descriptions of X-ray images. By combining OpenAI's CLIP with GPT-2, and following the system designed by [CLIP_prefix_caption]([https://drive.google.com/file/d/1PyVsMYO8bhy8e_o1trA4xk7G08ywsOF3/view?usp=sharing](https://github.com/rmokady/CLIP_prefix_caption)) we were able to train a model with significantly improved prediction accuracy, achieving **54.98% test accuracy**, on Chest Radiography images outperforming the baseline CLIP model.  

## Features  

- **Multi-Modal Understanding** – Integrates both image and text data to generate meaningful and accurate radiology reports.  
- **Deep Learning-Powered** – Leverages state-of-the-art models like CLIP and GPT-2 for improved diagnostic insights.  
- **Performance Boost** – Delivers **54.98% test accuracy**, surpassing the vanilla CLIP model in prediction tasks.  

## Getting Started  

Follow these steps to set up and run the project on your local machine.  

### 1. Clone the Repository  
Run the following command to clone the repository:  
```bash
git clone git@github.com:Yash-Haque/CLIP-CRD.git
cd CLIP-CRD
```

### 2. Set Up Your Environment  
Make sure you have Python installed. If you're using Conda, activate your environment:  
```bash
conda activate CLIP-CRD  
```
If you're using Python’s built-in virtual environment:  
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies  
Install the required packages:  
```bash
pip install -r requirements.txt
```

### 4. Prepare the Data  
Create a folder for storing data:  
```bash
mkdir data
```
Download the **OpenI dataset** from [this link](https://drive.google.com/file/d/1PyVsMYO8bhy8e_o1trA4xk7G08ywsOF3/view?usp=sharing) and unzip it into the `data/` folder.  

### 5. Run the Preprocessing Script  
Navigate to the `scripts/` folder:  
```bash
cd scripts
```
Run the preprocessing script:  
```bash
./parse_openi.sh
```

### 6. Run the Main Script  
Once preprocessing is complete, execute the main script:  
```bash
./run.sh
```

This will process the dataset, generate CLIP embeddings, and store the outputs in the `outputs/` directory.  

## Notes  
- If you run into issues with missing dependencies, make sure you have PyTorch and OpenAI's CLIP installed.  
- If `config.py` isn't found, ensure that the `sys.path` is correctly set to `..` i.e. the project directory.  
- The scripts are designed to automatically create missing folders, so no need to manually create `outputs/`.  
