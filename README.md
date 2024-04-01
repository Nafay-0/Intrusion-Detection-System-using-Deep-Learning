# Intrusion Detection System using Deep Learning

This project aims to perform a comparative analysis of multiple Deep Learning based models to detect intrusion from incoming network packets. The project is implemented in Python and uses pip for package management.

## Dataset Used

## Project Structure

The project has the following structure:

- `data_preperation.ipynb`: This Jupyter notebook contains the code for preparing the data for the deep learning models. It includes steps for loading the data, cleaning it, and splitting it into training and testing sets.

- `ANN`: This directory contains the code for the Artificial Neural Network model.

- `RNN`: This directory contains the code for the Recurrent Neural Network model.

- `LSTM`: This directory contains the code for the Long Short-Term Memory model.

- `GRU`: This directory contains the code for the Gated Recurrent Unit model.

- `DenseNet`: This directory contains the code for the DenseNet model.

- `multi_clf.ipynb`: This notebook contains the code for training and evaluating the models for Multi-class classification.

- `binary_clf.ipynb`: This notebook contains the code for training and evaluating the models for Binary classification.

- `Data`: This directory is where the dataset should be placed. It is accessible in code as `./data/`.

- `Deployment` : This directory contains the code for deploying the model using Flask.



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/GlitChed2k2/Intrusion-Detection-System-using-Deep-Learning.git
   ```
2. Navigate into the cloned repository:
   ```
   cd Intrusion-Detection-System-using-Deep-Learning
   ```
3. Install the required Python packages:
   ```
   pip install -r requirements.txt
    ```
4. Download the dataset from the following link and place in Data folder
    ```
   https://research.unsw.edu.au/projects/unsw-nb15-dataset
    ```

## Usage

1. Run the `data_preperation.ipynb` notebook to prepare the data for the models.

2. Navigate to the directory of the model you want to run (e.g., `ANN`, `RNN`, `LSTM`, `GRU`, `DenseNet`).


## Running Flask App

1. Navigate to the `Deployment` directory:
   ```
   cd Deployment
   ```
2. Run the Flask app:
   ```
    python app.py
    ```
   
3. Open a web browser and go to 
    ```
    http://localhost:8000/
    ```
   


## License

This project is licensed under the MIT License - see the `LICENSE` file for details
```

Please note that you might need to adjust the file paths, commands, and other details to match your actual project setup.