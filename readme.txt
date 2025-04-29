- Card Class Classifier

This project is a machine learning-based web application that classifies cards from images using 
a Convolutional Neural Network (CNN). The CNN model was trained on a labeled dataset of card images to 
accurately predict the specific card ( "Ace of Hearts", "ten of Spades" ). A FastAPI backend serves 
the trained model through a RESTful API, enabling real-time inference. Users can upload an image via a 
web interface, and the application returns the predicted card class instantly.


How to Run the Project ?
Follow these steps to set up and run the project on your local machine:

1- Clone the github repo to your machine
    command : git clone <repository-url>
  
2- Open the project folder in the VScode or other code editor

3- Open a terminal (PowerShell or your terminal of choice)

4- Create a virtual enviroment:
    command: py -3.10 -m venv venv 
  
5- Activate the virtual enviroment:
    command: .venv\Scripts\activate 
    
6- Install project dependencies: 
    command: pip install -r requirements.txt 
    
7- Run the application using Uvicorn: 
    command: uvicorn app:app --reload 