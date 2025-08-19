The Grand Opening: Your Final AI Challenge

Welcome, innovators! You've spent weeks learning the theory and practicing the techniques. Now it's time to put it all together and launch your very own AI-powered application.

Your Mission

Your mission, should you choose to accept it, is to take one of the powerful AI models you've worked on, build a functional API to serve its predictions, and create a simple web interface for a user to interact with it. You are taking your model out of the notebook and bringing it to life!

The Blueprint: Three Core Tasks

This challenge is broken down into three main parts, just like opening a restaurant:

Choose Your Signature Dish (The Model)

Go back through your projects and hands-on labs.

Choose one AI model that you feel confident with. This will be the "brain" of your application.

Make sure you have the saved model file (.pkl, .h5, .joblib, etc.) ready to go.

Build Your Food Truck (The API)

Using Flask, create a new web server application.

Write a Python script that loads your saved AI model when the server starts.

Create at least one API endpoint (e.g., /predict) that accepts a POST request.

This endpoint must:

Receive data from a user (e.g., numbers, text, or an uploaded image file).

Preprocess this data so it's in the correct format for your model.

Feed the data to the model to get a prediction.

Return that prediction as a clean JSON response.

Design Your Menu & Ordering Window (The Front-end)

Create a simple index.html file.

This page must be user-friendly and include:

A clear title and instructions.

An HTML form with the necessary input fields (e.g., text boxes for the Titanic model, a file upload button for an image model).

A "Submit" or "Predict" button.

A designated area to display the model's prediction.

Use JavaScript (and the fetch API) to:

Capture the user's input when they click the button.

Send that data to your Flask API endpoint.

Receive the JSON response from the API and display the result to the user on the page.

The Menu: Your Model Choices

You can choose any of the models, including:

The Titanic Survivor Predictor (Neural Network): A web form where a user inputs passenger details (Class, Age, Sex, etc.) and gets back a "Survived" or "Did Not Survive" prediction.

The Handwritten Digit Reader (CNN): A page where a user can upload a small image of a handwritten number (0-9) and get back the model's prediction.

The Fashion Item Classifier (CNN): A page where a user can upload a photo of an item of clothing (from the Fashion MNIST dataset) and get back its category (e.g., "T-Shirt", "Boot").

The Time-Series Forecaster (RNN): A page where a user can input a sequence of numbers and get the model's prediction for the next number in the sequence.

Submission Guidelines & Presentation

You will need to submit your complete project folder, including your Flask app, your saved model, and your HTML/JS files.

Be prepared to give a short (3-5 minute) live demo of your application to the class.

Your demo should explain:

Which model you chose and why.

How your API works.

A live demonstration of your web page making a prediction.

This challenge is your final project. It's your chance to be creative, solve problems, and build a complete, end-to-end AI application.

Good luck, developers!