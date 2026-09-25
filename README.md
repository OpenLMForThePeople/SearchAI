# Introduction

Hello there! I invented SearchAI to help you search the depths of YouTube!

**WARNING**

Most of the content of the source code of this repository is made with the help of Artificial Intelligence.

## How does it work?

It utilizes a Python library called `PyTorch` to make a "DQN" model for your query. It can be checked via criteria of your choice.

# Installation

Since the Graphical User Interface is not made yet (which we apologize for all the helpless people new to technology in the 21st century), here is how to install this program into your system:

## Executable for Ease

1. Download the latest `.exe` file from the releases page.
2. Double-click the downloaded file to open it.
3. Enjoy the tool!

## Terminal for the Techies

1. Install Python if you have not already
2. Download this repository's zip file.
3. Extract the zip and move the contents to an accessible folder.
4. Open Command Prompt or PowerShell (Yes, I know it seems scary but trust me!). Go to your folder via `cd <folder name>` to go to a folder, `cd ..` to go back a level, and `dir` to view all the contents of the directory.
5. Run `python -m venv .venv`. (Optional but recommended)
6. Run `pip install -r requirements.txt` to install all required packages.
7. Run `python main.py` (or `main.py` to save time) and enjoy the tool!

## Website for the really challenging

This one is even more advanced than terminal, so:

1. Install Python if you have not already
2. Download this repository's zip file.
3. Extract the zip and move the contents to an accessible folder.
4. Open Command Prompt or PowerShell (Yes, I know it seems scary but trust me!). Go to your folder via `cd <folder name>` to go to a folder, `cd ..` to go back a level, and `dir` to view all the contents of the directory.
5. Run `python -m venv .venv`. (Optional but recommended)
6. Run `pip install -r requirements.txt -r website_requirements.txt` to install all required packages.
7. Run `uvicorn server:app --reload` to open a server
8. Go to `http://localhost:8000` and enjoy your website!

# Usage

Once you have installed the tool, you can create your model. 

## Terminal

1. Navigate to the folder where you want to create your model via `cd [folder]`
2. Give it a name.
3. Give it your search query.
4. Give it criteria to follow. You can add as many criteria as you need!
5. Decide the shape of your model's neural network. The output size must match the number of criteria. A good structure is `input{o}  hidden{2o 2o} output{o}` where `o` is the output size (Do not forget, it must match the amount of criteria!) and `2o` is double the output size.
6. Once done, type `open <model>` and type `1` to train it.
7. When you reach the first video (indicated by the `1/10`), answer a number between 0 and 1 depending on how close the video is to satisfying your criteria. If you need to interrupt the training session, either enter `I` or `Ctrl+C`. If you see numbers between 0 and 1, no worries! Those are just your model's predictions!
8. After answering for all 10 videos, the model will learn until it is done. Then, it will output a `csv` file for you to view.

## Website

1. Navigate to the folder where you want to create your model via clicking
2. Give it a name.
3. Give it your search query.
4. Give it criteria to follow. You can add as many criteria as you need!
5. Decide the shape of your model's neural network. The output size must match the number of criteria. A good structure is `input{o}  hidden{2o 2o} output{o}` where `o` is the output size (Do not forget, it must match the amount of criteria!) and `2o` is double the output size.
5. Once done, click on a button with your model's name on it.
6. When you reach the first video (indicated by the `1/10`), answer a number between 0 and 1 depending on how close the video is to satisfying your criteria. If you need to interrupt the training session, press "Interrupt". If you see numbers between 0 and 1, no worries! Those are just your model's predictions!
7. After answering for all 10 videos, the model will learn until it is done. Then, it will output a `csv` file for you to view.
