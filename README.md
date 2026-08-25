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

# Usage

Once you have installed the tool, you can create your model. 

1. Give it a name.
2. Give it your search query.
3. Give it criteria to follow. You can add as many criteria as you need!
4. Once done, type `open <model>` and type `1` to train it.
5. When you reach the first video (indicated by the `1/10`), answer a number between 0 and 1 depending on how close the video is to satisfying your criteria. If you need to interrupt the training session, either enter `I` or `Ctrl+C`. If you see numbers between 0 and 1, no worries! Those are just your model's predictions!
6. After answering for all 10 videos, the model will learn until it is done. Then, it will output a `csv` file for you to view.