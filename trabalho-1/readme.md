# how to run the project
In order to show the images, you will need to install tkinter, please run:

```
sudo apt-get install python3-tk
```

Then, you need create an virtual environment, activate it and to install the requirements:

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

The last step is to actually run the script:

```
python3 main.py baboon.png h1 mono
python3 main.py baboon.png h2 mono
python3 main.py baboon.png h1_and_h2 mono
python3 main.py baboon.png 1a colored
...
```
