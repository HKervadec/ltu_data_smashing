How to prepare enviroment for this app?

1. Install python on your computer
2. Install wirtual enviroment on your computer
    http://docs.python-guide.org/en/latest/dev/virtualenvs/
3. Make new virtual enviroment on your cmputer (it is not important where you careate it):
    Windows: python -m venv data_smashing
    Linux: python -m venv data_smashing
    * data_smashing is just example of the name you can use any other name
4. Open virtual enviroment:
    Windows: data_smashing\Scripts\activate
    Linux:  source data_smashing/bin/activate
    * to make it you have to be in virtual enviroment directory
5. Clone app from github in any space on computer
6. Install pip if it is not installed
    windows
    - download https://raw.githubusercontent.com/pypa/pip/master/contrib/get-pip.py
    - install python get-pip.py
    linux
    sudo apt-get install python-pip
7. Install requirements in virtual enviroment
    pip install -r /path/to/requirements.txt
    * requirements.txt are in webapp directory of our project
8. move to directory webapp/data_smashing_app/
9. Run python manage.py migrate
10. Run python manage.py runserver
11. Application is now working, you acess to app using url: http://localhost:8000/

If any problems just write me!!!

When everything up is installed, if you want to run app you just need to:

1. Run virtual enviroment
    Windows: data_smashing\Scripts\activate
    Linux:  source data_smashing/bin/activate
2. Move to directory webapp/data_smashing_app/
10. Run python manage.py runserver
11. Application is now working, you acess to app using url: http://localhost:8000/