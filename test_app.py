from flask import Flask

app = Flask(__name__)
application = app

@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    application.run() 