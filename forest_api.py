# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, jsonify, request
from flask import Flask, render_template
import wrapper

# creating a Flask app
app = Flask(__name__,template_folder='templates')


@app.route('/', methods = ['GET'])
def dynamic_page():
    data = wrapper.run()
    return data

@app.route('/html')
def static_page():
  return render_template('index.html')
  
# driver function
if __name__ == '__main__':
  
    app.run(host='0.0.0.0')