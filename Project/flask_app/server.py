from flask import Flask, render_template, url_for
 
app = Flask(__name__)
 
@app.route('/')
def index():
   return render_template("index.html")
 
@app.route('/about')
def about():
   pass
 
if __name__ == 'main':
    app.run(debug=True)
    
app.run('127.0.0.1', port=5500)