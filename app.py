from flask import Flask,request,render_template
from src.predict_pipeline import Predict_Pipeline
from utility.loggers import logger

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict_data():
    try:
        if request.method == 'GET':
            return render_template('predict_data.html')
        else:
            message = request.form.get('message')
            obj = Predict_Pipeline(message)
            customdata = obj.customdata()
            prediction_result = obj.predict(customdata['text'])
            return render_template("predict_data.html",prediction=prediction_result)

            
    
    except Exception as e:
        logger.error(logger.error(f"Error during prediction: {e}"))
        return render_template("predict_data.html", prediction="Error: Could not process the prediction.")

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)