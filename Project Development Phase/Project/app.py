from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
import os
import numpy as np

app = Flask(__name__)
model=load_model(r"alphabet.hdf5",compile=False)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files['image']
        basepath=os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        import numpy as np
        from keras import utils
        from string import ascii_uppercase

        import pandas as pd

        
        test_image = image.load_img(filepath,target_size=(224,224))
        test_image = utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        df = pd.DataFrame({ "T/F" :[True if round(i) else False for i in result[0]], "Prob" : [round(i, 5) for i in result[0]]}, index=list(ascii_uppercase) + ['delete', 'nothing', 'space'])
        # # Find the index (letter) with the highest probability
        max_prob_index = df['Prob'].idxmax()

        # # Get the letter associated with the highest probability
        letter_highest_prob = max_prob_index if max_prob_index in ascii_uppercase else None

        prediction = letter_highest_prob 
    return prediction

if __name__=='__main__':
    app.run(debug=True)
    