import pickle
import json
from flask import Flask,request,render_template,jsonify
from model_files.model import fpredict,features
import numpy as np

app = Flask("pridiction")

with open('./model_files/model.bin','rb') as f_in: 
        model = pickle.load(f_in)
        f_in.close()

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def prediction():
  print(request.form)
  float_features=[float(x) for x in request.form.values()]
  final=float_features
  final= np.float32([final])
  final=np.nan_to_num(final, nan=-9999, posinf=33333333, neginf=33333333)
  """final= np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -124.3979304, -49.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -66.86091034]])"""
  prediction=fpredict(final,model)
  lists = prediction.tolist()
  str = json.dumps(lists)
  return render_template('index.html',pred='your property prize is {}'.format(str))
  
  
  
if __name__ == '__main__':
  app.run(debug=True,host='127.1.1.1',port=9696)