from flask import Flask, request
from flask import escape
from nemo.collections.nlp.models import TokenClassificationModel
   

# to get the list of pre-trained models
TokenClassificationModel.list_available_models()

# Download and load the pre-trained BERT-based model
model = TokenClassificationModel.from_pretrained("ner_en_bert")

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/nemo/', methods=['POST'])
def get_nemo_string():
    request_data = request.get_json()
    inText = request_data['inText']
# try the model on a few examples
    s= model.add_predictions([inText])
    # show the user profile for that user
    return s[0]
