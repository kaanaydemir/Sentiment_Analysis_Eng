from flask import Flask,request,jsonify
import numpy as np

import tensorflow as tf

from transformers import TFBertForSequenceClassification, BertTokenizer,glue_convert_examples_to_features, InputExample,BertConfig,InputFeatures
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gdown
import zipfile


url = 'https://drive.google.com/uc?id=1-7dj0Yx6MTQ0ZqrHzxQ8sNrt11-31-TY'
output = 'eng.zip'
gdown.download(url, 'output', quiet=False)

with zipfile.ZipFile("output","r") as zip_ref:
    zip_ref.extractall("model")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
sentiment_model = TFBertForSequenceClassification.from_pretrained('model/eng') 
def example_to_features_predict(input_ids, attention_masks, token_type_ids):
    """
        Convert the test examples into Bert compatible format.
    """
    return {"input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids}


def get_prediction(model,in_sentences):
    """
        Prepare the test comments and return the predictions.
    """
    labels = ["0", "1","2"]
    sentiment = ['negative','neutral','positive']
    label_list = ['0','1','2']
    MAX_SEQ_LENGTH = 160
    
    txt = in_sentences['content']
    input_examples = [InputExample(guid="", text_a = txt, text_b = None, label = '0')]
    predict_input_fn = glue_convert_examples_to_features(examples=input_examples, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, task='sst-2', label_list=label_list)
    x_test_input, y_test_input = my_solution(predict_input_fn)
    test_ds   = tf.data.Dataset.from_tensor_slices((x_test_input[0], x_test_input[1], x_test_input[2])).map(example_to_features_predict).batch(32)

    predictions = model.predict(test_ds)
    predictions_classes = np.argmax(predictions[0], axis = 1)
    return sentiment[predictions_classes[0]]

def my_solution(bdset):
    """ Create a list of input tensors required to be in the first argument of the
        model call function for training. e.g. `model([input_ids, attention_mask, token_type_ids])`.
    """
    input_ids, attention_mask, token_type_ids, label = [], [], [], []
    for in_ex in bdset:
        input_ids.append(in_ex.input_ids)
        attention_mask.append(in_ex.attention_mask)
        token_type_ids.append(in_ex.token_type_ids)
        label.append(in_ex.label)

    input_ids = np.vstack(input_ids)
    attention_mask = np.vstack(attention_mask)
    token_type_ids = np.vstack(token_type_ids)
    label = np.vstack(label)
    return ([input_ids, attention_mask, token_type_ids], label)

app = Flask(__name__)


@app.route('/api/sentiment',methods=['POST'])
def sentiment_prediction():
	content = request.json
	result = get_prediction(sentiment_model,content)
	return jsonify(result)

@app.route("/")
def index():
	return '<h1>FLASK APP IS RUNNING</h1>'

if(__name__ == '__main__'):
	app.run()
