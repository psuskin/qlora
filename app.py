import flask

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
from peft import PeftModel

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import re
import json
import secrets
import numpy as np

app = flask.Flask(__name__)
app.debug = True


class PrefixMiddleware(object):
    def __init__(self, app, prefix=''):
        self.app = app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        if environ['PATH_INFO'].startswith(self.prefix):
            environ['PATH_INFO'] = environ['PATH_INFO'][len(self.prefix):]
            environ['SCRIPT_NAME'] = self.prefix
            return self.app(environ, start_response)
        else:
            start_response('404', [('Content-Type', 'text/plain')])
            return ["This url does not belong to the app.".encode()]


app.wsgi_app = PrefixMiddleware(app.wsgi_app)

app.secret_key = "bacon"


class Model:
    def __init__(self, pipe):
        self.pipe = pipe

    def output(self, text):
        response = self.pipe(text)
        reply, docs = response["result"], response["source_documents"]

        answer = reply.split("### Response:", 1)[1].strip()
        answer.replace("SAP", "classix")

        modules = []
        for doc in docs:
            module = re.search(r'"([^"]+)"', doc.page_content).group(1)
            if module not in modules:
                modules.append(module)

        def generate_html(tokens, arrays, probabilities):
            idx = secrets.token_hex(16)
            html_code = f"<div id='{idx}' style='cursor: default'>"

            for i, token in enumerate(tokens):
                if i >= (diff := len(tokens) - len(arrays)):
                    arr = arrays[i - diff]
                    probability = probabilities[i - diff]
                    html_code += f"""
<div class="token" onmouseover="updateBarsAndValues('{idx}', {i}, {[0. if np.isnan(a) else a for a in arr]}, {probability})" onmouseout="reset()">
    {token}
    <div class="bar"></div>
    <div class="value"></div>
</div>
"""
                else:
                    html_code += f"""
<div class="token input">
    {token}
    <div class="bar"></div>
    <div class="value"></div>
</div>
"""

            html_code += "</div>"
            return html_code

        with open("KAL/outputs.txt", "a", encoding="utf-8") as f:
            stringified = json.dumps({"query": text, "response": answer, "modules": modules, "full": reply},
                                     ensure_ascii=False, indent=4)
            f.write(stringified + '\n')

        from numpy import array, nan, float32
        saliency = {"tokens": ['The', 'ĠItalian', 'Ġman', 'Ġworked', 'Ġas', 'Ġa', 'Ġwaiter', 'Ġat', 'Ġthe', 'Ġrestaurant', ',', 'Ġand', 'Ġhe', 'Ġwas', 'Ġa', 'Ġgood', 'Ġfriend', 'Ġof', 'Ġthe', 'Ġowner'], "arrays": [array([0.00281611, 0.00364545, 0.00213171, 0.00252734, 0.00494051,       0.01618105,        nan,        nan,        nan,        nan,              nan,        nan,        nan,        nan,        nan,              nan,        nan,        nan,        nan], dtype=float32), array([0.01515741, 0.02697004, 0.01988638, 0.01438098, 0.01081599,       0.00976275, 0.09552417,        nan,        nan,        nan,              nan,        nan,        nan,        nan,        nan,              nan,        nan,        nan,        nan], dtype=float32), array([0.02451318, 0.01489752, 0.00990465, 0.00896524, 0.03293276,       0.060025  , 0.02741622, 0.03470256,        nan,        nan,              nan,        nan,        nan,        nan,        nan,              nan,        nan,        nan,        nan], dtype=float32), array([0.00478708, 0.01034752, 0.00818659, 0.0049166 , 0.00232616,       0.00309016, 0.01745407, 0.00182156, 0.00373987,        nan,              nan,        nan,        nan,        nan,        nan,              nan,        nan,        nan,        nan], dtype=float32), array([0.01678652, 0.01175672, 0.00970312, 0.00557236, 0.01509926,       0.0242974 , 0.01148202, 0.01005575, 0.0127493 , 0.00931392,              nan,        nan,        nan,        nan,        nan,              nan,        nan,        nan,        nan], dtype=float32), array([0.01577661, 0.01318702, 0.00911494, 0.00645523, 0.01475324,       0.02423487, 0.01251248, 0.00920456, 0.01823439, 0.00969612,       0.04076581,        nan,        nan,        nan,        nan,              nan,        nan,        nan,        nan], dtype=float32), array([0.00435984, 0.00525343, 0.00333427, 0.00308241, 0.00271656,       0.00420872, 0.00453406, 0.00195634, 0.0032407 , 0.00422698,       0.00470024, 0.00623057,        nan,        nan,        nan,              nan,        nan,        nan,        nan], dtype=float32), array([0.0075996 , 0.00600707, 0.00453472, 0.00383128, 0.0078052 ,       0.0138847 , 0.00698502, 0.00471925, 0.01041345, 0.00359636,       0.01387625, 0.00768297, 0.00635854,        nan,        nan,              nan,        nan,        nan,        nan], dtype=float32), array([0.00637821, 0.00661779, 0.00562561, 0.00334654, 0.00468132,       0.00955556, 0.00551376, 0.00283656, 0.0065363 , 0.0025512 ,              nan,        nan,        nan,        nan], dtype=float32), array([0.00365457, 0.00559501, 0.0050163 , 0.00325146, 0.00265017,       0.00550501, 0.006602  , 0.00183556, 0.00422216, 0.00299608,       0.00524381, 0.00279836, 0.00344955, 0.00223757, 0.00293196,              nan,        nan,        nan,        nan], dtype=float32), array([0.01588702, 0.02146394, 0.0173907 , 0.02083934, 0.00727287,       0.00815521, 0.03822549, 0.00538291, 0.00556945, 0.01731191,       0.00766621, 0.00827269, 0.01417106, 0.00958808, 0.01202462,       0.0382867 ,        nan,        nan,        nan], dtype=float32), array([0.00945477, 0.01095424, 0.00892754, 0.01000391, 0.00464457,       0.00468137, 0.02061913, 0.00530285, 0.00394721, 0.01329887,       0.00658287, 0.00719809, 0.0110837 , 0.02110757, 0.01959706,       0.05684447, 0.08165075,        nan,        nan], dtype=float32), array([0.01618847, 0.0112477 , 0.0086441 , 0.00601509, 0.01988119,       0.04187708, 0.01661257, 0.01263047, 0.0336639 , 0.00915174,       0.03219088, 0.02252959, 0.00460613, 0.0057932 , 0.01192344,       0.00575729, 0.00942016, 0.01209147,        nan], dtype=float32), array([0.00953179, 0.01286669, 0.00751429, 0.00536961, 0.00220899,       0.00203821, 0.01398605, 0.00348988, 0.00244822, 0.01728696,       0.00292684, 0.00262578, 0.00268303, 0.00461552, 0.00202137,       0.00702033, 0.01341578, 0.00276748, 0.00728203], dtype=float32)], "probabilities": [0.03370072320103645, 0.41444823145866394, 0.28984105587005615, 0.05912625044584274, 0.1991393268108368, 0.1678282767534256, 0.10112572461366653, 0.14058329164981842, 0.04196571186184883, 0.051482733339071274, 0.4118659198284149, 0.5827254056930542, 0.18700218200683594, 0.14579828083515167]}

        return {"output": answer, "modules": modules,
                "saliency": generate_html(saliency["tokens"], saliency["arrays"], saliency["probabilities"])}


def setup():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"
    model_id = 'meta-llama/Llama-2-13b-hf'

    def load_model(model_name_or_path, adapter_path=None):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Fixing some of the early LLaMA HF conversion issues.
        tokenizer.bos_token_id = 1

        # Load the model (use bf16 for faster inference)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        )

        if adapter_path:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()
        else:
            model = base_model

        return model, tokenizer

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
        embed_instruction="Represent the document for retrieval:",
        query_instruction="Represent the question for retrieving supporting documents:",
    )

    db = Chroma(persist_directory="DB_KLIO_ALPACA", embedding_function=embeddings,
                client_settings=Settings(anonymized_telemetry=False, is_persistent=True))
    retriever = db.as_retriever()

    model, tokenizer = load_model(model_id, 'output/klio-alpaca-2-13b-r64-noeval/checkpoint-1875/adapter_model')
    generation_config = GenerationConfig.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=4096,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n###Input:\n{context}\n\n### Response:"
    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
        },
    )

    model = Model(qa)

    return model


try:
    model = setup()
except Exception as e:
    print("Error while loading generative model:")


@app.route('/query/<string:intention>', methods=['POST'])
def query(intention):
    if flask.request.method == 'POST' and intention == 'generation':
        text = flask.request.form["text"]

        try:
            output = model.output(text)
            return flask.make_response(output)
        except Exception as e:
            print(e)
            return flask.make_response(flask.jsonify({"error": "Sorry, an error occurred. Please try again."}))

    return flask.make_response(flask.jsonify({"error": "Sorry, an error occurred. Please try again."}))


@app.route('/', methods=['GET', 'POST'])
def index():
    return flask.render_template('index.html')


if __name__ == '__main__':
    app.run()
