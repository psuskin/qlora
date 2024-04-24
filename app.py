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
        reply, docs, saliency = response["result"], response["source_documents"], response["saliency"]

        answer = reply.split("### Response:")[1].strip()
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
<div class="token" onmouseover="updateBarsAndValues('{idx}', {i}, {[0. if isnan(a) else a for a in arr]}, {probability})" onmouseout="reset()">
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
            stringified = json.dumps({"query": text, "response": answer, "modules": modules, "input": reply},
                                     ensure_ascii=False, indent=4)
            f.write(stringified)

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
