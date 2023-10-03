FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel
WORKDIR /home/qlora
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8888
ENV PORT=8888
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.token='cx'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_origin='*'" >> /root/.jupyter/jupyter_notebook_config.py
CMD ["jupyter", "lab", "--ip='*'", "--NotebookApp.allow_origin='*'", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='cx'"]

RUN pip install -U openpyxl
RUN git config --global credential.helper store
#huggingface-cli login

RUN mv /home/qlora/lora.py /usr/local/lib/python3.10/dist-packages/peft/tuners/lora.py