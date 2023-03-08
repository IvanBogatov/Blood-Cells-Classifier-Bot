FROM python:slim
ENV TOKEN='<YOUR TOKEN>'
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip3 install torch==1.13.1+cpu torchvision --extra-index-url https://download.pytorch.org/whl/cpu
CMD python bot.py