FROM python:3.7

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"


COPY . /user/app


EXPOSE 8080
WORKDIR /user/app
RUN pip install -r requirements.txt

CMD python flask_code.py