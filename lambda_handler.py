# lambda_handler.py
import awsgi
from app import app  # your Flask app

def handler(event, context):
    # This will convert AWS Lambda event into Flask WSGI request and return Flask response
    return awsgi.response(app, event, context)