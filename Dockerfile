FROM public.ecr.aws/lambda/python:3.9

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app files
COPY . ./

# Set the CMD to your lambda_handler.handler function
CMD ["lambda_handler.handler"]

