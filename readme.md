
# The API

This is Python web application that generates textual descriptions of uploaded images using a pre-trained image-text model called CLIP. Hugging Face library transformers are used to load the CLIP model. The image description generation is done by comparing the uploaded image features to a set of text features corresponding to different image classes in the CIFAR100 dataset. The image's top 5 most similar labels are then returned as the description.

After you get the API up and running, you can use a REST Client such as Postman or curl to call the API with the image, this is an example with curl. Please note that to run this example command, you might need to make changes to the url and a file tree.png must exist. If you run the API in the provided IDE, use the Ports tab to see what url you need to use.

```
curl --location --request POST 'http://0.0.0.0:8080/description' --form 'image=@"tree.png"'
```

# Setup
To run the application, first install the dependencies. You can do this by installing all the requirements, like this:

```
pip3 install -r requirements.txt
```

If you are running in your own environment and need to install dependencies manually for some reason, here they are:

```bash
pip3 install uvicorn
pip3 install fastapi
pip3 install python-multipart
pip3 install mozuma-clip # This package includes OpenAI's CLIP
                         # (pip3 install git+https://github.com/openai/CLIP.git)
```

# Launch the API

Now you can do this to launch the API.

```
uvicorn app:app --host 0.0.0.0 --port 8080
```

