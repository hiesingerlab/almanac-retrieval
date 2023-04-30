# Almanac: Knowledge-Grounded Language Models for Clinical Medicine

## Installation
First make sure to install all the required Python packages:
```
pip install -r requirements.txt
```
### Docker Setup
The vector database [Qdrant](https://qdrant.tech) relies on the installation of Docker. Please follow the instructions [here](https://docs.docker.com/get-docker/) to download Docker.

After installing and configuring Docker, please run the following commands to pull the selected image and run a local database instance:
```
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```
### OpenAI AI Key
To run the examples in this repository, you will need to obtain an API key from the [official website](https://platform.openai.com/) and run the following command to set it up.

```
export OPENAI_API_KEY='YOUR_API_KEY_HERE'
echo $OPENAI_API_KEY
``` 

## Running
To run the example script, simply run after navigating to the app folder:
```
cd app
python local.py
```
The script `local.py` contains example prompts for clinical queries and calculations. Modify as needed.