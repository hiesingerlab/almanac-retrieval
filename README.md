# Almanac — Retrieval-Augmented Language Models for Clinical Medicine

Official codebase for [Almanac — Retrieval-Augmented Language Models for Clinical Medicine](https://ai.nejm.org/doi/full/10.1056/AIoa2300068) to showcase the ability of LLMs to use tools to improve the safety and reliability of their responses to clinical queries.

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
The official manuscript uses the OpenAI API for reproducibility but the codebase can easily be extended to other language and embedding models. We provide wrapper classes for the OpenAI APIs in `models/llm.py` and `models/embedding.py` for ease-of-use. The embedding model also provides a sample token chunking strategy. To run the examples in this repository, you will need to obtain an API key from the [official website](https://platform.openai.com/) and run the following command to set it up.

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
The script `local.py` contains example prompts for clinical queries and calculations which make use of the `qdrant` vector database embedding and retrieval to answer clinical queries. The code showcases retrieval (local and web-based), chunking, embedding, and scoring of documents prior to LLM generation. For user convenience, a sample calculator and three documents are provided.

The `repl.py` and `search.py` scripts offer the template web-scraping and code execution modules respectively to further extend the capabilities of the language model as needed. The latter will need to be modified on a per-site basis. However we leave the reader with the following note of caution:

> **Note:** 
> Always consider the ethical and legal implications of your actions when working with data scraping and execution of arbitrary code. The developers of this repository do not endorse or support the misuse of these capabilities.
> 1. When **web scraping**, please ensure that you comply with the terms of service of the websites you are scraping. Web scraping can put heavy loads on websites and violate their terms of service. Always use respectful scraping practices such as limiting request rates and operating during off-peak hours. It is the responsibility of the user to ensure that their scraping activities are lawful and ethical.
> 2. **Execution of arbitrary code**, can pose significant security risks if not managed properly. Ensure that the environment in which this script is used is secure and that any input to the REPL is thoroughly validated. This tool should only be used in controlled settings and not exposed to untrusted inputs. Misuse of this script can lead to unintended consequences, including but not limited to, system compromise and data leakage. Proceed with caution and prioritize security in its use.

 