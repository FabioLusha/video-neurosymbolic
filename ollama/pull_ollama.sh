docker pull ollama/ollama:latest

docker tag ollama/ollama:latest lusha/ollama:latest

docker rmi ollama/ollama:latest

echo "Ollama Docker image has been downloaded and renamed to lusha/ollama:latest"
