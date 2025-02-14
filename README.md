# Steps:
1. Create the docker image.
2. Run the docker image, creating a container.
3. While the docker image is running, open another terminal and enter the docker container.
4. Execute "python3 prediction.py".

<br>

### Command to build docker image:

```
docker build -f graphcast.dockerfile -t graphcast .
```

### Command to run docker image using the compose file:

```
docker compose -f graphcast.yaml up
```

### Command to enter the terminal and execute instructions:

```
docker exec -it graphcast /bin/bash
```

### Running the prediction file:

```
python3 prediction.py
```

![giphy](https://github.com/user-attachments/assets/2267d22f-7e27-4a14-be6d-0ac7d9af6667)
