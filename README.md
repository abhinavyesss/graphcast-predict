Steps:
1. Create the docker iamge.
2. Run the docker image, creating a container.
3. While the docker image is running, open another terminal and enter the docker container.
4. Execute "python3 prediction.py".

Command to build docker image:

```
docker build -f graphcast.dockerfile -t graphcast .
```

Command to run docker iamge using the compose file:

```
docker compose -f graphcast.yaml up
```

Command to enter the terminal and execute instructions:

```
docker exec -it graphcast /bin/bash
```
