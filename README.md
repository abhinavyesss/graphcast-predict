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
