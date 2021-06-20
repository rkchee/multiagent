results_root = /home/mladmin/ray_results
result_dir = $(shell ls -td $(results_root)/*/ | head -1)
env:
	pipenv shell
ray:
	ray start --head --redis-port=6000
server:
	python nitinol5-server.py --run=PPO
client:
	python thor9-client.py --inference-mode=remote
tensorboard:
	tensorboard --logdir=${results_root} --bind_all
	