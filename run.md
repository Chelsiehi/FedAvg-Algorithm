
# Experiment setup

## mnist iid
B = 10 E = 1/5/20
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 1 --iid
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 5 --iid
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 20 --iid

B = 50 E = 1/5/20
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 1 --iid
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 5 --iid
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 20 --iid

## mnist non-iid
B = 10 E = 1/5/20
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 1
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 5
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 10 --local_ep 20

B = 50 E = 1/5/20
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 1
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 5
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset mnist --local_bs 50 --local_ep 20


## cifar10 iid
B = 10 E = 1/5/20
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 1 --iid
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 5 --iid
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 20 --iid

B = 50 E = 1/5/20
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 1 --iid
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 5 --iid
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 20 --iid

## cifar10 non-iid
B = 10 E = 1/5/20
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 1
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 5
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 10 --local_ep 20

B = 50 E = 1/5/20
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 1
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 5
python main.py --epochs 300 --num_users 100 --frac 0.1 --dataset cifar --local_bs 50 --local_ep 20

