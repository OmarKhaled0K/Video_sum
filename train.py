from configs import get_config
from solver import Solver
from data_loader import get_loader
from pathlib import Path

if __name__ == '__main__':
    config = get_config(mode='train')
    test_config = get_config(mode='test')
    print(config)
    path = Path(r"E:\Artificial_Intelligence\Video Summarization\Datasets\tvsum\features")
    train_loader = get_loader(path, 'train')
    test_loader = get_loader(path, 'test')
    solver = Solver(config, train_loader, test_loader)

    solver.build()
    solver.train()
