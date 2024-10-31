import argparse
from constants import DEFAULT_ARGS
from act.imitate_episodes import main as imitate_main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    args = parser.parse_args()
    
    # Build the argument dictionary for imitate_episodes.py
    imitate_args = DEFAULT_ARGS.copy()
    imitate_args['task_name'] = args.task
    imitate_args['num_epochs'] = args.num_epochs

    # Call the main function of imitate_episodes.py
    imitate_main(imitate_args)

if __name__ == '__main__':
    main()