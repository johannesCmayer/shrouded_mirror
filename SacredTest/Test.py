from sacred import Experiment
import time

ex = Experiment()

@ex.config
def my_config():
    recipient = 'Joe'
    message = f'hello {recipient}!'
    #sleep_time = 1

@ex.automain
def main(message, sleep_time=1):
    time.sleep(sleep_time)
    print(message)