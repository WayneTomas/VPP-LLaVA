# import debugpy; debugpy.connect(('22.9.35.97', 5673))
from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
