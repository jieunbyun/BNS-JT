import BNS_JT.model as model
import BNS_JT.config as config
from pathlib import Path

HOME = Path(__file__).parent

def main():

    cfg = config.Config(HOME.joinpath('./config.json'))

    cpms, varis = model.setup_model(cfg)


if __name__=='__main__':

    main()
