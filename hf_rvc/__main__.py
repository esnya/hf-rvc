from argh import ArghParser

from .converters import convert_hubert, convert_rvc, convert_vits
from .tools import eval_dataset

parser = ArghParser()
parser.add_commands([convert_hubert, convert_vits, convert_rvc, eval_dataset])
parser.dispatch()
