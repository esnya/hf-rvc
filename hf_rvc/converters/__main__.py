from argh import ArghParser

from . import convert_hubert, convert_rvc, convert_vits

parser = ArghParser()
parser.add_commands([convert_hubert, convert_vits, convert_rvc])
parser.set_default_command(convert_rvc)
parser.dispatch()
