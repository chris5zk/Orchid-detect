from import_package import *

print(f"{Bcolors.OKBLUE}cuda{Bcolors.OKBLUE}" if torch.cuda.is_available() else f"{Bcolors.OKBLUE}cuda{Bcolors.OKBLUE}", file=sys.stderr)
