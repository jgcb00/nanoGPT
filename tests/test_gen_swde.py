import os
from dataclasses import dataclass
from typing import List
import tyro
import pickle
import json
import tiktoken
from pathlib import Path
from arch.utils import get_model
import torch
from config import NanoConfig

from arch.lm import NanoLM

ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


@dataclass
class Args:
    run_dir: Path  # something like logs/... (the dir that contains the .pt model)
    tasks: str  # list of tasks to evaluate on (hellaswag, winogrande, ...)
    batch_size: int = 32

    def __post_init__(self):
        self.tasks = self.tasks.split(",")
        assert self.run_dir.exists(), f"Run directory {self.run_dir} does not exist."


args = tyro.cli(Args)

# read config
with open(args.run_dir / "config.pkl", "rb") as f:
    config: NanoConfig = pickle.load(f)
config.rmsnorm = False
config.disable_scalable_softmax_for_local = (
    False  # False for loading old runs, True for newer ones
)

# define and load model, tokenizer
model = get_model(config)
model.cuda()

model_file = sorted(args.run_dir.glob("state_step*.pt"))[-1]
assert model_file.exists(), f"Model file {model_file} does not exist."

checkpoint = torch.load(model_file)
state_dict = checkpoint["model"]

new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

with open("data/enc.pkl", "rb") as f:
    enc_pickled = pickle.load(f)
enc = tiktoken.core.Encoding(enc_pickled.pop("name"), **enc_pickled)

lm = NanoLM(
    model=model,
    config=config,
    enc=enc,
    batch_size=args.batch_size,
)

#############

input_str = "\ufeff Suicide Kings Movie Facts and Details click here amc home | movie guide Genres\nLists\nRatings amctv.com>movie guide>Suicide Kings>details Suicide Kings details\nOverall Rating Total Ratings: 1 Overview\nDetails\nCast & Credits\nAwards\nReview Movie Details: Director: Peter O'Fallon\nProduced By: Eyes 'n Rice, Live Entertainment\nYear: 1997\nRun Time: 106 minutes\nCountry: USA\nLanguage: English MPAA Rating: R (for strong violence and language, and for some nudity and drug use)\nCategory: Feature\nGenre/Type: Thriller, Crime\nFilmed In: Color Key Cast: Christopher Walken, Denis Leary, Henry Thomas, Sean Patrick Flanery, Jay Mohr, Jeremy Sisto, Johnny Galecki, Laura San Giacomo, Full Credits Television director Peter O'Fallon made his feature film debut with this independent film that pays obvious homage to the style of Quentin Tarantino, with plenty of violence and funny, talkative hit men. Suave gangster Charlie Barrett (Christopher Walken) meets four young men who have taken over his regular booth at a popular bistro. Charmed by the swaggering kids, he agrees to take a ride with them, but they give him a sedative and he awakens in a deserted mansion, taped to a chair with one of his fingers missing. One of his abductors, Avery (Henry Thomas), says that he has a sister who has been kidnapped and they need two million dollars to get her back, as well as a finger to exchange for her severed digit. Charlie phones his lawyer Marty (Cliff De Young), who calls a henchman, Lono (Denis Leary), who investigates the kidnappings and gives Charlie enough information to start playing each of his inexperienced abductors against the others. by Michael Betzold, Rovi Keywords: betrayal\ngangster\nkidnapping ransom\ncollege-student Themes: Thrill Crime\nCrime Gone Awry\nKidnapping \ufeff\ufeff\ufeff\ufeff click here Currently 3/5\n1\n2\n3\n4\n5 write your review Enter your review: Log In or Register to\nwrite your review username\npassword Similar Movies The Hit (1984) The Ref (1994) Better Luck Tomorrow (2002) Dead Silence (1991) Northern Exposure: Aurora Borealis - A Fairytale for Big People (1990) Barquero (1970) Attack The Gas Station! (1999) Wannabes (2000) Niagara Niagara (1997) Can't Buy Me Love (1987) schedule\nmovies\namc originals\nblogs\nvideos\nphotos\npolls\ngames\nstore\nmobile\nsubscribe to the amc newsletter About & Contact\nFAQ\nTerms & Conditions\nPrivacy Policy network sites:\nAMCtv.com\nFilmsite.org\nFilmcritic.com Copyright 2010 American Movie Classics Company LLC. All rights reserved. \n\nSummary of information above...\ncategory: Feature\ncountry: USA\ngenre/type: Thriller, Crime\ndirector:"
input_enc = enc.encode(input_str)

print(input_enc)
print(len(input_enc))

stop_tokens = ["\n", "."]
stop_tokens = [enc.encode(token)[0] for token in stop_tokens]
print(stop_tokens)

with ctx:
    output_enc = lm.generate(
        [torch.tensor(input_enc)],
        n_tokens=[48],
        samples=[False],
        stop_tokens=[stop_tokens],
    )

print(output_enc[0])
print(enc.decode(output_enc[0].tolist()))
