
tests = [
    {"input": "had the peach exact hand , ' til his niggas knew it they would  ",
     "output": "had the peach exact hand, 'til his niggas knew it they would"},

    {"input": """screamin' , " 4th revolution " killas on the rope nigga look stupid , look """,
     "output": """screamin', "4th revolution" killas on the rope nigga look stupid, look"""},

    {"input": "( git niggas ready ",
     "output": "(git niggas ready"},

    {"input": "shit we play in the club we poppin' up on it , hop on em ' so i drop it , c ' mon ",
     "output": "shit we play in the club we poppin' up on it, hop on em 'so i drop it, c'mon"},

    {"input": "i wish i would have left em ' until they toss me in the ground ",
     "output": "i wish i would have left em 'til they toss me in the ground"},

    {"input": "it ' s jump off my grind ",
     "output": "it's jump off my grind"},

    {"input": "don't get us into shit ' round here ",
     "output": "don't get us into shit 'round here"},

    {"input": '" love ones " once more ',
     "output": '"love ones" once more'},

    {"input": 'yo ',
     "output": "yo"},

    {'input': "but don't worry ' bout success , real life for ' er ",
     "output": "but don't worry 'bout success, real life for 'er"},

    {"input": "1 7 8", "output": "178"},

    {"input": "niggas ' ain't talkin' ' bout ya daddy ",
     "output": "niggas 'ain't talkin' 'bout ya daddy"},

    {"input": "a nigga playin' with me cause i'm pimpin ? 106",
     "output": "a nigga playin' with me cause i'm pimpin? 106"},

    {"input": "i disrespect you ( huh !!!! ) like its up to me for sure",
     "output": "i disrespect you (huh!!!!) like its up to me for sure"},

    {"input": "i don't play , i get money , y ' know what we control say ",
     "output": "i don't play, i get money, y'know what we control say"},

    {"input": "{ * blam * } nah , thought that y ' all stuck in the middle class ",
     "output": "{* blam *} nah, thought that y'all stuck in the middle class"},

    {"input": "a shakespeare intruders , 21 % of them ",
     "output": "a shakespeare intruders, 21% of them"}
]

if __name__ == '__main__':
    from generation import utils

    for test in tests:
        true, out = test['output'], utils.detokenize(test['input'])
        if true != out:
            print("Error:\n\t{}\n\t{}".format(true, out))
