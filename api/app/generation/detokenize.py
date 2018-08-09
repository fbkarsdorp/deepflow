
tests = [
    {"input": "had the peach exact hand , ' til his niggas knew it they would  ",
     "output": "had the peach exact hand, 'til his niggas knew it they would"},

    {"input": """screamin' , " 4th revolution " killas on the rope nigga look stupid , look """,
     "output": """screamin', "4th revolution" killas on the rope nigga look stupid, look"""},

    {"input": "( git niggas ready ",
     "output": "(git niggas ready"},

    {"input": "shit we play in the club we poppin' up on it , hop on em ' so i drop it , c ' mon ",
     "output": "shit we play in the club we poppin' up on it, hop on em 'so i drop it, c'mon"},

    {"input": "it ' s jump off my grind ",
     "output": "it's jump off my grind"},

    {"input": "don't get us into shit ' round here ",
     "output": "don't get us into shit 'round here"},

    {"input": '" love ones " once more ',
     "output": '"love ones" once more'},
]

if __name__ == '__main__':
    from generation import utils

    for test in tests:
        true, out = test['output'], utils.detokenize(test['input'])
        if true != out:
            print("Error:\n\t{}\n\t{}".format(true, out))
