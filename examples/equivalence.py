import numpy as np
from np_rasp import kqv, indices, full, equals, seq_map, tok_map, where

def equivalence(x, BOS=1, EOS=2, RESPS={3}, INPTS={4,5,6}, TRIGS={7}):
    """
    Solves a numeric equivalence task where the goal is to output
    the same number of response tokens after the trigger as there are
    input tokens before the trigger. The algorithm uses a -1 as the
    value for input tokens and 1 as the value for response tokens and
    knows to stop when the mean over the input and response tokens is
    0 after the trigger token.

    examples -> BOS INPT INPT TRIGGER RESP RESP EOS
                 1   5    4      7     3    3    2

             -> BOS INPT INPT INPT TRIGGER RESP RESP RESP EOS
                 1   5    4    6      7     3    3    3    2

    x: sequence of tokens (B,S)
        a context to be operated on. include a batch dimension.
    """
    start_idx = kqv(
        k=x, q=full(x, BOS), v=indices(x), pred=equals, reduction="max")
    trig = list(TRIGS)[0]
    trig_idx = kqv(
        k=x, q=full(x,trig), v=indices(x), pred=equals, reduction="max")

    # Is the token before the trigger token
    pre_trig = seq_map(x=indices(x), y=trig_idx, func=lambda x,y: x<y)
    # Is the token after the beginning of sequence token?
    is_valid = seq_map(x=indices(x), y=start_idx, func=lambda x,y: x>=y)
    # Is the token a response type token
    is_resp = tok_map(x=x, func=lambda x: x in RESPS)
    # Is the token an input type token
    is_inpt = tok_map(x=x, func=lambda x: x in INPTS)
    respinps = RESPS.union(INPTS)
    do_count = tok_map(x=x, func=lambda x: x in respinps)
    vals = where(is_inpt, full(x,-1), full(x,0))
    vals = where(is_resp, full(x,1), vals)
    mean = kqv(
        k=np.asarray(do_count),
        q=np.asarray(is_valid),
        v=np.asarray(vals),
        pred=lambda x,y: int(x and y))
    is_zero = seq_map(x=mean, y=full(mean,0), func=equals)
    not_first = seq_map(indices(x), full(x, start_idx), lambda x,y: x!=y)
    is_eos = seq_map(x=is_zero, y=not_first, func=lambda x,y: x and y)
    rid = list(RESPS)[0]
    inpt = np.random.randint(0,len(INPTS), size=x.shape)
    inpt = np.asarray(list(INPTS))[inpt.reshape(-1)].reshape(x.shape)
    next_toks = where(pre_trig, inpt, full(x,rid))
    next_toks = where(is_eos, full(x, EOS), next_toks)
    return next_toks

if __name__ == '__main__':
    ## Test it on an input:
    BOS = 1
    EOS = 2
    RESPS = {3}
    INPTS = {4,5,6}
    TRIGS = {7}

    ## Generate auto-regressively
    max_num = 10
    offset = sorted(list(INPTS))[0]
    prompts = [
        np.array(
            [BOS]+
            [np.random.randint(len(INPTS))+offset for _ in range(i)]+
            [list(TRIGS)[np.random.randint(len(TRIGS))]]
        ) for i in range(max_num)
    ]
    for i,prompt in enumerate(prompts):
        print('prompt:', prompt)
        x = prompt.copy()
        while x[-1] != EOS:
            next_tok = equivalence(x)[-1]
            print(f'{x} -> {next_tok}')
            x = np.concatenate((x, [next_tok]))

        assert np.isin(x,list(RESPS)).sum()==i and x[-1]==EOS
        print('generation:', x)
